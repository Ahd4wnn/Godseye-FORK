"""Track B service: question -> grounded answer with timeouts, bounded concurrency and bounded sessions. Failures here
cannot touch Track A (it only reads the stores; run it in its own process for full isolation)."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..answer.engine import AnswerEngine
from ..answer.response import Answer
from ..query.schema import Clarification
from ..query.understand import Failed, NeedsClarification, QueryUnderstanding, Rejected, Resolved, Session
from .config import RuntimeConfig
from .metrics import Metrics

log = logging.getLogger("godseye.runtime")


@dataclass
class Result:
    kind: str                                   # answer | clarify | rejected | busy | timeout | error
    answer: Answer | None = None
    clarification: Clarification | None = None
    message: str = ""
    elapsed_ms: float = 0.0


@dataclass
class _Entry:
    session: Session
    last: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class QueryService:
    def __init__(self, qu: QueryUnderstanding, engine: AnswerEngine, cfg: RuntimeConfig, metrics: Metrics | None = None):
        self.qu, self.engine, self.cfg = qu, engine, cfg
        self.m = metrics or Metrics()
        self._pool = cf.ThreadPoolExecutor(max_workers=cfg.query_max_inflight, thread_name_prefix="query")
        self._slots = threading.BoundedSemaphore(cfg.query_max_inflight)
        self._sessions: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- sessions (bounded, expiring)
    def session(self, sid: str | None) -> tuple[str, _Entry]:
        now = time.time()
        with self._lock:
            for k in [k for k, e in self._sessions.items() if now - e.last > self.cfg.session_ttl_s]:
                del self._sessions[k]
            if sid in self._sessions:
                e = self._sessions[sid]
            else:
                if len(self._sessions) >= self.cfg.max_sessions:
                    del self._sessions[min(self._sessions, key=lambda k: self._sessions[k].last)]
                sid = uuid.uuid4().hex
                e = self._sessions[sid] = _Entry(Session(self.qu.cfg.recent_identities))
            e.last = now
            return sid, e

    # ---------------------------------------------------------------- questions
    def ask(self, sid: str | None, text: str) -> tuple[str, Result]:
        return self._run(sid, lambda s: self.qu.handle(s, text))

    def answer_clarification(self, sid: str, cid: str, response) -> tuple[str, Result]:
        return self._run(sid, lambda s: self.qu.respond(s, cid, response))

    def _run(self, sid, fn) -> tuple[str, Result]:
        sid, entry = self.session(sid)
        if not self._slots.acquire(blocking=False):
            self.m.inc("query_busy")
            return sid, Result("busy", message="too many questions are being processed; try again shortly")
        t0 = time.perf_counter()

        def work():
            try:
                with entry.lock:
                    out = fn(entry.session)
                    if isinstance(out, Resolved):
                        return Result("answer", answer=self.engine.run(out.context, entry.session))
                    if isinstance(out, NeedsClarification):
                        return Result("clarify", clarification=out.clarification)
                    if isinstance(out, Failed):
                        return Result("error", message=out.reason)
                    return Result("rejected", message=out.reason if isinstance(out, Rejected) else str(out))
            finally:
                self._slots.release()

        fut = self._pool.submit(work)
        try:
            res = fut.result(timeout=self.cfg.query_timeout_s)
        except cf.TimeoutError:
            self.m.inc("query_timeout")
            log.error("query timed out after %.0fs (worker continues in the background; result discarded)", self.cfg.query_timeout_s)
            res = Result("timeout", message=f"the question took longer than {self.cfg.query_timeout_s:.0f}s")
        except Exception as e:                                  # noqa: BLE001 - a query bug must not escape to the caller / Track A
            self.m.inc("query_errors")
            log.exception("query failed")
            res = Result("error", message=f"{type(e).__name__}: {e}")
        res.elapsed_ms = (time.perf_counter() - t0) * 1e3
        self.m.observe("query_ms", res.elapsed_ms)
        return sid, res
