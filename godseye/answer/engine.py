"""AnswerEngine: Resolved Query Context -> Query Planner -> Retrieval -> Path Reconstructor -> Response Builder."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..query.schema import ResolvedQueryContext
from ..query.understand import Session
from . import response as R
from .config import AnswerConfig
from .path import reconstruct
from .plan import GID, Op, PlanError, QueryPlan, make_plan, validate_plan
from .retrieval import Retriever
from .topology import Topology

log = logging.getLogger("godseye.answer")
ROOT = Path(__file__).resolve().parents[2]


class _Vectors:
    """Read access to Track A's `track_clip` collection (get_collection only: never creates anything)."""
    def __init__(self, path: Path):
        self.col = {}
        try:
            c = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
            self.col["track_clip"] = c.get_collection("track_clip")
        except Exception as e:                                       # noqa: BLE001 - surfaced as SYSTEM_ERROR on use
            self.error = e


class AnswerEngine:
    def __init__(self, cfg: AnswerConfig, embedder=None, llm=None, base_dir: Path = ROOT, vectors=None):
        root = Path(cfg.store_root)
        root = root if root.is_absolute() else base_dir / root
        db = root / "godseye.db"
        if not db.exists():
            raise FileNotFoundError(f"{db} not found: run Track A (storage) first")
        self.cfg, self.llm, self.root = cfg, llm, root
        self.conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")
        self.schema_version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        self.topology = Topology.load(cfg.topology if Path(cfg.topology).is_absolute() else base_dir / cfg.topology) if cfg.topology else Topology()
        self.embedder = embedder
        self.vectors = vectors if vectors is not None else (_Vectors(root / "chroma") if (root / "chroma").exists() else None)
        self.retriever = Retriever(cfg, self.conn, self.vectors, embedder, root / "clips")

    # ---------------------------------------------------------------- helpers used by plan validation
    def _cameras(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT camera_id FROM cameras")}

    def _identity_exists(self, gid: str) -> bool:
        return self.conn.execute("SELECT 1 FROM identities WHERE global_id=?", (gid,)).fetchone() is not None

    def _track_exists(self, cam, epoch, tid) -> bool:
        return self.conn.execute("SELECT 1 FROM tracks WHERE camera_id=? AND stream_epoch=? AND track_id=?", (cam, epoch, tid)).fetchone() is not None

    def plan(self, ctx: ResolvedQueryContext) -> QueryPlan:
        return validate_plan(make_plan(ctx, self.cfg), self.cfg, self._cameras(), self._identity_exists, self._track_exists)

    # ---------------------------------------------------------------- main entry
    def run(self, ctx: ResolvedQueryContext, session: Session | None = None) -> R.Answer:
        plan = None
        try:
            plan = self.plan(ctx)
            if self.schema_version < 2:
                plan.notes.append("Track A schema v1: label/time indexes are missing, queries may be slow (run Track A once to migrate)")
            if plan.semantic_query and (self.embedder is None or self.vectors is None or "track_clip" not in self.vectors.col):
                raise RuntimeError("visual (description) search is unavailable: text embedder or the track_clip collection could not be opened")
            ans = self._execute(plan)
        except PlanError as e:
            log.warning("query plan rejected: %s", e)
            return R.system_error(plan, f"the query could not be turned into a valid plan ({e})")
        except Exception as e:                                       # noqa: BLE001 - never fabricate an answer on failure
            log.exception("answering failed")
            return R.system_error(plan, f"{type(e).__name__}: {e}")
        if self.cfg.llm_wording and self.llm is not None:
            ans = R.polish(ans, self.llm, self._cameras())
        if session is not None:                                      # so "he" / "that car" can refer to what we just showed
            seen = [t for t in ans.tracklets if t.global_id] or []
            for t in reversed(seen[:5]):
                session.remember(t.global_id, t.reid_class, t.label)
            if ans.path is not None:
                session.remember(ans.path.global_id, ans.path.global_id.split("-")[0])
        return ans

    def _execute(self, plan: QueryPlan) -> R.Answer:
        if plan.operation != Op.path:
            return R.build(plan, self.retriever.run(plan), None, self.cfg)
        i = plan.identity_filter
        gid = i.global_id if i else None
        if i and i.track_ref:
            row = self.conn.execute("SELECT global_id FROM tracks WHERE camera_id=? AND stream_epoch=? AND track_id=?", i.track_ref).fetchone()
            gid = row[0]
            if gid is None:                                          # a single-camera track without a cross-camera identity
                plan = plan.model_copy(update={"operation": Op.find_tracks, "reconstruction_required": False,
                                               "notes": plan.notes + ["this track has no cross-camera identity, so only this single sighting can be reported, not a path"]})
                a = R.build(plan, self.retriever.run(plan), None, self.cfg)
                a.status = R.Status.PARTIAL_EVIDENCE if a.status == R.Status.OK else a.status
                return a
        if gid is None:                                              # object only: ask which individual
            cands = self.retriever.run(plan.model_copy(update={"operation": Op.find_tracks, "reconstruction_required": False, "limit": 50}))
            seen, out = set(), []
            for t in cands.tracklets:
                if t.global_id and t.global_id not in seen and len(out) < 5:
                    seen.add(t.global_id); out.append((t.global_id, t.camera_id, t.last_seen_ts))
            return R.build(plan, None, None, self.cfg, candidates=out)
        retr = self.retriever
        path = reconstruct(gid, retr, self.cfg, self.topology, plan.time_filter, tuple(plan.camera_filter))
        a = R.build(plan, None, path, self.cfg)
        return a
