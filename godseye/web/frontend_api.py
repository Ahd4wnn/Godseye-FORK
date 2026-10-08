"""Adapter between the React console (frontend/, contract in frontend/README.md) and the existing Track B service.
Nothing here widens what Track B can do: questions go through QueryService (NLU -> resolver -> clarification -> planner ->
retrieval -> grounded answer); every value shown comes from SQLite rows or the answer; media is addressed by clip_id."""
from __future__ import annotations

import json
import threading
import time
import uuid

from .server_util import iso

NAME = "Godseye"
CLARIFY_WAIT_S = 600
MAX_PENDING = 64


class Pending:
    def __init__(self, sid: str, c):
        self.sid, self.c, self.ev, self.resp, self.skip = sid, c, threading.Event(), None, False


class FrontendApi:
    def __init__(self, app):
        self.app = app
        self.kn = app.qs.qu.kn
        self.ent = app.qs.qu.ent
        self._mu = threading.Lock()
        self._pending: dict[str, Pending] = {}
        self._sess: dict[str, str] = {}                      # browser session_id -> QueryService session id (bounded)

    # ------------------------------------------------------------------ read views
    def _latest_clips(self) -> dict[str, dict]:
        rows = self.app.q("SELECT clip_id,camera_id,start_ts,end_ts,fps,frames,width,height FROM clips WHERE status='complete' "
                          "AND clip_id IN (SELECT max(clip_id) FROM clips WHERE status='complete' GROUP BY camera_id)")
        return {r["camera_id"]: dict(r) for r in rows}

    def _camera_names(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for alias, target in self.kn.aliases("camera").items():
            out.setdefault(target, alias)
        return out

    def health(self) -> dict:
        q = self.app.q
        now = time.time()
        return {"ok": True, "index_size": q("SELECT count(*) n FROM tracks")[0]["n"], "facts": len(self.facts()),
                "cameras_online": q("SELECT count(*) n FROM cameras WHERE last_seen>=?", now - 120)[0]["n"],
                "capabilities": {"upload": False, "restart": False}}

    def site(self) -> dict:
        return {"name": NAME}                                # no layout: Godseye does not infer a floor plan (never invent one)

    def cameras(self) -> list[dict]:
        names = self._camera_names()
        out = []
        for cam, c in sorted(self._latest_clips().items()):
            out.append({"id": cam, "name": names.get(cam, cam), "video_url": f"/media/clip/{c['clip_id']}",
                        "thumb_url": f"/api/frame/{c['clip_id']}/0.jpg", "fps": c["fps"], "duration_s": max(c["end_ts"] - c["start_ts"], 0.001), "video_s": c["frames"] / c["fps"],
                        "start_wall": iso(c["start_ts"])})
        return out

    def facts(self) -> list[dict]:
        out = []
        names = {"camera": "camera", "identity": "identity"}
        for r in self.kn.fact_rows():
            if r["kind"] in names:
                cam = r["target"] if r["kind"] == "camera" else ((self.ent.identity(r["target"]) or {}).get("last_camera") or "")
                out.append({"id": f"{r['kind']}:{r['alias']}", "referent": r["alias"], "camera_id": cam, "source_query": r["source"],
                            "created_at": iso(r["created_ts"])})
            elif r["kind"] == "zone":
                out.append({"id": f"zone:{r['camera_id']}:{r['name']}", "referent": r["name"], "camera_id": r["camera_id"],
                            "region": [[r["x1"], r["y1"]], [r["x2"], r["y1"]], [r["x2"], r["y2"]], [r["x1"], r["y2"]]],
                            "source_query": "user", "created_at": iso(r["created_ts"])})
        return out

    def forget(self, fid: str) -> bool:
        kind, _, rest = fid.partition(":")
        if kind in ("camera", "identity"):
            return self.kn.forget(kind, rest) > 0
        if kind == "zone":
            cam, _, name = rest.partition(":")
            return self.kn.forget_zone(cam, name) > 0
        return False

    # ------------------------------------------------------------------ evidence -> Hit
    def hit(self, ev) -> dict | None:
        if ev.clip_id is None or ev.clip_frame is None:
            return None                                      # not playable: it stays in the answer text, never as a fake clip
        c = self.app.q("SELECT fps,frames,width,height FROM clips WHERE clip_id=? AND status='complete'", ev.clip_id)
        if not c:
            return None
        c = c[0]
        cam, epoch, tid = ev.track_ref
        tr = self.app.q("SELECT track_row,label FROM tracks WHERE camera_id=? AND stream_epoch=? AND track_id=?", cam, epoch, tid)
        boxes = []
        if tr:
            for o in self.app.q("SELECT clip_frame,x1,y1,x2,y2 FROM observations WHERE track_row=? AND clip_id=? ORDER BY clip_frame",
                                tr[0]["track_row"], ev.clip_id):
                x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in (o["x1"] / c["width"], o["y1"] / c["height"], o["x2"] / c["width"], o["y2"] / c["height"]))
                boxes.append({"t": o["clip_frame"] / c["fps"], "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "track_id": ev.global_id or f"track {tid}",
                              "kind": "object", "label": tr[0]["label"]})
        return {"id": f"{ev.clip_id}:{cam}:{epoch}:{tid}", "camera_id": cam, "t_start": 0.0, "t_end": c["frames"] / c["fps"],
                "t_best": ev.clip_frame / c["fps"], "wall_time": iso(ev.ts), "score": ev.confidence, "clip_url": f"/media/clip/{ev.clip_id}",
                "thumb_url": f"/api/frame/{ev.clip_id}/{ev.clip_frame}.jpg", "label": (tr[0]["label"] if tr else "object"), "boxes": boxes}

    # ------------------------------------------------------------------ query (server-sent events)
    def query(self, body: dict, emit) -> None:
        """emit(event, data) writes one SSE event. Blocks until the answer (and any clarifications) are done."""
        text, browser = body.get("text"), str(body.get("session_id") or "")[:64]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("send {text}")
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        emit("parsed", {"query_id": uuid.uuid4().hex[:8], "terms": text.split()[:12], "referents": []})
        with self._mu:
            sid = self._sess.get(browser)
        sid, res = self.app.qs.ask(sid, text)
        with self._mu:
            self._sess[browser] = sid
            while len(self._sess) > 256:
                del self._sess[next(iter(self._sess))]
        while True:
            stages["understand+retrieve"] = res.elapsed_ms
            emit("stage", {"name": "understand+retrieve", "ms": round(res.elapsed_ms), "detail": res.kind})
            if res.kind == "clarify":
                c = res.clarification
                p = Pending(sid, c)
                with self._mu:
                    if len(self._pending) >= MAX_PENDING:
                        del self._pending[next(iter(self._pending))]
                    self._pending[c.cid] = p
                emit("clarify", self._clarify_event(c))
                got = p.ev.wait(CLARIFY_WAIT_S)
                with self._mu:
                    self._pending.pop(c.cid, None)
                if not got:
                    return emit("error", {"message": "No answer to the clarifying question; the question was dropped."})
                if p.skip:
                    emit("answer", {"text": f"Understood. I did not run the search because I still need an answer to: {c.prompt}", "hits": [], "memory_used": []})
                    break
                sid, res = self.app.qs.answer_clarification(sid, c.cid, p.resp)
                continue
            if res.kind == "answer":
                a = res.answer
                hits = [h for h in (self.hit(e) for e in a.evidence) if h]
                msg = a.text + ("".join(f"\n⚠ {w}" for w in a.warnings))
                emit("answer", {"text": f"[{a.status.value}] {msg}", "hits": hits, "memory_used": []})
            else:
                emit("error", {"message": res.message or res.kind})
            break
        emit("done", {"total_ms": round((time.perf_counter() - t0) * 1e3), "stages": stages})

    @staticmethod
    def _clarify_event(c) -> dict:
        kind = {"camera_grid": "camera", "region_selection": "region", "point_and_click": "entity"}.get(c.kind, "choice" if c.kind == "multiple_choice" else "text")
        ev = {"question_id": c.cid, "question": c.prompt, "referent": c.mention or c.slot, "kind": kind, "suggested_cameras": []}
        if c.kind == "camera_grid":
            ev["suggested_cameras"] = [o.value for o in c.options]
        elif c.camera_id:
            ev["suggested_cameras"] = [c.camera_id]
        if c.kind == "multiple_choice":
            ev["options"] = [{"value": o.value, "label": o.label} for o in c.options]
        return ev

    # ------------------------------------------------------------------ clarification answers
    def clarify(self, body: dict) -> dict | None:
        qid = body.get("question_id")
        with self._mu:
            p = self._pending.get(qid) if isinstance(qid, str) else None
        if p is None:
            raise LookupError("no such open question")
        c = p.c
        if body.get("skip") is True:
            p.skip = True
            p.ev.set()
            return None
        cam, reg, resp = body.get("camera_id"), body.get("region"), body.get("response")
        if c.kind in ("camera_grid", "multiple_choice"):
            p.resp = cam if c.kind == "camera_grid" else resp
            if not isinstance(p.resp, str):
                raise ValueError("bad answer")
        elif c.kind == "free_text":
            if not isinstance(resp, str):
                raise ValueError("send {response: text}")
            p.resp = resp
        elif c.kind == "region_selection":
            x1, y1, x2, y2 = self._rect(reg)
            p.resp = {"camera_id": cam, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        else:                                                # point_and_click: centre of the picked area at the frame the operator was looking at
            x1, y1, x2, y2 = self._rect(reg) if reg else (0.5, 0.5, 0.5, 0.5)
            clip = self._latest_clips().get(cam)
            if clip is None:
                raise ValueError("unknown camera")
            wall = body.get("wall")
            fts = json.loads(self.app.q("SELECT frame_ts FROM clips WHERE clip_id=?", clip["clip_id"])[0]["frame_ts"])
            i = 0 if not isinstance(wall, (int, float)) else max(0, min(len(fts) - 1, round((wall - clip["start_ts"]) * clip["fps"])))
            p.resp = {"camera_id": cam, "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "ts": fts[i], "frame_w": clip["width"], "frame_h": clip["height"]}
        before = {f["id"] for f in self.facts()}
        p.ev.set()
        time.sleep(0.05)
        new = [f for f in self.facts() if f["id"] not in before]
        return new[0] if new else {"id": f"ephemeral:{qid}", "referent": c.mention or c.slot, "camera_id": cam or c.camera_id or "",
                                   "source_query": "clarification", "created_at": iso(time.time())}

    @staticmethod
    def _rect(reg) -> tuple[float, float, float, float]:
        if not (isinstance(reg, list) and 2 <= len(reg) <= 8 and all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in reg)):
            raise ValueError("region must be a list of [x, y] points")
        xs, ys = [p[0] for p in reg], [p[1] for p in reg]
        return min(xs), min(ys), max(xs), max(ys)
