"""Retrieval Engine: hybrid SQLite metadata filtering + ChromaDB vector ranking, strictly read-only on Track A's stores.
SQL is assembled ONLY from constant fragments; every value from the plan is bound as a parameter, and ORDER BY comes from an
allow-list. All queries are index-backed (tests assert EXPLAIN QUERY PLAN shows no table scan)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import AnswerConfig
from .plan import Op, QueryPlan, Ranking

_ORDER = {Ranking.recency: "t.last_seen_ts DESC", Ranking.duration: "(t.last_seen_ts - t.start_ts) DESC",
          Ranking.confidence: "t.hits DESC", Ranking.semantic: "t.last_seen_ts DESC"}
_COLS = ("t.track_row, t.camera_id, t.stream_epoch, t.track_id, t.label, t.reid_class, t.start_ts, t.last_seen_ts, t.end_ts, "
         "t.hits, t.global_id, t.identity_status, t.identity_score, t.embedding_id")
CHUNK = 400


@dataclass
class Evidence:
    camera_id: str
    ts: float
    confidence: float
    bbox: tuple
    track_ref: tuple
    global_id: str | None
    clip_id: int | None = None          # clips.clip_id (set only when the clip is playable: this is what a UI uses, never a path)
    clip_path: str | None = None        # relative to <store>/clips, only when the file exists
    clip_frame: int | None = None
    clip_status: str = "not_recorded"   # ok | not_recorded | missing_file | corrupt_file | marked_missing | marked_corrupt


@dataclass
class Tracklet:
    track_row: int
    camera_id: str
    stream_epoch: int
    track_id: int
    label: str
    reid_class: str
    start_ts: float
    last_seen_ts: float
    hits: int
    global_id: str | None
    identity_status: str
    identity_score: float | None
    mean_confidence: float | None = None
    semantic_sim: float | None = None
    evidence: Evidence | None = None

    @property
    def ref(self) -> tuple:
        return (self.camera_id, self.stream_epoch, self.track_id)


@dataclass
class Retrieved:
    tracklets: list = field(default_factory=list)
    total_matching: int = 0              # before the limit (capped by candidate_cap)
    notes: list = field(default_factory=list)
    partial: bool = False                # something requested could not be fully evaluated
    counts: dict | None = None           # count_tracks
    coverage: dict = field(default_factory=dict)


class RetrievalError(RuntimeError):
    pass


class Retriever:
    def __init__(self, cfg: AnswerConfig, conn: sqlite3.Connection, vectors, embedder, clips_dir: Path):
        self.cfg, self.conn, self.vectors, self.embedder, self.clips_dir = cfg, conn, vectors, embedder, clips_dir

    # ---------------------------------------------------------------- where clause
    def _where(self, plan: QueryPlan) -> tuple[str, list]:
        w, a = [], []
        if plan.camera_filter:
            w.append("t.camera_id IN (%s)" % ",".join("?" * len(plan.camera_filter))); a += plan.camera_filter
        if plan.time_filter:
            w.append("t.start_ts <= ? AND t.last_seen_ts >= ?"); a += [plan.time_filter.end, plan.time_filter.start]
        if plan.object_filter:
            w.append("t.label = ?"); a.append(plan.object_filter.label)
        i = plan.identity_filter
        if i and i.global_id:
            w.append("t.global_id = ?"); a.append(i.global_id)
        elif i and i.track_ref:
            w.append("t.camera_id = ? AND t.stream_epoch = ? AND t.track_id = ?"); a += list(i.track_ref)
        if plan.min_confidence > 0:
            w.append("(SELECT avg(o.confidence) FROM observations o WHERE o.track_row = t.track_row) >= ?"); a.append(plan.min_confidence)
        return (" WHERE " + " AND ".join(w)) if w else "", a

    # ---------------------------------------------------------------- operations
    def run(self, plan: QueryPlan) -> Retrieved:
        if plan.operation == Op.count_tracks:
            return self._count(plan)
        r = self._tracks(plan)
        if plan.operation == Op.locate_latest:
            r.tracklets = r.tracklets[:1] if plan.identity_filter else r.tracklets
        return r

    def _count(self, plan: QueryPlan) -> Retrieved:
        where, args = self._where(plan)
        row = self.conn.execute("SELECT count(*), count(DISTINCT t.global_id), coalesce(sum(t.global_id IS NULL), 0) "
                                "FROM tracks t" + where, args).fetchone()
        r = Retrieved(total_matching=row[0], counts={"tracks": row[0], "identities": row[1], "unidentified_tracks": int(row[2])})
        r.coverage = self.coverage(plan)
        return r

    def _tracks(self, plan: QueryPlan) -> Retrieved:
        where, args = self._where(plan)
        sql = f"SELECT {_COLS}, (SELECT avg(o.confidence) FROM observations o WHERE o.track_row = t.track_row) AS mean_conf " \
              f"FROM tracks t{where} ORDER BY {_ORDER[plan.ranking_strategy]} LIMIT ?"
        rows = self.conn.execute(sql, [*args, self.cfg.candidate_cap]).fetchall()
        out = Retrieved(total_matching=len(rows), coverage=self.coverage(plan))
        if len(rows) == self.cfg.candidate_cap:
            out.notes.append(f"candidate set capped at {self.cfg.candidate_cap} tracks; narrow the camera/time filters")
            out.partial = True
        tl = [Tracklet(*(r[k] for k in ("track_row", "camera_id", "stream_epoch", "track_id", "label", "reid_class", "start_ts",
                                        "last_seen_ts", "hits", "global_id", "identity_status", "identity_score")),
                       mean_confidence=r["mean_conf"]) for r in rows]
        emb = {t.track_row: r["embedding_id"] for t, r in zip(tl, rows)}
        if plan.spatial_filter:
            tl = self._spatial(plan, tl, out)
        if plan.semantic_query and tl:
            tl = self._semantic(plan, tl, emb, out)
        elif plan.ranking_strategy == Ranking.confidence:
            tl.sort(key=lambda t: -(t.mean_confidence or 0))
        out.total_matching = len(tl)
        out.tracklets = tl[:plan.limit]
        for t in out.tracklets:
            t.evidence = self.best_evidence(t)
        return out

    # ---------------------------------------------------------------- spatial filter
    def camera_dims(self, camera_id: str):
        r = self.conn.execute("SELECT width, height FROM clips WHERE camera_id=? ORDER BY end_ts DESC LIMIT 1", (camera_id,)).fetchone()
        return (r[0], r[1]) if r else None

    def _spatial(self, plan, tl, out):
        keep, unknown = [], set()
        for cam in {t.camera_id for t in tl}:
            regs = [r for r in plan.spatial_filter if r.camera_id in (None, cam)]
            rows = [t for t in tl if t.camera_id == cam]
            if not regs:
                keep += rows                       # the region constrains other cameras only
                continue
            dims = self.camera_dims(cam)
            if dims is None:
                unknown.add(cam)
                continue
            hit = set()
            for i in range(0, len(rows), CHUNK):
                ids = [t.track_row for t in rows[i:i + CHUNK]]
                for r in regs:
                    q = ("SELECT DISTINCT track_row FROM observations WHERE track_row IN (%s) AND (x1+x2)/2 BETWEEN ? AND ? "
                         "AND (y1+y2)/2 BETWEEN ? AND ?" % ",".join("?" * len(ids)))
                    hit |= {x[0] for x in self.conn.execute(q, [*ids, r.x1 * dims[0], r.x2 * dims[0], r.y1 * dims[1], r.y2 * dims[1]])}
            keep += [t for t in rows if t.track_row in hit]
        if unknown:
            out.partial = True
            out.notes.append(f"spatial filter could not be evaluated for {sorted(unknown)} (no frame size recorded); those cameras' tracks were excluded")
        keep.sort(key=lambda t: -t.last_seen_ts)
        return keep

    # ---------------------------------------------------------------- semantic ranking (ChromaDB)
    def _semantic(self, plan, tl, emb, out):
        q = self.embedder.embed([plan.semantic_query.text])[0]
        ids = [emb[t.track_row] for t in tl if emb[t.track_row]]
        if len(ids) < len(tl):
            out.partial = True
            out.notes.append(f"{len(tl) - len(ids)} matching track(s) have no stored visual embedding yet and could not be ranked by description")
        if not ids:
            return []
        sims = {}
        for i in range(0, len(ids), CHUNK):
            part = ids[i:i + CHUNK]
            r = self.vectors.col["track_clip"].query(query_embeddings=[q.tolist()], n_results=len(part), ids=part, include=["distances"])
            sims.update({k: 1.0 - d for k, d in zip(r["ids"][0], r["distances"][0])})
        ranked = []
        for t in tl:
            e = emb[t.track_row]
            if e in sims:
                t.semantic_sim = float(sims[e])
                ranked.append(t)
        ranked.sort(key=lambda t: -t.semantic_sim)
        out.notes.append("ordered by visual similarity to the description (CLIP cosine; a relative ranking, not a probability)")
        return ranked

    # ---------------------------------------------------------------- evidence
    def best_evidence(self, t: Tracklet) -> Evidence | None:
        r = self.conn.execute(
            "SELECT o.ts, o.confidence, o.x1, o.y1, o.x2, o.y2, o.clip_id, o.clip_frame, c.path, c.status, c.size_bytes FROM observations o "
            "LEFT JOIN clips c ON c.clip_id = o.clip_id WHERE o.track_row = ? ORDER BY o.confidence DESC LIMIT 1", (t.track_row,)).fetchone()
        if r is None:
            return None
        ev = Evidence(t.camera_id, r["ts"], r["confidence"], (r["x1"], r["y1"], r["x2"], r["y2"]), t.ref, t.global_id)
        if r["clip_id"] is not None:
            if r["status"] != "complete":
                ev.clip_status = f"marked_{r['status']}"
            elif not (self.clips_dir / r["path"]).is_file():
                ev.clip_status = "missing_file"
            elif (self.clips_dir / r["path"]).stat().st_size != r["size_bytes"]:
                ev.clip_status = "corrupt_file"
            else:
                ev.clip_status, ev.clip_path, ev.clip_frame, ev.clip_id = "ok", r["path"], r["clip_frame"], r["clip_id"]
        return ev

    def identity_tracks(self, global_id: str) -> list[Tracklet]:
        rows = self.conn.execute(f"SELECT {_COLS}, (SELECT avg(o.confidence) FROM observations o WHERE o.track_row = t.track_row) AS mean_conf "
                                 "FROM tracks t WHERE t.global_id = ? ORDER BY t.start_ts LIMIT ?", (global_id, self.cfg.candidate_cap)).fetchall()
        return [Tracklet(*(r[k] for k in ("track_row", "camera_id", "stream_epoch", "track_id", "label", "reid_class", "start_ts",
                                          "last_seen_ts", "hits", "global_id", "identity_status", "identity_score")),
                         mean_confidence=r["mean_conf"]) for r in rows]

    def end_positions(self, track_row: int) -> tuple:
        a = self.conn.execute("SELECT ts, (x1+x2)/2 AS cx, (y1+y2)/2 AS cy FROM observations WHERE track_row = ? ORDER BY ts ASC LIMIT 1", (track_row,)).fetchone()
        b = self.conn.execute("SELECT ts, (x1+x2)/2 AS cx, (y1+y2)/2 AS cy FROM observations WHERE track_row = ? ORDER BY ts DESC LIMIT 1", (track_row,)).fetchone()
        return (tuple(a) if a else None, tuple(b) if b else None)

    def coverage(self, plan: QueryPlan) -> dict:
        """Per camera: did the system have the camera online during the requested time window (cameras.first_seen..last_seen)?"""
        if plan.camera_filter:
            rows = self.conn.execute("SELECT camera_id, first_seen, last_seen FROM cameras WHERE camera_id IN (%s)" %
                                     ",".join("?" * len(plan.camera_filter)), plan.camera_filter).fetchall()
        else:
            rows = self.conn.execute("SELECT camera_id, first_seen, last_seen FROM cameras").fetchall()
        out = {}
        for r in rows:
            if plan.time_filter:
                lo, hi = max(plan.time_filter.start, r["first_seen"]), min(plan.time_filter.end, r["last_seen"])
                out[r["camera_id"]] = max(0.0, hi - lo) / max(plan.time_filter.end - plan.time_filter.start, 1e-9)
            else:
                out[r["camera_id"]] = 1.0
        return out
