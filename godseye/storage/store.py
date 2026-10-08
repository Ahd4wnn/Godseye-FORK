"""EvidenceStore: EnrichedFrame -> ChromaDB + SQLite + clip files, with explicit failure/consistency behaviour.

Single writer (fcntl lock on <root>/.lock), driven from the pipeline thread.

Failure model
  * SQLite / ChromaDB write failure  -> exception to the caller (never swallowed; nothing is silently dropped).
  * Clip failure (no disk, ffmpeg dead, validation failed) -> DEGRADED, not fatal: the open segment is aborted and its
    `.part` removed, the observations it covered keep clip_id NULL, `health()['clip_failures']` is incremented and the
    error logged; that camera is not retried for `retry_after_s`.
  * Ordering: vectors first (idempotent upserts) -> clip published (validated, atomic rename) -> SQLite commit that
    references it. A crash can only leave orphans (stale .part, unreferenced clip, orphan vector, open track), never a
    reference to something that does not exist; `recover()` (on open) and `reconcile()` clean/report them.
"""
from __future__ import annotations

import fcntl
import json
import logging
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..perception.types import EnrichedFrame, TrackedObject
from .clips import ClipError, ClipWriter, probe, sha256_file
from .config import StorageConfig
from .db import DatabaseError, connect
from .vectors import PersistentGallery, VectorStore

log = logging.getLogger("godseye.storage")
_CAM = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ROOT = Path(__file__).resolve().parents[2]


class StorageError(RuntimeError):
    pass


class _Segment:
    def __init__(self, writer: ClipWriter, rel: str, epoch: int, w: int, h: int, ts: float):
        self.writer, self.rel, self.epoch, self.w, self.h = writer, rel, epoch, w, h
        self.start_ts = self.last_ts = self.last_track_ts = ts
        self.frame_ts: list[float] = []
        self.pending: list[int] = []        # obs_ids waiting for this clip to be published


class EvidenceStore:
    def __init__(self, cfg: StorageConfig, base_dir: Path = ROOT):
        self.cfg = cfg
        root = Path(cfg.root)
        self.root = root if root.is_absolute() else base_dir / root
        self.clips_dir = self.root / "clips"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.root / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise StorageError(f"{self.root} is locked by another writer process") from e
        self.conn = connect(self.root / "godseye.db")
        self.vectors = VectorStore(self.root / "chroma")
        self.gallery = PersistentGallery(cfg, self.vectors, self.conn)
        self._seg: dict[str, _Segment] = {}
        self._blocked: dict[str, float] = {}
        self._open: dict[tuple, TrackedObject] = {}          # (camera, epoch, track_id) -> last snapshot
        self.stats = {"clips_written": 0, "clip_failures": 0, "frames_written": 0, "last_clip_error": None}
        self.recovered = self.recover()

    # ------------------------------------------------------------------ transactions
    def _txn(self):
        class T:
            def __enter__(s):
                self.conn.execute("BEGIN IMMEDIATE")
            def __exit__(s, et, ev, tb):
                self.conn.execute("ROLLBACK" if et else "COMMIT")
        return T()

    # ------------------------------------------------------------------ write path
    def write(self, ef: EnrichedFrame) -> None:
        src = ef.source
        cam = src.camera_id
        if not _CAM.match(cam):
            raise StorageError(f"camera_id {cam!r} is not path-safe ([A-Za-z0-9_.-]{{1,64}})")
        ts = src.capture_ts
        has_tracks = bool(ef.tracks)
        if has_tracks and src.original_bgr is not None and src.privacy_masked and not src.original_privacy_masked:
            raise StorageError("privacy masks configured but not applied to original_bgr: refusing to store evidence")
        for t in ef.tracks:
            self._open[(cam, t.stream_epoch, t.track_id)] = t
        self._vector_writes(ef)                                         # 1. vectors (idempotent)
        seg, idx = self._clip_step(src, has_tracks)                     # 2. media (degrades, never raises)
        with self._txn():                                               # 3. metadata, one transaction per frame
            self.conn.execute("INSERT INTO cameras VALUES(?,?,?) ON CONFLICT(camera_id) DO UPDATE SET last_seen=?", (cam, ts, ts, ts))
            for t in ef.tracks:
                row = self._upsert_track(t, ts)
                d = t.detection.bbox_xyxy if t.detection is not None else (None,) * 4
                b = t.bbox_xyxy
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO observations(track_row,camera_id,frame_seq,ts,x1,y1,x2,y2,dx1,dy1,dx2,dy2,confidence,label,clip_frame)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row, cam, src.frame_seq, ts, *(float(v) for v in b), *(None if v is None else float(v) for v in d),
                     t.confidence, t.label, idx))
                if seg is not None and cur.rowcount:
                    seg.pending.append(cur.lastrowid)
            for e in ef.ended_tracks:
                self._end_track(e)
        for e in ef.ended_tracks:
            self._open.pop((e.camera_id, e.stream_epoch, e.track_id), None)
        self.stats["frames_written"] += 1

    def _vector_writes(self, ef: EnrichedFrame) -> None:
        ids, vecs, metas = [], [], []
        for t in ef.tracks:
            if t.hits % self.cfg.track_embed_every == 0:
                ids.append(self._eid(t.camera_id, t.stream_epoch, t.track_id)); vecs.append(t.clip_embedding)
                metas.append(self._emeta(t.camera_id, t.stream_epoch, t.track_id, t.global_id, t.reid_class))
        for e in ef.ended_tracks:                                       # final embedding with the final global_id
            last = self._open.get((e.camera_id, e.stream_epoch, e.track_id))
            if last is not None:
                ids.append(self._eid(e.camera_id, e.stream_epoch, e.track_id)); vecs.append(last.clip_embedding)
                metas.append(self._emeta(e.camera_id, e.stream_epoch, e.track_id, e.global_id, e.reid_class))
        if ids:
            last = {i: k for k, i in enumerate(ids)}            # one id may appear twice (live + ended in one frame): last wins
            keep = sorted(last.values())
            self.vectors.upsert("track_clip", [ids[k] for k in keep], [vecs[k] for k in keep], [metas[k] for k in keep])

    @staticmethod
    def _eid(cam, epoch, tid) -> str:
        return f"{cam}:{epoch}:{tid}"

    @staticmethod
    def _emeta(cam, epoch, tid, gid, cls) -> dict:
        return {"camera_id": cam, "stream_epoch": epoch, "track_id": tid, "global_id": gid or "", "reid_class": cls}

    def _upsert_track(self, t: TrackedObject, ts: float) -> int:
        if t.global_id:
            self.conn.execute("INSERT INTO identities VALUES(?,?,?,?,?) ON CONFLICT(global_id) DO UPDATE SET "
                              "last_seen=max(last_seen,excluded.last_seen), last_camera=excluded.last_camera",
                              (t.global_id, t.reid_class, ts, ts, t.camera_id))
        emb = self._eid(t.camera_id, t.stream_epoch, t.track_id) if t.hits >= self.cfg.track_embed_every else None
        self.conn.execute(
            "INSERT INTO tracks(camera_id,stream_epoch,track_id,label,reid_class,start_ts,last_seen_ts,hits,global_id,"
            "identity_status,identity_score,identity_candidates,embedding_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(camera_id,stream_epoch,track_id) DO UPDATE SET label=excluded.label, last_seen_ts=excluded.last_seen_ts, "
            "hits=excluded.hits, global_id=excluded.global_id, identity_status=excluded.identity_status, "
            "identity_score=excluded.identity_score, identity_candidates=excluded.identity_candidates, "
            "embedding_id=COALESCE(excluded.embedding_id, embedding_id)",
            (t.camera_id, t.stream_epoch, t.track_id, t.label, t.reid_class, t.first_seen_ts, t.last_seen_ts, t.hits,
             t.global_id, t.identity_status, t.identity_score, json.dumps(t.identity_candidates) if t.identity_candidates else None, emb))
        return self.conn.execute("SELECT track_row FROM tracks WHERE camera_id=? AND stream_epoch=? AND track_id=?",
                                 (t.camera_id, t.stream_epoch, t.track_id)).fetchone()[0]

    def _end_track(self, e) -> None:
        if e.global_id:
            self.conn.execute("INSERT OR IGNORE INTO identities VALUES(?,?,?,?,?)",
                              (e.global_id, e.reid_class, e.first_seen_ts, e.last_seen_ts, e.camera_id))
        self.conn.execute(
            "UPDATE tracks SET end_ts=?, last_seen_ts=?, hits=?, end_reason=?, global_id=COALESCE(?, global_id), identity_status=?, "
            "embedding_id=? WHERE camera_id=? AND stream_epoch=? AND track_id=?",
            (e.last_seen_ts, e.last_seen_ts, e.hits, e.reason, e.global_id, e.identity_status,
             self._eid(e.camera_id, e.stream_epoch, e.track_id) if (e.camera_id, e.stream_epoch, e.track_id) in self._open else None,
             e.camera_id, e.stream_epoch, e.track_id))

    # ------------------------------------------------------------------ clips
    def _clip_step(self, src, has_tracks: bool):
        cam, ts = src.camera_id, src.capture_ts
        if src.original_bgr is None:                       # no media to store (keep_original off): metadata only
            return None, None
        seg = self._seg.get(cam)
        if seg is not None:
            h, w = src.original_bgr.shape[:2]
            if (seg.epoch != src.stream_epoch or (seg.w, seg.h) != (w, h) or ts - seg.start_ts > self.cfg.max_segment_s
                    or (not has_tracks and ts - seg.last_track_ts > self.cfg.idle_close_s)):
                self._close(cam)
                seg = None
        if seg is None:
            if not has_tracks or src.original_bgr is None or ts < self._blocked.get(cam, 0.0):
                return None, None
            seg = self._open_segment(src)
            if seg is None:
                return None, None
        try:
            idx = seg.writer.write(src.original_bgr)
        except ClipError as e:
            self._clip_failed(cam, seg, e, ts)
            return None, None
        seg.frame_ts.append(ts)
        seg.last_ts = ts
        if has_tracks:
            seg.last_track_ts = ts
        return seg, idx

    def _open_segment(self, src):
        cam, ts = src.camera_id, src.capture_ts
        h, w = src.original_bgr.shape[:2]
        try:
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.clips_dir).free
            if free < self.cfg.min_free_bytes:
                raise ClipError(f"low disk: {free} bytes free < min_free_bytes {self.cfg.min_free_bytes}")
            day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            rel = f"{day}/{cam}/{int(ts * 1000)}_{uuid.uuid4().hex[:8]}.mp4"
            seg = _Segment(ClipWriter(self.cfg, self.clips_dir / rel, w, h), rel, src.stream_epoch, w, h, ts)
        except (ClipError, OSError) as e:
            self._clip_failed(cam, None, e, ts)
            return None
        self._seg[cam] = seg
        return seg

    def _clip_failed(self, cam: str, seg: _Segment | None, err: Exception, ts: float) -> None:
        self.stats["clip_failures"] += 1
        self.stats["last_clip_error"] = f"{cam}: {err}"
        self._blocked[cam] = ts + self.cfg.retry_after_s
        log.error("clip failure on %s (metadata continues, clip_id stays NULL): %s", cam, err)
        if seg is not None:
            seg.writer.abort()
            self._seg.pop(cam, None)
            if seg.pending:
                with self._txn():
                    self.conn.executemany("UPDATE observations SET clip_frame=NULL WHERE obs_id=? AND clip_id IS NULL",
                                          [(i,) for i in seg.pending])

    def _close(self, cam: str) -> None:
        seg = self._seg.pop(cam, None)
        if seg is None:
            return
        try:
            info = seg.writer.finish()
        except ClipError as e:
            self._seg[cam] = seg
            self._clip_failed(cam, seg, e, seg.last_ts)
            return
        with self._txn():
            cid = self.conn.execute(
                "INSERT INTO clips(camera_id,stream_epoch,start_ts,end_ts,path,fps,frames,width,height,frame_ts,size_bytes,sha256,created_ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cam, seg.epoch, seg.start_ts, seg.last_ts, seg.rel, self.cfg.clip_fps, info["frames"], info["width"],
                 info["height"], json.dumps(seg.frame_ts), info["size_bytes"], info["sha256"], time.time())).lastrowid
            self.conn.executemany("UPDATE observations SET clip_id=? WHERE obs_id=?", [(cid, i) for i in seg.pending])
        self.stats["clips_written"] += 1

    # ------------------------------------------------------------------ lifecycle
    def close_camera(self, camera_id: str) -> None:
        self._close(camera_id)

    def close(self) -> None:
        for cam in list(self._seg):
            self._close(cam)
        self.conn.close()
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()

    def health(self) -> dict:
        return {**self.stats, "open_segments": len(self._seg), "open_tracks": len(self._open), "recovered": self.recovered}

    # ------------------------------------------------------------------ recovery / reconciliation
    def recover(self) -> dict:
        """Run on open (we hold the writer lock, so nothing here is in use): drop stale .part files, close tracks left open
        by a crash, and detach observations that pointed at an unpublished clip."""
        parts = list(self.clips_dir.rglob("*.part")) if self.clips_dir.exists() else []
        for p in parts:
            p.unlink(missing_ok=True)
        with self._txn():
            stale = self.conn.execute("UPDATE tracks SET end_ts=last_seen_ts, end_reason='crash_recovered' WHERE end_ts IS NULL").rowcount
            detached = self.conn.execute("UPDATE observations SET clip_frame=NULL WHERE clip_id IS NULL AND clip_frame IS NOT NULL").rowcount
        r = {"part_files_removed": len(parts), "stale_tracks_closed": stale, "observations_detached": detached}
        if any(r.values()):
            log.warning("storage recovery after unclean shutdown: %s", r)
        return r

    def reconcile(self, fix: bool = False, deep: bool = False) -> dict:
        """Cross-check SQLite <-> ChromaDB <-> clip files. Report always; repair only with fix=True.
        deep=True re-hashes every clip. Repairs never delete evidence that SQLite still references."""
        rep = {k: [] for k in ("clips_missing_file", "clips_bad_size", "clips_bad_hash", "orphan_clip_files",
                               "orphan_vectors", "identities_without_vectors", "tracks_missing_embedding")}
        # clips
        known = {}
        for r in self.conn.execute("SELECT clip_id,path,size_bytes,sha256,status FROM clips"):
            known[r["path"]] = r["clip_id"]
            p = self.clips_dir / r["path"]
            bad = None
            if not p.is_file():
                rep["clips_missing_file"].append(r["path"]); bad = "missing"
            elif p.stat().st_size != r["size_bytes"]:
                rep["clips_bad_size"].append(r["path"]); bad = "corrupt"
            elif deep and sha256_file(p) != r["sha256"]:
                rep["clips_bad_hash"].append(r["path"]); bad = "corrupt"
            if bad and fix and r["status"] != bad:
                self.conn.execute("UPDATE clips SET status=? WHERE clip_id=?", (bad, r["clip_id"]))
        if self.clips_dir.exists():
            for p in self.clips_dir.rglob("*.mp4"):
                rel = p.relative_to(self.clips_dir).as_posix()
                if rel not in known and rel not in {s.rel for s in self._seg.values()}:
                    rep["orphan_clip_files"].append(rel)
                    if fix:
                        p.unlink()
        # vectors
        gids = {r[0] for r in self.conn.execute("SELECT global_id FROM identities")}
        have = set()
        for n in ("reid_person", "reid_vehicle"):
            g = self.vectors.get(n)
            orphan = [i for i, m in zip(g["ids"], g["metadatas"]) if m["global_id"] not in gids]
            have |= {m["global_id"] for m in g["metadatas"]}
            rep["orphan_vectors"] += [f"{n}/{i}" for i in orphan]
            if fix:
                self.vectors.delete(n, orphan)
        rep["identities_without_vectors"] = sorted(gids - have)
        tr = {self._eid(r[0], r[1], r[2]) for r in self.conn.execute("SELECT camera_id,stream_epoch,track_id FROM tracks")}
        g = self.vectors.get("track_clip")
        orphan = [i for i in g["ids"] if i not in tr]
        rep["orphan_vectors"] += [f"track_clip/{i}" for i in orphan]
        if fix:
            self.vectors.delete("track_clip", orphan)
        stored = set(g["ids"])
        rep["tracks_missing_embedding"] = [r[0] for r in self.conn.execute(
            "SELECT embedding_id FROM tracks WHERE embedding_id IS NOT NULL") if r[0] not in stored]
        return rep
