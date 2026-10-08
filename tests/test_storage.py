"""Phase 3 tests: real SQLite, real ChromaDB (persistent, on disk), real ffmpeg clips. Perception output is synthetic here
(see test_storage_real.py for the real-model pipeline -> store run)."""
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from godseye.ingest.preprocess import compute_letterbox
from godseye.ingest.types import PreprocessedFrame
from godseye.perception.config import IdentityConfig
from godseye.perception.identity import IdentityResolver
from godseye.perception.tracker import TrackMeta
from godseye.perception.types import Detection, EnrichedFrame, Provenance, TrackedObject, TrackEnded
from godseye.storage.clips import probe
from godseye.storage.config import StorageConfig, StorageConfigError, load_config, parse
from godseye.storage.store import EvidenceStore, StorageError

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed for real evidence clips")
H, W = 120, 160
PROV = Provenance("nanoowl", "t", "torch", "fp32", None, None, "x", 768)
rng = np.random.default_rng(1)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def cfg(tmp_path, **kw):
    base = dict(root=str(tmp_path / "store"), min_free_bytes=0, idle_close_s=2.0, track_embed_every=3, retry_after_s=5.0)
    return StorageConfig(**{**base, **kw})


def frame(cam="cam1", seq=0, ts=None, epoch=0, shade=None, masked=(False, False)):
    img = np.full((H, W, 3), (seq * 7) % 255 if shade is None else shade, np.uint8)
    return PreprocessedFrame(cam, seq, seq, epoch, float(seq) if ts is None else ts, None, np.zeros((1, 1, 3), np.float32),
                             compute_letterbox(W, H, 640), img, "base", None, masked[0], masked[1])


def track(cam="cam1", tid=1, seq=0, hits=None, gid=None, epoch=0, cls="person", emb=None):
    box = np.array([10 + seq, 20, 60 + seq, 100], np.float32)
    det = Detection(0, cam, seq, epoch, float(seq), None, box, cls, cls, 0.8, emb if emb is not None else unit(np.ones(512)), 0, PROV)
    return TrackedObject(cam, tid, cls, cls, "tracked", box, 0.8, det, hits or seq + 1, 0.0, float(seq), (), unit(np.ones(512)),
                         gid, "new" if gid else "unresolved", None, (), epoch)


def ended(t, reason="lost_timeout"):
    return TrackEnded(t.camera_id, t.track_id, reason, t.label, t.reid_class, t.first_seen_ts, t.last_seen_ts, t.hits,
                      t.global_id, t.identity_status, t.stream_epoch)


def ef(f, tracks=(), ends=()):
    return EnrichedFrame(f, (), tuple(tracks), tuple(ends))


def feed(store, n=6, cam="cam1", gid=None, start=0):
    last = None
    for k in range(start, start + n):
        last = track(cam, seq=k, hits=k - start + 1, gid=gid)
        store.write(ef(frame(cam, k), [last]))
    return last


def rows(store, sql, *a):
    return [dict(r) for r in store.conn.execute(sql, a)]


# ---------------------------------------------------------------- clips + metadata
def test_clip_published_with_consistent_metadata(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    last = feed(s, 6)
    s.write(ef(frame(seq=6), [], [ended(last)]))                      # track ends
    for k in (7, 8, 9):                                               # idle frames beyond idle_close_s close the segment
        s.write(ef(frame(seq=k)))
    (clip,) = rows(s, "SELECT * FROM clips")
    p = s.clips_dir / clip["path"]
    assert clip["path"].count("/") == 2 and clip["path"].split("/")[1] == "cam1"          # {date}/{camera}/{file}
    info = probe(s.cfg, p)
    assert info["codec"] == "h264" and info["frames"] == clip["frames"] == len(json.loads(clip["frame_ts"]))
    assert clip["size_bytes"] == p.stat().st_size and clip["status"] == "complete"
    obs = rows(s, "SELECT * FROM observations ORDER BY frame_seq")
    assert len(obs) == 6 and all(o["clip_id"] == clip["clip_id"] and o["clip_frame"] == i for i, o in enumerate(obs))
    (tr,) = rows(s, "SELECT * FROM tracks")
    assert tr["end_reason"] == "lost_timeout" and tr["end_ts"] is not None and tr["hits"] == 6
    assert not list(s.clips_dir.rglob("*.part"))
    assert all(not v for v in s.reconcile().values())                 # healthy store reconciles clean
    s.close()


def test_clip_frame_matches_observation_time(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    feed(s, 4)
    s.close()
    s = EvidenceStore(cfg(tmp_path))
    (clip,) = rows(s, "SELECT * FROM clips")
    ts = json.loads(clip["frame_ts"])
    for o in rows(s, "SELECT ts, clip_frame FROM observations"):
        assert ts[o["clip_frame"]] == o["ts"]
    s.close()


def test_stream_epoch_change_closes_segment(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    s.write(ef(frame(seq=0), [track(seq=0)]))
    s.write(ef(frame(seq=1, epoch=1), [track(seq=1, epoch=1)]))
    s.close()
    assert len(rows(EvidenceStore(cfg(tmp_path)), "SELECT 1 FROM clips")) == 2


# ---------------------------------------------------------------- persistence / restart / vectors
def test_restart_persists_everything(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    g = s.gallery.new("person", unit(rng.normal(size=2048)), "cam1", 1.0)
    last = feed(s, 6, gid=g)
    s.write(ef(frame(seq=6), [], [ended(last, "shutdown")]))           # what pipeline.shutdown() hands the store
    s.close()
    s2 = EvidenceStore(cfg(tmp_path))
    assert s2.recovered["stale_tracks_closed"] == 0
    assert len(rows(s2, "SELECT 1 FROM observations")) == 6 and len(s2.gallery) == 1
    assert rows(s2, "SELECT global_id FROM tracks")[0]["global_id"] == g
    assert s2.vectors.count("track_clip") == 1 and s2.vectors.count("reid_person") == 1
    assert all(not v for v in s2.reconcile().values())
    s2.close()


def test_vector_search_and_persistent_gallery_resolver(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    res = IdentityResolver(IdentityConfig(), s.gallery)
    a, b = unit(rng.normal(size=2048)), unit(rng.normal(size=2048))
    ms = []
    for cam, base in (("c1", a), ("c2", b)):
        m = TrackMeta(cam, 1, "person", 0, 0)
        for _ in range(4):
            res.observe(m, unit(base + 0.02 * rng.normal(size=2048)), 1.0)
        ms.append(m)
    assert ms[0].global_id != ms[1].global_id and len(s.gallery) == 2
    s.close()
    s = EvidenceStore(cfg(tmp_path))                                   # restart: identities must still resolve
    res = IdentityResolver(IdentityConfig(), s.gallery)
    m = TrackMeta("c3", 9, "person", 0, 0)
    for _ in range(4):
        res.observe(m, unit(a + 0.02 * rng.normal(size=2048)), 2.0)
    assert m.identity_status == "matched" and m.global_id == ms[0].global_id
    hit = s.vectors.search("reid_person", a, 3)
    assert hit[0][2]["global_id"] == ms[0].global_id and hit[0][1] > hit[-1][1]
    s.close()


def test_exemplar_cap_and_track_embedding_search(tmp_path):
    s = EvidenceStore(cfg(tmp_path, max_exemplars=4))
    g = s.gallery.new("person", unit(rng.normal(size=2048)), "c1", 0.0)
    for k in range(1, 10):
        s.gallery.add("person", g, unit(rng.normal(size=2048)), "c1", float(k))
    assert s.vectors.count("reid_person") == 4
    e1, e2 = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    for tid, e in ((1, e1), (2, e2)):
        for k in range(3):
            t = track(tid=tid, seq=k, hits=k + 1, emb=e)
            object.__setattr__(t, "clip_embedding", e)
            s.write(ef(frame(seq=k + 10 * tid), [t]))
    top = s.vectors.search("track_clip", e2, 2)
    assert top[0][2]["track_id"] == 2 and top[0][1] > 0.99
    s.close()


# ---------------------------------------------------------------- duplicates / safety
def test_duplicate_writes_are_idempotent_and_single_writer(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    t = track(seq=0, hits=3)
    for _ in range(3):
        s.write(ef(frame(seq=0), [t]))
    assert len(rows(s, "SELECT 1 FROM observations")) == 1 and len(rows(s, "SELECT 1 FROM tracks")) == 1
    assert s.vectors.count("track_clip") == 1
    with pytest.raises(StorageError, match="locked"):
        EvidenceStore(cfg(tmp_path))
    s.close()


def test_unsafe_camera_id_and_unmasked_original_refused(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    with pytest.raises(StorageError, match="path-safe"):
        s.write(ef(frame(cam="../evil"), [track(cam="../evil")]))
    with pytest.raises(StorageError, match="privacy"):
        s.write(ef(frame(masked=(True, False)), [track()]))
    assert not rows(s, "SELECT 1 FROM observations")
    s.close()


# ---------------------------------------------------------------- failures
def test_missing_ffmpeg_degrades_but_keeps_metadata(tmp_path):
    s = EvidenceStore(cfg(tmp_path, ffmpeg="/nonexistent/ffmpeg"))
    feed(s, 4)
    h = s.health()
    assert h["clip_failures"] >= 1 and "ffmpeg" in h["last_clip_error"] and h["clips_written"] == 0
    obs = rows(s, "SELECT clip_id, clip_frame FROM observations")
    assert len(obs) == 4 and all(o["clip_id"] is None and o["clip_frame"] is None for o in obs)
    assert not rows(s, "SELECT 1 FROM clips")
    s.close()


def test_low_disk_blocks_clips_then_retries_after_backoff(tmp_path):
    s = EvidenceStore(cfg(tmp_path, min_free_bytes=10**15, retry_after_s=3.0))
    feed(s, 3)
    assert s.health()["clip_failures"] == 1 and "low disk" in s.health()["last_clip_error"]     # not once per frame
    s.cfg = cfg(tmp_path, min_free_bytes=0, retry_after_s=3.0)
    s.write(ef(frame(seq=3, ts=2.5), [track(seq=3)]))
    assert not s._seg                                                                         # still in back-off
    feed(s, 3, start=10)
    s.close()
    assert len(rows(EvidenceStore(cfg(tmp_path)), "SELECT 1 FROM clips")) == 1


def test_encoder_dies_mid_segment(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    feed(s, 3)
    s._seg["cam1"].writer.proc.kill()
    feed(s, 3, start=3)
    assert s.health()["clip_failures"] >= 1 and not list(s.clips_dir.rglob("*.part"))
    assert not rows(s, "SELECT 1 FROM observations WHERE clip_id IS NULL AND clip_frame IS NOT NULL AND frame_seq<3")
    s.close()
    assert not rows(EvidenceStore(cfg(tmp_path)), "SELECT 1 FROM clips WHERE frames=0")


def test_vector_failure_raises_and_leaves_no_metadata(tmp_path, monkeypatch):
    s = EvidenceStore(cfg(tmp_path))
    monkeypatch.setattr(s.vectors, "upsert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chroma down")))
    with pytest.raises(RuntimeError, match="chroma down"):
        s.write(ef(frame(seq=0), [track(seq=0, hits=3)]))
    assert not rows(s, "SELECT 1 FROM observations") and not rows(s, "SELECT 1 FROM tracks")
    monkeypatch.undo()
    s.close()


def test_sqlite_failure_raises_and_rolls_back(tmp_path, monkeypatch):
    s = EvidenceStore(cfg(tmp_path))
    s.write(ef(frame(seq=0), [track(seq=0)]))
    bad = track(seq=1)
    object.__setattr__(bad, "reid_class", "bogus")                    # violates the identities CHECK via global_id path
    object.__setattr__(bad, "global_id", "person-x")
    with pytest.raises(sqlite3.Error):
        s.write(ef(frame(seq=1), [bad]))
    assert len(rows(s, "SELECT 1 FROM observations")) == 1            # whole frame rolled back
    s.close()


# ---------------------------------------------------------------- crash recovery / reconciliation
def test_crash_recovery_and_orphans(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    feed(s, 4)                                                         # segment still open, track still open
    part = next(s.clips_dir.rglob("*.part"))
    (s.clips_dir / "2000-01-01" / "cam1").mkdir(parents=True)
    stray = s.clips_dir / "2000-01-01" / "cam1" / "stray.mp4"
    stray.write_bytes(b"not a clip")
    s._seg["cam1"].writer.abort()                                      # simulate kill -9: no clean close, lock released
    part.write_bytes(b"half-written")                                    # what a killed ffmpeg leaves behind
    s._lock.close()
    s2 = EvidenceStore(cfg(tmp_path))
    assert s2.recovered["part_files_removed"] == 1 and s2.recovered["stale_tracks_closed"] == 1 and s2.recovered["observations_detached"] == 4
    assert not part.exists()
    assert rows(s2, "SELECT end_reason FROM tracks")[0]["end_reason"] == "crash_recovered"
    assert not rows(s2, "SELECT 1 FROM observations WHERE clip_frame IS NOT NULL")
    rep = s2.reconcile()
    assert rep["orphan_clip_files"] == ["2000-01-01/cam1/stray.mp4"] and stray.exists()      # report only
    s2.reconcile(fix=True)
    assert not stray.exists()
    s2.close()


def test_missing_and_corrupt_media_detected_not_silently_served(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    feed(s, 4, cam="a")
    feed(s, 4, cam="b")
    s.close()
    s = EvidenceStore(cfg(tmp_path))
    pa, pb = (s.clips_dir / r["path"] for r in rows(s, "SELECT path FROM clips ORDER BY camera_id"))
    pa.unlink()
    pb.write_bytes(pb.read_bytes()[:-50])
    rep = s.reconcile()
    assert len(rep["clips_missing_file"]) == 1 and len(rep["clips_bad_size"]) == 1
    assert {r["status"] for r in rows(s, "SELECT status FROM clips")} == {"complete"}     # report-only did not touch rows
    s.reconcile(fix=True)
    assert sorted(r["status"] for r in rows(s, "SELECT status FROM clips")) == ["corrupt", "missing"]
    s.close()


def test_deep_check_detects_same_size_corruption(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    feed(s, 4)
    s.close()
    s = EvidenceStore(cfg(tmp_path))
    p = s.clips_dir / rows(s, "SELECT path FROM clips")[0]["path"]
    b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))
    assert not s.reconcile()["clips_bad_hash"] and len(s.reconcile(deep=True)["clips_bad_hash"]) == 1
    s.close()


def test_orphan_vectors_found_and_removed(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    g = s.gallery.new("person", unit(rng.normal(size=2048)), "c1", 1.0)
    s.vectors.upsert("reid_person", ["person-ghost:1"], [unit(rng.normal(size=2048))], [{"global_id": "person-ghost", "camera_id": "c", "ts": 0.0}])
    s.vectors.upsert("track_clip", ["c:0:99"], [unit(rng.normal(size=512))], [{"camera_id": "c", "stream_epoch": 0, "track_id": 99, "global_id": "", "reid_class": "person"}])
    rep = s.reconcile()
    assert sorted(rep["orphan_vectors"]) == ["reid_person/person-ghost:1", "track_clip/c:0:99"]
    s.reconcile(fix=True)
    assert all(not v for v in s.reconcile().values()) and len(s.gallery) == 1
    s.conn.execute("DELETE FROM identities")                               # opposite direction: row lost -> reported
    assert s.reconcile()["orphan_vectors"] and g
    s.close()


def test_config_validation():
    with pytest.raises(StorageConfigError):
        parse({"bogus": 1})
    with pytest.raises(StorageConfigError):
        parse({"crf": 99})
    assert load_config("config/storage.yaml").root == "data/store"


def test_track_live_and_ended_in_the_same_frame_does_not_break_vector_upsert(tmp_path):
    s = EvidenceStore(cfg(tmp_path))
    last = feed(s, 2)
    t = track(seq=2, hits=3, gid=None)                                  # hits % track_embed_every == 0 AND ends this frame
    s.write(ef(frame(seq=2), [t], [ended(t)]))
    assert s.vectors.count("track_clip") == 1
    s.close()
