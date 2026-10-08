"""A small but complete Track A store built with the REAL EvidenceStore (SQLite + ChromaDB + ffmpeg clips) and synthetic
perception output, shaped to exercise Phase 5: multi-camera paths (plausible / overlapping / impossible / long gap), vehicles
with distinct visual embeddings, an unidentified track and a low-confidence track."""
from __future__ import annotations

import numpy as np

from godseye.ingest.preprocess import compute_letterbox
from godseye.ingest.types import PreprocessedFrame
from godseye.perception.types import Detection, EnrichedFrame, Provenance, TrackedObject, TrackEnded
from godseye.storage.config import StorageConfig
from godseye.storage.store import EvidenceStore

T0 = 1_700_000_000.0
H, W = 120, 160
P1, P2, P3, P4 = ("person-111111111111", "person-222222222222", "person-333333333333", "person-444444444444")
C1, C2 = "vehicle-aaaaaaaaaaaa", "vehicle-bbbbbbbbbbbb"
PROV = Provenance("nanoowl", "t", "torch", "fp32", None, None, "x", 768)
TOPOLOGY = """links:
  - {a: lobby_cam, b: cam2, min_travel_s: 8, max_travel_s: 90}
  - {a: cam2, b: north_gate, min_travel_s: 0, max_travel_s: 20, overlap: true}
"""

# (camera, track_id, global_id, label, reid_class, start offset, end offset, confidence, text for the visual embedding)
TRACKS = [
    ("lobby_cam", 1, P1, "person", "person", 0, 20, .85, "a person"),
    ("cam2", 2, P1, "person", "person", 40, 60, .85, "a person"),
    ("north_gate", 3, P1, "person", "person", 62, 80, .85, "a person"),
    ("lobby_cam", 4, P2, "person", "person", 100, 110, .85, "a person"),
    ("cam2", 5, P2, "person", "person", 103, 120, .85, "a person"),          # overlaps P2 on lobby_cam; cameras not overlapping
    ("lobby_cam", 6, P3, "person", "person", 200, 210, .85, "a person"),
    ("cam2", 7, P3, "person", "person", 213, 225, .85, "a person"),          # 3 s lobby->cam2, fastest route 8 s
    ("cam2", 8, C1, "car", "vehicle", 300, 320, .9, "a red car"),
    ("cam2", 9, C2, "car", "vehicle", 330, 350, .9, "a blue car"),
    ("lobby_cam", 10, None, "person", "person", 400, 410, .8, "a person"),    # no cross-camera identity
    ("lobby_cam", 11, P4, "person", "person", 500, 510, .85, "a person"),
    ("lobby_cam", 12, P4, "person", "person", 1500, 1510, .85, "a person"),   # 990 s later: new episode
    ("north_gate", 13, None, "person", "person", 600, 606, .2, "a person"),   # low confidence
]


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def build(root, embedder, max_segment_s=30.0):
    """-> EvidenceStore (closed) at `root`; returns the config used."""
    cfg = StorageConfig(root=str(root), min_free_bytes=0, idle_close_s=2.0, max_segment_s=max_segment_s, track_embed_every=3)
    store = EvidenceStore(cfg)
    emb = {t: unit(embedder.embed([t])[0]) for t in {x[-1] for x in TRACKS}}
    timeline: dict[float, list] = {}
    for cam, tid, gid, label, cls, a, b, conf, text in TRACKS:
        for k, off in enumerate(range(a, b + 1, 1)):
            timeline.setdefault(T0 + off, []).append((cam, tid, gid, label, cls, a, conf, text, k + 1, off, b))
    ends = {}
    for ts in sorted(timeline):
        by_cam: dict[str, list] = {}
        for cam, tid, gid, label, cls, a, conf, text, hits, off, b in timeline[ts]:
            box = np.array([10 + hits, 20, 60 + hits, 100], np.float32)
            det = Detection(0, cam, int(ts), 0, ts, None, box, label, cls, conf, emb[text], 0, PROV)
            t = TrackedObject(cam, tid, label, cls, "tracked", box, conf, det, hits, T0 + a, ts, (), emb[text], gid,
                              "matched" if gid else "unresolved", None, (), 0)
            by_cam.setdefault(cam, []).append(t)
            if off == b:
                ends.setdefault((cam, ts), []).append(TrackEnded(cam, tid, "lost_timeout", label, cls, T0 + a, ts, hits, gid,
                                                                  t.identity_status, 0))
        for cam, ts_ in by_cam.items():
            img = np.full((H, W, 3), int(ts) % 200 + 20, np.uint8)
            f = PreprocessedFrame(cam, int(ts), int(ts), 0, ts, None, np.zeros((1, 1, 3), np.float32), compute_letterbox(W, H, 640), img, "base", None)
            store.write(EnrichedFrame(f, (), tuple(ts_), tuple(ends.get((cam, ts), ()))))
    store.close()
    return cfg
