"""Phase 2 logic tests with synthetic detections/features (the tracker is the REAL ByteTrack; only inputs are synthetic).
Real-model tests live in test_perception_real.py."""
import numpy as np
import pytest

from godseye.ingest.preprocess import compute_letterbox
from godseye.ingest.types import PreprocessedFrame
from godseye.perception.config import IdentityConfig, PerceptionConfigError, TrackerConfig, load_config, parse
from godseye.perception.identity import IdentityResolver
from godseye.perception.pipeline import PerceptionPipeline
from godseye.perception.tracker import CameraTracker, TrackMeta
from godseye.perception.types import Detection, Provenance

PROV = Provenance("nanoowl", "t", "torch", "fp32", None, None, "x", 768)
H, W = 480, 640
rng = np.random.default_rng(0)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def det(box, cls="person", conf=0.8, cam="c1", i=0, emb=None):
    return Detection(i, cam, 0, 0, 0.0, None, np.array(box, np.float32), cls if cls != "vehicle" else "car", cls, conf,
                     emb if emb is not None else unit(rng.normal(size=512)), 0, PROV)


def run(tr, frames):
    """frames: list of lists of detections -> list of (live ids, ended ids)"""
    out = []
    for k, ds in enumerate(frames):
        live, ended = tr.update(tuple(ds), H, W, float(k))
        out.append(({m.track_id for m, *_ in live}, {m.track_id for m, _ in ended}))
    return out


def mover(x0, dx=4):
    return lambda k: [x0 + dx * k, 100, x0 + dx * k + 60, 220]


def test_lifecycle_stable_id_and_two_objects():
    tr = CameraTracker("c1", TrackerConfig(), 0)
    a, b = mover(50), mover(300)
    res = run(tr, [[det(a(k)), det(b(k))] for k in range(8)])
    ids = res[-1][0]
    assert len(ids) == 2 and all(r[0] == ids for r in res[2:])      # ids never change
    m = next(iter(tr.meta.values()))
    assert m.hits >= 7 and len(m.trajectory) >= 7 and m.trajectory[-1][1] > m.trajectory[0][1]


def test_occlusion_gap_keeps_id_and_long_gap_ends_track():
    tr = CameraTracker("c1", TrackerConfig(), 0)
    a = mover(50)
    res = run(tr, [[det(a(k))] for k in range(6)] + [[] for _ in range(3)] + [[det(a(k))] for k in range(9, 12)])
    tid = next(iter(res[5][0]))
    assert res[-1][0] == {tid}                                   # re-found after a 3-frame occlusion
    tr = CameraTracker("c1", TrackerConfig(), 0)
    res = run(tr, [[det(a(k))] for k in range(6)] + [[] for _ in range(10)] + [[det(a(k)) for _ in [0]] for k in range(16, 20)])
    tid = next(iter(res[5][0]))
    assert any(tid in e for _, e in res)                         # timed out -> ended once
    assert tid not in res[-1][0] and res[-1][0]                  # the returning object got a new camera-local id


def test_person_and_vehicle_never_share_a_track():
    tr = CameraTracker("c1", TrackerConfig(), 0)
    res = run(tr, [[det([100, 100, 160, 220], "person"), det([100, 100, 160, 220], "vehicle")] for _ in range(5)])
    assert len(res[-1][0]) == 2
    assert {m.reid_class for m in tr.meta.values()} == {"person", "vehicle"}


def test_track_ids_are_camera_local():
    a, b = CameraTracker("c1", TrackerConfig(), 0), CameraTracker("c2", TrackerConfig(), 0)
    ra = run(a, [[det(mover(50)(k))] for k in range(4)])
    rb = run(b, [[det(mover(50)(k), cam="c2")] for k in range(4)])
    assert ra[-1][0] == rb[-1][0] == {1}          # same local id on two cameras: it is NOT a global identity


# ---- identity -----------------------------------------------------------------------------------------------
def meta(cam, tid, cls="person"):
    return TrackMeta(cam, tid, cls, 0.0, 0.0)


def person_feats(base, n=4, noise=0.02):
    return [unit(base + noise * rng.normal(size=512)) for _ in range(n)]


def feed(res, m, feats):
    for f in feats:
        res.observe(m, f, 1.0)


def test_identity_new_then_cross_camera_match():
    res = IdentityResolver(IdentityConfig())
    p = unit(rng.normal(size=512))
    m1 = meta("c1", 1); feed(res, m1, person_feats(p))
    assert m1.identity_status == "new" and m1.global_id.startswith("person-")
    m2 = meta("c2", 7); feed(res, m2, person_feats(p))
    assert m2.identity_status == "matched" and m2.global_id == m1.global_id
    assert m2.track_id != m1.global_id                                  # track_id and global_id are different things


def test_identity_different_people_not_merged():
    res = IdentityResolver(IdentityConfig())
    ms = [meta("c1", i) for i in range(5)]
    for m in ms:
        feed(res, m, person_feats(unit(rng.normal(size=512))))
    assert len({m.global_id for m in ms}) == 5 and all(m.identity_status == "new" for m in ms)


def test_identity_ambiguous_is_not_merged_and_rechecked_once():
    res = IdentityResolver(IdentityConfig())
    p, q = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    for cam, base in (("c1", p), ("c2", q)):
        feed(res, meta(cam, 1), []) ; m = meta(cam, 1); feed(res, m, person_feats(base))
    amb = unit(p + q)                                          # equally similar (~0.7) to both identities
    m3 = meta("c3", 1); feed(res, m3, person_feats(amb, noise=0.01))
    assert m3.identity_status == "ambiguous" and m3.global_id is None and len(m3.identity_candidates) == 2
    n = len(res.gallery)
    feed(res, m3, person_feats(amb, n=10, noise=0.01))
    assert m3.global_id is None and len(res.gallery) == n      # still never force-merged / no new identity invented


def test_same_camera_simultaneous_tracks_never_share_identity():
    res = IdentityResolver(IdentityConfig())
    p = unit(rng.normal(size=512))
    m1 = meta("c1", 1); feed(res, m1, person_feats(p))
    m2 = meta("c1", 2); feed(res, m2, person_feats(p))         # same look, same camera, both alive => two people
    assert m2.global_id != m1.global_id
    res.end(m1)
    m3 = meta("c1", 3); feed(res, m3, person_feats(p))         # m1 gone: the look may legitimately re-appear
    assert m3.global_id in {m1.global_id, m2.global_id}


def test_vehicle_and_person_galleries_are_separate():
    res = IdentityResolver(IdentityConfig())
    p = unit(rng.normal(size=512))
    mp, mv = meta("c1", 1), meta("c2", 1, "vehicle")
    feed(res, mp, person_feats(p)); feed(res, mv, person_feats(p))
    assert mv.global_id.startswith("vehicle-") and mv.global_id != mp.global_id and mv.identity_status == "new"


# ---- config / pipeline plumbing -----------------------------------------------------------------------------
def test_shipped_config_loads_and_rejects_unknown_keys():
    cfg = load_config("config/perception.yaml")
    assert cfg.detector.prompts and set(cfg.reid) == {"person", "vehicle"}
    with pytest.raises(PerceptionConfigError):
        parse({"detector": {"prompts": [{"text": "a", "label": "a"}], "bogus": 1}})


class FakeDet:                                             # isolated test double; never imported by runtime code
    def __init__(self): self.script = []
    def detect(self, frame): return tuple(self.script.pop(0))


class FakeExtractor:
    def crop(self, bgr, box): return np.zeros((10, 10, 3), np.uint8)
    def extract(self, crops): return np.stack([unit(np.ones(512)) for _ in crops])


def _frame(cam="c1", seq=0, epoch=0):
    img = np.zeros((H, W, 3), np.uint8)
    return PreprocessedFrame(cam, seq, seq, epoch, float(seq), None, np.zeros((1, 1, 3), np.float32),
                             compute_letterbox(W, H, 640), img, "base", None)


def test_pipeline_stream_epoch_resets_tracks_and_reports_ended():
    cfg = load_config("config/perception.yaml")
    fd = FakeDet()
    pipe = PerceptionPipeline(cfg, detector=fd, extractors={"person": FakeExtractor(), "vehicle": FakeExtractor()})
    last = None
    for k in range(6):
        fd.script.append([det(mover(50)(k))])
        last = pipe.process(_frame(seq=k))
    t = last.tracks[0]
    assert t.track_id == 1 and t.global_id and t.identity_status == "new" and t.hits >= 5
    fd.script.append([det(mover(50)(6))])
    out = pipe.process(_frame(seq=6, epoch=1))                  # reconnect
    assert [(e.track_id, e.reason, e.global_id, e.stream_epoch) for e in out.ended_tracks] == [(1, "stream_restart", t.global_id, 0)]
    assert out.tracks[0].track_id == 1 and out.tracks[0].global_id is None   # fresh tracker, fresh identity evidence
