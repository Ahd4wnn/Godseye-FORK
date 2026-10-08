"""Phase 2 tests with the REAL models (NanoOWL TensorRT engine, FastReID) on real images/video. Skipped, loudly, when
the GPU or the real data is missing -- never silently replaced by mocks."""
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model tests need CUDA")

from godseye.ingest.preprocess import compute_letterbox  # noqa: E402
from godseye.ingest.types import PreprocessedFrame  # noqa: E402
from godseye.perception.config import load_config  # noqa: E402
from godseye.perception.identity import IdentityResolver  # noqa: E402
from godseye.perception.pipeline import PerceptionPipeline  # noqa: E402
from godseye.perception.tracker import TrackMeta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO = next(iter(glob.glob(os.environ.get("GODSEYE_TEST_VIDEO", "/home/albhy/Downloads/Busy_airport*.mp4"))), None)
MARKET = ROOT / "data/market1501_raw/Market-1501-v15.09.15/bounding_box_test"


@pytest.fixture(scope="module")
def pipe():
    return PerceptionPipeline(load_config(ROOT / "config/perception.yaml"))


def pframe(bgr, seq=0, epoch=0, cam="cam"):
    h, w = bgr.shape[:2]
    return PreprocessedFrame(cam, seq, seq, epoch, seq / 5.0, None, np.zeros((1, 1, 3), np.float32),
                             compute_letterbox(w, h, 640), bgr, "base", None)


@pytest.fixture(scope="module")
def video_frames():
    if not VIDEO:
        pytest.skip("no real test video (set GODSEYE_TEST_VIDEO)")
    cap = cv2.VideoCapture(VIDEO)
    step, out, k = max(1, round(cap.get(cv2.CAP_PROP_FPS) / 5)), [], 0
    while len(out) < 40:
        ok, bgr = cap.read()
        if not ok:
            break
        if k % step == 0:
            out.append(bgr)
        k += 1
    return out


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(iw, 0) * max(ih, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_real_detection_contract(pipe, video_frames):
    f = pframe(video_frames[0])
    dets = pipe.detector.detect(f)
    h, w = f.original_bgr.shape[:2]
    assert len([d for d in dets if d.label == "person"]) >= 2                 # multiple real objects
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h                        # original-frame pixels
        assert d.embedding.shape == (512,) and abs(np.linalg.norm(d.embedding) - 1) < 1e-3
        assert d.camera_id == "cam" and d.provenance.detector == "nanoowl" and d.provenance.engine_sha256
        assert d.confidence > 0.1 and d.patch_index >= 0


def test_detection_coordinates_follow_the_image(pipe, video_frames):
    """Pad the frame by a known offset: boxes must move by exactly that offset (coordinate mapping is real)."""
    a = video_frames[3]
    dx, dy = 160, 96
    b = cv2.copyMakeBorder(a, dy, 0, dx, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    da = [d for d in pipe.detector.detect(pframe(a)) if d.confidence > 0.4]
    db = [d for d in pipe.detector.detect(pframe(b)) if d.confidence > 0.4]
    assert da
    hit = sum(any(iou(x.bbox_xyxy + [dx, dy, dx, dy], y.bbox_xyxy) > 0.6 for y in db) for x in da)
    assert hit / len(da) >= 0.7, f"only {hit}/{len(da)} boxes moved with the image"


def test_detection_and_embedding_are_deterministic(pipe, video_frames):
    f = pframe(video_frames[5])
    d1, d2 = pipe.detector.detect(f), pipe.detector.detect(f)
    assert len(d1) == len(d2)
    for x, y in zip(d1, d2):
        assert float(x.embedding @ y.embedding) > 0.999 and iou(x.bbox_xyxy, y.bbox_xyxy) > 0.98


def test_video_tracking_and_identity_invariants(pipe, video_frames):
    seen = defaultdict(list)
    per_frame = []
    for k, bgr in enumerate(video_frames):
        out = pipe.process(pframe(bgr, seq=k, cam="vid"))
        per_frame.append(out)
        gids = [t.global_id for t in out.tracks if t.global_id]
        assert len(gids) == len(set(gids)), "two simultaneous tracks on one camera share a global_id"
        for t in out.tracks:
            seen[t.track_id].append(t)
            assert t.reid_class != "person" or t.global_id is None or t.global_id.startswith("person-")
            assert abs(np.linalg.norm(t.clip_embedding) - 1) < 1e-3
    assert max(len(o.tracks) for o in per_frame) >= 2                         # multiple simultaneous tracks
    longest = max(seen.values(), key=len)
    assert len(longest) >= 10 and longest[-1].hits == len(longest)           # lifecycle: one id, growing hits
    assert len(longest[-1].trajectory) >= 10
    for ts in seen.values():                                                  # a track's global_id never changes
        g = {t.global_id for t in ts if t.global_id}
        assert len(g) <= 1
    assert any(t.global_id for ts in seen.values() for t in ts)               # real FastReID produced identities
    ended = pipe.shutdown()
    assert all(e.reason == "shutdown" for e in ended)


# ---- real FastReID on real person crops --------------------------------------------------------------------
def _market():
    if not MARKET.is_dir():
        pytest.skip("Market-1501 not present")
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(MARKET.glob("*.jpg")):
        m = re.match(r"(-?\d+)_c(\d)", f.name)
        if m and int(m[1]) > 0:
            by[int(m[1])][int(m[2])].append(f)
    return {p: c for p, c in by.items() if len(c) >= 2 and all(len(v) >= 4 for v in list(c.values())[:2])}


def _feats(ex, files):
    return ex.extract([cv2.imread(str(f)) for f in files])


def test_real_reid_embedding_consistency(pipe):
    ex = pipe.extractors["person"]
    by = _market()
    pids = sorted(by)[:40]
    same, diff = [], []
    cache = {}
    for p in pids:
        c1, c2 = sorted(by[p])[:2]
        cache[p] = (_feats(ex, by[p][c1][:4]), _feats(ex, by[p][c2][:4]))
    for p in pids:
        a, b = cache[p]
        assert a.shape == (4, 2048) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-4)   # dims + normalisation
        same.append(float((a @ b.T).mean()))
        q = pids[(pids.index(p) + 7) % len(pids)]
        diff.append(float((a @ cache[q][1].T).mean()))
    print(f"same-id cos {np.mean(same):.3f}  diff-id cos {np.mean(diff):.3f}")
    assert np.mean(same) > np.mean(diff) + 0.15
    assert np.mean(np.array(same) > np.array(diff)) > 0.85


def test_real_cross_camera_identity_conservative(pipe):
    """Enrol 25 people from one camera, probe with 25 known + 15 unknown people from another. Known people should
    mostly match their own identity; a false merge (wrong or unknown person merged) must be rare."""
    ex, by = pipe.extractors["person"], _market()
    pids = sorted(by)[100:140]
    known, unknown = pids[:25], pids[25:]
    res = IdentityResolver(load_config(ROOT / "config/perception.yaml").identity)
    gid_of = {}
    for p in known:
        c1 = sorted(by[p])[0]
        m = TrackMeta("camA", p, "person", 0, 0)
        for f in _feats(ex, by[p][c1][:4]):
            res.observe(m, f, 0.0)
        gid_of[p] = m.global_id
        assert m.global_id
    right = wrong = ambiguous = new = 0
    for p in known + unknown:
        c2 = sorted(by[p])[1]
        m = TrackMeta("camB", p, "person", 0, 0)
        for f in _feats(ex, by[p][c2][:4]):
            res.observe(m, f, 1.0)
        if m.identity_status == "matched":
            if p in gid_of and m.global_id == gid_of[p]:
                right += 1
            else:
                wrong += 1
        elif m.identity_status == "ambiguous":
            ambiguous += 1
        else:
            new += 1
    print(f"right={right} wrong={wrong} ambiguous={ambiguous} new={new}")
    assert right >= 5
    assert wrong <= 0.15 * (right + wrong)


def test_tensorrt_runs_with_gstreamer_loaded(gst, pipe, video_frames):
    """Regression: libunwind (pulled in by GStreamer) used to abort TensorRT's Dims iteration; both live in one process."""
    assert pipe.detector.detect(pframe(video_frames[1]))


def test_decode_matches_nanoowl_decode(pipe, video_frames):
    """Our patch-index-preserving decode must select exactly what NanoOWL's own `decode` selects."""
    d = pipe.detector
    bgr = video_frames[2]
    h, w = bgr.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).cuda().permute(2, 0, 1)[None].float()
    enc = d.predictor.encode_rois(d.predictor.image_preprocessor(t), torch.tensor([[0, 0, w, h]], dtype=torch.float32).cuda(), pad_square=True)
    ref = d.predictor.decode(enc, d.text, [p.threshold for p in d.cfg.prompts])
    ours = d._decode(enc)
    assert len(ref.scores) == len(ours["score"]) > 0
    assert torch.allclose(ref.scores.sort().values, ours["score"].sort().values, atol=1e-5)
    assert torch.allclose(ref.boxes.sort(dim=0).values, enc.pred_boxes[0][ours["patch"]].sort(dim=0).values, atol=1e-3)
