"""Deterministic sampler tests on synthetic arrays + explicit timestamps (isolated unit tests; not a pipeline path)."""
import numpy as np

from godseye.ingest.config import SamplingConfig
from godseye.ingest.sampler import FrameSampler

W, H, SRC_FPS = 320, 180, 25


def static_frame():
    return np.full((H, W, 3), 64, np.uint8)


def moving_frame(i):
    f = static_frame()
    x = (i * 7) % (W - 60)
    f[60:120, x:x + 60] = 255
    return f


def run(cfg, frames, seconds):
    s, out = FrameSampler(cfg), []
    for i in range(int(seconds * SRC_FPS)):
        d = s.offer(frames(i), i / SRC_FPS)
        out.append((i / SRC_FPS, d))
    return out


def emitted(res):
    return [(t, d) for t, d in res if d.emit]


def test_base_rate_two_fps_without_gating():
    cfg = SamplingConfig(motion_gating=False, redundancy_threshold=0, keepalive_s=0)
    em = emitted(run(cfg, lambda i: moving_frame(i), 20))
    assert 38 <= len(em) <= 41                               # 2 fps * 20 s
    assert {d.reason for _, d in em} <= {"first", "base"}
    gaps = np.diff([t for t, _ in em])
    assert 0.44 <= gaps.min() and gaps.max() <= 0.56


def test_burst_rate_five_fps_on_motion_then_decays():
    cfg = SamplingConfig(redundancy_threshold=0, keepalive_s=0, burst_hold_s=2.0)
    res = run(cfg, lambda i: moving_frame(i) if i < 15 * SRC_FPS else static_frame(), 25)
    moving = [t for t, d in emitted(res) if 5 <= t < 15]
    assert 45 <= len(moving) <= 52                           # ~5 fps * 10 s
    assert any(d.reason == "burst" for _, d in emitted(res))
    later = [t for t, d in emitted(res) if 19 <= t < 25]     # after hold expired -> back to base
    assert 10 <= len(later) <= 13


def test_static_scene_is_gated_to_keepalive():
    cfg = SamplingConfig(keepalive_s=5.0)
    res = run(cfg, lambda i: static_frame(), 30)
    em = emitted(res)
    assert [d.reason for _, d in em][0] == "first"
    assert all(d.reason == "keepalive" for _, d in em[1:])
    assert 5 <= len(em) <= 7                                  # first + one per 5 s
    assert sum(1 for _, d in res if d.reason == "redundant") > 0


def test_motion_overrides_redundancy_and_warmup_ignored():
    cfg = SamplingConfig(keepalive_s=100)
    res = run(cfg, lambda i: static_frame() if i < 5 * SRC_FPS else moving_frame(i), 10)
    first_motion_emit = next(t for t, d in emitted(res) if t > 5 and d.reason in ("burst", "base"))
    assert first_motion_emit < 6.0
    assert all(d.motion == 0.0 for _, d in res[: 5])          # warm-up evals report no motion


def test_disabled_gating_reports_no_motion():
    cfg = SamplingConfig(motion_gating=False, redundancy_threshold=0)
    assert run(cfg, lambda i: static_frame(), 1)[0][1].motion is None


def test_late_frames_resync_without_burst_of_catchup():
    cfg = SamplingConfig(motion_gating=False, redundancy_threshold=0, keepalive_s=0)
    s = FrameSampler(cfg)
    ts = [0, 10, 10.04, 10.08, 10.5, 11.0]
    out = [s.offer(static_frame(), t).emit for t in ts]
    assert out == [True, True, False, False, True, True]
