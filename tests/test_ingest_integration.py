"""Integration tests: the production CameraPipeline/IngestManager against real encoded streams
(file, RTSP, HLS). The streams are real H.264; only their *source servers* are test fixtures."""
import os
import threading
import time

import pytest

from godseye.ingest import CameraState, IngestManager
from godseye.ingest.camera import CameraPipeline, FatalCameraError
from godseye.ingest.config import parse_camera
from tests.support.streams import HlsServer, RtspServer, free_port

pytestmark = pytest.mark.integration
NO_GATE = {"motion_gating": False, "redundancy_threshold": 0, "keepalive_s": 0}


def file_url(p):
    return f"file://{p}"


def collect(m, seconds, stop_when=None):
    out, t = [], time.time()
    while time.time() - t < seconds:
        f = m.get_frame(0.2)
        if f:
            out.append(f)
        if stop_when and stop_when(out):
            break
    return out


def wait_state(m, cid, state, timeout=15):
    t = time.time()
    while time.time() - t < timeout:
        if m.stats()[cid].state == state:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def mgr(gst, make_cfg):
    made = []

    def make(*cams, **kw):
        m = IngestManager(make_cfg(*cams, **kw)); m.start(); made.append(m); return m
    yield make
    for m in made:
        m.stop()


def test_file_source_base_rate_and_frame_contract(clips, mgr):
    m = mgr({"camera_id": "f", "url": file_url(clips["motion"]), "sampling": NO_GATE})
    frames = collect(m, 20, lambda o: m.stats()["f"].state == CameraState.ENDED and m.bus.depth("f") == 0)
    s = m.stats()["f"]
    assert s.state == CameraState.ENDED and s.decoder == "avdec_h264"
    assert 20 <= len(frames) <= 28                      # 12 s clip at 2 fps
    f = frames[0]
    assert f.camera_id == "f" and f.sample_reason == "first" and f.stream_epoch == 1
    assert f.image.shape == (640, 640, 3) and 0 <= f.image.min() and f.image.max() <= 1
    assert f.original_bgr.shape == (360, 640, 3)
    assert (f.transform.orig_w, f.transform.orig_h, f.transform.pad_top) == (640, 360, 140)
    assert [x.frame_seq for x in frames] == list(range(len(frames)))
    assert all(b.capture_ts >= a.capture_ts for a, b in zip(frames, frames[1:]))
    assert s.frames_decoded >= 12 * 25 - 5 and s.malformed == 0


def test_motion_gating_burst_vs_static(clips, mgr):
    m = mgr({"camera_id": "s", "url": file_url(clips["static_then_motion"]), "sampling": {"redundancy_threshold": 1.0}})
    frames = collect(m, 20, lambda o: m.stats()["s"].state == CameraState.ENDED and m.bus.depth("s") == 0)
    reasons = [f.sample_reason for f in frames]
    assert "burst" in reasons
    first_burst = next(i for i, r in enumerate(reasons) if r == "burst")
    static_part = [f for f in frames[:first_burst]]
    assert len(static_part) <= 3                        # static first half is gated (first + keepalive at most)
    assert m.stats()["s"].dropped_redundant > 20


def test_static_clip_dropped_as_redundant(clips, mgr):
    m = mgr({"camera_id": "s", "url": file_url(clips["static"])})
    frames = collect(m, 20, lambda o: m.stats()["s"].state == CameraState.ENDED and m.bus.depth("s") == 0)
    assert len(frames) <= 4 and m.stats()["s"].dropped_redundant > 50


def test_multi_camera_isolation_and_fairness(clips, mgr):
    m = mgr({"camera_id": "a", "url": file_url(clips["motion"]), "sampling": NO_GATE},
            {"camera_id": "b", "url": file_url(clips["motion"]), "sampling": NO_GATE},
            {"camera_id": "dead", "url": f"rtsp://127.0.0.1:{free_port()}/none"})
    frames = collect(m, 8)
    ids = {f.camera_id for f in frames}
    assert ids == {"a", "b"}
    st = m.stats()
    assert st["dead"].state in (CameraState.RECONNECTING, CameraState.CONNECTING) and st["dead"].last_error
    assert st["dead"].reconnects >= 1 and st["dead"].frames_decoded == 0
    assert st["a"].state == st["b"].state == CameraState.STREAMING


def test_rtsp_stream(clips, mgr):
    with RtspServer(clips["motion"]) as srv:
        m = mgr({"camera_id": "r", "url": srv.url, "sampling": NO_GATE})
        frames = collect(m, 15, lambda o: len(o) >= 6)
        s = m.stats()["r"]
        assert len(frames) >= 6 and s.state == CameraState.STREAMING and s.decoder == "avdec_h264"
        assert frames[0].pts_ns is not None and frames[0].image.shape == (640, 640, 3)


def test_rtsp_disconnect_and_reconnect_new_epoch(clips, mgr):
    srv = RtspServer(clips["motion"]).start()
    try:
        m = mgr({"camera_id": "r", "url": srv.url, "sampling": NO_GATE})
        assert collect(m, 15, lambda o: len(o) >= 3)
        srv.stop()
        assert wait_state(m, "r", CameraState.RECONNECTING, 15)
        srv.start()
        got = collect(m, 20, lambda o: any(f.stream_epoch == 2 for f in o))
        assert any(f.stream_epoch == 2 for f in got)
        s = m.stats()["r"]
        assert s.reconnects >= 1 and s.state == CameraState.STREAMING and s.stream_epoch == 2
    finally:
        srv.stop()


def test_hls_stream(clips, mgr, tmp_path):
    with HlsServer(clips["motion"], tmp_path) as srv:
        m = mgr({"camera_id": "h", "url": srv.url, "sampling": NO_GATE})
        frames = collect(m, 20, lambda o: len(o) >= 5)
        assert len(frames) >= 5 and m.stats()["h"].state == CameraState.STREAMING


@pytest.mark.skipif(not os.environ.get("GODSEYE_TEST_RTMP_URL"),
                    reason="needs a real RTMP server URL in GODSEYE_TEST_RTMP_URL (none available locally: ffmpeg cannot serve RTMP to players)")
def test_rtmp_stream(mgr):
    m = mgr({"camera_id": "m", "url": os.environ["GODSEYE_TEST_RTMP_URL"], "sampling": NO_GATE})
    assert len(collect(m, 30, lambda o: len(o) >= 3)) >= 3


def test_invalid_streams_retry_without_crash(clips, mgr, tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not video" * 100)
    m = mgr({"camera_id": "junk", "url": file_url(junk)},
            {"camera_id": "missing", "url": file_url(tmp_path / "nope.mp4")},
            {"camera_id": "http404", "url": f"http://127.0.0.1:{free_port()}/x.m3u8"})
    time.sleep(4)
    for cid, s in m.stats().items():
        assert s.frames_decoded == 0 and s.last_error, cid
        assert s.state != CameraState.STREAMING, cid


def test_max_attempts_marks_failed(mgr):
    m = mgr({"camera_id": "x", "url": f"rtsp://127.0.0.1:{free_port()}/x", "reconnect": {"max_attempts": 2}})
    assert wait_state(m, "x", CameraState.FAILED, 15)
    assert "gave up" in m.stats()["x"].last_error


def test_hw_accel_unavailable_fails_loudly_not_silently(clips, mgr, gst):
    if gst.ElementFactory.find("nvh264dec"):
        pytest.skip("nvh264dec present on this machine")
    m = mgr({"camera_id": "hw", "url": file_url(clips["motion"]), "hw_accel": "nvcodec"},
            {"camera_id": "ok", "url": file_url(clips["motion"]), "sampling": NO_GATE})
    assert wait_state(m, "hw", CameraState.FAILED, 10)
    assert "nvh264dec" in m.stats()["hw"].last_error and m.stats()["hw"].frames_decoded == 0
    assert collect(m, 5, lambda o: any(f.camera_id == "ok" for f in o))   # other camera unaffected


def test_queue_pressure_drops_oldest_and_never_grows(clips, mgr):
    m = mgr({"camera_id": "q", "url": file_url(clips["motion"]), "sampling": NO_GATE, "queue_size": 2})
    time.sleep(8)                                       # no consumer
    assert m.bus.depth("q") == 2
    s = m.stats()["q"]
    assert s.dropped_queue > 5 and s.state in (CameraState.STREAMING, CameraState.ENDED)
    newest = m.get_frame(1)
    assert newest.frame_seq >= s.dropped_queue          # oldest frames were the ones dropped


def test_loop_file_restarts_at_eos(clips, mgr):
    m = mgr({"camera_id": "l", "url": file_url(clips["motion"]), "loop_file": True, "sampling": NO_GATE})
    frames = collect(m, 16)
    assert len(frames) > 24 and m.stats()["l"].state == CameraState.STREAMING   # >12 s worth at 2 fps


def test_graceful_shutdown_is_fast_and_clean(clips, mgr):
    m = mgr(*[{"camera_id": f"c{i}", "url": file_url(clips["motion"]), "loop_file": True} for i in range(4)],
            {"camera_id": "down", "url": f"rtsp://127.0.0.1:{free_port()}/x"})
    collect(m, 3)
    t = time.time()
    assert m.stop(10) is True and time.time() - t < 5
    assert all(s.state == CameraState.STOPPED for s in m.stats().values())
    assert not [t for t in threading.enumerate() if t.name.startswith("cam-")]


def test_add_and_remove_camera_at_runtime(clips, mgr, make_cfg):
    m = mgr({"camera_id": "a", "url": file_url(clips["motion"]), "loop_file": True, "sampling": NO_GATE})
    m.add_camera(make_cfg({"camera_id": "b", "url": file_url(clips["motion"]), "loop_file": True, "sampling": NO_GATE}).cameras[0])
    assert {f.camera_id for f in collect(m, 6)} == {"a", "b"}
    assert m.remove_camera("a") is True
    time.sleep(0.5)
    while m.get_frame(0.01):
        pass
    assert {f.camera_id for f in collect(m, 3)} == {"b"}


def test_environment_failure_is_loud(make_cfg, gst, monkeypatch):
    from godseye.ingest import camera
    monkeypatch.setitem(camera.SOURCE_ELEMENTS, "rtsp", ("definitely_not_an_element",))
    with pytest.raises(camera.IngestEnvironmentError, match="definitely_not_an_element"):
        IngestManager(make_cfg({"camera_id": "x", "url": "rtsp://h/x"})).start()


class _FakeBuf:                                         # isolated unit-test double for GStreamer buffer/sample
    def __init__(self, size, corrupted=False, ok=True):
        self.size, self.corrupted, self.ok, self.pts = size, corrupted, ok, 0

    def has_flags(self, f): return self.corrupted
    def map(self, flags):
        class MI: pass
        mi = MI(); mi.size = self.size; mi.data = bytes(self.size)
        return self.ok, mi
    def unmap(self, mi): pass


class _FakeSample:
    def __init__(self, buf, w=64, h=48):
        self.buf, self.w, self.h = buf, w, h
    def get_buffer(self): return self.buf
    def get_caps(self):
        s = self
        class St:
            def get_value(self, k): return s.w if k == "width" else s.h
        class C:
            def get_structure(self, i): return St()
        return C()


@pytest.mark.parametrize("buf", [_FakeBuf(10), _FakeBuf(64 * 48 * 3, corrupted=True), _FakeBuf(0, ok=False)])
def test_malformed_buffers_counted_and_not_published(gst, buf):
    from godseye.ingest.sampler import FrameSampler
    published = []
    cam = CameraPipeline(parse_camera({"camera_id": "m", "url": "rtsp://h/x"}, {}, 0), lambda f: published.append(f) or "ok", gst)
    cam._on_sample(_FakeSample(buf), FrameSampler(cam.cfg.sampling), 0.0)
    assert cam.stats().malformed == 1 and not published and cam.stats().last_error


def test_stalled_stream_detected_and_recovers(clips, mgr):
    import signal
    srv = RtspServer(clips["motion"]).start()
    try:
        m = mgr({"camera_id": "r", "url": srv.url, "sampling": NO_GATE})
        assert collect(m, 15, lambda o: len(o) >= 3)
        srv.proc.send_signal(signal.SIGSTOP)            # connection stays open but no data flows
        assert wait_state(m, "r", CameraState.RECONNECTING, 20)
        assert m.stats()["r"].last_error                # stall watchdog or rtspsrc timeout; either way surfaced
        srv.proc.send_signal(signal.SIGCONT)
        assert collect(m, 25, lambda o: any(f.stream_epoch >= 2 for f in o))
    finally:
        srv.proc.send_signal(signal.SIGCONT)
        srv.stop()
