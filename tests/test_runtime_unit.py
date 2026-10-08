"""TrackAService logic (quarantine, circuit breaker, carried track endings, health) with test doubles for ingest / pipeline /
store, so it runs without a GPU. The real-component versions of these scenarios are in test_integration.py."""
import time

import numpy as np
import pytest

from godseye.ingest.preprocess import compute_letterbox
from godseye.ingest.types import CameraState, CameraStats, PreprocessedFrame
from godseye.perception.types import EnrichedFrame, TrackEnded
from godseye.runtime.config import RuntimeConfig, RuntimeConfigError, parse
from godseye.runtime.metrics import Metrics
from godseye.runtime.service import TrackAService

CFG = RuntimeConfig(poll_s=0.05, quarantine_after=3, quarantine_s=0.5, store_fail_threshold=2, store_backoff_s=0.3, store_backoff_max_s=0.6)


def frame(cam, seq=0, epoch=1):
    return PreprocessedFrame(cam, seq, seq, epoch, time.time(), None, np.zeros((1, 1, 3), np.float32), compute_letterbox(4, 4, 4),
                             np.zeros((4, 4, 3), np.uint8), "base", None)


def ended(cam, tid=1):
    return TrackEnded(cam, tid, "lost_timeout", "person", "person", 0.0, 1.0, 3, None, "unresolved", 1)


class FakeBus:
    def depth(self, cid): return 0


class FakeIngest:
    bus = FakeBus()
    def __init__(self): self.states = {}
    def stats(self): return {c: CameraStats(state=s) for c, s in self.states.items()}
    def get_frame(self, timeout=None): time.sleep(min(timeout or 0, 0.01)); return None


class FakePipeline:
    identity_errors = 0
    def __init__(self): self.fail = set(); self.calls = []; self.closed = []
    def process(self, f):
        self.calls.append(f.camera_id)
        if f.camera_id in self.fail:
            raise RuntimeError("injected model failure")
        return EnrichedFrame(f, (), (), (ended(f.camera_id),) if f.frame_seq == 9 else (), {"detect": 1.0, "track": 0.5, "identity": 2.0})
    def close_camera(self, cam, reason="shutdown"):
        first = all(c != cam for c, _ in self.closed)                 # like the real tracker: a closed camera has nothing left to end
        self.closed.append((cam, reason))
        return (ended(cam, 99),) if first else ()
    def shutdown(self): return (ended("a", 77),)


class FakeStore:
    def __init__(self): self.fail = False; self.written = []; self.ends = []; self._seg = {}
    def write(self, ef):
        if self.fail:
            raise OSError("injected: database is locked")
        self.written.append(ef)
    def end_tracks(self, cam, ended_, ts, stream_epoch=0): self.ends.append((cam, [e.track_id for e in ended_]))
    def close_camera(self, cam): pass
    def health(self): return {"clips_written": 0}


def svc():
    s = TrackAService(FakeIngest(), FakePipeline(), FakeStore(), CFG, Metrics())
    return s


def test_failing_camera_is_quarantined_and_others_continue():
    s = svc()
    s.pipeline.fail = {"bad"}
    for i in range(3):
        assert s.process_frame(frame("bad", i)) is None
        assert s.process_frame(frame("good", i)) is not None
    assert s.m.counters["quarantined.bad"] == 1 and s.m.counters["model_error.bad"] == 3
    n = len(s.pipeline.calls)
    assert s.process_frame(frame("bad", 4)) is None and len(s.pipeline.calls) == n            # quarantined: model not even called
    assert s.process_frame(frame("good", 4)) is not None
    time.sleep(0.6)
    s.pipeline.fail = set()
    assert s.process_frame(frame("bad", 5)) is not None                                        # back after the quarantine
    assert s.health()["cameras"] == {}


def test_store_breaker_pauses_persistence_but_not_analytics_and_carries_endings():
    s = svc()
    s.store.fail = True
    for i in range(2):
        assert s.process_frame(frame("a", i)) is not None                                      # analytics keep running
    assert s.m.counters["store_errors"] == 2 and s.m.counters["store_breaker_opened"] == 1
    s.process_frame(frame("a", 9))                                                             # ending produced while storage is down
    assert s.m.counters["frames_not_persisted"] == 3 and not s.store.written
    assert s.health()["persistence"]["paused"] and s.health()["persistence"]["degraded"]
    s.store.fail = False
    time.sleep(0.35)
    s.process_frame(frame("a", 10))
    assert len(s.store.written) == 1 and [e.track_id for e in s.store.written[0].ended_tracks] == [1]       # the missed ending was carried
    assert not s._carry and not s.health()["persistence"]["degraded"]


def test_source_end_closes_tracks_once_and_shutdown_persists_the_rest():
    s = svc()
    s.ingest.states = {"a": CameraState.ENDED, "b": CameraState.STREAMING}
    s.process_frame(frame("a"))
    s._check_sources()
    assert s.pipeline.closed == [("a", "source_ended")] and s.store.ends == [("a", [99])]
    s._check_sources()
    assert s.pipeline.closed == [("a", "source_ended")]                                        # not repeated
    s._finish_all("shutdown")
    assert ("a", [77]) in s.store.ends


def test_source_end_retries_when_storage_is_down():
    s = svc()
    s.ingest.states = {"a": CameraState.FAILED}
    boom = {"n": 0}
    real = s.store.end_tracks
    def flaky(*a, **k):
        boom["n"] += 1
        if boom["n"] == 1: raise OSError("down")
        return real(*a, **k)
    s.store.end_tracks = flaky
    s._check_sources()
    assert "a" not in s._closed and s._carry["a"]
    s._retry_at["a"] = 0
    s._check_sources()
    assert "a" in s._closed and s.store.ends == [("a", [99])]


def test_oom_is_classified_and_config_is_validated():
    s = svc()
    class OutOfMemoryError(RuntimeError): pass
    s.pipeline.process = lambda f: (_ for _ in ()).throw(OutOfMemoryError("CUDA out of memory"))
    assert s.process_frame(frame("a")) is None and s.m.counters["gpu_oom.a"] == 1
    with pytest.raises(RuntimeConfigError):
        parse({"bogus": 1})
    with pytest.raises(RuntimeConfigError):
        parse({"quarantine_after": 0})
    assert parse({}).query_timeout_s == 120.0
