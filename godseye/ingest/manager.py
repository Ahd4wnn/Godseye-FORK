"""Multi-camera supervisor + bounded per-camera frame bus (the Phase 1 -> Phase 2 boundary)."""
from __future__ import annotations

import logging
import threading
from collections import deque

from ..gst_env import init_gst
from .camera import CameraPipeline, check_environment
from .config import CameraConfig, IngestConfig
from .types import CameraStats, PreprocessedFrame

log = logging.getLogger("godseye.ingest")


class FrameBus:
    """Bounded queue per camera, fair round-robin consumption. Producers never block (backpressure = drop)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._q: dict[str, deque] = {}
        self._policy: dict[str, tuple[int, str]] = {}
        self._order: list[str] = []
        self._rr = 0

    def register(self, cid: str, size: int, policy: str) -> None:
        with self._cv:
            self._q[cid] = deque()
            self._policy[cid] = (size, policy)
            self._order.append(cid)

    def unregister(self, cid: str) -> None:
        with self._cv:
            self._q.pop(cid, None)
            self._policy.pop(cid, None)
            self._order.remove(cid)

    def put(self, frame: PreprocessedFrame) -> str:
        with self._cv:
            q = self._q.get(frame.camera_id)
            if q is None:
                return "dropped_unregistered"
            size, policy = self._policy[frame.camera_id]
            result = "ok"
            if len(q) >= size:
                if policy == "drop_newest":
                    return "dropped_newest"
                q.popleft()
                result = "dropped_oldest"
            q.append(frame)
            self._cv.notify()
            return result

    def get(self, timeout: float | None = None) -> PreprocessedFrame | None:
        with self._cv:
            def take():
                n = len(self._order)
                for i in range(n):
                    cid = self._order[(self._rr + i) % n]
                    if self._q[cid]:
                        self._rr = (self._rr + i + 1) % n
                        return self._q[cid].popleft()
            return take() or (self._cv.wait_for(lambda: any(self._q.values()), timeout) and take()) or None

    def depth(self, cid: str) -> int:
        with self._cv:
            return len(self._q.get(cid, ()))


class IngestManager:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.bus = FrameBus()
        self._cams: dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._Gst = None
        self._started = False

    def start(self) -> None:
        self._Gst = init_gst(list(self.cfg.plugin_paths), self.cfg.base_dir, list(self.cfg.preload_libs))
        enabled = [c for c in self.cfg.cameras if c.enabled]
        check_environment(self._Gst, {c.protocol for c in enabled})
        self._started = True
        for c in enabled:
            self.add_camera(c)

    def add_camera(self, cfg: CameraConfig) -> None:
        if not self._started:
            raise RuntimeError("IngestManager.start() first")
        check_environment(self._Gst, {cfg.protocol})
        with self._lock:
            if cfg.camera_id in self._cams:
                raise ValueError(f"camera {cfg.camera_id} already exists")
            self.bus.register(cfg.camera_id, cfg.queue_size, cfg.queue_policy)
            cam = CameraPipeline(cfg, self.bus.put, self._Gst)
            self._cams[cfg.camera_id] = cam
        cam.start()

    def remove_camera(self, camera_id: str, timeout: float = 10.0) -> bool:
        with self._lock:
            cam = self._cams.pop(camera_id)
        ok = cam.stop(timeout)
        self.bus.unregister(camera_id)
        return ok

    def get_frame(self, timeout: float | None = None) -> PreprocessedFrame | None:
        return self.bus.get(timeout)

    def stats(self) -> dict[str, CameraStats]:
        with self._lock:
            return {cid: c.stats() for cid, c in self._cams.items()}

    def stop(self, timeout: float = 10.0) -> bool:
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:           # signal all first so shutdown is parallel
            c.request_stop()
        ok = all(c.join(timeout) for c in cams)
        if not ok:
            log.error("some camera threads did not stop within %ss", timeout)
        return ok
