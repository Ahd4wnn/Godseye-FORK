"""Track A service: IngestManager -> PerceptionPipeline -> EvidenceStore, with failure isolation and measurements.

Isolation rules
  * Cameras are independent GStreamer threads feeding bounded drop-based queues; this loop is the only consumer.
  * A failure on one camera's frame (model error, GPU OOM, bad input) drops that frame and, after repeated failures,
    pauses ONLY that camera (quarantine); other cameras keep flowing.
  * Storage failures never stop ingestion or analytics: a circuit breaker pauses persistence with back-off, every
    non-persisted frame is counted (`frames_not_persisted`), and track endings that could not be written are carried to
    the next successful write so no track is silently left open.
  * Nothing here is shared with Track B except the stores on disk."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

from ..ingest.manager import IngestManager
from ..ingest.types import CameraState
from ..perception.nanoowl_detector import DetectorInputError
from ..perception.pipeline import PerceptionPipeline
from ..perception.types import EnrichedFrame
from ..storage.store import EvidenceStore
from .config import RuntimeConfig
from .metrics import Metrics

log = logging.getLogger("godseye.runtime")


class TrackAService:
    def __init__(self, ingest: IngestManager, pipeline: PerceptionPipeline, store: EvidenceStore, cfg: RuntimeConfig, metrics: Metrics | None = None):
        self.ingest, self.pipeline, self.store, self.cfg = ingest, pipeline, store, cfg
        self.m = metrics or Metrics()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fails: dict[str, int] = defaultdict(int)
        self._quarantine: dict[str, float] = {}
        self._store_fails = 0
        self._store_paused_until = 0.0
        self._store_backoff = cfg.store_backoff_s
        self._carry: dict[str, list] = defaultdict(list)       # camera -> track endings not yet persisted
        self._closed: set[str] = set()
        self._last_check = 0.0
        self._retry_at: dict[str, float] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="track-a", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        alive = bool(self._thread and self._thread.is_alive())
        if not alive:
            self._finish_all("shutdown")
        return not alive

    def _run(self) -> None:
        while not self._stop.is_set():
            f = self.ingest.get_frame(timeout=self.cfg.poll_s)
            if f is not None:
                self.process_frame(f)
            if time.time() - self._last_check >= self.cfg.poll_s:
                self._last_check = time.time()
                self._check_sources()

    # ------------------------------------------------------------------ one frame
    def process_frame(self, f) -> EnrichedFrame | None:
        cam = f.camera_id
        if time.time() < self._quarantine.get(cam, 0.0):
            self.m.inc(f"frames_dropped_quarantine.{cam}")
            return None
        self._closed.discard(cam)                               # a frame means the camera is live again
        t0 = time.perf_counter()
        try:
            ef = self.pipeline.process(f)
        except Exception as e:                                  # noqa: BLE001 - isolate: one camera's failure must not stop the loop
            self._camera_failed(cam, e)
            return None
        self._fails[cam] = 0
        for k, v in ef.timings_ms.items():
            self.m.observe(f"{k}_ms", v)
        self.m.observe("pipeline_ms", (time.perf_counter() - t0) * 1e3)
        self.m.inc(f"frames_processed.{cam}")
        self.m.tick("processed_fps")
        self._persist(cam, ef)
        self.m.observe("e2e_ms", (time.time() - f.capture_ts) * 1e3)
        return ef

    def _camera_failed(self, cam: str, e: Exception) -> None:
        kind = "gpu_oom" if type(e).__name__ == "OutOfMemoryError" else "input_error" if isinstance(e, DetectorInputError) else "model_error"
        self.m.inc(f"{kind}.{cam}")
        self.last_error = f"{cam}: {kind}: {e}"
        log.error("frame from %s dropped (%s): %s", cam, kind, e, exc_info=kind == "model_error")
        if kind == "gpu_oom":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                   # noqa: BLE001
                pass
        self._fails[cam] += 1
        if self._fails[cam] >= self.cfg.quarantine_after or kind == "input_error":
            self._quarantine[cam] = time.time() + self.cfg.quarantine_s
            self._fails[cam] = 0
            self.m.inc(f"quarantined.{cam}")
            log.error("camera %s paused for %.0fs after repeated failures", cam, self.cfg.quarantine_s)

    # ------------------------------------------------------------------ persistence with circuit breaker
    def _persist(self, cam: str, ef: EnrichedFrame) -> None:
        carried = self._carry.get(cam)
        if carried:
            ef = EnrichedFrame(ef.source, ef.detections, ef.tracks, tuple(carried) + ef.ended_tracks, ef.timings_ms)
        if time.time() < self._store_paused_until:
            self._defer(cam, ef)
            return
        t0 = time.perf_counter()
        try:
            self.store.write(ef)
        except Exception as e:                                  # noqa: BLE001 - storage must never take ingestion down
            self._store_fails += 1
            self.m.inc("store_errors")
            self.last_error = f"store: {type(e).__name__}: {e}"
            log.error("storage write failed (%d in a row): %s", self._store_fails, e)
            if self._store_fails >= self.cfg.store_fail_threshold:
                self._store_paused_until = time.time() + self._store_backoff
                self._store_backoff = min(self._store_backoff * 2, self.cfg.store_backoff_max_s)
                self.m.inc("store_breaker_opened")
                log.error("persistence paused for %.0fs (circuit breaker)", self._store_backoff / 2)
            self._defer(cam, ef)
            return
        self.m.observe("store_ms", (time.perf_counter() - t0) * 1e3)
        self._carry.pop(cam, None)
        self._store_fails = 0
        self._store_backoff = self.cfg.store_backoff_s

    def _defer(self, cam: str, ef: EnrichedFrame) -> None:
        self.m.inc("frames_not_persisted")
        if ef.ended_tracks:
            self._carry[cam] = list(ef.ended_tracks)

    # ------------------------------------------------------------------ sources that ended or were removed
    def _check_sources(self) -> None:
        for cam, st in self.ingest.stats().items():
            if st.state in (CameraState.ENDED, CameraState.FAILED, CameraState.STOPPED) and cam not in self._closed \
                    and self.ingest.bus.depth(cam) == 0 and time.time() >= self._retry_at.get(cam, 0.0):
                self._finish_camera(cam, "source_ended" if st.state == CameraState.ENDED else "source_failed")

    def _finish_camera(self, cam: str, reason: str) -> None:
        ended = self.pipeline.close_camera(cam, reason)
        ended = tuple(self._carry.pop(cam, [])) + tuple(ended)
        try:
            self.store.end_tracks(cam, ended, time.time())
            self._closed.add(cam)
        except Exception as e:                                  # noqa: BLE001
            self.m.inc("store_errors")
            self._carry[cam] = list(ended)
            self._retry_at[cam] = time.time() + 5.0
            log.error("could not persist track endings for %s: %s (retry in 5s)", cam, e)

    def _finish_all(self, reason: str) -> None:
        for e in self.pipeline.shutdown():
            self._carry[e.camera_id].append(e)
        for cam in list(self._carry):
            try:
                self.store.end_tracks(cam, tuple(self._carry.pop(cam)), time.time())
            except Exception as e:                              # noqa: BLE001
                log.error("could not persist final track endings for %s: %s (recover() will close them on next start)", cam, e)
        for cam in list(self.store._seg):
            self.store.close_camera(cam)

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        cams = {}
        for cid, st in self.ingest.stats().items():
            cams[cid] = {"state": st.state.value, "decoder": st.decoder, "stream_epoch": st.stream_epoch, "reconnects": st.reconnects,
                         "decoded": st.frames_decoded, "sampled": st.frames_sampled,
                         "dropped": {"rate": st.dropped_rate, "redundant": st.dropped_redundant, "queue": st.dropped_queue},
                         "malformed": st.malformed, "queue_depth": self.ingest.bus.depth(cid), "last_error": st.last_error,
                         "quarantined": time.time() < self._quarantine.get(cid, 0.0)}
        degraded = time.time() < self._store_paused_until or self._store_fails > 0
        return {"cameras": cams, "persistence": {"degraded": degraded, "paused": time.time() < self._store_paused_until,
                                                 "consecutive_failures": self._store_fails, **self.store.health()},
                "identity_errors": self.pipeline.identity_errors, "last_error": self.last_error,
                "processed_fps": self.m.rate("processed_fps"), **self.m.snapshot()}
