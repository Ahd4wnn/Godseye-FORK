"""Phase 2 pipeline: PreprocessedFrame -> NanoOWL (+reused embeddings) -> ByteTrack -> FastReID -> EnrichedFrame."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..ingest.types import PreprocessedFrame
from .config import PerceptionConfig
from .identity import IdentityGallery, IdentityResolver
from .nanoowl_detector import NanoOwlDetector
from .reid import FastReIDExtractor
from .tracker import CameraTracker, TrackMeta, ended_record
from .types import Detection, EnrichedFrame, TrackedObject

log = logging.getLogger("godseye.perception")


def _overlap_frac(box: np.ndarray, others: list[np.ndarray]) -> float:
    """Largest fraction of `box` covered by any other box (occlusion proxy)."""
    best, area = 0.0, max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
    for o in others:
        iw, ih = min(box[2], o[2]) - max(box[0], o[0]), min(box[3], o[3]) - max(box[1], o[1])
        if iw > 0 and ih > 0:
            best = max(best, float(iw * ih) / area)
    return best


class PerceptionPipeline:
    """Not thread-safe: drive it from one worker thread (GPU models are shared across cameras)."""

    def __init__(self, cfg: PerceptionConfig, detector=None, extractors: dict | None = None,
                 gallery: IdentityGallery | None = None):
        self.cfg = cfg
        self.detector = detector or NanoOwlDetector(cfg.detector)
        if extractors is None:
            extractors = {c: FastReIDExtractor(r) for c, r in cfg.reid.items()} if cfg.identity.enabled else {}
        self.extractors = extractors
        self.resolver = IdentityResolver(cfg.identity, gallery)
        self._cams: dict[str, CameraTracker] = {}

    def process(self, frame: PreprocessedFrame) -> EnrichedFrame:
        t = {}
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        t["detect"] = (time.perf_counter() - t0) * 1e3

        ended = []
        cam = self._cams.get(frame.camera_id)
        if cam is not None and cam.stream_epoch != frame.stream_epoch:      # reconnect: motion continuity is broken
            ended += cam.drain("stream_restart")
            cam = None
        if cam is None:
            cam = self._cams[frame.camera_id] = CameraTracker(frame.camera_id, self.cfg.tracker, frame.stream_epoch)
        h, w = frame.original_bgr.shape[:2]
        t0 = time.perf_counter()
        live, gone = cam.update(dets, h, w, frame.capture_ts)
        ended += gone
        t["track"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        self._identify(frame, dets, live)
        t["identity"] = (time.perf_counter() - t0) * 1e3

        for m, _ in ended:
            self.resolver.end(m)
        tracks = tuple(self._snapshot(m, box, score, det) for m, box, score, det in live)
        return EnrichedFrame(frame, dets, tracks, tuple(ended_record(m, r) for m, r in ended), t)

    def _identify(self, frame, dets: tuple[Detection, ...], live) -> None:
        pending: dict[str, list] = {}
        cfg = self.cfg.identity
        for m, _, _, det in live:
            ex = self.extractors.get(m.reid_class)
            if ex is None or not self.resolver.wants_feature(m):
                continue
            others = [d.bbox_xyxy for d in dets if d is not det and d.reid_class == m.reid_class]
            if _overlap_frac(det.bbox_xyxy, others) > cfg.max_overlap:
                continue                                  # occluded / crowded crop: would pollute the identity
            crop = ex.crop(frame.original_bgr, det.bbox_xyxy)
            if crop is not None:
                pending.setdefault(m.reid_class, []).append((m, crop))
        for cls, items in pending.items():
            feats = self.extractors[cls].extract([c for _, c in items])
            for (m, _), f in zip(items, feats):
                self.resolver.observe(m, f, frame.capture_ts)

    @staticmethod
    def _snapshot(m: TrackMeta, box, score, det) -> TrackedObject:
        status = m.identity_status if m.reid_class != "other" else "not_applicable"
        return TrackedObject(
            camera_id=m.camera_id, track_id=m.track_id, label=m.label, reid_class=m.reid_class, state="tracked",
            bbox_xyxy=box, confidence=score, detection=det, hits=m.hits, first_seen_ts=m.first_seen_ts,
            last_seen_ts=m.last_seen_ts, trajectory=tuple(m.trajectory), clip_embedding=m.clip_embedding,
            global_id=m.global_id, identity_status=status, identity_score=m.identity_score,
            identity_candidates=m.identity_candidates, stream_epoch=m.stream_epoch)

    def close_camera(self, camera_id: str, reason: str = "shutdown"):
        cam = self._cams.pop(camera_id, None)
        out = cam.drain(reason) if cam else []
        for m, _ in out:
            self.resolver.end(m)
        return tuple(ended_record(m, r) for m, r in out)

    def shutdown(self):
        return tuple(e for cid in list(self._cams) for e in self.close_camera(cid))
