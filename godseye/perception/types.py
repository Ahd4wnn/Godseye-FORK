"""Phase 2 output contract (consumed by Phase 3 storage)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ingest.types import PreprocessedFrame

REID_CLASSES = ("person", "vehicle", "other")


@dataclass(frozen=True)
class Provenance:
    """Where a detection came from; stored with every detection."""
    detector: str            # "nanoowl"
    model_name: str          # e.g. google/owlvit-base-patch32
    backend: str             # tensorrt | torch
    precision: str           # fp16 | fp32
    nanoowl_commit: str | None
    engine_sha256: str | None
    prompt_hash: str         # hash of (label, text, threshold) set
    image_size: int


@dataclass(frozen=True, eq=False)
class Detection:
    det_id: int                      # index within the frame
    camera_id: str
    frame_seq: int
    stream_epoch: int
    capture_ts: float
    pts_ns: int | None
    bbox_xyxy: np.ndarray            # float32[4], ORIGINAL-frame pixels, clipped to the frame
    label: str
    reid_class: str                  # person | vehicle | other
    confidence: float
    embedding: np.ndarray            # float32[512], L2-normalised, reused from the detection forward pass
    patch_index: int                 # OWL-ViT patch the box/embedding came from (source association)
    provenance: Provenance


@dataclass(frozen=True, eq=False)
class TrackedObject:
    camera_id: str
    track_id: int                    # camera-local (ByteTrack); NOT a global identity
    label: str
    reid_class: str
    state: str                       # tracked
    bbox_xyxy: np.ndarray            # float32[4] original px (Kalman-smoothed ByteTrack estimate)
    confidence: float
    detection: Detection | None      # the detection matched this frame
    hits: int                        # number of matched frames
    first_seen_ts: float
    last_seen_ts: float
    trajectory: tuple                # ((capture_ts, cx, cy), ...) most recent N observations
    clip_embedding: np.ndarray       # float32[512] mean of reused detection embeddings
    global_id: str | None            # cross-camera identity (FastReID), None until resolved / for reid_class other
    identity_status: str             # unresolved | new | matched | ambiguous | not_applicable
    identity_score: float | None     # similarity to the matched identity (matched) or best candidate
    identity_candidates: tuple = ()  # ((global_id, score), ...) when ambiguous
    stream_epoch: int = 0            # track_id restarts at 1 per (camera, stream_epoch): key tracks by all three


@dataclass(frozen=True)
class TrackEnded:
    camera_id: str
    track_id: int
    reason: str                      # lost_timeout | stream_restart | shutdown
    label: str
    reid_class: str
    first_seen_ts: float
    last_seen_ts: float
    hits: int
    global_id: str | None
    identity_status: str
    stream_epoch: int = 0


@dataclass(frozen=True, eq=False)
class EnrichedFrame:
    source: PreprocessedFrame
    detections: tuple[Detection, ...]
    tracks: tuple[TrackedObject, ...]
    ended_tracks: tuple[TrackEnded, ...]
    timings_ms: dict = field(default_factory=dict)
