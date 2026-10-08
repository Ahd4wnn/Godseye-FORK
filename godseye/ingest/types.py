"""Output contract of Phase 1 (consumed by Phase 2 perception)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class CameraState(str, enum.Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    ENDED = "ended"      # file source reached EOS
    FAILED = "failed"    # fatal / retries exhausted; will not retry
    STOPPED = "stopped"


@dataclass(frozen=True)
class LetterboxTransform:
    """original(w,h) -> scaled by `scale` -> padded to target x target."""
    orig_w: int
    orig_h: int
    target: int
    scale: float
    pad_left: int
    pad_top: int
    new_w: int
    new_h: int

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes (N,4) in letterboxed model pixels to original-frame pixels, clipped."""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_left) / self.scale
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_top) / self.scale
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.orig_w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.orig_h)
        return b

    def to_model(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes in original pixels to letterboxed model pixels."""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale + self.pad_left
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale + self.pad_top
        return b


@dataclass(frozen=True, eq=False)
class PreprocessedFrame:
    """One sampled, preprocessed frame. Phase 2 input contract.

    image:        float32, shape (target, target, 3), RGB, values in [0,1], C-contiguous.
    original_bgr: uint8 (H,W,3) BGR full-resolution frame (None if keep_original=False).
    capture_ts:   wall-clock epoch seconds when the frame was pulled from the pipeline.
    stream_epoch: increments on every (re)connect; downstream trackers must reset on change.
    """
    camera_id: str
    frame_seq: int          # index among sampled frames of this camera
    source_seq: int         # index among all decoded frames of this camera (this process)
    stream_epoch: int
    capture_ts: float
    pts_ns: int | None
    image: np.ndarray
    transform: LetterboxTransform
    original_bgr: np.ndarray | None
    sample_reason: str      # first | base | burst | keepalive
    motion_score: float | None
    privacy_masked: bool = False          # masks applied to `image`
    original_privacy_masked: bool = False  # masks also applied to `original_bgr`


@dataclass
class CameraStats:
    state: CameraState = CameraState.IDLE
    decoder: str | None = None
    stream_epoch: int = 0
    reconnects: int = 0
    frames_decoded: int = 0
    frames_sampled: int = 0
    dropped_rate: int = 0
    dropped_redundant: int = 0
    dropped_queue: int = 0
    malformed: int = 0
    last_error: str | None = None
    last_frame_ts: float | None = None
    extra: dict = field(default_factory=dict)
