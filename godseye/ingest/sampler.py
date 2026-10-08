"""Frame sampling: base/burst rate, MOG2 motion gating, redundant-frame dropping.

Pure function of (frame, timestamp) so it is deterministic and testable without GStreamer.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SamplingConfig


@dataclass(frozen=True)
class SampleDecision:
    emit: bool
    reason: str            # emit: first|base|burst|keepalive ; drop: rate|redundant
    motion: float | None   # latest motion score (None when gating disabled)


class FrameSampler:
    def __init__(self, cfg: SamplingConfig):
        self.cfg = cfg
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history, varThreshold=cfg.mog2_var_threshold, detectShadows=True
        ) if cfg.motion_gating else None
        self._motion_evals = 0
        self._motion_due = 0.0
        self._motion: float | None = None
        self._burst_until = float("-inf")
        self._due: float | None = None       # next scheduled emit time
        self._last_emit_ts: float | None = None
        self._last_emit_small: np.ndarray | None = None

    def _small_gray(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        sw = self.cfg.motion_width
        small = cv2.resize(bgr, (sw, max(1, round(h * sw / w))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def _update_motion(self, small: np.ndarray, ts: float) -> None:
        fg = self._mog.apply(small)
        self._motion_evals += 1
        if self._motion_evals <= self.cfg.motion_warmup_evals:
            self._motion = 0.0
        else:
            self._motion = float(np.count_nonzero(fg == 255)) / fg.size
            if self._motion >= self.cfg.motion_threshold:
                self._burst_until = ts + self.cfg.burst_hold_s
        self._motion_due = max(self._motion_due + 1.0 / self.cfg.burst_fps, ts - 1.0 / self.cfg.burst_fps)

    def offer(self, bgr: np.ndarray, ts: float) -> SampleDecision:
        """ts: monotonically non-decreasing seconds."""
        c = self.cfg
        need_motion = self._mog is not None and ts >= self._motion_due - 1e-9
        small = None
        if need_motion:
            small = self._small_gray(bgr)
            self._update_motion(small, ts)
        burst = ts < self._burst_until
        interval = 1.0 / (c.burst_fps if burst else c.base_fps)

        if self._last_emit_ts is None:
            return self._emit(bgr, ts, small, "first", interval)
        if self._due is not None and burst:
            self._due = min(self._due, self._last_emit_ts + interval)
        if ts < self._due - 1e-9:
            return SampleDecision(False, "rate", self._motion)

        keepalive = c.keepalive_s > 0 and ts - self._last_emit_ts >= c.keepalive_s
        if c.redundancy_threshold > 0 and not burst and not keepalive:
            small = small if small is not None else self._small_gray(bgr)
            diff = float(cv2.absdiff(small, self._last_emit_small).mean())
            if diff < c.redundancy_threshold:
                return SampleDecision(False, "redundant", self._motion)
        return self._emit(bgr, ts, small, "keepalive" if keepalive else ("burst" if burst else "base"), interval)

    def _emit(self, bgr, ts, small, reason, interval) -> SampleDecision:
        if self.cfg.redundancy_threshold > 0:
            self._last_emit_small = small if small is not None else self._small_gray(bgr)
        due = (self._due if self._due is not None else ts) + interval
        self._due = due if due > ts else ts + interval  # resync when late by >= one interval
        self._last_emit_ts = ts
        return SampleDecision(True, reason, self._motion)
