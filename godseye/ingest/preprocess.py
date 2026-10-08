"""Privacy masking, aspect-preserving letterbox resize, [0,1] normalisation."""
from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig
from .types import LetterboxTransform


def compute_letterbox(w: int, h: int, target: int) -> LetterboxTransform:
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid frame size {w}x{h}")
    scale = min(target / w, target / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return LetterboxTransform(w, h, target, scale, (target - new_w) // 2, (target - new_h) // 2, new_w, new_h)


def apply_privacy(bgr: np.ndarray, masks) -> np.ndarray:
    """Return a masked copy (never mutates the input)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    for m in masks:
        x1, y1, x2, y2 = (int(m.rect[0] * w), int(m.rect[1] * h), int(round(m.rect[2] * w)), int(round(m.rect[3] * h)))
        if m.mode == "black":
            out[y1:y2, x1:x2] = 0
        else:
            k = max(3, (min(x2 - x1, y2 - y1) // 2) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, LetterboxTransform, np.ndarray | None]:
        """Returns (model_image float32 RGB HWC [0,1], transform, original_bgr or None).

        The original is a copy that is safe to keep after the source buffer is released.
        """
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 HxWx3 BGR, got {bgr.dtype} {bgr.shape}")
        c = self.cfg
        masked = apply_privacy(bgr, c.privacy_masks) if c.privacy_masks else bgr
        h, w = masked.shape[:2]
        t = compute_letterbox(w, h, c.target_size)
        interp = cv2.INTER_AREA if t.scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(masked, (t.new_w, t.new_h), interpolation=interp)
        canvas = np.full((t.target, t.target, 3), c.pad_value, dtype=np.uint8)
        canvas[t.pad_top:t.pad_top + t.new_h, t.pad_left:t.pad_left + t.new_w] = resized
        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        original = None
        if c.keep_original:
            # masked is a private copy when masks exist; otherwise copy so the source buffer can be released
            original = masked if (c.privacy_masks and c.privacy_apply_to_original) else bgr.copy()
        return np.ascontiguousarray(image), t, original
