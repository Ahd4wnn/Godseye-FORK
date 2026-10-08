"""FastReID feature extraction (person / vehicle models) with a strict checkpoint loader.

The FastReID v0.1.1 release checkpoints (e.g. MSMT17 SBS-R50-ibn) store the BN-neck under `heads.bnneck.*` while the
current FastReID source expects `heads.bottleneck.0.*`. A plain `Checkpointer.load` silently leaves that BatchNorm at its
random/default init, degrading every embedding. This loader remaps the legacy keys and refuses to run if any
inference-relevant weight is still missing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ._paths import ROOT, add_third_party
from .config import ReidModelConfig

log = logging.getLogger("godseye.perception")

_KEY_REMAP = (("heads.bnneck.", "heads.bottleneck.0."),)
_ALLOWED_MISSING = {"heads.weight"}                       # training-only classifier
_IGNORED_UNEXPECTED = ("heads.classifier.weight", "num_batches_tracked")


class ReidError(RuntimeError):
    pass


def load_checkpoint_strict(model: torch.nn.Module, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # trusted: pinned release file
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    remapped = {}
    for k, v in sd.items():
        for old, new in _KEY_REMAP:
            if k.startswith(old):
                k = new + k[len(old):]
        remapped[k] = v
    for name in ("pixel_mean", "pixel_std"):           # non-persistent buffers: apply the training-time values
        if name in remapped:
            v = remapped.pop(name).reshape(getattr(model, name).shape).float()
            if not torch.allclose(v, getattr(model, name).cpu(), atol=1e-3):
                log.warning("%s: checkpoint %s differs from config; using checkpoint value", path.name, name)
            getattr(model, name).copy_(v)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    bad_missing = [k for k in missing if k not in _ALLOWED_MISSING]
    bad_unexpected = [k for k in unexpected if not any(s in k for s in _IGNORED_UNEXPECTED)]
    if bad_missing or bad_unexpected:
        raise ReidError(f"{path.name}: checkpoint/model mismatch. missing={bad_missing[:6]} unexpected={bad_unexpected[:6]}")


class FastReIDExtractor:
    def __init__(self, cfg: ReidModelConfig):
        add_third_party("fast-reid")
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model

        self.cfg = cfg
        root = lambda p: (Path(p) if Path(p).is_absolute() else ROOT / p)  # noqa: E731
        fr = get_cfg()
        fr.merge_from_file(str(root(cfg.config_file)))
        fr.MODEL.BACKBONE.PRETRAIN = False          # weights come from our checkpoint, not an ImageNet download
        fr.MODEL.DEVICE = cfg.device
        fr.freeze()
        self.size_hw = tuple(fr.INPUT.SIZE_TEST)    # (H, W)
        self.device = cfg.device
        w = root(cfg.weights)
        if not w.exists():
            raise ReidError(f"FastReID weights not found: {w}")
        self.model = build_model(fr).eval()
        load_checkpoint_strict(self.model, w)
        self.model.to(self.device)
        self.half = cfg.precision == "fp16" and str(self.device).startswith("cuda")
        if self.half:
            self.model.half()
        self.weights_name = w.name

    def crop(self, bgr: np.ndarray, bbox) -> np.ndarray | None:
        """Crop a detection for Re-ID; None if it is too small / implausible (not an error: low-quality observation)."""
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < self.cfg.min_box_px:
            return None
        ar = bh / max(bw, 1)
        if (self.cfg.min_aspect and ar < self.cfg.min_aspect) or (self.cfg.max_aspect and ar > self.cfg.max_aspect):
            return None
        return bgr[y1:y2, x1:x2]

    @torch.no_grad()
    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops -> (N, D) float32, L2-normalised (cosine space)."""
        if not crops:
            return np.zeros((0, 0), np.float32)
        outs = []
        for s in range(0, len(crops), self.cfg.batch_size):
            batch = []
            for c in crops[s:s + self.cfg.batch_size]:
                rgb = c[:, :, ::-1]                                             # model expects RGB
                img = cv2.resize(rgb, self.size_hw[::-1], interpolation=cv2.INTER_CUBIC)   # as FastReID demo
                batch.append(torch.as_tensor(img.astype("float32").transpose(2, 0, 1)))
            x = torch.stack(batch).to(self.device)
            f = self.model(x.half() if self.half else x).float()
            if not torch.isfinite(f).all():
                raise ReidError("non-finite ReID features")
            outs.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)
