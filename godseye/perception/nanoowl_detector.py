"""NanoOWL open-vocabulary detector with embedding reuse.

Uses NanoOWL's own `OwlPredictor` (OWL-ViT image/text encoders, box + class heads, roi-align preprocessing) unchanged.
The only additions are (1) the TensorRT 11 engine loader (`trt_runtime`), (2) a decode that keeps the patch index of
every detection so the detection's CLIP-aligned class embedding can be reused, and (3) NMS / clipping / provenance.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import batched_nms

from ..ingest.types import PreprocessedFrame
from ._paths import ROOT, add_third_party, git_commit
from .config import DetectorConfig
from .types import Detection, Provenance

log = logging.getLogger("godseye.perception")


class DetectorError(RuntimeError):
    pass


class DetectorInputError(DetectorError):
    """The frame cannot be fed to the detector (e.g. original frame missing / privacy-masking policy violated)."""


class NanoOwlDetector:
    def __init__(self, cfg: DetectorConfig):
        add_third_party("nanoowl")
        from nanoowl.owl_predictor import OwlPredictor

        self.cfg = cfg
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise DetectorError("CUDA requested but torch.cuda.is_available() is False")
        self.predictor = OwlPredictor(cfg.model_name, device=cfg.device)
        self.image_size = self.predictor.image_size
        self.engine_sha: str | None = None
        if cfg.backend == "tensorrt":
            self._attach_engine()
        self.labels = [p.label for p in cfg.prompts]
        self.reid_classes = [p.reid_class for p in cfg.prompts]
        self.thresholds = torch.tensor([p.threshold for p in cfg.prompts], device=cfg.device)
        with torch.no_grad():
            self.text = self.predictor.encode_text([p.text for p in cfg.prompts])
        phash = hashlib.sha256(repr([(p.label, p.text, p.threshold, p.reid_class) for p in cfg.prompts]).encode()).hexdigest()[:16]
        self.provenance = Provenance(
            detector="nanoowl", model_name=cfg.model_name, backend=cfg.backend,
            precision=cfg.precision if cfg.backend == "tensorrt" else "fp32",
            nanoowl_commit=git_commit(add_third_party("nanoowl")), engine_sha256=self.engine_sha,
            prompt_hash=phash, image_size=self.image_size)

    # ---- engine -----------------------------------------------------------------------------------
    def _engine_path(self) -> Path:
        import tensorrt as trt
        cap = "sm%d%d" % torch.cuda.get_device_capability(self.cfg.device)
        name = f"{self.cfg.model_name.split('/')[-1]}-{self.cfg.precision}-trt{trt.__version__}-{cap}-b{self.cfg.max_batch}.engine"
        d = Path(self.cfg.engine_dir)
        return (d if d.is_absolute() else ROOT / d) / name

    def _attach_engine(self) -> None:
        from .trt_runtime import TrtError, TrtImageEncoder, build_engine, export_onnx
        path = self._engine_path()
        try:
            if not path.exists():
                import tempfile
                log.warning("building TensorRT engine %s (one-time, cached)", path.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as td:
                    onnx = Path(td) / "enc.onnx"
                    export_onnx(self.predictor, onnx, half=self.cfg.precision == "fp16")
                    build_engine(onnx, path, self.image_size, self.cfg.max_batch)
            self.engine_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.predictor.image_encoder_engine = TrtImageEncoder(path, self.cfg.max_batch)
        except (TrtError, ImportError, RuntimeError) as e:
            raise DetectorError(f"TensorRT backend requested but unavailable: {e}. "
                                f"Set detector.backend: torch to run NanoOWL's PyTorch path explicitly.") from e

    # ---- inference --------------------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, frame: PreprocessedFrame) -> tuple[Detection, ...]:
        if frame.original_bgr is None:
            raise DetectorInputError("frame.original_bgr is None: set preprocess.keep_original: true for this camera")
        if frame.privacy_masked and not frame.original_privacy_masked:
            raise DetectorInputError("privacy masks configured but not applied to original_bgr: set preprocess.privacy_apply_to_original: true")
        bgr = frame.original_bgr
        h, w = bgr.shape[:2]
        dev = self.cfg.device
        t = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(dev).permute(2, 0, 1)[None].float()   # RGB 0-255
        image = self.predictor.image_preprocessor(t)
        rois = torch.tensor([[0, 0, w, h]], dtype=image.dtype, device=dev)
        enc = self.predictor.encode_rois(image, rois, pad_square=True)
        keep = self._decode(enc)
        boxes = enc.pred_boxes[0][keep["patch"]].clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        scores, labels, patches = keep["score"], keep["label"], keep["patch"]
        ok = ((boxes[:, 2] - boxes[:, 0]) >= self.cfg.min_box_px) & ((boxes[:, 3] - boxes[:, 1]) >= self.cfg.min_box_px)
        boxes, scores, labels, patches = boxes[ok], scores[ok], labels[ok], patches[ok]
        if len(boxes):
            idx = batched_nms(boxes, scores, labels, self.cfg.nms_iou)[: self.cfg.max_detections]
            boxes, scores, labels, patches = boxes[idx], scores[idx], labels[idx], patches[idx]
        emb = enc.image_class_embeds[0][patches]
        emb = emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-6)
        boxes, scores, labels, patches, emb = (x.cpu().numpy() for x in (boxes, scores, labels, patches, emb))
        out = []
        for i in range(len(boxes)):
            e = emb[i].astype(np.float32)
            self._validate_embedding(e)
            out.append(Detection(
                det_id=i, camera_id=frame.camera_id, frame_seq=frame.frame_seq, stream_epoch=frame.stream_epoch,
                capture_ts=frame.capture_ts, pts_ns=frame.pts_ns, bbox_xyxy=boxes[i].astype(np.float32),
                label=self.labels[labels[i]], reid_class=self.reid_classes[labels[i]], confidence=float(scores[i]),
                embedding=e, patch_index=int(patches[i]), provenance=self.provenance))
        return tuple(out)

    def _decode(self, enc) -> dict:
        """NanoOWL `OwlPredictor.decode` math (cosine logits -> shift/scale -> sigmoid -> per-label threshold), but
        returning patch indices. Equivalence with `OwlPredictor.decode` is asserted in tests."""
        e = enc.image_class_embeds
        e = e / (torch.linalg.norm(e, dim=-1, keepdim=True) + 1e-6)
        q = self.text.text_embeds
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        logits = (torch.einsum("...pd,...qd->...pq", e, q) + enc.logit_shift) * enc.logit_scale
        mx = torch.sigmoid(logits)[0].max(dim=-1)
        mask = mx.values > self.thresholds[mx.indices]
        patch = mask.nonzero(as_tuple=True)[0]
        return {"patch": patch, "score": mx.values[patch], "label": mx.indices[patch]}

    @staticmethod
    def _validate_embedding(e: np.ndarray) -> None:
        if e.shape != (512,) or not np.isfinite(e).all() or abs(float(np.linalg.norm(e)) - 1.0) > 1e-3:
            raise DetectorError(f"invalid detection embedding: shape={e.shape} norm={np.linalg.norm(e):.4f}")

    def warmup(self, n: int = 3) -> None:
        dummy = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        from ..ingest.preprocess import compute_letterbox
        f = PreprocessedFrame("warmup", 0, 0, 0, time.time(), None, np.zeros((1, 1, 3), np.float32),
                              compute_letterbox(self.image_size, self.image_size, self.image_size), dummy, "first", None)
        for _ in range(n):
            self.detect(f)
