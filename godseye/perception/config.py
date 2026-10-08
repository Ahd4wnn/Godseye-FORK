"""Perception configuration (YAML -> dataclasses). Unknown keys are errors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import REID_CLASSES


class PerceptionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PromptConfig:
    text: str
    label: str
    reid_class: str = "other"        # person | vehicle | other
    threshold: float = 0.1


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str = "google/owlvit-base-patch32"
    backend: str = "tensorrt"        # tensorrt | torch  (explicit; never a silent fallback)
    precision: str = "fp16"          # tensorrt only: fp16 | fp32
    engine_dir: str = "models/nanoowl"
    max_batch: int = 1
    device: str = "cuda"
    nms_iou: float = 0.5
    max_detections: int = 100
    min_box_px: int = 8              # drop boxes smaller than this (original px) on either side
    prompts: tuple[PromptConfig, ...] = ()


@dataclass(frozen=True)
class TrackerConfig:
    track_thresh: float = 0.3        # high-score split (OWL scores are not YOLO-calibrated: tune per deployment)
    low_thresh: float = 0.1
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: float = 5.0          # effective sampled fps; max_time_lost = frame_rate/30*track_buffer frames
    mot20: bool = False
    trajectory_len: int = 50


@dataclass(frozen=True)
class ReidModelConfig:
    config_file: str
    weights: str
    device: str = "cuda"
    min_box_px: int = 48             # skip crops with a shorter side below this
    min_aspect: float = 0.0          # h/w bounds for a plausible crop (0 = off)
    max_aspect: float = 0.0
    batch_size: int = 32


@dataclass(frozen=True)
class IdentityConfig:
    enabled: bool = True
    min_observations: int = 3        # good ReID features a track needs before it is resolved
    recheck_observations: int = 8    # unmatched tracks are re-evaluated once at this count
    extract_every: int = 3           # after min_observations, refresh a track's ReID feature every N matched frames
    max_track_features: int = 10
    match_threshold: float = 0.70    # cosine; CALIBRATE on deployment data (see docs/TESTING.md)
    margin: float = 0.05             # best must beat the runner-up identity by this much
    ambiguous_floor: float = 0.55    # between floor and match_threshold (or low margin) => ambiguous, never merged
    max_exemplars: int = 20          # per identity
    max_overlap: float = 0.3         # skip Re-ID crops >30% covered by another same-class box (occlusion)
    allow_simultaneous_cross_camera: bool = True
    gallery_ttl_s: float = 0.0       # 0 = keep identities for the process lifetime


@dataclass(frozen=True)
class PerceptionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: dict = field(default_factory=dict)             # reid_class -> ReidModelConfig
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    base_dir: str = "."


def _build(cls, data, where):
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    if set(data) - names:
        raise PerceptionConfigError(f"{where}: unknown keys {sorted(set(data) - names)}")
    return cls(**data)


def parse(doc: dict, base_dir: str = ".") -> PerceptionConfig:
    unknown = set(doc) - {"detector", "tracker", "reid", "identity"}
    if unknown:
        raise PerceptionConfigError(f"unknown top-level keys {sorted(unknown)}")
    d = dict(doc.get("detector") or {})
    prompts = tuple(_build(PromptConfig, p, "detector.prompts") for p in d.pop("prompts", []))
    det = _build(DetectorConfig, d, "detector")
    det = dataclasses.replace(det, prompts=prompts)
    if not prompts:
        raise PerceptionConfigError("detector.prompts must define at least one open-vocabulary prompt")
    if det.backend not in ("tensorrt", "torch"):
        raise PerceptionConfigError("detector.backend must be tensorrt|torch")
    if det.precision not in ("fp16", "fp32"):
        raise PerceptionConfigError("detector.precision must be fp16|fp32")
    for p in prompts:
        if p.reid_class not in REID_CLASSES:
            raise PerceptionConfigError(f"prompt '{p.text}': reid_class must be one of {REID_CLASSES}")
        if not 0 < p.threshold < 1:
            raise PerceptionConfigError(f"prompt '{p.text}': threshold must be in (0,1)")
    trk = _build(TrackerConfig, doc.get("tracker"), "tracker")
    if not 0 < trk.low_thresh < trk.track_thresh < 1:
        raise PerceptionConfigError("tracker: need 0 < low_thresh < track_thresh < 1")
    reid = {}
    for cls_name, r in (doc.get("reid") or {}).items():
        if cls_name not in ("person", "vehicle"):
            raise PerceptionConfigError(f"reid: only person|vehicle models are supported, got '{cls_name}'")
        reid[cls_name] = _build(ReidModelConfig, r, f"reid.{cls_name}")
    ident = _build(IdentityConfig, doc.get("identity"), "identity")
    if not 0 <= ident.ambiguous_floor <= ident.match_threshold <= 1:
        raise PerceptionConfigError("identity: need ambiguous_floor <= match_threshold within [0,1]")
    needed = {p.reid_class for p in prompts} & {"person", "vehicle"}
    if ident.enabled and needed - set(reid):
        raise PerceptionConfigError(f"identity enabled but no reid model configured for {sorted(needed - set(reid))}")
    return PerceptionConfig(det, trk, reid, ident, base_dir)


def load_config(path: str | Path) -> PerceptionConfig:
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise PerceptionConfigError(f"cannot read {path}: {e}") from e
    return parse(doc, str(path.parent.parent if path.parent.name == "config" else path.parent))
