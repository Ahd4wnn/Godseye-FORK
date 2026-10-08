"""Typed, validated ingestion configuration (YAML -> dataclasses). Unknown keys are errors."""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class ConfigError(ValueError):
    pass


SCHEMES = {"rtsp": "rtsp", "rtsps": "rtsp", "rtmp": "rtmp", "rtmps": "rtmp",
           "http": "hls", "https": "hls", "file": "file"}
HW_ACCEL = ("none", "nvcodec", "vaapi")


@dataclass(frozen=True)
class SamplingConfig:
    base_fps: float = 2.0            # architecture target
    burst_fps: float = 5.0           # architecture target
    burst_hold_s: float = 3.0        # stay in burst this long after last motion
    motion_gating: bool = True       # MOG2 foreground gating; False => base rate only
    motion_threshold: float = 0.005  # fraction of foreground pixels that counts as motion
    motion_width: int = 160          # analysis width (px) for MOG2 / redundancy
    mog2_history: int = 200
    mog2_var_threshold: float = 16.0
    motion_warmup_evals: int = 10    # MOG2 evals ignored while the model learns the background
    redundancy_threshold: float = 1.0  # mean abs gray diff (0-255) below which a frame is redundant; 0 disables
    keepalive_s: float = 5.0         # emit at least this often even if static


@dataclass(frozen=True)
class PrivacyMask:
    rect: tuple[float, float, float, float]  # normalised x1,y1,x2,y2
    mode: str = "black"                      # black | blur


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: int = 640
    pad_value: int = 114             # letterbox fill, 0-255
    keep_original: bool = True       # keep full-res BGR for evidence/clips
    privacy_masks: tuple[PrivacyMask, ...] = ()
    privacy_apply_to_original: bool = False


@dataclass(frozen=True)
class ReconnectConfig:
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    max_attempts: int = 0            # 0 = unlimited
    connect_timeout_s: float = 15.0  # no first frame within this => failed attempt
    stall_timeout_s: float = 10.0    # no frame for this long while streaming => reconnect


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str
    enabled: bool = True
    rtsp_transport: str = "tcp"      # tcp | udp
    latency_ms: int = 200
    hw_accel: str = "none"           # none | nvcodec | vaapi (fails loudly if unavailable)
    loop_file: bool = False          # file:// only: restart at EOS
    queue_size: int = 4
    queue_policy: str = "drop_oldest"  # drop_oldest | drop_newest
    reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    @property
    def protocol(self) -> str:
        return SCHEMES[urlparse(self.url).scheme]


@dataclass(frozen=True)
class IngestConfig:
    plugin_paths: tuple[str, ...] = ()
    preload_libs: tuple[str, ...] = ()
    cameras: tuple[CameraConfig, ...] = ()
    base_dir: str = "."


def redact_url(url: str) -> str:
    return re.sub(r"//[^/@]*@", "//***@", url)


def _build(cls, data: dict | None, where: str, **overrides):
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**{**data, **overrides})


def _merge(base: Any, data: dict | None, where: str):
    """Return a copy of dataclass `base` with `data` applied (unknown keys rejected)."""
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(base)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return dataclasses.replace(base, **data)


def _expand_env(url: str, where: str) -> str:
    def sub(m):
        if m.group(1) not in os.environ:
            raise ConfigError(f"{where}: environment variable {m.group(1)} is not set")
        return os.environ[m.group(1)]
    return re.sub(r"\$\{(\w+)\}", sub, url)


def _validate_sampling(s: SamplingConfig, where: str) -> None:
    if not (0 < s.base_fps <= s.burst_fps):
        raise ConfigError(f"{where}: need 0 < base_fps <= burst_fps")
    if s.motion_width < 16:
        raise ConfigError(f"{where}: motion_width must be >= 16")
    for k in ("burst_hold_s", "motion_threshold", "redundancy_threshold", "keepalive_s"):
        if getattr(s, k) < 0:
            raise ConfigError(f"{where}: {k} must be >= 0")


def _validate_preprocess(p: PreprocessConfig, where: str) -> None:
    if p.target_size < 32:
        raise ConfigError(f"{where}: target_size must be >= 32")
    if not 0 <= p.pad_value <= 255:
        raise ConfigError(f"{where}: pad_value must be 0-255")
    for m in p.privacy_masks:
        x1, y1, x2, y2 = m.rect
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ConfigError(f"{where}: privacy rect must be normalised x1<x2,y1<y2 within [0,1]")
        if m.mode not in ("black", "blur"):
            raise ConfigError(f"{where}: privacy mode must be black|blur")


def parse_camera(raw: dict, defaults: dict, idx: int) -> CameraConfig:
    where = f"cameras[{idx}]"
    raw = dict(raw)
    cid = raw.get("camera_id")
    if not cid or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(cid)):
        raise ConfigError(f"{where}: camera_id required, [A-Za-z0-9_.-]+ only")
    where = f"camera '{cid}'"
    if "url" not in raw:
        raise ConfigError(f"{where}: url required")
    url = _expand_env(str(raw.pop("url")), where)
    scheme = urlparse(url).scheme
    if scheme not in SCHEMES:
        raise ConfigError(f"{where}: unsupported URL scheme '{scheme}' (supported: {sorted(SCHEMES)})")
    sampling = _merge(SamplingConfig(), {**defaults.get("sampling", {}), **raw.pop("sampling", {})}, where + ".sampling")
    pre_raw = {**defaults.get("preprocess", {}), **raw.pop("preprocess", {})}
    masks = tuple(_build(PrivacyMask, {**m, "rect": tuple(m["rect"])}, where + ".privacy_masks")
                  for m in pre_raw.pop("privacy_masks", []))
    pre = _merge(PreprocessConfig(), pre_raw, where + ".preprocess")
    pre = dataclasses.replace(pre, privacy_masks=masks)
    rec = _merge(ReconnectConfig(), {**defaults.get("reconnect", {}), **raw.pop("reconnect", {})}, where + ".reconnect")
    cam_defaults = {k: v for k, v in defaults.items() if k not in ("sampling", "preprocess", "reconnect")}
    cam = _build(CameraConfig, {**cam_defaults, **raw}, where, url=url, sampling=sampling, preprocess=pre, reconnect=rec)
    _validate_sampling(sampling, where)
    _validate_preprocess(pre, where)
    if cam.hw_accel not in HW_ACCEL:
        raise ConfigError(f"{where}: hw_accel must be one of {HW_ACCEL}")
    if cam.rtsp_transport not in ("tcp", "udp"):
        raise ConfigError(f"{where}: rtsp_transport must be tcp|udp")
    if cam.queue_policy not in ("drop_oldest", "drop_newest"):
        raise ConfigError(f"{where}: queue_policy must be drop_oldest|drop_newest")
    if cam.queue_size < 1:
        raise ConfigError(f"{where}: queue_size must be >= 1")
    return cam


def load_config(path: str | Path) -> IngestConfig:
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    unknown = set(doc) - {"gstreamer", "defaults", "cameras"}
    if unknown:
        raise ConfigError(f"top-level unknown keys {sorted(unknown)}")
    gst = doc.get("gstreamer") or {}
    if set(gst) - {"plugin_paths", "preload_libs"}:
        raise ConfigError(f"gstreamer: unknown keys {sorted(set(gst) - {'plugin_paths', 'preload_libs'})}")
    cams = tuple(parse_camera(c, doc.get("defaults") or {}, i) for i, c in enumerate(doc.get("cameras") or []))
    ids = [c.camera_id for c in cams]
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate camera_id")
    return IngestConfig(plugin_paths=tuple(gst.get("plugin_paths", [])),
                        preload_libs=tuple(gst.get("preload_libs", [])), cameras=cams,
                        base_dir=str(path.parent.parent if path.parent.name == "config" else path.parent))
