"""Storage configuration (YAML -> dataclass). Unknown keys are errors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


class StorageConfigError(ValueError):
    pass


@dataclass(frozen=True)
class StorageConfig:
    root: str = "data/store"             # holds godseye.db, chroma/, clips/, .lock (relative to the project root)
    clip_fps: float = 5.0                # playback fps of evidence clips (frames are sampled, not real-time; see frame_ts)
    idle_close_s: float = 5.0            # close a camera's clip segment this long after its last track was seen
    max_segment_s: float = 60.0          # hard cap on one clip's span
    min_free_bytes: int = 2 * 1024**3    # refuse to start a clip below this much free disk
    retry_after_s: float = 30.0          # after a clip failure, do not retry on that camera for this long
    crf: int = 26
    preset: str = "veryfast"
    encoder_threads: int = 2             # per clip encoder; ffmpeg's default (all cores) cost ~240 MB RSS per camera (measured)
    x264_params: str = "rc-lookahead=0:sync-lookahead=0:bframes=0"   # low-latency, low-memory encoding
    encoder_threads: int = 2             # per clip encoder; ffmpeg's default (all cores) cost ~240 MB RSS per camera (measured)
    x264_params: str = "rc-lookahead=0:sync-lookahead=0:bframes=0"   # low-latency, low-memory encoding
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    chroma_query_k: int = 100            # exemplars fetched per gallery query (identity = best exemplar among them)
    track_embed_every: int = 10          # refresh a live track's stored CLIP embedding every N hits (+ at track end)
    max_exemplars: int = 20              # Re-ID exemplars kept per identity


def parse(doc: dict | None) -> StorageConfig:
    doc = dict(doc or {})
    names = {f.name for f in dataclasses.fields(StorageConfig)}
    if set(doc) - names:
        raise StorageConfigError(f"unknown storage keys {sorted(set(doc) - names)}")
    c = StorageConfig(**doc)
    if c.clip_fps <= 0 or c.idle_close_s < 0 or c.max_segment_s <= 0 or not 0 <= c.crf <= 51:
        raise StorageConfigError("storage: clip_fps>0, idle_close_s>=0, max_segment_s>0, 0<=crf<=51 required")
    import re
    if not 1 <= c.encoder_threads <= 32 or not re.fullmatch(r"[A-Za-z0-9=:_.\-]*", c.x264_params):
        raise StorageConfigError("storage: encoder_threads in 1..32 and x264_params limited to [A-Za-z0-9=:_.-]")
    import re
    if not 1 <= c.encoder_threads <= 32 or not re.fullmatch(r"[A-Za-z0-9=:_.\-]*", c.x264_params):
        raise StorageConfigError("storage: encoder_threads in 1..32 and x264_params limited to [A-Za-z0-9=:_.-]")
    if c.chroma_query_k < 2 or c.max_exemplars < 2 or c.track_embed_every < 1:
        raise StorageConfigError("storage: chroma_query_k>=2, max_exemplars>=2, track_embed_every>=1 required")
    return c


def load_config(path: str | Path) -> StorageConfig:
    try:
        return parse(yaml.safe_load(Path(path).read_text()))
    except (OSError, yaml.YAMLError) as e:
        raise StorageConfigError(f"cannot read {path}: {e}") from e
