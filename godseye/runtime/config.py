"""Runtime (integration) configuration. Unknown keys are errors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


class RuntimeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    ingest: str = "config/ingest.yaml"
    perception: str = "config/perception.yaml"
    storage: str = "config/storage.yaml"
    query: str = "config/query.yaml"
    answer: str = "config/answer.yaml"
    poll_s: float = 0.5                   # frame wait; also the cadence of source-ended checks
    quarantine_after: int = 5             # consecutive per-camera processing failures before the camera is paused
    quarantine_s: float = 30.0
    store_fail_threshold: int = 3         # consecutive storage failures before writes pause (circuit breaker)
    store_backoff_s: float = 5.0
    store_backoff_max_s: float = 60.0
    query_timeout_s: float = 120.0        # overall wall-clock limit for one question
    query_max_inflight: int = 2           # concurrent questions; more are refused as busy
    session_ttl_s: float = 3600.0
    max_sessions: int = 200


def parse(doc: dict | None) -> RuntimeConfig:
    doc = dict(doc or {})
    names = {f.name for f in dataclasses.fields(RuntimeConfig)}
    if set(doc) - names:
        raise RuntimeConfigError(f"unknown runtime keys {sorted(set(doc) - names)}")
    c = RuntimeConfig(**doc)
    if min(c.poll_s, c.quarantine_s, c.store_backoff_s, c.query_timeout_s) <= 0 or min(c.quarantine_after, c.store_fail_threshold, c.query_max_inflight, c.max_sessions) < 1:
        raise RuntimeConfigError("runtime: durations must be > 0 and counts >= 1")
    return c


def load_config(path: str | Path) -> RuntimeConfig:
    try:
        return parse(yaml.safe_load(Path(path).read_text()))
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeConfigError(f"cannot read {path}: {e}") from e
