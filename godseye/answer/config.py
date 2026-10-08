"""Phase 5 configuration (Track B, back half). Unknown keys are errors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


class AnswerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AnswerConfig:
    store_root: str = "data/store"        # Track A's root: godseye.db + chroma/ + clips/ are opened READ-ONLY by Track B
    topology: str = ""                    # path to the camera topology YAML ("" = none configured: transitions are 'unverified')
    default_limit: int = 10
    max_limit: int = 50
    candidate_cap: int = 2000             # max tracks pulled from SQLite before ranking
    merge_gap_s: float = 30.0             # same-camera tracks of one identity closer than this form one visit
    max_gap_s: float = 600.0              # a longer gap between sightings starts a new episode (no link is asserted)
    travel_tolerance_s: float = 2.0       # slack on min travel time (clock/sampling error)
    low_confidence: float = 0.35          # detection confidence below this is flagged
    count_merge_note: bool = True
    timezone: str = "local"               # "local" or an IANA name used to render timestamps
    llm_wording: bool = False             # let the local LLM reword answers (facts re-verified; falls back to the template)
    object_classes: tuple = (("person", "person"), ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"),
                             ("motorcycle", "other"), ("bicycle", "other"))


def parse(doc: dict | None) -> AnswerConfig:
    doc = dict(doc or {})
    names = {f.name for f in dataclasses.fields(AnswerConfig)}
    if set(doc) - names:
        raise AnswerConfigError(f"unknown answer keys {sorted(set(doc) - names)}")
    if "object_classes" in doc:
        doc["object_classes"] = tuple(tuple(x) for x in doc["object_classes"])
    c = AnswerConfig(**doc)
    if not (1 <= c.default_limit <= c.max_limit <= 500) or c.candidate_cap < c.max_limit:
        raise AnswerConfigError("answer: need 1 <= default_limit <= max_limit <= 500 and candidate_cap >= max_limit")
    if min(c.merge_gap_s, c.max_gap_s, c.travel_tolerance_s) < 0 or not 0 <= c.low_confidence <= 1:
        raise AnswerConfigError("answer: gaps/tolerance must be >= 0 and low_confidence in [0,1]")
    return c


def load_config(path: str | Path) -> AnswerConfig:
    try:
        return parse(yaml.safe_load(Path(path).read_text()))
    except (OSError, yaml.YAMLError) as e:
        raise AnswerConfigError(f"cannot read {path}: {e}") from e
