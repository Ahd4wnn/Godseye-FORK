"""Query data model. `ParsedQuery` is what the LLM may produce (untrusted, validated, inert data). `ResolvedQueryContext`
is what leaves Phase 4: every field is verified against Track A's data / the knowledge store, never raw LLM text."""
from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class Intent(str, enum.Enum):
    find = "find"                # show me / find occurrences of X
    locate = "locate"            # where is / was X
    track_path = "track_path"    # where did X go / path across cameras
    count = "count"              # how many X
    when_seen = "when_seen"      # when did we see X
    unknown = "unknown"          # not a surveillance question


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ObjectSpec(_Strict):
    label: Short | None = None                      # the user's noun ("guy", "SUV") -- NOT yet a canonical label
    attributes: list[Short] = Field(default_factory=list, max_length=8)    # "red jacket", "white"


class TimeSpec(_Strict):
    """What the user said about time. The LLM only extracts it; code does the calendar arithmetic."""
    kind: Literal["none", "last", "day", "between"] = "none"
    amount: int | None = Field(default=None, ge=1, le=10000)
    unit: Literal["minutes", "hours", "days"] | None = None
    day: Short | None = None                        # today | yesterday | "N days ago" | YYYY-MM-DD
    part_of_day: Literal["morning", "afternoon", "evening", "night"] | None = None
    start_clock: Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")] | None = None
    end_clock: Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")] | None = None


class ReferentSpec(_Strict):
    kind: Literal["none", "pronoun", "named", "demonstrative"] = "none"      # he/it | "the delivery guy" | "that person"
    text: Short | None = None


class SpatialSpec(_Strict):
    relation: Literal["none", "left", "right", "top", "bottom", "center", "in_zone", "near_zone"] = "none"
    zone_text: Short | None = None                  # "the loading dock"


class ParsedQuery(_Strict):
    intent: Intent
    object: ObjectSpec = Field(default_factory=ObjectSpec)
    location_text: Short | None = None              # "the lobby" (a place, usually a camera / camera group)
    camera_mentions: list[Short] = Field(default_factory=list, max_length=8)
    time: TimeSpec = Field(default_factory=TimeSpec)
    referent: ReferentSpec = Field(default_factory=ReferentSpec)
    spatial: SpatialSpec = Field(default_factory=SpatialSpec)

    @field_validator("camera_mentions", mode="before")
    @classmethod
    def _dedupe(cls, v):
        return list(dict.fromkeys(v)) if isinstance(v, list) else v


# ---- clarification ---------------------------------------------------------------------------------------------
class Option(_Strict):
    value: str
    label: str


class Clarification(_Strict):
    cid: str
    slot: Literal["camera", "object", "referent", "time", "zone", "intent"]
    kind: Literal["multiple_choice", "camera_grid", "point_and_click", "region_selection", "free_text"]
    prompt: str
    options: list[Option] = Field(default_factory=list)
    mention: str | None = None                      # the user phrase being resolved (learned as an alias on success)
    camera_id: str | None = None                    # point_and_click / region_selection: which camera's frame


# ---- output ----------------------------------------------------------------------------------------------------
class Region(_Strict):
    camera_id: str | None
    x1: float = Field(ge=0, le=1); y1: float = Field(ge=0, le=1); x2: float = Field(ge=0, le=1); y2: float = Field(ge=0, le=1)
    name: str | None = None


class Provenance(_Strict):
    slot: str
    mention: str | None
    method: Literal["exact", "fuzzy", "semantic", "llm", "user", "context", "learned"]
    target: str
    score: float | None = None


class ResolvedQueryContext(_Strict):
    original_query: str
    intent: Intent
    object_label: str | None                         # canonical label (one of cfg.objects)
    attributes: list[str] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)     # verified to exist; empty = no camera constraint
    time_start: float | None = None                  # epoch seconds
    time_end: float | None = None
    global_id: str | None = None                     # resolved identity referent (cross-camera)
    track_ref: tuple[str, int, int] | None = None    # (camera_id, stream_epoch, track_id) when only a local track is known
    regions: list[Region] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
