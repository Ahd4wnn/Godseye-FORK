"""QueryPlan: the validated, allow-listed description of what to retrieve. Built deterministically from a
`ResolvedQueryContext` (already verified by Phase 4). The LLM does not touch plans: nothing it says can widen what runs.
Execution only ever consumes a validated QueryPlan whose operation is one of `Op`."""
from __future__ import annotations

import enum
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..query.schema import Intent, Region, ResolvedQueryContext
from .config import AnswerConfig

GID = re.compile(r"^(person|vehicle)-[0-9a-f]{12}$")
CAM = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class PlanError(ValueError):
    pass


class Op(str, enum.Enum):
    find_tracks = "find_tracks"
    count_tracks = "count_tracks"
    locate_latest = "locate_latest"
    when_seen = "when_seen"
    path = "path"


class Ranking(str, enum.Enum):
    recency = "recency"
    confidence = "confidence"
    duration = "duration"
    semantic = "semantic"


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticQuery(_S):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class TimeFilter(_S):
    start: float
    end: float


class ObjectFilter(_S):
    label: str
    reid_class: Literal["person", "vehicle", "other"]


class IdentityFilter(_S):
    global_id: str | None = None
    track_ref: tuple[str, int, int] | None = None


class QueryPlan(_S):
    operation: Op
    semantic_query: SemanticQuery | None = None
    camera_filter: list[str] = Field(default_factory=list, max_length=64)
    time_filter: TimeFilter | None = None
    spatial_filter: list[Region] = Field(default_factory=list, max_length=64)
    object_filter: ObjectFilter | None = None
    identity_filter: IdentityFilter | None = None
    ranking_strategy: Ranking = Ranking.recency
    reconstruction_required: bool = False
    limit: int = Field(default=10, ge=1, le=500)
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    original_query: str = ""
    notes: list[str] = Field(default_factory=list)


OPS = {Intent.find: Op.find_tracks, Intent.locate: Op.locate_latest, Intent.track_path: Op.path,
       Intent.count: Op.count_tracks, Intent.when_seen: Op.when_seen}


def make_plan(ctx: ResolvedQueryContext, cfg: AnswerConfig) -> QueryPlan:
    if ctx.intent not in OPS:
        raise PlanError(f"intent {ctx.intent.value!r} cannot be answered from stored data")
    op = OPS[ctx.intent]
    classes = dict(cfg.object_classes)
    obj = ObjectFilter(label=ctx.object_label, reid_class=classes[ctx.object_label]) if ctx.object_label in classes else None
    ident = IdentityFilter(global_id=ctx.global_id, track_ref=ctx.track_ref) if (ctx.global_id or ctx.track_ref) else None
    sem = None
    notes: list[str] = []
    if ctx.attributes and op in (Op.find_tracks, Op.locate_latest):
        sem = SemanticQuery(text=" ".join([*ctx.attributes, ctx.object_label or ""]).strip())
    elif ctx.attributes:
        notes.append(f"attributes {ctx.attributes} are ignored for '{op.value}' (visual attributes only rank find/locate results)")
    return QueryPlan(operation=op, semantic_query=sem, camera_filter=list(ctx.camera_ids),
                     time_filter=TimeFilter(start=ctx.time_start, end=ctx.time_end) if ctx.time_start is not None else None,
                     spatial_filter=list(ctx.regions), object_filter=obj, identity_filter=ident,
                     ranking_strategy=Ranking.semantic if sem else Ranking.recency, reconstruction_required=op == Op.path,
                     limit=cfg.default_limit, original_query=ctx.original_query, notes=notes)


def validate_plan(plan: QueryPlan, cfg: AnswerConfig, cameras: set[str], identity_exists, track_exists) -> QueryPlan:
    """Reject anything that is not safe/meaningful BEFORE any retrieval. Raises PlanError."""
    labels = {l for l, _ in cfg.object_classes}
    for c in plan.camera_filter:
        if not CAM.match(c) or c not in cameras:
            raise PlanError(f"unknown camera {c!r}")
    for r in plan.spatial_filter:
        if r.camera_id is not None and r.camera_id not in cameras:
            raise PlanError(f"region on unknown camera {r.camera_id!r}")
        if not (r.x1 < r.x2 and r.y1 < r.y2):
            raise PlanError("degenerate region")
    if plan.time_filter and not plan.time_filter.start < plan.time_filter.end:
        raise PlanError("time_filter.start must be before end")
    if plan.object_filter and plan.object_filter.label not in labels:
        raise PlanError(f"object label {plan.object_filter.label!r} is not in the vocabulary")
    i = plan.identity_filter
    if i:
        if (i.global_id is None) == (i.track_ref is None):
            raise PlanError("identity_filter needs exactly one of global_id / track_ref")
        if i.global_id and not (GID.match(i.global_id) and identity_exists(i.global_id)):
            raise PlanError(f"unknown identity {i.global_id!r}")
        if i.track_ref and not track_exists(*i.track_ref):
            raise PlanError(f"unknown track {i.track_ref!r}")
    if plan.limit > cfg.max_limit:
        raise PlanError(f"limit {plan.limit} exceeds max_limit {cfg.max_limit}")
    if plan.operation == Op.path and not plan.reconstruction_required:
        raise PlanError("path requires reconstruction_required")
    if plan.operation != Op.path and plan.reconstruction_required:
        raise PlanError("only the path operation may require reconstruction")
    if plan.ranking_strategy == Ranking.semantic and not plan.semantic_query:
        raise PlanError("semantic ranking needs a semantic_query")
    if plan.semantic_query and plan.operation not in (Op.find_tracks, Op.locate_latest):
        raise PlanError("semantic_query is only allowed for find_tracks / locate_latest")
    if plan.operation in (Op.find_tracks, Op.locate_latest, Op.count_tracks, Op.when_seen) and not (
            plan.object_filter or plan.identity_filter or plan.semantic_query):
        raise PlanError("a subject (object, identity or description) is required")
    return plan
