"""Track B (query understanding) configuration. Unknown keys are errors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


class QueryConfigError(ValueError):
    pass


@dataclass(frozen=True)
class QueryConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"            # DEVIATION from the Llama 3 8B in the spec: user-approved 2026-10-08 (only Qwen models installed)
    num_ctx: int = 4096
    timeout_s: float = 90.0
    llm_retries: int = 1               # extra attempts when the LLM output fails validation
    max_query_chars: int = 500
    store_root: str = "data/store"     # Track A's root: godseye.db is opened READ-ONLY; knowledge.db lives here too
    fuzzy_threshold: float = 0.85
    fuzzy_margin: float = 0.08
    semantic_threshold: float = 0.55   # cosine of OWL-ViT/CLIP text embeddings (short-phrase CLIP cosines are low: "guy"->person 0.68, "SUV"->car 0.75)
    semantic_margin: float = 0.10
    embedder_device: str = "cpu"       # keep the GPU for Track A and the LLM
    owl_model: str = "google/owlvit-base-patch32"
    objects: tuple = ("person", "car", "truck", "bus", "motorcycle", "bicycle")   # canonical labels (match detector prompt labels)
    object_aliases: tuple = ()         # extra (alias, label) pairs seeded at startup
    recent_identities: int = 5         # how many recent results a pronoun / "that person" may refer to
    timezone: str = "local"            # "local" or an IANA name


def parse(doc: dict | None) -> QueryConfig:
    doc = dict(doc or {})
    names = {f.name for f in dataclasses.fields(QueryConfig)}
    if set(doc) - names:
        raise QueryConfigError(f"unknown query keys {sorted(set(doc) - names)}")
    for k in ("objects", "object_aliases"):
        if k in doc:
            doc[k] = tuple(tuple(x) if isinstance(x, list) else x for x in doc[k])
    c = QueryConfig(**doc)
    if not c.ollama_url.startswith(("http://127.0.0.1", "http://localhost")):
        raise QueryConfigError("ollama_url must be local (no cloud LLM in the query path)")
    if not (0 < c.fuzzy_threshold <= 1 and 0 < c.semantic_threshold <= 1) or c.max_query_chars < 10:
        raise QueryConfigError("query: thresholds must be in (0,1], max_query_chars >= 10")
    return c


def load_config(path: str | Path) -> QueryConfig:
    try:
        return parse(yaml.safe_load(Path(path).read_text()))
    except (OSError, yaml.YAMLError) as e:
        raise QueryConfigError(f"cannot read {path}: {e}") from e
