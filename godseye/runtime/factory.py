"""Wire the two tracks from config. Track A and Track B are built independently so they can run in separate processes."""
from __future__ import annotations

from pathlib import Path

from ..answer.config import load_config as load_answer
from ..answer.engine import AnswerEngine
from ..ingest.config import load_config as load_ingest
from ..ingest.manager import IngestManager
from ..perception.config import load_config as load_perception
from ..perception.pipeline import PerceptionPipeline
from ..query.config import load_config as load_query
from ..query.factory import build as build_query
from ..storage.config import load_config as load_storage
from ..storage.store import EvidenceStore
from .config import RuntimeConfig, load_config
from .metrics import Metrics
from .service import TrackAService
from .trackb import QueryService

ROOT = Path(__file__).resolve().parents[2]


def build_track_a(rcfg: RuntimeConfig, base: Path = ROOT, metrics: Metrics | None = None) -> TrackAService:
    store = EvidenceStore(load_storage(base / rcfg.storage), base)
    pipe = PerceptionPipeline(load_perception(base / rcfg.perception), gallery=store.gallery)
    ing = IngestManager(load_ingest(base / rcfg.ingest))
    return TrackAService(ing, pipe, store, rcfg, metrics)


def build_track_b(rcfg: RuntimeConfig, base: Path = ROOT, metrics: Metrics | None = None) -> QueryService:
    qcfg = load_query(base / rcfg.query)
    qu = build_query(qcfg, base)
    eng = AnswerEngine(load_answer(base / rcfg.answer), embedder=qu.res.emb, llm=qu.llm, base_dir=base)
    return QueryService(qu, eng, rcfg, metrics)
