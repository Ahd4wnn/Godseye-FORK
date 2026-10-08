from __future__ import annotations

from pathlib import Path

from .config import QueryConfig
from .embedder import OwlTextEmbedder
from .entities import EntityIndex
from .knowledge import KnowledgeStore
from .llm import OllamaClient
from .resolver import MentionResolver
from .understand import QueryUnderstanding

ROOT = Path(__file__).resolve().parents[2]


def build(cfg: QueryConfig, base_dir: Path = ROOT, embedder=None) -> QueryUnderstanding:
    """Wire Phase 4. Fails loudly if Ollama/the model is unavailable or Track A has not produced godseye.db."""
    root = Path(cfg.store_root)
    root = root if root.is_absolute() else base_dir / root
    llm = OllamaClient(cfg)
    llm.verify()
    kn = KnowledgeStore(root / "knowledge.db")
    ent = EntityIndex(root / "godseye.db")
    emb = embedder or OwlTextEmbedder(cfg.owl_model, cfg.embedder_device)
    return QueryUnderstanding(cfg, llm, kn, ent, MentionResolver(cfg, kn, emb, llm))
