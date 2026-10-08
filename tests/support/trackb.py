"""Fixtures for Phase 4 tests: a Track A database created with the REAL schema (storage.db), plus a scripted LLM double
(used only where a test needs to control model output, e.g. malformed / hostile output)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from godseye.query.config import QueryConfig
from godseye.query.entities import EntityIndex
from godseye.query.knowledge import KnowledgeStore
from godseye.query.resolver import MentionResolver
from godseye.query.understand import QueryUnderstanding
from godseye.storage.db import connect

NOW = datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc)
PERSON_A = "person-aaaaaaaaaaaa"
PERSON_B = "person-bbbbbbbbbbbb"
CAR_A = "vehicle-cccccccccccc"


def make_track_a(root):
    """cameras lobby_cam, cam2, north_gate; 2 persons + 1 car; person A is tracked at (0.4-0.6, 0.2-0.8) of 1000x500 at ts=100."""
    c = connect(root / "godseye.db")
    for cam in ("lobby_cam", "cam2", "north_gate"):
        c.execute("INSERT INTO cameras VALUES(?,?,?)", (cam, 0, 1000))
    for gid, cls in ((PERSON_A, "person"), (PERSON_B, "person"), (CAR_A, "vehicle")):
        c.execute("INSERT INTO identities VALUES(?,?,?,?,?)", (gid, cls, 0, 1000, "lobby_cam"))
    for row, (cam, tid, gid, label, cls) in enumerate([("lobby_cam", 1, PERSON_A, "person", "person"), ("lobby_cam", 2, None, "person", "person"),
                                                       ("cam2", 1, CAR_A, "car", "vehicle")], 1):
        c.execute("INSERT INTO tracks(track_row,camera_id,stream_epoch,track_id,label,reid_class,start_ts,last_seen_ts,hits,global_id,identity_status)"
                  " VALUES(?,?,?,?,?,?,?,?,?,?,?)", (row, cam, 0, tid, label, cls, 90, 110, 5, gid, "new" if gid else "unresolved"))
    box = {1: (400, 100, 600, 400), 2: (50, 50, 150, 300), 3: (100, 100, 300, 300)}
    for row, cam in ((1, "lobby_cam"), (2, "lobby_cam"), (3, "cam2")):
        c.execute("INSERT INTO observations(track_row,camera_id,frame_seq,ts,x1,y1,x2,y2,confidence,label) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (row, cam, 1, 100.0, *box[row], 0.9, "x"))
    c.close()


class ScriptedLLM:
    """Test double for OllamaClient: returns queued dicts (or raises). Records (system, user) for security assertions."""
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def chat_json(self, system, user, schema):
        self.calls.append((system, user))
        o = self.outputs.pop(0) if self.outputs else {"choice": None}
        if isinstance(o, Exception):
            raise o
        return o

    def verify(self):
        return {}


class FakeEmbedder:
    """Deterministic embedder double for unit tests of the semantic stage (real embedder is used in test_query_real)."""
    def __init__(self, table):
        self.table = table

    def embed(self, texts):
        import numpy as np
        return np.stack([np.asarray(self.table.get(t, [0, 0, 1]), np.float32) / np.linalg.norm(self.table.get(t, [0, 0, 1])) for t in texts])


def make_qu(tmp_path, llm, cfg=None, embedder=None, clock=lambda: NOW):
    cfg = cfg or QueryConfig(object_aliases=(("guy", "person"), ("man", "person"), ("woman", "person")))
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    if not (root / "godseye.db").exists():
        make_track_a(root)
    kn = KnowledgeStore(root / "knowledge.db")
    ent = EntityIndex(root / "godseye.db")
    return QueryUnderstanding(cfg, llm, kn, ent, MentionResolver(cfg, kn, embedder, llm), clock), kn, ent


def pq(**kw):
    """A ParsedQuery-shaped dict as the LLM would emit it."""
    kw.setdefault("intent", "find")
    return kw


__all__ = ["make_qu", "ScriptedLLM", "FakeEmbedder", "pq", "NOW", "PERSON_A", "PERSON_B", "CAR_A", "json"]
