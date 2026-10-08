"""ChromaDB: persistent vector storage. Collections:
  reid_person / reid_vehicle  FastReID exemplars (the identity gallery), cosine space, metadata {global_id, camera_id, ts}
  track_clip                  per-track mean of the reused NanoOWL/CLIP embeddings (512-d), metadata {camera_id, stream_epoch, track_id, global_id, reid_class}
Ids are deterministic, so every write is an idempotent upsert. SQLite stays the source of truth for metadata."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from .config import StorageConfig
from .db import DatabaseError

COLLECTIONS = ("reid_person", "reid_vehicle", "track_clip")


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.col = {n: self.client.get_or_create_collection(n, metadata={"hnsw:space": "cosine"}, embedding_function=None)
                    for n in COLLECTIONS}

    def upsert(self, name: str, ids: list[str], vecs, metas: list[dict]) -> None:
        self.col[name].upsert(ids=ids, embeddings=[np.asarray(v, np.float32).tolist() for v in vecs], metadatas=metas)

    def search(self, name: str, vec, k: int, where: dict | None = None) -> list[tuple[str, float, dict]]:
        """-> [(id, cosine similarity, metadata)] best first."""
        c = self.col[name]
        n = c.count()
        if n == 0:
            return []
        r = c.query(query_embeddings=[np.asarray(vec, np.float32).tolist()], n_results=min(k, n), where=where,
                    include=["metadatas", "distances"])
        return [(i, 1.0 - d, m) for i, d, m in zip(r["ids"][0], r["distances"][0], r["metadatas"][0])]

    def get(self, name: str, ids: list[str] | None = None, where: dict | None = None, include_vectors: bool = False):
        inc = ["metadatas"] + (["embeddings"] if include_vectors else [])
        return self.col[name].get(ids=ids, where=where, include=inc)

    def delete(self, name: str, ids: list[str]) -> None:
        if ids:
            self.col[name].delete(ids=ids)

    def count(self, name: str) -> int:
        return self.col[name].count()


class PersistentGallery:
    """Chroma-backed drop-in for perception.identity.IdentityGallery (same query / new / add / __len__).

    Write order (cross-store atomicity is impossible, so the order makes failures harmless): vector first (idempotent),
    then the SQLite identity row. A crash in between leaves an orphan vector that `reconcile` removes; the reverse
    (row without vectors) can never be produced by this order."""

    def __init__(self, cfg: StorageConfig, vectors: VectorStore, conn):
        self.cfg, self.v, self.conn = cfg, vectors, conn
        self._ex: dict[str, list[tuple[float, str]]] = {}          # gid -> [(ts, vector id)] oldest first (lazy cache)

    @staticmethod
    def _cls(gid: str) -> str:
        c = gid.split("-", 1)[0]
        if c not in ("person", "vehicle"):
            raise DatabaseError(f"malformed global_id {gid!r}")
        return c

    def query(self, reid_class: str, feat: np.ndarray) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for _, sim, m in self.v.search(f"reid_{reid_class}", feat, self.cfg.chroma_query_k):
            g = m["global_id"]
            best[g] = max(best.get(g, -1.0), sim)
        return sorted(best.items(), key=lambda x: -x[1])

    def _put(self, gid: str, feat: np.ndarray, camera_id: str, ts: float) -> None:
        vid = hashlib.sha1(gid.encode() + camera_id.encode() + repr(ts).encode() + np.float32(feat).tobytes()).hexdigest()[:24]
        self.v.upsert(f"reid_{self._cls(gid)}", [f"{gid}:{vid}"], [feat], [{"global_id": gid, "camera_id": camera_id, "ts": ts}])
        ex = self._load(gid)
        if all(i != f"{gid}:{vid}" for _, i in ex):
            ex.append((ts, f"{gid}:{vid}"))
        while len(ex) > self.cfg.max_exemplars:                      # keep the first (cleanest) + most recent
            self.v.delete(f"reid_{self._cls(gid)}", [ex.pop(1)[1]])

    def _load(self, gid: str):
        if gid not in self._ex:
            r = self.v.get(f"reid_{self._cls(gid)}", where={"global_id": gid})
            self._ex[gid] = sorted((m["ts"], i) for i, m in zip(r["ids"], r["metadatas"]))
        return self._ex[gid]

    def new(self, reid_class: str, feat: np.ndarray, camera_id: str, ts: float) -> str:
        gid = f"{reid_class}-{uuid.uuid4().hex[:12]}"
        self._put(gid, feat, camera_id, ts)
        self._touch(gid, reid_class, camera_id, ts)
        return gid

    def add(self, reid_class: str, gid: str, feat: np.ndarray, camera_id: str, ts: float) -> None:
        self._put(gid, feat, camera_id, ts)
        self._touch(gid, reid_class, camera_id, ts)

    def _touch(self, gid: str, reid_class: str, camera_id: str, ts: float) -> None:
        self.conn.execute("INSERT INTO identities(global_id,reid_class,first_seen,last_seen,last_camera) VALUES(?,?,?,?,?) "
                          "ON CONFLICT(global_id) DO UPDATE SET last_seen=max(last_seen,excluded.last_seen), "
                          "last_camera=CASE WHEN excluded.last_seen>=last_seen THEN excluded.last_camera ELSE last_camera END",
                          (gid, reid_class, ts, ts, camera_id))

    def __len__(self) -> int:
        return self.conn.execute("SELECT count(*) FROM identities").fetchone()[0]
