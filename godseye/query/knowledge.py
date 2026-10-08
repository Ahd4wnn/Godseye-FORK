"""Knowledge store: learned user mappings (aliases, zones). Track B's own SQLite file (`knowledge.db`); it never writes to
Track A's `godseye.db`. All SQL is parameterised; no value from the LLM or the user is ever interpolated into SQL."""
from __future__ import annotations

import functools
import json
import threading
import re
import sqlite3
import time
from pathlib import Path

KINDS = ("camera", "object", "identity", "zone")
SCHEMA = """
CREATE TABLE IF NOT EXISTS aliases (
    kind TEXT NOT NULL CHECK (kind IN ('camera','object','identity','zone')),
    alias TEXT NOT NULL, scope TEXT NOT NULL DEFAULT '',           -- scope: camera_id for zones, else ''
    target TEXT NOT NULL, source TEXT NOT NULL, created_ts REAL NOT NULL, uses INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, alias, scope));
CREATE TABLE IF NOT EXISTS zones (
    camera_id TEXT NOT NULL, name TEXT NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    created_ts REAL NOT NULL, PRIMARY KEY (camera_id, name));
"""


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"[\s_\-]+", " ", s.lower())).strip()


def _locked(fn):
    @functools.wraps(fn)
    def w(self, *a, **k):
        with self._mu:
            return fn(self, *a, **k)
    return w


class KnowledgeStore:
    def __init__(self, path: Path):
        self._mu = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    @_locked
    def learn(self, kind: str, alias: str, target: str, source: str = "user", scope: str = "") -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown alias kind {kind!r}")
        a = norm(alias)
        if not a or not target:
            raise ValueError("empty alias/target")
        with self.conn:
            self.conn.execute("INSERT INTO aliases(kind,alias,scope,target,source,created_ts) VALUES(?,?,?,?,?,?) "
                              "ON CONFLICT(kind,alias,scope) DO UPDATE SET target=excluded.target, source=excluded.source",
                              (kind, a, scope, target, source, time.time()))

    @_locked
    def lookup(self, kind: str, alias: str, scope: str | None = None) -> list[str]:
        q, args = "SELECT target FROM aliases WHERE kind=? AND alias=?", [kind, norm(alias)]
        if scope is not None:
            q += " AND scope=?"; args.append(scope)
        rows = [r[0] for r in self.conn.execute(q, args)]
        if rows:
            with self.conn:
                self.conn.execute("UPDATE aliases SET uses=uses+1 WHERE kind=? AND alias=?", [kind, norm(alias)])
        return list(dict.fromkeys(rows))

    @_locked
    def aliases(self, kind: str) -> dict[str, str]:
        return {r["alias"]: r["target"] for r in self.conn.execute("SELECT alias,target FROM aliases WHERE kind=?", (kind,))}

    @_locked
    def forget(self, kind: str, alias: str) -> int:
        with self.conn:
            return self.conn.execute("DELETE FROM aliases WHERE kind=? AND alias=?", (kind, norm(alias))).rowcount

    @_locked
    def save_zone(self, camera_id: str, name: str, x1: float, y1: float, x2: float, y2: float) -> None:
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("zone must be a non-empty rectangle in normalised [0,1] coordinates")
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO zones VALUES(?,?,?,?,?,?,?)", (camera_id, norm(name), x1, y1, x2, y2, time.time()))
        self.learn("zone", name, json.dumps([camera_id, norm(name)]), "user", scope=camera_id)

    @_locked
    def zones(self, name: str, camera_ids: list[str] | None = None) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM zones WHERE name=?", (norm(name),)).fetchall()
        return [dict(r) for r in rows if camera_ids is None or not camera_ids or r["camera_id"] in camera_ids]

    @_locked
    def zone_names(self) -> list[str]:
        return sorted({r[0] for r in self.conn.execute("SELECT name FROM zones")})

    @_locked
    def fact_rows(self) -> list[dict]:
        """Everything the user taught the system, for the console's memory panel (read-only view)."""
        out = [dict(r) for r in self.conn.execute("SELECT kind,alias,target,source,created_ts FROM aliases WHERE kind IN ('camera','identity') ORDER BY created_ts DESC")]
        out += [{"kind": "zone", **dict(r)} for r in self.conn.execute("SELECT camera_id,name,x1,y1,x2,y2,created_ts FROM zones ORDER BY created_ts DESC")]
        return out

    @_locked
    def forget_zone(self, camera_id: str, name: str) -> int:
        with self.conn:
            n = self.conn.execute("DELETE FROM zones WHERE camera_id=? AND name=?", (camera_id, norm(name))).rowcount
            self.conn.execute("DELETE FROM aliases WHERE kind='zone' AND alias=? AND scope=?", (norm(name), camera_id))
        return n

    def close(self) -> None:
        self.conn.close()
