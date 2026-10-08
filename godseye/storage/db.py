"""SQLite: source of truth for structured metadata. One writer process (enforced by the store's lock file)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE cameras (
    camera_id TEXT PRIMARY KEY, first_seen REAL NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE identities (
    global_id TEXT PRIMARY KEY, reid_class TEXT NOT NULL CHECK (reid_class IN ('person','vehicle')),
    first_seen REAL NOT NULL, last_seen REAL NOT NULL, last_camera TEXT);
CREATE TABLE clips (
    clip_id INTEGER PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(camera_id), stream_epoch INTEGER NOT NULL,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, path TEXT NOT NULL UNIQUE,      -- path relative to <root>/clips
    fps REAL NOT NULL, frames INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    frame_ts TEXT NOT NULL,                                                         -- JSON [capture_ts per frame]
    size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('complete','missing','corrupt')),
    created_ts REAL NOT NULL);
CREATE TABLE tracks (
    track_row INTEGER PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
    stream_epoch INTEGER NOT NULL, track_id INTEGER NOT NULL,                      -- camera-local (ByteTrack)
    label TEXT NOT NULL, reid_class TEXT NOT NULL,
    start_ts REAL NOT NULL, last_seen_ts REAL NOT NULL, end_ts REAL,               -- end_ts NULL = still open
    hits INTEGER NOT NULL,
    global_id TEXT REFERENCES identities(global_id),                               -- cross-camera (FastReID)
    identity_status TEXT NOT NULL, identity_score REAL, identity_candidates TEXT,
    end_reason TEXT, embedding_id TEXT,
    UNIQUE (camera_id, stream_epoch, track_id));
CREATE TABLE observations (
    obs_id INTEGER PRIMARY KEY, track_row INTEGER NOT NULL REFERENCES tracks(track_row),
    camera_id TEXT NOT NULL, frame_seq INTEGER NOT NULL, ts REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,        -- tracker estimate, original px
    dx1 REAL, dy1 REAL, dx2 REAL, dy2 REAL,                                        -- raw detection box, original px
    confidence REAL NOT NULL, label TEXT NOT NULL,
    clip_id INTEGER REFERENCES clips(clip_id), clip_frame INTEGER,                 -- both set only once the clip is published
    UNIQUE (track_row, frame_seq));
CREATE INDEX ix_tracks_global ON tracks(global_id);
CREATE INDEX ix_tracks_cam_time ON tracks(camera_id, start_ts, end_ts);
CREATE INDEX ix_tracks_open ON tracks(end_ts) WHERE end_ts IS NULL;
CREATE INDEX ix_obs_cam_ts ON observations(camera_id, ts);
CREATE INDEX ix_obs_track ON observations(track_row, ts);
CREATE INDEX ix_obs_clip ON observations(clip_id);
CREATE INDEX ix_clips_cam_time ON clips(camera_id, start_ts, end_ts);
CREATE INDEX ix_ident_last ON identities(last_seen);
"""


class DatabaseError(RuntimeError):
    pass


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)   # autocommit; we open explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver == 0:
        conn.executescript("BEGIN;" + SCHEMA + f"PRAGMA user_version={SCHEMA_VERSION};COMMIT;")
    elif ver != SCHEMA_VERSION:
        raise DatabaseError(f"{path}: schema version {ver}, code expects {SCHEMA_VERSION} (no migration defined)")
    return conn
