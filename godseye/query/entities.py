"""Read-only view of Track A's data (cameras, identities, tracks). Track B never writes to godseye.db."""
from __future__ import annotations

import sqlite3
from pathlib import Path


class EntityIndex:
    def __init__(self, db_path: Path):
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} not found: run Track A (storage) first")
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")

    def cameras(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT camera_id FROM cameras ORDER BY camera_id")]

    def identity(self, global_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM identities WHERE global_id=?", (global_id,)).fetchone()
        return dict(r) if r else None

    def track(self, camera_id: str, stream_epoch: int, track_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM tracks WHERE camera_id=? AND stream_epoch=? AND track_id=?",
                              (camera_id, stream_epoch, track_id)).fetchone()
        return dict(r) if r else None

    def objects_at(self, camera_id: str, x: float, y: float, ts: float, frame_wh: tuple[int, int], max_dt: float = 3.0) -> list[dict]:
        """Tracked objects whose box contains the clicked point (x, y normalised 0-1) in the observation nearest `ts`."""
        px, py = x * frame_wh[0], y * frame_wh[1]
        rows = self.conn.execute(
            "SELECT o.track_row, o.ts, t.track_id, t.stream_epoch, t.global_id, t.label, t.reid_class, "
            "(o.x2-o.x1)*(o.y2-o.y1) AS area FROM observations o JOIN tracks t USING(track_row) "
            "WHERE o.camera_id=? AND abs(o.ts-?)<=? AND o.x1<=? AND o.x2>=? AND o.y1<=? AND o.y2>=? "
            "ORDER BY abs(o.ts-?), area", (camera_id, ts, max_dt, px, px, py, py, ts)).fetchall()
        seen, out = set(), []
        for r in rows:
            if r["track_row"] not in seen:
                seen.add(r["track_row"]); out.append(dict(r))
        return out

    def close(self) -> None:
        self.conn.close()
