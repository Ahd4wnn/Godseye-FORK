"""Cross-camera identity: `global_id` assignment from FastReID features. In-memory gallery; Phase 3 persists it.

Conservative by design: a track is merged into an existing identity only if it is clearly the best candidate
(score >= match_threshold AND beats the runner-up by `margin`) and that identity is not simultaneously another
track on the SAME camera. Anything in between is `ambiguous` (no global_id, candidates reported) -- a missed merge is
recoverable, a false merge corrupts the path of two people.
"""
from __future__ import annotations

import uuid

import numpy as np

from .config import IdentityConfig
from .tracker import TrackMeta


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class IdentityGallery:
    """Exemplar gallery per Re-ID class. Interface (query / new / add) is what Phase 3 re-implements on ChromaDB."""

    def __init__(self, max_exemplars: int):
        self.max_exemplars = max_exemplars
        self._ex: dict[str, dict[str, list[np.ndarray]]] = {}     # reid_class -> gid -> exemplars (unit vectors)
        self.last_seen: dict[str, tuple[str, float]] = {}          # gid -> (camera_id, ts)

    def query(self, reid_class: str, feat: np.ndarray) -> list[tuple[str, float]]:
        """Identities ranked by best-exemplar cosine similarity."""
        out = [(gid, float(np.max(np.stack(ex) @ feat))) for gid, ex in self._ex.get(reid_class, {}).items()]
        return sorted(out, key=lambda x: -x[1])

    def new(self, reid_class: str, feat: np.ndarray, camera_id: str, ts: float) -> str:
        gid = f"{reid_class}-{uuid.uuid4().hex[:12]}"
        self._ex.setdefault(reid_class, {})[gid] = [feat]
        self.last_seen[gid] = (camera_id, ts)
        return gid

    def add(self, reid_class: str, gid: str, feat: np.ndarray, camera_id: str, ts: float) -> None:
        ex = self._ex[reid_class][gid]
        ex.append(feat)
        if len(ex) > self.max_exemplars:
            del ex[1]                       # keep the first (oldest, usually cleanest) and the most recent ones
        self.last_seen[gid] = (camera_id, ts)

    def __len__(self) -> int:
        return sum(len(g) for g in self._ex.values())


class IdentityResolver:
    def __init__(self, cfg: IdentityConfig, gallery: IdentityGallery | None = None):
        self.cfg = cfg
        self.gallery = gallery or IdentityGallery(cfg.max_exemplars)
        self._active: dict[str, set[tuple[str, int]]] = {}         # gid -> {(camera_id, track_id)} currently alive

    def wants_feature(self, m: TrackMeta) -> bool:
        """Should the pipeline spend a Re-ID forward pass on this track this frame?"""
        c = self.cfg
        if not c.enabled or m.reid_class not in ("person", "vehicle"):
            return False
        if len(m.features) < c.min_observations:
            return True
        if m.identity_status == "ambiguous" and m.rechecked:
            return False                    # gave it its one re-check; stays ambiguous (never force-merged)
        m.matched_since_feature += 1
        return m.matched_since_feature >= c.extract_every

    def observe(self, m: TrackMeta, feat: np.ndarray, ts: float) -> None:
        c = self.cfg
        feat = _unit(feat.astype(np.float32))
        m.features.append(feat)
        m.matched_since_feature = 0
        if len(m.features) > c.max_track_features:
            del m.features[0]
        n_obs = len(m.features)
        if m.global_id is not None:                                 # resolved: keep growing the identity's exemplars
            self.gallery.add(m.reid_class, m.global_id, feat, m.camera_id, ts)
            return
        if n_obs < c.min_observations:
            return
        if m.identity_status == "ambiguous" and (m.rechecked or n_obs < c.recheck_observations):
            return
        if m.identity_status == "ambiguous":
            m.rechecked = True
        self._resolve(m, ts)

    def _resolve(self, m: TrackMeta, ts: float) -> None:
        c = self.cfg
        q = _unit(np.mean(np.stack(m.features), axis=0))
        ranked = [(g, s) for g, s in self.gallery.query(m.reid_class, q)
                  if not any(cam == m.camera_id and tid != m.track_id for cam, tid in self._active.get(g, ()))]
        if not c.allow_simultaneous_cross_camera:
            ranked = [(g, s) for g, s in ranked if not self._active.get(g)]
        best = ranked[0] if ranked else None
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        if best is None or best[1] < c.ambiguous_floor:
            gid = self.gallery.new(m.reid_class, q, m.camera_id, ts)
            self._bind(m, gid, "new", None)
        elif best[1] >= c.match_threshold and best[1] - runner >= c.margin:
            self.gallery.add(m.reid_class, best[0], q, m.camera_id, ts)
            self._bind(m, best[0], "matched", best[1])
        else:
            m.identity_status, m.identity_score = "ambiguous", best[1]
            m.identity_candidates = tuple((g, round(s, 4)) for g, s in ranked[:3] if s >= c.ambiguous_floor)

    def _bind(self, m: TrackMeta, gid: str, status: str, score: float | None) -> None:
        m.global_id, m.identity_status, m.identity_score, m.identity_candidates = gid, status, score, ()
        self._active.setdefault(gid, set()).add((m.camera_id, m.track_id))

    def end(self, m: TrackMeta) -> None:
        if m.global_id is not None:
            s = self._active.get(m.global_id)
            if s:
                s.discard((m.camera_id, m.track_id))
