"""Per-camera tracking: real ByteTrack, one instance per Re-ID class group, camera-unique `track_id`.

Groups (person / vehicle / other) are tracked separately so a person box can never be associated with a vehicle track.
`track_id` is camera-local (the allocator is shared by the groups of one camera); it is NOT a global identity.
"""
from __future__ import annotations

import itertools
from argparse import Namespace
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .bytetrack.basetrack import TrackState
from .bytetrack.byte_tracker import BYTETracker
from .config import TrackerConfig
from .types import REID_CLASSES, Detection, TrackEnded


@dataclass
class TrackMeta:
    """Mutable per-track state kept by the pipeline (the emitted TrackedObject is an immutable snapshot)."""
    camera_id: str
    track_id: int
    reid_class: str
    first_seen_ts: float
    last_seen_ts: float
    hits: int = 0
    labels: Counter = field(default_factory=Counter)
    trajectory: deque = field(default_factory=deque)
    emb_sum: np.ndarray = field(default_factory=lambda: np.zeros(512, np.float32))
    # identity bookkeeping (owned by IdentityResolver)
    features: list = field(default_factory=list)
    matched_since_feature: int = 0
    global_id: str | None = None
    identity_status: str = "unresolved"
    identity_score: float | None = None
    identity_candidates: tuple = ()
    rechecked: bool = False
    stream_epoch: int = 0

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]

    @property
    def clip_embedding(self) -> np.ndarray:
        n = np.linalg.norm(self.emb_sum)
        return (self.emb_sum / n).astype(np.float32) if n > 0 else self.emb_sum.copy()


class CameraTracker:
    def __init__(self, camera_id: str, cfg: TrackerConfig, stream_epoch: int):
        self.camera_id, self.cfg, self.stream_epoch = camera_id, cfg, stream_epoch
        ids = itertools.count(1)
        self._alloc = lambda: next(ids)
        args = Namespace(track_thresh=cfg.track_thresh, low_thresh=cfg.low_thresh, match_thresh=cfg.match_thresh,
                         track_buffer=cfg.track_buffer, mot20=cfg.mot20)
        self._bt = {c: BYTETracker(args, frame_rate=cfg.frame_rate, id_allocator=self._alloc) for c in REID_CLASSES}
        self.meta: dict[int, TrackMeta] = {}

    def update(self, dets: tuple[Detection, ...], h: int, w: int, ts: float):
        """-> (list[(TrackMeta, bbox_xyxy, score, Detection)], list[(TrackEnded-less TrackMeta, reason)])"""
        live, ended = [], []
        for cls, bt in self._bt.items():
            group = [d for d in dets if d.reid_class == cls]
            arr = (np.array([[*d.bbox_xyxy, d.confidence] for d in group], np.float64)
                   if group else np.zeros((0, 5), np.float64))
            out = bt.update(arr, (h, w), (h, w), payloads=group)
            for st in out:
                det = st.payload
                m = self.meta.get(st.track_id)
                if m is None:
                    m = self.meta[st.track_id] = TrackMeta(self.camera_id, st.track_id, cls, ts, ts,
                                                           trajectory=deque(maxlen=self.cfg.trajectory_len),
                                                           stream_epoch=self.stream_epoch)
                m.hits += 1
                m.last_seen_ts = ts
                m.labels[det.label] += 1
                m.emb_sum += det.embedding
                box = np.clip(st.tlbr, 0, [w, h, w, h]).astype(np.float32)
                m.trajectory.append((ts, float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                live.append((m, box, float(st.score), det))
            for st in bt.removed_stracks:               # never-activated (unconfirmed) tracks were never emitted
                m = self.meta.pop(st.track_id, None)
                if m is not None:
                    ended.append((m, "lost_timeout"))
            # ByteTrack drops a removed track from `lost_stracks` one frame late, using this list: keep only those
            # still pending (clearing it outright resurrects dead tracks; never pruning it leaks memory).
            bt.removed_stracks = [t for t in bt.removed_stracks if any(t is x for x in bt.lost_stracks)]
        return live, ended

    def drain(self, reason: str):
        out = [(m, reason) for m in self.meta.values()]
        self.meta.clear()
        return out


def ended_record(m: TrackMeta, reason: str) -> TrackEnded:
    return TrackEnded(m.camera_id, m.track_id, reason, m.label, m.reid_class, m.first_seen_ts, m.last_seen_ts, m.hits,
                      m.global_id, m.identity_status, m.stream_epoch)
