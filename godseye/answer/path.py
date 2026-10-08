"""Path Reconstructor: an identity's movement across cameras from its stored tracks.

Model
  Visit      OBSERVED  -- one or more tracks of the identity on one camera (tracks closer than merge_gap_s are merged).
  Transition INFERRED  -- the identity was NOT observed between two consecutive visits; the link is an inference from
                          timestamps. It is never presented as an observation and carries a plausibility verdict:
     plausible            gap >= shortest possible travel time (topology) [and <= max_gap_s]
     unverified           no topology configured (or camera pair not covered): nothing can be checked
     impossible_too_fast  gap < shortest possible travel time - tolerance  (probable identity merge error)
     impossible_no_route  topology configured but no route between the cameras
     impossible_overlap   seen on two cameras at once, which are not declared overlapping (probable identity merge error)
     long_gap             gap > max_gap_s: treated as a new episode, no link asserted
  Impossible / long-gap transitions split the path into separate episodes."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import AnswerConfig
from .retrieval import Evidence, Retriever, Tracklet
from .topology import Topology

OBSERVED, INFERRED = "OBSERVED", "INFERRED"
BREAKS = {"impossible_too_fast", "impossible_no_route", "impossible_overlap", "long_gap"}


@dataclass
class Visit:
    camera_id: str
    start_ts: float
    end_ts: float
    tracks: list = field(default_factory=list)          # [(camera, epoch, track_id)]
    first_pos: tuple | None = None                      # (ts, cx, cy) pixels, from the stored observations
    last_pos: tuple | None = None
    mean_confidence: float | None = None
    evidence: Evidence | None = None
    kind: str = OBSERVED


@dataclass
class Transition:
    from_camera: str
    to_camera: str
    gap_s: float
    plausibility: str
    detail: str
    kind: str = INFERRED


@dataclass
class Path:
    global_id: str
    visits: list = field(default_factory=list)
    transitions: list = field(default_factory=list)     # transitions[i] links visits[i] -> visits[i+1]
    episodes: list = field(default_factory=list)        # [[visit index, ...]] split at BREAKS
    warnings: list = field(default_factory=list)
    topology_configured: bool = False


def reconstruct(global_id: str, retr: Retriever, cfg: AnswerConfig, topo: Topology, time_filter=None, cameras=()) -> Path:
    tracks: list[Tracklet] = retr.identity_tracks(global_id)
    if time_filter:
        tracks = [t for t in tracks if t.start_ts <= time_filter.end and t.last_seen_ts >= time_filter.start]
    if cameras:
        tracks = [t for t in tracks if t.camera_id in cameras]
    path = Path(global_id, topology_configured=topo.configured)
    visits: list[Visit] = []
    for t in sorted(tracks, key=lambda x: x.start_ts):
        v = visits[-1] if visits and visits[-1].camera_id == t.camera_id and t.start_ts - visits[-1].end_ts <= cfg.merge_gap_s else None
        if v is not None:
            v.end_ts = max(v.end_ts, t.last_seen_ts)
            v.tracks.append(t.ref)
        else:
            v = Visit(t.camera_id, t.start_ts, t.last_seen_ts, [t.ref])
            visits.append(v)
        first, last = retr.end_positions(t.track_row)
        v.first_pos = v.first_pos or first
        v.last_pos = last or v.last_pos
        ev = retr.best_evidence(t)
        if ev and (v.evidence is None or ev.confidence > v.evidence.confidence):
            v.evidence = ev
        if t.mean_confidence is not None:
            v.mean_confidence = t.mean_confidence if v.mean_confidence is None else (v.mean_confidence + t.mean_confidence) / 2
    path.visits = visits
    for a, b in zip(visits, visits[1:]):
        path.transitions.append(_judge(a, b, cfg, topo))
    path.episodes = [[0]] if visits else []
    for i, tr in enumerate(path.transitions):
        if tr.plausibility in BREAKS:
            path.episodes.append([i + 1])
            if tr.plausibility.startswith("impossible"):
                path.warnings.append(f"identity {global_id}: {tr.detail} -- the same global_id may cover two different objects (identity merge error)")
        else:
            path.episodes[-1].append(i + 1)
    if not topo.configured and len(visits) > 1:
        path.warnings.append("no camera topology is configured: cross-camera transitions are UNVERIFIED inferences from timestamps")
    return path


def _judge(a: Visit, b: Visit, cfg: AnswerConfig, topo: Topology) -> Transition:
    gap = b.start_ts - a.end_ts
    if a.camera_id == b.camera_id:
        if gap > cfg.max_gap_s:
            return Transition(a.camera_id, b.camera_id, gap, "long_gap", f"{gap:.0f}s between sightings on {a.camera_id} (> max_gap_s)")
        return Transition(a.camera_id, b.camera_id, gap, "plausible", f"left and re-entered {a.camera_id}")
    if gap < 0:                                                    # simultaneous presence on two cameras
        link = topo.link(a.camera_id, b.camera_id)
        if topo.configured and not (link and link.overlap):
            return Transition(a.camera_id, b.camera_id, gap, "impossible_overlap",
                              f"seen on {a.camera_id} and {b.camera_id} at the same time but they are not declared overlapping")
        return Transition(a.camera_id, b.camera_id, gap, "plausible" if topo.configured else "unverified",
                          f"simultaneous on {a.camera_id}/{b.camera_id}" + ("" if topo.configured else " (topology not configured)"))
    if gap > cfg.max_gap_s:
        return Transition(a.camera_id, b.camera_id, gap, "long_gap", f"{gap:.0f}s between {a.camera_id} and {b.camera_id} (> max_gap_s)")
    if not topo.configured:
        return Transition(a.camera_id, b.camera_id, gap, "unverified", "no camera topology configured")
    dmin = topo.min_travel(a.camera_id, b.camera_id)
    if dmin is None:
        return Transition(a.camera_id, b.camera_id, gap, "impossible_no_route", f"no route from {a.camera_id} to {b.camera_id} in the topology")
    if gap + cfg.travel_tolerance_s < dmin:
        return Transition(a.camera_id, b.camera_id, gap, "impossible_too_fast",
                          f"{gap:.0f}s from {a.camera_id} to {b.camera_id} but the fastest route takes {dmin:.0f}s")
    return Transition(a.camera_id, b.camera_id, gap, "plausible", f"{gap:.0f}s >= fastest route {dmin:.0f}s")
