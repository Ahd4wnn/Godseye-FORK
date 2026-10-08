"""Measured performance of the integrated Track A (and optionally Track B) on THIS machine.
   python -m godseye.runtime.bench --video PATH --cameras 2 --seconds 60 [--queries 4] [--out report.json]
Each camera loops the same file (decode + sampling + inference + Re-ID + storage are all real)."""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

from ..gst_env import init_gst  # noqa: F401
from ..ingest.config import IngestConfig, parse_camera
from ..ingest.manager import IngestManager
from ..perception.config import load_config as load_perception
from ..perception.pipeline import PerceptionPipeline
from ..storage.config import StorageConfig
from ..storage.store import EvidenceStore
from .config import RuntimeConfig
from .metrics import HostSampler, Metrics
from .service import TrackAService

ROOT = Path(__file__).resolve().parents[2]
PLUGINS = [".gst-local/plugins"]
PRELOAD = [".gst-local/lib/libgstcodecparsers-1.0.so.0", ".gst-local/lib/libgstmpegts-1.0.so.0"]


def run(video: str, cameras: int, seconds: float, base_fps: float, burst_fps: float, clips: bool = True) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="godseye-bench-"))
    store = EvidenceStore(StorageConfig(root=str(tmp / "store"), min_free_bytes=0 if clips else 2**62, track_embed_every=10))
    t0 = time.time()
    pipe = PerceptionPipeline(load_perception(ROOT / "config/perception.yaml"), gallery=store.gallery)
    pipe.detector.warmup()
    init_s = time.time() - t0
    cams = [{"camera_id": f"cam{i}", "url": f"file://{video}", "loop_file": True, "sampling": {"base_fps": base_fps, "burst_fps": burst_fps}}
            for i in range(cameras)]
    ing = IngestManager(IngestConfig(tuple(PLUGINS), tuple(PRELOAD), tuple(parse_camera(c, {}, i) for i, c in enumerate(cams)), str(ROOT)))
    m = Metrics()
    svc = TrackAService(ing, pipe, store, RuntimeConfig(), m)
    host = HostSampler(1.0)
    ing.start()
    svc.start()
    time.sleep(8)                                              # warm-up (stream start, first inferences) is excluded
    base = {c: (s.frames_decoded, s.frames_sampled, s.dropped_queue, s.dropped_rate, s.dropped_redundant) for c, s in ing.stats().items()}
    base_proc = {c: m.counters.get(f"frames_processed.{c}", 0) for c in base}
    m.__init__()                                               # reset latency windows & counters after warm-up
    host.start()
    t1 = time.time()
    depth_max = {c: 0 for c in base}
    while time.time() - t1 < seconds:
        for c in base:
            depth_max[c] = max(depth_max[c], ing.bus.depth(c))
        time.sleep(0.1)
    dt = time.time() - t1
    stats = ing.stats()
    host.stop()
    out = {"machine": "RTX 5050 Laptop 8GB, see CLAUDE.md DEPENDENCIES", "video": Path(video).name, "cameras": cameras, "seconds": round(dt, 1),
           "sampling": {"base_fps": base_fps, "burst_fps": burst_fps}, "init_s": round(init_s, 1), "per_camera": {}}
    for c, s in stats.items():
        d0 = base[c]
        out["per_camera"][c] = {
            "ingest_fps": round((s.frames_decoded - d0[0]) / dt, 1), "sampled_fps": round((s.frames_sampled - d0[1]) / dt, 2),
            "processed_fps": round(m.counters.get(f"frames_processed.{c}", 0) / dt, 2), "queue_depth_max": depth_max[c],
            "dropped_queue": s.dropped_queue - d0[2], "dropped_rate": s.dropped_rate - d0[3], "dropped_redundant": s.dropped_redundant - d0[4]}
    out["latency_ms"] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in m.snapshot()["latency_ms"].items()}
    out["total_processed_fps"] = round(sum(v["processed_fps"] for v in out["per_camera"].values()), 2)
    out["host"] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in host.summary().items()}
    out["counters"] = {k: v for k, v in m.counters.items() if not k.startswith("frames_processed")}
    out["persistence"] = store.health()
    ing.stop()
    svc.stop()
    store.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cameras", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--base-fps", type=float, default=2.0)
    ap.add_argument("--burst-fps", type=float, default=5.0)
    ap.add_argument("--no-clips", action="store_true", help="diagnostic: disable clip encoding to see its cost")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    res = run(a.video, a.cameras, a.seconds, a.base_fps, a.burst_fps, clips=not a.no_clips)
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
