"""`python -m godseye.runtime trackA|web` -- run Track A (ingest + perception + storage) or the local web chat (Track B)."""
from __future__ import annotations

import logging
import signal
import sys
import threading

from .config import load_config
from .factory import build_track_a


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rcfg = load_config("config/runtime.yaml")
    if argv[:1] == ["trackA"]:
        svc = build_track_a(rcfg)
        svc.ingest.start()
        svc.start()
        stop = threading.Event()
        for sg in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sg, lambda *_: stop.set())
        while not stop.wait(30):
            h = svc.health()
            logging.info("health: fps=%.1f cameras=%s persistence=%s", h["processed_fps"],
                         {k: v["state"] for k, v in h["cameras"].items()}, "DEGRADED" if h["persistence"]["degraded"] else "ok")
        svc.ingest.stop()
        ok = svc.stop()
        svc.store.close()
        return 0 if ok else 1
    if argv[:1] == ["web"]:
        from ..web.server import serve
        return serve(rcfg)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
