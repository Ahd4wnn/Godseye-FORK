"""Run in a subprocess: write frames while the process file-size limit (RLIMIT_FSIZE) makes every large write fail with EFBIG,
the closest real-OS stand-in for a full disk that needs no root. Prints a JSON summary."""
import json
import resource
import signal
import sys

import numpy as np

sys.path.insert(0, ".")
from godseye.ingest.preprocess import compute_letterbox  # noqa: E402
from godseye.ingest.types import PreprocessedFrame  # noqa: E402
from godseye.perception.types import Detection, EnrichedFrame, Provenance, TrackedObject  # noqa: E402
from godseye.storage.config import StorageConfig  # noqa: E402
from godseye.storage.store import EvidenceStore  # noqa: E402

root, limit = sys.argv[1], int(sys.argv[2])
H, W = 240, 320
PROV = Provenance("nanoowl", "t", "torch", "fp32", None, None, "x", 768)
emb = np.ones(512, np.float32) / np.sqrt(512)
store = EvidenceStore(StorageConfig(root=root, min_free_bytes=0, track_embed_every=2, retry_after_s=0.0))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
errors, ok = [], 0
for k in range(60):
    box = np.array([10 + k, 20, 60 + k, 100], np.float32)
    det = Detection(0, "cam", k, 0, float(k), None, box, "person", "person", .8, emb, 0, PROV)
    t = TrackedObject("cam", 1, "person", "person", "tracked", box, .8, det, k + 1, 0.0, float(k), (), emb, None, "unresolved", None, (), 0)
    f = PreprocessedFrame("cam", k, k, 0, float(k), None, np.zeros((1, 1, 3), np.float32), compute_letterbox(W, H, 640),
                          np.random.default_rng(k).integers(0, 255, (H, W, 3), dtype=np.uint8), "base", None)
    try:
        store.write(EnrichedFrame(f, (), (t,), ()))
        ok += 1
    except Exception as e:                          # noqa: BLE001 - the point is that nothing crashes the process
        errors.append(type(e).__name__)
try:
    store.close()
except Exception as e:                              # noqa: BLE001
    errors.append("close:" + type(e).__name__)
print(json.dumps({"ok_writes": ok, "errors": sorted(set(errors)), "n_errors": len(errors), "health": {k: v for k, v in store.health().items() if k != "recovered"}}))
