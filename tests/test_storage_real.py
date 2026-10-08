"""Real models -> real store: video frames through PerceptionPipeline (NanoOWL TRT + ByteTrack + FastReID), persisted by
EvidenceStore (SQLite + ChromaDB + ffmpeg clips), then restarted and re-verified."""
import shutil

import pytest

from tests.test_perception_real import VIDEO, cv2, np, pframe, torch  # noqa: F401  (shared helpers/skips)

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg"),
              pytest.mark.skipif(not VIDEO, reason="no real test video")]

from godseye.perception.config import load_config  # noqa: E402
from godseye.perception.pipeline import PerceptionPipeline  # noqa: E402
from godseye.storage.clips import probe  # noqa: E402
from godseye.storage.config import StorageConfig  # noqa: E402
from godseye.storage.store import EvidenceStore  # noqa: E402


def test_pipeline_to_store_roundtrip(tmp_path):
    scfg = StorageConfig(root=str(tmp_path / "store"), min_free_bytes=0, track_embed_every=5)
    store = EvidenceStore(scfg)
    pipe = PerceptionPipeline(load_config("config/perception.yaml"), gallery=store.gallery)
    cap = cv2.VideoCapture(VIDEO)
    step, k, seq, seen = max(1, round(cap.get(cv2.CAP_PROP_FPS) / 5)), 0, 0, {}
    while seq < 45:
        ok, bgr = cap.read()
        if not ok:
            break
        if k % step == 0:
            out = pipe.process(pframe(bgr, seq=seq, cam="lobby"))
            store.write(out)
            for t in out.tracks:
                seen[t.track_id] = t
            seq += 1
        k += 1
    final = pipe.shutdown()
    from godseye.ingest.types import PreprocessedFrame  # noqa: F401
    last = pframe(bgr, seq=seq, cam="lobby")
    from godseye.perception.types import EnrichedFrame
    store.write(EnrichedFrame(last, (), (), final))
    store.close()

    s = EvidenceStore(scfg)                                            # restart
    assert s.recovered["stale_tracks_closed"] == 0 and s.recovered["part_files_removed"] == 0
    tr = {r["track_id"]: r for r in s.conn.execute("SELECT * FROM tracks")}
    assert set(tr) == set(seen) and all(r["end_ts"] is not None for r in tr.values())
    assert any(r["global_id"] for r in tr.values())
    for tid, t in seen.items():                                        # DB agrees with what the pipeline emitted
        assert tr[tid]["hits"] == t.hits and tr[tid]["global_id"] == t.global_id and tr[tid]["label"] == t.label
    clips = s.conn.execute("SELECT * FROM clips").fetchall()
    assert clips and all(probe(s.cfg, s.clips_dir / c["path"])["frames"] == c["frames"] for c in clips)
    n_obs = s.conn.execute("SELECT count(*) FROM observations").fetchone()[0]
    assert n_obs == s.conn.execute("SELECT count(*) FROM observations WHERE clip_id IS NOT NULL").fetchone()[0] > 20
    gids = {r["global_id"] for r in tr.values() if r["global_id"]}
    assert len(s.gallery) == s.conn.execute("SELECT count(*) FROM identities").fetchone()[0] >= len(gids)
    assert s.vectors.count("track_clip") == len(tr)
    assert all(not v for v in s.reconcile(deep=True).values())
    s.close()
