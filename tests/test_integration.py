"""Phase 6: Track A and Track B integrated with REAL components (GStreamer file source -> NanoOWL/ByteTrack/FastReID ->
SQLite/ChromaDB/ffmpeg clips -> real local LLM -> planner/retrieval/response). Fault injection uses monkeypatching of the
real objects (the injected fault is the only fake)."""
import shutil
import sqlite3
import threading
import time

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
from tests.test_perception_real import VIDEO  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg"), pytest.mark.skipif(not VIDEO, reason="no real test video")]

from godseye.answer.config import AnswerConfig  # noqa: E402
from godseye.answer.engine import AnswerEngine  # noqa: E402
from godseye.answer.response import Status  # noqa: E402
from godseye.ingest import CameraState, IngestManager  # noqa: E402
from godseye.perception.config import load_config as load_perception  # noqa: E402
from godseye.perception.pipeline import PerceptionPipeline  # noqa: E402
from godseye.query.config import QueryConfig  # noqa: E402
from godseye.query.embedder import OwlTextEmbedder  # noqa: E402
from godseye.query.entities import EntityIndex  # noqa: E402
from godseye.query.knowledge import KnowledgeStore  # noqa: E402
from godseye.query.llm import LLMError, OllamaClient  # noqa: E402
from godseye.query.resolver import MentionResolver  # noqa: E402
from godseye.query.understand import QueryUnderstanding  # noqa: E402
from godseye.runtime.config import RuntimeConfig  # noqa: E402
from godseye.runtime.service import TrackAService  # noqa: E402
from godseye.runtime.trackb import QueryService  # noqa: E402
from godseye.storage.config import StorageConfig  # noqa: E402
from godseye.storage.store import EvidenceStore  # noqa: E402

SAMPLING = {"base_fps": 5.0, "burst_fps": 5.0}
RCFG = RuntimeConfig(poll_s=0.2, quarantine_after=3, quarantine_s=2.0, store_fail_threshold=2, store_backoff_s=0.5, store_backoff_max_s=1.0,
                     query_timeout_s=3.0)


@pytest.fixture(scope="module")
def pipe():
    return PerceptionPipeline(load_perception("config/perception.yaml"))


@pytest.fixture(scope="module")
def embedder():
    return OwlTextEmbedder("google/owlvit-base-patch32", "cpu")


class Stack:
    def __init__(self, gst, make_cfg, pipe, tmp_path, cams, rcfg=RCFG):
        self.root = tmp_path / "store"
        self.store = EvidenceStore(StorageConfig(root=str(self.root), min_free_bytes=0, track_embed_every=5))
        pipe._cams.clear()
        pipe.resolver.gallery = self.store.gallery
        self.pipe = pipe
        self.ingest = IngestManager(make_cfg(*cams))
        self.svc = TrackAService(self.ingest, pipe, self.store, rcfg)

    def run(self, timeout=120, until=None):
        self.ingest.start()
        self.svc.start()
        t = time.time()
        while time.time() - t < timeout:
            st = self.ingest.stats()
            if until and until(self):
                break
            if not until and all(s.state in (CameraState.ENDED, CameraState.FAILED) for s in st.values()) and set(st) <= self.svc._closed:
                break
            time.sleep(0.2)
        return self

    def stop(self):
        self.ingest.stop()
        ok = self.svc.stop()
        self.store.close()
        return ok

    def db(self):
        c = sqlite3.connect(f"file:{self.root / 'godseye.db'}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c


def cam(cid, url=None, **kw):
    return {"camera_id": cid, "url": url or f"file://{VIDEO}", "sampling": SAMPLING, **kw}


@pytest.fixture
def stack(gst, make_cfg, pipe, tmp_path):
    made = []

    def make(*cams, **kw):
        s = Stack(gst, make_cfg, pipe, tmp_path, cams, **kw)
        made.append(s)
        return s
    yield make
    for s in made:
        try:
            s.stop()
        except Exception:                                    # noqa: BLE001 - already stopped by the test
            pass


def query_service(root, embedder, url="http://127.0.0.1:11434", rcfg=RuntimeConfig(query_timeout_s=120.0)):
    qcfg = QueryConfig(store_root=str(root), ollama_url=url, timeout_s=30)
    llm = OllamaClient(qcfg)
    kn = KnowledgeStore(root / "knowledge.db")
    qu = QueryUnderstanding(qcfg, llm, kn, EntityIndex(root / "godseye.db"), MentionResolver(qcfg, kn, embedder, llm))
    eng = AnswerEngine(AnswerConfig(store_root=str(root), timezone="UTC"), embedder=embedder, llm=llm)
    return QueryService(qu, eng, rcfg)


def real_llm():
    try:
        OllamaClient(QueryConfig()).verify()
    except LLMError as e:
        pytest.skip(f"real LLM unavailable: {e}")


# ---------------------------------------------------------------- the whole chain, and what must propagate
def test_end_to_end_propagation(stack, embedder):
    real_llm()
    t_start = time.time()
    s = stack(cam("lobby")).run()
    t_end = time.time()
    assert s.stop()
    db = s.db()
    tracks = db.execute("SELECT * FROM tracks").fetchall()
    assert len(tracks) >= 3 and all(t["camera_id"] == "lobby" and t["end_ts"] is not None for t in tracks)         # no track left open
    # timestamps are the capture wall-clock times, inside the run
    ts = [r[0] for r in db.execute("SELECT ts FROM observations")]
    assert ts and t_start - 1 <= min(ts) and max(ts) <= t_end + 1
    # track_id is camera-local and unique per (camera, epoch); global ids are cross-camera and only on identified tracks
    assert len({(t["camera_id"], t["stream_epoch"], t["track_id"]) for t in tracks}) == len(tracks)
    gids = {t["global_id"] for t in tracks if t["global_id"]}
    assert gids and all(g.startswith("person-") or g.startswith("vehicle-") for g in gids)
    assert gids <= {r[0] for r in db.execute("SELECT global_id FROM identities")}
    # embeddings: one track_clip vector per track, 512-d, and a gallery vector per identity
    eng = AnswerEngine(AnswerConfig(store_root=str(s.root), timezone="UTC"), embedder=embedder)
    got = eng.vectors.col["track_clip"].get(include=["embeddings", "metadatas"])
    assert len(got["ids"]) == len(tracks) and len(got["embeddings"][0]) == 512
    assert {m["camera_id"] for m in got["metadatas"]} == {"lobby"}
    # clips: every referenced clip exists with the recorded size and frame count
    clips = db.execute("SELECT * FROM clips").fetchall()
    assert clips
    from godseye.storage.clips import probe
    for c in clips:
        assert probe(s.store.cfg, s.root / "clips" / c["path"])["frames"] == c["frames"] and c["camera_id"] == "lobby"
    assert all(r[0] for r in db.execute("SELECT clip_id FROM observations WHERE clip_frame IS NOT NULL"))
    # Track B over the same data: the answer's numbers equal the database's numbers
    qs = query_service(s.root, embedder)
    sid, res = qs.ask(None, "how many people were seen on the lobby camera?")
    if res.kind == "clarify":
        sid, res = qs.answer_clarification(sid, res.clarification.cid, "lobby")
    assert res.kind == "answer" and res.answer.counts["tracks"] == len(tracks)
    assert res.answer.counts["identities"] == len(gids)
    assert s.svc.health()["counters"]["frames_processed.lobby"] > 10
    assert not any(t["end_reason"] == "crash_recovered" for t in tracks)


def test_stream_epoch_change_ends_and_restarts_tracks(stack, pipe):
    s = stack()  # no ingest needed: feed the service real frames directly
    cap = cv2.VideoCapture(VIDEO)
    from tests.test_perception_real import pframe
    seq = 0
    for epoch in (1, 2):
        for _ in range(12):
            ok, bgr = cap.read()
            for _ in range(3):
                cap.read()
            s.svc.process_frame(pframe(bgr, seq=seq, epoch=epoch, cam="lobby"))
            seq += 1
    s.svc._finish_all("shutdown")
    s.store.close()
    db = s.db()
    eps = {r["stream_epoch"] for r in db.execute("SELECT stream_epoch FROM tracks")}
    assert eps == {1, 2}
    assert db.execute("SELECT count(*) FROM tracks WHERE end_reason='stream_restart'").fetchone()[0] >= 1
    assert db.execute("SELECT count(*) FROM tracks WHERE end_ts IS NULL").fetchone()[0] == 0


# ---------------------------------------------------------------- failure isolation
def test_dead_camera_does_not_affect_good_camera_or_queries(stack, embedder):
    bad = cam("dead", url="rtsp://127.0.0.1:9/none", reconnect={"initial_backoff_s": 0.2, "max_backoff_s": 0.5, "connect_timeout_s": 2})
    s = stack(cam("lobby"), bad)
    s.run(until=lambda st: st.svc.m.counters.get("frames_processed.lobby", 0) >= 15, timeout=90)
    assert s.ingest.stats()["dead"].state in (CameraState.CONNECTING, CameraState.RECONNECTING, CameraState.FAILED)
    assert s.svc.m.counters.get("frames_processed.lobby", 0) >= 15 and not s.svc.m.counters.get("frames_processed.dead")
    h = s.svc.health()
    assert h["cameras"]["dead"]["state"] != "streaming" and h["cameras"]["lobby"]["state"] in ("streaming", "ended")
    s.stop()
    assert s.db().execute("SELECT count(*) FROM tracks WHERE camera_id='lobby'").fetchone()[0] >= 1


def test_model_failure_on_one_camera_quarantines_only_that_camera(stack, pipe):
    s = stack(cam("good"), cam("bad"))
    real = pipe.detector.detect

    def flaky(frame):
        if frame.camera_id == "bad":
            raise RuntimeError("injected model failure")
        return real(frame)
    pipe.detector.detect = flaky
    try:
        s.run(until=lambda st: st.svc.m.counters.get("frames_processed.good", 0) >= 12 and st.svc.m.counters.get("quarantined.bad", 0) >= 1, timeout=90)
    finally:
        pipe.detector.detect = real
    c = s.svc.m.counters
    assert c.get("model_error.bad", 0) >= 3 and c.get("quarantined.bad", 0) >= 1 and c.get("frames_processed.good", 0) >= 12
    assert not c.get("frames_processed.bad")
    assert s.svc.health()["cameras"]["bad"]["quarantined"] in (True, False)


def test_gpu_memory_pressure_drops_frames_and_recovers(stack, pipe):
    from tests.test_perception_real import pframe
    cap = cv2.VideoCapture(VIDEO)
    ok, bgr = cap.read()
    s = stack()
    assert s.svc.process_frame(pframe(bgr, seq=0)) is not None
    torch.cuda.empty_cache()
    hog = []
    try:
        while True:                                                                             # take (almost) all remaining VRAM
            hog.append(torch.empty(4 * 2**20, dtype=torch.uint8, device="cuda"))
    except torch.OutOfMemoryError:
        pass
    try:
        res = [s.svc.process_frame(pframe(bgr, seq=i + 1)) for i in range(2)]
    finally:
        del hog
        torch.cuda.empty_cache()
    assert res == [None, None] and s.svc.m.counters.get("gpu_oom.cam", 0) == 2
    assert s.svc.process_frame(pframe(bgr, seq=9)) is not None                              # recovered once memory is back


def test_storage_failure_never_stops_ingestion_and_no_track_stays_open(stack):
    s = stack(cam("lobby"), cam("lobby2"))
    real = s.store.write
    state = {"n": 0, "fail": True}

    def flaky(ef):
        state["n"] += 1
        if state["fail"] and 3 <= state["n"] <= 12:
            raise sqlite3.OperationalError("injected: disk I/O error")
        return real(ef)
    s.store.write = flaky
    s.run()
    c = s.svc.m.counters
    assert c.get("store_errors", 0) >= 2 and c.get("store_breaker_opened", 0) >= 1 and c.get("frames_not_persisted", 0) >= 5
    assert c.get("frames_processed.lobby", 0) > 10                                           # analytics kept running throughout
    assert s.stop()
    db = s.db()
    assert db.execute("SELECT count(*) FROM tracks WHERE end_ts IS NULL").fetchone()[0] == 0   # endings were carried, not lost
    assert db.execute("SELECT count(*) FROM observations").fetchone()[0] > 0
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_chromadb_failure_is_isolated_and_recovers(stack):
    s = stack(cam("lobby"))
    real = s.store.vectors.upsert
    n = {"i": 0}

    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] <= 4:
            raise RuntimeError("injected: chroma unavailable")
        return real(*a, **k)
    s.store.vectors.upsert = flaky
    s.run()
    assert s.svc.m.counters.get("store_errors", 0) >= 1 and s.svc.m.counters.get("frames_processed.lobby", 0) > 10
    assert s.stop()
    assert s.db().execute("SELECT count(*) FROM tracks WHERE end_ts IS NULL").fetchone()[0] == 0


def test_sqlite_lock_contention(stack, tmp_path):
    s = stack(cam("lobby"))
    s.store.conn.execute("PRAGMA busy_timeout=300")
    blocker = sqlite3.connect(s.root / "godseye.db", isolation_level=None, check_same_thread=False)
    blocker.execute("BEGIN EXCLUSIVE")
    holder = threading.Timer(2.5, lambda: blocker.execute("ROLLBACK"))
    holder.start()
    s.run()
    holder.join()
    c = s.svc.m.counters
    assert c.get("store_errors", 0) >= 1 and c.get("frames_processed.lobby", 0) > 10        # contention degraded persistence, not analytics
    assert s.stop()
    assert s.db().execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert s.db().execute("SELECT count(*) FROM observations").fetchone()[0] > 0           # persistence resumed after the lock cleared


def test_query_failure_does_not_stop_ingestion(stack, embedder):
    s = stack(cam("lobby", loop_file=True))
    qs = None
    s.run(until=lambda st: st.svc.m.counters.get("frames_processed.lobby", 0) >= 6, timeout=60)
    qs = query_service(s.root, embedder, url="http://127.0.0.1:9")                          # LLM down
    before = s.svc.m.counters["frames_processed.lobby"]
    kinds = [qs.ask(None, "show people on lobby")[1].kind for _ in range(3)]
    assert kinds == ["error"] * 3
    time.sleep(3)
    assert s.svc.m.counters["frames_processed.lobby"] > before                               # ingestion progressed while queries failed
    s.stop()


# ---------------------------------------------------------------- query-side reliability
def test_query_timeout_and_busy_with_slow_llm(tmp_path, embedder):
    import http.server
    import socketserver

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(4)
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def log_message(self, *a): pass

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from tests.support.trackb import make_track_a
    root = tmp_path / "store"; root.mkdir(); make_track_a(root)
    rc = RuntimeConfig(query_timeout_s=1.0, query_max_inflight=1)
    qs = query_service(root, embedder, url=f"http://127.0.0.1:{srv.server_address[1]}", rcfg=rc)
    t = time.time()
    _, r1 = qs.ask(None, "show cars")
    assert r1.kind == "timeout" and time.time() - t < 2.5
    _, r2 = qs.ask(None, "show cars")                                                       # the slow worker still holds the only slot
    assert r2.kind == "busy"
    srv.shutdown()


def test_corrupt_and_missing_media_degrade_answers(stack, embedder):
    s = stack(cam("lobby"))
    s.run()
    s.stop()
    eng = AnswerEngine(AnswerConfig(store_root=str(s.root), timezone="UTC"), embedder=embedder)
    from godseye.query.schema import Intent, ResolvedQueryContext
    c = ResolvedQueryContext(original_query="q", intent=Intent.find, object_label="person", camera_ids=["lobby"])
    first = eng.run(c)
    assert first.status in (Status.OK, Status.PARTIAL_EVIDENCE)
    paths = {t.evidence.clip_path for t in first.tracklets if t.evidence and t.evidence.clip_path}
    assert paths
    victim = s.root / "clips" / sorted(paths)[0]
    victim.write_bytes(victim.read_bytes()[:-100])
    a = eng.run(c)
    assert a.status == Status.PARTIAL_EVIDENCE and "damaged" in a.text
    victim.unlink()
    assert "missing on disk" in eng.run(c).text
