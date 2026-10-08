"""Phase 5 end to end with everything real: video -> NanoOWL/ByteTrack/FastReID -> EvidenceStore -> real local LLM
understanding -> planner -> hybrid retrieval (ChromaDB + SQLite, OWL-ViT text tower) -> path -> grounded answer."""
import re
import shutil

import numpy as np
import pytest

from tests.test_perception_real import VIDEO, cv2, pframe, torch  # noqa: F401

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg"),
              pytest.mark.skipif(not VIDEO, reason="no real test video")]

from godseye.answer.config import AnswerConfig  # noqa: E402
from godseye.answer.engine import AnswerEngine  # noqa: E402
from godseye.answer.response import Status  # noqa: E402
from godseye.perception.config import load_config  # noqa: E402
from godseye.perception.pipeline import PerceptionPipeline  # noqa: E402
from godseye.perception.types import EnrichedFrame  # noqa: E402
from godseye.query.config import QueryConfig  # noqa: E402
from godseye.query.embedder import OwlTextEmbedder  # noqa: E402
from godseye.query.entities import EntityIndex  # noqa: E402
from godseye.query.knowledge import KnowledgeStore  # noqa: E402
from godseye.query.llm import LLMError, OllamaClient  # noqa: E402
from godseye.query.resolver import MentionResolver  # noqa: E402
from godseye.query.understand import NeedsClarification, QueryUnderstanding, Resolved, Session  # noqa: E402
from godseye.storage.config import StorageConfig  # noqa: E402
from godseye.storage.store import EvidenceStore  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("real") / "store"
    store = EvidenceStore(StorageConfig(root=str(root), min_free_bytes=0, track_embed_every=5))
    pipe = PerceptionPipeline(load_config("config/perception.yaml"), gallery=store.gallery)
    cap = cv2.VideoCapture(VIDEO)
    step, k, seq = max(1, round(cap.get(cv2.CAP_PROP_FPS) / 5)), 0, 0
    while seq < 45:
        ok, bgr = cap.read()
        if not ok:
            break
        if k % step == 0:
            store.write(pipe.process(pframe(bgr, seq=seq, cam="lobby")))
            seq += 1
        k += 1
    store.write(EnrichedFrame(pframe(bgr, seq=seq, cam="lobby"), (), (), pipe.shutdown()))
    store.close()
    return root


@pytest.fixture(scope="module")
def embedder():
    return OwlTextEmbedder("google/owlvit-base-patch32", "cpu")


@pytest.fixture
def engine(world, embedder):
    return AnswerEngine(AnswerConfig(store_root=str(world), timezone="UTC"), embedder=embedder)


def ctx(**kw):
    from godseye.query.schema import Intent, ResolvedQueryContext
    intent = kw.pop("intent")
    return ResolvedQueryContext(original_query="q", intent=Intent(intent), object_label=kw.pop("object_label", None), **kw)


def test_real_embeddings_align_with_text_queries(world, engine, embedder):
    got = engine.vectors.col["track_clip"].get(include=["embeddings"])
    E = np.asarray(got["embeddings"])
    assert len(E) >= 3 and E.shape[1] == 512
    t = embedder.embed(["a person", "a bus", "a photo of a cat"])
    sims = E @ t.T
    assert sims[:, 0].mean() > sims[:, 1].mean() and sims[:, 0].mean() > sims[:, 2].mean()      # real people embed closer to "person"


def test_real_count_find_locate_and_path(world, engine):
    c = engine.run(ctx(intent="count", object_label="person", camera_ids=["lobby"]))
    assert c.counts["tracks"] >= 3 and c.status in (Status.OK, Status.PARTIAL_EVIDENCE)
    f = engine.run(ctx(intent="find", object_label="person", camera_ids=["lobby"], attributes=["standing"]))
    assert f.tracklets and f.tracklets[0].semantic_sim is not None
    ordered = [t.semantic_sim for t in f.tracklets]
    assert ordered == sorted(ordered, reverse=True)
    sc = [t for t in f.tracklets if t.evidence and t.evidence.clip_status == "ok"]
    assert sc and all((engine.root / "clips" / t.evidence.clip_path).is_file() for t in sc)
    gid = next(t.global_id for t in f.tracklets if t.global_id)
    p = engine.run(ctx(intent="track_path", global_id=gid))
    assert p.path and p.path.visits and all(v.kind == "OBSERVED" for v in p.path.visits)
    assert p.status in (Status.OK, Status.PARTIAL_EVIDENCE) and "[OBSERVED]" in p.text
    assert engine.run(ctx(intent="find", object_label="bus")).status == Status.NO_RESULT


def test_real_clip_evidence_frame_is_decodable(world, engine):
    f = engine.run(ctx(intent="find", object_label="person", camera_ids=["lobby"]))
    ev = next(t.evidence for t in f.tracklets if t.evidence.clip_status == "ok")
    cap = cv2.VideoCapture(str(engine.root / "clips" / ev.clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ev.clip_frame)
    ok, img = cap.read()
    assert ok and img.shape[0] > 100


def test_real_question_to_grounded_answer(world, engine, embedder):
    try:
        llm = OllamaClient(QueryConfig())
        llm.verify()
    except LLMError as e:
        pytest.skip(f"real LLM unavailable: {e}")
    cfg = QueryConfig(store_root=str(world))
    kn = KnowledgeStore(world / "knowledge.db")
    qu = QueryUnderstanding(cfg, llm, kn, EntityIndex(world / "godseye.db"), MentionResolver(cfg, kn, embedder, llm))
    s = Session()
    out = qu.handle(s, "how many people were seen on the lobby camera?")
    if isinstance(out, NeedsClarification):
        out = qu.respond(s, out.clarification.cid, "lobby")
    assert isinstance(out, Resolved), out
    a = engine.run(out.context, s)
    assert a.counts and a.counts["tracks"] >= 3 and a.status in (Status.OK, Status.PARTIAL_EVIDENCE)
    assert all(c in ("lobby",) for c in re.findall(r"\blobby\b", a.text))
    out = qu.handle(s, "show me the person on lobby")
    if isinstance(out, NeedsClarification):
        out = qu.respond(s, out.clarification.cid, "lobby")
    a = engine.run(out.context, s)
    assert a.tracklets and a.status in (Status.OK, Status.PARTIAL_EVIDENCE) and s.focus


def test_real_llm_wording_never_adds_facts(world, engine):
    try:
        llm = OllamaClient(QueryConfig())
        llm.verify()
    except LLMError as e:
        pytest.skip(f"real LLM unavailable: {e}")
    from godseye.answer.response import polish
    a = engine.run(ctx(intent="find", object_label="person", camera_ids=["lobby"]))
    orig = a.text
    b = polish(a, llm, {"lobby"})
    tok = lambda t: set(re.findall(r"[A-Za-z0-9_.:/\-]*\d[A-Za-z0-9_.:/\-]*", t))  # noqa: E731
    assert b.text == orig or tok(b.text) == tok(orig)                         # reworded text carries exactly the same data
