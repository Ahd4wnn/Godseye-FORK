"""Phase 5: planner, retrieval, path reconstruction, response builder against a REAL Track A store (SQLite + ChromaDB +
ffmpeg clips). Text embeddings are the real OWL-ViT text tower; the LLM appears only as a scripted double in the wording tests."""
import shutil

import pytest

from godseye.answer.config import AnswerConfig, AnswerConfigError, parse
from godseye.answer.engine import AnswerEngine
from godseye.answer.plan import GID, Op, PlanError, Ranking, make_plan
from godseye.answer.response import Status, polish
from godseye.answer.topology import Topology, TopologyError
from godseye.query.embedder import OwlTextEmbedder
from godseye.query.schema import Intent, Region, ResolvedQueryContext
from godseye.query.understand import Session
from tests.support import scenario as S
from tests.support.trackb import ScriptedLLM

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed to build the real Track A store")


@pytest.fixture(scope="module")
def embedder():
    return OwlTextEmbedder("google/owlvit-base-patch32", "cpu")


@pytest.fixture(scope="module")
def world(tmp_path_factory, embedder):
    root = tmp_path_factory.mktemp("world") / "store"
    S.build(root, embedder)
    topo = root.parent / "topology.yaml"
    topo.write_text(S.TOPOLOGY)
    return root, topo


def engine(world, embedder, topology=True, **kw):
    root, topo = world
    cfg = AnswerConfig(store_root=str(root), topology=str(topo) if topology else "", timezone="UTC", **kw)
    return AnswerEngine(cfg, embedder=embedder)


def ctx(intent, **kw):
    return ResolvedQueryContext(original_query="q", intent=intent, object_label=kw.pop("object_label", None), **kw)


# ---------------------------------------------------------------- query -> plan, validation
def test_context_to_plan_mapping(world, embedder):
    e = engine(world, embedder)
    p = e.plan(ctx(Intent.find, object_label="car", camera_ids=["cam2"], time_start=S.T0, time_end=S.T0 + 500, attributes=["red"]))
    assert p.operation == Op.find_tracks and p.object_filter.reid_class == "vehicle" and p.camera_filter == ["cam2"]
    assert p.semantic_query.text == "red car" and p.ranking_strategy == Ranking.semantic and not p.reconstruction_required
    p = e.plan(ctx(Intent.track_path, global_id=S.P1))
    assert p.operation == Op.path and p.reconstruction_required and p.identity_filter.global_id == S.P1
    assert e.plan(ctx(Intent.count, object_label="person")).operation == Op.count_tracks
    assert e.plan(ctx(Intent.when_seen, object_label="person")).operation == Op.when_seen
    assert e.plan(ctx(Intent.locate, global_id=S.P1)).operation == Op.locate_latest
    with pytest.raises(PlanError):
        make_plan(ctx(Intent.unknown), e.cfg)


@pytest.mark.parametrize("bad", [
    dict(intent=Intent.find, object_label="car", camera_ids=["nope"]),
    dict(intent=Intent.find, object_label="car", camera_ids=["../../etc"]),
    dict(intent=Intent.find, object_label="car", time_start=10.0, time_end=5.0),
    dict(intent=Intent.track_path, global_id="person-ffffffffffff"),
    dict(intent=Intent.track_path, global_id="person-1111'; DROP TABLE tracks;--"),
    dict(intent=Intent.find, object_label="unicorn"),
    dict(intent=Intent.find, track_ref=("cam2", 0, 999)),
    dict(intent=Intent.find),
    dict(intent=Intent.find, object_label="car", regions=[Region(camera_id="ghost", x1=0, y1=0, x2=1, y2=1)]),
])
def test_invalid_plans_are_rejected_before_execution(world, embedder, bad):
    e = engine(world, embedder)
    with pytest.raises(PlanError):
        e.plan(ResolvedQueryContext.model_construct(original_query="q", **{"object_label": None, "attributes": [], "camera_ids": [], "time_start": None,
                                                    "time_end": None, "global_id": None, "track_ref": None, "regions": [], "provenance": [], **bad}))
    a = e.run(ResolvedQueryContext.model_construct(original_query="q", **{"object_label": None, "attributes": [], "camera_ids": [], "time_start": None,
                                                   "time_end": None, "global_id": None, "track_ref": None, "regions": [], "provenance": [], **bad}))
    assert a.status == Status.SYSTEM_ERROR and "valid plan" in a.text


def test_plan_invariants(world, embedder):
    e = engine(world, embedder)
    p = e.plan(ctx(Intent.find, object_label="car"))
    for upd in ({"limit": 999}, {"operation": Op.path}, {"reconstruction_required": True}, {"ranking_strategy": Ranking.semantic}):
        with pytest.raises(PlanError):
            from godseye.answer.plan import validate_plan
            validate_plan(p.model_copy(update=upd), e.cfg, e._cameras(), e._identity_exists, e._track_exists)
    with pytest.raises(Exception):
        type(p).model_validate({**p.model_dump(), "sql": "DROP TABLE x"})                 # extra keys rejected


# ---------------------------------------------------------------- retrieval: metadata filters
def refs(ans):
    return {(t.camera_id, t.track_id) for t in ans.tracklets}


def test_metadata_filters(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.find, object_label="car", camera_ids=["cam2"]))
    assert refs(a) == {("cam2", 8), ("cam2", 9)}
    a = e.run(ctx(Intent.find, object_label="person", camera_ids=["lobby_cam"], time_start=S.T0 + 95, time_end=S.T0 + 215))
    assert refs(a) == {("lobby_cam", 4), ("lobby_cam", 6)}
    a = e.run(ctx(Intent.find, global_id=S.P1))
    assert refs(a) == {("lobby_cam", 1), ("cam2", 2), ("north_gate", 3)} and all(t.global_id == S.P1 for t in a.tracklets)
    a = e.run(ctx(Intent.find, track_ref=("cam2", 0, 7)))
    assert refs(a) == {("cam2", 7)}
    assert e.run(ctx(Intent.find, object_label="car", time_start=S.T0 + 305, time_end=S.T0 + 306)).tracklets[0].track_id == 8


def test_spatial_filter_uses_observation_positions(world, embedder):
    e = engine(world, embedder)
    left = Region(camera_id="lobby_cam", x1=0, y1=0, x2=.5, y2=1)
    right = Region(camera_id="lobby_cam", x1=.6, y1=0, x2=1, y2=1)
    assert e.run(ctx(Intent.find, object_label="person", camera_ids=["lobby_cam"], regions=[left])).tracklets
    a = e.run(ctx(Intent.find, object_label="person", camera_ids=["lobby_cam"], regions=[right]))
    assert a.status == Status.NO_RESULT


def test_retrieval_never_scans_tracks(world, embedder):
    e = engine(world, embedder)
    seen = []
    e.conn.set_trace_callback(lambda s: seen.append(s) if s.lstrip().upper().startswith("SELECT") and "FROM tracks t" in s else None)
    for c in (ctx(Intent.find, object_label="car"), ctx(Intent.find, object_label="person", camera_ids=["cam2"], time_start=S.T0, time_end=S.T0 + 99),
              ctx(Intent.find, global_id=S.P1), ctx(Intent.count, object_label="person", time_start=S.T0, time_end=S.T0 + 99),
              ctx(Intent.track_path, global_id=S.P1)):
        e.run(c)
    assert seen
    for sql in seen:
        plan = " | ".join(r[3] for r in e.conn.execute("EXPLAIN QUERY PLAN " + sql.replace("?", "0")) if True) if "?" in sql else ""
        assert "SCAN t" not in plan and "SCAN tracks" not in plan, (sql[:120], plan)


def test_sql_values_are_bound_not_interpolated(world, embedder):
    from godseye.answer.plan import ObjectFilter, QueryPlan
    e = engine(world, embedder)
    plan = QueryPlan(operation=Op.find_tracks, object_filter=ObjectFilter(label="x'; DROP TABLE tracks;--", reid_class="person"))
    r = e.retriever.run(plan)                                                      # bypasses validation on purpose
    assert r.tracklets == [] and e.conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == len(S.TRACKS)


# ---------------------------------------------------------------- semantic retrieval (ChromaDB + CLIP text tower)
@pytest.mark.parametrize("attr,first", [("red", 8), ("blue", 9)])
def test_semantic_ranking_uses_stored_embeddings(world, embedder, attr, first):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.find, object_label="car", attributes=[attr]))
    assert [t.track_id for t in a.tracklets][0] == first
    assert a.tracklets[0].semantic_sim > a.tracklets[1].semantic_sim and "visual similarity" in a.text
    assert any("relative ranking" in w for w in a.warnings)


def test_semantic_unavailable_is_a_system_error_not_a_guess(world):
    root, topo = world
    e = AnswerEngine(AnswerConfig(store_root=str(root), timezone="UTC"), embedder=None)
    a = e.run(ctx(Intent.find, object_label="car", attributes=["red"]))
    assert a.status == Status.SYSTEM_ERROR and not a.tracklets


# ---------------------------------------------------------------- responses: statuses
def test_ok_answer_has_grounded_evidence(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.locate, global_id=S.P1))
    assert a.status == Status.OK and a.tracklets[0].camera_id == "north_gate"
    ev = a.evidence[0]
    assert ev.clip_status == "ok" and (e.root / "clips" / ev.clip_path).is_file() and "not a live position" in a.text
    assert "north_gate" in a.text and S.P1 in a.text and ev.clip_path in a.text


def test_no_result_is_distinct_and_honest(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.find, object_label="bus"))
    assert a.status == Status.NO_RESULT and "does not prove" in a.text and not a.tracklets
    a = e.run(ctx(Intent.find, object_label="car", camera_ids=["north_gate"]))
    assert a.status == Status.NO_RESULT
    a = e.run(ctx(Intent.count, object_label="bus"))
    assert a.status == Status.NO_RESULT


def test_coverage_warning_when_camera_was_not_recording(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.find, object_label="car", camera_ids=["cam2"], time_start=S.T0 + 5000, time_end=S.T0 + 6000))
    assert a.status == Status.NO_RESULT and any("only 0%" in w for w in a.warnings)


def test_low_confidence_and_missing_clip_give_partial_evidence(world, embedder, tmp_path):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.find, object_label="person", camera_ids=["north_gate"], time_start=S.T0 + 590, time_end=S.T0 + 610))
    assert a.status == Status.PARTIAL_EVIDENCE and any("low detection confidence" in w for w in a.warnings)
    # delete a clip file: the answer must say so instead of citing it
    row = e.conn.execute("SELECT path FROM clips c JOIN observations o USING(clip_id) JOIN tracks t USING(track_row) WHERE t.track_id=8 LIMIT 1").fetchone()
    f = e.root / "clips" / row[0]
    hidden = tmp_path / "moved.mp4"
    shutil.move(f, hidden)
    try:
        a = e.run(ctx(Intent.find, object_label="car", camera_ids=["cam2"], time_start=S.T0 + 295, time_end=S.T0 + 325))
        assert a.status == Status.PARTIAL_EVIDENCE and "clip file is missing on disk" in a.text and a.evidence[0].clip_path is None
    finally:
        shutil.move(hidden, f)
    assert e.run(ctx(Intent.find, object_label="car", camera_ids=["cam2"], time_start=S.T0 + 295, time_end=S.T0 + 325)).status == Status.OK


def test_ambiguous_results(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.locate, object_label="person"))
    assert a.status == Status.AMBIGUOUS_RESULT and "different individuals" in a.text
    a = e.run(ctx(Intent.track_path, object_label="person"))
    assert a.status == Status.AMBIGUOUS_RESULT and S.P1 in a.text and "Which one" in a.text and a.path is None
    a = e.run(ctx(Intent.track_path, track_ref=("lobby_cam", 0, 10)))                    # track without a cross-camera identity
    assert a.status == Status.PARTIAL_EVIDENCE and "no cross-camera identity" in a.text and a.path is None


def test_count_reports_tracks_not_people(world, embedder):
    a = engine(world, embedder).run(ctx(Intent.count, object_label="person"))
    assert a.counts == {"tracks": 11, "identities": 4, "unidentified_tracks": 2}
    assert "upper bound" in a.text and a.status == Status.PARTIAL_EVIDENCE


def test_when_seen_merges_adjacent_tracks(world, embedder):
    a = engine(world, embedder).run(ctx(Intent.when_seen, global_id=S.P1))
    assert a.status == Status.OK and a.text.count("- ") == 3


# ---------------------------------------------------------------- path reconstruction
def test_plausible_path_distinguishes_observed_and_inferred(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.track_path, global_id=S.P1))
    p = a.path
    assert [v.camera_id for v in p.visits] == ["lobby_cam", "cam2", "north_gate"] and len(p.episodes) == 1
    assert [t.plausibility for t in p.transitions] == ["plausible", "plausible"] and all(t.kind == "INFERRED" for t in p.transitions)
    assert all(v.kind == "OBSERVED" for v in p.visits) and a.text.count("[OBSERVED]") == 3 and a.text.count("[INFERRED") == 2
    assert p.visits[0].first_pos[1] < p.visits[0].last_pos[1] and a.status == Status.OK       # trajectory from observations
    assert all(v.evidence.clip_status == "ok" for v in p.visits)


def test_impossible_transitions_are_flagged_and_split(world, embedder):
    e = engine(world, embedder)
    a = e.run(ctx(Intent.track_path, global_id=S.P2))
    assert a.path.transitions[0].plausibility == "impossible_overlap" and len(a.path.episodes) == 2
    assert a.status == Status.PARTIAL_EVIDENCE and any("identity merge error" in w for w in a.warnings) and "no link asserted" in a.text
    a = e.run(ctx(Intent.track_path, global_id=S.P3))
    assert a.path.transitions[0].plausibility == "impossible_too_fast" and "fastest route takes 8s" in a.text
    a = e.run(ctx(Intent.track_path, global_id=S.P4))
    assert a.path.transitions[0].plausibility == "long_gap" and len(a.path.episodes) == 2


def test_without_topology_transitions_are_unverified(world, embedder):
    a = engine(world, embedder, topology=False).run(ctx(Intent.track_path, global_id=S.P1))
    assert [t.plausibility for t in a.path.transitions] == ["unverified", "unverified"]
    assert a.status == Status.PARTIAL_EVIDENCE and any("UNVERIFIED" in w for w in a.warnings)


def test_path_respects_time_and_camera_filters(world, embedder):
    a = engine(world, embedder).run(ctx(Intent.track_path, global_id=S.P1, camera_ids=["lobby_cam", "cam2"]))
    assert [v.camera_id for v in a.path.visits] == ["lobby_cam", "cam2"]
    a = engine(world, embedder).run(ctx(Intent.track_path, global_id=S.P1, time_start=S.T0 + 100, time_end=S.T0 + 200))
    assert a.status == Status.NO_RESULT


def test_topology_loader_and_routing():
    t = Topology.load.__func__(Topology, __import__("pathlib").Path("config/topology.example.yaml"))
    assert t.configured and t.min_travel("lobby_cam", "north_gate") == 8.0 and t.min_travel("lobby_cam", "elsewhere") is None
    assert Topology().min_travel("a", "b") is None and not Topology().configured
    import tempfile, pathlib
    for bad in ("links: [{a: x, b: x, min_travel_s: 1, max_travel_s: 2}]", "links: [{a: x, b: y, min_travel_s: 9, max_travel_s: 2}]", "bogus: 1"):
        p = pathlib.Path(tempfile.mktemp(suffix=".yaml")); p.write_text(bad)
        with pytest.raises(TopologyError):
            Topology.load(p)


# ---------------------------------------------------------------- hallucination resistance / errors / security
def test_llm_rewording_cannot_add_facts(world, embedder):
    e = engine(world, embedder)
    base = e.run(ctx(Intent.locate, global_id=S.P1))
    cams = e._cameras()
    good = base.text.replace("The most recent stored sighting", "The latest recorded sighting")
    assert polish(e.run(ctx(Intent.locate, global_id=S.P1)), ScriptedLLM({"text": good}), cams).reworded
    for evil in (base.text + " It was seen on cam2 at 14:55 with 97% certainty.", base.text.replace("north_gate", "cam2"),
                 "He went to the loading dock.", base.text * 5):
        a = e.run(ctx(Intent.locate, global_id=S.P1))
        orig = a.text
        a = polish(a, ScriptedLLM({"text": evil}), cams)
        assert a.text == orig and not a.reworded
    a = e.run(ctx(Intent.track_path, global_id=S.P1))
    orig = a.text
    assert polish(a, ScriptedLLM({"text": a.text.replace("[INFERRED", "[OBSERVED")}), cams).text == orig       # label tampering rejected
    for bad in ({"nope": 1}, RuntimeError):
        from godseye.query.llm import LLMError
        a = e.run(ctx(Intent.locate, global_id=S.P1)); orig = a.text
        assert polish(a, ScriptedLLM(bad if isinstance(bad, dict) else LLMError("x")), cams).text == orig


def test_every_fact_in_an_answer_exists_in_the_store(world, embedder):
    import re
    e = engine(world, embedder)
    known_cams = e._cameras()
    gids = {r[0] for r in e.conn.execute("SELECT global_id FROM identities")}
    for c in (ctx(Intent.find, object_label="person"), ctx(Intent.track_path, global_id=S.P2), ctx(Intent.when_seen, object_label="car")):
        a = e.run(c)
        assert set(re.findall(r"\b(?:lobby_cam|cam2|north_gate|[a-z]+_[a-z]+|cam\d)\b", a.text)) & {"lobby_cam", "cam2", "north_gate"} <= known_cams
        assert set(re.findall(r"(?:person|vehicle)-[0-9a-f]{12}", a.text)) <= gids


def test_system_error_never_looks_like_no_result(world, embedder):
    e = engine(world, embedder)
    e.conn.close()
    a = e.run(ctx(Intent.find, object_label="car"))
    assert a.status == Status.SYSTEM_ERROR and "No result was produced" in a.text and not a.tracklets
    with pytest.raises(FileNotFoundError):
        AnswerEngine(AnswerConfig(store_root="/nonexistent/store"))


def test_engine_is_read_only_and_remembers_entities(world, embedder):
    e = engine(world, embedder)
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        e.conn.execute("DELETE FROM tracks")
    s = Session()
    e.run(ctx(Intent.find, object_label="car"), s)
    e.run(ctx(Intent.track_path, global_id=S.P1), s)
    assert S.P1 in [f.global_id for f in s.focus]


def test_config_validation_and_no_execution_primitives():
    import re
    from pathlib import Path
    with pytest.raises(AnswerConfigError):
        parse({"bogus": 1})
    with pytest.raises(AnswerConfigError):
        parse({"default_limit": 100, "max_limit": 10})
    src = "\n".join(p.read_text() for p in Path("godseye/answer").glob("*.py"))
    assert not re.search(r"(?<![.\w])(eval|exec)\(", src)
    for banned in ("subprocess", "os.system", "shutil", "unlink(", "pickle", "write_text", "write_bytes", "INSERT ", "UPDATE ", "DELETE ", "DROP "):
        assert banned not in src, banned
