"""Web chat: authentication, request validation, media access control (traversal / symlink / poisoned DB path / corrupt clip),
Range support and XSS surface. Real server on a loopback port, real Track A store and clips; the LLM is a scripted double."""
import http.client
import json
import shutil
import threading
from pathlib import Path

import pytest

from godseye.answer.config import AnswerConfig
from godseye.answer.engine import AnswerEngine
from godseye.query.config import QueryConfig
from godseye.query.embedder import OwlTextEmbedder
from godseye.query.entities import EntityIndex
from godseye.query.knowledge import KnowledgeStore
from godseye.query.resolver import MentionResolver
from godseye.query.understand import QueryUnderstanding
from godseye.runtime.config import RuntimeConfig
from godseye.runtime.trackb import QueryService
from godseye.web.server import WebApp, load_or_create_token, make_server
from tests.support import scenario as S
from tests.support.trackb import ScriptedLLM, pq

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg needed to build the real Track A store")
TOKEN = "t" * 43


@pytest.fixture(scope="module")
def embedder():
    return OwlTextEmbedder("google/owlvit-base-patch32", "cpu")


@pytest.fixture(scope="module")
def world(tmp_path_factory, embedder):
    root = tmp_path_factory.mktemp("web") / "store"
    S.build(root, embedder)
    return root


class Site:
    def __init__(self, root, embedder, llm=None, chat_per_min=1000):
        cfg = QueryConfig(store_root=str(root), object_aliases=(("guy", "person"),))
        kn = KnowledgeStore(root / "knowledge.db")
        qu = QueryUnderstanding(cfg, llm or ScriptedLLM(), kn, EntityIndex(root / "godseye.db"), MentionResolver(cfg, kn, embedder, llm))
        eng = AnswerEngine(AnswerConfig(store_root=str(root), timezone="UTC"), embedder=embedder)
        self.qs = QueryService(qu, eng, RuntimeConfig(query_timeout_s=60))
        self.app = WebApp(self.qs, root, TOKEN, chat_per_min)
        self.srv = make_server(self.app, 0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def req(self, method, path, body=None, headers=None, host=None, auth=True, raw=None):
        h = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
        if auth:
            h["Authorization"] = f"Bearer {TOKEN}"
        if method == "POST":
            h.setdefault("X-Requested-With", "godseye"); h.setdefault("Content-Type", "application/json")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in h.items():
            c.putheader(k, v)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None:
            c.putheader("Content-Length", str(len(data)))
        c.endheaders(data)
        r = c.getresponse()
        out = r.read()
        c.close()
        return r.status, dict(r.getheaders()), out

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


@pytest.fixture
def site(world, embedder):
    s = Site(world, embedder, ScriptedLLM(*[pq(intent="find", object={"label": "car"}, camera_mentions=["cam2"])] * 20))
    yield s
    s.close()


def test_authentication_and_login_cookie(site):
    for path in ("/", "/api/cameras", "/media/clip/1", "/api/clip/1", "/api/frame/1/0.jpg"):
        assert site.req("GET", path, auth=False)[0] == 401, path
    assert site.req("POST", "/api/chat", {"text": "x"}, auth=False)[0] == 401
    assert site.req("GET", "/", headers={"Authorization": "Bearer wrong"}, auth=False)[0] == 401
    st, h, _ = site.req("GET", f"/?token={TOKEN}", auth=False)
    assert st == 303 and "HttpOnly" in h["Set-Cookie"] and "SameSite=Strict" in h["Set-Cookie"]
    assert site.req("GET", f"/?token={'x' * 43}", auth=False)[0] == 401
    st, _, body = site.req("GET", "/", headers={"Cookie": f"gs={TOKEN}"}, auth=False)
    assert st == 200 and b"<title>Godseye</title>" in body
    assert site.req("GET", "/app.js", auth=False)[0] == 200 and site.req("GET", "/healthz", auth=False)[0] == 200


def test_dns_rebinding_and_csrf_defences(site):
    assert site.req("GET", "/api/cameras", host="evil.example.com")[0] == 403
    assert site.req("GET", "/api/cameras", host=f"127.0.0.1:{site.port + 1}")[0] == 403
    assert site.req("POST", "/api/chat", {"text": "x"}, headers={"X-Requested-With": ""})[0] == 403
    assert site.req("POST", "/api/chat", {"text": "x"}, headers={"Origin": "http://evil.example.com"})[0] == 403
    st, _, _ = site.req("POST", "/api/chat", {"text": "show cars"}, headers={"Origin": f"http://127.0.0.1:{site.port}"})
    assert st == 200


def test_request_validation_limits_and_rate(world, embedder, site):
    assert site.req("POST", "/api/chat", raw=b"x" * 9000)[0] == 413
    assert site.req("POST", "/api/chat", raw=b"{nope")[0] == 400
    assert site.req("POST", "/api/chat", raw=b"[1,2]")[0] == 400
    assert site.req("POST", "/api/chat", {"nothing": 1})[0] == 400
    assert site.req("POST", "/api/chat", {"cid": "x", "session": "0" * 32, "response": {"a": [1]}})[0] == 400
    assert site.req("POST", "/api/other", {"text": "x"})[0] == 404
    limited = Site(world, embedder, ScriptedLLM(*[pq(intent="unknown")] * 10), chat_per_min=3)
    try:
        codes = [limited.req("POST", "/api/chat", {"text": "weather"})[0] for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
    finally:
        limited.close()


def test_chat_returns_grounded_json_with_playable_evidence(site):
    st, h, body = site.req("POST", "/api/chat", {"text": "show cars on cam2"})
    out = json.loads(body)
    assert st == 200 and out["kind"] == "answer" and out["status"] in ("OK", "PARTIAL_EVIDENCE") and len(out["session"]) == 32
    ev = [e for e in out["evidence"] if e["clip_id"]]
    assert ev and all(e["camera_id"] == "cam2" and e["clip_status"] == "ok" for e in ev)
    st, h, vid = site.req("GET", f"/media/clip/{ev[0]['clip_id']}")
    assert st == 200 and h["Content-Type"] == "video/mp4" and h["Accept-Ranges"] == "bytes" and vid[4:8] == b"ftyp"
    assert h["X-Content-Type-Options"] == "nosniff" and "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert "python" not in h.get("Server", "").lower() and "BaseHTTP" not in h.get("Server", "")


def test_clarification_roundtrip_over_http(world, embedder):
    s = Site(world, embedder, ScriptedLLM(pq(intent="count", object={"label": "car"}, camera_mentions=["the roof"]), {"choice": None},
                                          pq(intent="count", object={"label": "car"}, camera_mentions=["the roof"])))
    try:
        out = json.loads(s.req("POST", "/api/chat", {"text": "how many cars on the roof"})[2])
        assert out["kind"] == "clarify" and out["clarification"]["kind"] == "camera_grid"
        c = out["clarification"]
        bad = json.loads(s.req("POST", "/api/chat", {"session": out["session"], "cid": c["cid"], "response": "rm -rf /"})[2])
        assert bad["kind"] == "clarify" and "not one of" in bad["clarification"]["prompt"]
        done = json.loads(s.req("POST", "/api/chat", {"session": out["session"], "cid": c["cid"], "response": "cam2"})[2])
        assert done["kind"] == "answer" and "cam2" in done["text"]
    finally:
        s.close()


def test_media_range_requests(site, world):
    row = site.app.q("SELECT clip_id, size_bytes FROM clips WHERE status='complete' ORDER BY clip_id")[0]
    cid, size = row["clip_id"], row["size_bytes"]
    full = site.req("GET", f"/media/clip/{cid}")[2]
    assert len(full) == size
    st, h, part = site.req("GET", f"/media/clip/{cid}", headers={"Range": "bytes=10-99"})
    assert st == 206 and part == full[10:100] and h["Content-Range"] == f"bytes 10-99/{size}"
    st, h, tail = site.req("GET", f"/media/clip/{cid}", headers={"Range": "bytes=-50"})
    assert st == 206 and tail == full[-50:]
    assert site.req("GET", f"/media/clip/{cid}", headers={"Range": f"bytes={size + 5}-"})[0] == 416
    assert site.req("GET", f"/media/clip/{cid}", headers={"Range": "bytes=abc"})[0] == 416
    assert site.req("GET", f"/media/clip/{cid}", headers={"Range": "bytes=50-10"})[0] == 416


def test_media_cannot_escape_the_clips_directory(world, embedder, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(world, root, ignore=shutil.ignore_patterns("chroma"))
    import sqlite3
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"TOP SECRET")
    db = sqlite3.connect(root / "godseye.db")
    cam = db.execute("SELECT camera_id FROM cameras LIMIT 1").fetchone()[0]
    base = dict(camera_id=cam, stream_epoch=0, start_ts=0, end_ts=1, fps=5, frames=1, width=2, height=2, frame_ts="[0]", sha256="x", created_ts=0)
    rows = [("../../secret.mp4", 10), ("2023-11-14/lobby_cam/../../../../secret.mp4", 10), ("link/secret.mp4", 10)]
    (root / "clips" / "link").symlink_to(tmp_path)                                # a symlink inside clips/ pointing outside
    ids = []
    for path, size in rows:
        ids.append(db.execute("INSERT INTO clips(camera_id,stream_epoch,start_ts,end_ts,path,fps,frames,width,height,frame_ts,size_bytes,sha256,created_ts) "
                              "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (cam, 0, 0, 1, path, 5, 1, 2, 2, "[0]", size, "x", 0)).lastrowid)
    db.commit(); db.close()
    s = Site(root, embedder)
    try:
        for cid in ids:
            st, _, body = s.req("GET", f"/media/clip/{cid}")
            assert st == 404 and b"TOP SECRET" not in body, cid
        for evil in ("/media/clip/../../../etc/passwd", "/media/clip/%2e%2e%2f%2e%2e%2fetc%2fpasswd", "/media/clip/1/../../x", "/media/clip/abc",
                     "/media/clip/99999999999", "/api/frame/1/../../x.jpg", "/media/%2e%2e/godseye.db", "/godseye.db", "/../../etc/passwd"):
            st, _, body = s.req("GET", evil)
            assert st in (400, 404) and b"root:" not in body and b"SQLite" not in body, evil
    finally:
        s.close()


def test_damaged_missing_and_marked_clips_are_refused(world, embedder, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(world, root, ignore=shutil.ignore_patterns("chroma"))
    import sqlite3
    db = sqlite3.connect(root / "godseye.db")
    a, b, c = [r[0] for r in db.execute("SELECT clip_id FROM clips ORDER BY clip_id LIMIT 3")]
    pa = root / "clips" / db.execute("SELECT path FROM clips WHERE clip_id=?", (a,)).fetchone()[0]
    pb = root / "clips" / db.execute("SELECT path FROM clips WHERE clip_id=?", (b,)).fetchone()[0]
    pa.write_bytes(pa.read_bytes()[:-10])                                         # damaged (size differs)
    pb.unlink()                                                                   # missing
    db.execute("UPDATE clips SET status='corrupt' WHERE clip_id=?", (c,)); db.commit(); db.close()
    s = Site(root, embedder)
    try:
        assert s.req("GET", f"/media/clip/{a}")[0] == 410 and s.req("GET", f"/media/clip/{b}")[0] == 410 and s.req("GET", f"/media/clip/{c}")[0] == 404
        assert s.req("GET", f"/api/frame/{a}/0.jpg")[0] == 410
    finally:
        s.close()


def test_frame_endpoint_returns_jpeg(site):
    row = site.app.q("SELECT clip_id, frames FROM clips WHERE status='complete' LIMIT 1")[0]
    st, h, body = site.req("GET", f"/api/frame/{row['clip_id']}/1.jpg")
    assert st == 200 and h["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8"
    assert site.req("GET", f"/api/frame/{row['clip_id']}/{row['frames']}.jpg")[0] == 404
    cams = json.loads(site.req("GET", "/api/cameras")[2])["cameras"]
    assert {c["camera_id"] for c in cams} == {"lobby_cam", "cam2", "north_gate"}
    meta = json.loads(site.req("GET", f"/api/clip/{cams[0]['clip_id']}")[2])
    assert len(meta["frame_ts"]) == meta["frames"]


def test_internal_errors_do_not_leak_details(site, monkeypatch):
    monkeypatch.setattr(site.app, "chat", lambda b: (_ for _ in ()).throw(RuntimeError("secret /home/albhy/path")))
    st, _, body = site.req("POST", "/api/chat", {"text": "x"})
    assert st == 500 and b"secret" not in body and b"Traceback" not in body


def test_ui_has_no_html_injection_sinks_and_server_is_loopback_only(world, embedder):
    from godseye.web.assets import INDEX, JS
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "<script>"):
        assert sink not in JS, sink
    assert "<script src=" in INDEX and "onclick=" not in INDEX
    with pytest.raises(ValueError):
        make_server(WebApp.__new__(WebApp), 0, "0.0.0.0")
    p = Path(__file__).parent / "tok"
    try:
        t = load_or_create_token(p)
        assert len(t) >= 40 and oct(p.stat().st_mode & 0o777) == "0o600" and load_or_create_token(p) == t
    finally:
        p.unlink(missing_ok=True)
