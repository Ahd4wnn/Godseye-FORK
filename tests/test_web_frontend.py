"""React console API (frontend/README.md contract) over the real Track B stack: shapes, grounding of hits in stored clips,
the clarification round trip on an open event stream, memory, CSRF/auth, honest 501s, static serving safety."""
import http.client
import json
import threading
import time
from urllib.parse import quote

from tests.support.trackb import ScriptedLLM, pq
from tests.test_web import TOKEN, Site, embedder, site, world  # noqa: F401  (fixtures)


def get(site, path):
    st, h, body = site.req("GET", path)
    return st, (json.loads(body) if body and h.get("Content-Type", "").startswith("application/json") else body)


def sse(site, body, out: list, headers=None):
    """Read /api/query to the end on a thread; events are appended as (event, data)."""
    def run():
        c = http.client.HTTPConnection("127.0.0.1", site.port, timeout=60)
        h = {"Host": f"127.0.0.1:{site.port}", "Authorization": f"Bearer {TOKEN}", "X-Requested-With": "godseye", "Content-Type": "application/json"}
        c.request("POST", "/api/query", json.dumps(body), h)
        r = c.getresponse()
        out.append(("status", r.status))
        buf = ""
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            buf += chunk.decode()
            while "\n\n" in buf:
                blk, buf = buf.split("\n\n", 1)
                ev = dict(line.split(": ", 1) for line in blk.split("\n") if ": " in line and not line.startswith(":"))
                if "event" in ev:
                    out.append((ev["event"], json.loads(ev["data"])))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def wait_for(out, name, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        for e in out:
            if e[0] == name:
                return e[1]
        time.sleep(0.05)
    raise AssertionError(f"no {name!r} event in {out}")


def test_read_endpoints_match_the_contract_and_the_store(site):
    st, h = get(site, "/api/health")
    assert st == 200 and h["ok"] and h["index_size"] > 0 and h["capabilities"] == {"upload": False, "restart": False}
    assert get(site, "/api/site") == (200, {"name": "Godseye"})              # no invented floor plan
    st, cams = get(site, "/api/cameras")
    ids = {r[0] for r in site.app.q("SELECT camera_id FROM cameras")}
    assert st == 200 and cams and {c["id"] for c in cams} <= ids
    for c in cams:
        assert c["video_url"].startswith("/media/clip/") and c["fps"] > 0 and c["duration_s"] > 0 and c["video_s"] > 0
        assert site.req("GET", c["video_url"])[0] == 200 and site.req("GET", c["thumb_url"])[0] == 200
        assert c["start_wall"].endswith("Z")
    assert get(site, "/api/memory") == (200, [])


def test_query_stream_returns_hits_backed_by_stored_clips(site):
    out = []
    sse(site, {"text": "show cars on cam2", "session_id": "browser-1"}, out).join(30)
    names = [e[0] for e in out if e[0] != "status"]
    assert out[0] == ("status", 200) and names[0] == "parsed" and names[-1] == "done" and "answer" in names
    ans = next(d for e, d in out if e == "answer")
    assert ans["text"].startswith("[") and ans["hits"]
    for h in ans["hits"]:
        assert h["camera_id"] == "cam2" and h["clip_url"].startswith("/media/clip/") and 0 <= h["t_best"] <= h["t_end"]
        assert site.req("GET", h["clip_url"])[0] == 200                                   # evidence exists and plays
        assert all(0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 and b["w"] >= 0 and b["x"] + b["w"] <= 1.0001 for b in h["boxes"])
        assert h["wall_time"].endswith("Z")


def test_clarification_continues_on_the_same_stream_and_is_remembered(world, embedder):
    s = Site(world, embedder, ScriptedLLM(pq(intent="count", object={"label": "car"}, camera_mentions=["the roof"]), {"choice": None},
                                          pq(intent="count", object={"label": "car"}, camera_mentions=["the roof"])))
    try:
        out = []
        t = sse(s, {"text": "how many cars on the roof", "session_id": "b2"}, out)
        q = wait_for(out, "clarify")
        assert q["kind"] == "camera" and "cam2" in q["suggested_cameras"] and not any(e[0] == "answer" for e in out)   # stream stays open
        st, _, _ = s.req("POST", "/api/clarify", {"question_id": "nope", "camera_id": "cam2"})
        assert st == 404
        st, _, body = s.req("POST", "/api/clarify", {"question_id": q["question_id"], "camera_id": "cam2"})
        fact = json.loads(body)
        assert st == 200 and fact["camera_id"] == "cam2"
        t.join(30)
        assert "cam2" in wait_for(out, "answer")["text"] and out[-1][0] == "done"
        facts = get(s, "/api/memory")[1]
        assert [f["referent"] for f in facts] == ["the roof"] and facts[0]["camera_id"] == "cam2"
        assert s.req("DELETE", "/api/memory/" + quote(facts[0]["id"], safe=""), headers={"X-Requested-With": "godseye"})[0] == 204
        assert get(s, "/api/memory")[1] == []
        assert s.req("DELETE", "/api/memory/camera:gone", headers={"X-Requested-With": "godseye"})[0] == 404
    finally:
        s.close()


def test_skipping_a_clarification_runs_no_search(world, embedder):
    s = Site(world, embedder, ScriptedLLM(pq(intent="count", object={"label": "car"}, camera_mentions=["the rooftop deck"]), {"choice": None}))
    try:
        out = []
        t = sse(s, {"text": "how many cars on the rooftop deck", "session_id": "b3"}, out)
        q = wait_for(out, "clarify")
        assert s.req("POST", "/api/clarify", {"question_id": q["question_id"], "skip": True})[0] == 204
        t.join(30)
        a = wait_for(out, "answer")
        assert a["hits"] == [] and "did not run the search" in a["text"]
    finally:
        s.close()


def test_writes_need_auth_and_the_csrf_header_and_unsupported_routes_say_so(site):
    assert site.req("POST", "/api/query", {"text": "x"}, headers={"X-Requested-With": ""})[0] == 403
    assert site.req("POST", "/api/query", {"text": "x"}, auth=False)[0] == 401
    assert site.req("DELETE", "/api/memory/camera:x", headers={})[0] == 403
    assert site.req("GET", "/api/health", auth=False)[0] == 401
    for path in ("/api/videos", "/api/videos/links", "/api/admin/restart"):
        st, _, body = site.req("POST", path, {})
        assert st == 501 and "Track A" in json.loads(body)["error"]                   # no silent fake success
    out = []
    sse(site, {"text": "   "}, out).join(10)
    assert any(e[0] == "error" for e in out)


def test_static_console_is_served_safely(site, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>console</title>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("top secret")
    site.app.dist = dist.resolve()
    assert b"console" in site.req("GET", "/")[2] and site.req("GET", "/legacy")[0] == 200
    st, h, body = site.req("GET", "/assets/a.js")
    assert st == 200 and "javascript" in h["Content-Type"] and "Content-Security-Policy" in h
    assert site.req("GET", "/some/client/route")[2].startswith(b"<!doctype html>")         # SPA fallback
    assert site.req("GET", "/assets/missing.js")[0] == 404
    for evil in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt", "/..%2fsecret.txt"):
        st, _, body = site.req("GET", evil)
        assert b"top secret" not in body, evil
    assert site.req("GET", "/assets/a.js", auth=False)[0] == 401
