"""Ray-cluster placement of the LLM: remote proxy validation, loopback gateway, local fallback. Tests that need the cluster are
skipped (loudly) when it is not reachable."""
import http.server
import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from godseye.query.config import QueryConfig, QueryConfigError, parse
from godseye.query.llm import LLMUnavailable, OllamaClient
from godseye_remote.llm_proxy import LLMProxy

ROOT = Path(__file__).resolve().parents[1]
RAY_PY = ROOT / ".venv-ray/bin/python"
SCHEMA = {"type": "object", "properties": {"intent": {"type": "string", "enum": ["find", "count"]}}, "required": ["intent"]}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeOllama:
    """Isolated test double of an Ollama server (used only for the fallback unit tests)."""
    def __init__(self, reply):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, o):
                b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self): self._send({"models": [{"name": "qwen3:8b"}]})
            def do_POST(self):
                outer.calls += 1
                self.rfile.read(int(self.headers["Content-Length"])); self._send({"message": {"content": json.dumps(reply)}})
        self.calls = 0
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


# ---------------------------------------------------------------- remote proxy validation (pure python)
def test_remote_proxy_rejects_anything_but_a_plain_chat_request():
    p = LLMProxy(["qwen3:8b"])
    base = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}]}
    for bad in ({**base, "model": "llama3:70b"}, {**base, "tools": [{"type": "function"}]}, {**base, "keep_alive": -1},
                {**base, "options": {"num_gpu": 99}}, {**base, "messages": [{"role": "user", "content": "x" * 70000}]}, "not a dict", None):
        with pytest.raises(ValueError):
            p.chat(bad)


def test_remote_proxy_module_is_stdlib_only():
    src = (ROOT / "godseye_remote/llm_proxy.py").read_text()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines() if l.startswith(("import ", "from "))}
    assert imports <= {"json", "urllib", "platform"}
    import re
    for banned in ("subprocess", "os.system", "socket"):
        assert banned not in src
    assert not re.search(r"(?<![\w.])(open|eval|exec)\(", src)


# ---------------------------------------------------------------- config + fallback
def test_config_requires_loopback_llm_urls():
    for bad in ({"ollama_url": "http://10.90.158.67:11434"}, {"llm_fallback_url": "https://example.com"}):
        with pytest.raises(QueryConfigError):
            parse(bad)
    assert parse({"ollama_url": "http://127.0.0.1:11435", "llm_fallback_url": "http://127.0.0.1:11434"}).llm_fallback_url


def test_fallback_is_used_loudly_and_only_when_primary_is_unavailable(caplog):
    good = FakeOllama({"intent": "count"})
    dead = free_port()
    try:
        c = OllamaClient(QueryConfig(ollama_url=f"http://127.0.0.1:{dead}", llm_fallback_url=f"http://127.0.0.1:{good.port}"))
        assert c.verify()["model"] == "qwen3:8b"
        assert c.chat_json("s", "u", SCHEMA) == {"intent": "count"} and c.fallbacks_used == 1 and good.calls == 1
        # primary healthy -> fallback untouched
        other = FakeOllama({"intent": "find"})
        c2 = OllamaClient(QueryConfig(ollama_url=f"http://127.0.0.1:{other.port}", llm_fallback_url=f"http://127.0.0.1:{good.port}"))
        assert c2.chat_json("s", "u", SCHEMA) == {"intent": "find"} and c2.fallbacks_used == 0 and good.calls == 1
        other.close()
        # no fallback configured -> a clear unavailable error, never a silent substitute
        with pytest.raises(LLMUnavailable):
            OllamaClient(QueryConfig(ollama_url=f"http://127.0.0.1:{dead}")).chat_json("s", "u", SCHEMA)
        # both down
        with pytest.raises(LLMUnavailable):
            OllamaClient(QueryConfig(ollama_url=f"http://127.0.0.1:{dead}", llm_fallback_url=f"http://127.0.0.1:{free_port()}")).chat_json("s", "u", SCHEMA)
    finally:
        good.close()
    assert any("using the fallback LLM" in r.message for r in caplog.records)


# ---------------------------------------------------------------- the real cluster
@pytest.fixture(scope="module")
def gateway():
    if not RAY_PY.exists():
        pytest.skip(".venv-ray (Python 3.11 + ray 2.40) not present")
    port = free_port()
    p = subprocess.Popen([str(RAY_PY), str(ROOT / "tools/ray_llm_gateway.py"), "config/cluster.yaml", str(port)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    t = time.time()
    while time.time() - t < 90:
        if p.poll() is not None:
            pytest.skip("Ray cluster unreachable or LLM node not alive: " + p.stdout.read()[-300:])
        try:
            if httpx.get(url + "/healthz", timeout=3).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        p.kill()
        pytest.skip("gateway did not become healthy")
    yield url, p
    p.terminate()
    p.wait(10)


def test_gateway_serves_the_remote_llm(gateway):
    url, _ = gateway
    assert httpx.get(url + "/healthz").json()["python"].startswith("3.11")
    c = OllamaClient(QueryConfig(ollama_url=url))
    assert c.verify()["model"] == "qwen3:8b"
    t = time.time()
    out = c.chat_json("Extract JSON.", "how many cars passed", SCHEMA)
    assert out == {"intent": "count"}
    print(f"remote LLM call: {time.time() - t:.2f}s")


def test_gateway_surface_is_minimal_and_validated(gateway):
    url, _ = gateway
    assert httpx.get(url + "/api/pull").status_code == 404
    assert httpx.post(url + "/api/generate", json={}).status_code == 404
    assert httpx.post(url + "/api/chat", content=b"x" * 70000).status_code == 413
    assert httpx.post(url + "/api/chat", json={"model": "qwen3:8b", "tools": []}).status_code == 400
    assert httpx.post(url + "/api/chat", json={"model": "llama3:70b", "messages": []}).status_code == 400
    assert httpx.post(url + "/api/chat", content=b"{not json").status_code == 400


def test_real_nlu_through_the_remote_llm(gateway):
    from datetime import datetime, timezone

    from godseye.query.nlu import Nlu
    url, _ = gateway
    cfg = QueryConfig(ollama_url=url)
    pq = Nlu(cfg, OllamaClient(cfg)).parse("how many cars were on cam2 in the last 30 minutes?", datetime(2026, 10, 8, 18, tzinfo=timezone.utc))
    assert pq.intent.value == "count" and pq.time.kind == "last" and pq.time.amount == 30


def test_gateway_death_falls_back_to_local_ollama_loudly(gateway):
    url, proc = gateway
    try:
        local_ok = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        local_ok = False
    if not local_ok:
        pytest.skip("local Ollama not running for the fallback leg")
    c = OllamaClient(QueryConfig(ollama_url=url, llm_fallback_url="http://127.0.0.1:11434"))
    assert c.chat_json("Extract JSON.", "how many cars passed", SCHEMA)["intent"] == "count" and c.fallbacks_used == 0
    proc.terminate(); proc.wait(10)                                                        # the peer / gateway disappears
    assert c.chat_json("Extract JSON.", "how many cars passed", SCHEMA)["intent"] == "count" and c.fallbacks_used == 1
