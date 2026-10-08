"""Loopback Ollama-compatible gateway that forwards /api/chat and /api/tags to the Ollama on a Ray peer node.
Run with the Python 3.11 environment that matches the cluster:   .venv-ray/bin/python tools/ray_llm_gateway.py
Exposes ONLY: GET /api/tags, POST /api/chat, GET /healthz. Binds 127.0.0.1. Request bodies are never logged."""
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ray
import yaml
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import godseye_remote  # noqa: E402
from godseye_remote.llm_proxy import LLMProxy  # noqa: E402

log = logging.getLogger("ray-llm-gateway")
MAX_BODY = 64_000


class Backend:
    def __init__(self, cfg):
        self.cfg, self.lock, self.actor = cfg, threading.Lock(), None
        ray.init(address=cfg["ray_address"], logging_level="ERROR", namespace="godseye", runtime_env={"py_modules": [godseye_remote]})
        self._make()

    def _make(self):
        nodes = [n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == self.cfg["llm_node_ip"]]
        if not nodes:
            raise RuntimeError(f"Ray node {self.cfg['llm_node_ip']} is not alive in the cluster")
        self.actor = ray.remote(num_cpus=0.1, max_restarts=0)(LLMProxy).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0]["NodeID"], soft=False)).remote(self.cfg["allowed_models"])
        ray.get(self.actor.whoami.remote(), timeout=60)

    def call(self, method, *args):
        for attempt in (1, 2):
            try:
                return ray.get(getattr(self.actor, method).remote(*args), timeout=self.cfg["call_timeout_s"] + 10)
            except (ray.exceptions.RayActorError, ray.exceptions.ActorDiedError):
                if attempt == 2:
                    raise
                log.warning("remote LLM proxy died; recreating")
                with self.lock:
                    self._make()


def handler(backend, timeout):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            try:
                if self.path == "/healthz":
                    return self._send(200, {"ok": True, **backend.call("whoami")})
                if self.path == "/api/tags":
                    return self._send(200, backend.call("tags"))
                self._send(404, {"error": "not found"})
            except Exception as e:                                  # noqa: BLE001
                self._send(502, {"error": f"remote LLM unavailable: {type(e).__name__}"})

        def do_POST(self):
            if self.path != "/api/chat":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= MAX_BODY:
                    return self._send(413, {"error": "bad body size"})
                body = json.loads(self.rfile.read(n))
                self._send(200, backend.call("chat", body, timeout))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {str(e)[:100]}"})
            except Exception as e:                                  # noqa: BLE001
                self._send(502, {"error": f"remote LLM unavailable: {type(e).__name__}: {str(e)[:150]}"})
    return H


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load((ROOT / (argv[0] if argv else "config/cluster.yaml")).read_text())
    port = int(argv[1]) if argv and len(argv) > 1 else cfg["gateway_port"]
    if cfg["gateway_host"] not in ("127.0.0.1", "localhost"):
        raise SystemExit("gateway_host must be loopback")
    backend = Backend(cfg)
    srv = ThreadingHTTPServer((cfg["gateway_host"], port), handler(backend, cfg["call_timeout_s"]))
    log.info("ray LLM gateway on http://%s:%d -> node %s (models %s)", cfg["gateway_host"], port, cfg["llm_node_ip"], cfg["allowed_models"])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
