"""Ray actor that forwards chat requests to the Ollama server running on ITS OWN node (127.0.0.1). It exposes exactly two
operations, validates the request shape and model name, and never returns anything but Ollama's JSON. It runs no
commands, opens no other URLs and touches no files."""
import json
import urllib.error
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
MAX_BODY = 64_000
ALLOWED_KEYS = {"model", "messages", "stream", "think", "format", "options"}
ALLOWED_OPTIONS = {"temperature", "num_ctx", "num_predict", "seed"}


class LLMProxy:
    def __init__(self, allowed_models):
        self.allowed = set(allowed_models)

    def _call(self, path, body=None, timeout=120.0):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read(5_000_000))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama on the remote node is not reachable: {e}") from e

    def whoami(self):
        import platform
        return {"python": platform.python_version(), "host": platform.node()}

    def tags(self):
        return self._call("/api/tags", timeout=10)

    def chat(self, body, timeout=120.0):
        if not isinstance(body, dict) or set(body) - ALLOWED_KEYS:
            raise ValueError("unexpected request fields")
        if body.get("model") not in self.allowed:
            raise ValueError(f"model {body.get('model')!r} is not allowed on this proxy")
        if len(json.dumps(body)) > MAX_BODY:
            raise ValueError("request too large")
        opts = body.get("options") or {}
        if set(opts) - ALLOWED_OPTIONS:
            raise ValueError("unexpected option")
        body = {**body, "stream": False}
        return self._call("/api/chat", body, timeout=min(float(timeout), 300.0))
