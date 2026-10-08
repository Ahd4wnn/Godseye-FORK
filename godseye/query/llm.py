"""Local LLM client (Ollama over loopback HTTP). The LLM is a text->JSON function: its output is returned as data and is
validated by the caller. There is no tool-calling, no code execution and no filesystem/network access derived from it."""
from __future__ import annotations

import json

import httpx

from .config import QueryConfig


class LLMError(RuntimeError):
    pass


_DROP = {"title", "default", "pattern", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "description"}


def grammar_schema(schema: dict) -> dict:
    """Pydantic JSON schema -> the subset Ollama's grammar compiler accepts: $refs inlined, validation-only keywords
    dropped (pydantic re-validates everything strictly afterwards, so nothing is lost)."""
    defs = schema.get("$defs", {})

    def walk(n):
        if isinstance(n, list):
            return [walk(x) for x in n]
        if not isinstance(n, dict):
            return n
        if "$ref" in n:
            return walk(defs[n["$ref"].split("/")[-1]])
        return {k: walk(v) for k, v in n.items() if k not in _DROP and k != "$defs"}

    return walk(schema)


class OllamaClient:
    def __init__(self, cfg: QueryConfig):
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.ollama_url, timeout=cfg.timeout_s)

    def verify(self) -> dict:
        """Fail loudly if Ollama is down or the configured model is not installed (never silently substitute)."""
        try:
            tags = self.http.get("/api/tags").json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"Ollama not reachable at {self.cfg.ollama_url}: {e}") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.cfg.model not in names and f"{self.cfg.model}:latest" not in names:
            raise LLMError(f"model {self.cfg.model!r} is not installed in Ollama (have: {sorted(names)})")
        return {"model": self.cfg.model, "installed": sorted(names)}

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        body = {"model": self.cfg.model, "stream": False, "think": False, "format": grammar_schema(schema),
                "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = self.http.post("/api/chat", json=body)
            r.raise_for_status()
            text = r.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"LLM request failed: {e}") from e
        if len(text) > 20000:
            raise LLMError("LLM output unreasonably long")
        try:
            out = json.loads(text)
        except ValueError as e:
            raise LLMError(f"LLM returned non-JSON: {text[:120]!r}") from e
        if not isinstance(out, dict):
            raise LLMError("LLM returned JSON that is not an object")
        return out
