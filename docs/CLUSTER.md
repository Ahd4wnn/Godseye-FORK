# Ray cluster placement (verified 2026-10-08)

Cluster as found: Ray 2.40.0, Python 3.11.17; head `10.90.158.130` (this laptop, RTX 5050 8 GiB; started from another project's env,
`~/Hacknex/.venv-ray`), peer `10.90.158.67` (RTX 3050 6 GiB, 15 GiB RAM, Ollama with qwen3:8b already installed, torch only: no TensorRT,
transformers, ChromaDB, OpenCV or GStreamer). Link measured at about 13 MB/s each way.

## What runs where
| Component | Where | Why |
|---|---|---|
| GStreamer ingest, sampling, NanoOWL (TensorRT), ByteTrack, FastReID, SQLite, ChromaDB, clips | this laptop | needs the camera network, the local stores and the TensorRT/Blackwell build; frames are too large for the ~13 MB/s link |
| **LLM (qwen3:8b, Ollama)** | **peer laptop** | biggest GPU tenant (4.5-6 GiB), only needed per question; the peer GPU was idle |
| Query understanding, planner, retrieval, response, web chat | this laptop | read the local stores |

Not moved, deliberately: perception/Re-ID (would need the full TensorRT/torch stack replicated on the peer, an Ampere TRT engine, and
~100-200 KB JPEG frames per sample over the LAN); VRAM was never the limit for them (2.1 GiB measured).

## How the LLM reaches the peer
`tools/ray_llm_gateway.py` runs under **Python 3.11** (`.venv-ray`, `requirements-ray.lock`) because Ray 2.40 refuses a Python 3.12 driver on a
3.11 cluster (no override). It binds `127.0.0.1:11435` and serves only `GET /api/tags`, `POST /api/chat`, `GET /healthz`. Each chat is forwarded
to a Ray actor (`godseye_remote/llm_proxy.py`, standard library only, shipped with `runtime_env.py_modules`) pinned to the peer node, which calls
the peer's own Ollama on `127.0.0.1:11434`. The actor accepts only the keys `model, messages, stream, think, format, options`, only the model names in
`config/cluster.yaml`, bodies under 64 KB and options `temperature, num_ctx, num_predict, seed`. Nothing is exposed on the LAN by us.

Start: `.venv-ray/bin/python tools/ray_llm_gateway.py` then set `config/query.yaml: ollama_url: http://127.0.0.1:11435` and (recommended)
`llm_fallback_url: http://127.0.0.1:11434` so a lost peer degrades to the local Ollama with a WARNING instead of failing (off by default; the
fallback needs ~6 GiB of local VRAM when it loads).

## Operational notes and risks
* The Ray cluster itself has no authentication: anyone who can reach the head's ports (6379, 10001, 8265...) can run code on every node. Keep it on
  a trusted network or behind a firewall/VPN. This is a property of the existing cluster, not of this project.
* The peer's Ollama must stay running and keep `qwen3:8b` installed; the actor fails with a clear error otherwise (`LLMUnavailable`).
* Clip/frame data never leaves this laptop. Question text and the structured prompts do (to the peer's Ollama).
* Remote Ollama is "local" in the sense of the architecture (no cloud service); it is a LAN peer, not the loopback. This is a deviation from
  "local LLM on the same machine" that the project owner requested.
