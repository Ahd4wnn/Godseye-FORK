import shutil
from pathlib import Path

import pytest

from godseye.gst_env import init_gst
from godseye.ingest.config import IngestConfig, parse_camera
from tests.support.streams import ROOT, make_clip

PLUGINS = [".gst-local/plugins"]
PRELOAD = [".gst-local/lib/libgstcodecparsers-1.0.so.0", ".gst-local/lib/libgstmpegts-1.0.so.0"]


@pytest.fixture(scope="session")
def gst():
    return init_gst(PLUGINS, ROOT, PRELOAD)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed to generate real H.264 test clips")
    d = tmp_path_factory.mktemp("clips")
    return {k: make_clip(d / f"{k}.mp4", k, seconds=s) for k, s in
            [("motion", 12), ("static", 12), ("static_then_motion", 12)]}


@pytest.fixture
def make_cfg():
    def build(*cams: dict, defaults: dict | None = None) -> IngestConfig:
        base = {"reconnect": {"initial_backoff_s": 0.2, "max_backoff_s": 0.5, "connect_timeout_s": 6, "stall_timeout_s": 3}}
        return IngestConfig(plugin_paths=tuple(PLUGINS), preload_libs=tuple(PRELOAD), base_dir=str(ROOT),
                            cameras=tuple(parse_camera(c, {**base, **(defaults or {})}, i) for i, c in enumerate(cams)))
    return build


# ---- GPU hygiene: the local LLM (qwen3:8b, ~6 GB) and the perception models do not fit in 8 GB VRAM together at load time.
# In production the LLM runs on the Ray peer; in tests GPU-heavy modules unload the local one first (the real-LLM tests reload it).
GPU_MODULES = {"test_perception_real", "test_storage_real", "test_answer_real", "test_integration"}


def unload_local_llm(model: str = "qwen3:8b") -> None:
    try:
        import httpx
        httpx.post("http://127.0.0.1:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
        import time
        time.sleep(1.5)
    except Exception:                                    # noqa: BLE001 - Ollama not running: nothing to unload
        pass


@pytest.fixture(autouse=True, scope="module")
def _free_gpu_for_perception(request):
    if request.module.__name__.rsplit(".", 1)[-1] in GPU_MODULES:
        unload_local_llm()
    yield
