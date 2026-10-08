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
