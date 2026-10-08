"""TEST FIXTURES ONLY. Real H.264 clips (ffmpeg) served over real RTSP / HLS servers.

Never imported by runtime code. Nothing here fabricates frames for the production pipeline; it only
provides genuine encoded streams for the production pipeline to ingest.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GST_LOCAL = ROOT / ".gst-local"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_clip(path: Path, kind: str = "motion", seconds: int = 10, size: str = "640x360", fps: int = 25) -> Path:
    """kind: motion (moving test pattern) | static (flat gray) | static_then_motion (half/half)."""
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-g", str(fps), "-pix_fmt", "yuv420p"]
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    mot = f"testsrc2=size={size}:rate={fps}"
    sta = f"color=c=0x404040:size={size}:rate={fps}"
    if kind == "motion":
        cmd = base + ["-f", "lavfi", "-i", mot, "-t", str(seconds)] + enc
    elif kind == "static":
        cmd = base + ["-f", "lavfi", "-i", sta, "-t", str(seconds)] + enc
    elif kind == "static_then_motion":
        h = seconds // 2
        cmd = base + ["-f", "lavfi", "-t", str(h), "-i", sta, "-f", "lavfi", "-t", str(seconds - h), "-i", mot,
                      "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]"] + enc
    else:
        raise ValueError(kind)
    subprocess.run(cmd + [str(path)], check=True)
    return path


class RtspServer:
    """Real GstRtspServer in a subprocess so it can be killed/restarted to test reconnection."""

    def __init__(self, clip: Path, port: int | None = None):
        self.clip, self.port, self.proc = clip, port or free_port(), None

    @property
    def url(self) -> str:
        return f"rtsp://127.0.0.1:{self.port}/cam"

    def start(self) -> "RtspServer":
        env = dict(os.environ,
                   GST_PLUGIN_PATH=str(GST_LOCAL / "plugins"),
                   GST_REGISTRY=str(GST_LOCAL / "registry-rtspd.bin"), GST_REGISTRY_FORK="no",
                   LD_LIBRARY_PATH=f"{GST_LOCAL / 'lib'}:{GST_LOCAL / 'test-rtsp' / 'lib'}",
                   GI_TYPELIB_PATH=str(GST_LOCAL / "test-rtsp" / "typelib"))
        self.proc = subprocess.Popen(["/usr/bin/python3", str(Path(__file__).parent / "rtspd.py"), str(self.clip), str(self.port)],
                                     env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline()
        if line.strip() != "ready":
            raise RuntimeError(f"RTSP test server failed: {self.proc.stderr.read()}")
        return self

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()


class HlsServer:
    """ffmpeg segments the clip live (looping) into a dir served by http.server."""

    def __init__(self, clip: Path, workdir: Path):
        self.clip, self.dir, self.port = clip, workdir / "hls", free_port()
        self.ff = self.http = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/index.m3u8"

    def start(self) -> "HlsServer":
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(self.clip),
                                    "-c", "copy", "-f", "hls", "-hls_time", "1", "-hls_list_size", "5",
                                    "-hls_flags", "delete_segments", str(self.dir / "index.m3u8")])
        self.http = subprocess.Popen(["python3", "-m", "http.server", str(self.port), "--bind", "127.0.0.1", "--directory", str(self.dir)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t = time.time()
        while not (self.dir / "index.m3u8").exists() and time.time() - t < 10:
            time.sleep(0.1)
        time.sleep(1.5)
        return self

    def stop(self) -> None:
        for p in (self.ff, self.http):
            if p and p.poll() is None:
                p.terminate()
                p.wait(5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()
