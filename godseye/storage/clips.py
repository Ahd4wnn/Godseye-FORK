"""Evidence clip writer: raw BGR frames -> ffmpeg (libx264, yuv420p, faststart) -> validated, atomically published mp4.

A clip is written to `<name>.part`; only after ffmpeg exits cleanly AND ffprobe confirms codec / size / frame count is
it fsynced and renamed to its final path. Callers publish metadata only after `finish()` returns.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .config import StorageConfig


class ClipError(RuntimeError):
    pass


def fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def probe(cfg: StorageConfig, path: Path) -> dict:
    """-> {codec, width, height, frames}; raises ClipError when the file is not a decodable h264 mp4."""
    try:
        r = subprocess.run([cfg.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                            "stream=codec_name,width,height,nb_read_packets", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout)["streams"][0]
        if r.returncode != 0:
            raise ValueError(r.stderr)
        return {"codec": s["codec_name"], "width": int(s["width"]), "height": int(s["height"]),
                "frames": int(s["nb_read_packets"])}
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as e:
        raise ClipError(f"cannot probe {path.name}: {e}") from e


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class ClipWriter:
    def __init__(self, cfg: StorageConfig, final_path: Path, width: int, height: int):
        self.cfg, self.final, self.width, self.height = cfg, final_path, width, height
        self.part = final_path.with_name(final_path.name + ".part")
        self.frames = 0
        final_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(cfg.clip_fps), "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(self.part)]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise ClipError(f"cannot start {cfg.ffmpeg}: {e}") from e

    def write(self, bgr: np.ndarray) -> int:
        if bgr.shape != (self.height, self.width, 3) or bgr.dtype != np.uint8:
            raise ClipError(f"frame {bgr.shape}/{bgr.dtype} does not match clip {self.height}x{self.width}x3 uint8")
        try:
            self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        except (BrokenPipeError, OSError) as e:
            raise ClipError(f"ffmpeg write failed: {e}; {self._stderr()}") from e
        self.frames += 1
        return self.frames - 1

    def _stderr(self) -> str:
        try:
            self.proc.kill()
            return self.proc.stderr.read().decode(errors="replace")[-300:]
        except Exception:                                       # noqa: BLE001 - diagnostics only
            return ""

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:                                       # noqa: BLE001
            pass
        for s in (self.proc.stdin, self.proc.stderr):
            try:
                s and s.close()
            except OSError:
                pass
        self.part.unlink(missing_ok=True)

    def finish(self) -> dict:
        """Close, validate, publish atomically. -> {width, height, frames, size_bytes, sha256}."""
        if self.frames == 0:
            self.abort()
            raise ClipError("empty clip")
        try:
            self.proc.stdin.close()
            rc = self.proc.wait(timeout=120)
            err = self.proc.stderr.read().decode(errors="replace")[-300:]
            self.proc.stderr.close()
            if rc != 0:
                raise ClipError(f"ffmpeg exited {rc}: {err}")
            info = probe(self.cfg, self.part)
            if info["codec"] != "h264" or info["frames"] != self.frames or (info["width"], info["height"]) != (
                    self.width + self.width % 2, self.height + self.height % 2):
                raise ClipError(f"clip failed validation: wrote {self.frames} frames {self.width}x{self.height}, probe={info}")
            size, digest = self.part.stat().st_size, sha256_file(self.part)
            with open(self.part, "rb") as f:
                os.fsync(f.fileno())
            os.replace(self.part, self.final)
            fsync_dir(self.final.parent)
        except (ClipError, OSError, subprocess.SubprocessError) as e:
            self.abort()
            raise e if isinstance(e, ClipError) else ClipError(str(e)) from e
        return {"width": info["width"], "height": info["height"], "frames": self.frames, "size_bytes": size, "sha256": digest}
