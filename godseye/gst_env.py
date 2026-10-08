"""GStreamer bootstrap. Must run before the first Gst.init() in the process."""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

_lock = threading.Lock()
_Gst = None


def init_gst(plugin_paths: list[str] | None = None, base_dir: str | Path = ".", preload_libs: list[str] | None = None):
    """Idempotently initialise GStreamer; extra plugin dirs are prepended to GST_PLUGIN_PATH.

    Plugin dirs only take effect on the first call in a process. `preload_libs` are shared libraries
    dlopen'ed RTLD_GLOBAL first, so locally-extracted plugins can resolve their sonames without LD_LIBRARY_PATH.
    """
    global _Gst
    with _lock:
        if _Gst is not None:
            return _Gst
        for lib in preload_libs or []:
            ctypes.CDLL(str((Path(base_dir) / lib).resolve()), mode=ctypes.RTLD_GLOBAL)
        paths = [str((Path(base_dir) / p).resolve()) for p in (plugin_paths or [])]
        existing = os.environ.get("GST_PLUGIN_PATH")
        if existing:
            paths.append(existing)
        if paths:
            # private registry: the user-wide cache may have blacklisted these plugins before their libs were resolvable
            os.environ.setdefault("GST_REGISTRY_FORK", "no")  # scan in-process so preloaded libs resolve (the forked scanner would not see them)
            os.environ.setdefault("GST_REGISTRY", str(Path(paths[0]).parent / "registry.bin"))
            os.environ["GST_PLUGIN_PATH"] = os.pathsep.join(paths)
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        _Gst = Gst
        return Gst
