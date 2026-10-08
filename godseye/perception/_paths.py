"""Make the pinned, un-pip-installable third-party source trees importable (NanoOWL, FastReID)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"


def add_third_party(name: str) -> Path:
    p = THIRD_PARTY / name
    if not p.is_dir():
        raise ImportError(f"{p} missing: clone it (see docs/DECISIONS.md, Phase 2 pins)")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
