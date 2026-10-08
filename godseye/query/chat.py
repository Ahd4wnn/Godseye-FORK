"""Terminal chat for Phase 4: prints the Resolved Query Context (Phase 5 will execute it). `python -m godseye.query.chat`"""
from __future__ import annotations

import sys

from .config import load_config
from .factory import build
from .understand import NeedsClarification, Rejected, Resolved, Session


def main(argv=None) -> int:
    cfg = load_config("config/query.yaml")
    qu = build(cfg)
    s = Session(cfg.recent_identities)
    print(f"Godseye query understanding ({cfg.model}). Ctrl-D to quit.")
    out = None
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        line = line.strip()
        if not line:
            continue
        out = qu.handle(s, line)
        while isinstance(out, NeedsClarification):
            c = out.clarification
            print(f"? {c.prompt}  [{c.kind}]")
            for i, o in enumerate(c.options):
                print(f"   {i + 1}. {o.label}")
            if c.kind in ("point_and_click", "region_selection"):
                print("   (a graphical client must answer this; type free text to restate the question instead)")
                out = qu.handle(s, input("> "))
                continue
            ans = input("> ").strip()
            if c.options and ans.isdigit() and 0 < int(ans) <= len(c.options):
                ans = c.options[int(ans) - 1].value
            out = qu.respond(s, c.cid, ans)
        print(out.context.model_dump_json(indent=2, exclude_defaults=True) if isinstance(out, Resolved) else out.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
