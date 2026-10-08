"""Camera topology (travel-time constraints). Nothing is assumed: with no topology every cross-camera transition is
'unverified'; pairs absent from a configured topology are routed through the link graph or judged unreachable."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import yaml


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class Link:
    a: str
    b: str
    min_travel_s: float
    max_travel_s: float
    overlap: bool = False
    directed: bool = False


class Topology:
    def __init__(self, links: list[Link] | None = None):
        self.links = list(links or [])
        self.configured = bool(self.links)
        self._adj: dict[str, list[tuple[str, float]]] = {}
        for l in self.links:
            self._adj.setdefault(l.a, []).append((l.b, l.min_travel_s))
            if not l.directed:
                self._adj.setdefault(l.b, []).append((l.a, l.min_travel_s))

    @classmethod
    def load(cls, path: str | Path) -> "Topology":
        try:
            doc = yaml.safe_load(Path(path).read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            raise TopologyError(f"cannot read topology {path}: {e}") from e
        if set(doc) - {"links"}:
            raise TopologyError(f"unknown topology keys {sorted(set(doc) - {'links'})}")
        links = []
        for i, d in enumerate(doc.get("links") or []):
            if set(d) - {"a", "b", "min_travel_s", "max_travel_s", "overlap", "directed"}:
                raise TopologyError(f"link {i}: unknown keys")
            try:
                l = Link(str(d["a"]), str(d["b"]), float(d["min_travel_s"]), float(d["max_travel_s"]),
                         bool(d.get("overlap", False)), bool(d.get("directed", False)))
            except (KeyError, TypeError, ValueError) as e:
                raise TopologyError(f"link {i}: {e}") from e
            if l.a == l.b or not 0 <= l.min_travel_s <= l.max_travel_s:
                raise TopologyError(f"link {i}: need a != b and 0 <= min_travel_s <= max_travel_s")
            links.append(l)
        return cls(links)

    def cameras(self) -> set[str]:
        return {c for l in self.links for c in (l.a, l.b)}

    def min_travel(self, a: str, b: str) -> float | None:
        """Shortest possible travel time a -> b over the link graph; None if unreachable (or either camera unknown)."""
        if a == b:
            return 0.0
        best, heap = {a: 0.0}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                return d
            if d > best.get(u, float("inf")):
                continue
            for v, w in self._adj.get(u, ()):
                if d + w < best.get(v, float("inf")):
                    best[v] = d + w
                    heapq.heappush(heap, (d + w, v))
        return None

    def link(self, a: str, b: str) -> Link | None:
        for l in self.links:
            if (l.a, l.b) == (a, b) or (not l.directed and (l.b, l.a) == (a, b)):
                return l
        return None
