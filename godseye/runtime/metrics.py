"""Runtime metrics: thread-safe counters and latency summaries plus host/GPU sampling. Everything reported is measured."""
from __future__ import annotations

import subprocess
import threading
import time
from collections import defaultdict, deque

import psutil


class Metrics:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self.t0 = time.time()
        self.counters: dict[str, int] = defaultdict(int)
        self._lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._rates: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self._lat[name].append(ms)

    def tick(self, name: str) -> None:
        with self._lock:
            self._rates[name].append(time.time())

    def rate(self, name: str, horizon: float = 10.0) -> float:
        with self._lock:
            ts = [t for t in self._rates[name] if t >= time.time() - horizon]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    @staticmethod
    def _pct(v: list[float], q: float) -> float:
        return v[min(len(v) - 1, int(q * len(v)))]

    def latency(self, name: str) -> dict:
        with self._lock:
            v = sorted(self._lat[name])
        return {} if not v else {"n": len(v), "mean": sum(v) / len(v), "p50": self._pct(v, .5), "p95": self._pct(v, .95), "max": v[-1]}

    def snapshot(self) -> dict:
        with self._lock:
            names = list(self._lat)
            counters = dict(self.counters)
        return {"uptime_s": time.time() - self.t0, "counters": counters, "latency_ms": {n: self.latency(n) for n in names}}


def gpu_sample() -> dict | None:
    """Real NVIDIA readings via nvidia-smi (fixed argv, no shell). None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()[0]
        u, used, total = (float(x) for x in out.split(","))
        return {"util_pct": u, "vram_used_mib": used, "vram_total_mib": total}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class HostSampler(threading.Thread):
    """Samples CPU / RAM / GPU every `period` seconds; `summary()` returns mean/max over the run."""
    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True, name="host-sampler")
        self.period, self._stop_ev, self.samples = period, threading.Event(), []
        self.proc = psutil.Process()

    def run(self):
        psutil.cpu_percent(None)
        self.proc.cpu_percent(None)
        while not self._stop_ev.wait(self.period):
            g = gpu_sample() or {}
            self.samples.append({"cpu_pct": psutil.cpu_percent(None), "proc_cpu_pct": self.proc.cpu_percent(None),
                                 "ram_used_mib": psutil.virtual_memory().used / 2**20, "ram_available_mib": psutil.virtual_memory().available / 2**20,
                                 "swap_used_mib": psutil.swap_memory().used / 2**20, "proc_rss_mib": self.proc.memory_info().rss / 2**20,
                                 "children_rss_mib": sum(c.memory_info().rss for c in self._children()) / 2**20, **g})

    def _children(self):
        try:
            return self.proc.children(recursive=True)
        except psutil.Error:
            return []

    def stop(self):
        self._stop_ev.set()
        self.join(5)

    def summary(self) -> dict:
        out = {}
        for k in {k for s in self.samples for k in s}:
            v = [s[k] for s in self.samples if k in s]
            out[k] = {"mean": sum(v) / len(v), "max": max(v)}
        return out
