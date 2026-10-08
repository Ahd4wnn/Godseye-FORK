# Performance (measured 2026-10-08, this machine)

Machine: RTX 5050 Laptop 8 GiB, 16 logical CPUs, 15 GiB RAM (about 10 GiB used by other applications, 4 GiB swap that was already full),
Linux 7.0, GStreamer 1.24 (software H.264 decode), TensorRT 11.3 FP16 NanoOWL, FastReID R50-ibn.
Method: `python -m godseye.runtime.bench --video <1280x720 H.264, 24 fps> --cameras N --seconds 40`. Each camera loops the same file; decode,
sampling, detection, tracking, Re-ID, SQLite, ChromaDB and clip encoding are all real. An 8 s warm-up is excluded. Single runs on one
machine with other applications active: treat differences below about 10% as noise. Sampling was base 2 fps / burst 5 fps; the clip is
continuously moving, so every camera sampled at about 5 fps (the burst rate).

| Run | Cameras | Processed fps (total) | Dropped (queue, 40 s) | e2e p50 / p95 | Notes |
|---|---|---|---|---|---|
| Baseline | 1 | 4.99 | 0 | 29 / 43 ms | source-limited (5 fps) |
| Baseline | 4 | 19.99 | 0 | 57 / 184 ms | no drops, queue depth max 2 |
| Baseline | 8 | **3.28** | 682 | 768 / 1883 ms (max 19.6 s) | collapse: RAM available 385 MiB, swap 4 GiB full |
| Diagnostic: clip encoding off | 8 | 29.6 | 415 | 725 / 847 ms | isolates the cause |
| After encoder fix (2 threads, no lookahead) | 8 | 27.7 | 485 | 727 / 845 ms | ffmpeg children 1.87 GB -> 0.72 GB RSS |
| + FastReID fp16 | 8 | 29.1 | 435 | 733 / 841 ms | Re-ID p50 15.0 -> 13.3 ms |

## Measured bottlenecks and what was done
1. **8 cameras collapsed because of host RAM, not VRAM.** Each default ffmpeg/x264 encoder used about 240 MB (all cores, lookahead); eight of
   them plus the process drove the machine into swap (detect stalls up to 13.5 s). Fix: `storage.encoder_threads: 2` and
   `x264_params: rc-lookahead=0:sync-lookahead=0:bframes=0` (evidence clips, so low latency matters more than compression). 3.3 -> 27.7 fps.
2. **FastReID in fp16** (`reid.*.precision`, default fp16): 12 crops 69.7 -> 20.7 ms in isolation; features keep cosine >= 0.99998 to fp32 and
   similarities differ by at most 0.0024 (tested: `test_reid_fp16_matches_fp32`). End-to-end gain is small (+5%) because...
3. **After 1 and 2 the limit is the single Python pipeline thread** (about 25 ms per sampled frame: detect 10.4 + Re-ID 13.3 + track 0.8 + store 3.3 ms,
   while eight camera threads compete for the GIL). Capacity is about 29 fps of sampled frames: roughly 5-6 cameras that move continuously at
   5 fps, or about 14 cameras at the 2 fps base rate. Beyond that frames are dropped by design (drop-oldest queues, counted). Next lever, not built:
   several Track A worker processes, each owning a camera group.

## LLM placement (what the Ray peer buys)
Track A at 8 cameras while questions arrive every 2 s (identical load, 85 s):

| LLM location | Track A fps | detect p95 | VRAM peak | RAM available (mean) | Question latency (median) |
|---|---|---|---|---|---|
| This GPU (Ollama local) | 24.6 | 48.8 ms | 7.2 GiB | 0.9 GB | 1.2 s |
| Peer RTX 3050 via Ray gateway | 28.6 | 15.7 ms | 1.8 GiB | 1.7 GB | 1.9 s |

Query latency through the gateway is about 0.5-0.7 s higher (Ray hop on a ~13 MB/s link). Cold model load: 5-7 s either way.

## Stage latencies, single camera (p50 / p95)
detect 10.3 / 12.0 ms (TensorRT FP16, NanoOWL `detect()` on a 1280x720 frame); track 0.9 / 1.3; Re-ID 13.0 / 26.8; storage write 2.8 / 10.3 (SQLite+Chroma,
clip frame pipe included); pipeline 24.4 / 38.4; end-to-end capture->persisted 29 / 43 ms. GPU utilisation 8% (1 camera), 26% (4 cameras), 24-34% (8 cameras).
Process CPU: 64% of one core (1 camera), about 390% (8 cameras). Process RSS 2.3 GB (1 camera), 3.1-3.5 GB (8 cameras).

## Not measured
Soak (hours), real RTSP cameras / network jitter, 1080p or 4K sources, hardware decode (unavailable), more than 8 cameras, multi-process Track A,
the ChromaDB/SQLite growth rate over days, and anything on the peer's GPU beyond the LLM.
