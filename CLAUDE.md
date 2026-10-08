# CLAUDE.md — Godseye Project Operating Manual

Last updated: 2026-10-08 (Phase 1). Facts marked UNKNOWN / NOT VERIFIED have not been checked. Never promote them without verifying.

# PROJECT

Godseye is a production-grade multi-camera CCTV video analytics and natural-language surveillance search system. It has two tracks:

- Track A ingests live camera streams, detects and tracks people/objects, re-identifies them across cameras, and stores structured metadata, embeddings and clips.
- Track B lets a user ask questions in natural language and answers them from stored data, with grounded evidence (clips, timestamps, camera IDs).

# ARCHITECTURE

These boundaries are architectural requirements. Do not simplify or replace them.

## TRACK A — VIDEO ANALYTICS
CCTV → RTSP/RTMP/HLS → GStreamer → Frame Sampling → Preprocessing → NanoOwl Detection → Embedding Reuse → ByteTrack → FastReID → ChromaDB → SQLite → Clip Storage

| Component | Responsibility |
|---|---|
| GStreamer | Stream ingestion + decoding (RTSP/RTMP/HLS). Per-camera pipelines, reconnect handling. |
| Frame Sampling | Choose which decoded frames go to inference (rate control). |
| Preprocessing | Resize/normalize/convert frames to model input. |
| NanoOwl | Open-vocabulary detection (text-prompted). Also produces image embeddings reused downstream. |
| Embedding Reuse | Reuse NanoOwl-produced embeddings instead of recomputing where applicable. |
| ByteTrack | Per-camera tracking; assigns `track_id` (local to one camera/session). |
| FastReID | Cross-camera Re-ID embeddings; assigns/associates `global_id`. |
| ChromaDB | Vector storage (Re-ID / appearance embeddings). |
| SQLite | Structured metadata + knowledge: cameras, tracks, detections, global identities, events, clip index. |
| Clip Storage | Filesystem storage of evidence clips; indexed from SQLite. |

## TRACK B — QUERY / SEARCH
User → Chat UI → Local LLM / NLU → Referent Resolver → Clarification Engine → Resolved Query Context → Query Planner → Retrieval Engine → Path Reconstructor → Response Builder → Grounded Answer + Evidence

| Component | Responsibility |
|---|---|
| Chat UI | User-facing conversation interface. |
| Local LLM / NLU | Parses the user message into a structured intent (output is data, never executed). |
| Referent Resolver | Resolves "he", "that person", "the same car" etc. against conversation/query context. |
| Clarification Engine | Asks follow-ups when intent/referent is ambiguous. |
| Resolved Query Context | Validated, fully-resolved structured query object. |
| Query Planner | Turns the resolved context into a retrieval plan (allow-listed operations only). |
| Retrieval Engine | Executes the plan against SQLite / ChromaDB / clip index. |
| Path Reconstructor | Builds a cross-camera movement path from `global_id` observations. |
| Response Builder | Produces the answer strictly grounded in retrieved records, with evidence references. |

Exact module layout and per-component interfaces: NOT YET DESIGNED (no code exists).

# CURRENT PHASE
Phase 7 (final audit) done; awaiting user decisions on the blockers in PHASE 7. Previous: Phase 6 — End-to-end integration & hardening: implemented, measured and tested (see PHASE 6 at the end), awaiting user acceptance before Phase 7 (final audit). The two tracks run end to end on this machine; the LLM can run on the Ray peer. Open items are listed in PHASE 6 LIMITATIONS.
Phase prompts live in `cctv_claude_code_prompts/` (master contract `00_*`, phases `01_`–`08_`). Docs required by the contract are in `docs/`.

# PHASE STATUS
- Phase 0 — Architecture & CLAUDE.md: COMPLETE
- Phase 1 — Video ingestion (GStreamer → sampling → preprocessing): COMPLETE except RTMP integration verification (BLOCKED on an RTMP server)
- Phase 2 — AI perception, tracking & cross-camera identity (NanoOwl, ByteTrack, FastReID): COMPLETE pending acceptance (vehicle Re-ID unvalidated; Re-ID thresholds uncalibrated for any real site)
- Phase 3 — Storage & evidence (ChromaDB, SQLite, clips): COMPLETE pending acceptance (retrieval NOT implemented by design; single-writer; clips need ffmpeg)
- Phase 4 — Query understanding: COMPLETE pending acceptance (terminal chat only; LLM = qwen3:8b by user approval, NOT Llama 3)
- Phase 5 — Retrieval, path reconstruction, response: COMPLETE pending acceptance (topology is a placeholder example until measured on site; attribute search is approximate)
- Phase 6 — Integration & hardening: COMPLETE pending acceptance (not soak-tested; no real cameras; web UI JavaScript not exercised in a browser; Ray cluster unauthenticated)
- Phase 7 — Final audit: COMPLETE (2026-10-08) — see PHASE 7 at the end. Verdict: NOT READY for production

# IMPLEMENTED COMPONENTS
Verified by the tests listed in TEST STATUS:
- `godseye/gst_env.py` — GStreamer bootstrap (plugin paths, lib preload, private registry).
- `godseye/ingest/config.py` — validated YAML config (unknown keys rejected, `${ENV}` URL expansion, per-camera overrides).
- `godseye/ingest/camera.py` — per-camera GStreamer pipeline + worker thread: RTSP/HLS/file verified against real servers; RTMP element wired but NOT verified. Reconnect with backoff, connect/stall watchdogs, `hw_accel` fail-loud, malformed-buffer handling.
- `godseye/ingest/sampler.py` — base 2 fps / burst 5 fps (configurable), MOG2 gating, redundancy drop, keepalive.
- `godseye/ingest/preprocess.py` — privacy masks, letterbox 640, RGB float32 [0,1], coordinate transform with round-trip.
- `godseye/ingest/manager.py` — `IngestManager` + bounded `FrameBus`.
- `config/ingest.yaml` (ships with `cameras: []`), `docs/*.md`, `tests/`.
Phase 2 (`godseye/perception/`): see PHASE 2 below. Phase 3 (`godseye/storage/`): see PHASE 3 below. Phase 4 (`godseye/query/`): see PHASE 4 below. Phase 5 (`godseye/answer/`): see PHASE 5 below. Phase 6 (`godseye/runtime/`, `godseye/web/`, `godseye_remote/`, `tools/`): see PHASE 6 below.
Phase 7 produced no code, only the audit.

# TECHNOLOGY REQUIREMENTS

Required (never replace with weaker alternatives without explicit architectural approval):

- GStreamer — ingestion/decoding
- NanoOwl — detection
- ByteTrack — local tracking
- FastReID — cross-camera Re-ID
- ChromaDB — vector store
- SQLite — structured store
- Local LLM via Ollama (Llama 3 where specified) — query understanding/planning

DECIDED 2026-10-08 (user): Track B uses `qwen3:8b` (the only 8B installed), an explicit deviation from the specified Llama 3 8B. The model is a config value (`config/query.yaml: model`); switching to Llama 3 = `ollama pull llama3:8b` + change the value + re-run tests/test_query_real.py. Do not substitute any other model silently.

# ARCHITECTURAL INVARIANTS

Never violate:

1. GStreamer is the video ingestion/decoding architecture.
2. NanoOwl is the detection architecture.
3. ByteTrack is the local tracking architecture.
4. FastReID is the cross-camera Re-ID architecture.
5. ChromaDB is the vector storage architecture.
6. SQLite is the structured metadata/knowledge storage architecture.
7. A local LLM is used for query understanding/planning where specified. No cloud LLM in the query path.
8. `track_id` (per-camera, ByteTrack) and `global_id` (cross-camera, FastReID) are distinct concepts. Never conflate or reuse one for the other.
9. LLM output must never directly execute database, filesystem or shell operations. It is parsed into a validated structure, then the planner issues allow-listed operations.
10. Production paths must never use fake inference.
11. Test mocks stay isolated from production paths (tests/ only; never imported by runtime code).
12. Track A and Track B stay separate, communicating only through SQLite / ChromaDB / clip storage.
13. Answers must be grounded in retrieved records. No fabricated evidence.

# INTERFACE CONTRACTS
Verified: Phase 1 → Phase 2 `PreprocessedFrame` and `IngestManager` API. Full field table in `docs/INTERFACES.md`.
Key points: `image` float32 (640,640,3) RGB [0,1]; `transform.to_original()` maps model-space boxes to original pixels; `original_bgr` kept for evidence; `stream_epoch` increments on reconnect (trackers must reset).
OPEN for Phase 2: NanoOwl's real input requirements are NOT VERIFIED (OWL-ViT-style models normally use their own resolution and CLIP mean/std). Phase 2 must verify and, if needed, derive the model input from `original_bgr`.

# DEPENDENCIES
Verified 2026-10-08.
System: Linux 7.0.0-34; Python 3.12.3; GStreamer 1.24.2 + PyGObject 3.48.2; ffmpeg 6.1.1 (with libx264, used for test clips only); SQLite 3.45.1; Docker 29.6.2; Ollama 0.32.4 (models: qwen3:8b, qwen3:4b, qwen3-vl:4b, qwen3-8k; no Llama 3).
Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, driver 595.84, CUDA 13.2 (per nvidia-smi); 14 GiB RAM.
Project venv `.venv` (created with `--system-site-packages` so `gi` works): numpy 2.5.3, opencv-python-headless 5.0.0, pytest 9.1.1, PyYAML 6.0.1.
GStreamer elements: present in system: rtspsrc, decodebin, decodebin3, videoconvert, appsink, souphttpsrc, hlsdemux2, flvdemux, filesrc. Missing in system (no sudo): avdec_h264, rtmp2src, h264parse, tsdemux.
Local, rootless (`.gst-local/`, extracted from Ubuntu noble `.deb`s, same GStreamer 1.24.x): `plugins/` = libgstlibav (avdec_h264/h265), libgstrtmp2 (rtmp2src), libgstvideoparsersbad (h264parse), libgstmpegtsdemux (tsdemux); `lib/` = libgstcodecparsers, libgstmpegts; `test-rtsp/` = GstRtspServer typelib+lib for the TEST RTSP server only.
Hardware decode: nvh264dec/vah264dec NOT present, so `hw_accel` can only be `none` on this machine; the fail-loud path is tested.
NOT installed (Phase 2+ will need): torch, torchvision, transformers, onnxruntime, tensorrt, torch2trt, nanoowl, fastreid, ByteTrack deps (scipy, lap, cython_bbox), chromadb, ollama client, fastapi/pydantic. Model weights NOT present. PyTorch/TensorRT support for RTX 5050 (Blackwell) + CUDA 13.2: NOT VERIFIED.

# CONFIGURATION
`config/ingest.yaml` — sections `gstreamer` (plugin_paths, preload_libs), `defaults` (sampling / preprocess / reconnect / camera keys), `cameras[]`. Loader: `godseye.ingest.config.load_config`. Credentials via `${ENV_VAR}` in URLs. Per-field reference: dataclasses in `godseye/ingest/config.py`. Later phases' config location: UNDECIDED.

# DATABASE / STORAGE

Planned responsibilities (none created yet):
- ChromaDB: appearance/Re-ID embedding vectors for similarity search. Not the source of truth for metadata.
- SQLite: cameras, tracks, detections, global identities, events, clip index. Source of truth for structured data. Schema: NOT DESIGNED.
- Filesystem clips: evidence video clips. SQLite stores paths and time ranges. Layout: NOT DESIGNED.

# TEST STATUS
Command: `.venv/bin/python -m pytest tests -q` (2026-10-08, after Phase 6): **222 passed, 1 skipped** in about 10.5 min (Phase 6 added 47; the real-model/LLM/cluster tests dominate the time) (skipped = RTMP integration test, needs `GODSEYE_TEST_RTMP_URL`). Phase 1 alone: 65 passed. Phase 2 tests: see PHASE 2 TESTS.
Covered: config validation, preprocessing/coordinate round-trips, sampler rates (2 fps base, ~5 fps burst, static gating, keepalive), bus policies/fairness, multi-camera isolation, real RTSP and HLS ingest, disconnect/reconnect (stream_epoch), stall recovery, invalid/missing/404 streams, max_attempts, hw_accel failure, queue pressure, loop_file, runtime add/remove, shutdown, environment failure, malformed buffers.
NOT covered: RTMP, real cameras, hardware decode, soak/performance. No performance numbers measured.

# KNOWN ISSUES
1. RTMP is not integration-verified. `rtmp2src` connects to an ffmpeg `-listen 1` endpoint but receives no data (ffmpeg is not a real RTMP play server), and no other RTMP server exists on this machine. Needs a real RTMP server (e.g. MediaMTX, nginx-rtmp) or camera; then run `GODSEYE_TEST_RTMP_URL=... pytest -k rtmp`.
2. System GStreamer lacks the H.264 decoder and related plugins; the project works around it with extracted `.deb`s in `.gst-local/` (see docs/DECISIONS.md #4). Not portable until `gstreamer1.0-libav gstreamer1.0-plugins-bad` are installed system-wide.
3. Stale `.gst-local/registry*.bin` can cache plugin-load failures; delete to rescan.

# KNOWN LIMITATIONS
- Only software H.264 (libav) decode verified; H.265 plugin present but untested.
- Sampler timing uses wall-clock at pull time, not stream PTS.
- Frames hold full-res `original_bgr` (about 6 MB at 1080p) times `queue_size` per camera; memory sizing for many cameras is not measured.
- 8 GB VRAM budget for Phase 2+ models is NOT VERIFIED; free RAM was about 5.5 GiB at check time.
- No test video of real CCTV footage; fixtures are synthetic ffmpeg patterns (encoded H.264, ingested genuinely).
- No git repository.

# ARCHITECTURAL DECISIONS
Mandated: two-track split; `track_id` / `global_id` separation; LLM never executes anything directly.
Phase 1 (rationale in `docs/DECISIONS.md`): programmatic GStreamer property setting; `emit("try-pull-sample")`; HLS via decodebin3; rootless local plugins; OpenCV only for resize/MOG2; pure-function sampler; drop-based backpressure at three levels; fail-loud hw_accel.
Pending (need user input): LLM choice (Qwen vs Llama 3), NanoOwl/FastReID weight sourcing, RTMP test server.

# DO NOT CHANGE

- The architecture pipelines and component boundaries above.
- The technology choices in TECHNOLOGY REQUIREMENTS.
- The `track_id` / `global_id` distinction.
- The rule that LLM output is never executed directly.

# NEXT ACTION (superseded by the Phase 2 line at the end of this file)

1. User decides how to verify RTMP (provide a server/URL, or approve fetching MediaMTX) and accepts or rejects Phase 1.
2. Phase 2 (`cctv_claude_code_prompts/03_PHASE_2_AI_PERCEPTION.md`): FIRST verify NanoOwl package/weights/inference API and torch/TensorRT on the RTX 5050 (CUDA 13.2). If unverifiable, STOP and report per the master contract. Do not substitute YOLO or another detector.

# NEXT ACTION
1. User accepts or rejects Phase 6. Useful inputs: real camera URLs + a soak run, the real camera topology (travel times), site footage for Re-ID threshold calibration, vehicle Re-ID data, an SHA-256 pin for the FastReID weights.
2. Phase 7 (`08_*`): final audit of everything against the master contract and CLAUDE.md.
3. Still open: RTMP verification (Phase 1); Llama 3 vs Qwen (Phase 4); 640-vs-768 input spec; multi-process Track A for more than ~6 busy cameras.

# PHASE 2 — AI PERCEPTION (COMPLETE pending acceptance; verified 2026-10-08)
## Pipeline / modules (`godseye/perception/`)
`PerceptionPipeline.process(PreprocessedFrame) -> EnrichedFrame` (single worker thread; models shared across cameras).
- `nanoowl_detector.py` NanoOWL `OwlPredictor` (unmodified, `third_party/nanoowl` @ fb553dee) + TensorRT FP16 encoder engine (`trt_runtime.py`, built on first start, cached in `models/nanoowl/`) or explicit `backend: torch`. Open-vocabulary prompts from config. Embedding reuse: each detection's 512-d CLIP-aligned class embedding is the OWL-ViT head output of the same forward pass (no second CLIP), L2-normalised and validated; `patch_index` associates it with its box. Boxes are in ORIGINAL-frame pixels, clipped. Provenance (model, backend, precision, nanoowl commit, engine sha256, prompt hash) on every Detection.
- `tracker.py` real ByteTrack (vendored, `bytetrack/README`, upstream d1bf0191), one instance per Re-ID class group (person/vehicle/other) per camera so groups never associate; `track_id` from a per-camera allocator. Emits trajectory, hits, first/last seen, label vote, mean reused embedding, `TrackEnded` (`lost_timeout | stream_restart | shutdown`).
- `reid.py` real FastReID (`third_party/fast-reid` @ c9bc3ce, SBS-R50-ibn; person=MSMT17, vehicle=VeRi; 2048-d L2-normalised). Strict checkpoint loader remaps legacy `heads.bnneck.*` keys and fails on any other mismatch.
- `identity.py` `global_id` resolution (`IdentityResolver`, in-memory `IdentityGallery`). Per-track mean of >= `min_observations` quality Re-ID features -> gallery query (best-exemplar cosine, per Re-ID class). `matched` only if score >= `match_threshold` AND beats runner-up by `margin`; between `ambiguous_floor` and that -> `ambiguous` (no global_id, candidates listed, one re-check at `recheck_observations`, never force-merged); below floor -> `new` identity. An identity alive on the same camera under another track is excluded as a candidate. Crops overlapped > `max_overlap` by another same-class box, or outside size/aspect limits, are not used (occlusion/low quality).
- `pipeline.py`, `config.py` (strict YAML, `config/perception.yaml`), `types.py` (contract: `Detection`, `TrackedObject`, `TrackEnded`, `EnrichedFrame`).
## Interface notes
- `track_id` is camera-local AND restarts at 1 after a reconnect (`stream_epoch` change drains all tracks with reason `stream_restart` and starts a fresh tracker). Key by (camera_id, stream_epoch, track_id). `TrackedObject`/`TrackEnded` carry `stream_epoch`.
- `global_id` format `person-<12 hex>` / `vehicle-<12 hex>`; None for reid_class `other` (status `not_applicable`) and until resolved. Gallery is in-memory: global_ids do NOT survive a restart until Phase 3 persists the gallery.
- ByteTrack quirk (upstream): a track is emitted from the first frame only if it is the tracker's first frame; later new tracks need a second match; new tracks need score > track_thresh + 0.1.
## Runtime requirements
torch 2.14.1+cu130, tensorrt 11.3.0.99, transformers 5.19.0, scipy, lap, cython_bbox, onnx in `.venv`; RTX 5050 (sm_120) verified. NanoOWL model `google/owlvit-base-patch32` (HF cache); FastReID weights in `models/fastreid/`. First start builds the TRT engine (about a minute). Measured once on this machine on a 1280x720 video: detect ~9-10 ms, per-frame Re-ID ~10-15 ms with ~5-15 people. Pipeline init ~8 s.
## Configuration
`config/perception.yaml` sections `detector` (prompts with per-prompt threshold and `reid_class`), `tracker`, `reid.person|vehicle` (model, crop min size / aspect), `identity`. Unknown keys rejected. Defaults are NOT calibrated for any real site.
## PHASE 2 TESTS
`tests/test_perception_unit.py` (11; real ByteTrack on synthetic detections, resolver on synthetic features, config, pipeline epoch reset with an isolated fake detector/extractor): lifecycle/stable ids, two objects, occlusion gap kept / long gap ended, person-vs-vehicle separation, camera-local ids, new/matched/different/ambiguous/same-camera/class-separated identity.
`tests/test_perception_real.py` (8; real NanoOWL TRT + real FastReID; needs CUDA, a real video via `GODSEYE_TEST_VIDEO` default `~/Downloads/Busy_airport*.mp4`, and `data/market1501_raw`): detection contract, coordinate mapping (padded image -> boxes shift), determinism, our decode == NanoOWL decode, 40-frame video tracking + identity invariants, Re-ID embedding consistency (Market-1501: same-id cos 0.575 vs diff-id 0.206), cross-camera identity (25 known + 15 unknown people: right=15 wrong=0 ambiguous=10 new=15), TensorRT-with-GStreamer regression.
## PHASE 2 LIMITATIONS / ISSUES
1. Re-ID accuracy: MSMT17 weights on Market-1501 (cross-domain): Rank-1 0.62, mAP 0.34. Investigated: RGB 0-255, RGB mean/std, eval mode and checkpoint BN stats are correct; BGR worse; no bug found. Most likely a genuine domain gap (NOT proven). Cross-camera merging is therefore conservative: many true matches end `ambiguous`/`new` (duplicate identities) rather than false merges. Thresholds (`match_threshold` .70, `margin` .05, `ambiguous_floor` .55) are uncalibrated defaults; the 40-person Market test showed no false merges at them, but that is a small sample on a different domain. They MUST be calibrated on footage from the real cameras.
2. Vehicle Re-ID (VeRi weights) loads and runs but is UNVALIDATED (no vehicle Re-ID data here). Treat vehicle global_ids as experimental.
3. Input size CONFLICT (decision needed): NanoOWL natively uses 768 px with CLIP normalisation; the detector consumes `original_bgr`, not Phase 1's 640 `image`. Phase 1 must therefore run with `keep_original: true` and privacy masks applied to the original (`privacy_apply_to_original`), else the detector raises.
4. Real-video testing used one AI-generated-looking airport clip (not real CCTV); no occlusion-heavy or night footage; small distant objects can be missed (a distant cyclist was).
5. Found and fixed: (a) clearing ByteTrack's `removed_stracks` resurrected dead tracks (now pruned, not cleared); (b) GStreamer in the same process made TensorRT abort (libunwind vs `tuple(Dims)`), fixed in `trt_runtime._dims`. Any other TRT call that throws internally under GStreamer may still abort.
6. Gallery is in-memory, unbounded growth (exemplars capped per identity, identities not expired: `gallery_ttl_s` is not implemented). Ambiguous tracks never receive a global_id.
7. Timing figures are single measurements; no soak/multi-camera load test; models are shared by one worker, so throughput across many cameras is unmeasured.
## PHASE 2 DECISIONS
Per-class-group ByteTrack; conservative identity rules (merge only when clearly best, ambiguity reported not resolved); occlusion/quality gating of Re-ID crops; embedding reuse from the OWL-ViT class-embedding head; explicit `backend` (no silent torch fallback); stale engine/ONNX experiments remain in `models/nanoowl/` (safe to delete all but `owlvit-base-patch32-fp16-trt11.3.0.99-sm120-b1.engine`).

# PHASE 3 — PERSISTENCE, VECTORS & EVIDENCE (COMPLETE pending acceptance; verified 2026-10-08)
## Architecture (`godseye/storage/`, config `config/storage.yaml`, root `data/store/`)
`EvidenceStore.write(EnrichedFrame)` (single writer thread; fcntl lock `<root>/.lock` -> second opener gets `StorageError`). Layout: `godseye.db` (SQLite WAL), `chroma/` (ChromaDB PersistentClient, telemetry off), `clips/{YYYY-MM-DD UTC}/{camera_id}/{start_ms}_{uuid8}.mp4`. Pass `store.gallery` to `PerceptionPipeline(gallery=...)`: the Chroma-backed `PersistentGallery` replaces the in-memory one, so `global_id`s now survive restarts. Call `pipeline.shutdown()` and `store.write()` its `ended_tracks`, then `store.close()` (else tracks look crash-closed on next start).
## SQLite schema (user_version 1; `db.py`)
`cameras`; `identities(global_id, reid_class person|vehicle, first/last_seen, last_camera)`; `clips(clip_id, camera_id, stream_epoch, start/end_ts, path relative, fps, frames, width, height, frame_ts JSON, size_bytes, sha256, status complete|missing|corrupt)`; `tracks(track_row, camera_id, stream_epoch, track_id, label, reid_class, start_ts, last_seen_ts, end_ts NULL=open, hits, global_id FK, identity_status/score/candidates, end_reason, embedding_id; UNIQUE(camera,epoch,track_id))`; `observations(obs_id, track_row FK, camera_id, frame_seq, ts, x1..y2 tracker box, dx1..dy2 detection box, confidence, label, clip_id FK, clip_frame; UNIQUE(track_row, frame_seq))`. Indexes on global_id, (camera, time), open tracks, observation (camera,ts)/(track,ts)/clip. FKs on. `track_id` and `global_id` are separate columns; tracks are keyed by (camera_id, stream_epoch, track_id).
To play an observation: `clips.path` + `observations.clip_frame` (seek = frame index; `clips.frame_ts[i]` gives each frame's capture time; playback fps is `clip_fps` while capture is sampled at 2-5 fps, so clips play faster than real time).
## ChromaDB collections (cosine space, deterministic ids => idempotent upserts)
`reid_person`, `reid_vehicle` (2048-d FastReID exemplars = the identity gallery; id `{gid}:{sha1}`, metadata global_id/camera_id/ts; capped at `max_exemplars` per identity, first + newest kept); `track_clip` (512-d reused NanoOWL/CLIP track embedding; id `{camera}:{epoch}:{track}`; refreshed every `track_embed_every` hits and finalised at track end with the final global_id). Search: `VectorStore.search(name, vec, k, where)`. Gallery query takes the best exemplar per identity among the top `chroma_query_k`=100 hits.
## Clips
Only frames while a camera has tracks (+ `idle_close_s` after), one segment per camera shared by all its tracks (not per-track), closed on idle / `max_segment_s` / `stream_epoch` or resolution change. Frames (`original_bgr`, privacy masks already applied; store refuses unmasked originals) are piped to ffmpeg (libx264, yuv420p, faststart) into `*.part`; on close the clip is probed (ffprobe: h264, size, packet count == frames written), sha256'd, fsynced, atomically renamed, and only then are the `clips` row and `observations.clip_id` committed. No pre-roll.
## Consistency / failure behaviour
- Order per frame: vectors (idempotent) -> clip frame -> one SQLite transaction. Publish order for clips: media validated -> rename -> SQLite commit.
- SQLite/Chroma failure: exception to caller, frame rolled back (nothing partial in SQLite). Clip failure (ffmpeg missing/dead, low disk `min_free_bytes`, validation): DEGRADED, not fatal -- `.part` deleted, affected observations keep `clip_id`/`clip_frame` NULL, `store.health()` (`clip_failures`, `last_clip_error`) and an ERROR log record it, camera retried after `retry_after_s`.
- Crash: `recover()` on open deletes `.part` files, closes open tracks (`end_reason=crash_recovered`), detaches observations of unpublished clips. At most one open segment of video per camera is lost; metadata is kept.
- `reconcile(fix=False, deep=False)` reports: clips with missing file / wrong size (/ wrong sha256 when `deep`), orphan clip files, orphan vectors (gallery vectors with no `identities` row; `track_clip` with no track), identities without vectors, tracks whose embedding is missing. `fix=True` marks clips `missing|corrupt` (never deletes referenced evidence), removes orphan files/vectors.
- Duplicates: re-writing a frame is a no-op (UNIQUE + deterministic vector ids).
## PHASE 3 TESTS
`tests/test_storage.py` (18; real SQLite/Chroma/ffmpeg, synthetic perception output): clip publish + metadata consistency, frame<->time mapping, epoch split, restart persistence, vector search + persistent gallery matching across restart, exemplar cap, duplicates, single-writer lock, unsafe camera id, privacy refusal, missing ffmpeg, low disk + back-off, encoder killed mid-segment, Chroma failure, SQLite failure rollback, crash recovery, missing/corrupt/same-size-corrupt clips, orphan vectors, config. `tests/test_storage_real.py` (1): real video -> real pipeline -> store -> restart -> DB equals pipeline output, clips probe OK, `reconcile(deep=True)` clean.
## PHASE 3 LIMITATIONS / ISSUES
1. Not covered: real disk-full (ENOSPC) on a full filesystem (low-disk is the pre-check; encoder death is covered), power loss, concurrent readers during writes, long soak, large-scale Chroma/SQLite volume (one observation row per tracked object per sampled frame; growth rate NOT measured).
2. Chroma and SQLite cannot be atomic: a crash can leave orphan vectors/clips (cleaned by reconcile) and a vector exists before its SQLite row. Reconcile is manual (`store.reconcile()`), not scheduled.
3. Gallery trimming and identity lookup cost one Chroma `get`/`query` per call; the top-100 window can hide a low-scoring identity behind a crowded one (affects runner-up margin only).
4. `ffmpeg`/`ffprobe` are runtime dependencies for clips (not covered by the GStreamer invariant; ingestion/decoding is still GStreamer). Clips are CPU-encoded (libx264 veryfast).
5. No retention/rotation/deletion policy for clips, tracks or identities; `gallery_ttl_s` still unimplemented. Single writer only (one process).
6. Found and fixed: `IdentityResolver(gallery=empty)` silently replaced a persistent gallery with an in-memory one (`gallery or ...` with falsy `__len__`).
7. Frame timestamps are wall-clock capture times (Phase 1 limitation), not stream PTS.

# PHASE 4 — QUERY UNDERSTANDING (COMPLETE pending acceptance; verified 2026-10-08)
Boundary: User -> (terminal) Chat -> Local LLM/NLU -> Referent Resolver -> Clarification Engine -> `ResolvedQueryContext`. STOPS before the Query Planner.
## LLM / model configuration (`config/query.yaml`, `godseye/query/config.py`)
Ollama 0.32.4 on loopback only (non-local `ollama_url` is rejected: no cloud LLM in the query path). Model `qwen3:8b` (DEVIATION from Llama 3 8B, user-approved), temperature 0, `num_ctx` 4096, `think: false`, JSON-schema constrained output (`format`), `verify()` at startup fails loudly if Ollama/the model is missing. Pydantic's schema is flattened for Ollama's grammar compiler (`llm.grammar_schema`: $refs inlined, validation-only keywords dropped; pydantic re-validates everything strictly afterwards). Measured once: ~2-8 s per LLM call; with qwen3:8b loaded the GPU shows ~6 GB used, so Track A models + the LLM together do NOT comfortably fit in 8 GB VRAM (run them in separate processes/times, or let Ollama offload). Semantic matching uses the OWL-ViT text tower (same CLIP space as the detector), on CPU by default.
## Query schema (`schema.py`)
`ParsedQuery` (LLM output, `extra=forbid`, length/range limits): intent {find, locate, track_path, count, when_seen, unknown}; object {label (user's noun), attributes}; location_text; camera_mentions; time (TimeSpec: none|last|day|between + amount/unit/day/part_of_day/clock); referent {none|pronoun|named|demonstrative, text}; spatial {none|left|right|top|bottom|center|in_zone|near_zone, zone_text}. `ResolvedQueryContext` (output): original_query (preserved), intent, object_label (canonical, one of `cfg.objects`), attributes, camera_ids (verified to exist), time_start/time_end (epoch), global_id or track_ref (camera, stream_epoch, track_id), regions (normalised rects, per camera), provenance (slot, mention, method exact|learned|fuzzy|semantic|llm|user|context, target, score).
Time arithmetic is done by code (`timeres.py`), not the LLM: qwen3:8b answered "yesterday" with the wrong year. The LLM only extracts a TimeSpec. Future or unparseable times trigger a clarification.
## Resolver (`resolver.py`, `understand.py`)
Mentions (object, camera, identity) resolve in order: exact (normalised + learned aliases + configured synonyms) -> fuzzy (difflib >= .85 and clear margin) -> semantic (CLIP text cosine >= .55, margin >= .10; objects only) -> LLM choice among the candidates (index validated against the offered list; anything else discarded). Ambiguous/unresolved -> clarification. Cameras: candidates come from Track A's `cameras` table (read-only). Pronouns/demonstratives resolve ONLY from `Session.focus` (recent result entities, filtered by person/vehicle): one candidate -> `context`, several -> multiple choice, none -> point-and-click. Named referents: raw `person-xxxxxxxxxxxx` ids (must exist) or learned identity aliases, else point-and-click. LLM-resolved items are marked `method=llm` in provenance (a UI should show "I read X as Y").
## Clarification (`Clarification`, `QueryUnderstanding.respond`)
Kinds: `multiple_choice` (object, ambiguous referent), `camera_grid` (camera), `point_and_click` (referent: answer = camera_id, x, y normalised, ts, frame_w, frame_h -> looked up in observations; overlapping objects -> follow-up multiple choice; untracked identity -> `track_ref`), `region_selection` (zone: normalised rectangle, corners in any order, camera must match), `free_text` (time/subject/zone; question phrased by the LLM `clarify()` with a template fallback, the answer is re-parsed by the NLU with a hint). Answers are validated; invalid ones re-ask; (slot, mention) asked > 3 times or > 6 invalid answers -> `Rejected`. Choices are applied to the SAME pending query (the original text is preserved).
## Knowledge store (`knowledge.py`, `<store_root>/knowledge.db`, Track B's own SQLite file)
`aliases(kind camera|object|identity|zone, alias normalised, scope, target, source, uses)`, `zones(camera_id, name, rect)`. Learned from user choices (camera/object/identity names, zones drawn by the user; pronouns are never learned). Verified to persist across instances. Track B never writes to `godseye.db` (opened `mode=ro` + `query_only`).
## Security boundary
LLM output is parsed into `ParsedQuery` (strict, inert data). Every value reaching the context is matched against Track A's cameras/identities, the object vocabulary, the knowledge store or computed by code. SQL is parameterised everywhere; there is no tool-calling, shell, filesystem or network use derived from LLM output; user text only travels in the LLM `user` role. A test scans `godseye/query` for execution primitives (`subprocess`, `eval/exec`, `shutil`, deletes, pickle, file writes).
## Interfaces / usage
`build(load_config("config/query.yaml"))` -> `QueryUnderstanding`; `handle(session, text)` and `respond(session, cid, answer)` return `Resolved | NeedsClarification | Rejected`. `python -m godseye.query.chat` (needs `data/store/godseye.db` from Track A; point/region questions need a graphical client and are only described in the terminal).
## PHASE 4 TESTS
`tests/test_query_unit.py` (21; scripted LLM double only to control malformed/hostile output; real SQLite with the real schema): time arithmetic, schema strictness, local-only config, learned mappings persist, read-only Track A view, resolver stages (fuzzy must not guess between near-equals; LLM choice limited to offered candidates), normal/incomplete/unknown/oversized queries, camera ambiguity + learned mapping reused, missing referent -> click -> identity (+ learned alias), untracked click -> track_ref, ambiguous pronoun -> choices -> context reuse, zone selection learned, future time, loop termination, malformed LLM output (never crashes, one retry recovers), hostile LLM output inert, only verified values in the context, no execution primitives.
`tests/test_query_real.py` (10; real qwen3:8b + real OWL-ViT text embedder; skipped if Ollama/model missing): full query parse incl. clock-based time, count + relative time, semantic object resolution (pedestrian/lorry/SUV), pronoun from context, missing referent, unknown camera -> clarification -> learned across instances, prompt-injection attempts (DB untouched, only verified values), non-surveillance/gibberish handled.
## PHASE 4 LIMITATIONS / ISSUES
1. Deviation: qwen3:8b instead of Llama 3 8B (user-approved). Behaviour with Llama 3 is untested. LLM output varies between prompts; the real-LLM tests assert safe/verified output and key fields, not exact phrasing. qwen3:8b sometimes drops a clause (e.g. omitted "this morning" before an example was added to the prompt); there is no automatic check that every part of the question was captured.
2. `method=llm` camera/object resolutions are accepted without confirmation (e.g. "north parking entrance" -> `north_gate` was accepted by the LLM). Wrong guesses are possible; provenance exposes them. A stricter policy (always confirm LLM-resolved cameras) is a one-line change if wanted.
3. Generic words resolve to one class ("vehicle" -> car by CLIP margin); no "any vehicle" semantics yet (Phase 5 planner decision). Attribute descriptions ("red jacket") are carried as text only: visual attribute search is Phase 5.
4. Cameras have no human-friendly names in Track A: names are learned from clarifications. The camera grid lists ids only (no thumbnails); point-and-click and region selection need a graphical client that does not exist yet (only the terminal chat). The "Chat UI" is a terminal loop, not a web UI.
5. Time zones: "local" system zone; no DST/zone edge-case tests; 'night' spans 21:00-06:00 (next day).
6. Single-user in-process sessions; `Session.focus` is filled only by `remember()` (Phase 5 must call it) or by a resolved referent.
7. Intent is only extracted here; whether an intent can be answered with the stored data is a Phase 5 concern.
## PHASE 4 DECISIONS
Code (not the LLM) does calendar arithmetic; LLM output is schema-constrained AND strictly re-validated; resolver cascade never jumps to the LLM first; semantic stage reuses the detector's CLIP text tower instead of a new embedding model; knowledge lives in Track B's own SQLite file, Track A's DB is read-only to Track B; clarification answers are validated and learned only from explicit user choices.

# PHASE 5 — PLANNING, RETRIEVAL, PATHS, RESPONSES (COMPLETE pending acceptance; verified 2026-10-08)
Boundary: `ResolvedQueryContext` -> Query Planner -> Retrieval Engine -> Path Reconstructor -> Response Builder -> `Answer`. Code: `godseye/answer/`; config `config/answer.yaml`, `config/topology.example.yaml`; entry `AnswerEngine.run(ctx, session=None) -> Answer`; the terminal chat (`python -m godseye.query.chat`) now prints `[STATUS]` + answer.
## QueryPlan (`plan.py`)
Strict pydantic (`extra=forbid`): operation (allow-list: find_tracks, count_tracks, locate_latest, when_seen, path), semantic_query, camera_filter, time_filter, spatial_filter (normalised regions), object_filter (label + reid_class), identity_filter (global_id XOR track_ref), ranking_strategy (recency|confidence|duration|semantic), reconstruction_required, limit, min_confidence, notes. `make_plan` is DETERMINISTIC (intent -> operation; attributes -> semantic text); the LLM does not construct plans (spec says it may; not used, so nothing the model says can widen what runs). `validate_plan` runs before any retrieval: cameras exist and are path-safe, ids well-formed AND present, time ordered, label in vocabulary, limit <= max_limit, op/flag consistency, a subject is present. Invalid plan -> `SYSTEM_ERROR` (never a guessed answer).
## Retrieval (`retrieval.py`)
Read-only SQLite (`mode=ro`) + ChromaDB `track_clip` (`get_collection` only). SQL assembled from constant fragments with bound parameters; ORDER BY from an allow-list; candidate cap 2000 (flagged PARTIAL when hit). Track A schema is now v2 (migrated automatically by the Track A writer): added indexes `tracks(start_ts)`, `tracks(label,start_ts)`, `tracks(reid_class,start_ts)`; a test asserts `EXPLAIN QUERY PLAN` shows no table scan for the main queries. Semantic ranking: query text embedded with the OWL-ViT text tower (the same CLIP space as the stored reused detection embeddings), then Chroma `query(ids=<SQL candidate ids>)` cosine; tracks with no stored embedding are reported, not ranked. Spatial filter: observation box centres vs normalised regions using frame size taken from the camera's latest clip (cameras without any clip cannot be evaluated -> excluded + PARTIAL). Evidence per track = highest-confidence observation with clip path/frame; the clip file is checked on disk at answer time (`ok | not_recorded | missing_file | marked_missing | marked_corrupt`). Coverage: per camera, fraction of the requested window inside `cameras.first_seen..last_seen`.
## Path model (`path.py`, `topology.py`)
`Visit` = OBSERVED (tracks of one global_id on one camera, merged if closer than `merge_gap_s`), with first/last observed position and best evidence. `Transition` = always INFERRED (identity not seen in between) with a verdict: `plausible` | `unverified` (no topology / pair not checkable) | `impossible_too_fast` (gap < shortest route - tolerance) | `impossible_no_route` | `impossible_overlap` (seen on two non-overlapping cameras at once) | `long_gap` (> max_gap_s). Impossible/long-gap transitions split the path into episodes with no link asserted, and impossible ones add a warning that the global_id may cover two objects. Topology = YAML links with min/max travel seconds (+overlap, directed); shortest route by Dijkstra on min travel. `config/topology.example.yaml` holds PLACEHOLDER numbers, not a real site topology; with none configured every cross-camera transition is `unverified` and the answer says so.
## Response behaviour (`response.py`)
Statuses: `OK`, `NO_RESULT` (valid query, nothing stored; text says this does not prove absence), `PARTIAL_EVIDENCE` (missing/marked clips, low detection confidence < `low_confidence`, unverified/impossible transitions, capped or unevaluable filters, counts containing unidentified tracks), `AMBIGUOUS_RESULT` (locate/path for a class of object with several individuals -> lists candidate identities to choose from), `SYSTEM_ERROR` (invalid plan, store/Chroma/embedder failure; says no result was produced). Factual text comes only from templates filled with retrieved rows (camera, time, confidence, clip path+frame, identity, status). Counts report tracks, distinct identities and unidentified tracks (an upper bound on unique objects). `locate` says it is the last recorded sighting, not a live position. Optional LLM rewording (`llm_wording`, default OFF): accepted only if every number/id/camera token survives, none is added, OBSERVED/INFERRED label counts are unchanged, length is bounded and few new content words appear; otherwise the template text is used. After each answer the returned identities are added to `Session.focus`.
## PHASE 5 TESTS
`tests/test_answer.py` (35; REAL Track A store built by `tests/support/scenario.py` with the real EvidenceStore/ffmpeg/Chroma + real OWL-ViT text tower; LLM only as a scripted double in the wording tests): query->plan for every intent, invalid-plan rejection (hostile ids/cameras/labels), plan invariants, metadata filters (camera/time/label/identity/track_ref), spatial filter, no-table-scan EXPLAIN check, bound-parameter check with a hostile label, semantic ranking (red vs blue car), semantic unavailable -> SYSTEM_ERROR, grounded OK answer with an existing clip, NO_RESULT, coverage warning, low confidence + missing clip -> PARTIAL, ambiguous results, count semantics, when_seen merging, plausible path (OBSERVED vs INFERRED), impossible_overlap / too_fast / long_gap, unverified without topology, path filters, topology loader/routing, LLM-wording rejection cases, every id/camera in answers exists in the store, closed DB -> SYSTEM_ERROR, read-only engine + session memory, config + no-execution-primitive scan.
`tests/test_answer_real.py` (5; real video -> real perception -> real store -> real qwen3:8b understanding -> engine): real embeddings are closer to "a person" than "a bus"/"a cat", count/find/path on real data with existing decodable clips, real question -> grounded answer, real LLM rewording adds no data tokens.
## PHASE 5 LIMITATIONS / ISSUES
1. Attribute search ("red jacket") is approximate: CLIP cosine of the stored per-track mean embedding vs a text embedding, uncalibrated, relative only. Verified on real data only that person embeddings align with "person" more than "bus"/"cat"; colour/clothing attributes were verified only with controlled text-derived vectors, NOT on real images.
2. Topology must be measured on site; the shipped example is a placeholder. Camera coverage uses only `cameras.first_seen/last_seen` (gaps inside that span are not detected).
3. Spatial filters need at least one stored clip per camera (frame size comes from clips); stored positions are pixels of the original frame.
4. Concurrency: Track B opens Track A's SQLite read-only (WAL, fine) but opens the ChromaDB persistent directory from a second process; multi-process Chroma access while Track A writes is NOT verified (tests are sequential). Phase 6 should use a Chroma server or snapshot.
5. The Track A schema migrated to v2 (indexes); an older DB is migrated when the Track A writer opens it, Track B only warns on v1.
6. LLM wording is a convenience; the check blocks changed data, labels and mostly new words but cannot guarantee a harmless qualitative phrase is absent, so it is off by default. The planner does not use the LLM.
7. Counts are track counts; same-person re-detections without identity inflate them (stated in every count answer). Identity quality is limited by the Phase 2 Re-ID caveats (false splits more likely than false merges).
8. Performance not measured at scale (candidate cap 2000, per-track subqueries). `locate_latest` without an identity returns recent sightings of different individuals as AMBIGUOUS rather than choosing one.
9. Bugs found by Phase 5 work and fixed: Phase 3 store raised a Chroma DuplicateIDError when a track was live and ended in the same frame (ids now deduped, regression test); the LLM-wording check first accepted a fabricated sentence and a relabelled INFERRED->OBSERVED step (check tightened, tests added).
## PHASE 5 DECISIONS
Deterministic planner; allow-listed operations; every transition between observed visits is INFERRED and judged against an explicit topology (no topology => unverified, never silently plausible); statuses are mutually distinct and SYSTEM_ERROR never masquerades as NO_RESULT; index migration instead of accepting table scans; LLM wording opt-in with fact-preservation checks.

# PHASE 6 — INTEGRATION & HARDENING (COMPLETE pending acceptance; verified 2026-10-08)
Details: `docs/PERFORMANCE.md`, `docs/SECURITY.md`, `docs/CLUSTER.md`. Locked dependencies: `requirements.lock` (Python 3.12 venv), `requirements-ray.lock` (Python 3.11 `.venv-ray`).
## Integration (`godseye/runtime/`)
* `TrackAService` (single consumer thread): `IngestManager.get_frame` -> `PerceptionPipeline.process` -> `EvidenceStore.write`. Entry: `python -m godseye.runtime trackA` (reads `config/runtime.yaml` + the sub-configs). `build_track_a/build_track_b` wire from config; Track A and B can run in separate processes.
* `QueryService` (Track B): question -> NLU -> resolver/clarification -> planner -> retrieval -> answer, with per-question timeout (`query_timeout_s`), bounded concurrency (`query_max_inflight`, excess = `busy`), bounded expiring sessions, exceptions converted to `error` results. `Failed` (LLM unavailable) is distinct from `Rejected` (question not understood).
* Web chat (`python -m godseye.runtime web` -> `godseye/web/server.py`): loopback only, token login link printed at start, clarification buttons, point-and-click / area selection on stored clip frames, evidence video with seek. See SECURITY for the model. The JavaScript UI is covered by server-side tests only: it was NOT run in a browser here.
* Terminal chat `python -m godseye.query.chat` now prints `[STATUS]` + the grounded answer.
## Propagation verified (real components, `tests/test_integration.py::test_end_to_end_propagation`)
camera_id (tracks, observations, clips, Chroma metadata), capture timestamps (inside the run window), `track_id` (unique per camera/epoch, camera-local), `global_id` (only on identified tracks, present in `identities`), embeddings (one 512-d `track_clip` vector per track), clip paths (file exists, ffprobe frame count equals the row, observation `clip_id` set only for published clips); Track B counts equal SQL counts.
## Failure isolation (all tested with the real stack unless noted)
* Dead camera (refused RTSP) does not affect the good camera, the service or queries. Model failure on one camera -> that camera is quarantined after `quarantine_after` failures (frames counted), others continue. GPU OOM (VRAM exhausted on purpose) -> frame dropped, counted, recovers when memory returns. Re-ID exceptions -> tracks emitted without a new identity (`identity_errors`).
* Storage failure (SQLite error, ChromaDB error, SQLite lock contention) never stops ingestion or analytics: circuit breaker (`store_fail_threshold`, exponential back-off), every non-persisted frame counted (`frames_not_persisted`), track endings carried to the next successful write; after the run no track is left open and `PRAGMA integrity_check` is ok. Sources that end/fail get their tracks closed and persisted (retried).
* Query failure (LLM down, timeout, busy, internal error) does not slow or stop ingestion (measured). Missing or damaged clips degrade answers to PARTIAL_EVIDENCE (`corrupt_file` = size differs from the recorded size) and the web server refuses them (410).
* OS write failures (RLIMIT_FSIZE: EFBIG, the nearest no-root stand-in for a full disk): writer survives, SQLite intact, no dangling clip or vector references after restart. A genuine ENOSPC on a full filesystem was NOT tested.
* LLM: `LLMUnavailable` vs invalid output; optional `llm_fallback_url` (loopback) used loudly when the primary fails; malformed output covered in Phase 4 tests.
* Camera reconnect: a `stream_epoch` change ends all tracks with `stream_restart` and persists them (integration test with real frames); GStreamer-level reconnect/stall behaviour is the Phase 1 test set.
## Cluster (Ray) — see `docs/CLUSTER.md`
LLM (qwen3:8b via Ollama) runs on the peer RTX 3050 through `tools/ray_llm_gateway.py` (Python 3.11 venv, loopback gateway) + `godseye_remote/llm_proxy.py` (stdlib-only Ray actor). Everything else stays on this laptop (measured: VRAM was not the limit for perception, 2.1 GiB). Usage: `.venv-ray/bin/python tools/ray_llm_gateway.py` and `config/query.yaml: ollama_url: http://127.0.0.1:11435` (`llm_fallback_url: http://127.0.0.1:11434` recommended). Default config still points at the local Ollama; the gateway is opt-in. Deviation to note: the LLM is then a LAN peer, not the same machine.
## Measurements (this machine; full tables in `docs/PERFORMANCE.md`)
1 camera: 5.0 fps processed, e2e 29/43 ms (p50/p95), detect 10.3 ms, Re-ID 13 ms, store 2.8 ms, GPU util 8%. 4 cameras: 20.0 fps, 0 drops. 8 cameras: **3.3 fps (swap thrash)** before fixes -> **27.7-29.1 fps** after limiting x264 encoders (the measured cause: host RAM) and fp16 Re-ID. Capacity of the single pipeline thread is about 29 sampled fps. LLM on this GPU vs on the peer under 8-camera load: 24.6 vs 28.6 fps, detect p95 48.8 vs 15.7 ms, VRAM peak 7.2 vs 1.8 GiB; question latency 1.2 s vs 1.9 s median.
## Security audit — see `docs/SECURITY.md`
Fixed: camera passwords leaking through GStreamer error text into logs/health; venv not self-contained (`python-dotenv`). Open: no TLS/per-user auth on the web chat, no encryption at rest, no per-identity deletion, weights checksums not verified (`torch.load(weights_only=False)` on pinned checkpoints), unauthenticated Ray cluster (existing infrastructure).
## PHASE 6 TESTS (47 new)
`test_runtime_unit.py` (5, doubles, no GPU), `test_integration.py` (11, real stack + fault injection), `test_cluster.py` (8; 4 need the live Ray cluster and skip otherwise), `test_web.py` (11), `test_security.py` (9), plus in earlier files: Re-ID fp16 parity, OS write-failure, `purge_clips`, concurrent live+ended ids. Test-suite note: GPU modules unload the local Ollama model first (`conftest.py`); without that, loading perception next to the 6 GB local LLM runs out of 8 GB VRAM (reproduced).
## PHASE 6 LIMITATIONS / ISSUES
1. No soak test (hours/days), no real RTSP cameras, no hardware decode, no more than 8 simulated cameras, one video source (looped file). Throughput numbers are single runs on a loaded laptop (about 10 GB of RAM used by other apps, swap full before the runs).
2. Track A is one pipeline thread: ~29 sampled fps. More cameras need multiple worker processes (not built). Cross-process Track A + Track B access to ChromaDB while the writer runs is tested only sequentially and via one process; Chroma's multi-process behaviour is unverified (use a snapshot or Chroma server if Track B runs in its own process under load). SQLite WAL readers are fine.
3. Ray: cluster unauthenticated; a Python 3.11 venv and gateway process must be running; the peer must keep Ollama and `qwen3:8b`. Perception was not distributed (justified above); the 3050 is otherwise unused.
4. Web UI: single shared token, HTTP (no TLS), global rate limit; JS never executed in a browser during this work.
5. `EvidenceStore` is single-writer/process (lock file). Retention is manual (`purge_clips`); metadata and embeddings are kept indefinitely.
6. Found and fixed in this phase: cross-thread SQLite use (store/knowledge/entities now `check_same_thread=False` + locks), `EvidenceStore` constructed on one thread and used on another, storage config `x264` memory, Retrieval clip-corruption check, credentials in error text, schema/venv issues listed above.
## PHASE 6 DECISIONS
Isolate by catching at the service boundary and counting (never silently); circuit breaker + carried track endings instead of dropping endings; encoder memory limits chosen from measurement; fp16 Re-ID with a parity test; LLM placed on the peer via a loopback gateway rather than distributing perception; web UI is local-only by construction; security checks are executable tests.

# PHASE 7 — FINAL ARCHITECTURE CONFORMANCE AUDIT (2026-10-08)
No functionality added. Re-ran the full suite: **222 passed, 1 skipped (RTMP, no server) in 13m42s**. Git state: Phase 6 files are staged but NOT committed.
## Downgrade audit (grep of `godseye/`, `godseye_remote/`, `tools/`; ~6.7k LOC) — no substitution found
* No YOLO/ultralytics (only comments). `cv2` is used for resize/MOG2/privacy masks/letterbox and the web server's decode of STORED clips; ingestion/decoding is GStreamer. `cv2.VideoCapture` appears once, in `web/server.py` (reading stored evidence, not a camera).
* Tracking = vendored ByteTrack; Re-ID = FastReID checkpoint; vectors = ChromaDB `PersistentClient` (no `EphemeralClient`/in-memory path in production); structured data = SQLite. `json` is used only for columns/HTTP/LLM payloads, not as a store.
* NLU = local LLM; the only `re` use in `godseye/query` is inside prompts' text, no keyword intent matcher. Planner is deterministic by design (documented), not a hidden hardcoded retrieval.
* All Fake/Mock/Scripted classes are defined under `tests/`; none is imported by `godseye/`. `factory.py` wires the real classes only. The few "dummy/stub" hits in `godseye/` are warm-up input and a flush placeholder frame, not fake inference.
## Component table (exists / implemented / right tech / tested)
CCTV sources, RTSP, HLS, GStreamer, sampling, preprocessing: yes / yes / yes / tested on synthetic streams via real RTSP+HLS servers. **RTMP: wired, NOT verified.**
NanoOWL, embedding reuse, ByteTrack, FastReID, ChromaDB, SQLite, clip storage: yes / yes / yes / tested (real models on one non-CCTV video; Market-1501 for Re-ID). Vehicle Re-ID unvalidated.
Knowledge store, local LLM (qwen3:8b, approved deviation), referent resolver, clarification engine, planner, retrieval, path reconstruction, response builder: yes / yes / yes / tested (unit + real LLM).
## Verdicts
* Architecture Conformance: **PASS** (deviations: qwen3:8b instead of Llama 3, user-approved; `ffmpeg` for clip encoding; `torch.load(weights_only=False)` on pinned checkpoints).
* Implementation Completeness: **PASS with gaps** (below).
* Testing: **PASS for what exists, thin for production** (no real cameras, no soak, no browser run of the web JS, no real ENOSPC, no RTMP).
* Security: **FAIL for any non-local deployment** (no TLS/per-user auth, no encryption at rest, no per-identity deletion, weights not checksum-verified, unauthenticated Ray). Acceptable for loopback single-user use only.
* Performance: **PASS for <= 8 simulated cameras (~29 sampled fps single thread)**; unmeasured beyond that and over time.
* Production Readiness: **NOT READY**.
## Blockers (need user input)
1. Real cameras and a soak run (hours/days); RTMP server or camera to verify RTMP.
2. Re-ID thresholds uncalibrated; need real-site footage. Vehicle Re-ID has no validation data.
3. Camera topology is a placeholder; path verdicts are `unverified` until measured.
4. Auth/TLS decision if the web chat is ever exposed beyond loopback.
## Technical debt
* Track A is one pipeline thread (~29 fps); multi-process needed for more cameras. ChromaDB multi-process access while the writer runs is unverified.
* No retention policy beyond manual `purge_clips`; `gallery_ttl_s` unimplemented; reconcile is manual.
* Timestamps are wall-clock, not stream PTS. 640 vs 768 input spec conflict (detector uses `original_bgr`).
* LLM-resolved camera/object matches are accepted without confirmation; attribute search (colour/clothing) unvalidated on real images.
* System GStreamer lacks H.264 plugins; relies on `.gst-local/` extracted debs (not portable).
* Test-count figures in per-phase notes are approximate; the pytest total above is authoritative.
## NEXT ACTION
Commit Phase 6/7 (user to say so), then supply the blocker inputs above. STOP per the Phase 7 contract.

# FRONTEND CONSOLE (branch `frontend`, after Phase 7; verified 2026-10-08)
React + Vite + TypeScript console (`frontend/`, built to `frontend/dist`) is served by `python -m godseye.runtime web` together with the `/api` contract of `frontend/README.md`, implemented in `godseye/web/frontend_api.py` on top of the unchanged Track B (QueryService). The old single-file chat remains at `/legacy` (its camera endpoint moved to `/api/legacy/cameras`). Details and mapping: `docs/FRONTEND.md`.
* Security model unchanged and extended to the new routes: loopback only, token cookie, Host/Origin + `X-Requested-With` on writes (the frontend client sends it), media by `clip_id`, static files confined to `frontend/dist`. External font CDN removed from the frontend.
* Not provided (501, hidden in the UI via `health.capabilities`): video upload/links, backend restart. No `site.layout` (no invented floor plan), no progress/candidate events. Clarification kinds `choice`/`text` added to the UI for multiple-choice and free-text questions.
* Verified: `tests/test_web_frontend.py` (6) + full suite **228 passed, 1 skipped** (RTMP); frontend `npm test` 5 passed, `tsc -b` clean; driven in headless Chromium against the live server (login, camera wall, query, evidence video with boxes, clarification by tile pick learned into memory, memory panel): no console errors. This supersedes "web UI JavaScript not exercised in a browser" for the new console only (the legacy page was not browser-tested).
* Limits: tile playhead uses a linear wall<->video map (approximate for sampled capture; evidence player/boxes are exact); only each camera's newest clip is on the wall; no TLS; all Phase 7 verdicts (NOT READY) stand.
