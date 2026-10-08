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
Phase 3 — Storage & evidence (ChromaDB, SQLite, clips): implemented and tested (see PHASE 3 at the end), awaiting user acceptance before Phase 4. Phase 2 and Phase 1 caveats remain (RTMP unverified; Re-ID cross-domain accuracy; vehicle Re-ID unvalidated; 640-vs-768 input decision).
Phase prompts live in `cctv_claude_code_prompts/` (master contract `00_*`, phases `01_`–`08_`). Docs required by the contract are in `docs/`.

# PHASE STATUS
- Phase 0 — Architecture & CLAUDE.md: COMPLETE
- Phase 1 — Video ingestion (GStreamer → sampling → preprocessing): COMPLETE except RTMP integration verification (BLOCKED on an RTMP server)
- Phase 2 — AI perception, tracking & cross-camera identity (NanoOwl, ByteTrack, FastReID): COMPLETE pending acceptance (vehicle Re-ID unvalidated; Re-ID thresholds uncalibrated for any real site)
- Phase 3 — Storage & evidence (ChromaDB, SQLite, clips): COMPLETE pending acceptance (retrieval NOT implemented by design; single-writer; clips need ffmpeg)
- Phase 4 — Query understanding: NOT STARTED
- Phase 5 — Retrieval, path reconstruction, response: NOT STARTED
- Phase 6 — Integration & hardening: NOT STARTED
- Phase 7 — Final audit: NOT STARTED

# IMPLEMENTED COMPONENTS
Verified by the tests listed in TEST STATUS:
- `godseye/gst_env.py` — GStreamer bootstrap (plugin paths, lib preload, private registry).
- `godseye/ingest/config.py` — validated YAML config (unknown keys rejected, `${ENV}` URL expansion, per-camera overrides).
- `godseye/ingest/camera.py` — per-camera GStreamer pipeline + worker thread: RTSP/HLS/file verified against real servers; RTMP element wired but NOT verified. Reconnect with backoff, connect/stall watchdogs, `hw_accel` fail-loud, malformed-buffer handling.
- `godseye/ingest/sampler.py` — base 2 fps / burst 5 fps (configurable), MOG2 gating, redundancy drop, keepalive.
- `godseye/ingest/preprocess.py` — privacy masks, letterbox 640, RGB float32 [0,1], coordinate transform with round-trip.
- `godseye/ingest/manager.py` — `IngestManager` + bounded `FrameBus`.
- `config/ingest.yaml` (ships with `cameras: []`), `docs/*.md`, `tests/`.
Phase 2 (`godseye/perception/`): see PHASE 2 below. Phase 3 (`godseye/storage/`): see PHASE 3 below.
NOT implemented: Phases 4–7 (no query system, LLM integration, UI, retrieval).

# TECHNOLOGY REQUIREMENTS

Required (never replace with weaker alternatives without explicit architectural approval):

- GStreamer — ingestion/decoding
- NanoOwl — detection
- ByteTrack — local tracking
- FastReID — cross-camera Re-ID
- ChromaDB — vector store
- SQLite — structured store
- Local LLM via Ollama (Llama 3 where specified) — query understanding/planning

Note: the Ollama models present on this machine are Qwen-family, not Llama 3 (see DEPENDENCIES). Which model Track B uses is an OPEN DECISION. Do not silently substitute.

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
Command: `.venv/bin/python -m pytest tests -q` (2026-10-08, after Phase 3): **103 passed, 1 skipped** in about 123 s (Phase 3 added 20) (skipped = RTMP integration test, needs `GODSEYE_TEST_RTMP_URL`). Phase 1 alone: 65 passed. Phase 2 tests: see PHASE 2 TESTS.
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
1. User accepts or rejects Phase 3 (and Phase 2 open items: input-size spec, Re-ID calibration data, vehicle Re-ID data).
2. Phase 4 (`05_*`): query understanding (local LLM, referent resolver, clarification). Phase 5 will read via the schema below; Phase 3 deliberately has no retrieval API except `VectorStore.search`.
3. Still open from Phase 1: RTMP integration verification.

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
