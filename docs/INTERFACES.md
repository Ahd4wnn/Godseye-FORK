# Interfaces (verified by tests)

## Phase 1 → Phase 2: `PreprocessedFrame` (`godseye/ingest/types.py`, frozen dataclass)
| field | type | meaning |
|---|---|---|
| camera_id | str | from config |
| frame_seq | int | index among *sampled* frames of this camera |
| source_seq | int | index among all decoded frames (per process) |
| stream_epoch | int | +1 on every (re)connect; trackers must reset when it changes |
| capture_ts | float | wall-clock epoch seconds at pull |
| pts_ns | int \| None | GStreamer buffer PTS |
| image | np.float32 (T,T,3) | RGB, [0,1], C-contiguous, letterboxed (T=`target_size`, default 640) |
| transform | LetterboxTransform | `to_model(xyxy)` / `to_original(xyxy)` (clipped) |
| original_bgr | np.uint8 (H,W,3) \| None | full-res frame for evidence/clips (copy; safe to retain) |
| sample_reason | str | first \| base \| burst \| keepalive |
| motion_score | float \| None | MOG2 foreground fraction (None if gating off) |
| privacy_masked | bool | masks applied to `image` (and `original_bgr` if `privacy_apply_to_original`) |

`image` is NOT the OWL-ViT/NanoOwl native input (CLIP mean/std, native resolution): Phase 2 must verify NanoOwl's real input requirements and adapt/re-derive from `original_bgr` if needed. NOT VERIFIED.

## `IngestManager`
`start()` (raises `IngestEnvironmentError` if required GStreamer elements missing) · `add_camera(cfg)` · `remove_camera(id)` · `get_frame(timeout) -> PreprocessedFrame|None` (fair round-robin) · `stats() -> {camera_id: CameraStats}` · `stop(timeout) -> bool`.

## `CameraStats`
state (idle/connecting/streaming/reconnecting/ended/failed/stopped), decoder (actual GStreamer decoder factory names), stream_epoch, reconnects, frames_decoded, frames_sampled, dropped_rate, dropped_redundant, dropped_queue, malformed, last_error, last_frame_ts.

## Error semantics
Retryable stream errors → backoff reconnect (state RECONNECTING). `FatalCameraError` (e.g. `hw_accel` requested but unavailable or not actually used) → that camera FAILED, others unaffected, no silent software fallback. `max_attempts` exhausted → FAILED. Malformed buffers/frames → counted in `malformed`, dropped, never published.
