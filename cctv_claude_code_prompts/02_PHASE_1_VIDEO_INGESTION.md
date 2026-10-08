# PHASE 1 — COMPLETE VIDEO INGESTION PIPELINE

Read `CLAUDE.md` and `00_MASTER_IMPLEMENTATION_CONTRACT.md`.

## Boundary

CCTV  
→ RTSP / RTMP / HLS  
→ GStreamer  
→ Frame Sampling  
→ Preprocessing  
→ Preprocessed Frame Output

## Stream Layer

Implement:

- multiple independently configured cameras
- RTSP / RTMP / HLS as applicable
- camera lifecycle
- connection/reconnection
- one pipeline per camera
- camera_id propagation
- timestamps
- bounded buffering
- backpressure
- graceful shutdown

## GStreamer

Use the intended architecture:

`rtspsrc → decodebin → videoconvert → appsink`

Adapt only when required by actual protocol or hardware.

Do NOT replace GStreamer with OpenCV VideoCapture.

Where hardware acceleration is supported, expose it through configuration and verify actual runtime capability.

Do not falsely claim acceleration.

## Frame Sampling

Implement configurable:

- base sampling, architecture target 2 FPS
- burst sampling, architecture target 5 FPS
- motion/change gating
- frame-difference / MOG2-style gating where appropriate
- redundant frame dropping
- bounded queues

Do not hardcode production values.

## Preprocessing

Implement:

- resize
- aspect-ratio preservation
- letterboxing
- 640×640 target
- [0,1] normalization
- coordinate transformation tracking
- optional privacy preprocessing

Preserve original-frame information needed for later evidence/clip generation.

## Reliability

Handle:

- unavailable camera
- invalid stream
- interruption
- reconnect
- malformed frames
- queue pressure
- graceful shutdown

One camera failure must not crash unrelated cameras.

## Testing

Test:

- multiple cameras
- valid/invalid streams
- disconnect/reconnect
- sampling
- burst mode
- motion gating
- preprocessing
- coordinate transforms
- queue pressure
- shutdown

Use test fixtures only where necessary.

Never use fake frames in production.

## Do NOT Implement

- NanoOwl
- ByteTrack
- FastReID
- ChromaDB persistence
- query system
- LLM

Only create the output contract required by Phase 2.

## Completion

Before finishing:

1. Verify implementation.
2. Run tests.
3. Audit architecture.
4. Update `CLAUDE.md`.
5. Record dependencies.
6. Record configuration.
7. Record interfaces.
8. Record issues and limitations.
9. Record exact next action.

STOP.
