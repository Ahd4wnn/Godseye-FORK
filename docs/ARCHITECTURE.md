# Architecture

Authoritative pipeline definitions: `cctv_claude_code_prompts/00_MASTER_IMPLEMENTATION_CONTRACT.md`. Summary:

**Track A** CCTV → RTSP/RTMP/HLS → GStreamer → Frame Sampling → Preprocessing → NanoOwl → Embedding Reuse → ByteTrack → FastReID → ChromaDB → SQLite → Clip Storage
**Track B** User → Chat UI → Local LLM/NLU → Referent Resolver → Clarification Engine → Resolved Query Context → Query Planner → Retrieval Engine → Path Reconstructor → Response Builder → Grounded Answer + Evidence

## Implemented: Phase 1 (ingestion boundary)

```
CameraConfig ──► CameraPipeline (1 thread + 1 GStreamer pipeline per camera)
                   src ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink
                   │ pull (100 ms poll; bus ERROR/EOS; connect + stall watchdogs; backoff reconnect)
                   ▼
                 FrameSampler  (base/burst rate, MOG2 gating, redundancy drop, keepalive)
                   ▼ (only emitted frames are copied/preprocessed)
                 Preprocessor  (privacy mask → letterbox 640 → RGB float32 [0,1])
                   ▼
                 FrameBus (bounded deque per camera, drop_oldest|drop_newest, round-robin get)
                   ▼
                 PreprocessedFrame  ──►  Phase 2 (NanoOwl)  [not implemented]
```
Source elements: rtsp=`rtspsrc`, rtmp=`rtmp2src`, hls=`souphttpsrc`+`decodebin3` (hlsdemux2 needs a streams-aware parent), file=`filesrc` (dev/replay, clock-paced).
Module map: `godseye/gst_env.py` (GStreamer bootstrap), `godseye/ingest/{config,types,preprocess,sampler,camera,manager}.py`.
Boundaries: ingestion knows nothing about detection/tracking/storage; the only output is `PreprocessedFrame` via `IngestManager.get_frame()`.
