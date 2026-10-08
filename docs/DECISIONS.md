# Decisions
1. **GStreamer via PyGObject, programmatic property setting.** URL set with `set_property`, never interpolated into a launch string (untrusted input).
2. **appsink pulled with `emit("try-pull-sample")`.** The GstApp typelib is not installed; the action signal needs none.
3. **HLS uses `decodebin3`** (hlsdemux2 fails with plain decodebin: "Element requires a streams-aware context"). Protocol-driven adaptation allowed by the Phase 1 spec.
4. **Local GStreamer plugins without root.** The system lacks gst-libav/plugins-bad. `.gst-local/plugins` holds `libgstlibav`, `libgstrtmp2`, `libgstvideoparsersbad`, `libgstmpegtsdemux` extracted from the Ubuntu `.deb`s (same versions as installed GStreamer 1.24.2); `.gst-local/lib` holds `libgstcodecparsers`/`libgstmpegts`. `gst_env.init_gst` preloads libs (ctypes RTLD_GLOBAL), sets `GST_REGISTRY_FORK=no` (the forked scanner can't see preloaded libs) and a private registry. Replace with `apt install gstreamer1.0-libav gstreamer1.0-plugins-bad` when sudo is available, then drop `plugin_paths`/`preload_libs` from config.
5. **OpenCV is used only for resize/MOG2/colour conversion**, never for capture (`VideoCapture` not used).
6. **Sampler is a pure function of (frame, timestamp)** → deterministic tests. Motion evaluated at ≤ burst_fps; emit schedule resyncs when late; redundancy only suppresses non-burst candidates; keepalive guarantees liveness.
7. **Backpressure = drop at three levels**: appsink (max-buffers=2, drop), sampler, bounded FrameBus. Producers never block.
8. **`hw_accel` fails loudly**: rank bump of NVDEC/VA decoders, then verifies the decoder actually used after the first frame.
9. **Letterbox pad value 114 and RGB HWC float32** are config/contract choices, not model-verified (see INTERFACES).

## Phase 2 (details in CLAUDE.md, PHASE 2)
10. **TensorRT 11 engine via ONNX + Python API** (torch2trt/trtexec unusable on TRT 11); outputs wrapped in NanoOWL's own output type.
11. **Detector consumes `original_bgr` at NanoOWL's native 768 px / CLIP stats**, not the 640 letterbox (spec conflict, user decision pending).
12. **ByteTrack per Re-ID class group**; ids per camera; reset on `stream_epoch`.
13. **Identity merges are conservative**: threshold + margin + same-camera exclusion; ambiguity is reported, never merged.
14. **Never `tuple(trt.Dims)`**: libunwind from GStreamer turns TensorRT's internal C++ IndexError into an abort.

## Phase 3 (details in CLAUDE.md, PHASE 3)
15. **SQLite is the source of truth; Chroma holds vectors only.** Write order vectors -> media -> SQLite commit so crashes leave orphans, never dangling references; `recover()` + `reconcile()` handle orphans.
16. **One clip segment per camera, not per track**; observations point at (clip_id, clip_frame); published only after ffprobe validation + atomic rename.
17. **Clip failure degrades, metadata failure raises.** Never silent: counters, ERROR log, NULL clip refs.
18. **ffmpeg subprocess for H.264 evidence encoding** (OpenCV's writer has no H.264 here; system GStreamer has no x264enc).

## Phase 4 (details in CLAUDE.md, PHASE 4)
19. **qwen3:8b instead of Llama 3 8B** (only 8B installed; user-approved 2026-10-08); model name is config.
20. **Time arithmetic in code**; the LLM extracts a `TimeSpec` only.
21. **Resolver cascade exact -> fuzzy -> semantic (OWL-ViT text tower) -> LLM**; LLM may only pick from offered candidates.
22. **Track B owns `knowledge.db`; `godseye.db` is opened read-only.**
