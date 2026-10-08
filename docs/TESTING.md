# Testing
Run: `.venv/bin/python -m pytest tests -q` (≈100 s; needs ffmpeg + the local GStreamer plugins in `.gst-local`).
- Unit: `test_config.py`, `test_preprocess.py`, `test_sampler.py` (synthetic arrays + explicit timestamps, isolated), `test_bus.py`.
- Integration (`-m integration`): `test_ingest_integration.py` ingests *real* H.264 (ffmpeg-encoded clips) over file, real RTSP (GstRtspServer subprocess), real HLS (ffmpeg + http.server). Fixtures live only in `tests/support/`.
- Covered: multi-camera, fairness, invalid/missing/404 streams, disconnect→reconnect (new `stream_epoch`), stall detection, max_attempts, hw_accel failure loudness, queue pressure, loop_file, runtime add/remove, shutdown (<5 s, no threads left), environment-failure loudness, malformed buffers (fake buffer double, unit scope only).
- NOT covered: RTMP (needs a real RTMP server; `GODSEYE_TEST_RTMP_URL`), real IP cameras, hardware decode (no NVDEC/VA decoder installed), long-duration soak, performance/latency.
Results: see CLAUDE.md → TEST STATUS.
