# PHASE 6 — END-TO-END INTEGRATION & PRODUCTION HARDENING

Read `CLAUDE.md` completely before proceeding.

## Integrate both tracks

### Track A

CCTV  
→ GStreamer  
→ Sampling  
→ Preprocessing  
→ NanoOwl  
→ Embedding Reuse  
→ ByteTrack  
→ FastReID  
→ ChromaDB / SQLite / Clips

### Track B

User  
→ NLU  
→ Referent Resolution  
→ Clarification  
→ Query Planner  
→ Retrieval  
→ Path Reconstruction  
→ Response

## Integration Requirements

Preserve all verified interfaces.

Do not rewrite working components merely for convenience.

Verify correct propagation of:

- camera_id
- timestamps
- track_id
- global_id
- embeddings
- clip_path

## Failure Isolation

Verify:

- failed camera does not crash query services
- storage failure does not silently corrupt analytics
- query failure does not stop camera ingestion
- one camera failure does not affect unrelated cameras

## Performance Instrumentation

Measure actual:

- ingest FPS
- sampled FPS
- inference latency
- tracking latency
- Re-ID latency
- storage latency
- query latency
- end-to-end latency
- GPU utilization
- CPU utilization
- RAM/VRAM
- queue depth
- dropped frames

Optimize measured bottlenecks only.

Do not remove architectural components for benchmark gains.

## Security Audit

Review:

- SQL injection
- prompt injection
- filesystem traversal
- credentials
- LLM tool execution
- media access
- API security
- PII exposure

## Reliability

Test:

- camera disconnect/reconnect
- GStreamer failure
- model failure
- GPU memory pressure
- ChromaDB failure
- SQLite locking/failure
- disk full
- clip write failure
- LLM unavailable
- malformed LLM output
- query timeout
- corrupt media

## Completion

Update `CLAUDE.md` with:

- integration status
- test results
- measurements
- security findings
- remaining blockers
- next phase

STOP.
