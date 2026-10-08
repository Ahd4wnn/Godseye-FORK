# PHASE 7 — FINAL ARCHITECTURE CONFORMANCE & EVALUATION

Read `CLAUDE.md`.

Do NOT add new functionality unless required to fix a verified conformance defect.

## Audit Every Major Component

Verify:

- CCTV
- RTSP/RTMP/HLS
- GStreamer
- Frame Sampling
- Preprocessing
- NanoOwl
- Embedding Reuse
- ByteTrack
- FastReID
- ChromaDB
- SQLite
- Clip Storage
- Knowledge Store
- Local LLM
- Referent Resolver
- Clarification Engine
- Query Planner
- Retrieval Engine
- Path Reconstruction
- Response Builder

For each determine:

1. Exists?
2. Actually implemented?
3. Correct technology?
4. Correct responsibility?
5. Correct interface?
6. Tested?
7. Production path or test-only?
8. Any downgrade?
9. Any hallucinated behavior?
10. Any undocumented assumption?

## Explicit Downgrade Audit

Search for accidental substitutions such as:

- OpenCV replacing GStreamer
- YOLO replacing NanoOwl
- custom tracking replacing ByteTrack
- heuristics replacing FastReID
- JSON replacing SQLite
- in-memory vectors replacing ChromaDB
- keyword matching replacing LLM NLU
- hardcoded retrieval
- fake clips
- fake detections
- fake embeddings
- fake identities
- fake path reconstruction
- mock data in production paths

## CLAUDE.md Audit

Every claim must be verified.

Do not say:

- implemented without implementation
- tested without test evidence
- production ready while critical blockers exist

## Final Report

Provide:

Architecture Conformance: PASS / FAIL

Implementation Completeness: PASS / FAIL

Testing: PASS / FAIL

Security: PASS / FAIL

Performance: PASS / FAIL

Production Readiness: READY / NOT READY / BLOCKED

List every remaining issue, technical debt item, and blocker.

Update `CLAUDE.md` with the final verified state.

STOP.
