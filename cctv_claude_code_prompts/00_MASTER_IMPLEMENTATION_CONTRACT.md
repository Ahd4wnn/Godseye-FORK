# MASTER IMPLEMENTATION CONTRACT

You are implementing a production-grade multi-camera CCTV video analytics + natural-language surveillance search system.

This is NOT a demo, toy, mockup, simplified prototype, or downgraded proof-of-concept.

## Architecture

### Track A — Video Analytics

CCTV  
→ RTSP / RTMP / HLS  
→ GStreamer  
→ Frame Sampling  
→ Preprocessing  
→ NanoOwl Detection  
→ Embedding Reuse  
→ ByteTrack  
→ FastReID  
→ ChromaDB  
→ SQLite  
→ Clip Storage

### Track B — Query / Search

User  
→ Chat UI  
→ Local LLM / NLU  
→ Referent Resolver  
→ Clarification Engine  
→ Resolved Query Context  
→ Query Planner  
→ Retrieval Engine  
→ Path Reconstructor  
→ Response Builder  
→ Grounded Answer + Evidence

## Non-Negotiable Rules

1. Do not downgrade the architecture.
2. Do not silently substitute required technologies.
3. Do not use fake inference, fake embeddings, fake identities, fake clips, or fake query results in production paths.
4. Never invent APIs, model capabilities, dependencies, performance numbers, or requirements.
5. Read `CLAUDE.md` before every phase.
6. Update `CLAUDE.md` after every phase.
7. Never claim implementation or testing unless it actually exists and was verified.
8. Preserve architecture boundaries.
9. Mocks are allowed only in isolated tests, never in production execution paths.
10. If a required capability cannot be verified, STOP and report the blocker.
11. Do not implement future phases unless explicitly requested.
12. Prefer fixing the current phase over accumulating technical debt.

## Required Technologies

Where defined by the architecture, preserve:

- GStreamer for video ingestion/decoding
- NanoOwl for open-vocabulary detection
- NanoOwl-associated/shared CLIP vision encoder for embedding reuse
- ByteTrack for camera-local tracking
- FastReID for cross-camera Re-ID
- ChromaDB for persistent vector storage
- SQLite for structured metadata/knowledge storage
- Local LLM / Ollama / Llama 3 where specified

Never replace these with weaker alternatives merely because they are easier.

## Security

The LLM must never directly execute arbitrary:

- SQL
- shell commands
- filesystem operations
- database operations
- network operations

LLM outputs must be structured, validated, and passed through controlled application interfaces.

## Completion Gate

A phase is complete only when:

- implementation exists
- interfaces are documented
- configuration exists
- tests exist
- tests pass, or failures are explicitly documented
- failure handling exists
- no architectural downgrade was introduced
- `CLAUDE.md` is updated
- known issues are documented

## Stop Conditions

STOP and report instead of guessing when:

- a required model/API is unavailable or unclear
- requirements conflict
- a dependency cannot support the required capability
- hardware capability is uncertain
- architectural correctness would require a downgrade
- a critical assumption cannot be verified
