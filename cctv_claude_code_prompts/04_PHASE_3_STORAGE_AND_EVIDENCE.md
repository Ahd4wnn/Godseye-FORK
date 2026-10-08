# PHASE 3 — PERSISTENCE, VECTOR STORAGE & VIDEO EVIDENCE

Read `CLAUDE.md` and the master contract.

## Boundary

Enriched Track/Identity Output  
→ ChromaDB  
→ SQLite  
→ Clip Storage

## ChromaDB

Implement persistent vector storage for visual embeddings.

Support:

- persistent collections
- embedding insertion
- metadata association
- ANN/vector search interface
- persistence across restart

Do not replace ChromaDB with in-memory arrays or JSON.

## SQLite

Persist structured surveillance metadata including architecture-required information such as:

- camera_id
- start/end time
- track_id
- global_id
- bbox
- label
- clip_path

Use proper schema design and indexes.

## Clip Storage

Store actual evidence clips using deterministic organization, for example:

`/clips/{date}/{camera}/{id}/...`

Do not create fake clip references.

Publish metadata references only after media exists and is validated.

## Consistency

Handle failures between:

- SQLite
- ChromaDB
- filesystem

Prevent silent orphan records and broken references.

Where cross-system atomicity is impossible, implement explicit reconciliation/consistency handling.

## Testing

Test:

- persistence
- restart
- vector storage
- metadata consistency
- clip generation
- failed writes
- missing media
- disk errors
- duplicates

Do not implement query retrieval yet.

## Completion

Update `CLAUDE.md` with:

- storage architecture
- schemas/collections
- paths
- interfaces
- tests
- failure behavior
- issues
- next phase

STOP.
