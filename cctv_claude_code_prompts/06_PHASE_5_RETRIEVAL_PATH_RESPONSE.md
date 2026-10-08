# PHASE 5 — QUERY PLANNING, RETRIEVAL, PATH RECONSTRUCTION & RESPONSE

Read `CLAUDE.md` and the master contract.

## Boundary

Resolved Query Context  
→ Query Planner  
→ Retrieval Engine  
→ Path Reconstructor  
→ Response Builder  
→ Grounded Answer + Evidence

## Query Planner

Create a validated QueryPlan supporting:

- semantic_query
- camera_filter
- time_filter
- spatial_filter
- object_filter
- identity_filter
- ranking_strategy
- reconstruction_required

LLM may help construct the structured plan.

LLM must NOT directly execute database/filesystem/shell operations.

Validate all plans before execution.

## Retrieval

Use:

- ChromaDB
- SQLite

for hybrid retrieval.

Use the existing NanoOwl-associated CLIP text encoder for semantic query embeddings where required.

Do not introduce an unrelated embedding model.

Apply metadata filters and vector retrieval.

Do not perform naive full-database scans.

Return ranked tracklets/events with provenance sufficient for path reconstruction.

## Path Reconstruction

Use:

- global_id
- ordered timestamps
- camera topology
- travel-time constraints
- spatio-temporal relationships
- trajectories where available

Validate physical plausibility.

Distinguish:

- OBSERVED
- INFERRED

Never present inferred transitions as direct observations.

## Response Builder

Build grounded responses from verified evidence.

Use deterministic templates for factual fields.

A local LLM may improve wording but must not invent surveillance facts.

Where available attach:

- camera
- timestamp
- confidence
- clip
- path

Clearly distinguish:

- NO RESULT
- PARTIAL EVIDENCE
- AMBIGUOUS RESULT
- SYSTEM ERROR

## Testing

Test:

- query → plan
- plan validation
- semantic retrieval
- metadata filtering
- empty results
- path reconstruction
- impossible transitions
- partial evidence
- missing clips
- low confidence
- hallucination resistance

## Completion

Update `CLAUDE.md` with:

- QueryPlan schema
- retrieval behavior
- path model
- response behavior
- tests
- limitations
- issues
- next phase

STOP.
