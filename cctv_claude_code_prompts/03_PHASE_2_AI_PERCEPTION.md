# PHASE 2 — AI PERCEPTION, TRACKING & CROSS-CAMERA IDENTITY

Read `CLAUDE.md` and the master contract.

## Boundary

Preprocessed Frame  
→ NanoOwl  
→ Embedding Reuse  
→ ByteTrack  
→ FastReID  
→ Enriched Track/Identity Output

## NanoOwl

Use the actual NanoOwl architecture/model required by the project.

Before implementation verify:

- package/version
- model availability
- inference API
- runtime requirements
- GPU/CPU compatibility

Do NOT substitute YOLO, OpenCV detection, or another detector.

Implement configurable open-vocabulary prompts.

Detection output must include appropriate:

- bbox
- label
- confidence
- camera_id
- timestamp
- model provenance

Correctly map model coordinates back to original-frame coordinates.

## Embedding Reuse

Reuse the NanoOwl-associated/shared CLIP vision encoder.

Do NOT introduce an unnecessary independent CLIP model.

Generate embeddings from detected regions and validate:

- dimensions
- normalization
- source association

## ByteTrack

Use actual ByteTrack.

Maintain:

- camera-local `track_id`
- bbox
- timestamp
- confidence
- trajectory
- track lifecycle/state

Do not replace it with centroid tracking or a homemade tracker.

## FastReID

Use actual FastReID for cross-camera Re-ID.

Maintain:

- `track_id` = camera-local identity
- `global_id` = cross-camera identity

Respect person/vehicle Re-ID requirements.

Do not replace Re-ID with color heuristics or handcrafted matching.

Avoid aggressive false merges.

## Testing

Test:

- real detection
- multiple objects
- track lifecycle
- occlusion
- cross-camera identity
- ambiguous identity
- person/vehicle separation where applicable
- embedding consistency

Mocks are permitted only in isolated unit tests.

## Completion

Update `CLAUDE.md` with:

- model versions
- runtime requirements
- configuration
- interfaces
- tests
- limitations
- issues
- decisions
- next phase

STOP.
