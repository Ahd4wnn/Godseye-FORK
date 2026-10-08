# PHASE 4 — USER QUERY UNDERSTANDING

Read `CLAUDE.md` and the master contract.

## Boundary

User  
→ Chat UI  
→ Local LLM / NLU  
→ Referent Resolver  
→ Clarification Engine  
→ Resolved Query Context

STOP BEFORE Query Planner.

## Local LLM

Use the intended local LLM architecture.

Where specified:

- Llama 3 8B
- Ollama

Verify actual model/runtime availability.

Do not replace the LLM with keyword matching.

## NLU

Extract and validate:

- intent
- object
- location
- time
- identity/referent
- camera constraints
- spatial constraints

Preserve the original query.

## Referent Resolution

Use:

1. exact
2. fuzzy
3. semantic/embedding
4. LLM fallback

Do not skip directly to guessing with the LLM.

## Clarification

Support architecture-defined mechanisms where applicable:

- multiple choice
- camera grid
- point-and-click
- region selection
- LLM clarify()

User responses must feed back into resolution.

Persist learned mappings through the knowledge store.

## Security

LLM output is untrusted.

It must never directly execute:

- SQL
- shell
- filesystem operations
- arbitrary database operations
- arbitrary network operations

Use structured validated interfaces.

## Testing

Test:

- normal queries
- incomplete queries
- ambiguous queries
- missing referents
- malformed LLM output
- clarification
- learned mappings
- security boundaries

## Completion

Update `CLAUDE.md` with:

- LLM/model configuration
- query schema
- resolver behavior
- clarification behavior
- tests
- limitations
- issues
- next phase

STOP.
