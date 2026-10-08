# PHASE 0 — ARCHITECTURE LOCK, REPOSITORY AUDIT & CLAUDE.md

Read `00_MASTER_IMPLEMENTATION_CONTRACT.md` first.

## Objective

Establish the permanent engineering context before implementation.

## Instructions

1. Inspect the entire repository:
   - source code
   - dependencies
   - configuration
   - tests
   - Docker/compose files
   - scripts
   - documentation
   - model files
   - database files
   - UI/API
   - deployment files

2. Do not modify implementation code initially.

3. Determine the actual status of every major architecture component:
   - IMPLEMENTED
   - PARTIALLY IMPLEMENTED
   - SCAFFOLD ONLY
   - MISSING
   - UNKNOWN

4. Map architecture components to repository modules.

5. Audit actual dependencies and runtime requirements.

6. Run the existing test suite without fixing unrelated problems yet.

## Create / Update `CLAUDE.md`

It must contain:

- PROJECT
- ARCHITECTURE
- CURRENT PHASE
- PHASE STATUS for Phase 0–7
- IMPLEMENTED COMPONENTS
- TECHNOLOGY REQUIREMENTS
- ARCHITECTURAL INVARIANTS
- INTERFACE CONTRACTS
- DEPENDENCIES
- CONFIGURATION
- DATABASE / STORAGE
- TEST STATUS
- KNOWN ISSUES
- KNOWN LIMITATIONS
- ARCHITECTURAL DECISIONS
- DO NOT CHANGE
- NEXT ACTION

Only record verified facts.

Use `UNKNOWN` or `NOT VERIFIED` when necessary.

## Anti-Hallucination Rule

Never write "implemented" unless verified.

Never write "tested" unless tests were actually executed.

Never invent model capabilities, dependencies, or performance numbers.

## Restrictions

Do NOT implement Phase 1 or later.

## Completion

Update `CLAUDE.md`, report repository findings, baseline tests, blockers, and exact next phase.

STOP.
