# Claude Code Implementation Pack

This directory contains the phase-by-phase Claude Code prompts for the CCTV multi-camera analytics and natural-language search system.

## Put these files in your project root

```text
your-project/
├── CLAUDE.md
├── 00_MASTER_IMPLEMENTATION_CONTRACT.md
├── 01_PHASE_0_ARCHITECTURE_AND_CLAUDE_MD.md
├── 02_PHASE_1_VIDEO_INGESTION.md
├── 03_PHASE_2_AI_PERCEPTION.md
├── 04_PHASE_3_STORAGE_AND_EVIDENCE.md
├── 05_PHASE_4_QUERY_UNDERSTANDING.md
├── 06_PHASE_5_RETRIEVAL_PATH_RESPONSE.md
├── 07_PHASE_6_INTEGRATION_HARDENING.md
├── 08_PHASE_7_FINAL_AUDIT.md
└── README.md
```

## Workflow

### First Claude Code session

Use:

```text
Read 00_MASTER_IMPLEMENTATION_CONTRACT.md and 01_PHASE_0_ARCHITECTURE_AND_CLAUDE_MD.md.

Execute Phase 0 only.

Do not implement Phase 1.

Stop when Phase 0 is complete.
```

### Every later phase

Use:

```text
Read CLAUDE.md first.

Then read the relevant phase file.

Execute ONLY that phase.

Do not redo verified work from previous phases.

Update CLAUDE.md when finished.

Run the completion gate.

Stop.
```

## Phase order

1. `01_PHASE_0_ARCHITECTURE_AND_CLAUDE_MD.md`
2. `02_PHASE_1_VIDEO_INGESTION.md`
3. `03_PHASE_2_AI_PERCEPTION.md`
4. `04_PHASE_3_STORAGE_AND_EVIDENCE.md`
5. `05_PHASE_4_QUERY_UNDERSTANDING.md`
6. `06_PHASE_5_RETRIEVAL_PATH_RESPONSE.md`
7. `07_PHASE_6_INTEGRATION_HARDENING.md`
8. `08_PHASE_7_FINAL_AUDIT.md`

## Important

`CLAUDE.md` is the project's persistent implementation state and must be treated as the source of truth for future Claude Code sessions.

Do not accept a phase as complete merely because the application starts. Require tests and architecture verification.
