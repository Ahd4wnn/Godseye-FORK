# React console integration (branch `frontend`)

The console in `frontend/` (React + Vite + TypeScript) talks to the existing Track B web server through `godseye/web/frontend_api.py`.
Nothing in Track A/B behaviour changed: questions still go NLU -> resolver -> clarification -> planner -> retrieval -> grounded answer.

## Mapping (contract in `frontend/README.md`)
| UI | Godseye source |
|---|---|
| `/api/health` | counts from `godseye.db` (`index_size` = tracks, `cameras_online` = cameras seen in the last 120 s), `capabilities` |
| `/api/cameras` | each camera's newest `complete` clip: `video_url=/media/clip/<id>`, `duration_s` = wall span, `video_s` = file length |
| `/api/query` (SSE) | `QueryService.ask`; events `parsed`, `stage` (one measured stage), `clarify`, `answer`, `error`, `done`. The stream stays open while a clarification is pending (10 min) |
| answer `hits` | `Evidence` rows that have a playable clip (`clip_id` set): clip URL by id, thumbnail frame, boxes from `observations` normalised by clip size. Evidence without a clip stays in the answer text, never as a fake clip |
| `/api/clarify` | `camera_grid` -> camera; `region_selection` -> region rectangle; `point_and_click` -> centre of the dragged area at the frame under the playhead (looked up in `observations`); `multiple_choice` / `free_text` -> `response` |
| `/api/memory` | `knowledge.db`: learned camera and identity aliases and zones; DELETE forgets one |

## Not provided (honest 501 / hidden in the UI)
Video upload, link import, backend restart (cameras are ingested by Track A from `config/ingest.yaml`), a floor-plan layout (`site.layout`), candidate/progress streaming, per-hit "why" scores. "Skip" on a clarification ends the question without running a search.

## Verification
`tests/test_web_frontend.py` (6, real Track B stack), `frontend` unit tests (`npm test`, 5), `npx tsc -b`. Also driven in headless Chromium against the live server (load, ask, evidence playback with boxes, clarification by tile pick, memory panel; no console errors).
## Limits
Tile playback uses a linear wall<->video mapping; capture is sampled at 2-5 fps so inside one clip it is approximate. The evidence player and boxes use exact clip time. Only each camera's newest clip is on the wall; older evidence is reached through answers. No TLS, single shared token (unchanged).
