# Gods Eye: frontend

React + Vite + TypeScript, Zustand (one shared store), Motion. There is no mock data: every pixel comes from the FastAPI backend, and when it is down the UI shows an offline state.

## Run

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api to VITE_BACKEND_URL (default http://localhost:8000)
npm test           # SSE parser + time model
npm run build      # outputs dist/
```

Serve the build from FastAPI, with every API route under `/api`:

```python
app.include_router(api_router)  # APIRouter(prefix="/api")
app.mount("/", StaticFiles(directory="frontend/dist", html=True))
```

## API contract

Conventions: ids are strings; `t` = seconds into that camera's video; wall times are ISO 8601 UTC; boxes and points are normalised 0–1 (`x, y` top-left). Media URLs (`video_url`, `thumb_url`, `clip_url`) must be reachable from the browser: serve them under `/api/...` or as absolute URLs. MP4 needs HTTP range support. Types live in `src/api/types.ts`.

| Method | Path | Returns / does |
|---|---|---|
| GET | `/api/health` | `{ ok, index_size, facts, cameras_online }`, polled every 5 s |
| GET | `/api/site` | `{ name, layout? }`, where `layout` is the **AI-inferred blueprint**: `{ aspect?, zones[{id,name,polygon}], cameras[{camera_id,x,y,heading_deg,fov_deg}], links?[{from,to,transit_s?}] }`. Read-only in the UI. |
| GET | `/api/cameras` | `Camera[]`: `{ id, name, video_url, thumb_url, fps, duration_s, start_wall }`. Frames per video = `fps × duration_s`. |
| POST | `/api/query` `{ text, session_id }` | `text/event-stream`, see below |
| POST | `/api/clarify` `{ question_id, camera_id, region? }` | `Fact`. `{ question_id, skip: true }` → 204, answer without the referent |
| GET | `/api/memory` | `Fact[]`: `{ id, referent, camera_id, region?, source_query, created_at }` |
| DELETE | `/api/memory/:id` | 204 |
| POST | `/api/admin/restart` | 202. The UI polls `/health` every 500 ms for up to 30 s, then reloads memory. |
| POST | `/api/videos` multipart `files` + `camera_tags` (1..n, same order) | `{ job_id }`. One video or a whole building: each file becomes the camera named by its tag. FastAPI: `files: list[UploadFile] = File(...), camera_tags: list[str] = Form(...)` |
| POST | `/api/videos/links` `{ items: [{ url, camera_tag }] }` | `{ job_id }`. The backend downloads each URL, then indexes it the same way. |
| GET | `/api/jobs/:id/events` | SSE: `progress`, `stage`, `error`, then `done { cameras }` |

`/api/query` events, in order:

```
parsed     { query_id, terms[], time_window?, referents[] }
stage      { name, ms, detail? }                      // detail becomes a log line
progress   { frames_done, frames_total, cameras: { [camera_id]: 0..1 } }
candidates { items: [{ camera_id, t, score }] }
clarify    { question_id, question, referent, kind: 'camera'|'region'|'entity', suggested_cameras[] }
           // keep the stream open; it continues after POST /clarify
answer     { text, hits: Hit[], memory_used: fact_id[] }
error      { message }
done       { total_ms, stages: { [name]: ms } }       // shown as the latency breakdown
```

`Hit = { id, camera_id, t_start, t_end, t_best, wall_time (of t_best), score, clip_url, thumb_url, label, boxes[{ t, x, y, w, h, track_id?, kind?, label? }], why?{ similarity, matched_terms[] } }`. Box `t` is camera video time, and the clip starts at `t_start`.

## Where to connect the backend

Every integration point in the code carries a `// BACKEND:` comment naming the route and what it must return:

```bash
grep -rn "BACKEND:" src
```

## How the panels talk

All panels read one store (`src/store.ts`):
- Clicking an answer chip, timeline tick or filmstrip thumb calls `selectHit`. The camera locks on (2×2), the playhead seeks to it, the evidence clip opens and the lane and blueprint pin highlight.
- One master playhead drives every tile.
- A `clarify` event puts the wall in teach mode (pick a tile, drag a region, then Enter or Esc).
