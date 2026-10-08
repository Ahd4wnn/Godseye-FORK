// BACKEND: keep the FastAPI Pydantic models identical to these types (field names and units).
// Mirrors the backend contract in README.md. `t` = seconds into that camera's video, `wall` = ISO 8601 UTC.

export type Health = { ok: boolean; index_size: number; facts: number; cameras_online: number; capabilities?: { upload: boolean; restart: boolean } }
// Blueprint the backend infers from the footage (normalised 0–1 coordinates). Read-only in the UI.
export type SiteLayout = {
  aspect?: number // width / height of the blueprint, default 1.6
  zones: { id: string; name: string; polygon: [number, number][] }[]
  cameras: { camera_id: string; x: number; y: number; heading_deg: number; fov_deg: number }[]
  links?: { from: string; to: string; transit_s?: number }[]
}
export type Site = { name: string; layout?: SiteLayout }
export type Camera = {
  id: string
  name: string
  video_url: string
  thumb_url: string
  fps: number
  duration_s: number // wall-clock span of the recording
  video_s?: number // length of the video file when it differs from the wall span (sampled capture played at a fixed fps)
  start_wall: string
}

export type Box = { t: number; x: number; y: number; w: number; h: number; track_id?: string; kind?: 'object' | 'face'; label?: string }

export type Hit = {
  id: string
  camera_id: string
  t_start: number
  t_end: number
  t_best: number
  wall_time: string
  score: number
  clip_url: string
  thumb_url: string
  label: string
  boxes: Box[]
  why?: { similarity: number; matched_terms: string[] }
}

export type Fact = { id: string; referent: string; camera_id: string; region?: [number, number][]; source_query: string; created_at: string }

export type Progress = { frames_done: number; frames_total: number; cameras: Record<string, number> }
export type Candidate = { camera_id: string; t: number; score: number }

// Server-sent events from POST /api/query
export type QueryEvent =
  | { event: 'parsed'; data: { query_id: string; terms: string[]; time_window?: [string, string]; referents: string[] } }
  | { event: 'stage'; data: { name: string; ms: number; detail?: string } }
  | { event: 'progress'; data: Progress }
  | { event: 'candidates'; data: { items: Candidate[] } }
  | { event: 'clarify'; data: { question_id: string; question: string; referent: string; kind: 'camera' | 'region' | 'entity' | 'choice' | 'text'; suggested_cameras: string[]; options?: { value: string; label: string }[] } }
  | { event: 'answer'; data: { text: string; hits: Hit[]; memory_used: string[] } }
  | { event: 'error'; data: { message: string } }
  | { event: 'done'; data: { total_ms: number; stages: Record<string, number> } }

// Server-sent events from GET /api/jobs/:id/events (upload indexing)
export type JobEvent =
  | { event: 'progress'; data: Progress }
  | { event: 'stage'; data: { name: string; ms: number; detail?: string } }
  | { event: 'error'; data: { message: string } }
  | { event: 'done'; data: { cameras: number } }
