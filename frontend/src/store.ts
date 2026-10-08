import { create } from 'zustand'
import { api, postForm, stream } from './api/client'
import type { Camera, Candidate, Fact, Health, Hit, JobEvent, Progress, QueryEvent, Site } from './api/types'
import { span } from './time'

export type Message =
  | { id: string; role: 'user'; text: string }
  | { id: string; role: 'eye'; kind: 'answer'; text: string; hits: Hit[]; recalled: string[] }
  | { id: string; role: 'eye'; kind: 'clarify' | 'learned' | 'info'; text: string }
  | { id: string; role: 'eye'; kind: 'error'; text: string; retry?: string }

export type Clarify = Extract<QueryEvent, { event: 'clarify' }>['data']

export type Processing = {
  kind: 'query' | 'upload' | 'restart'
  title: string
  started: number
  progress?: Progress
  sent?: number // share of the upload body sent, 0..1
  candidates: Candidate[]
  log: { name: string; detail?: string }[]
}

/** What the Upload dialog sends: video files from this device, or links the backend downloads. Each becomes one tagged camera. */
export type UploadSource =
  | { kind: 'files'; items: { file: File; tag: string }[] }
  | { kind: 'links'; items: { url: string; tag: string }[] }

export type LastRun = { total_ms: number; stages: Record<string, number> }

type State = {
  backend: 'connecting' | 'online' | 'offline' | 'restarting'
  health: Health | null
  site: Site | null
  cameras: Camera[]
  facts: Fact[]
  messages: Message[]
  activeCameraId: string | null
  activeHit: Hit | null
  answerHits: Hit[]
  answerRecalled: string[]
  wall: number | null
  playing: boolean
  processing: Processing | null
  lastRun: LastRun | null
  clarify: Clarify | null
  memoryOpen: boolean
  newFactId: string | null

  boot: () => void
  ask: (text: string) => Promise<void>
  answerClarify: (camera_id: string, region?: [number, number][]) => Promise<void>
  answerWith: (response: string) => Promise<void>
  skipClarify: () => Promise<void>
  upload: (source: UploadSource) => Promise<void>
  forget: (id: string) => Promise<void>
  restart: () => Promise<void>
  selectHit: (hit: Hit, group?: { hits: Hit[]; recalled: string[] }) => void
  selectCamera: (id: string) => void
  seek: (wall: number) => void
  setPlaying: (p: boolean) => void
  setMemoryOpen: (o: boolean) => void
}

const uid = () => crypto.randomUUID()
// BACKEND: per-tab session id sent with /query; swap for a server-issued id if the backend tracks sessions.
const SESSION = uid()
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export const useStore = create<State>((set, get) => {
  type EyeMsg = Extract<Message, { role: 'eye' }>
  const say = (m: EyeMsg extends infer T ? (T extends EyeMsg ? Omit<T, 'id' | 'role'> : never) : never) =>
    set((s) => ({ messages: [...s.messages, { id: uid(), role: 'eye', ...m } as Message] }))

  const patchProcessing = (fn: (p: Processing) => Partial<Processing>) =>
    set((s) => (s.processing ? { processing: { ...s.processing, ...fn(s.processing) } } : {}))

  // BACKEND: GET /api/health, /api/site, /api/cameras, /api/memory. Everything on screen starts here.
  async function loadAll() {
    const [health, site, cameras, facts] = await Promise.all([
      api.get<Health>('/health'),
      api.get<Site>('/site'),
      api.get<Camera[]>('/cameras'),
      api.get<Fact[]>('/memory'),
    ])
    const s = get()
    const sp = span(cameras)
    set({
      backend: 'online',
      health,
      site,
      cameras,
      facts,
      // keep the operator's playhead if it is still inside the footage
      wall: s.wall !== null && sp && s.wall >= sp[0] && s.wall <= sp[1] ? s.wall : sp ? sp[0] : null,
      activeCameraId: cameras.some((c) => c.id === s.activeCameraId) ? s.activeCameraId : null,
    })
  }

  let healthTimer: ReturnType<typeof setTimeout> | undefined
  async function healthLoop() {
    clearTimeout(healthTimer)
    if (get().backend === 'restarting') return
    try {
      // BACKEND: GET /api/health → { ok, index_size, facts, cameras_online }, polled every 5 s.
      const health = await api.get<Health>('/health')
      if (get().backend !== 'online') await loadAll()
      else set({ health })
    } catch {
      if (get().backend !== 'restarting') set({ backend: 'offline' })
    }
    healthTimer = setTimeout(healthLoop, get().backend === 'online' ? 5000 : 2000)
  }

  let queryAbort: AbortController | null = null
  let booted = false

  return {
    backend: 'connecting',
    health: null,
    site: null,
    cameras: [],
    facts: [],
    messages: [],
    activeCameraId: null,
    activeHit: null,
    answerHits: [],
    answerRecalled: [],
    wall: null,
    playing: false,
    processing: null,
    lastRun: null,
    clarify: null,
    memoryOpen: false,
    newFactId: null,

    boot: () => {
      if (booted) return
      booted = true
      healthLoop()
    },

    async ask(text) {
      queryAbort?.abort()
      queryAbort = new AbortController()
      set((s) => ({
        messages: [...s.messages, { id: uid(), role: 'user', text }],
        processing: { kind: 'query', title: `Matching "${text}"`, started: Date.now(), candidates: [], log: [] },
        clarify: null,
      }))
      let failed = false
      try {
        await stream<QueryEvent>(
          // BACKEND: POST /api/query → text/event-stream (parsed, stage, progress, candidates, clarify, answer, error, done).
          '/query',
          (e) => {
            switch (e.event) {
              case 'parsed':
                patchProcessing(() => ({ title: `Matching "${e.data.terms.join(' ') || text}"` }))
                break
              case 'stage':
                patchProcessing((p) => ({ log: [...p.log, { name: e.data.name, detail: e.data.detail }] }))
                break
              case 'progress':
                patchProcessing(() => ({ progress: e.data }))
                break
              case 'candidates':
                patchProcessing((p) => ({ candidates: [...p.candidates, ...e.data.items] }))
                break
              case 'clarify':
                set({ clarify: e.data })
                say({ kind: 'clarify', text: e.data.question })
                break
              case 'answer': {
                const { text: answer, hits, memory_used } = e.data
                say({ kind: 'answer', text: answer, hits, recalled: memory_used })
                if (hits[0]) get().selectHit(hits[0], { hits, recalled: memory_used })
                else set({ activeHit: null, answerHits: [], answerRecalled: [] })
                if (memory_used.length) set({ newFactId: memory_used[0] })
                break
              }
              case 'error':
                failed = true
                say({ kind: 'error', text: e.data.message, retry: text })
                break
              case 'done':
                set({ lastRun: e.data })
                break
            }
          },
          { body: { text, session_id: SESSION }, signal: queryAbort.signal },
        )
      } catch (err) {
        if ((err as Error).name !== 'AbortError' && !failed) say({ kind: 'error', text: (err as Error).message, retry: text })
      } finally {
        set({ processing: null, clarify: null })
      }
    },

    async answerClarify(camera_id, region) {
      const q = get().clarify
      if (!q) return
      set({ clarify: null })
      try {
        // BACKEND: POST /api/clarify { question_id, camera_id, region? } → Fact; persist it so it survives a restart.
        const fact = await api.post<Fact>('/clarify', { question_id: q.question_id, camera_id, region, wall: get().wall === null ? undefined : get().wall! / 1000 })
        const cam = get().cameras.find((c) => c.id === camera_id)
        set((s) => ({ facts: [fact, ...s.facts.filter((f) => f.id !== fact.id)], newFactId: fact.id }))
        say({ kind: 'learned', text: `Learned: ${fact.referent} = ${cam?.name ?? camera_id}` })
      } catch (err) {
        set({ clarify: q })
        say({ kind: 'error', text: (err as Error).message })
      }
    },

    async answerWith(response) {
      const q = get().clarify
      if (!q) return
      set({ clarify: null })
      try {
        // BACKEND: POST /api/clarify { question_id, response } for choice / free-text questions.
        const fact = await api.post<Fact>('/clarify', { question_id: q.question_id, response })
        if (fact && !fact.id.startsWith('ephemeral:')) set((s) => ({ facts: [fact, ...s.facts.filter((f) => f.id !== fact.id)], newFactId: fact.id }))
      } catch (err) {
        set({ clarify: q })
        say({ kind: 'error', text: (err as Error).message })
      }
    },

    async skipClarify() {
      const q = get().clarify
      if (!q) return
      set({ clarify: null })
      // BACKEND: POST /api/clarify { question_id, skip: true } → 204; resume the open /query stream without the referent.
      await api.post('/clarify', { question_id: q.question_id, skip: true }).catch((err) => say({ kind: 'error', text: (err as Error).message }))
    },

    async upload(source) {
      const n = source.items.length
      if (!n) return
      const tags = source.items.map((i) => i.tag).join(', ')
      const videos = `${n} video${n > 1 ? 's' : ''}`
      set({ processing: { kind: 'upload', title: source.kind === 'files' ? `Uploading ${videos}` : `Indexing ${videos}`, started: Date.now(), candidates: [], log: [{ name: source.kind === 'files' ? 'upload' : 'fetch', detail: tags }] } })
      try {
        let start: Promise<{ job_id: string }>
        if (source.kind === 'files') {
          // camera_tags[i] belongs to files[i]
          const form = new FormData()
          source.items.forEach(({ file, tag }) => { form.append('files', file); form.append('camera_tags', tag) })
          // BACKEND: POST /api/videos multipart → { job_id }; FastAPI: files: list[UploadFile], camera_tags: list[str] = Form(...).
          start = postForm('/videos', form, (sent) => patchProcessing(() => ({ sent })))
        } else {
          // BACKEND: POST /api/videos/links { items: [{ url, camera_tag }] } → { job_id }; the backend downloads each URL.
          start = api.post('/videos/links', { items: source.items.map(({ url, tag }) => ({ url, camera_tag: tag })) })
        }
        const { job_id } = await start
        patchProcessing(() => ({ title: `Indexing ${videos}`, sent: undefined }))
        // BACKEND: GET /api/jobs/:id/events → text/event-stream (progress, stage, error, done).
        await stream<JobEvent>(`/jobs/${encodeURIComponent(job_id)}/events`, (e) => {
          if (e.event === 'progress') patchProcessing(() => ({ progress: e.data }))
          else if (e.event === 'stage') patchProcessing((p) => ({ log: [...p.log, { name: e.data.name, detail: e.data.detail }] }))
          else if (e.event === 'error') say({ kind: 'error', text: e.data.message })
        })
        await loadAll()
        say({ kind: 'info', text: `Indexed ${n} video${n > 1 ? 's' : ''}. Ask me anything about the footage.` })
      } catch (err) {
        say({ kind: 'error', text: `Upload failed: ${(err as Error).message}` })
      } finally {
        set({ processing: null })
      }
    },

    async forget(id) {
      const prev = get().facts
      set({ facts: prev.filter((f) => f.id !== id) })
      try {
        // BACKEND: DELETE /api/memory/:id → 204.
        await api.del(`/memory/${encodeURIComponent(id)}`)
      } catch (err) {
        set({ facts: prev })
        say({ kind: 'error', text: (err as Error).message })
      }
    },

    async restart() {
      clearTimeout(healthTimer)
      set({ backend: 'restarting', processing: { kind: 'restart', title: 'Restarting backend', started: Date.now(), candidates: [], log: [] } })
      const log = (name: string, detail?: string) => patchProcessing((p) => ({ log: [...p.log, { name, detail }] }))
      try {
        // BACKEND: POST /api/admin/restart → 202, then /api/health must come back ok within 30 s with memory reloaded from disk.
        await api.post('/admin/restart')
        log('restart', 'requested')
        const deadline = Date.now() + 30000
        let up = false
        while (Date.now() < deadline && !up) {
          await sleep(500)
          up = await api.get<Health>('/health').then((h) => h.ok).catch(() => false)
        }
        if (!up) throw new Error('Backend did not come back within 30 s')
        await loadAll()
        const { facts, health } = get()
        log('memory', `${facts.length} facts restored`)
        log('index', `${health?.index_size.toLocaleString()} entries`)
        say({ kind: 'info', text: `Backend restarted. ${facts.length} remembered fact${facts.length === 1 ? '' : 's'} restored.` })
      } catch (err) {
        set({ backend: 'offline' })
        say({ kind: 'error', text: `Restart failed: ${(err as Error).message}` })
      } finally {
        set({ processing: null })
        if (get().backend !== 'restarting') healthTimer = setTimeout(healthLoop, 2000)
      }
    },

    selectHit(hit, group) {
      set((s) => ({
        activeHit: hit,
        activeCameraId: hit.camera_id,
        wall: Date.parse(hit.wall_time),
        playing: false,
        ...(group && { answerHits: group.hits, answerRecalled: group.recalled }),
        ...(!group && !s.answerHits.some((h) => h.id === hit.id) && { answerHits: [hit], answerRecalled: [] }),
      }))
    },

    selectCamera: (id) => set((s) => ({ activeCameraId: s.activeCameraId === id ? null : id })),
    seek: (wall) => set({ wall }),
    setPlaying: (playing) => set({ playing }),


    setMemoryOpen: (memoryOpen) => set({ memoryOpen }),
  }
})
