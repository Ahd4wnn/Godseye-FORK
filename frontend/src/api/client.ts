import { sseParser, type RawEvent } from './sse'

// BACKEND: FastAPI serves every route under APIRouter(prefix="/api"); Vite proxies /api in dev (vite.config.ts).
export const API = '/api'
// Godseye's server refuses state-changing requests without this header (CSRF defence).
const CSRF = { 'X-Requested-With': 'godseye' }
// BACKEND: set VITE_BACKEND_URL in frontend/.env if FastAPI is not on localhost:8000.
export const BACKEND_LABEL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8080'

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const isForm = body instanceof FormData
  const res = await fetch(API + path, {
    method,
    headers: { ...CSRF, ...(body && !isForm ? { 'Content-Type': 'application/json' } : {}) },
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${method} ${path} → ${res.status} ${await res.text().catch(() => '')}`.trim())
  return res.status === 204 || res.status === 202 ? (undefined as T) : res.json()
}

export const api = {
  get: <T>(p: string) => req<T>('GET', p),
  post: <T>(p: string, b?: unknown) => req<T>('POST', p, b),
  put: <T>(p: string, b?: unknown) => req<T>('PUT', p, b),
  del: (p: string) => req<void>('DELETE', p),
}

// fetch can't report upload progress, so multipart uploads go through XHR.
export function postForm<T>(path: string, form: FormData, onProgress: (sent: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', API + path)
    xhr.responseType = 'json'
    xhr.setRequestHeader('X-Requested-With', 'godseye')
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onload = () => (xhr.status < 300 ? resolve(xhr.response) : reject(new Error(`POST ${path} → ${xhr.status}`)))
    xhr.onerror = () => reject(new Error(`POST ${path} → network error`))
    xhr.send(form)
  })
}

// Reads a text/event-stream response (GET or POST) and calls onEvent per event until it ends.
export async function stream<E extends RawEvent>(path: string, onEvent: (e: E) => void, opts: { body?: unknown; signal?: AbortSignal } = {}) {
  const res = await fetch(API + path, {
    method: opts.body === undefined ? 'GET' : 'POST',
    headers: { ...CSRF, Accept: 'text/event-stream', ...(opts.body !== undefined && { 'Content-Type': 'application/json' }) },
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
    signal: opts.signal,
  })
  if (!res.ok || !res.body) throw new Error(`${path} → ${res.status}`)
  const feed = sseParser(onEvent as (e: RawEvent) => void)
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    feed(value)
  }
}
