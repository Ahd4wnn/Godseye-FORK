import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import type { Camera } from '../api/types'
import { useStore, type Processing } from '../store'
import { clock, duration, wallOf } from '../time'
import { BoxOverlay, boxesAt, useVideoFrame } from './BoxOverlay'
import { Head } from './Head'

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(t)
  }, [])
  return <>{duration((now - since) / 1000)}</>
}

const Bar = ({ value, label, tone = 'blue' }: { value: number; label: string; tone?: 'blue' | 'amber' }) => (
  <div className="bar" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
    <motion.i className={`bar__fill bar__fill--${tone}`} animate={{ width: `${Math.min(1, value) * 100}%` }} transition={{ duration: 0.25, ease: 'easeOut' }} />
  </div>
)

function Evidence() {
  const { activeHit: hit, answerHits, answerRecalled, facts, cameras, selectHit } = useStore()
  const video = useRef<HTMLVideoElement>(null)
  const frame = useVideoFrame(video, !!hit)

  // Open the clip at the best frame.
  useEffect(() => {
    const v = video.current
    if (!v || !hit) return
    const go = () => { v.currentTime = Math.max(0, hit.t_best - hit.t_start) }
    v.readyState >= 1 ? go() : v.addEventListener('loadedmetadata', go, { once: true })
  }, [hit?.id])

  const camName = (id: string) => cameras.find((c) => c.id === id)?.name ?? id
  const tracks = hit ? [...new Set(hit.boxes.map((b) => b.track_id).filter(Boolean))] : []
  const memory = answerRecalled.map((id) => facts.find((f) => f.id === id)?.referent ?? id)

  return (
    <section className="section section--evidence" aria-label="Evidence">
      <Head title="Evidence">{hit && answerHits.length > 1 && <span>{answerHits.findIndex((h) => h.id === hit.id) + 1} of {answerHits.length}</span>}</Head>
      {!hit ? (
        <p className="section__empty">The clip for each answer plays here.</p>
      ) : (
        <div className="section__body">
          <div className="player">
            <video key={hit.clip_url} ref={video} src={hit.clip_url} poster={hit.thumb_url} controls autoPlay muted playsInline />
            {frame && <BoxOverlay boxes={boxesAt(hit.boxes, hit.t_start + frame.t)} rect={frame.rect} />}
          </div>
          {answerHits.length > 1 && (
            <div className="filmstrip" role="list">
              {answerHits.map((h) => (
                <button key={h.id} role="listitem" className={`film${h.id === hit.id ? ' film--on' : ''}`} onClick={() => selectHit(h)} aria-label={`${camName(h.camera_id)} at ${clock(Date.parse(h.wall_time))}`}>
                  <img src={h.thumb_url} alt="" />
                  <span>{clock(Date.parse(h.wall_time))}</span>
                </button>
              ))}
            </div>
          )}
          <dl className="props">
            <dt>Camera</dt><dd className="mono">{camName(hit.camera_id)}</dd>
            <dt>Time</dt><dd className="mono">{clock(Date.parse(hit.wall_time), true)} UTC</dd>
            <dt>Match</dt><dd>{hit.label}</dd>
            {hit.why && <><dt>Similarity</dt><dd className="mono">{hit.why.similarity.toFixed(2)}</dd></>}
            <dt>Score</dt><dd className="mono">{hit.score.toFixed(2)}</dd>
            {hit.why?.matched_terms.length ? <><dt>Terms</dt><dd>{hit.why.matched_terms.join(' · ')}</dd></> : null}
            {memory.length > 0 && <><dt>Memory</dt><dd className="amber">{memory.join(' · ')}</dd></>}
            {tracks.length > 0 && <><dt>Track</dt><dd className="mono">{tracks.map((t) => `#${t}`).join(' ')}</dd></>}
          </dl>
        </div>
      )}
    </section>
  )
}

function Activity() {
  const { processing, lastRun, cameras } = useStore()
  // Keep the last run on screen after it finishes so the log and latency stay readable.
  const [snap, setSnap] = useState<Processing | null>(processing)
  useEffect(() => { if (processing) setSnap(processing) }, [processing])
  const p = processing ?? snap
  const idx = (id: string) => cameras.findIndex((c) => c.id === id)
  const pct = p?.progress && p.progress.frames_total ? p.progress.frames_done / p.progress.frames_total : null
  const latency = !processing && p?.kind === 'query' ? lastRun : null
  const maxStage = latency ? Math.max(1, ...Object.values(latency.stages)) : 1

  return (
    <section className="section" aria-label="Activity">
      <Head title="Activity">
        {processing ? (
          <span className="live"><i className="dot dot--busy" />{pct !== null ? `${Math.round(pct * 100)}%` : p?.sent !== undefined ? `${Math.round(p.sent * 100)}%` : 'Working'}</span>
        ) : latency ? (
          <span>{Math.round(latency.total_ms)} ms</span>
        ) : null}
      </Head>
      {!p ? (
        <p className="section__empty">Search and indexing progress shows here.</p>
      ) : (
        <div className="section__body">
          <div className="activity__title">
            <p>{p.title}</p>
            {processing && <span className="mono"><Elapsed since={p.started} /></span>}
          </div>

          {latency && (
            <div className="group">
              {Object.entries(latency.stages).map(([name, ms]) => (
                <div key={name} className="metric">
                  <span>{name}</span>
                  <Bar value={ms / maxStage} label={`${name} ${ms} ms`} tone="amber" />
                  <span className="mono">{Math.round(ms)} ms</span>
                </div>
              ))}
            </div>
          )}

          {processing && p.sent !== undefined && (
            <div className="group">
              <div className="metric">
                <span>Upload</span>
                <Bar value={p.sent} label="Upload" />
                <span className="mono">{Math.round(p.sent * 100)}%</span>
              </div>
            </div>
          )}

          {processing && p.progress && (
            <div className="group">
              <div className="metric">
                <span>All</span>
                <Bar value={pct ?? 0} label="All frames" />
                <span className="mono">{p.progress.frames_done.toLocaleString()} / {p.progress.frames_total.toLocaleString()}</span>
              </div>
              {Object.entries(p.progress.cameras).map(([id, v]) => {
                const cam = cameras[idx(id)]
                const total = cam ? frameCount(cam) : null
                return (
                  <div key={id} className="metric">
                    <span>{cam?.name ?? id}</span>
                    <Bar value={v} label={cam?.name ?? id} />
                    {/* cameras still being uploaded are not in /cameras yet, so only a share is known */}
                    <span className="mono">{total ? `${Math.round(v * total).toLocaleString()} / ${total.toLocaleString()}` : `${Math.round(v * 100)}%`}</span>
                  </div>
                )
              })}
            </div>
          )}

          {p.candidates.length > 0 && (
            <div className="group">
              <p className="group__label">Candidates</p>
              <div className="chips">
                {[...p.candidates].sort((a, b) => b.score - a.score).slice(0, 12).map((c, i) => {
                  const cam = cameras[idx(c.camera_id)]
                  return (
                    <button
                      key={i}
                      className="chip chip--cand"
                      disabled={!cam}
                      title={`score ${c.score.toFixed(2)}`}
                      onClick={() => cam && useStore.setState({ activeCameraId: cam.id, wall: wallOf(cam, c.t), playing: false })}
                    >
                      {cam?.name ?? c.camera_id} · {duration(c.t)}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {p.log.length > 0 && (
            <ol className="log">
              {p.log.map((l, i) => (
                <li key={i}><span>{l.name}</span>{l.detail}</li>
              ))}
            </ol>
          )}
        </div>
      )}
    </section>
  )
}

// BACKEND: fps and duration_s must be the real stream metadata (e.g. from ffprobe) or these counts are wrong.
const frameCount = (c: Camera) => Math.round(c.fps * c.duration_s)

function Footage() {
  const cameras = useStore((s) => s.cameras)
  if (!cameras.length) return null
  const total = cameras.reduce((n, c) => n + frameCount(c), 0)
  return (
    <section className="section" aria-label="Footage">
      <Head title="Footage"><span>{total.toLocaleString()} frames</span></Head>
      <div className="section__body">
        <div className="footage" role="table" aria-label="Frames per video">
          <div role="row" className="footage__head">
            <span role="columnheader">Camera</span>
            <span role="columnheader">Frames</span>
            <span role="columnheader">FPS</span>
            <span role="columnheader">Length</span>
          </div>
          {cameras.map((c) => (
            <div role="row" key={c.id}>
              <span role="cell">{c.name}</span>
              <span role="cell">{frameCount(c).toLocaleString()}</span>
              <span role="cell">{+c.fps.toFixed(2)}</span>
              <span role="cell">{duration(c.duration_s)}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export function Inspector() {
  return (
    <aside className="inspector">
      <Evidence />
      <Activity />
      <Footage />
    </aside>
  )
}
