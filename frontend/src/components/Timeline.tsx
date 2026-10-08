import { useEffect, type CSSProperties, type MouseEvent } from 'react'
import { useStore } from '../store'
import { clock, duration, span, videoTime, wallOf } from '../time'

const STEPS = [5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]

export function Timeline() {
  const { cameras, wall, playing, setPlaying, seek, answerHits, activeHit, activeCameraId, selectHit, selectCamera, processing } = useStore()
  const sp = span(cameras)

  // Master clock: advance the shared playhead; every tile follows it.
  useEffect(() => {
    if (!playing || !sp) return
    let raf = 0, last = performance.now()
    const tick = (now: number) => {
      const w = (useStore.getState().wall ?? sp[0]) + (now - last)
      last = now
      if (w >= sp[1]) return useStore.setState({ wall: sp[1], playing: false })
      seek(w)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, sp?.[0], sp?.[1]])

  if (!sp || wall === null) return null

  const [t0, t1] = sp
  const len = Math.max(1, t1 - t0)
  const frac = (w: number) => (w - t0) / len
  const pct = (w: number) => `${frac(w) * 100}%`
  const step = STEPS.find((s) => len / 1000 / s <= 10) ?? 7200
  // drop a label that would hang off the right edge
  const ticks = Array.from({ length: Math.floor(len / 1000 / step) + 1 }, (_, i) => i * step).filter((s) => (s * 1000) / len < 0.96)
  const seekFromClick = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    seek(t0 + ((e.clientX - r.left) / r.width) * len)
  }

  return (
    <section className="timeline" aria-label="Timeline" style={{ '--p': frac(wall) } as CSSProperties}>
      <div className="transport">
        <button className="icon-btn" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? (
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5h2.5v11H4zM9.5 2.5H12v11H9.5z" fill="currentColor" /></svg>
          ) : (
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.5v11l9-5.5z" fill="currentColor" /></svg>
          )}
        </button>
        <span className="transport__clock">{clock(wall)}</span>
        <input className="scrub" type="range" min={t0} max={t1} step={100} value={wall} onChange={(e) => seek(Number(e.target.value))} aria-label="Playhead" />
        <span className="transport__dur">{duration((wall - t0) / 1000)} / {duration(len / 1000)}</span>
      </div>

      <div className="lanes">
        {cameras.map((cam) => {
          const start = Date.parse(cam.start_wall)
          const live = videoTime(cam, wall) !== null
          return (
            <div key={cam.id} className={`lane${cam.id === activeCameraId ? ' lane--on' : ''}`}>
              <button className="lane__name" onClick={() => selectCamera(cam.id)} aria-pressed={cam.id === activeCameraId}>
                <i className={`dot dot--${live ? 'live' : 'idle'}`} />
                {cam.name}
              </button>
              <div className="lane__track" onClick={seekFromClick}>
                <div className="lane__rec" style={{ left: pct(start), width: `${((cam.duration_s * 1000) / len) * 100}%` }} />
                {processing?.candidates.filter((c) => c.camera_id === cam.id).map((c, i) => (
                  <span key={i} className="tick tick--cand" style={{ left: pct(wallOf(cam, c.t)), opacity: 0.35 + 0.65 * c.score }} />
                ))}
                {answerHits.filter((h) => h.camera_id === cam.id).map((h) => (
                  <button
                    key={h.id}
                    className={`tick tick--hit${activeHit?.id === h.id ? ' tick--on' : ''}`}
                    style={{ left: pct(Date.parse(h.wall_time)) }}
                    onClick={(e) => { e.stopPropagation(); selectHit(h) }}
                    aria-label={`${h.label} on ${cam.name} at ${clock(Date.parse(h.wall_time))}`}
                  />
                ))}
              </div>
            </div>
          )
        })}
        <div className="playhead" />
      </div>

      <div className="ruler">
        {ticks.map((s) => (
          <span key={s} style={{ '--x': (s * 1000) / len } as CSSProperties}>{duration(s)}</span>
        ))}
      </div>
    </section>
  )
}
