import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { motion } from 'motion/react'
import type { Camera } from '../api/types'
import { useStore } from '../store'
import { clock, videoTime } from '../time'
import { BoxOverlay, boxesAt, fitRect, useVideoFrame } from './BoxOverlay'

type Teach = { suggested: boolean; picked: boolean; region?: [number, number][]; drawRegion: boolean; onPick: (region?: [number, number][]) => void }

export function CameraTile({ cam, active, span, teach }: { cam: Camera; active: boolean; span: boolean; teach?: Teach }) {
  const wall = useStore((s) => s.wall)
  const playing = useStore((s) => s.playing)
  const activeHit = useStore((s) => s.activeHit)
  const selectCamera = useStore((s) => s.selectCamera)
  const video = useRef<HTMLVideoElement>(null)
  const [failed, setFailed] = useState(false)
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)

  const vt = wall === null ? null : videoTime(cam, wall)
  const hit = activeHit?.camera_id === cam.id ? activeHit : null
  const frame = useVideoFrame(video, !!hit || !!teach)

  useEffect(() => {
    const v = video.current
    if (!v || failed) return
    if (vt === null) return void v.pause()
    if (Math.abs(v.currentTime - vt) > 0.35) v.currentTime = vt
    if (playing && v.paused) v.play().catch(() => {})
    if (!playing && !v.paused) v.pause()
  }, [vt, playing, failed])

  // Teach mode: drag a rectangle in video-normalised coordinates.
  const norm = (e: PointerEvent) => {
    const el = video.current!
    const box = el.getBoundingClientRect()
    const r = fitRect(el.videoWidth, el.videoHeight, box.width, box.height)
    const clamp = (n: number) => Math.min(1, Math.max(0, n))
    return { x: clamp((e.clientX - box.left - r.left) / r.width), y: clamp((e.clientY - box.top - r.top) / r.height) }
  }
  const finishDrag = () => {
    if (!drag || !teach) return
    const [x0, x1] = [drag.x0, drag.x1].sort((a, b) => a - b)
    const [y0, y1] = [drag.y0, drag.y1].sort((a, b) => a - b)
    setDrag(null)
    teach.onPick(x1 - x0 > 0.01 && y1 - y0 > 0.01 ? [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] : undefined)
  }
  const shown = drag
    ? [[Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1)], [Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)]]
    : teach?.picked && teach.region
      ? [teach.region[0], teach.region[2]]
      : null
  const state = failed ? 'error' : vt === null ? 'idle' : 'live'

  return (
    <motion.div
      layout
      transition={{ type: 'spring', stiffness: 420, damping: 40 }}
      style={span ? { gridColumn: 'span 2', gridRow: 'span 2', order: -1 } : undefined}
      className={['tile', active && 'tile--active', teach && 'tile--teach', teach?.suggested && 'tile--suggested', teach?.picked && 'tile--picked'].filter(Boolean).join(' ')}
      onClick={teach ? undefined : () => selectCamera(cam.id)}
      onPointerDown={teach ? (e) => { const p = norm(e); e.currentTarget.setPointerCapture(e.pointerId); setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }) } : undefined}
      onPointerMove={teach && drag ? (e) => { const p = norm(e); setDrag({ ...drag, x1: p.x, y1: p.y }) } : undefined}
      onPointerUp={teach ? () => (drag && teach.drawRegion ? finishDrag() : (setDrag(null), teach.onPick())) : undefined}
      role="button"
      tabIndex={0}
      aria-pressed={active}
      aria-label={teach ? `Choose ${cam.name}` : `Focus ${cam.name}`}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') teach ? teach.onPick() : selectCamera(cam.id) }}
    >
      {/* BACKEND: video_url must support HTTP range requests (FastAPI FileResponse / StaticFiles do) so seeking works. */}
      {!failed && <video ref={video} src={cam.video_url} poster={cam.thumb_url} muted playsInline preload="auto" onError={() => setFailed(true)} />}
      {hit && frame && <BoxOverlay boxes={boxesAt(hit.boxes, frame.t)} rect={frame.rect} />}
      {teach && frame && shown && (
        <div className="boxes" style={frame.rect}>
          <div className="region" style={{ left: `${shown[0][0] * 100}%`, top: `${shown[0][1] * 100}%`, width: `${(shown[1][0] - shown[0][0]) * 100}%`, height: `${(shown[1][1] - shown[0][1]) * 100}%` }} />
        </div>
      )}
      <span className="tag tag--name"><i className={`dot dot--${state}`} />{cam.name}</span>
      {state === 'live' && wall !== null && <span className="tag tag--time">{clock(wall)}</span>}
    </motion.div>
  )
}
