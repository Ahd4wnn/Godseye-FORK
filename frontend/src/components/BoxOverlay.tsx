import { useEffect, useState, type RefObject } from 'react'
import type { Box } from '../api/types'

type Rect = { left: number; top: number; width: number; height: number }

/** Where an object-fit: contain video actually draws inside its element. */
export function fitRect(vw: number, vh: number, ew: number, eh: number): Rect {
  if (!vw || !vh) return { left: 0, top: 0, width: ew, height: eh }
  const s = Math.min(ew / vw, eh / vh)
  const width = vw * s, height = vh * s
  return { left: (ew - width) / 2, top: (eh - height) / 2, width, height }
}

/** Current time + drawn rect of a video, tracked every frame while `on`. */
export function useVideoFrame(ref: RefObject<HTMLVideoElement | null>, on: boolean) {
  const [frame, setFrame] = useState<{ t: number; rect: Rect } | null>(null)
  useEffect(() => {
    const v = ref.current
    if (!on || !v) return setFrame(null)
    let raf = 0
    const tick = () => {
      const rect = fitRect(v.videoWidth, v.videoHeight, v.clientWidth, v.clientHeight)
      setFrame((f) => (f && f.t === v.currentTime && f.rect.width === rect.width && f.rect.height === rect.height ? f : { t: v.currentTime, rect }))
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [ref, on])
  return frame
}

/** Boxes closest to time t (one per track, within 1 s), so detections don't linger. */
export function boxesAt(boxes: Box[], t: number) {
  const best = new Map<string, Box>()
  boxes.forEach((b, i) => {
    if (Math.abs(b.t - t) > 1) return
    const key = b.track_id ?? `#${i}`
    const cur = best.get(key)
    if (!cur || Math.abs(b.t - t) < Math.abs(cur.t - t)) best.set(key, b)
  })
  return [...best.values()]
}

export function BoxOverlay({ boxes, rect }: { boxes: Box[]; rect: Rect }) {
  return (
    <div className="boxes" style={rect}>
      {boxes.map((b, i) => (
        <div key={b.track_id ?? i} className="box" style={{ left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }}>
          {(b.label || b.track_id) && <span className="box__label">{b.label ?? b.track_id}</span>}
        </div>
      ))}
    </div>
  )
}
