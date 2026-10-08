import type { Camera } from './api/types'

const ms = (iso: string) => Date.parse(iso)

/** Wall-clock span (epoch ms) covering every camera's recording. */
// BACKEND: start_wall must be each recording's real UTC start, or the cameras won't line up on the timeline.
export function span(cams: Camera[]): [number, number] | null {
  if (!cams.length) return null
  const starts = cams.map((c) => ms(c.start_wall))
  const ends = cams.map((c) => ms(c.start_wall) + c.duration_s * 1000)
  return [Math.min(...starts), Math.max(...ends)]
}

/** Seconds into the camera's video for a wall time, or null if the camera wasn't recording then. */
export function videoTime(cam: Camera, wall: number): number | null {
  const t = (wall - ms(cam.start_wall)) / 1000
  return t < 0 || t > cam.duration_s ? null : t * scale(cam)
}

const scale = (cam: Camera) => (cam.video_s && cam.duration_s ? cam.video_s / cam.duration_s : 1)
export const wallOf = (cam: Camera, t: number) => ms(cam.start_wall) + (t / scale(cam)) * 1000

/** 14:22:07 in UTC */
export const clock = (wall: number, withMs = false) => new Date(wall).toISOString().slice(11, withMs ? 23 : 19)

/** 03:21 (or 1:03:21 past an hour) */
export function duration(sec: number) {
  const s = Math.max(0, Math.floor(sec))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60
  const mm = String(m).padStart(2, '0'), ss = String(r).padStart(2, '0')
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}
