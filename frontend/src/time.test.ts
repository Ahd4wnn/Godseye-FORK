import { describe, expect, it } from 'vitest'
import { clock, duration, span, videoTime, wallOf } from './time'
import type { Camera } from './api/types'

const cam = (id: string, start: string, dur: number): Camera =>
  ({ id, name: id, video_url: '', thumb_url: '', fps: 25, duration_s: dur, start_wall: start })

describe('time model', () => {
  const a = cam('a', '2026-10-08T14:00:00Z', 600)
  const b = cam('b', '2026-10-08T14:05:00Z', 600)
  it('spans all cameras', () => {
    expect(span([a, b])).toEqual([Date.parse('2026-10-08T14:00:00Z'), Date.parse('2026-10-08T14:15:00Z')])
    expect(span([])).toBeNull()
  })
  it('maps wall time to each camera video time and back', () => {
    const w = Date.parse('2026-10-08T14:06:00Z')
    expect(videoTime(a, w)).toBe(360)
    expect(videoTime(b, w)).toBe(60)
    expect(videoTime(b, Date.parse('2026-10-08T14:01:00Z'))).toBeNull()
    expect(wallOf(b, 60)).toBe(w)
    // sampled capture: 10 s of wall time stored as a 20 s video -> video time runs twice as fast as the wall clock
    const s = { ...a, duration_s: 10, video_s: 20 }
    expect(videoTime(s, Date.parse(s.start_wall) + 5000)).toBe(10)
    expect(wallOf(s, 10)).toBe(Date.parse(s.start_wall) + 5000)
  })
  it('formats', () => {
    expect(clock(Date.parse('2026-10-08T14:22:07.400Z'), true)).toBe('14:22:07.400')
    expect(duration(201)).toBe('03:21')
    expect(duration(3801)).toBe('1:03:21')
  })
})
