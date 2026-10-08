import { useLayoutEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { fitRect } from './BoxOverlay'

const PAD = 24

/** Read-only site blueprint inferred by the backend from the footage. Drawn in real pixels so text stays crisp. */
export function Blueprint() {
  const { site, cameras, activeCameraId, answerHits, selectCamera } = useStore()
  const box = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(box.current!)
    return () => ro.disconnect()
  }, [])

  const layout = site?.layout
  if (!layout) {
    return (
      <div className="blueprint" ref={box}>
        <p className="wall__empty">The blueprint appears once the backend has mapped the cameras.</p>
      </div>
    )
  }

  // BACKEND: send layout.aspect (width / height of the inferred site); 1.6 is only the fallback shape.
  const aspect = layout.aspect ?? 1.6
  const r = fitRect(aspect * 1000, 1000, Math.max(0, size.w - PAD * 2), Math.max(0, size.h - PAD * 2))
  const X = (x: number) => PAD + r.left + x * r.width
  const Y = (y: number) => PAD + r.top + y * r.height
  const pos = new Map(layout.cameras.map((c) => [c.camera_id, c]))
  const name = (id: string) => cameras.find((c) => c.id === id)?.name ?? id
  const withHits = new Set(answerHits.map((h) => h.camera_id))
  const R = Math.max(28, Math.min(r.width, r.height) * 0.12)

  return (
    <div className="blueprint" ref={box}>
      <svg width={size.w} height={size.h} role="img" aria-label={`${site!.name} blueprint`}>
        <defs>
          <pattern id="bp-dots" width="16" height="16" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="0.8" className="bp-dot" />
          </pattern>
        </defs>
        <rect width={size.w} height={size.h} fill="url(#bp-dots)" />

        {layout.zones.map((z) => {
          const pts = z.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ')
          const minX = Math.min(...z.polygon.map((p) => p[0])), minY = Math.min(...z.polygon.map((p) => p[1]))
          return (
            <g key={z.id}>
              <polygon points={pts} className="bp-zone" />
              <text x={X(minX) + 8} y={Y(minY) + 16} className="bp-label">{z.name}</text>
            </g>
          )
        })}

        {layout.links?.map((l, i) => {
          const a = pos.get(l.from), b = pos.get(l.to)
          if (!a || !b) return null
          return (
            <g key={i}>
              <line x1={X(a.x)} y1={Y(a.y)} x2={X(b.x)} y2={Y(b.y)} className="bp-link" />
              {l.transit_s !== undefined && (
                <text x={(X(a.x) + X(b.x)) / 2} y={(Y(a.y) + Y(b.y)) / 2 - 6} className="bp-label bp-label--mid">{Math.round(l.transit_s)}s</text>
              )}
            </g>
          )
        })}

        {layout.cameras.map((c) => {
          const on = c.camera_id === activeCameraId
          const half = ((c.fov_deg / 2) * Math.PI) / 180
          const wedge = `M0 0 L${-Math.sin(half) * R} ${-Math.cos(half) * R} A${R} ${R} 0 0 1 ${Math.sin(half) * R} ${-Math.cos(half) * R} Z`
          return (
            <g
              key={c.camera_id}
              className={`bp-cam${on ? ' bp-cam--on' : ''}${withHits.has(c.camera_id) ? ' bp-cam--hit' : ''}`}
              transform={`translate(${X(c.x)} ${Y(c.y)})`}
              onClick={() => selectCamera(c.camera_id)}
              role="button"
              tabIndex={0}
              aria-label={`Focus ${name(c.camera_id)}`}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && selectCamera(c.camera_id)}
            >
              <path d={wedge} transform={`rotate(${c.heading_deg})`} className="bp-fov" />
              <rect x="-4" y="-4" width="8" height="8" rx="1.5" className="bp-pin" />
              <text x="10" y="4" className="bp-label bp-label--cam">{name(c.camera_id)}</text>
            </g>
          )
        })}
      </svg>
      <span className="blueprint__note">Inferred from footage</span>
    </div>
  )
}
