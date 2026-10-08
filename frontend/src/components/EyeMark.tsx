import { useId } from 'react'
import { motion } from 'motion/react'

// Rays around the almond, as in the logo.
const RAYS = [-60, -30, 0, 30, 60].flatMap((a) => [a, a + 180]).map((deg) => {
  const r = (deg - 90) * (Math.PI / 180)
  return { x1: 32 + Math.cos(r) * 24, y1: 22 + Math.sin(r) * 19, x2: 32 + Math.cos(r) * 29, y2: 22 + Math.sin(r) * 23 }
})
const ALMOND = 'M8 22 Q32 2 56 22 Q32 42 8 22 Z'

/** The Gods Eye mark. `searching` makes the iris look around and the lids blink. */
export function EyeMark({ size = 40, searching = false }: { size?: number; searching?: boolean }) {
  const clip = 'eye' + useId().replace(/[^a-z0-9]/gi, '')
  return (
    <svg width={size} height={size * 0.69} viewBox="0 0 64 44" fill="none" aria-hidden="true">
      <defs>
        <clipPath id={clip}>
          <path d={ALMOND} />
        </clipPath>
      </defs>
      <motion.g
        stroke="var(--amber)"
        strokeWidth="2.2"
        strokeLinecap="round"
        animate={searching ? { opacity: [0.35, 1, 0.35] } : undefined}
        transition={searching ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : undefined}
      >
        {RAYS.map((l, i) => <line key={i} {...l} />)}
      </motion.g>
      <motion.g
        style={{ originY: '50%' }}
        animate={searching ? { scaleY: [1, 1, 0.08, 1, 1] } : undefined}
        transition={searching ? { duration: 3.2, times: [0, 0.82, 0.86, 0.9, 1], repeat: Infinity } : undefined}
      >
        <path d={ALMOND} stroke="var(--amber)" strokeWidth="2.4" />
        <g clipPath={`url(#${clip})`}>
          <motion.g
            animate={searching ? { x: [0, -11, -11, 11, 11, 0], y: [0, -2, 2, 2, -2, 0] } : undefined}
            transition={searching ? { duration: 3.2, repeat: Infinity, ease: 'easeInOut' } : undefined}
          >
            <circle cx="32" cy="22" r="9" stroke="var(--amber)" strokeWidth="2.4" />
            <circle cx="32" cy="22" r="3.6" fill="var(--amber)" />
          </motion.g>
        </g>
      </motion.g>
    </svg>
  )
}
