import { useEffect, useLayoutEffect, useRef } from 'react'
import { useAnimate, useReducedMotion, type AnimationPlaybackControls } from 'motion/react'
import { EyeMark } from './EyeMark'

// Easter egg (Ctrl+K twice). Content is intentionally hand-written: it is the team's credits, not app data.
type Row = [role: string, name: string, alias?: string]
type Block = { title: string; rows?: Row[]; lines?: string[]; fine?: boolean }

const BLOCKS: Block[] = [
  {
    title: 'Starring',
    rows: [
      ['The Eye', 'As itself'],
      ['The Main Gate', 'Camera 1', 'asked once, remembered forever'],
      ['A Red Car', 'May or may not have passed the main gate'],
      ['Person Carrying a Large Bag', 'Uncredited'],
      ['The Baseline', 'As the one we beat'],
    ],
  },
  {
    title: 'The Anatomy of God',
    rows: [
      ['UI / UX', 'Adon Joseph', "God's Skin"],
      ['Clustering', 'Allen Shinoj', "God's Nerves"],
      ['Backend', 'Albhy Mathew', "God's Brain"],
      ['Backend', 'Lenin Jacob', "God's Heart"],
    ],
  },
  {
    title: 'Crew',
    rows: [
      ['Director of Photography', 'Every CCTV camera that never blinked'],
      ['Head of Continuity', 'Clarify-once memory', 'never asks twice'],
      ['Re-identification Specialist', 'Same car, different camera, trust us'],
      ['Timestamp Wrangler', 'UTC, the one true timezone'],
      ['Latency Department', 'Every millisecond we bragged about'],
      ['Stunt Coordinator', 'git push --force'],
      ['Merge Conflict Mediator', 'Whoever pushed last'],
      ['Key Grip', 'The extension cord'],
      ['Best Boy', "The laptop that didn't crash"],
      ['Gaffer', 'One desk lamp, 24 hours'],
    ],
  },
  {
    title: 'Music',
    rows: [
      ['Original Score', 'Laptop fans at full speed'],
      ['Percussion', 'Mechanical keyboards, 3 a.m.'],
      ['Featured Track', '“Works on My Machine”', 'live at demo day'],
    ],
  },
  {
    title: 'Catering',
    rows: [
      ['Fuel', 'Coffee, then more coffee'],
      ['Emergency Rations', 'Instant noodles'],
      ['Sleep', 'Out of stock'],
    ],
  },
  {
    title: 'Thanks to',
    lines: [
      'The judges, for bringing footage we had never seen',
      'Our mentors, for asking the hard questions',
      'The Wi-Fi, on the occasions it worked',
      'Stack Overflow, as always',
      'Ctrl+Z, our most reliable teammate',
    ],
  },
  {
    title: 'Special thanks',
    lines: [
      'Vinod Sir, for yesterday’s TED talk',
      'Our parents, for not calling during the demo',
      'The red car, for finally passing the main gate',
      'Whoever invented the energy drink',
      'You, for pressing Ctrl+K twice',
    ],
  },
  {
    title: 'The fine print',
    fine: true,
    lines: [
      'No red cars were harmed in the making of this film.',
      'Any resemblance to real CCTV footage is purely indexed.',
      'Every frame was watched by the Eye. None were judged. Except by the judges.',
      'Filmed entirely in 24 hours, on location, mostly at one table.',
    ],
  },
]

const SPEED = 48 // px per second, about the pace of film credits

export function Credits({ onExit }: { onExit: () => void }) {
  const [scope, animate] = useAnimate<HTMLDivElement>()
  const end = useRef<HTMLDivElement>(null)
  const run = useRef<AnimationPlaybackControls>()
  const paused = useRef(false)
  const still = useReducedMotion()

  // Roll from below the screen until "The End" sits in the middle, then stop there.
  useLayoutEffect(() => {
    if (still) return
    const roll = scope.current, last = end.current!
    const vh = window.innerHeight
    const to = vh / 2 - (last.offsetTop + last.offsetHeight / 2)
    run.current = animate(roll, { y: [vh, to] }, { duration: (vh - to) / SPEED, ease: 'linear' })
    return () => run.current?.stop()
  }, [still])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onExit()
      if (e.key === ' ' && run.current) {
        e.preventDefault()
        paused.current ? run.current.play() : run.current.pause()
        paused.current = !paused.current
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onExit])

  return (
    <div className={`credits${still ? ' credits--still' : ''}`} role="dialog" aria-modal="true" aria-label="Credits">
      <div className="credits__viewport">
        <div className="credits__roll" ref={scope}>
          <header className="credits__title">
            <EyeMark size={96} searching={!still} />
            <h1>Gods Eye</h1>
            <p>A Team Gods Eye picture</p>
            <p className="credits__code">Problem statement HNX26EPS05 · made in 24 hours</p>
          </header>

          {BLOCKS.map((b) => (
            <section key={b.title} className={`credits__block${b.fine ? ' credits__block--fine' : ''}`}>
              <h2>{b.title}</h2>
              {b.rows?.map(([role, name, alias], i) => (
                <div key={i} className="credits__row">
                  <span className="credits__role">{role}</span>
                  <span className="credits__name">
                    {name}
                    {alias && <span className="credits__alias">{alias}</span>}
                  </span>
                </div>
              ))}
              {b.lines?.map((l) => <p key={l} className="credits__line">{l}</p>)}
            </section>
          ))}

          <div className="credits__end" ref={end}>
            <h2>The End</h2>
            <p>Gods Eye will return… after a nap.</p>
            <p className="credits__code">© 2026 Team Gods Eye · all frames indexed</p>
          </div>
        </div>
      </div>
      <p className="credits__hint">{still ? 'Esc to return' : 'Space to pause · Esc to return'}</p>
    </div>
  )
}
