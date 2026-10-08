import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, LayoutGroup, motion } from 'motion/react'
import { BACKEND_LABEL } from '../api/client'
import { useStore } from '../store'
import { Blueprint } from './Blueprint'
import { CameraTile } from './CameraTile'
import { EyeLoader } from './EyeLoader'
import { Head } from './Head'

const GAP = 8

/** Column count that gives the largest 16:9 tiles in a w×h panel. A locked-on tile spans 2x2, so it needs 3+ columns. */
function grid(n: number, focus: boolean, w: number, h: number) {
  const span = focus && n >= 3
  const cells = span ? n + 3 : n
  let best = { cols: 1, rows: 1, tileW: 0 }
  for (let cols = span ? 3 : 1; cols <= Math.max(span ? 3 : 1, n); cols++) {
    const rows = Math.max(1, Math.ceil(cells / cols))
    const tileW = Math.max(0, Math.min((w - GAP * (cols - 1)) / cols, ((h - GAP * (rows - 1)) / rows) * (16 / 9)))
    if (tileW > best.tileW + 0.5) best = { cols, rows, tileW }
  }
  return { span, ...best }
}

export function VideoWall() {
  const { backend, cameras, activeCameraId, processing, clarify, answerClarify, skipClarify } = useStore()
  const [view, setView] = useState<'cameras' | 'blueprint'>('cameras')
  const [pick, setPick] = useState<{ camera_id: string; region?: [number, number][] } | null>(null)
  const body = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useLayoutEffect(() => {
    const el = body.current!
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => setPick(null), [clarify?.question_id])
  useEffect(() => { if (clarify) setView('cameras') }, [clarify])
  useEffect(() => {
    if (!clarify) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') skipClarify()
      if (e.key === 'Enter' && pick && !(e.target instanceof HTMLInputElement)) answerClarify(pick.camera_id, pick.region)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clarify, pick, answerClarify, skipClarify])

  const focus = !clarify && cameras.some((c) => c.id === activeCameraId)
  const g = grid(cameras.length, focus, size.w, size.h)
  const tileW = g.tileW
  const last = processing?.log.at(-1)
  const empty =
    backend === 'offline' || backend === 'connecting'
      ? `Waiting for backend at ${BACKEND_LABEL}`
      : cameras.length === 0 && !processing
        ? 'No cameras yet. Upload recordings to index them.'
        : null

  return (
    <section className="wall" aria-label="Video wall">
      <Head
        title={
          <span className="segmented" role="tablist">
            {(['cameras', 'blueprint'] as const).map((v) => (
              <button key={v} role="tab" aria-selected={view === v} className={view === v ? 'on' : undefined} onClick={() => setView(v)} disabled={!!clarify && v === 'blueprint'}>
                {v === 'cameras' ? 'Cameras' : 'Blueprint'}
              </button>
            ))}
          </span>
        }
      >
        {cameras.length > 0 && <span>{cameras.length} {cameras.length === 1 ? 'feed' : 'feeds'}</span>}
      </Head>

      <div className="wall__body" ref={body}>
        {view === 'blueprint' ? (
          <Blueprint />
        ) : (
          <div
            className={`wall__grid${clarify ? ' wall__grid--teach' : ''}`}
            style={{ gridTemplateColumns: `repeat(${g.cols}, ${tileW}px)`, gridAutoRows: `${(tileW * 9) / 16}px`, gap: GAP }}
          >
            <LayoutGroup>
              {cameras.map((cam) => (
                <CameraTile
                  key={cam.id}
                  cam={cam}
                  active={focus && cam.id === activeCameraId}
                  span={focus && g.span && cam.id === activeCameraId}
                  teach={
                    clarify
                      ? {
                          suggested: clarify.suggested_cameras.includes(cam.id),
                          picked: pick?.camera_id === cam.id,
                          region: pick?.camera_id === cam.id ? pick.region : undefined,
                          drawRegion: clarify.kind === 'region',
                          onPick: (region) => setPick({ camera_id: cam.id, region }),
                        }
                      : undefined
                  }
                />
              ))}
            </LayoutGroup>
          </div>
        )}

        {empty && view === 'cameras' && <p className="wall__empty">{empty}</p>}

        <AnimatePresence>
          {clarify && (
            <motion.div
              key="teach"
              className="hud"
              role="dialog"
              aria-label="Clarify"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
            >
              <div>
                <p className="hud__q">{clarify.question}</p>
                <p className="hud__hint">{clarify.kind === 'region' ? 'Pick a camera and drag the area' : 'Pick a camera'} · Enter to confirm · Esc to skip</p>
              </div>
              <button className="btn" onClick={skipClarify}>Skip</button>
              <button className="btn btn--primary" disabled={!pick} onClick={() => pick && answerClarify(pick.camera_id, pick.region)}>Confirm</button>
            </motion.div>
          )}
          {processing && !clarify && (
            <EyeLoader
              key="eye"
              title={processing.title}
              detail={last ? `${last.name}${last.detail ? ` · ${last.detail}` : ''}` : undefined}
              progress={processing.progress && processing.progress.frames_total ? processing.progress.frames_done / processing.progress.frames_total : processing.sent}
            />
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}
