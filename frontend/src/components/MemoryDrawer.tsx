import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useStore } from '../store'
import { Head } from './Head'

export function MemoryDrawer() {
  const { memoryOpen, setMemoryOpen, facts, cameras, forget, restart, backend, newFactId } = useStore()
  const [confirming, setConfirming] = useState(false)
  const restarting = backend === 'restarting'

  useEffect(() => {
    if (!memoryOpen) return setConfirming(false)
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMemoryOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [memoryOpen, setMemoryOpen])

  return (
    <AnimatePresence>
      {memoryOpen && (
        <>
          <motion.div className="scrim" onClick={() => setMemoryOpen(false)} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
          <motion.aside
            className="drawer"
            role="dialog"
            aria-label="Memory"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            <Head title="Memory">
              <button className="icon-btn" onClick={() => setMemoryOpen(false)} aria-label="Close memory">
                <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
              </button>
            </Head>
            <p className="drawer__intro">What I learned from your answers. I use these instead of asking again, and they survive a restart.</p>
            {facts.length === 0 ? (
              <p className="section__empty">Nothing learned yet. When I ask which camera or area you mean, your answer is saved here.</p>
            ) : (
              <ul className="facts">
                <AnimatePresence initial={false}>
                  {facts.map((f, i) => (
                    <motion.li
                      key={f.id}
                      layout
                      className={`fact${f.id === newFactId ? ' fact--new' : ''}`}
                      initial={{ opacity: 0, scale: 1.15 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, x: 40 }}
                      transition={{ duration: 0.2, delay: restarting ? 0 : i * 0.04 }}
                    >
                      <div className="fact__row">
                        <p className="fact__ref">{f.referent}</p>
                        <button className="link" onClick={() => forget(f.id)}>Forget</button>
                      </div>
                      <p className="fact__cam">{cameras.find((c) => c.id === f.camera_id)?.name ?? f.camera_id}{f.region && ' · region'}</p>
                      <p className="fact__src">“{f.source_query}” · {new Date(f.created_at).toLocaleString()}</p>
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ul>
            )}
            <div className="drawer__foot">
              {confirming ? (
                <>
                  <p>Restart the backend? Memory should survive.</p>
                  <button className="btn" onClick={() => setConfirming(false)}>Cancel</button>
                  <button className="btn btn--primary" onClick={() => { setConfirming(false); restart() }}>Restart</button>
                </>
              ) : (
                <button className="btn" onClick={() => setConfirming(true)} disabled={backend !== 'online'}>
                  {restarting ? 'Restarting…' : 'Restart backend'}
                </button>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
