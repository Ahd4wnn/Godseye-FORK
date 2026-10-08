import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { useStore, type Message } from '../store'
import { clock } from '../time'
import { Head } from './Head'

function EyeLine({ m }: { m: Extract<Message, { role: 'eye' }> }) {
  const { cameras, facts, activeHit, selectHit, ask } = useStore()
  const camName = (id: string) => cameras.find((c) => c.id === id)?.name ?? id
  return (
    <div className={`msg msg--eye msg--${m.kind}`}>
      <p className="msg__who">God's Eye:</p>
      <p className="msg__text">{m.text}</p>
      {m.kind === 'answer' && (m.hits.length > 0 || m.recalled.length > 0) && (
        <div className="chips">
          {m.hits.map((h) => (
            <button
              key={h.id}
              className={`chip${activeHit?.id === h.id ? ' chip--on' : ''}`}
              onClick={() => selectHit(h, { hits: m.hits, recalled: m.recalled })}
              title={`${h.label} · score ${h.score.toFixed(2)}`}
            >
              {camName(h.camera_id)} · {clock(Date.parse(h.wall_time))}
            </button>
          ))}
          {m.recalled.map((id) => (
            <span key={id} className="chip chip--recalled">
              Recalled · {facts.find((f) => f.id === id)?.referent ?? id}
            </span>
          ))}
        </div>
      )}
      {m.kind === 'error' && m.retry && (
        <p className="msg__hint"><button className="link" onClick={() => ask(m.retry!)}>Retry</button></p>
      )}
    </div>
  )
}

export function ChatRail() {
  const { messages, ask, processing, backend, cameras } = useStore()
  const [text, setText] = useState('')
  const log = useRef<HTMLDivElement>(null)
  const asking = processing?.kind === 'query'
  const canAsk = backend === 'online' && cameras.length > 0 && !asking

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: 'smooth' })
  }, [messages.length])

  return (
    <aside className="rail">
      <Head title="Conversation" />
      <div className="rail__log" ref={log} aria-live="polite">
        {messages.length === 0 && backend === 'online' && (
          <div className="rail__empty">
            <p>{cameras.length ? 'Ask about the footage in plain English.' : 'Upload recordings to begin.'}</p>
            <p>{cameras.length ? 'Every answer cites the camera, the time and the clip.' : 'Each video becomes one camera.'}</p>
          </div>
        )}
        {messages.map((m) => (
          <motion.div key={m.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.18, ease: 'easeOut' }}>
            {m.role === 'user' ? (
              <div className="msg msg--user"><p className="msg__who">User:</p><p className="msg__bubble">{m.text}</p></div>
            ) : (
              <EyeLine m={m} />
            )}
          </motion.div>
        ))}
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          const q = text.trim()
          if (!q || !canAsk) return
          setText('')
          ask(q)
        }}
      >
        <input
          className="composer__input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask God's Eye"
          aria-label="Ask God's Eye"
          disabled={backend !== 'online'}
        />
        <button className="composer__send" disabled={!canAsk || !text.trim()} aria-label="Send">
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
      </form>
    </aside>
  )
}
