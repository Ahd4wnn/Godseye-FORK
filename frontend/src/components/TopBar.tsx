import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { clock } from '../time'
import { EyeMark } from './EyeMark'
import { UploadDialog } from './UploadDialog'

function UtcClock() {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])
  return <>{clock(now)}</>
}

const STATUS = { online: 'Online', connecting: 'Connecting', offline: 'Offline', restarting: 'Restarting' } as const

export function TopBar() {
  const { backend, site, health, cameras, facts, processing, setMemoryOpen } = useStore()
  const [adding, setAdding] = useState(false)
  const online = backend === 'online'

  return (
    <header className="topbar">
      <div className="brand">
        <EyeMark size={22} />
        <span>Gods Eye</span>
      </div>
      <span className={`status status--${backend}`}>
        <i />
        {online && site ? site.name : STATUS[backend]}
      </span>
      <dl className="stats">
        <div><dt>UTC</dt><dd><UtcClock /></dd></div>
        <div><dt>Cams</dt><dd>{health ? `${health.cameras_online}/${cameras.length}` : '—'}</dd></div>
        <div><dt>Index</dt><dd>{health ? health.index_size.toLocaleString() : '—'}</dd></div>
      </dl>
      <div className="topbar__actions">
        <button className="btn" onClick={() => setMemoryOpen(true)} disabled={!health}>
          Memory <span className="count">{health ? facts.length : '—'}</span>
        </button>
        <button className="btn btn--primary" onClick={() => setAdding(true)} disabled={!online || !!processing} aria-haspopup="dialog">
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 11V2.5M4.5 6 8 2.5 11.5 6M2.5 10.5v3h11v-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
          {processing?.kind === 'upload' ? 'Indexing…' : 'Upload video'}
        </button>
      </div>
      <UploadDialog open={adding} onClose={() => setAdding(false)} />
    </header>
  )
}
