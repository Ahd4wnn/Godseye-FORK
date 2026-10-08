import { useEffect, useState } from 'react'
import { MotionConfig } from 'motion/react'
import { useStore } from './store'
import { TopBar } from './components/TopBar'
import { ChatRail } from './components/ChatRail'
import { VideoWall } from './components/VideoWall'
import { Timeline } from './components/Timeline'
import { Inspector } from './components/Inspector'
import { MemoryDrawer } from './components/MemoryDrawer'
import { Credits } from './components/Credits'

export function App() {
  const boot = useStore((s) => s.boot)
  useEffect(boot, [boot])

  // Easter egg: Ctrl+K (⌘K) twice opens the credits at #credits, so Back closes them too.
  const [credits, setCredits] = useState(location.hash === '#credits')
  useEffect(() => {
    let last = 0
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== 'k' || e.repeat) return
      e.preventDefault()
      const now = performance.now()
      if (now - last < 500) { location.hash = 'credits'; last = 0 } else last = now
    }
    const onHash = () => setCredits(location.hash === '#credits')
    window.addEventListener('keydown', onKey)
    window.addEventListener('hashchange', onHash)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('hashchange', onHash) }
  }, [])
  const closeCredits = () => { history.replaceState(null, '', location.pathname + location.search); setCredits(false) }

  return (
    <MotionConfig reducedMotion="user">
      <div className="app">
        <TopBar />
        <main className="layout">
          <ChatRail />
          <div className="center">
            <VideoWall />
            <Timeline />
          </div>
          <Inspector />
        </main>
      </div>
      <MemoryDrawer />
      {credits && <Credits onExit={closeCredits} />}
    </MotionConfig>
  )
}
