import type { ReactNode } from 'react'

/** The one section header used everywhere, so every column's hairline lines up. */
export function Head({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <header className="head">
      <h2 className="head__title">{title}</h2>
      {children && <div className="head__meta">{children}</div>}
    </header>
  )
}
