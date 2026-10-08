import { useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useStore } from '../store'
import { Head } from './Head'

type FileRow = { id: string; file: File; tag: string }
type LinkRow = { id: string; url: string; tag: string }
type Step = 'choose' | 'files' | 'links'

const TITLES: Record<Step, string> = { choose: 'Add footage', files: 'From this device', links: 'From links' }
const uid = () => crypto.randomUUID()
const stem = (name: string) => name.replace(/\.[^.]+$/, '')
const isUrl = (s: string) => {
  try {
    return ['http:', 'https:'].includes(new URL(s.trim()).protocol)
  } catch {
    return false
  }
}
const size = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(0.1, b / 1e6).toFixed(1)} MB`)

/** First problem with the rows, in the order a person would fix them. */
function problem(rows: { tag: string; url?: string }[], kind: 'files' | 'links') {
  if (!rows.length) return kind === 'files' ? 'Add at least one video.' : 'Add at least one link.'
  if (kind === 'links' && rows.some((r) => !isUrl(r.url ?? ''))) return 'Every link must be a full http(s) URL.'
  if (rows.some((r) => !r.tag.trim())) return 'Every video needs a camera tag.'
  const tags = rows.map((r) => r.tag.trim().toLowerCase())
  if (new Set(tags).size !== tags.length) return 'Camera tags must be unique.'
  return null
}

const Remove = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <button type="button" className="icon-btn" onClick={onClick} aria-label={label}>
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
  </button>
)

function Option({ icon, title, text, onClick }: { icon: ReactNode; title: string; text: string; onClick: () => void }) {
  return (
    <button type="button" className="option" onClick={onClick}>
      <span className="option__icon">{icon}</span>
      <span className="option__text">
        <span className="option__title">{title}</span>
        <span>{text}</span>
      </span>
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
  )
}

export function UploadDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const upload = useStore((s) => s.upload)
  const [step, setStep] = useState<Step>('choose')
  const [files, setFiles] = useState<FileRow[]>([])
  const [links, setLinks] = useState<LinkRow[]>([])
  const [tried, setTried] = useState(false)
  const [dragging, setDragging] = useState(false)
  const picker = useRef<HTMLInputElement>(null)

  // Fresh form each time it opens; never on re-render, or typed rows would vanish on the next health poll.
  useEffect(() => {
    if (open) { setStep('choose'); setFiles([]); setLinks([]); setTried(false) }
  }, [open])
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const addFiles = (list: Iterable<File>) => {
    const fresh = [...list].filter((f) => f.type.startsWith('video/'))
    setFiles((rows) => [
      ...rows,
      ...fresh
        .filter((f) => !rows.some((r) => r.file.name === f.name && r.file.size === f.size && r.file.lastModified === f.lastModified))
        .map((file) => ({ id: uid(), file, tag: stem(file.name) })),
    ])
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  const rows = step === 'files' ? files : links
  const err = step === 'choose' ? null : problem(rows, step)
  const n = rows.length
  const tagged = new Set(rows.map((r) => r.tag.trim().toLowerCase()).filter(Boolean)).size
  const submit = () => {
    setTried(true)
    if (err || step === 'choose') return
    upload(
      step === 'files'
        ? { kind: 'files', items: files.map((r) => ({ file: r.file, tag: r.tag.trim() })) }
        : { kind: 'links', items: links.map((r) => ({ url: r.url.trim(), tag: r.tag.trim() })) },
    )
    onClose()
  }
  const dupTag = (tag: string) => !!tag.trim() && rows.filter((r) => r.tag.trim().toLowerCase() === tag.trim().toLowerCase()).length > 1
  const tagInvalid = (tag: string) => tried && (!tag.trim() || dupTag(tag))

  return (
    <AnimatePresence>
      {open && (
        <div className="modal" role="presentation">
          <motion.div className="scrim" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }} />
          <motion.form
            className="dialog"
            role="dialog"
            aria-modal="true"
            aria-label={TITLES[step]}
            noValidate
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            onSubmit={(e) => { e.preventDefault(); submit() }}
            onDragOver={step === 'files' ? (e) => { e.preventDefault(); setDragging(true) } : undefined}
            onDragLeave={step === 'files' ? (e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false) } : undefined}
            onDrop={step === 'files' ? onDrop : undefined}
          >
            <Head title={TITLES[step]}>
              <Remove label="Close" onClick={onClose} />
            </Head>

            {step === 'choose' && (
              <div className="dialog__body dialog__body--choose">
                <Option
                  title="From this device"
                  text="One video or a whole building's CCTV. Tag each file with its camera."
                  onClick={() => setStep('files')}
                  icon={<svg width="18" height="18" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 10.5V2.5M4.5 6 8 2.5 11.5 6M2.5 10v3.5h11V10" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                />
                <Option
                  title="From links"
                  text="Paste video URLs. The backend downloads and indexes them."
                  onClick={() => { setLinks([{ id: uid(), url: '', tag: '' }]); setStep('links') }}
                  icon={<svg width="18" height="18" viewBox="0 0 16 16" aria-hidden="true"><path d="M6.8 9.2a2.8 2.8 0 0 0 4 0l2.2-2.2a2.8 2.8 0 0 0-4-4l-.8.8M9.2 6.8a2.8 2.8 0 0 0-4 0L3 9a2.8 2.8 0 0 0 4 4l.8-.8" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>}
                />
              </div>
            )}

            {step === 'files' && (
              <div className="dialog__body">
                <input ref={picker} type="file" accept="video/*" multiple hidden onChange={(e) => { addFiles(e.target.files ?? []); e.target.value = '' }} />
                <button type="button" className={`dropzone${dragging ? ' dropzone--on' : ''}${files.length ? ' dropzone--compact' : ''}`} onClick={() => picker.current?.click()} autoFocus>
                  <span>{dragging ? 'Drop to add' : files.length ? 'Drop more videos here or ' : 'Drop videos here or '}</span>
                  {!dragging && <span className="dropzone__link">choose files</span>}
                </button>
                {files.length > 0 && (
                  <div className="rows" role="table" aria-label="Videos to upload">
                    <div className="rows__head" role="row"><span role="columnheader">Video</span><span role="columnheader">Camera tag</span><span /></div>
                    {files.map((r) => (
                      <div className="rows__row" role="row" key={r.id}>
                        <span className="rows__file" role="cell" title={r.file.name}>
                          <span>{r.file.name}</span>
                          <span className="rows__meta">{size(r.file.size)}</span>
                        </span>
                        <input
                          className="field"
                          role="cell"
                          value={r.tag}
                          onChange={(e) => setFiles((rs) => rs.map((x) => (x.id === r.id ? { ...x, tag: e.target.value } : x)))}
                          aria-label={`Camera tag for ${r.file.name}`}
                          aria-invalid={tagInvalid(r.tag)}
                        />
                        <Remove label={`Remove ${r.file.name}`} onClick={() => setFiles((rs) => rs.filter((x) => x.id !== r.id))} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {step === 'links' && (
              <div className="dialog__body">
                <div className="rows" role="table" aria-label="Video links">
                  <div className="rows__head" role="row"><span role="columnheader">Video URL</span><span role="columnheader">Camera tag</span><span /></div>
                  {links.map((r, i) => (
                    <div className="rows__row" role="row" key={r.id}>
                      <input
                        className="field"
                        role="cell"
                        type="url"
                        inputMode="url"
                        placeholder="https://"
                        value={r.url}
                        onChange={(e) => setLinks((rs) => rs.map((x) => (x.id === r.id ? { ...x, url: e.target.value } : x)))}
                        aria-label={`Video URL ${i + 1}`}
                        aria-invalid={tried && !isUrl(r.url)}
                        autoFocus={i === links.length - 1}
                      />
                      <input
                        className="field"
                        role="cell"
                        value={r.tag}
                        onChange={(e) => setLinks((rs) => rs.map((x) => (x.id === r.id ? { ...x, tag: e.target.value } : x)))}
                        aria-label={`Camera tag ${i + 1}`}
                        aria-invalid={tagInvalid(r.tag)}
                      />
                      <Remove label={`Remove link ${i + 1}`} onClick={() => setLinks((rs) => rs.filter((x) => x.id !== r.id))} />
                    </div>
                  ))}
                </div>
                <button type="button" className="btn rows__add" onClick={() => setLinks((rs) => [...rs, { id: uid(), url: '', tag: '' }])}>
                  <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5v9M1.5 6h9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
                  Add link
                </button>
              </div>
            )}

            {step !== 'choose' && (
              <div className="dialog__foot">
                <p className={tried && err ? 'dialog__error' : undefined} role={tried && err ? 'alert' : undefined}>
                  {tried && err ? err : n ? `${n} video${n > 1 ? 's' : ''} · ${tagged} camera${tagged === 1 ? '' : 's'}` : ''}
                </p>
                <button type="button" className="btn" onClick={() => { setStep('choose'); setTried(false) }}>Back</button>
                <button type="submit" className="btn btn--primary">{n > 0 ? `Index ${n} video${n > 1 ? 's' : ''}` : 'Index'}</button>
              </div>
            )}
          </motion.form>
        </div>
      )}
    </AnimatePresence>
  )
}
