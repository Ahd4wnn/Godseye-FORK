import { motion } from 'motion/react'
import { EyeMark } from './EyeMark'

/** Shown over the video wall while the backend is looking for an answer, indexing or restarting. */
export function EyeLoader({ title, detail, progress }: { title: string; detail?: string; progress?: number }) {
  return (
    <motion.div
      className="eye-loader"
      role="status"
      aria-live="polite"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <motion.div className="eye-loader__sweep" animate={{ top: ['0%', '100%'] }} transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }} />
      <EyeMark size={72} searching />
      <p className="eye-loader__title">{title}</p>
      {progress !== undefined && (
        <div className="eye-loader__bar">
          <motion.i animate={{ width: `${Math.min(1, progress) * 100}%` }} transition={{ duration: 0.25, ease: 'easeOut' }} />
        </div>
      )}
      {detail && <p className="eye-loader__detail">{detail}</p>}
    </motion.div>
  )
}
