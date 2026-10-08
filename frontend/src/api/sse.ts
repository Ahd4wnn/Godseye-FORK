export type RawEvent = { event: string; data: unknown }

// Incremental text/event-stream parser. EventSource can't POST, so /query is read with fetch.
export function sseParser(onEvent: (e: RawEvent) => void) {
  let buf = ''
  return (chunk: string) => {
    buf += chunk.replace(/\r\n?/g, '\n')
    let end: number
    while ((end = buf.indexOf('\n\n')) !== -1) {
      const block = buf.slice(0, end)
      buf = buf.slice(end + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
      }
      if (!data.length) continue
      const text = data.join('\n')
      let parsed: unknown = text
      try { parsed = JSON.parse(text) } catch { /* plain-text data stays a string */ }
      onEvent({ event, data: parsed })
    }
  }
}
