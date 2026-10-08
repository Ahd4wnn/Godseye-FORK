import { describe, expect, it } from 'vitest'
import { sseParser, type RawEvent } from './sse'

describe('sseParser', () => {
  it('parses events split across chunks and CRLF', () => {
    const out: RawEvent[] = []
    const feed = sseParser((e) => out.push(e))
    feed('event: stage\r\ndata: {"name":"retr')
    feed('ieve","ms":12}\r\n\r\nevent: done\ndata: {"total_ms":40,"stages":{}}\n\n: keepalive\n\n')
    expect(out).toEqual([
      { event: 'stage', data: { name: 'retrieve', ms: 12 } },
      { event: 'done', data: { total_ms: 40, stages: {} } },
    ])
  })
  it('defaults event name and keeps non-JSON data as text', () => {
    const out: RawEvent[] = []
    sseParser((e) => out.push(e))('data: hello\n\n')
    expect(out).toEqual([{ event: 'message', data: 'hello' }])
  })
})
