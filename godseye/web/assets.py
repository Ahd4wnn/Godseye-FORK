"""Static UI. No inline script, no external resources; all dynamic text is inserted with textContent (no HTML injection)."""
INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Godseye</title><link rel="stylesheet" href="/app.css"></head><body>
<header><h1>Godseye</h1><span id="state">ready</span></header>
<main><div id="log" aria-live="polite"></div><div id="picker" hidden></div>
<form id="form"><input id="q" autocomplete="off" maxlength="500" placeholder="Ask about the cameras, e.g. how many people were on lobby in the last hour?" required>
<button>Ask</button></form></main><script src="/app.js"></script></body></html>"""

CSS = """:root{color-scheme:light dark;--bg:#fff;--fg:#1b1f24;--mut:#59636e;--card:#f3f5f7;--acc:#0b5cd5;--warn:#9a6700;--err:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--fg:#e6edf3;--mut:#9aa7b4;--card:#171b22;--acc:#6aa7ff;--warn:#e3b341;--err:#ff7b72}}
*{box-sizing:border-box}body{margin:0;font:16px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;flex-direction:column;min-height:100vh}
header{display:flex;gap:12px;align-items:baseline;padding:12px 16px;border-bottom:1px solid var(--card)}h1{margin:0;font-size:18px}#state{color:var(--mut);font-size:13px}
main{flex:1;display:flex;flex-direction:column;max-width:880px;width:100%;margin:0 auto;padding:0 16px}#log{flex:1;padding:16px 0}
.msg{margin:10px 0;padding:10px 12px;border-radius:8px;background:var(--card);white-space:pre-wrap;overflow-wrap:anywhere}.me{background:transparent;border:1px solid var(--card)}
.status{font-weight:600;font-size:12px;letter-spacing:.04em}.OK{color:var(--acc)}.PARTIAL_EVIDENCE,.AMBIGUOUS_RESULT{color:var(--warn)}.NO_RESULT{color:var(--mut)}.SYSTEM_ERROR,.error{color:var(--err)}
form{display:flex;gap:8px;padding:12px 0;position:sticky;bottom:0;background:var(--bg)}input{flex:1;padding:10px;border-radius:8px;border:1px solid var(--mut);background:var(--bg);color:var(--fg);font:inherit}
button{padding:8px 14px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:#fff;font:inherit;cursor:pointer}button.alt{background:transparent;color:var(--acc)}
.opts{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}video,img.frame{max-width:100%;border-radius:8px;margin-top:8px;display:block}img.frame{cursor:crosshair}
.ev{font-size:13px;color:var(--mut);margin-top:6px}:focus-visible{outline:2px solid var(--acc);outline-offset:2px}"""

JS = """'use strict';
const $ = id => document.getElementById(id);
let sid = null, pending = null;
function el(tag, cls, text){ const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function add(node){ $('log').appendChild(node); node.scrollIntoView({block:'end'}); }
async function api(path, body){
  const r = await fetch(path, {method: body ? 'POST' : 'GET', headers: {'X-Requested-With':'godseye','Content-Type':'application/json'}, body: body ? JSON.stringify(body) : undefined});
  if (!r.ok) throw new Error((await r.json().catch(()=>({error:r.statusText}))).error || r.statusText);
  return r.json();
}
async function send(body){
  $('state').textContent = 'thinking…';
  try { const out = await api('/api/chat', {session: sid, ...body}); sid = out.session; render(out); $('state').textContent = 'ready'; }
  catch(e){ add(el('div','msg error', 'Error: ' + e.message)); $('state').textContent = 'error'; }
}
function render(out){
  const m = el('div','msg');
  if (out.kind === 'answer'){
    m.appendChild(el('div','status ' + out.status, out.status.replace('_',' ')));
    m.appendChild(el('div', '', out.text));
    for (const ev of out.evidence){
      const d = el('div','ev', `${ev.camera_id} · ${new Date(ev.ts*1000).toISOString()} · confidence ${ev.confidence.toFixed(2)} · ${ev.clip_status}`);
      m.appendChild(d);
      if (ev.clip_id){ const v = el('video'); v.controls = true; v.preload = 'none'; v.src = `/media/clip/${ev.clip_id}#t=${ev.seek_s}`; m.appendChild(v); }
    }
  } else if (out.kind === 'clarify'){ clarify(m, out.clarification); }
  else { m.appendChild(el('div','error', out.message || out.kind)); }
  add(m);
}
function clarify(m, c){
  m.appendChild(el('div','', c.prompt)); pending = c;
  if (c.kind === 'multiple_choice' || c.kind === 'camera_grid'){
    const o = el('div','opts'); for (const op of c.options){ const b = el('button','alt', op.label); b.onclick = () => { add(el('div','msg me', op.label)); send({cid:c.cid, response:op.value}); }; o.appendChild(b); } m.appendChild(o);
  } else if (c.kind === 'point_and_click' || c.kind === 'region_selection'){ const b = el('button','alt', 'Choose on a camera view'); b.onclick = () => picker(c); m.appendChild(b); }
}
async function picker(c){
  const p = $('picker'); p.hidden = false; p.replaceChildren();
  const cams = (await api('/api/legacy/cameras')).cameras.filter(x => !c.camera_id || x.camera_id === c.camera_id);
  if (!cams.length){ p.appendChild(el('div','msg error','No recorded video is available to click on yet.')); return; }
  const cam = cams[0], clip = await api(`/api/clip/${cam.clip_id}`);
  let n = Math.floor(clip.frames / 2), first = null;
  const img = el('img','frame'); img.alt = 'camera view: click to choose'; img.src = `/api/frame/${clip.clip_id}/${n}.jpg`;
  p.appendChild(el('div','', `${cam.camera_id}: ` + (c.kind === 'point_and_click' ? 'click the person or vehicle' : 'click two opposite corners of the area')));
  const slider = el('input'); slider.type = 'range'; slider.min = 0; slider.max = clip.frames - 1; slider.value = n; slider.setAttribute('aria-label','frame');
  slider.oninput = () => { n = +slider.value; img.src = `/api/frame/${clip.clip_id}/${n}.jpg`; };
  img.onclick = ev => {
    const r = img.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height;
    if (c.kind === 'point_and_click'){ p.hidden = true; add(el('div','msg me','(clicked on camera view)')); send({cid:c.cid, response:{camera_id:cam.camera_id, x, y, ts:clip.frame_ts[n], frame_w:clip.width, frame_h:clip.height}}); }
    else if (!first){ first = {x, y}; } else { p.hidden = true; add(el('div','msg me','(marked an area)')); send({cid:c.cid, response:{camera_id:cam.camera_id, x1:first.x, y1:first.y, x2:x, y2:y}}); }
  };
  p.append(img, slider);
}
$('form').onsubmit = e => {
  e.preventDefault(); const t = $('q').value.trim(); if (!t) return; $('q').value = ''; add(el('div','msg me', t)); $('picker').hidden = true;
  if (pending && pending.kind === 'free_text'){ const c = pending; pending = null; send({cid:c.cid, response:t}); } else { pending = null; send({text:t}); }
};
"""
