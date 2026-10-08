"""Local web chat (the architecture's Chat UI) over the Track B service. Security model: loopback bind only, per-install bearer
token (HttpOnly SameSite=Strict cookie), Host/Origin/X-Requested-With checks (DNS-rebinding + CSRF), request-size and rate
limits, media addressed by clip_id from SQLite (never by client-supplied path), no inline script, no HTML injection."""
from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..runtime.trackb import QueryService
from .assets import CSS, INDEX, JS
from .frontend_api import FrontendApi

log = logging.getLogger("godseye.web")
MAX_BODY = 8192
SID = re.compile(r"^[0-9a-f]{32}$")
SECURITY = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cross-Origin-Resource-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; "
                                       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"}


def load_or_create_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    tok = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    return tok


class WebApp:
    def __init__(self, qs: QueryService, store_root: Path, token: str, chat_per_min: int = 30, dist: Path | None = None):
        self.qs, self.root, self.token = qs, store_root, token
        self.dist = dist.resolve() if dist is not None and (dist / "index.html").is_file() else None      # built React console, if present
        self.clips_dir = (store_root / "clips").resolve()
        self.db = sqlite3.connect(f"file:{store_root / 'godseye.db'}?mode=ro", uri=True, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
        self._mu = threading.Lock()
        self._hits: dict[str, deque] = defaultdict(deque)
        self.chat_per_min = chat_per_min
        self.fe = FrontendApi(self)

    def authed(self, cookie_header: str | None, auth_header: str | None) -> bool:
        cand = None
        if auth_header and auth_header.startswith("Bearer "):
            cand = auth_header[7:]
        elif cookie_header:
            for part in cookie_header.split(";"):
                k, _, v = part.strip().partition("=")
                if k == "gs":
                    cand = v
        return bool(cand) and hmac.compare_digest(cand.encode(), self.token.encode())

    def rate_ok(self) -> bool:
        now = time.time()
        with self._mu:
            q = self._hits["chat"]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.chat_per_min:
                return False
            q.append(now)
            return True

    def q(self, sql: str, *a):
        with self._mu:
            return self.db.execute(sql, a).fetchall()

    # ---------------------------------------------------------------- JSON views
    def evidence_json(self, ev) -> dict:
        seek = 0.0
        if ev.clip_id is not None and ev.clip_frame is not None:
            r = self.q("SELECT fps FROM clips WHERE clip_id=?", ev.clip_id)
            seek = ev.clip_frame / r[0]["fps"] if r else 0.0
        return {"camera_id": ev.camera_id, "ts": ev.ts, "confidence": ev.confidence, "clip_id": ev.clip_id, "clip_frame": ev.clip_frame,
                "clip_status": ev.clip_status, "global_id": ev.global_id, "seek_s": round(seek, 2)}

    def chat(self, body: dict) -> dict:
        sid = body.get("session") if isinstance(body.get("session"), str) and SID.match(body.get("session", "")) else None
        if isinstance(body.get("text"), str):
            sid, res = self.qs.ask(sid, body["text"])
        elif isinstance(body.get("cid"), str) and sid and "response" in body:
            resp = body["response"]
            if not (isinstance(resp, str) and len(resp) <= 500) and not (isinstance(resp, dict) and len(resp) <= 12 and all(
                    isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in resp.items())):
                raise ValueError("bad response")
            sid, res = self.qs.answer_clarification(sid, body["cid"], resp)
        else:
            raise ValueError("send {text} or {session, cid, response}")
        out = {"session": sid, "kind": res.kind, "message": res.message, "elapsed_ms": round(res.elapsed_ms)}
        if res.answer is not None:
            a = res.answer
            out.update(status=a.status.value, text=a.text, warnings=a.warnings, evidence=[self.evidence_json(e) for e in a.evidence])
        if res.clarification is not None:
            out["clarification"] = res.clarification.model_dump(mode="json")
        return out


def make_server(app: WebApp, port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the web chat binds loopback only (no TLS / multi-user auth); front it with a VPN or reverse proxy if remote access is needed")
    srv = ThreadingHTTPServer((host, port), type("H", (Handler,), {"app": app}))
    srv.daemon_threads = True
    return srv


class Handler(BaseHTTPRequestHandler):
    app: WebApp
    protocol_version = "HTTP/1.1"
    server_version = "godseye"
    sys_version = ""

    def log_message(self, *a):                       # never log URLs/bodies (tokens, questions)
        pass

    # ------------------------------------------------------------ helpers
    def _hdrs(self, extra=None):
        for k, v in {**SECURITY, **(extra or {})}.items():
            self.send_header(k, v)

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._hdrs({"Cache-Control": "no-store", **(extra or {})})
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj, extra=None):
        self._send(code, json.dumps(obj).encode(), extra=extra)

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")

    def _guard(self, write: bool) -> bool:
        if not self._host_ok():
            self._json(403, {"error": "bad host"})
            return False
        if write:
            o = self.headers.get("Origin")
            if self.headers.get("X-Requested-With") != "godseye" or (o and o not in (f"http://{self.headers.get('Host')}",)):
                self._json(403, {"error": "cross-site request refused"})
                return False
        return True

    def _auth(self) -> bool:
        if self.app.authed(self.headers.get("Cookie"), self.headers.get("Authorization")):
            return True
        self._json(401, {"error": "unauthorised: open the login link printed by the server"})
        return False

    # ------------------------------------------------------------ routes
    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:                            # noqa: BLE001 - no stack traces to clients
            log.exception("GET failed")
            self._json(500, {"error": "internal error"})

    do_HEAD = do_GET

    def _get(self):
        if not self._guard(False):
            return
        u = urlparse(self.path)
        if u.path == "/healthz":
            return self._json(200, {"ok": True})
        if u.path == "/" and "token" in parse_qs(u.query):                      # one-time login link -> cookie, then redirect
            tok = parse_qs(u.query)["token"][0]
            if not hmac.compare_digest(tok.encode(), self.app.token.encode()):
                return self._json(401, {"error": "bad token"})
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", f"gs={tok}; HttpOnly; SameSite=Strict; Path=/")
            self.send_header("Content-Length", "0")
            self._hdrs()
            self.end_headers()
            return
        if u.path == "/app.js":
            return self._send(200, JS.encode(), "text/javascript; charset=utf-8")
        if u.path == "/app.css":
            return self._send(200, CSS.encode(), "text/css; charset=utf-8")
        if not self._auth():
            return
        if u.path == "/legacy":
            return self._send(200, INDEX.encode(), "text/html; charset=utf-8")
        if u.path == "/" and self.app.dist is None:
            return self._send(200, INDEX.encode(), "text/html; charset=utf-8")
        fe = {"/api/health": self.app.fe.health, "/api/site": self.app.fe.site, "/api/cameras": self.app.fe.cameras, "/api/memory": self.app.fe.facts}
        if u.path in fe:
            return self._json(200, fe[u.path]())
        m = re.fullmatch(r"/media/clip/(\d{1,9})", u.path)
        if m:
            return self._media(int(m[1]))
        m = re.fullmatch(r"/api/frame/(\d{1,9})/(\d{1,6})\.jpg", u.path)
        if m:
            return self._frame(int(m[1]), int(m[2]))
        m = re.fullmatch(r"/api/clip/(\d{1,9})", u.path)
        if m:
            r = self.app.q("SELECT clip_id,camera_id,frames,fps,width,height,frame_ts FROM clips WHERE clip_id=? AND status='complete'", int(m[1]))
            if not r:
                return self._json(404, {"error": "no such clip"})
            d = dict(r[0]); d["frame_ts"] = json.loads(d["frame_ts"])
            return self._json(200, d)
        if u.path == "/api/legacy/cameras":
            rows = self.app.q("SELECT c.camera_id, (SELECT clip_id FROM clips k WHERE k.camera_id=c.camera_id AND k.status='complete' ORDER BY end_ts DESC LIMIT 1) AS clip_id "
                              "FROM cameras c ORDER BY c.camera_id")
            return self._json(200, {"cameras": [dict(r) for r in rows if r["clip_id"] is not None]})
        if self.app.dist is not None and not u.path.startswith("/api/"):
            return self._static(u.path)
        self._json(404, {"error": "not found"})

    def _static(self, path: str):
        """The built console: only files under frontend/dist; unknown paths fall back to index.html (client-side app)."""
        p = (self.app.dist / path.lstrip("/")).resolve() if path != "/" else self.app.dist / "index.html"
        if not (p.is_file() and p.is_relative_to(self.app.dist)):
            if "." in path.rsplit("/", 1)[-1]:
                return self._json(404, {"error": "not found"})
            p = self.app.dist / "index.html"
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        cache = "no-store" if p.name == "index.html" else "private, max-age=31536000, immutable"
        self._send(200, p.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ""), {"Cache-Control": cache})

    def do_DELETE(self):
        try:
            if not self._guard(True) or not self._auth():
                return
            m = re.fullmatch(r"/api/memory/([^/]{1,200})", urlparse(self.path).path)
            if not m:
                return self._json(404, {"error": "not found"})
            from urllib.parse import unquote
            if not self.app.fe.forget(unquote(m[1])):
                return self._json(404, {"error": "no such fact"})
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._hdrs()
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:                            # noqa: BLE001
            log.exception("DELETE failed")
            self._json(500, {"error": "internal error"})

    def _sse(self, body: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self._hdrs({"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.end_headers()
        self.close_connection = True

        def emit(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        stop = threading.Event()

        def beat():                                  # keep-alive comments while the (slow) LLM works
            while not stop.wait(10):
                try:
                    self.wfile.write(b": keep-alive\n\n"); self.wfile.flush()
                except OSError:
                    return
        threading.Thread(target=beat, daemon=True).start()
        try:
            self.app.fe.query(body, emit)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as e:
            emit("error", {"message": f"bad request: {str(e)[:100]}"})
        except Exception:                            # noqa: BLE001
            log.exception("query stream failed")
            emit("error", {"message": "internal error"})
        finally:
            stop.set()

    def do_POST(self):
        try:
            if not self._guard(True) or not self._auth():
                return
            path = urlparse(self.path).path
            if path in ("/api/videos", "/api/videos/links", "/api/admin/restart"):
                return self._json(501, {"error": "not available in Godseye: cameras are ingested by Track A (config/ingest.yaml); restart the services from the shell"})
            if path not in ("/api/chat", "/api/query", "/api/clarify"):
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                return self._json(413, {"error": "body too large"})
            if not self.app.rate_ok():
                return self._json(429, {"error": "too many questions; wait a moment"}, {"Retry-After": "10"})
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            if path == "/api/query":
                return self._sse(body)
            if path == "/api/clarify":
                fact = self.app.fe.clarify(body)
                if fact is None:
                    self.send_response(204); self.send_header("Content-Length", "0"); self._hdrs(); self.end_headers()
                    return
                return self._json(200, fact)
            self._json(200, self.app.chat(body))
        except LookupError as e:
            self._json(404, {"error": str(e)})
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"bad request: {str(e)[:100]}"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:                            # noqa: BLE001
            log.exception("POST failed")
            self._json(500, {"error": "internal error"})

    # ------------------------------------------------------------ media
    def _clip_file(self, clip_id: int):
        r = self.app.q("SELECT path,size_bytes,status FROM clips WHERE clip_id=?", clip_id)
        if not r or r[0]["status"] != "complete":
            return None, (404, "no such clip")
        p = (self.app.clips_dir / r[0]["path"]).resolve()
        if not p.is_relative_to(self.app.clips_dir):                             # defence in depth: DB value must stay inside clips/
            log.error("clip %d path escapes the clips directory", clip_id)
            return None, (404, "no such clip")
        if not p.is_file():
            return None, (410, "clip file is missing")
        if p.stat().st_size != r[0]["size_bytes"]:
            return None, (410, "clip file is damaged")
        return (p, r[0]["size_bytes"]), None

    def _media(self, clip_id: int):
        got, err = self._clip_file(clip_id)
        if err:
            return self._json(err[0], {"error": err[1]})
        p, size = got
        a, b, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or not (m[1] or m[2]):
                return self._json(416, {"error": "bad range"}, {"Content-Range": f"bytes */{size}"})
            if m[1]:
                a = int(m[1]); b = min(int(m[2]), size - 1) if m[2] else size - 1
            else:
                a = max(0, size - int(m[2]))
            if a > b or a >= size:
                return self._json(416, {"error": "bad range"}, {"Content-Range": f"bytes */{size}"})
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self._hdrs({"Cache-Control": "private, max-age=3600", "Content-Disposition": "inline"})
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _frame(self, clip_id: int, n: int):
        got, err = self._clip_file(clip_id)
        if err:
            return self._json(err[0], {"error": err[1]})
        import cv2
        frames = self.app.q("SELECT frames FROM clips WHERE clip_id=?", clip_id)[0]["frames"]
        if n >= frames:
            return self._json(404, {"error": "no such frame"})
        cap = cv2.VideoCapture(str(got[0]))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, img = cap.read()
        finally:
            cap.release()
        if not ok:
            return self._json(410, {"error": "frame could not be decoded"})
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._send(200, buf.tobytes(), "image/jpeg")


def serve(rcfg, host: str = "127.0.0.1", port: int = 8080) -> int:
    from ..query.config import load_config as load_query
    from ..runtime.factory import ROOT, build_track_b
    logging.basicConfig(level=logging.INFO)
    qs = build_track_b(rcfg)
    root = Path(load_query(ROOT / rcfg.query).store_root)
    root = root if root.is_absolute() else ROOT / root
    tok = load_or_create_token(root / "web_token")
    srv = make_server(WebApp(qs, root, tok, dist=ROOT / "frontend" / "dist"), port, host)
    ui = "frontend/dist (React console; old UI at /legacy)" if (ROOT / "frontend" / "dist" / "index.html").is_file() else "legacy UI (run `npm run build` in frontend/ for the React console)"
    print(f"Godseye web chat, {ui}: http://{host}:{port}/?token={tok}   (token file: {root / 'web_token'}, mode 0600)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
