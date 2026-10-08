"""One GStreamer pipeline + worker thread per camera: connect, pull, sample, preprocess, publish, reconnect.

Pipeline: <src> ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink
  rtsp: rtspsrc | rtmp: rtmp2src | hls: souphttpsrc (decodebin autoplugs hlsdemux2) | file: filesrc
"""
from __future__ import annotations

import copy
import logging
import random
import threading
import time
from typing import Callable

import numpy as np

from ..gst_env import init_gst
from .config import CameraConfig, redact_text, redact_url
from .preprocess import Preprocessor
from .sampler import FrameSampler
from .types import CameraState, CameraStats, PreprocessedFrame

log = logging.getLogger("godseye.ingest")

HW_DECODERS = {"nvcodec": ("nvh264dec", "nvh265dec"), "vaapi": ("vah264dec", "vah265dec")}
H264_DECODERS = ("avdec_h264", "nvh264dec", "vah264dec", "openh264dec", "v4l2slh264dec")
SOURCE_ELEMENTS = {"rtsp": ("rtspsrc",), "rtmp": ("rtmp2src",), "hls": ("souphttpsrc", "hlsdemux2"), "file": ("filesrc",)}
PIPELINES = {
    "rtsp": "rtspsrc name=src protocols={tr} latency={lat} ! decodebin name=dec",
    "rtmp": "rtmp2src name=src ! decodebin name=dec",
    "hls": "souphttpsrc name=src ! decodebin3 name=dec",  # hlsdemux2 requires a streams-aware parent => decodebin3
    "file": "filesrc name=src ! decodebin name=dec",
}
TAIL = " dec. ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false max-buffers=2 drop=true sync={sync}"


class StreamError(RuntimeError):
    """Retryable: this connection attempt failed."""


class FatalCameraError(RuntimeError):
    """Not retryable: configuration/environment cannot support this camera."""


class IngestEnvironmentError(RuntimeError):
    """A required GStreamer element is missing."""


def check_environment(Gst, protocols: set[str]) -> None:
    missing = []
    for e in ("decodebin", "videoconvert", "appsink"):
        if Gst.ElementFactory.find(e) is None:
            missing.append(e)
    for p in protocols:
        missing += [e for e in SOURCE_ELEMENTS[p] if Gst.ElementFactory.find(e) is None]
    if not any(Gst.ElementFactory.find(d) for d in H264_DECODERS):
        missing.append("an H.264 decoder (one of %s)" % ", ".join(H264_DECODERS))
    if missing:
        raise IngestEnvironmentError(f"missing GStreamer elements: {missing}")


class CameraPipeline:
    def __init__(self, cfg: CameraConfig, publish: Callable[[PreprocessedFrame], str], gst=None):
        self.cfg = cfg
        self._publish = publish
        self._Gst = gst or init_gst()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stats = CameraStats()
        self._rng = random.Random()
        self._pre = Preprocessor(cfg.preprocess)
        self._seq_sampled = 0
        self._seq_source = 0
        self._decoders: list[str] = []
        self._malformed_logged = 0
        self._epoch_delivered = False

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"camera {self.cfg.camera_id} already started")
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cfg.camera_id}", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 10.0) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def stop(self, timeout: float = 10.0) -> bool:
        self.request_stop()
        return self.join(timeout)

    def stats(self) -> CameraStats:
        with self._lock:
            return copy.deepcopy(self._stats)

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._stats, k, v)

    def _bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self._stats, name, getattr(self._stats, name) + n)

    # ---- worker ----------------------------------------------------------------------------
    def _run(self) -> None:
        r = self.cfg.reconnect
        failures = 0
        try:
            while not self._stop.is_set():
                self._set(state=CameraState.CONNECTING)
                try:
                    delivered = self._session()
                    if self._stats.state == CameraState.ENDED:
                        return
                    err = None
                except StreamError as e:
                    delivered, err = self._epoch_delivered, redact_text(str(e))
                except FatalCameraError as e:
                    log.error("camera %s fatal: %s", self.cfg.camera_id, redact_text(str(e)))
                    self._set(state=CameraState.FAILED, last_error=redact_text(str(e)))
                    return
                except Exception as e:  # unexpected: keep the camera alive but never hide it
                    log.error("camera %s unexpected error: %s", self.cfg.camera_id, redact_text(repr(e)))
                    log.debug("traceback (may contain the source URL)", exc_info=True)
                    delivered, err = False, redact_text(f"unexpected: {e!r}")
                if self._stop.is_set():
                    break
                failures = 0 if delivered else failures + 1
                if err:
                    self._set(last_error=err)
                    log.warning("camera %s stream failed: %s", self.cfg.camera_id, err)
                if r.max_attempts and failures >= r.max_attempts:
                    self._set(state=CameraState.FAILED, last_error=f"gave up after {failures} failed attempts: {err}")
                    return
                delay = min(r.max_backoff_s, r.initial_backoff_s * 2 ** max(failures - 1, 0))
                delay *= self._rng.uniform(0.8, 1.2)
                self._set(state=CameraState.RECONNECTING)
                with self._lock:
                    self._stats.reconnects += 1
                self._stop.wait(delay)
        finally:
            if self._stats.state not in (CameraState.FAILED, CameraState.ENDED):
                self._set(state=CameraState.STOPPED)

    def _build(self):
        Gst, c = self._Gst, self.cfg
        if c.hw_accel != "none":
            found = [Gst.ElementFactory.find(n) for n in HW_DECODERS[c.hw_accel]]
            found = [f for f in found if f]
            if not found:
                raise FatalCameraError(f"hw_accel={c.hw_accel} requested but none of {HW_DECODERS[c.hw_accel]} is available")
            for f in found:
                f.set_rank(Gst.Rank.PRIMARY + 100)
        proto = c.protocol
        desc = PIPELINES[proto].format(tr=c.rtsp_transport, lat=c.latency_ms) + TAIL.format(sync="true" if proto == "file" else "false")
        try:
            pipe = Gst.parse_launch(desc)
        except Exception as e:
            raise FatalCameraError(f"cannot build pipeline: {e}") from e
        src = pipe.get_by_name("src")
        src.set_property("location", c.url[len("file://"):] if proto == "file" else c.url)  # property, never string-built
        self._decoders = []

        def on_added(_bin, _sub, element):
            f = element.get_factory()
            if f and "Decoder" in f.get_metadata("klass") and "Video" in f.get_metadata("klass"):
                self._decoders.append(f.get_name())
        pipe.connect("deep-element-added", on_added)
        return pipe, pipe.get_by_name("sink")

    def _session(self) -> bool:
        """Run one connection until failure/stop/EOS. Returns True if it delivered frames."""
        Gst, c = self._Gst, self.cfg
        pipe, sink = self._build()
        bus = pipe.get_bus()
        sampler = FrameSampler(c.sampling)
        self._epoch_delivered = False
        t0 = last = time.monotonic()
        try:
            if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise StreamError("pipeline failed to enter PLAYING")
            while not self._stop.is_set():
                msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None:
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        raise StreamError(f"{err.message} ({(dbg or '').splitlines()[0] if dbg else ''})")
                    if not self._on_eos(pipe):
                        return self._epoch_delivered
                    continue
                sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)  # action signal: GstApp typelib not required
                now = time.monotonic()
                if sample is None:
                    if not self._epoch_delivered and now - t0 > c.reconnect.connect_timeout_s:
                        raise StreamError(f"no frame within connect_timeout_s={c.reconnect.connect_timeout_s}")
                    if self._epoch_delivered and now - last > c.reconnect.stall_timeout_s:
                        raise StreamError(f"stalled: no frame for {c.reconnect.stall_timeout_s}s")
                    continue
                last = now
                self._on_sample(sample, sampler, now)
            return self._epoch_delivered
        finally:
            pipe.set_state(Gst.State.NULL)

    def _on_eos(self, pipe) -> bool:
        """True = keep running (file looped); False = session over (file ended). Live EOS raises."""
        Gst = self._Gst
        if self.cfg.protocol != "file":
            raise StreamError("EOS on live stream")
        if self.cfg.loop_file and pipe.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0):
            return True
        self._set(state=CameraState.ENDED)
        return False

    def _on_sample(self, sample, sampler: FrameSampler, mono: float) -> None:
        Gst = self._Gst
        buf = sample.get_buffer()
        st = sample.get_caps().get_structure(0)
        w, h = st.get_value("width"), st.get_value("height")
        if not self._epoch_delivered:
            self._first_frame()
        with self._lock:
            self._stats.frames_decoded += 1
            self._stats.last_frame_ts = time.time()
        self._seq_source += 1
        if buf.has_flags(Gst.BufferFlags.CORRUPTED):
            return self._malformed("buffer flagged CORRUPTED")
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return self._malformed("buffer map failed")
        try:
            stride = (w * 3 + 3) & ~3
            if not w or not h or mi.size < stride * h:
                return self._malformed(f"buffer {mi.size}B too small for {w}x{h} BGR")
            view = np.ndarray((h, w, 3), np.uint8, buffer=mi.data, strides=(stride, 3, 1))
            d = sampler.offer(view, mono)
            if not d.emit:
                return self._bump("dropped_rate" if d.reason == "rate" else "dropped_redundant")
            try:
                image, tf, original = self._pre(view)
            except Exception as e:  # cv2.error / ValueError on bad pixel data
                return self._malformed(f"preprocess failed: {e}")
        finally:
            buf.unmap(mi)
        pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else None
        frame = PreprocessedFrame(
            camera_id=self.cfg.camera_id, frame_seq=self._seq_sampled, source_seq=self._seq_source,
            stream_epoch=self._stats.stream_epoch, capture_ts=time.time(), pts_ns=pts, image=image,
            transform=tf, original_bgr=original, sample_reason=d.reason, motion_score=d.motion,
            privacy_masked=bool(self.cfg.preprocess.privacy_masks),
            original_privacy_masked=bool(self.cfg.preprocess.privacy_masks and self.cfg.preprocess.privacy_apply_to_original))
        self._seq_sampled += 1
        self._bump("frames_sampled")
        if self._publish(frame) != "ok":
            self._bump("dropped_queue")

    def _first_frame(self) -> None:
        c = self.cfg
        if c.hw_accel != "none" and not any(d.startswith(HW_DECODERS[c.hw_accel][0][:2]) for d in self._decoders):
            raise FatalCameraError(f"hw_accel={c.hw_accel} requested but decoders actually used: {self._decoders}")
        self._epoch_delivered = True
        with self._lock:
            self._stats.stream_epoch += 1
            self._stats.decoder = ",".join(self._decoders) or None
            self._stats.state = CameraState.STREAMING
        log.info("camera %s streaming (%s) decoder=%s", c.camera_id, redact_url(c.url), self._decoders)

    def _malformed(self, why: str) -> None:
        self._bump("malformed")
        why = redact_text(why)
        self._set(last_error=why)
        if self._malformed_logged < 5 or self._stats.malformed % 100 == 0:
            self._malformed_logged += 1
            log.warning("camera %s malformed frame #%d: %s", self.cfg.camera_id, self._stats.malformed, why)
