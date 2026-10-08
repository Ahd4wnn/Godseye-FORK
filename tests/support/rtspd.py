"""TEST FIXTURE ONLY: RTSP server that serves an existing H.264 mp4 at rtsp://127.0.0.1:<port>/cam (no re-encode)."""
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402

Gst.init(None)
server = GstRtspServer.RTSPServer()
server.set_address("127.0.0.1")
server.set_service(sys.argv[2])
factory = GstRtspServer.RTSPMediaFactory()
factory.set_launch(f"( filesrc location={sys.argv[1]} ! qtdemux ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )")
factory.set_shared(False)
server.get_mount_points().add_factory("/cam", factory)
server.attach(None)
print("ready", flush=True)
GLib.MainLoop().run()
