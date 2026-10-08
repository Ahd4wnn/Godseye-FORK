import pytest

from godseye.ingest.config import ConfigError, load_config, parse_camera, redact_url


def cam(**kw):
    return parse_camera({"camera_id": "c1", "url": "rtsp://h/x", **kw}, {}, 0)


def test_defaults_match_architecture_targets():
    c = cam()
    assert (c.sampling.base_fps, c.sampling.burst_fps, c.preprocess.target_size) == (2.0, 5.0, 640)
    assert c.protocol == "rtsp"


@pytest.mark.parametrize("url,proto", [("rtmp://h/a/b", "rtmp"), ("https://h/i.m3u8", "hls"), ("file:///x.mp4", "file")])
def test_protocols(url, proto):
    assert parse_camera({"camera_id": "c", "url": url}, {}, 0).protocol == proto


@pytest.mark.parametrize("bad", [
    {"url": "ftp://h/x"}, {"camera_id": "has space"}, {"bogus": 1}, {"sampling": {"base_fps": 9}},
    {"sampling": {"nope": 1}}, {"hw_accel": "cuda"}, {"queue_size": 0}, {"queue_policy": "block"},
    {"preprocess": {"privacy_masks": [{"rect": [0.5, 0.5, 0.2, 0.9]}]}}, {"preprocess": {"target_size": 8}},
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ConfigError):
        parse_camera({"camera_id": "c1", "url": "rtsp://h/x", **bad}, {}, 0)


def test_env_expansion_and_missing_var(monkeypatch):
    monkeypatch.setenv("CAM_PW", "s3cret")
    assert cam(url="rtsp://u:${CAM_PW}@h/x").url == "rtsp://u:s3cret@h/x"
    monkeypatch.delenv("CAM_PW")
    with pytest.raises(ConfigError):
        cam(url="rtsp://u:${CAM_PW}@h/x")


def test_redact():
    assert "s3cret" not in redact_url("rtsp://u:s3cret@h/x")


def test_defaults_and_override_merge(tmp_path):
    f = tmp_path / "config" / "i.yaml"
    f.parent.mkdir()
    f.write_text("defaults: {sampling: {base_fps: 1.0}}\ncameras:\n- {camera_id: a, url: 'rtsp://h/1'}\n"
                 "- {camera_id: b, url: 'rtsp://h/2', sampling: {base_fps: 3.0}}\n")
    a, b = load_config(f).cameras
    assert (a.sampling.base_fps, b.sampling.base_fps, b.sampling.burst_fps) == (1.0, 3.0, 5.0)


def test_duplicate_ids_and_bad_toplevel(tmp_path):
    f = tmp_path / "i.yaml"
    f.write_text("cameras:\n- {camera_id: a, url: 'rtsp://h/1'}\n- {camera_id: a, url: 'rtsp://h/2'}\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(f)
    f.write_text("camreas: []\n")
    with pytest.raises(ConfigError):
        load_config(f)


def test_shipped_config_loads():
    from tests.support.streams import ROOT
    assert load_config(ROOT / "config" / "ingest.yaml").plugin_paths
