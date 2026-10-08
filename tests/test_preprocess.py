import numpy as np
import pytest

from godseye.ingest.config import PreprocessConfig, PrivacyMask
from godseye.ingest.preprocess import Preprocessor, compute_letterbox


def frame(w, h, val=200):
    return np.full((h, w, 3), val, np.uint8)


def test_letterbox_geometry_landscape_portrait_square():
    t = compute_letterbox(1920, 1080, 640)
    assert (t.new_w, t.new_h, t.pad_left, t.pad_top) == (640, 360, 0, 140)
    t = compute_letterbox(1080, 1920, 640)
    assert (t.new_w, t.new_h, t.pad_left, t.pad_top) == (360, 640, 140, 0)
    t = compute_letterbox(640, 640, 640)
    assert (t.scale, t.pad_left, t.pad_top) == (1.0, 0, 0)


def test_output_contract():
    img, t, orig = Preprocessor(PreprocessConfig())(frame(1920, 1080))
    assert img.shape == (640, 640, 3) and img.dtype == np.float32 and img.flags.c_contiguous
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert np.isclose(img[0, 0, 0], 114 / 255)        # pad
    assert np.isclose(img[320, 320, 0], 200 / 255, atol=1e-3)
    assert orig.shape == (1080, 1920, 3)


def test_channel_order_is_rgb():
    f = np.zeros((100, 100, 3), np.uint8)
    f[..., 2] = 255  # BGR red
    img, _, _ = Preprocessor(PreprocessConfig(target_size=64))(f)
    assert img[32, 32].tolist() == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1920), (640, 480), (333, 777), (100, 100)])
def test_box_roundtrip(w, h):
    t = compute_letterbox(w, h, 640)
    boxes = np.array([[10, 20, w / 2, h / 2], [0, 0, w, h]], float)
    back = t.to_original(t.to_model(boxes))
    np.testing.assert_allclose(back, boxes, atol=1e-6)


def test_to_original_clips_padding_region():
    t = compute_letterbox(1920, 1080, 640)
    out = t.to_original(np.array([[-50, -50, 700, 700]], float))
    np.testing.assert_allclose(out, [[0, 0, 1920, 1080]])


def test_object_position_survives_pipeline():
    f = frame(1920, 1080, 0)
    f[300:500, 900:1100] = 255                            # bright box in original coords
    img, t, _ = Preprocessor(PreprocessConfig())(f)
    ys, xs = np.where(img[..., 0] > 0.9)
    box = t.to_original(np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]))[0]
    np.testing.assert_allclose(box, [900, 300, 1100, 500], atol=4)


def test_privacy_black_and_original_policy():
    f = frame(200, 100, 255)
    m = (PrivacyMask((0.0, 0.0, 0.5, 0.5), "black"),)
    img, _, orig = Preprocessor(PreprocessConfig(target_size=200, privacy_masks=m))(f)
    assert img[50:100, 0:100].max() == 0 and img[100:150, :].min() > 0   # 200x100 padded to 200x200: 50px top pad, mask fills rows 50-100
    assert orig.min() == 255 and f.min() == 255          # original untouched by default, input never mutated
    _, _, orig2 = Preprocessor(PreprocessConfig(target_size=200, privacy_masks=m, privacy_apply_to_original=True))(f)
    assert orig2[:50, :100].max() == 0


def test_privacy_blur_changes_pixels():
    f = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    img, _, _ = Preprocessor(PreprocessConfig(target_size=100, privacy_masks=(PrivacyMask((0, 0, 1, 1), "blur"),)))(f)
    assert img.std() < f.std() / 255 * 0.8


def test_keep_original_false_and_copy_independence():
    f = frame(64, 64)
    _, _, o = Preprocessor(PreprocessConfig(target_size=32, keep_original=False))(f)
    assert o is None
    _, _, o = Preprocessor(PreprocessConfig(target_size=32))(f)
    f[:] = 0
    assert o.max() == 200


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8)])
def test_malformed_input_rejected(bad):
    with pytest.raises(ValueError):
        Preprocessor(PreprocessConfig())(bad)
