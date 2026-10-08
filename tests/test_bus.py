import threading
import time

import numpy as np

from godseye.ingest.manager import FrameBus
from godseye.ingest.types import PreprocessedFrame
from godseye.ingest.preprocess import compute_letterbox


def pf(cam, seq):  # minimal frame object for queue mechanics only
    return PreprocessedFrame(cam, seq, seq, 1, 0.0, None, np.zeros((1, 1, 3), np.float32),
                             compute_letterbox(4, 4, 4), None, "base", None)


def test_drop_oldest_keeps_newest():
    b = FrameBus(); b.register("a", 2, "drop_oldest")
    assert [b.put(pf("a", i)) for i in range(4)] == ["ok", "ok", "dropped_oldest", "dropped_oldest"]
    assert [b.get(0).frame_seq for _ in range(2)] == [2, 3]
    assert b.get(0) is None


def test_drop_newest_keeps_oldest():
    b = FrameBus(); b.register("a", 2, "drop_newest")
    assert [b.put(pf("a", i)) for i in range(4)] == ["ok", "ok", "dropped_newest", "dropped_newest"]
    assert [b.get(0).frame_seq for _ in range(2)] == [0, 1]


def test_round_robin_fairness_flooding_camera_cannot_starve_other():
    b = FrameBus(); b.register("a", 50, "drop_oldest"); b.register("b", 50, "drop_oldest")
    for i in range(30):
        b.put(pf("a", i))
    b.put(pf("b", 0))
    first = [b.get(0).camera_id for _ in range(2)]
    assert set(first) == {"a", "b"}


def test_blocking_get_wakes_on_put_and_times_out():
    b = FrameBus(); b.register("a", 2, "drop_oldest")
    assert b.get(0.05) is None
    threading.Timer(0.1, lambda: b.put(pf("a", 7))).start()
    t = time.time(); f = b.get(2)
    assert f.frame_seq == 7 and time.time() - t < 1


def test_unregistered_camera_dropped():
    assert FrameBus().put(pf("zz", 0)) == "dropped_unregistered"
