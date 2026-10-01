import time

from camera import CameraStream, open_capture


class FakeCapture:
    """Delivers `good_frames` frames, then read() fails forever."""

    def __init__(self, good_frames: int = 10**9, opened: bool = True, frame_delay: float = 0.002):
        self.remaining = good_frames
        self.opened = opened
        self.released = False
        self.frame_delay = frame_delay

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(self.frame_delay)
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, object()

    def release(self):
        self.released = True


class FakeOpener:
    def __init__(self, captures):
        self._captures = list(captures)
        self.opens = 0

    def __call__(self, index):
        self.opens += 1
        return self._captures.pop(0) if self._captures else FakeCapture(opened=False)


def wait_until(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_open_capture_returns_none_and_releases_when_not_opened():
    cap = FakeCapture(opened=False)

    assert open_capture(0, lambda i: cap) is None
    assert cap.released


def test_start_fails_when_camera_never_opens():
    stream = CameraStream(opener=FakeOpener([]))

    assert stream.start(permission_wait=0) is False


def test_latest_returns_new_frames_with_increasing_sequence():
    stream = CameraStream(opener=FakeOpener([FakeCapture()]))
    assert stream.start(permission_wait=0)
    try:
        assert wait_until(lambda: stream.latest()[0] >= 1)
        first_seq, frame = stream.latest()
        assert frame is not None
        assert wait_until(lambda: stream.latest()[0] > first_seq)
    finally:
        stream.stop()


def test_stalled_camera_is_reopened_and_frames_resume():
    dying = FakeCapture(good_frames=3)
    healthy = FakeCapture()
    stream = CameraStream(opener=FakeOpener([dying, healthy]), stall_seconds=0.1)
    assert stream.start(permission_wait=0)
    try:
        assert wait_until(lambda: stream.reconnects == 1)
        assert dying.released
        assert wait_until(lambda: stream.latest()[0] > 10)
    finally:
        stream.stop()


def test_stop_releases_camera():
    cap = FakeCapture()
    stream = CameraStream(opener=FakeOpener([cap]))
    stream.start(permission_wait=0)

    stream.stop()

    assert cap.released
