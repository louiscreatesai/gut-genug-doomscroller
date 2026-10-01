import numpy as np

import main
from gaze import GazeConfig, Machine, State
from main import KEY_ESC, KEY_RECALIBRATE, Tracker
from pause_button import PauseControl
from tests.test_gaze import CALIBRATION

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeWatcher:
    def __init__(self, state=State.IDLE):
        self.machine = Machine.start(GazeConfig(), CALIBRATION)
        self.state = state
        self.updates = 0
        self.resets = 0
        self.recalibrations = 0

    def update(self, rgb, now):
        self.updates += 1
        return {"face": False}

    def reset(self):
        self.resets += 1
        self.state = State.IDLE

    def recalibrate(self):
        self.recalibrations += 1


class FakeClipWindow:
    def __init__(self):
        self.visible = False

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


def make(state=State.DOOMSCROLLING):
    watcher, window, pause = FakeWatcher(state), FakeClipWindow(), PauseControl()
    return Tracker(watcher, window, pause, CALIBRATION), watcher, window


def test_doomscrolling_shows_the_clip():
    tracker, _, window = make()

    tracker.on_frame(FRAME, 0.0)

    assert window.visible


def test_looking_up_hides_the_clip():
    tracker, watcher, window = make()
    tracker.on_frame(FRAME, 0.0)
    watcher.state = State.IDLE

    tracker.on_frame(FRAME, 0.1)

    assert not window.visible


def test_pausing_hides_clip_and_skips_tracking():
    tracker, watcher, window = make()
    tracker.on_frame(FRAME, 0.0)

    tracker.on_key(ord("p"))
    tracker.on_frame(FRAME, 0.1)
    tracker.on_frame(FRAME, 0.2)

    assert not window.visible
    assert watcher.updates == 1


def test_space_also_toggles_pause():
    tracker, _, _ = make()

    tracker.on_key(ord(" "))

    assert tracker.pause.paused


def test_resume_resets_detector_and_tracks_again():
    tracker, watcher, _ = make()
    tracker.on_key(ord("p"))
    tracker.on_frame(FRAME, 0.0)

    tracker.on_key(ord("p"))
    tracker.on_frame(FRAME, 0.1)

    assert watcher.resets == 1
    assert watcher.updates == 1


def test_recalibrate_while_paused_resumes_tracking():
    tracker, watcher, _ = make()
    tracker.on_key(ord("p"))

    tracker.on_key(KEY_RECALIBRATE)

    assert watcher.recalibrations == 1
    assert not tracker.pause.paused


def test_escape_quits_and_other_keys_continue():
    tracker, _, _ = make()

    assert tracker.on_key(KEY_ESC) is False
    assert tracker.on_key(0xFF) is True


def test_new_calibration_is_saved(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    monkeypatch.setattr(main, "CALIBRATION_PATH", path)
    tracker = Tracker(FakeWatcher(State.IDLE), FakeClipWindow(), PauseControl(), saved_calibration=None)

    tracker.on_frame(FRAME, 0.0)

    assert path.exists()


def test_machine_reset_keeps_calibration_and_returns_to_idle():
    m = Machine(cfg=GazeConfig(), state=State.DOOMSCROLLING, calibration=CALIBRATION, score=1.2)

    fresh = m.reset()

    assert fresh.state is State.IDLE and fresh.calibration == CALIBRATION and fresh.score is None


def test_machine_reset_without_calibration_restarts_calibrating():
    assert Machine().reset().state is State.CALIBRATING_SCREEN
