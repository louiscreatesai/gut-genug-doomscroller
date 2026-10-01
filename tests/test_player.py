import threading
import time

import pytest
from selenium.common.exceptions import WebDriverException

from player import PAUSE_JS, PLAY_JS, ClipWindow


class FakeDriver:
    def __init__(self, fail_on_restore: bool = False):
        self.calls = []
        self.alive = True
        self.fail_on_restore = fail_on_restore
        self._lock = threading.Lock()

    def _log(self, name):
        with self._lock:
            self.calls.append(name)

    @property
    def window_handles(self):
        if not self.alive:
            raise WebDriverException("session gone")
        return ["main"]

    def execute_script(self, js, *args):
        self._log({PLAY_JS: "play", PAUSE_JS: "pause"}.get(js, "other-js"))

    def execute_cdp_cmd(self, cmd, params):
        if cmd == "Browser.getWindowForTarget":
            return {"windowId": 1}
        if self.fail_on_restore:
            raise WebDriverException("no such window")
        self._log(params["bounds"]["windowState"])
        return {}

    def quit(self):
        self._log("quit")
        self.alive = False

    def names(self):
        with self._lock:
            return list(self.calls)


class FakeLauncher:
    def __init__(self, drivers, delay: float = 0.0):
        self._drivers = list(drivers)
        self.delay = delay
        self.launches = 0

    def __call__(self):
        time.sleep(self.delay)
        self.launches += 1
        driver = self._drivers.pop(0)
        if isinstance(driver, Exception):
            raise driver
        return driver


def wait_until(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def window_factory():
    windows = []

    def make(launcher):
        w = ClipWindow(launcher=launcher)
        windows.append(w)
        w.start()
        return w

    yield make
    for w in windows:
        w.close()


def test_play_js_restarts_clip_from_the_top_with_sound():
    assert "currentTime = 0" in PLAY_JS and "muted = false" in PLAY_JS and "play()" in PLAY_JS


def test_start_prewarms_chrome_before_any_trigger(window_factory):
    launcher = FakeLauncher([FakeDriver()])

    window_factory(launcher)

    assert wait_until(lambda: launcher.launches == 1)


def test_show_starts_clip_then_restores_window(window_factory):
    driver = FakeDriver()
    w = window_factory(FakeLauncher([driver]))

    w.show()

    assert wait_until(lambda: driver.names() == ["play", "normal"])


def test_hide_pauses_clip_and_minimizes(window_factory):
    driver = FakeDriver()
    w = window_factory(FakeLauncher([driver]))
    w.show()
    assert wait_until(lambda: len(driver.names()) == 2)

    w.hide()

    assert wait_until(lambda: driver.names() == ["play", "normal", "pause", "minimized"])


def test_show_never_blocks_the_caller_even_while_chrome_starts(window_factory):
    w = window_factory(FakeLauncher([FakeDriver()], delay=1.0))

    started = time.monotonic()
    w.show()
    w.hide()
    w.show()

    assert time.monotonic() - started < 0.05


def test_rapid_toggles_collapse_to_latest_state(window_factory):
    driver = FakeDriver()
    w = window_factory(FakeLauncher([driver], delay=0.2))

    for _ in range(5):
        w.show()
        w.hide()
    w.show()

    assert wait_until(lambda: driver.names() == ["play", "normal"])
    time.sleep(0.1)
    assert driver.names() == ["play", "normal"]


def test_window_closed_by_hand_is_relaunched_on_next_show(window_factory):
    first, second = FakeDriver(), FakeDriver()
    launcher = FakeLauncher([first, second])
    w = window_factory(launcher)
    assert wait_until(lambda: launcher.launches == 1)
    first.alive = False

    w.show()

    assert wait_until(lambda: second.names() == ["play", "normal"])
    assert launcher.launches == 2


def test_driver_error_while_showing_relaunches_and_retries(window_factory):
    broken, fresh = FakeDriver(fail_on_restore=True), FakeDriver()
    w = window_factory(FakeLauncher([broken, fresh]))

    w.show()

    assert wait_until(lambda: fresh.names() == ["play", "normal"])
    assert "quit" in broken.names()


def test_chrome_failing_to_start_does_not_crash_or_spin(window_factory):
    launcher = FakeLauncher([WebDriverException("no chrome")] * 3)
    w = window_factory(launcher)

    w.show()
    time.sleep(0.2)

    assert launcher.launches == 2  # pre-warm + one attempt on show, no busy loop


def test_close_quits_chrome():
    driver = FakeDriver()
    launcher = FakeLauncher([driver])
    w = ClipWindow(launcher=launcher)
    w.start()
    assert wait_until(lambda: launcher.launches == 1)

    w.close()

    assert "quit" in driver.names()
    assert wait_until(lambda: not w._thread.is_alive())
