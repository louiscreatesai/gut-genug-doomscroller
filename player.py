"""The Gut Genug video window.

Chrome is launched once, in the background, as soon as the tracker starts:
the player page is loaded, the clip pre-buffered, and the window minimized.
Triggering then just restarts the clip from 0:00 (with sound) and restores
the window — instant, instead of cold-starting a player. All Selenium calls
run on one worker thread, so the webcam loop never blocks on the browser.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from chromedriver import ensure_chromedriver

# Window fraction of screen: width x height, anchored to the top-right so
# the webcam preview stays visible. The clip is vertical, so half-width x
# full-height shows it big with little letterboxing.
WIDTH_FRACTION = 1 / 2
HEIGHT_FRACTION = 1

CLOSE_TIMEOUT_SECONDS = 5
PRELOAD_TIMEOUT_SECONDS = 10
PRELOAD_POLL_SECONDS = 0.1
HAVE_ENOUGH_DATA = 4   # HTMLMediaElement.readyState

# Chrome won't buffer media in a minimized window, so the clip is loaded while
# the window is (briefly) visible at startup, then the window is minimized.
READY_JS = "return document.getElementById('clip').readyState >= arguments[0];"
# Restart the anthem from the top, with sound, every time you look down.
PLAY_JS = "const v = document.getElementById('clip'); v.currentTime = 0; v.muted = false; v.play();"
PAUSE_JS = "const v = document.getElementById('clip'); if (v) v.pause();"

Launcher = Callable[[], webdriver.Chrome]


def _chrome_options() -> Options:
    options = Options()
    options.page_load_strategy = "eager"
    for arg in (
        # play with sound without a click (we're not a website, we're an alarm)
        "--autoplay-policy=no-user-gesture-required",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--no-first-run",
        # keep the minimized page fully alive so restoring it is instant
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--disable-background-timer-throttling",
    ):
        options.add_argument(arg)
    # Best-effort: suppresses some of Chrome's "controlled by automated
    # software" indicators. Chrome may still show a small banner regardless.
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return options


def _wait_until_buffered(driver: webdriver.Chrome) -> None:
    deadline = time.monotonic() + PRELOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if driver.execute_script(READY_JS, HAVE_ENOUGH_DATA):
            return
        time.sleep(PRELOAD_POLL_SECONDS)
    print("The clip is still buffering; the first play may take a moment.")


def make_launcher(page_url: str) -> Launcher:
    def launch() -> webdriver.Chrome:
        """Starts Chrome on the player page, placed top-right, buffers the
        clip, then minimizes the window. Raises WebDriverException if Chrome
        can't be started."""
        driver_path = ensure_chromedriver()
        service = Service(executable_path=driver_path) if driver_path else Service()
        driver = webdriver.Chrome(service=service, options=_chrome_options())
        sw = driver.execute_script("return window.screen.availWidth")
        sh = driver.execute_script("return window.screen.availHeight")
        width, height = round(sw * WIDTH_FRACTION), round(sh * HEIGHT_FRACTION)
        driver.set_window_rect(x=sw - width, y=0, width=width, height=height)
        driver.get(page_url)
        _wait_until_buffered(driver)
        driver.minimize_window()
        return driver
    return launch


class ClipWindow:
    """Non-blocking handle on the video window. show()/hide() only record
    the desired state; a worker thread applies it, collapsing rapid toggles
    so the window just ends up in the latest requested state."""

    def __init__(self, launcher: Launcher):
        self._launcher = launcher
        self._cond = threading.Condition()
        self._want_visible = False
        self._is_visible = False
        self._stopping = False
        self._driver: Optional[webdriver.Chrome] = None
        self._thread = threading.Thread(target=self._run, name="clip-window", daemon=True)

    # --- called from the main (webcam) thread; never block -----------------

    def start(self) -> None:
        self._thread.start()

    def show(self) -> None:
        self._request(True)

    def hide(self) -> None:
        self._request(False)

    def close(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify()
        self._thread.join(CLOSE_TIMEOUT_SECONDS)

    def _request(self, visible: bool) -> None:
        with self._cond:
            if self._want_visible != visible:
                self._want_visible = visible
                self._cond.notify()

    # --- worker thread ------------------------------------------------------

    def _run(self) -> None:
        self._ensure_driver()  # pre-warm while you're still looking at the screen
        while True:
            with self._cond:
                while not self._stopping and self._want_visible == self._is_visible:
                    self._cond.wait()
                if self._stopping:
                    break
                target = self._want_visible
            self._apply(target)
        self._quit_driver()

    def _apply(self, visible: bool) -> None:
        try:
            self._show() if visible else self._hide()
        except WebDriverException as e:
            # Chrome crashed or the window was closed by hand: relaunch and
            # retry once, so the next trigger still works.
            print(f"Video window error ({type(e).__name__}); relaunching")
            self._quit_driver()
            if visible:
                try:
                    self._show()
                except WebDriverException as retry_error:
                    print(f"Could not show the video window: {retry_error.msg}")
        # Record the attempt either way, so a broken Chrome can't spin the loop.
        self._is_visible = visible

    def _ensure_driver(self) -> bool:
        if self._driver is not None and self._driver_alive():
            return True
        self._quit_driver()
        try:
            self._driver = self._launcher()
            return True
        except WebDriverException as e:
            print(f"Could not start Chrome via Selenium: {e.msg}")
            return False

    def _driver_alive(self) -> bool:
        try:
            return bool(self._driver.window_handles)
        except WebDriverException:
            return False

    def _show(self) -> None:
        if not self._ensure_driver():
            return
        # Start the clip first (the page stays live while minimized), so it's
        # already playing as the window appears.
        self._driver.execute_script(PLAY_JS)
        self._set_window_state("normal")

    def _hide(self) -> None:
        if self._driver is None:
            return
        self._driver.execute_script(PAUSE_JS)
        self._set_window_state("minimized")

    def _set_window_state(self, state: str) -> None:
        # Via the DevTools protocol rather than Selenium's set_window_rect /
        # minimize_window: those wait for macOS's whole (de)minimize animation,
        # this returns as soon as it starts. Restoring to "normal" keeps the
        # top-right placement from launch.
        window_id = self._driver.execute_cdp_cmd("Browser.getWindowForTarget", {})["windowId"]
        self._driver.execute_cdp_cmd(
            "Browser.setWindowBounds",
            {"windowId": window_id, "bounds": {"windowState": state}},
        )

    def _quit_driver(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.quit()
        except WebDriverException:
            pass  # already gone
        self._driver = None
