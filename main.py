import time
from pathlib import Path

import cv2

from camera import CameraStream
from clip import ensure_clip, stage_for_chrome
from gaze import (
    GazeWatcher, State, load_calibration, save_calibration,
    L_LID_TOP, L_LID_BOTTOM, R_LID_TOP, R_LID_BOTTOM,
)
from pause_button import PauseControl
from player import ClipWindow, make_launcher


CALIBRATION_PATH = Path(__file__).parent / "calibration.json"
WINDOW_TITLE = "gut genug doomscroll tracker"
CAMERA_STALLED_SECONDS = 1.0   # show a "reconnecting" notice after this
UI_POLL_MS = 5
KEY_ESC, KEY_RECALIBRATE = 27, ord("c")
KEYS_PAUSE = (ord("p"), ord(" "))

# colors are BGR (OpenCV's convention), not RGB
EYE_COLOR = (140, 50, 20)        # deep navy blue
BANNER_FILL = (12, 12, 12)       # black & white, like the video
BANNER_ACCENT = (245, 245, 245)
INFO_COLOR = (60, 220, 255)

BANNERS = {
    State.CALIBRATING_SCREEN: "calibrating 1/2: look at the screen",
    State.CALIBRATING_PHONE: "calibrating 2/2: now look down at your phone",
    State.SUSPECT: "eyes up...",
    State.DOOMSCROLLING: "handy weg... du bist gut genug",
}
PAUSED_BANNER = "paused - click resume or press p"


def draw_banner(frame, text: str) -> None:
    h, w = frame.shape[:2]
    bar_h = 64
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), BANNER_FILL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (0, bar_h), (w, bar_h + 4), BANNER_ACCENT, -1)

    font = cv2.FONT_HERSHEY_TRIPLEX
    label = f">>  {text.lower()}"
    (tw, th), _ = cv2.getTextSize(label, font, 0.95, 2)
    cv2.putText(
        frame, label, (24, (bar_h + th) // 2),
        font, 0.95, (240, 240, 255), 2, cv2.LINE_AA,
    )


def draw_eye_boxes(frame, lm) -> None:
    h, w = frame.shape[:2]
    for top_i, bot_i in [(L_LID_TOP, L_LID_BOTTOM), (R_LID_TOP, R_LID_BOTTOM)]:
        cx = int((lm[top_i].x + lm[bot_i].x) / 2 * w)
        cy = int((lm[top_i].y + lm[bot_i].y) / 2 * h)
        s, c = 45, 12  # half-size, corner length
        for dx in (-s, s):
            for dy in (-s, s):
                cv2.line(frame, (cx + dx, cy + dy), (cx + dx - c if dx > 0 else cx + dx + c, cy + dy), EYE_COLOR, 5)
                cv2.line(frame, (cx + dx, cy + dy), (cx + dx, cy + dy - c if dy > 0 else cy + dy + c), EYE_COLOR, 2)


def draw_info(frame, text: str) -> None:
    cv2.putText(
        frame, text, (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_COLOR, 2, cv2.LINE_AA,
    )


def describe(watcher: GazeWatcher, debug: dict, paused: bool) -> str:
    if paused:
        return "tracking paused   p / space = resume"
    m = watcher.machine
    if m.message:
        return m.message
    face = "" if debug.get("face") else "no face  "
    if m.score is None:
        return f"{face}state: {m.state.name}"
    return (f"{face}down-score {m.score:.2f} (triggers > {m.cfg.enter_score:.2f})  "
            f"state: {m.state.name}   c = recalibrate  p = pause")


def render(frame, watcher: GazeWatcher, debug: dict, pause: PauseControl, camera_stalled: bool) -> None:
    if debug.get("face"):
        draw_eye_boxes(frame, debug["landmarks"])
    if camera_stalled:
        banner = "camera reconnecting..."
    elif pause.paused:
        banner = PAUSED_BANNER
    else:
        banner = BANNERS.get(watcher.state)
    if banner:
        draw_banner(frame, banner)
    draw_info(frame, describe(watcher, debug, pause.paused))
    pause.draw(frame)
    cv2.imshow(WINDOW_TITLE, frame)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}")


class Tracker:
    """Glues detection to the alarm (the Gut Genug video) and handles
    pausing, calibration saving and key presses."""

    def __init__(self, watcher: GazeWatcher, clip_window: ClipWindow,
                 pause: PauseControl, saved_calibration):
        self.watcher = watcher
        self.clip_window = clip_window
        self.pause = pause
        self._saved_calibration = saved_calibration
        self._last_state = None

    def on_frame(self, frame, now: float) -> dict:
        self._apply_pause_change()
        if self.pause.paused:
            return {}
        debug = self.watcher.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)
        self._log_state()
        self._save_new_calibration()
        if self.watcher.state is State.DOOMSCROLLING:
            self.clip_window.show()
        else:
            self._alarm_off()
        return debug

    def on_key(self, key: int) -> bool:
        """Handles a key press; returns False when the app should quit."""
        if key == KEY_ESC:
            return False
        if key in KEYS_PAUSE:
            self.pause.toggle()
        elif key == KEY_RECALIBRATE:
            self.watcher.recalibrate()
            if self.pause.paused:
                self.pause.toggle()  # calibrating needs tracking on
        return True

    def _apply_pause_change(self) -> None:
        if not self.pause.consume_change():
            return
        if self.pause.paused:
            self._alarm_off()
            log("paused")
        else:
            self.watcher.reset()   # don't fire from state left over before the pause
            self._last_state = None
            log("resumed")

    def _alarm_off(self) -> None:
        self.clip_window.hide()

    def _log_state(self) -> None:
        state = self.watcher.state
        if state is not self._last_state:
            log(f"{self._last_state.name if self._last_state else '-'} -> {state.name}")
            self._last_state = state

    def _save_new_calibration(self) -> None:
        calibration = self.watcher.machine.calibration
        if calibration is not None and calibration is not self._saved_calibration:
            save_calibration(CALIBRATION_PATH, calibration)
            self._saved_calibration = calibration
            print(f"Calibrated and saved to {CALIBRATION_PATH.name}")


def run(camera: CameraStream, tracker: Tracker) -> None:
    last_seq, last_new_frame_at, frame, debug = -1, time.monotonic(), None, {}
    while True:
        seq, raw = camera.latest()
        now = time.monotonic()
        if raw is not None and seq != last_seq:
            last_seq, last_new_frame_at = seq, now
            frame = cv2.flip(raw, 1)
            debug = tracker.on_frame(frame, now)
            render(frame.copy(), tracker.watcher, debug, tracker.pause, camera_stalled=False)
        elif frame is not None and now - last_new_frame_at > CAMERA_STALLED_SECONDS:
            render(frame.copy(), tracker.watcher, debug, tracker.pause, camera_stalled=True)
        if not tracker.on_key(cv2.waitKey(UI_POLL_MS) & 0xFF):
            return


def main() -> None:
    clip = ensure_clip()
    if clip is None:
        return
    page_url = stage_for_chrome(clip).as_uri()

    camera = CameraStream()
    if not camera.start():
        print("Could not open webcam — allow camera access for your terminal in")
        print("System Settings → Privacy & Security → Camera, then run again.")
        return

    clip_window = ClipWindow(make_launcher(page_url))
    clip_window.start()  # pre-warms Chrome + buffers the clip in the background
    saved = load_calibration(CALIBRATION_PATH)
    print("Using saved calibration (press c to redo it)" if saved else "No saved calibration - calibrating now")
    watcher = GazeWatcher(calibration=saved)
    pause = PauseControl()
    cv2.namedWindow(WINDOW_TITLE)
    pause.attach(WINDOW_TITLE)
    tracker = Tracker(watcher, clip_window, pause, saved)

    try:
        run(camera, tracker)
    finally:
        clip_window.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
