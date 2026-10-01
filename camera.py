"""Webcam capture on a background thread, with automatic recovery.

Reading on its own thread means the preview always shows the newest frame
(no backlog of stale buffered frames after a slow iteration), and a watchdog
reopens the camera if it stops delivering frames — e.g. after macOS hands it
to another app, or the device hiccups.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import cv2

PERMISSION_WAIT_SECONDS = 60   # first run: time to click Allow on macOS's prompt
STALL_SECONDS = 2.0            # no new frame for this long -> reopen the camera
REOPEN_BACKOFF_SECONDS = 1.0
READ_FAIL_SLEEP_SECONDS = 0.01

Opener = Callable[[int], "cv2.VideoCapture"]


def open_capture(index: int, opener: Opener = cv2.VideoCapture) -> Optional["cv2.VideoCapture"]:
    cap = opener(index)
    if cap.isOpened():
        return cap
    cap.release()
    return None


class CameraStream:
    def __init__(self, index: int = 0, opener: Opener = cv2.VideoCapture,
                 stall_seconds: float = STALL_SECONDS):
        self._index = index
        self._opener = opener
        self._stall_seconds = stall_seconds
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stopping = threading.Event()
        self._cap = None
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.reconnects = 0

    def start(self, permission_wait: float = PERMISSION_WAIT_SECONDS) -> bool:
        """Opens the camera (retrying while the macOS permission prompt is up)
        and starts the reader thread. Returns False if it never opened."""
        deadline = time.monotonic() + permission_wait
        while True:
            self._cap = open_capture(self._index, self._opener)
            if self._cap is not None:
                break
            if time.monotonic() >= deadline:
                return False
            print("Waiting for camera access — click Allow if macOS asks...")
            time.sleep(2)
        self._thread.start()
        return True

    def latest(self) -> Tuple[int, Optional[object]]:
        """(sequence number, newest frame). The sequence number only changes
        when a new frame arrives, so callers can skip re-processing."""
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(2)
        if self._cap is not None:
            self._cap.release()

    def _run(self) -> None:
        last_frame_at = time.monotonic()
        while not self._stopping.is_set():
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            now = time.monotonic()
            if ok and frame is not None:
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                last_frame_at = now
                continue
            if now - last_frame_at >= self._stall_seconds:
                self._reopen()
                last_frame_at = time.monotonic()
            else:
                time.sleep(READ_FAIL_SLEEP_SECONDS)

    def _reopen(self) -> None:
        print("Camera stopped delivering frames — reopening it")
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        while not self._stopping.is_set():
            self._cap = open_capture(self._index, self._opener)
            if self._cap is not None:
                self.reconnects += 1
                return
            time.sleep(REOPEN_BACKOFF_SECONDS)
