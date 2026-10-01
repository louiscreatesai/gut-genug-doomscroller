"""A clickable Pause / Resume button drawn into the OpenCV preview window.

OpenCV has no widgets, so the button is painted onto each frame and clicks
are hit-tested via a mouse callback on the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

BUTTON_W, BUTTON_H, MARGIN = 170, 50, 18
ICON_HALF = 11

# colors are BGR (OpenCV's convention)
PAUSE_FILL = (0, 199, 255)      # golden arches yellow
RESUME_FILL = (60, 170, 40)     # green = go
DARK_INK = (25, 25, 25)
LIGHT_INK = (255, 255, 255)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def button_rect(frame_width: int, frame_height: int) -> Rect:
    """Bottom-right corner, clear of the top banner and bottom-left info text."""
    return Rect(frame_width - BUTTON_W - MARGIN, frame_height - BUTTON_H - MARGIN, BUTTON_W, BUTTON_H)


def _draw_icon(frame, rect: Rect, paused: bool, ink) -> None:
    cx, cy, s = rect.x + 30, rect.y + rect.h // 2, ICON_HALF
    if paused:  # "play" triangle: clicking resumes
        triangle = np.array([(cx - s + 3, cy - s), (cx - s + 3, cy + s), (cx + s, cy)], dtype=np.int32)
        cv2.fillPoly(frame, [triangle], ink, cv2.LINE_AA)
    else:       # "pause" bars: clicking pauses
        cv2.rectangle(frame, (cx - 9, cy - s), (cx - 3, cy + s), ink, -1)
        cv2.rectangle(frame, (cx + 3, cy - s), (cx + 9, cy + s), ink, -1)


def draw_button(frame, paused: bool) -> Rect:
    h, w = frame.shape[:2]
    rect = button_rect(w, h)
    fill, ink = (RESUME_FILL, LIGHT_INK) if paused else (PAUSE_FILL, DARK_INK)
    top_left, bottom_right = (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h)
    cv2.rectangle(frame, top_left, bottom_right, fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, top_left, bottom_right, DARK_INK, 2, cv2.LINE_AA)
    _draw_icon(frame, rect, paused, ink)
    cv2.putText(frame, "RESUME" if paused else "PAUSE", (rect.x + 52, rect.y + rect.h // 2 + 9),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, ink, 2, cv2.LINE_AA)
    return rect


class PauseControl:
    """Holds the paused flag. Toggled by clicking the button (after `attach`)
    or by calling `toggle()` from a key handler."""

    def __init__(self) -> None:
        self.paused = False
        self._rect: Optional[Rect] = None
        self._changed = False

    def attach(self, window_title: str) -> None:
        cv2.setMouseCallback(window_title, self.on_mouse)

    def on_mouse(self, event, x, y, flags=None, param=None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._rect is not None and self._rect.contains(x, y):
            self.toggle()

    def toggle(self) -> None:
        self.paused = not self.paused
        self._changed = True

    def consume_change(self) -> bool:
        """True once after each toggle, so the caller can react to it."""
        changed, self._changed = self._changed, False
        return changed

    def draw(self, frame) -> None:
        self._rect = draw_button(frame, self.paused)
