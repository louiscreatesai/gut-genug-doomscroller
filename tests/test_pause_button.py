import cv2
import numpy as np

from pause_button import BUTTON_H, BUTTON_W, MARGIN, PauseControl, button_rect

W, H = 1280, 720


def blank():
    return np.zeros((H, W, 3), dtype=np.uint8)


def click(control: PauseControl, x: int, y: int) -> None:
    control.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)


def test_button_sits_in_bottom_right_corner():
    rect = button_rect(W, H)

    assert (rect.x + rect.w, rect.y + rect.h) == (W - MARGIN, H - MARGIN)
    assert (rect.w, rect.h) == (BUTTON_W, BUTTON_H)


def test_click_inside_button_toggles_pause():
    control = PauseControl()
    control.draw(blank())
    rect = button_rect(W, H)

    click(control, rect.x + 5, rect.y + 5)

    assert control.paused


def test_second_click_resumes():
    control = PauseControl()
    control.draw(blank())
    rect = button_rect(W, H)

    click(control, rect.x + 5, rect.y + 5)
    click(control, rect.x + 5, rect.y + 5)

    assert not control.paused


def test_click_outside_button_does_nothing():
    control = PauseControl()
    control.draw(blank())

    click(control, 10, 10)

    assert not control.paused


def test_click_before_button_is_drawn_does_nothing():
    control = PauseControl()

    click(control, W - MARGIN - 5, H - MARGIN - 5)

    assert not control.paused


def test_mouse_move_and_release_are_ignored():
    control = PauseControl()
    control.draw(blank())
    rect = button_rect(W, H)

    control.on_mouse(cv2.EVENT_MOUSEMOVE, rect.x + 5, rect.y + 5)
    control.on_mouse(cv2.EVENT_LBUTTONUP, rect.x + 5, rect.y + 5)

    assert not control.paused


def test_consume_change_reports_each_toggle_once():
    control = PauseControl()

    control.toggle()

    assert control.consume_change() is True
    assert control.consume_change() is False


def test_draw_paints_different_button_when_paused():
    control = PauseControl()
    running, paused = blank(), blank()

    control.draw(running)
    control.toggle()
    control.draw(paused)

    rect = button_rect(W, H)
    region = (slice(rect.y, rect.y + rect.h), slice(rect.x, rect.x + rect.w))
    assert running[region].any() and paused[region].any()
    assert not np.array_equal(running[region], paused[region])
