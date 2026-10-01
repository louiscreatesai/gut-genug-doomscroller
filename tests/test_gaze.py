import json
import math
import statistics
from types import SimpleNamespace

import pytest

from gaze import (
    CALIBRATING, CHIN, FOREHEAD, Calibration, GazeConfig, GazeSample, Machine, State,
    extract_sample, fit_calibration, load_calibration, save_calibration,
    L_CORNER_IN, L_CORNER_OUT, L_IRIS, L_LID_BOTTOM, L_LID_TOP,
    R_CORNER_IN, R_CORNER_OUT, R_IRIS, R_LID_BOTTOM, R_LID_TOP,
)

# Medians measured on a real "eyes-only" doomscroller (head barely moves).
SCREEN = GazeSample(ear=0.36, iris_lid=0.45, iris_corner=-0.075, pitch_deg=3.0)
PHONE = GazeSample(ear=0.27, iris_lid=0.38, iris_corner=-0.045, pitch_deg=5.0)
FPS = 30
DT = 1 / FPS


def jittered(base: GazeSample, n: int = 60):
    """Deterministic small noise so spreads are realistic."""
    out = []
    for i in range(n):
        k = math.sin(i * 1.7)
        out.append(GazeSample(ear=base.ear + 0.012 * k, iris_lid=base.iris_lid + 0.012 * k,
                              iris_corner=base.iris_corner + 0.006 * k,
                              pitch_deg=base.pitch_deg + 0.8 * k))
    return out


CALIBRATION = fit_calibration(jittered(SCREEN), jittered(PHONE), min_separation=1.5)


def calibrated(cfg: GazeConfig = GazeConfig()) -> Machine:
    return Machine.start(cfg, CALIBRATION)


def feed(machine: Machine, sample, seconds: float, start: float = 0.0):
    """Feeds `sample` (or a list cycled) at 30fps; returns (machine, end time)."""
    t = start
    for i in range(round(seconds * FPS)):
        s = sample[i % len(sample)] if isinstance(sample, list) else sample
        machine = machine.step(s, t)
        t += DT
    return machine, t


def mix(a: GazeSample, b: GazeSample, frac: float) -> GazeSample:
    return GazeSample(*(x + (y - x) * frac for x, y in zip(a.values(), b.values())))


# --- fitting --------------------------------------------------------------

JITTER_TOLERANCE = (0.012, 0.012, 0.006, 0.8)


def close_to(actual, expected) -> bool:
    return all(abs(a - e) <= tol for a, e, tol in zip(actual, expected, JITTER_TOLERANCE))


def medians(samples):
    return tuple(statistics.median(s.values()[i] for s in samples) for i in range(4))


def test_fit_learns_screen_and_phone_poses():
    assert CALIBRATION.screen == pytest.approx(medians(jittered(SCREEN)))
    assert CALIBRATION.phone == pytest.approx(medians(jittered(PHONE)))
    assert sum(CALIBRATION.weights) == pytest.approx(1.0)


def test_fit_rejects_indistinguishable_poses():
    assert fit_calibration(jittered(SCREEN), jittered(SCREEN), min_separation=1.5) is None


def test_fit_gives_no_weight_to_a_signal_that_did_not_move():
    phone_same_pitch = GazeSample(ear=0.27, iris_lid=0.38, iris_corner=-0.045, pitch_deg=3.0)

    cal = fit_calibration(jittered(SCREEN), jittered(phone_same_pitch), min_separation=1.5)

    assert cal.weights[3] == 0.0


def test_noise_floor_stops_a_perfectly_still_signal_from_dominating():
    still_screen = [GazeSample(0.36, 0.45, -0.075, 3.0)] * 30
    still_phone = [GazeSample(0.27, 0.38, -0.045, 3.5)] * 30   # pitch moved 0.5° only

    cal = fit_calibration(still_screen, still_phone, min_separation=1.5)

    assert cal.weights[3] == 0.0


# --- scoring --------------------------------------------------------------

def test_score_is_about_zero_on_screen_and_one_on_phone():
    assert CALIBRATION.score(SCREEN, 15) == pytest.approx(0.0, abs=0.05)
    assert CALIBRATION.score(PHONE, 15) == pytest.approx(1.0, abs=0.05)


def test_score_is_clamped_for_garbage_values():
    garbage = GazeSample(ear=0.0, iris_lid=-5.0, iris_corner=3.0, pitch_deg=3.0)

    assert CALIBRATION.score(garbage, 15) <= 2.0


def test_big_head_tilt_always_counts_as_looking_down():
    head_down = GazeSample(ear=SCREEN.ear, iris_lid=SCREEN.iris_lid,
                           iris_corner=SCREEN.iris_corner, pitch_deg=SCREEN.pitch_deg + 20)
    cal = fit_calibration(jittered(SCREEN), jittered(PHONE), 1.5)

    assert cal.score(head_down, head_override_deg=15) >= 1.0


# --- persistence ----------------------------------------------------------

def test_calibration_round_trips_through_disk(tmp_path):
    path = tmp_path / "calibration.json"

    save_calibration(path, CALIBRATION)

    assert load_calibration(path) == CALIBRATION


def test_load_missing_or_corrupt_file_returns_none(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not json")

    assert load_calibration(tmp_path / "missing.json") is None
    assert load_calibration(corrupt) is None


@pytest.mark.parametrize("patch", [
    {"features": ["ear"]},
    {"screen": [1, 2]},
    {"weights": [0.5, 0.5, 0.5, 0.5]},
    {"weights": [-1, 1, 0.5, 0.5]},
    {"phone": ["x", 1, 2, 3]},
    {"screen": [float("nan"), 1, 2, 3]},
])
def test_invalid_calibration_data_is_rejected(patch):
    data = {**CALIBRATION.to_json(), **patch}

    assert Calibration.from_json(json.loads(json.dumps(data))) is None


def test_non_dict_calibration_data_is_rejected():
    assert Calibration.from_json(["nope"]) is None


# --- guided calibration flow ---------------------------------------------

def test_two_step_calibration_ends_idle_with_calibration():
    m, t = feed(Machine.start(), jittered(SCREEN), 2.6)
    assert m.state is State.CALIBRATING_PHONE

    m, _ = feed(m, jittered(PHONE), 3.6, start=t)

    assert m.state not in CALIBRATING
    assert close_to(m.calibration.screen, SCREEN.values())


def test_phone_settle_time_ignores_the_reaction_delay():
    m, t = feed(Machine.start(), jittered(SCREEN), 2.6)
    m, t = feed(m, jittered(SCREEN), 1.4, start=t)   # still looking up while reaching for phone
    m, _ = feed(m, jittered(PHONE), 2.2, start=t)

    assert m.state not in CALIBRATING
    assert close_to(m.calibration.phone, PHONE.values())


def test_calibration_restarts_with_hint_when_phone_pose_not_different():
    m, t = feed(Machine.start(), jittered(SCREEN), 2.6)
    m, _ = feed(m, jittered(SCREEN), 3.6, start=t)

    assert m.state is State.CALIBRATING_SCREEN
    assert "look further down" in m.message


def test_calibration_phase_repeats_when_face_not_seen():
    m, _ = feed(Machine.start(), None, 2.6)

    assert m.state is State.CALIBRATING_SCREEN
    assert "camera" in m.message


def test_start_with_saved_calibration_skips_calibrating():
    assert Machine.start(GazeConfig(), CALIBRATION).state is State.IDLE


def test_recalibrate_returns_fresh_machine_and_keeps_original():
    m, _ = feed(calibrated(), PHONE, 1.0)

    fresh = m.recalibrate()

    assert fresh.state is State.CALIBRATING_SCREEN and fresh.calibration is None
    assert m.state is State.DOOMSCROLLING


# --- detection ------------------------------------------------------------

def test_looking_at_screen_stays_idle():
    m, _ = feed(calibrated(), jittered(SCREEN), 5)

    assert m.state is State.IDLE


def test_sustained_phone_look_triggers_doomscrolling():
    m, _ = feed(calibrated(), jittered(PHONE), 0.7)

    assert m.state is State.DOOMSCROLLING


def test_quick_glance_down_does_not_trigger():
    m, t = feed(calibrated(), PHONE, 0.3)
    m, _ = feed(m, SCREEN, 1.0, start=t)

    assert m.state is State.IDLE


def test_single_frame_spike_does_not_trigger():
    m, t = feed(calibrated(), SCREEN, 0.5)

    m = m.step(mix(SCREEN, PHONE, 1.5), t)
    m, _ = feed(m, SCREEN, 1.0, start=t + DT)

    assert m.state is State.IDLE


def test_looking_back_up_releases():
    m, t = feed(calibrated(), PHONE, 1.0)
    m, _ = feed(m, SCREEN, 0.6, start=t)

    assert m.state is State.IDLE


def test_momentary_upward_flicker_does_not_release():
    m, t = feed(calibrated(), PHONE, 1.0)
    m, t = feed(m, SCREEN, 0.1, start=t)
    m, _ = feed(m, PHONE, 0.5, start=t)

    assert m.state is State.DOOMSCROLLING


def test_hysteresis_keeps_doomscrolling_between_thresholds():
    in_between = mix(SCREEN, PHONE, 0.4)   # below enter (0.5), above exit (0.3)
    m, t = feed(calibrated(), PHONE, 1.0)
    m, _ = feed(m, in_between, 2.0, start=t)

    assert m.state is State.DOOMSCROLLING


def test_in_between_does_not_start_doomscrolling():
    m, _ = feed(calibrated(), mix(SCREEN, PHONE, 0.4), 2.0)

    assert m.state is State.IDLE


def test_face_lost_briefly_keeps_doomscrolling():
    m, t = feed(calibrated(), PHONE, 1.0)
    m, _ = feed(m, None, 1.0, start=t)

    assert m.state is State.DOOMSCROLLING


def test_face_lost_for_long_resets_to_idle():
    m, t = feed(calibrated(), PHONE, 1.0)
    m, _ = feed(m, None, 2.5, start=t)

    assert m.state is State.IDLE


def test_face_reappearing_resets_lost_timer():
    m, t = feed(calibrated(), PHONE, 1.0)
    m, t = feed(m, None, 1.5, start=t)
    m, t = feed(m, PHONE, 0.1, start=t)
    m, _ = feed(m, None, 1.5, start=t)

    assert m.state is State.DOOMSCROLLING


def test_step_does_not_mutate_previous_snapshot():
    m0 = calibrated()

    m0.step(PHONE, 0.0)

    assert m0.state is State.IDLE and m0.score is None


# --- signal extraction ----------------------------------------------------

def _landmarks(pitch_z: float, lid_gap: float, iris_down: float):
    pts = {i: SimpleNamespace(x=0.5, y=0.5, z=0.0) for i in range(478)}
    pts[FOREHEAD] = SimpleNamespace(x=0.5, y=0.2, z=-pitch_z / 2)
    pts[CHIN] = SimpleNamespace(x=0.5, y=0.8, z=pitch_z / 2)
    for top, bot, a, b, iris in [(L_LID_TOP, L_LID_BOTTOM, L_CORNER_OUT, L_CORNER_IN, L_IRIS),
                                 (R_LID_TOP, R_LID_BOTTOM, R_CORNER_IN, R_CORNER_OUT, R_IRIS)]:
        pts[top] = SimpleNamespace(x=0.5, y=0.45 - lid_gap / 2, z=0.0)
        pts[bot] = SimpleNamespace(x=0.5, y=0.45 + lid_gap / 2, z=0.0)
        pts[a] = SimpleNamespace(x=0.45, y=0.45, z=0.0)
        pts[b] = SimpleNamespace(x=0.55, y=0.45, z=0.0)
        pts[iris] = SimpleNamespace(x=0.5, y=0.45 + iris_down, z=0.0)
    return pts


def test_extract_sample_level_head_centered_iris():
    s = extract_sample(_landmarks(pitch_z=0.0, lid_gap=0.02, iris_down=0.0), 640, 480)

    assert s.pitch_deg == pytest.approx(0.0)
    assert s.ear == pytest.approx(0.02 * 480 / (0.1 * 640))   # 9.6px / 64px
    assert s.iris_lid == pytest.approx(0.5)
    assert s.iris_corner == pytest.approx(0.0)


def test_extract_sample_looking_down_moves_iris_and_pitch():
    s = extract_sample(_landmarks(pitch_z=0.3, lid_gap=0.02, iris_down=0.005), 640, 480)

    assert s.pitch_deg > 20
    assert s.iris_lid > 0.5
    assert s.iris_corner > 0
