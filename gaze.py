"""Looking-down (doomscrolling) detection, personalised to how *you* look at
your phone.

People look down at a phone very differently: some drop their whole head
(15–25° of pitch), others keep their head level and only lower their eyes.
Fixed thresholds can't serve both, so a two-step calibration records you
looking at the screen and then at your phone, and learns how four signals
move for you:

    ear          eye openness (lids lower when you look down)
    iris_lid     iris position between the eyelids
    iris_corner  iris position below the eye-corner line (corners don't move
                 with gaze, so this tracks the eyeball itself)
    pitch_deg    head pitch (chin tucked = larger)

Each signal is rescaled so 0 = your screen pose and 1 = your phone pose, then
they're averaged, weighted by how cleanly each one separated the two poses
during calibration. Calibration is saved, so it only happens once (press `c`
to redo it).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence, Tuple

# FaceMesh landmark indices
L_LID_TOP, L_LID_BOTTOM, L_CORNER_OUT, L_CORNER_IN, L_IRIS = 159, 145, 33, 133, 468
R_LID_TOP, R_LID_BOTTOM, R_CORNER_IN, R_CORNER_OUT, R_IRIS = 386, 374, 362, 263, 473
FOREHEAD, CHIN = 10, 152

FEATURES = ("ear", "iris_lid", "iris_corner", "pitch_deg")
# Typical frame-to-frame noise per feature: a floor for the measured spread, so
# a feature that happened to be very still during calibration can't claim a
# huge (meaningless) separation.
NOISE_FLOOR = (0.01, 0.01, 0.005, 1.0)
MAX_SEPARATION = 6.0
MIN_USEFUL_SEPARATION = 0.5


class State(Enum):
    CALIBRATING_SCREEN = auto()
    CALIBRATING_PHONE = auto()
    IDLE = auto()
    SUSPECT = auto()
    DOOMSCROLLING = auto()


CALIBRATING = (State.CALIBRATING_SCREEN, State.CALIBRATING_PHONE)


@dataclass(frozen=True)
class GazeConfig:
    grace_seconds: float = 0.5           # sustained looking-down before trigger
    release_seconds: float = 0.3         # sustained looking-up before release
    face_lost_hold_seconds: float = 2.0  # a head tilted far down often loses tracking
    smoothing: float = 0.5               # EMA weight of the newest score
    enter_score: float = 0.5             # 0 = your screen pose, 1 = your phone pose
    exit_score: float = 0.3              # hysteresis: easier to stay than to enter
    head_down_override_deg: float = 15.0  # this much head tilt always counts as down
    # calibration: settle time after each prompt, then collect for this long
    screen_settle_seconds: float = 0.5
    phone_settle_seconds: float = 1.5   # people take ~1.5s to pick up / look at the phone
    collect_seconds: float = 2.0
    min_calibration_samples: int = 10
    min_separation: float = 1.5          # best signal must separate screen/phone this well


@dataclass(frozen=True)
class GazeSample:
    ear: float
    iris_lid: float
    iris_corner: float
    pitch_deg: float

    def values(self) -> Tuple[float, ...]:
        return tuple(getattr(self, f) for f in FEATURES)


@dataclass(frozen=True)
class Calibration:
    screen: Tuple[float, ...]   # per-feature median while looking at the screen
    phone: Tuple[float, ...]    # ... and while looking at the phone
    weights: Tuple[float, ...]  # sum to 1

    def score(self, sample: GazeSample, head_override_deg: float) -> float:
        """0 ≈ looking at the screen, 1 ≈ looking at your phone."""
        total = 0.0
        for x, a, b, w in zip(sample.values(), self.screen, self.phone, self.weights):
            if w > 0 and b != a:
                total += w * min(max((x - a) / (b - a), -1.0), 2.0)
        head_tilt = sample.pitch_deg - self.screen[FEATURES.index("pitch_deg")]
        return max(total, 1.0) if head_tilt > head_override_deg else total

    def to_json(self) -> dict:
        return {"features": list(FEATURES), "screen": list(self.screen),
                "phone": list(self.phone), "weights": list(self.weights)}

    @staticmethod
    def from_json(data: dict) -> Optional["Calibration"]:
        """Validates untrusted file content; returns None if unusable."""
        try:
            if data.get("features") != list(FEATURES):
                return None
            parts = [tuple(float(v) for v in data[k]) for k in ("screen", "phone", "weights")]
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if any(len(p) != len(FEATURES) or not all(map(math.isfinite, p)) for p in parts):
            return None
        if not math.isclose(sum(parts[2]), 1.0, abs_tol=1e-6) or min(parts[2]) < 0:
            return None
        return Calibration(*parts)


def _spread(values: Sequence[float], floor: float) -> float:
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values) * 1.4826
    return max(mad, floor)


def fit_calibration(screen: Sequence[GazeSample], phone: Sequence[GazeSample],
                    min_separation: float) -> Optional[Calibration]:
    """Learns per-feature screen/phone poses and weights. Returns None if the
    two poses weren't distinguishable (e.g. you didn't actually look down)."""
    screen_meds, phone_meds, separations = [], [], []
    for i, floor in enumerate(NOISE_FLOOR):
        s = [x.values()[i] for x in screen]
        p = [x.values()[i] for x in phone]
        a, b = statistics.median(s), statistics.median(p)
        spread = math.sqrt((_spread(s, floor) ** 2 + _spread(p, floor) ** 2) / 2)
        screen_meds.append(a)
        phone_meds.append(b)
        separations.append(min(abs(b - a) / spread, MAX_SEPARATION))
    if max(separations) < min_separation:
        return None
    raw = [max(d - MIN_USEFUL_SEPARATION, 0.0) for d in separations]
    weights = tuple(w / sum(raw) for w in raw)
    return Calibration(tuple(screen_meds), tuple(phone_meds), weights)


def load_calibration(path: Path) -> Optional[Calibration]:
    try:
        return Calibration.from_json(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def save_calibration(path: Path, calibration: Calibration) -> None:
    try:
        path.write_text(json.dumps(calibration.to_json(), indent=2))
    except OSError as e:
        print(f"Could not save calibration to {path}: {e}")


def _eye(lm, top, bottom, corner_a, corner_b, iris, width, height):
    eye_width = abs(lm[corner_b].x - lm[corner_a].x) * width or 1e-6
    lid_gap = (lm[bottom].y - lm[top].y) * height
    ear = abs(lid_gap) / eye_width
    iris_lid = (lm[iris].y - lm[top].y) * height / (lid_gap or 1e-6)
    corner_mid_y = (lm[corner_a].y + lm[corner_b].y) / 2
    iris_corner = (lm[iris].y - corner_mid_y) * height / eye_width
    return ear, iris_lid, iris_corner


def extract_sample(lm, width: int, height: int) -> GazeSample:
    """Gaze signals from FaceMesh landmarks. MediaPipe's z uses the same
    scale as x, so y is converted to that scale for the pitch angle."""
    left = _eye(lm, L_LID_TOP, L_LID_BOTTOM, L_CORNER_OUT, L_CORNER_IN, L_IRIS, width, height)
    right = _eye(lm, R_LID_TOP, R_LID_BOTTOM, R_CORNER_IN, R_CORNER_OUT, R_IRIS, width, height)
    fh, ch = lm[FOREHEAD], lm[CHIN]
    pitch = math.degrees(math.atan2((ch.z - fh.z) * width, (ch.y - fh.y) * height))
    ear, iris_lid, iris_corner = ((l + r) / 2 for l, r in zip(left, right))
    return GazeSample(ear=ear, iris_lid=iris_lid, iris_corner=iris_corner, pitch_deg=pitch)


@dataclass(frozen=True)
class Machine:
    """Immutable detector snapshot; `step` returns a new one. No camera or
    MediaPipe dependency, so it's unit-testable."""

    cfg: GazeConfig = field(default_factory=GazeConfig)
    state: State = State.CALIBRATING_SCREEN
    calibration: Optional[Calibration] = None
    score: Optional[float] = None         # smoothed down-score
    since: Optional[float] = None         # when the pending transition began
    face_lost_since: Optional[float] = None
    phase_started: Optional[float] = None
    samples: tuple = ()
    screen_samples: tuple = ()
    message: str = ""                     # shown in the preview (e.g. calibration hints)

    @staticmethod
    def start(cfg: GazeConfig = GazeConfig(), calibration: Optional[Calibration] = None) -> "Machine":
        if calibration is None:
            return Machine(cfg=cfg)
        return Machine(cfg=cfg, state=State.IDLE, calibration=calibration)

    def recalibrate(self) -> "Machine":
        return Machine(cfg=self.cfg)

    def reset(self) -> "Machine":
        """Fresh detection state, keeping the calibration (used on resume)."""
        return Machine.start(self.cfg, self.calibration)

    def step(self, sample: Optional[GazeSample], now: float) -> "Machine":
        if self.state in CALIBRATING:
            return self._calibrate(sample, now)
        if sample is None:
            return self._face_lost(now)
        raw = self.calibration.score(sample, self.cfg.head_down_override_deg)
        a = self.cfg.smoothing
        score = raw if self.score is None else a * raw + (1 - a) * self.score
        m = replace(self, score=score, face_lost_since=None)
        if m.state is State.DOOMSCROLLING:
            return m._step_doomscrolling(now)
        return m._step_watching(now)

    def _step_watching(self, now: float) -> "Machine":
        if self.score <= self.cfg.enter_score:
            return replace(self, state=State.IDLE, since=None)
        if self.state is State.IDLE:
            return replace(self, state=State.SUSPECT, since=now)
        if now - self.since >= self.cfg.grace_seconds:
            return replace(self, state=State.DOOMSCROLLING, since=None)
        return self

    def _step_doomscrolling(self, now: float) -> "Machine":
        if self.score >= self.cfg.exit_score:
            return replace(self, since=None)
        if self.since is None:
            return replace(self, since=now)
        if now - self.since >= self.cfg.release_seconds:
            return replace(self, state=State.IDLE, since=None)
        return self

    def _calibrate(self, sample: Optional[GazeSample], now: float) -> "Machine":
        cfg = self.cfg
        started = self.phase_started if self.phase_started is not None else now
        settle = cfg.screen_settle_seconds if self.state is State.CALIBRATING_SCREEN else cfg.phone_settle_seconds
        elapsed = now - started
        samples = self.samples
        if sample is not None and elapsed >= settle:
            samples = (*samples, sample)
        if elapsed < settle + cfg.collect_seconds:
            return replace(self, phase_started=started, samples=samples)
        if len(samples) < cfg.min_calibration_samples:
            return replace(self, phase_started=now, samples=(),
                           message="couldn't see your face - stay in view of the camera")
        if self.state is State.CALIBRATING_SCREEN:
            return replace(self, state=State.CALIBRATING_PHONE, phase_started=now,
                           samples=(), screen_samples=samples, message="")
        calibration = fit_calibration(self.screen_samples, samples, cfg.min_separation)
        if calibration is None:
            return replace(Machine(cfg=cfg),
                           message="couldn't tell screen from phone - look further down at your phone")
        return Machine(cfg=cfg, state=State.IDLE, calibration=calibration)

    def _face_lost(self, now: float) -> "Machine":
        lost_since = self.face_lost_since if self.face_lost_since is not None else now
        if now - lost_since < self.cfg.face_lost_hold_seconds:
            return replace(self, face_lost_since=lost_since)
        return replace(self, state=State.IDLE, since=None, score=None, face_lost_since=lost_since)


class GazeWatcher:
    """Runs MediaPipe FaceMesh on frames and feeds the state machine."""

    def __init__(self, config: Optional[GazeConfig] = None, calibration: Optional[Calibration] = None):
        import mediapipe as mp  # imported lazily: slow, and not needed by tests

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.machine = Machine.start(config or GazeConfig(), calibration)

    @property
    def state(self) -> State:
        return self.machine.state

    def recalibrate(self) -> None:
        self.machine = self.machine.recalibrate()

    def reset(self) -> None:
        self.machine = self.machine.reset()

    def update(self, rgb_frame, now: float) -> dict:
        """Processes one RGB frame; returns debug info for the preview UI."""
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            self.machine = self.machine.step(None, now)
            return {"face": False}
        lm = result.multi_face_landmarks[0].landmark
        h, w = rgb_frame.shape[:2]
        self.machine = self.machine.step(extract_sample(lm, w, h), now)
        return {"face": True, "landmarks": lm}
