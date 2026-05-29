from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackPoint:
    frame_idx: int
    x: float
    y: float
    source: str  # "detected" or "predicted"
    confidence: Optional[float] = None


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return cx, cy


# ─────────────────────────────────────────────
# Kalman noise profiles
# ─────────────────────────────────────────────

# Pre-launch: ball is stationary on tee — very low process noise so the
# filter stays locked and doesn't drift.
_PROCESS_NOISE_PRE_LAUNCH = 1e-2
_MEASUREMENT_NOISE_PRE_LAUNCH = 1e-1

# Post-launch: ball is moving 150-400 px/frame. High process noise lets
# the Kalman velocity state update aggressively frame-by-frame instead of
# lagging far behind the real trajectory.
_PROCESS_NOISE_POST_LAUNCH = 5e2
_MEASUREMENT_NOISE_POST_LAUNCH = 1e1

# Post-launch distance gate: ball can jump 400px in one frame, so we open
# the gate wide. Pre-launch we keep it tight so noise doesn't hijack it.
_DISTANCE_THRESHOLD_PRE_LAUNCH = 80.0
_DISTANCE_THRESHOLD_POST_LAUNCH = 600.0


class BallTracker:
    """
    Single-object Kalman tracker for a golf ball.

    State  : x, y, vx, vy
    Measurement: x, y

    Two operating modes selected by set_launched():
      - pre-launch : tight noise, nearest-to-prediction selection
      - post-launch: loose noise, highest-confidence selection
        (the ball moves 150-400 px/frame so "nearest" is meaningless;
         highest confidence correctly picks the real ball over debris)
    """

    def __init__(
        self,
        max_missed: int = 10,
        distance_threshold: float = 80.0,   # kept for API compat; overridden internally
    ) -> None:
        self.max_missed = max_missed
        self._launched = False

        self.kalman = cv2.KalmanFilter(4, 2)

        # Constant-velocity transition: x' = x + vx, y' = y + vy
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        self._apply_noise_profile(launched=False)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0
        self.history: List[TrackPoint] = []

    # ── noise / mode ────────────────────────────────────────────────────────

    def _apply_noise_profile(self, launched: bool) -> None:
        if launched:
            pn = _PROCESS_NOISE_POST_LAUNCH
            mn = _MEASUREMENT_NOISE_POST_LAUNCH
        else:
            pn = _PROCESS_NOISE_PRE_LAUNCH
            mn = _MEASUREMENT_NOISE_PRE_LAUNCH

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * pn
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * mn

    def set_launched(self) -> None:
        """Call this once when launch is detected to switch Kalman profile."""
        if self._launched:
            return
        self._launched = True
        self._apply_noise_profile(launched=True)

    @property
    def _distance_threshold(self) -> float:
        return (
            _DISTANCE_THRESHOLD_POST_LAUNCH
            if self._launched
            else _DISTANCE_THRESHOLD_PRE_LAUNCH
        )

    # ── public interface ─────────────────────────────────────────────────────

    def is_initialized(self) -> bool:
        return self.initialized

    def initialize(self, x: float, y: float) -> None:
        self.kalman.statePost = np.array(
            [[x], [y], [0.0], [0.0]],
            dtype=np.float32,
        )
        self.initialized = True
        self.missed_frames = 0

    def predict(self) -> Tuple[float, float]:
        predicted = self.kalman.predict()
        return float(predicted[0, 0]), float(predicted[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        self.missed_frames = 0
        return float(corrected[0, 0]), float(corrected[1, 0])

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def is_lost(self) -> bool:
        return self.missed_frames > self.max_missed

    def reset(self) -> None:
        self.initialized = False
        self.missed_frames = 0

    def get_position(self) -> Optional[Tuple[float, float]]:
        if not self.initialized:
            return None
        state = self.kalman.statePost
        return float(state[0, 0]), float(state[1, 0])

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def select_detection(
        self,
        detections: List[Detection],
        predicted_position: Optional[Tuple[float, float]],
    ) -> Optional[Detection]:
        """
        Pre-launch : pick the detection nearest to the predicted position
                     (ball is stationary; proximity == correctness).
        Post-launch: pick the detection with the highest confidence
                     (ball moves 150-400 px/frame so proximity is useless;
                     the real ball is the most confidently detected blob
                     anywhere in the frame).
        In both modes a distance gate still applies to avoid wild outliers.
        """
        if not detections:
            return None

        if not self.initialized or predicted_position is None:
            return max(detections, key=lambda d: d.confidence)

        if self._launched:
            # Highest confidence wins — but still reject anything impossibly far.
            candidates = [
                d for d in detections
                if self._distance(d.center, predicted_position) <= self._distance_threshold
            ]
            if not candidates:
                # Relax gate completely if nothing passes — better a weak
                # detection than a miss.
                candidates = detections
            return max(candidates, key=lambda d: d.confidence)

        # Pre-launch: nearest within threshold.
        best: Optional[Detection] = None
        best_dist = float("inf")
        for d in detections:
            dist = self._distance(d.center, predicted_position)
            if dist < best_dist:
                best_dist = dist
                best = d

        if best is None or best_dist > self._distance_threshold:
            return None
        return best

    def step(
        self,
        frame_idx: int,
        detections: List[Detection],
        predicted_position: Optional[Tuple[float, float]] = None,
    ) -> Optional[TrackPoint]:
        if self.initialized and predicted_position is None:
            predicted_position = self.predict()

        selected = self.select_detection(detections, predicted_position)

        if not self.initialized:
            if selected is None:
                return None
            x, y = selected.center
            self.initialize(x, y)
            point = TrackPoint(
                frame_idx=frame_idx,
                x=x,
                y=y,
                source="detected",
                confidence=selected.confidence,
            )
            self.history.append(point)
            return point

        if selected is not None:
            x, y = selected.center
            cx, cy = self.update(x, y)
            point = TrackPoint(
                frame_idx=frame_idx,
                x=cx,
                y=cy,
                source="detected",
                confidence=selected.confidence,
            )
            self.history.append(point)
            return point

        self.mark_missed()

        if self.is_lost():
            self.reset()
            return None

        fallback = self.get_position()
        if fallback is None:
            return None

        point = TrackPoint(
            frame_idx=frame_idx,
            x=fallback[0],
            y=fallback[1],
            source="predicted",
            confidence=None,
        )
        self.history.append(point)
        return point

    def get_history(self) -> List[TrackPoint]:
        return self.history


def yolo_result_to_detections(result) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(conf),
            )
        )
    return detections