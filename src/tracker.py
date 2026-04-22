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


class BallTracker:
    """
    Simple single-object Kalman tracker for golf ball tracking.

    State:
        x, y, vx, vy

    Measurement:
        x, y
    """

    def __init__(
        self,
        max_missed: int = 10,
        distance_threshold: float = 80.0,
    ) -> None:
        self.max_missed = max_missed
        self.distance_threshold = distance_threshold

        self.kalman = cv2.KalmanFilter(4, 2)

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

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0
        self.history: List[TrackPoint] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def initialize(self, x: float, y: float) -> None:
        self.kalman.statePost = np.array(
            [[x], [y], [0], [0]],
            dtype=np.float32,
        )
        self.initialized = True
        self.missed_frames = 0

    def predict(self) -> Tuple[float, float]:
        predicted = self.kalman.predict()
        px = float(predicted[0, 0])
        py = float(predicted[1, 0])
        return px, py

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        cx = float(corrected[0, 0])
        cy = float(corrected[1, 0])
        self.missed_frames = 0
        return cx, cy

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
        x = float(state[0, 0])
        y = float(state[1, 0])
        return x, y

    @staticmethod
    def _distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def select_detection(
        self,
        detections: List[Detection],
        predicted_position: Optional[Tuple[float, float]],
    ) -> Optional[Detection]:
        """
        Rules:
        - before initialization: highest confidence
        - after initialization: nearest to predicted position, if within threshold
        """
        if not detections:
            return None

        if not self.initialized or predicted_position is None:
            return max(detections, key=lambda det: det.confidence)

        best_detection = None
        best_distance = float("inf")

        for detection in detections:
            distance = self._distance(detection.center, predicted_position)
            if distance < best_distance:
                best_distance = distance
                best_detection = detection

        if best_detection is None:
            return None

        if best_distance > self.distance_threshold:
            return None

        return best_detection

    def step(
        self,
        frame_idx: int,
        detections: List[Detection],
        predicted_position: Optional[Tuple[float, float]] = None,
    ) -> Optional[TrackPoint]:
        if self.initialized and predicted_position is None:
            predicted_position = self.predict()

        selected_detection = self.select_detection(detections, predicted_position)

        if not self.initialized:
            if selected_detection is None:
                return None

            x, y = selected_detection.center
            self.initialize(x, y)

            point = TrackPoint(
                frame_idx=frame_idx,
                x=x,
                y=y,
                source="detected",
                confidence=selected_detection.confidence,
            )
            self.history.append(point)
            return point

        if selected_detection is not None:
            x, y = selected_detection.center
            corrected_x, corrected_y = self.update(x, y)

            point = TrackPoint(
                frame_idx=frame_idx,
                x=corrected_x,
                y=corrected_y,
                source="detected",
                confidence=selected_detection.confidence,
            )
            self.history.append(point)
            return point

        self.mark_missed()

        if self.is_lost():
            self.reset()
            return None

        fallback_position = self.get_position()
        if fallback_position is None:
            return None

        point = TrackPoint(
            frame_idx=frame_idx,
            x=fallback_position[0],
            y=fallback_position[1],
            source="predicted",
            confidence=None,
        )
        self.history.append(point)
        return point

    def get_history(self) -> List[TrackPoint]:
        return self.history


def yolo_result_to_detections(result) -> List[Detection]:
    """
    Convert one Ultralytics result object into Detection objects.
    """
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