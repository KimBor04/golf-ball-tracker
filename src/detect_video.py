"""
Golf-ball detection + tracking pipeline.

Key fixes vs. previous version
───────────────────────────────
BUG 1  update_tee_estimate was called unconditionally, so in-flight positions
       corrupted the tee estimate after launch.  Fixed: freeze tee once launched.

BUG 2  recent_detected_centers stored only (x, y); velocity was computed across
       non-consecutive frames when the tracker filled gaps with "predicted" points,
       causing false-launch triggers.  Fixed: store (frame_idx, x, y) and only
       measure velocity over truly consecutive detected frames.

BUG 3  Full-frame detection (conf=0.01) ran on every frame post-launch, flooding
       the pipeline with noise that was then discarded anyway.  Fixed: skip
       full-frame pass post-launch; trust the ROI pass.

IMPROVEMENT 1  Circularity filter — rejects non-round blobs before scoring.
IMPROVEMENT 2  Detection consensus boost — detections seen by both ROI and
               full-frame passes get their confidence boosted.
IMPROVEMENT 3  Aspect-ratio hard cut — anything with ratio > 1.6 is dropped
               before scoring.
IMPROVEMENT 4  Leading ROI center — post-launch the ROI is centered half a
               motion-vector step ahead of the prediction, keeping the ball
               closer to the ROI centre.
IMPROVEMENT 5  Dynamic trajectory-segment limit — scales with estimated speed
               so the visualisation never breaks on fast shots.
IMPROVEMENT 6  Post-launch full-frame pass disabled (see BUG 3 above).
IMPROVEMENT 7  choose_best_detection now returns Detection | None instead of a
               confusingly-typed list[Detection] that always had at most 1 item.
IMPROVEMENT 8  Post-launch scoring is now trajectory-aware:
               - soft distance penalty to predicted position
               - max-distance gate from prediction
               - backward-motion penalty
IMPROVEMENT 9  Trajectory drawing is now frame-aware:
               - trajectory points store frame_idx
               - line segments are not drawn across frame gaps
               - by default only real detected points are connected, not predicted
                 tracker points
"""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path
from typing import NamedTuple

import cv2
from ultralytics import YOLO

from tracker import BallTracker, Detection, yolo_result_to_detections


# ──────────────────────────────────────────────
# Core settings
# ──────────────────────────────────────────────
FULL_FRAME_CONF = 0.05          # raised from 0.01 — less noise
ROI_CONF = 0.001
ROI_CONF_POST_LAUNCH = 0.001
ROI_UPSCALE_POST_LAUNCH = 3.0
ROI_SIZE_POST_LAUNCH = 500      # slightly larger than 400 — ball moves fast
ROI_SIZE = 600
ROI_UPSCALE = 2.0

MIN_BOX_SIZE = 1.0
MAX_BOX_SIZE = 150.0
MAX_ASPECT_RATIO = 1.6          # hard cut — golf balls are round

DEDUP_DISTANCE = 10.0
MAX_TRAJECTORY_SEGMENT_BASE = 140.0   # px; scaled dynamically by speed
CONSENSUS_BOOST = 0.15                # confidence bonus for dual-pass detections

TRACKER_MAX_MISSED = 10
TRACKER_DISTANCE_THRESHOLD = 300.0

# candidate scoring (pre-launch)
CONF_WEIGHT = 4.0
DISTANCE_WEIGHT = 1.0 / 180.0
AREA_WEIGHT = 1.0 / 120.0
ASPECT_PENALTY_WEIGHT = 2.5
MOTION_CONTINUITY_WEIGHT = 1.0 / 160.0

# post-launch scoring
# Important:
# Do NOT use only confidence after launch.
# The real ball can be low-confidence, while false blobs can be high-confidence.
# So after launch we still use confidence, but also prefer candidates that are
# physically plausible relative to the predicted trajectory.
DISTANCE_WEIGHT_POST_LAUNCH = 1.0 / 300.0
MOTION_CONTINUITY_WEIGHT_POST_LAUNCH = 3.0 / 160.0
BACKWARD_MOTION_PENALTY = 2.0
MAX_POST_LAUNCH_CANDIDATE_DISTANCE = 350.0

# circularity filter
CIRCULARITY_THRESHOLD = 0.55          # 1.0 = perfect circle; golf ball ≈ 0.7–0.9
CIRCULARITY_MIN_BLOB_PX = 9           # skip tiny crops (unreliable contour)

# debug drawing
DRAW_ALL_CANDIDATES = True

# trajectory drawing
DRAW_PREDICTED_TRAJECTORY_POINTS = False
MAX_TRAJECTORY_FRAME_GAP = 1

# tee estimation
TEE_BOOTSTRAP_FRAMES = 20
TEE_BOOTSTRAP_MAX_RADIUS = 35.0

# launch detection
TEE_PROXIMITY_THRESHOLD = 60.0
TEE_FILTER_RADIUS_POST_LAUNCH = 80.0
LAUNCH_VELOCITY_THRESHOLD = 8.0
LAUNCH_VELOCITY_WINDOW = 3            # must be *consecutive* detected frames

# ROI leading factor post-launch (fraction of motion vector to look ahead)
ROI_LEAD_FACTOR = 0.5

# How many frames post-launch to keep the larger ROI before shrinking
POST_LAUNCH_LARGE_ROI_FRAMES = 10


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def distance_between_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_roi_bounds(
    image_width: int,
    image_height: int,
    center_x: float,
    center_y: float,
    roi_size: int,
) -> tuple[int, int, int, int]:
    half_size = roi_size // 2
    x1 = max(int(center_x) - half_size, 0)
    y1 = max(int(center_y) - half_size, 0)
    x2 = min(int(center_x) + half_size, image_width)
    y2 = min(int(center_y) + half_size, image_height)
    return x1, y1, x2, y2


def average_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None

    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)

    return x, y


# ──────────────────────────────────────────────
# Detected-center history (frame-aware)
#
# FIX (BUG 2): store frame_idx alongside position so velocity is only ever
# measured over *truly consecutive* detected frames, not across tracker gaps.
# ──────────────────────────────────────────────
class DetectedPoint(NamedTuple):
    frame_idx: int
    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y


class TrajectoryPoint(NamedTuple):
    frame_idx: int
    x: int
    y: int
    source: str


# ──────────────────────────────────────────────
# Launch detection
# ──────────────────────────────────────────────
def detect_launch(
    recent_detected: deque[DetectedPoint],
    tee_position: tuple[float, float] | None,
) -> bool:
    """
    Returns True if the ball appears to have been launched.

    Signal 1 — spatial: latest detection is far from tee.
    Signal 2 — velocity: average speed over the last LAUNCH_VELOCITY_WINDOW
                *consecutive* detected frames exceeds the threshold.

    FIX (BUG 2): velocity is now only computed when the frames in the window
    are consecutive (no gaps), preventing false triggers caused by the tracker
    bridging missed frames with predicted positions.
    """
    if len(recent_detected) < 2:
        return False

    latest = recent_detected[-1]

    # Signal 1: spatial separation from tee
    if tee_position is not None:
        if distance_between_points(latest.xy, tee_position) > TEE_PROXIMITY_THRESHOLD:
            return True

    # Signal 2: sustained high velocity over *consecutive* detected frames
    if len(recent_detected) >= LAUNCH_VELOCITY_WINDOW + 1:
        window = list(recent_detected)[-(LAUNCH_VELOCITY_WINDOW + 1):]

        frames_consecutive = all(
            window[i + 1].frame_idx == window[i].frame_idx + 1
            for i in range(len(window) - 1)
        )

        if frames_consecutive:
            total_dist = sum(
                distance_between_points(window[i].xy, window[i + 1].xy)
                for i in range(len(window) - 1)
            )
            avg_velocity = total_dist / LAUNCH_VELOCITY_WINDOW

            if avg_velocity >= LAUNCH_VELOCITY_THRESHOLD:
                return True

    return False


def extrapolate_position(
    recent_detected: deque[DetectedPoint],
    steps: int = 1,
) -> tuple[float, float] | None:
    """Linear extrapolation from the last two known detected positions."""
    if len(recent_detected) < 2:
        return recent_detected[-1].xy if recent_detected else None

    p1 = recent_detected[-2]
    p2 = recent_detected[-1]

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    return p2.x + dx * steps, p2.y + dy * steps


# ──────────────────────────────────────────────
# Detection conversion / drawing
# ──────────────────────────────────────────────
def convert_roi_result_to_full_frame_detections(
    result,
    offset_x: int,
    offset_y: int,
    scale_factor: float = 1.0,
) -> list[Detection]:
    detections: list[Detection] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = box.tolist()

        x1 /= scale_factor
        y1 /= scale_factor
        x2 /= scale_factor
        y2 /= scale_factor

        detections.append(
            Detection(
                x1=float(x1 + offset_x),
                y1=float(y1 + offset_y),
                x2=float(x2 + offset_x),
                y2=float(y2 + offset_y),
                confidence=float(conf),
            )
        )

    return detections


def draw_detection_boxes(
    image,
    detections: list[Detection],
    color=(0, 255, 0),
    prefix="ball",
    thickness: int = 2,
) -> None:
    for detection in detections:
        x1_i, y1_i = int(detection.x1), int(detection.y1)
        x2_i, y2_i = int(detection.x2), int(detection.y2)

        cx, cy = detection.center

        cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), color, thickness)

        label = f"{prefix} {detection.confidence:.3f}"
        cv2.putText(
            image,
            label,
            (x1_i, max(y1_i - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.circle(image, (int(cx), int(cy)), 3, color, -1)


# ──────────────────────────────────────────────
# Detection generation
# ──────────────────────────────────────────────
def run_full_frame_detection(
    model: YOLO,
    frame_path: Path,
    conf: float,
) -> list[Detection]:
    results = model.predict(
        source=str(frame_path),
        conf=conf,
        iou=0.7,
        save=False,
        verbose=False,
        device=0,
    )

    return yolo_result_to_detections(results[0])


def run_roi_detection_upscaled(
    model: YOLO,
    image,
    roi_center: tuple[float, float],
    roi_size: int,
    conf: float,
    upscale: float,
) -> tuple[list[Detection], tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_bounds(
        image_width=image_width,
        image_height=image_height,
        center_x=roi_center[0],
        center_y=roi_center[1],
        roi_size=roi_size,
    )

    roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi_image.size == 0:
        return [], (roi_x1, roi_y1, roi_x2, roi_y2)

    if upscale != 1.0:
        roi_image = cv2.resize(
            roi_image,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )

    results = model.predict(
        source=roi_image,
        conf=conf,
        iou=0.7,
        save=False,
        verbose=False,
        device=0,
    )

    detections = convert_roi_result_to_full_frame_detections(
        results[0],
        offset_x=roi_x1,
        offset_y=roi_y1,
        scale_factor=upscale,
    )

    return detections, (roi_x1, roi_y1, roi_x2, roi_y2)


# ──────────────────────────────────────────────
# Candidate filtering
# ──────────────────────────────────────────────
def filter_by_box_size(
    detections: list[Detection],
    min_size: float,
    max_size: float,
) -> list[Detection]:
    return [
        d for d in detections
        if min_size <= (d.x2 - d.x1) <= max_size
        and min_size <= (d.y2 - d.y1) <= max_size
    ]


def filter_by_aspect_ratio(
    detections: list[Detection],
    max_ratio: float = MAX_ASPECT_RATIO,
) -> list[Detection]:
    """Hard cut on aspect ratio — golf balls are round, not elongated."""
    kept: list[Detection] = []

    for d in detections:
        w = d.x2 - d.x1
        h = d.y2 - d.y1

        ratio = max(w, h) / max(min(w, h), 1e-6)

        if ratio <= max_ratio:
            kept.append(d)

    return kept


def filter_by_circularity(
    detections: list[Detection],
    image,
    threshold: float = CIRCULARITY_THRESHOLD,
    min_blob_px: int = CIRCULARITY_MIN_BLOB_PX,
) -> list[Detection]:
    """
    Reject non-circular blobs using contour-based circularity.
    circularity = 4π·area / perimeter²  (1.0 = perfect circle)

    Falls back to keeping the detection if the crop is too small or
    the contour can't be computed reliably.
    """
    kept: list[Detection] = []

    for d in detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < min_blob_px or crop.shape[1] < min_blob_px:
            kept.append(d)
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            kept.append(d)
            continue

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter < 1e-6 or area < 1.0:
            kept.append(d)
            continue

        circularity = 4.0 * math.pi * area / (perimeter ** 2)

        if circularity >= threshold:
            kept.append(d)

    return kept


def filter_tee_proximity(
    detections: list[Detection],
    tee_position: tuple[float, float] | None,
    ball_launched: bool,
    threshold: float = TEE_FILTER_RADIUS_POST_LAUNCH,
) -> list[Detection]:
    if not ball_launched or tee_position is None:
        return detections

    return [
        d for d in detections
        if distance_between_points(d.center, tee_position) > threshold
    ]


def filter_by_prediction_distance(
    detections: list[Detection],
    reference_position: tuple[float, float] | None,
    max_distance: float = MAX_POST_LAUNCH_CANDIDATE_DISTANCE,
) -> list[Detection]:
    """
    Post-launch safety gate.

    After launch, do not allow the tracker to jump to a random high-confidence
    blob that is very far away from the predicted trajectory.
    """
    if reference_position is None:
        return detections

    return [
        d for d in detections
        if distance_between_points(d.center, reference_position) <= max_distance
    ]


def deduplicate_detections(
    detections: list[Detection],
    center_distance_threshold: float,
) -> list[Detection]:
    unique: list[Detection] = []

    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(
            distance_between_points(detection.center, existing.center)
            > center_distance_threshold
            for existing in unique
        ):
            unique.append(detection)

    return unique


def apply_consensus_boost(
    roi_detections: list[Detection],
    full_detections: list[Detection],
    boost: float = CONSENSUS_BOOST,
    match_distance: float = DEDUP_DISTANCE * 2,
) -> list[Detection]:
    """
    If a detection from the full-frame pass is within match_distance of a
    ROI-pass detection, boost both confidences.

    This rewards candidates seen by two independent passes — a strong signal
    that it's a real ball.
    """
    boosted_roi = list(roi_detections)
    boosted_full = list(full_detections)

    for i, rd in enumerate(boosted_roi):
        for j, fd in enumerate(boosted_full):
            if distance_between_points(rd.center, fd.center) <= match_distance:
                boosted_roi[i] = Detection(
                    x1=rd.x1,
                    y1=rd.y1,
                    x2=rd.x2,
                    y2=rd.y2,
                    confidence=min(1.0, rd.confidence + boost),
                )

                boosted_full[j] = Detection(
                    x1=fd.x1,
                    y1=fd.y1,
                    x2=fd.x2,
                    y2=fd.y2,
                    confidence=min(1.0, fd.confidence + boost),
                )

    return boosted_roi + boosted_full


# ──────────────────────────────────────────────
# Candidate scoring / selection
# ──────────────────────────────────────────────
def estimate_motion_vector(
    recent_detected: deque[DetectedPoint],
) -> tuple[float, float] | None:
    if len(recent_detected) < 2:
        return None

    p1 = recent_detected[-2]
    p2 = recent_detected[-1]

    return p2.x - p1.x, p2.y - p1.y


def score_detection(
    detection: Detection,
    reference_position: tuple[float, float] | None,
    motion_vector: tuple[float, float] | None,
    ball_launched: bool = False,
) -> float:
    w = detection.x2 - detection.x1
    h = detection.y2 - detection.y1

    area = w * h
    aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
    compactness_penalty = abs(aspect_ratio - 1.0)

    score = detection.confidence * CONF_WEIGHT
    score -= area * AREA_WEIGHT
    score -= compactness_penalty * ASPECT_PENALTY_WEIGHT

    det_x, det_y = detection.center

    distance_w = (
        DISTANCE_WEIGHT_POST_LAUNCH
        if ball_launched
        else DISTANCE_WEIGHT
    )

    continuity_w = (
        MOTION_CONTINUITY_WEIGHT_POST_LAUNCH
        if ball_launched
        else MOTION_CONTINUITY_WEIGHT
    )

    # Soft distance penalty:
    # Before launch, this keeps the tracker near the tee.
    # After launch, this keeps the tracker near the predicted flight path.
    if reference_position is not None:
        dist = distance_between_points((det_x, det_y), reference_position)
        score -= dist * distance_w

    # Motion continuity:
    # Prefer detections that continue in the expected direction and speed.
    if motion_vector is not None and reference_position is not None:
        expected_x = reference_position[0] + motion_vector[0]
        expected_y = reference_position[1] + motion_vector[1]

        continuity_dist = distance_between_points(
            (det_x, det_y),
            (expected_x, expected_y),
        )

        score -= continuity_dist * continuity_w

    # Forward-motion penalty:
    # After launch, candidates behind the current motion direction are suspicious.
    # This prevents jumping backwards to tee noise or static false detections.
    if ball_launched and motion_vector is not None and reference_position is not None:
        candidate_vector = (
            det_x - reference_position[0],
            det_y - reference_position[1],
        )

        dot = (
            candidate_vector[0] * motion_vector[0]
            + candidate_vector[1] * motion_vector[1]
        )

        if dot < 0:
            score -= BACKWARD_MOTION_PENALTY

    return score


def choose_best_detection(
    detections: list[Detection],
    reference_position: tuple[float, float] | None,
    motion_vector: tuple[float, float] | None,
    ball_launched: bool = False,
) -> Detection | None:
    """Returns the single best-scoring detection, or None."""
    if not detections:
        return None

    return max(
        detections,
        key=lambda d: score_detection(
            detection=d,
            reference_position=reference_position,
            motion_vector=motion_vector,
            ball_launched=ball_launched,
        ),
    )


# ──────────────────────────────────────────────
# Tee position estimation
#
# FIX (BUG 1 / BUG 5): tee estimate is FROZEN as soon as the ball launches.
# Previously, in-flight positions were fed into update_tee_estimate, corrupting
# the tee location used for post-launch filtering.
# ──────────────────────────────────────────────
def update_tee_estimate(
    tee_samples: list[tuple[float, float]],
    detected_center: tuple[float, float] | None,
    frame_idx: int,
    ball_launched: bool,
) -> tuple[float, float] | None:
    # Once launched, freeze the tee position — never update from flight frames.
    if ball_launched:
        return average_point(tee_samples)

    if detected_center is None:
        return average_point(tee_samples)

    if frame_idx >= TEE_BOOTSTRAP_FRAMES:
        return average_point(tee_samples)

    current_estimate = average_point(tee_samples)

    if current_estimate is None:
        tee_samples.append(detected_center)
        return average_point(tee_samples)

    if distance_between_points(detected_center, current_estimate) <= TEE_BOOTSTRAP_MAX_RADIUS:
        tee_samples.append(detected_center)

    return average_point(tee_samples)


# ──────────────────────────────────────────────
# Dynamic trajectory-segment limit
# ──────────────────────────────────────────────
def dynamic_max_segment(
    recent_detected: deque[DetectedPoint],
    base: float = MAX_TRAJECTORY_SEGMENT_BASE,
    multiplier: float = 2.5,
) -> float:
    """
    Scale the max allowed trajectory segment by current estimated speed.
    Prevents the visualisation from showing broken lines on fast shots.

    This is only used for drawing. It should not influence tracking.
    """
    if len(recent_detected) < 2:
        return base

    p1 = recent_detected[-2]
    p2 = recent_detected[-1]

    speed = distance_between_points(p1.xy, p2.xy)

    return max(base, speed * multiplier)


def should_add_to_trajectory(track_source: str) -> bool:
    """
    By default, only draw the trajectory through real detections.

    Predicted tracker points can drift when the ball is missed, which makes the
    visual line look wrong even if the actual detector is doing okay.
    """
    if track_source == "detected":
        return True

    return DRAW_PREDICTED_TRAJECTORY_POINTS


def draw_trajectory(
    image,
    trajectory_points: list[TrajectoryPoint],
    recent_detected: deque[DetectedPoint],
) -> None:
    """
    Draws trajectory safely.

    Fixes two common visualisation bugs:
    1. It does not connect points across frame gaps.
    2. It does not connect huge jumps.
    """
    max_seg = dynamic_max_segment(recent_detected)

    for i in range(1, len(trajectory_points)):
        prev_point = trajectory_points[i - 1]
        curr_point = trajectory_points[i]

        frame_gap = curr_point.frame_idx - prev_point.frame_idx

        if frame_gap > MAX_TRAJECTORY_FRAME_GAP:
            continue

        segment_length = distance_between_points(
            (prev_point.x, prev_point.y),
            (curr_point.x, curr_point.y),
        )

        if segment_length > max_seg:
            continue

        line_color = (
            (0, 255, 0)
            if curr_point.source == "detected"
            else (0, 255, 255)
        )

        cv2.line(
            image,
            (prev_point.x, prev_point.y),
            (curr_point.x, curr_point.y),
            line_color,
            2,
        )


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root / "runs" / "detect" / "models" / "experiments"
        / "yolo11n_post_impact_12803" / "weights" / "best.pt"
    )

    frames_dir = (
        project_root / "data" / "extracted_frames" / "rory_005"
    )

    output_dir = (
        project_root / "output" / "detections" / "rory_005"
    )

    annotated_dir = output_dir / "annotated_frames"
    detection_csv_path = output_dir / "detections.csv"
    tracking_csv_path = output_dir / "tracking.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    tracker = BallTracker(
        max_missed=TRACKER_MAX_MISSED,
        distance_threshold=TRACKER_DISTANCE_THRESHOLD,
    )

    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    trajectory_points: list[TrajectoryPoint] = []

    # FIX (BUG 2): store DetectedPoint (frame_idx, x, y) not bare (x, y)
    recent_detected: deque[DetectedPoint] = deque(maxlen=5)

    tee_samples: list[tuple[float, float]] = []
    tee_position: tuple[float, float] | None = None

    ball_launched: bool = False
    frames_since_launch: int = 0

    last_detected_center: tuple[float, float] | None = None

    with (
        detection_csv_path.open("w", newline="", encoding="utf-8") as det_f,
        tracking_csv_path.open("w", newline="", encoding="utf-8") as trk_f,
    ):
        det_w = csv.writer(det_f)
        trk_w = csv.writer(trk_f)

        det_w.writerow(
            [
                "frame_name",
                "detection_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "width",
                "height",
            ]
        )

        trk_w.writerow(
            [
                "frame_name",
                "frame_idx",
                "x",
                "y",
                "source",
                "confidence",
            ]
        )

        for frame_idx, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path))

            if image is None:
                print(f"Warning: could not read image {frame_path.name}")
                continue

            image_height, image_width = image.shape[:2]

            predicted_position: tuple[float, float] | None = None
            roi_box_to_draw: tuple[int, int, int, int] | None = None
            roi_center_to_draw: tuple[float, float] | None = None
            selected_roi_box: tuple[int, int, int, int] | None = None

            roi_detections: list[Detection] = []
            full_detections: list[Detection] = []

            motion_vector = estimate_motion_vector(recent_detected)

            if tracker.is_initialized():
                predicted_position = tracker.predict()

                if ball_launched:
                    frames_since_launch += 1

                    # ── POST-LAUNCH ROI center with lead ─────────────────────
                    # Lead the ROI half a motion-vector step ahead so the ball
                    # stays near the centre of the crop.
                    if predicted_position is not None:
                        base_center = predicted_position
                    else:
                        base_center = (
                            extrapolate_position(recent_detected)
                            or last_detected_center
                        )

                    if base_center is not None and motion_vector is not None:
                        roi_center: tuple[float, float] | None = (
                            base_center[0] + motion_vector[0] * ROI_LEAD_FACTOR,
                            base_center[1] + motion_vector[1] * ROI_LEAD_FACTOR,
                        )
                    else:
                        roi_center = base_center

                    roi_conf = ROI_CONF_POST_LAUNCH
                    roi_upscale = ROI_UPSCALE_POST_LAUNCH

                    # Keep larger ROI for the first few post-launch frames.
                    roi_size = (
                        ROI_SIZE
                        if frames_since_launch <= POST_LAUNCH_LARGE_ROI_FRAMES
                        else ROI_SIZE_POST_LAUNCH
                    )

                else:
                    # ── PRE-LAUNCH ───────────────────────────────────────────
                    roi_center = (
                        last_detected_center
                        if last_detected_center is not None
                        else predicted_position
                    )

                    roi_conf = ROI_CONF
                    roi_upscale = ROI_UPSCALE
                    roi_size = ROI_SIZE

                if roi_center is not None:
                    roi_center_to_draw = roi_center

                    current_roi_detections, roi_box_to_draw = run_roi_detection_upscaled(
                        model=model,
                        image=image,
                        roi_center=roi_center,
                        roi_size=roi_size,
                        conf=roi_conf,
                        upscale=roi_upscale,
                    )

                    roi_detections.extend(current_roi_detections)

            # FIX (BUG 3): skip full-frame pass post-launch — ROI is sufficient
            # and the full-frame pass at low conf floods the pipeline with noise
            # that is then expensively filtered out anyway.
            if not ball_launched:
                full_detections = run_full_frame_detection(
                    model=model,
                    frame_path=frame_path,
                    conf=FULL_FRAME_CONF,
                )

            print(
                f"[{frame_path.name}] launched={ball_launched} "
                f"roi_raw={len(roi_detections)} "
                f"full_raw={len(full_detections)}"
            )

            # ── Consensus boost before merging ──────────────────────────────
            # Detections seen by both passes get a confidence bonus.
            if roi_detections and full_detections:
                all_detections = apply_consensus_boost(
                    roi_detections,
                    full_detections,
                )
            else:
                all_detections = roi_detections + full_detections

            # ── Standard filters ─────────────────────────────────────────────
            all_detections = filter_by_box_size(
                all_detections,
                MIN_BOX_SIZE,
                MAX_BOX_SIZE,
            )

            all_detections = filter_by_aspect_ratio(all_detections)
            all_detections = filter_by_circularity(all_detections, image)
            all_detections = deduplicate_detections(all_detections, DEDUP_DISTANCE)

            all_detections = filter_tee_proximity(
                detections=all_detections,
                tee_position=tee_position,
                ball_launched=ball_launched,
            )

            # ── Post-launch trajectory safety gate ───────────────────────────
            # Do not let the tracker jump to a random high-confidence blob far
            # away from the predicted flight path.
            if ball_launched:
                all_detections = filter_by_prediction_distance(
                    detections=all_detections,
                    reference_position=predicted_position,
                    max_distance=MAX_POST_LAUNCH_CANDIDATE_DISTANCE,
                )

            print(f"[{frame_path.name}] combined_filtered={len(all_detections)}")

            best = choose_best_detection(
                detections=all_detections,
                reference_position=predicted_position,
                motion_vector=motion_vector,
                ball_launched=ball_launched,
            )

            selected_detections = [best] if best is not None else []

            print(f"[{frame_path.name}] final_selected={len(selected_detections)}")

            if best is not None:
                selected_roi_box = get_roi_bounds(
                    image_width=image_width,
                    image_height=image_height,
                    center_x=best.center[0],
                    center_y=best.center[1],
                    roi_size=ROI_SIZE,
                )

                cx, cy = best.center

                det_w.writerow(
                    [
                        frame_path.name,
                        0,
                        0,
                        "golf_ball",
                        round(best.confidence, 6),
                        round(best.x1, 2),
                        round(best.y1, 2),
                        round(best.x2, 2),
                        round(best.y2, 2),
                        round(cx, 2),
                        round(cy, 2),
                        round(best.x2 - best.x1, 2),
                        round(best.y2 - best.y1, 2),
                    ]
                )

            if DRAW_ALL_CANDIDATES:
                draw_detection_boxes(
                    image,
                    all_detections,
                    color=(0, 255, 255),
                    prefix="cand",
                    thickness=1,
                )

            draw_detection_boxes(
                image,
                selected_detections,
                color=(0, 255, 0),
                prefix="ball",
                thickness=2,
            )

            track_point = tracker.step(
                frame_idx=frame_idx,
                detections=selected_detections,
                predicted_position=predicted_position,
            )

            # ── Update state from track result ───────────────────────────────
            detected_center_this_frame: tuple[float, float] | None = None

            if track_point is not None and track_point.source == "detected":
                detected_center_this_frame = (track_point.x, track_point.y)

                # FIX (BUG 2): store frame_idx for consecutive-frame velocity check.
                recent_detected.append(
                    DetectedPoint(frame_idx, track_point.x, track_point.y)
                )

                if not ball_launched:
                    last_detected_center = detected_center_this_frame

                elif (
                    tee_position is None
                    or distance_between_points(
                        detected_center_this_frame,
                        tee_position,
                    )
                    > TEE_FILTER_RADIUS_POST_LAUNCH
                ):
                    last_detected_center = detected_center_this_frame

            elif track_point is not None and track_point.source == "predicted":
                last_detected_center = (track_point.x, track_point.y)

            # ── Launch detection ─────────────────────────────────────────────
            if not ball_launched:
                ball_launched = detect_launch(
                    recent_detected=recent_detected,
                    tee_position=tee_position,
                )

                if ball_launched:
                    print(f"[{frame_path.name}] *** LAUNCH DETECTED ***")

            # FIX (BUG 1 / BUG 5): pass ball_launched so tee is frozen post-launch.
            tee_position = update_tee_estimate(
                tee_samples=tee_samples,
                detected_center=detected_center_this_frame,
                frame_idx=frame_idx,
                ball_launched=ball_launched,
            )

            # ── Write tracking CSV + overlay ─────────────────────────────────
            if track_point is not None:
                trk_w.writerow(
                    [
                        frame_path.name,
                        track_point.frame_idx,
                        round(track_point.x, 2),
                        round(track_point.y, 2),
                        track_point.source,
                        round(track_point.confidence, 6)
                        if track_point.confidence is not None
                        else "",
                    ]
                )

                track_x, track_y = int(track_point.x), int(track_point.y)

                if should_add_to_trajectory(track_point.source):
                    trajectory_points.append(
                        TrajectoryPoint(
                            frame_idx=frame_idx,
                            x=track_x,
                            y=track_y,
                            source=track_point.source,
                        )
                    )

                track_color = (
                    (0, 255, 0)
                    if track_point.source == "detected"
                    else (0, 255, 255)
                )

                cv2.circle(image, (track_x, track_y), 6, track_color, -1)

                launch_label = " [LAUNCHED]" if ball_launched else ""

                cv2.putText(
                    image,
                    f"track: {track_point.source}{launch_label}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ── ROI overlays ─────────────────────────────────────────────────
            if roi_box_to_draw is not None:
                rx1, ry1, rx2, ry2 = roi_box_to_draw
                cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

            if roi_center_to_draw is not None:
                cv2.circle(
                    image,
                    (int(roi_center_to_draw[0]), int(roi_center_to_draw[1])),
                    5,
                    (255, 0, 255),
                    -1,
                )

            if selected_roi_box is not None:
                sx1, sy1, sx2, sy2 = selected_roi_box
                cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)

            if tee_position is not None:
                cv2.circle(
                    image,
                    (int(tee_position[0]), int(tee_position[1])),
                    5,
                    (255, 255, 0),
                    -1,
                )

                cv2.putText(
                    image,
                    "tee_est",
                    (int(tee_position[0]) + 8, int(tee_position[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            mode_label = "post-launch" if ball_launched else "pre-launch"

            cv2.putText(
                image,
                f"mode: {mode_label}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ── Trajectory overlay, now frame-aware ──────────────────────────
            draw_trajectory(
                image=image,
                trajectory_points=trajectory_points,
                recent_detected=recent_detected,
            )

            cv2.imwrite(str(annotated_dir / frame_path.name), image)

    print("Detection + tracking run complete.")
    print(f"Frames processed : {len(frame_paths)}")
    print(f"Detection CSV    : {detection_csv_path}")
    print(f"Tracking CSV     : {tracking_csv_path}")
    print(f"Annotated frames : {annotated_dir}")


if __name__ == "__main__":
    main()