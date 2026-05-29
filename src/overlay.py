from pathlib import Path
import csv
import re

import cv2


# ──────────────────────────────────────────────
# Video selection
# ──────────────────────────────────────────────
VIDEO_NAME = "rory_005"
# VIDEO_NAME = "rory_dtl_ball_flight_001"


# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────
FPS = 30.0

DRAW_LINE_PROGRESSIVELY = True
DRAW_TRAJECTORY_POINTS = True
SAVE_FILTERED_DETECTIONS_CSV = True

# ──────────────────────────────────────────────
# General detection filtering
# ──────────────────────────────────────────────
START_FRAME = 0
END_FRAME: int | None = None

# Keep this low because the real ball can be low-confidence.
MIN_CONFIDENCE = 0.08

# General ball-shape filters.
# These should work across videos better than fixed x/y regions.
MIN_BOX_WIDTH = 1.0
MIN_BOX_HEIGHT = 1.0
MAX_BOX_WIDTH = 45.0
MAX_BOX_HEIGHT = 45.0
MAX_ASPECT_RATIO = 2.2

# ──────────────────────────────────────────────
# Automatic path finding
# ──────────────────────────────────────────────
# Important:
# Do not only try the highest-confidence detections as starts.
# Static tee/ground detections often have very high confidence.
MAX_START_CANDIDATES = 200

# Candidate linking.
MAX_FRAME_GAP = 25
MAX_SPEED_PX_PER_FRAME = 140.0
MIN_MOVE_DISTANCE = 2.5

# Path scoring.
PATH_LENGTH_WEIGHT = 1000.0
CONFIDENCE_WEIGHT = 120.0
SMOOTHNESS_WEIGHT = 2.0
DISTANCE_WEIGHT = 1.0
FRAME_GAP_WEIGHT = 0.5

# Avoid paths that are basically static bright blobs.
MIN_TOTAL_PATH_DISTANCE = 40.0

# Remove almost identical repeated points if they do not add visual value.
REMOVE_NEAR_DUPLICATES = True
MIN_DISTANCE_BETWEEN_DRAWN_POINTS = 4.0

# Smooth final line slightly.
SMOOTH_TRAJECTORY = True
SMOOTHING_WINDOW = 3

# ──────────────────────────────────────────────
# Optional visual trajectory extension
# ──────────────────────────────────────────────
DRAW_ESTIMATED_CONTINUATION = True

ESTIMATED_CONTINUATION_POINTS = 8
ESTIMATED_FRAME_STEP = 3
ESTIMATE_DECAY = 0.92

# Optional manual tee anchors by video.
# This is only used if available. It does not affect automatic path detection.
TEE_POINTS_BY_VIDEO: dict[str, tuple[float, float] | None] = {
    "rory_dtl_ball_flight_001": (1051.22, 1014.38),
    "lydia_ko_dtl_driver_001": None,
}

DRAW_TEE_TO_FIRST_DETECTION = True
TEE_FRAME_IDX = 0

# Colors in BGR
LINE_COLOR = (0, 255, 0)
POINT_COLOR = (0, 255, 0)
ESTIMATED_POINT_COLOR = (0, 255, 255)
TEE_POINT_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)


def extract_frame_idx(frame_name: str) -> int:
    matches = re.findall(r"\d+", frame_name)

    if not matches:
        raise ValueError(f"Could not extract frame index from: {frame_name}")

    return int(matches[-1])


def distance_between_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return (dx * dx + dy * dy) ** 0.5


def read_detection_candidates(
    detections_csv_path: Path,
) -> list[dict]:
    candidates: list[dict] = []

    if not detections_csv_path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {detections_csv_path}")

    with detections_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_name = row["frame_name"]
            frame_idx = extract_frame_idx(frame_name)

            confidence = float(row["confidence"])
            x = float(row["center_x"])
            y = float(row["center_y"])
            width = float(row["width"])
            height = float(row["height"])

            if frame_idx < START_FRAME:
                continue

            if END_FRAME is not None and frame_idx > END_FRAME:
                continue

            if confidence < MIN_CONFIDENCE:
                continue

            if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                continue

            if width > MAX_BOX_WIDTH or height > MAX_BOX_HEIGHT:
                continue

            aspect_ratio = max(width, height) / max(min(width, height), 1e-6)

            if aspect_ratio > MAX_ASPECT_RATIO:
                continue

            candidates.append(
                {
                    "frame_name": frame_name,
                    "frame_idx": frame_idx,
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "width": width,
                    "height": height,
                    "source": "real_detection",
                }
            )

    candidates.sort(key=lambda p: p["frame_idx"])
    return candidates


def group_candidates_by_frame(
    candidates: list[dict],
) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}

    for candidate in candidates:
        by_frame.setdefault(candidate["frame_idx"], []).append(candidate)

    return by_frame


def candidate_link_score(
    current: dict,
    candidate: dict,
    previous_motion: tuple[float, float] | None,
) -> float | None:
    frame_gap = candidate["frame_idx"] - current["frame_idx"]

    if frame_gap <= 0:
        return None

    if frame_gap > MAX_FRAME_GAP:
        return None

    dist = distance_between_points(
        (current["x"], current["y"]),
        (candidate["x"], candidate["y"]),
    )

    speed = dist / max(frame_gap, 1)

    if speed > MAX_SPEED_PX_PER_FRAME:
        return None

    score = 0.0
    score += candidate["confidence"] * CONFIDENCE_WEIGHT
    score -= dist * DISTANCE_WEIGHT
    score -= frame_gap * FRAME_GAP_WEIGHT

    if previous_motion is not None:
        expected_x = current["x"] + previous_motion[0] * frame_gap
        expected_y = current["y"] + previous_motion[1] * frame_gap

        expected_dist = distance_between_points(
            (candidate["x"], candidate["y"]),
            (expected_x, expected_y),
        )

        score -= expected_dist * SMOOTHNESS_WEIGHT

    return score


def build_path_from_start(
    start: dict,
    by_frame: dict[int, list[dict]],
    frames: list[int],
) -> list[dict]:
    path: list[dict] = [start]

    current = start
    previous_motion: tuple[float, float] | None = None

    for frame_idx in frames:
        if frame_idx <= current["frame_idx"]:
            continue

        if frame_idx - current["frame_idx"] > MAX_FRAME_GAP:
            continue

        scored_candidates: list[tuple[float, dict]] = []

        for candidate in by_frame[frame_idx]:
            score = candidate_link_score(
                current=current,
                candidate=candidate,
                previous_motion=previous_motion,
            )

            if score is None:
                continue

            scored_candidates.append((score, candidate))

        if not scored_candidates:
            continue

        _, best_candidate = max(scored_candidates, key=lambda item: item[0])

        move_dist = distance_between_points(
            (current["x"], current["y"]),
            (best_candidate["x"], best_candidate["y"]),
        )

        if move_dist < MIN_MOVE_DISTANCE:
            continue

        frame_gap = best_candidate["frame_idx"] - current["frame_idx"]

        previous_motion = (
            (best_candidate["x"] - current["x"]) / max(frame_gap, 1),
            (best_candidate["y"] - current["y"]) / max(frame_gap, 1),
        )

        path.append(best_candidate)
        current = best_candidate

    return path


def total_path_distance(points: list[dict]) -> float:
    if len(points) < 2:
        return 0.0

    total = 0.0

    for i in range(1, len(points)):
        total += distance_between_points(
            (points[i - 1]["x"], points[i - 1]["y"]),
            (points[i]["x"], points[i]["y"]),
        )

    return total


def path_smoothness_penalty(points: list[dict]) -> float:
    if len(points) < 3:
        return 0.0

    penalty = 0.0

    for i in range(2, len(points)):
        p0 = points[i - 2]
        p1 = points[i - 1]
        p2 = points[i]

        frame_gap_1 = max(p1["frame_idx"] - p0["frame_idx"], 1)
        frame_gap_2 = max(p2["frame_idx"] - p1["frame_idx"], 1)

        v1 = (
            (p1["x"] - p0["x"]) / frame_gap_1,
            (p1["y"] - p0["y"]) / frame_gap_1,
        )
        v2 = (
            (p2["x"] - p1["x"]) / frame_gap_2,
            (p2["y"] - p1["y"]) / frame_gap_2,
        )

        penalty += distance_between_points(v1, v2)

    return penalty


def score_path(points: list[dict]) -> float:
    if len(points) < 2:
        return -1_000_000.0

    movement = total_path_distance(points)

    if movement < MIN_TOTAL_PATH_DISTANCE:
        return -1_000_000.0

    avg_confidence = sum(p["confidence"] for p in points) / len(points)
    smoothness_penalty = path_smoothness_penalty(points)

    score = 0.0
    score += len(points) * PATH_LENGTH_WEIGHT
    score += avg_confidence * CONFIDENCE_WEIGHT
    score += movement
    score -= smoothness_penalty * SMOOTHNESS_WEIGHT

    return score


def get_start_candidates(
    candidates: list[dict],
) -> list[dict]:
    """
    Pick diverse start candidates across the whole video.

    Important:
    Do not only use highest-confidence detections. In golf-ball videos,
    static tee detections often have confidence 1.0 and would dominate the
    start list, while real flight detections can have lower confidence.
    """
    best_by_frame: dict[int, dict] = {}

    for candidate in candidates:
        frame_idx = candidate["frame_idx"]

        if frame_idx not in best_by_frame:
            best_by_frame[frame_idx] = candidate
            continue

        if candidate["confidence"] > best_by_frame[frame_idx]["confidence"]:
            best_by_frame[frame_idx] = candidate

    frame_candidates = sorted(
        best_by_frame.values(),
        key=lambda p: p["frame_idx"],
    )

    high_confidence_candidates = sorted(
        candidates,
        key=lambda p: p["confidence"],
        reverse=True,
    )

    combined: list[dict] = []
    seen: set[tuple[int, float, float]] = set()

    for candidate in frame_candidates + high_confidence_candidates:
        key = (
            candidate["frame_idx"],
            round(candidate["x"], 2),
            round(candidate["y"], 2),
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(candidate)

        if len(combined) >= MAX_START_CANDIDATES:
            break

    return combined


def build_best_path(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    by_frame = group_candidates_by_frame(candidates)
    frames = sorted(by_frame.keys())

    start_candidates = get_start_candidates(candidates)

    best_path: list[dict] = []
    best_score = -1_000_000.0

    for start in start_candidates:
        path = build_path_from_start(
            start=start,
            by_frame=by_frame,
            frames=frames,
        )

        score = score_path(path)

        if score > best_score:
            best_score = score
            best_path = path

    best_path.sort(key=lambda p: p["frame_idx"])

    if REMOVE_NEAR_DUPLICATES:
        best_path = remove_near_duplicate_points(best_path)

    if SMOOTH_TRAJECTORY:
        best_path = smooth_points(best_path, SMOOTHING_WINDOW)

    return best_path


def remove_near_duplicate_points(points: list[dict]) -> list[dict]:
    if len(points) <= 1:
        return points

    cleaned: list[dict] = [points[0]]

    for point in points[1:]:
        prev_point = cleaned[-1]

        dist = distance_between_points(
            (prev_point["x"], prev_point["y"]),
            (point["x"], point["y"]),
        )

        if dist >= MIN_DISTANCE_BETWEEN_DRAWN_POINTS:
            cleaned.append(point)

    return cleaned


def smooth_points(
    points: list[dict],
    window_size: int,
) -> list[dict]:
    if len(points) < window_size:
        return points

    smoothed: list[dict] = []
    half_window = window_size // 2

    for i, point in enumerate(points):
        start = max(0, i - half_window)
        end = min(len(points), i + half_window + 1)

        window = points[start:end]

        avg_x = sum(p["x"] for p in window) / len(window)
        avg_y = sum(p["y"] for p in window) / len(window)

        new_point = dict(point)
        new_point["x"] = avg_x
        new_point["y"] = avg_y

        smoothed.append(new_point)

    return smoothed


def add_visual_estimates(points: list[dict]) -> list[dict]:
    if not points:
        return points

    extended: list[dict] = []

    tee_point = TEE_POINTS_BY_VIDEO.get(VIDEO_NAME)

    if DRAW_TEE_TO_FIRST_DETECTION and tee_point is not None:
        extended.append(
            {
                "frame_name": "manual_tee_point",
                "frame_idx": TEE_FRAME_IDX,
                "x": tee_point[0],
                "y": tee_point[1],
                "confidence": 1.0,
                "width": 0.0,
                "height": 0.0,
                "source": "estimated_tee",
            }
        )

    extended.extend(points)

    if DRAW_ESTIMATED_CONTINUATION and len(points) >= 2:
        prev_point = points[-2]
        last_point = points[-1]

        dx = last_point["x"] - prev_point["x"]
        dy = last_point["y"] - prev_point["y"]

        current_x = last_point["x"]
        current_y = last_point["y"]
        current_frame_idx = last_point["frame_idx"]

        current_dx = dx
        current_dy = dy

        for i in range(ESTIMATED_CONTINUATION_POINTS):
            current_x += current_dx
            current_y += current_dy
            current_frame_idx += ESTIMATED_FRAME_STEP

            extended.append(
                {
                    "frame_name": f"estimated_continuation_{i + 1}",
                    "frame_idx": current_frame_idx,
                    "x": current_x,
                    "y": current_y,
                    "confidence": 1.0,
                    "width": 0.0,
                    "height": 0.0,
                    "source": "estimated_continuation",
                }
            )

            current_dx *= ESTIMATE_DECAY
            current_dy *= ESTIMATE_DECAY

    return extended


def save_points_csv(
    output_path: Path,
    points: list[dict],
) -> None:
    if not SAVE_FILTERED_DETECTIONS_CSV:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame_name",
                "frame_idx",
                "x",
                "y",
                "confidence",
                "width",
                "height",
                "source",
            ]
        )

        for point in points:
            writer.writerow(
                [
                    point["frame_name"],
                    point["frame_idx"],
                    round(point["x"], 2),
                    round(point["y"], 2),
                    round(point["confidence"], 6),
                    round(point["width"], 2),
                    round(point["height"], 2),
                    point.get("source", "real_detection"),
                ]
            )


def get_visible_points_for_frame(
    all_points: list[dict],
    current_frame_idx: int,
) -> list[dict]:
    if not DRAW_LINE_PROGRESSIVELY:
        return all_points

    return [
        p for p in all_points
        if p["frame_idx"] <= current_frame_idx
    ]


def point_color(point: dict) -> tuple[int, int, int]:
    source = point.get("source", "real_detection")

    if source == "estimated_tee":
        return TEE_POINT_COLOR

    if source == "estimated_continuation":
        return ESTIMATED_POINT_COLOR

    return POINT_COLOR


def draw_final_trajectory(
    frame,
    points: list[dict],
) -> None:
    if len(points) < 2:
        return

    for i in range(1, len(points)):
        prev_point = points[i - 1]
        curr_point = points[i]

        cv2.line(
            frame,
            (int(prev_point["x"]), int(prev_point["y"])),
            (int(curr_point["x"]), int(curr_point["y"])),
            LINE_COLOR,
            3,
        )

    if DRAW_TRAJECTORY_POINTS:
        for point in points:
            cv2.circle(
                frame,
                (int(point["x"]), int(point["y"])),
                5,
                point_color(point),
                -1,
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    frames_dir = (
        project_root
        / "data"
        / "extracted_frames"
        / VIDEO_NAME
    )

    output_dir = (
        project_root
        / "output"
        / "detections"
        / VIDEO_NAME
    )

    detections_csv_path = output_dir / "detections.csv"
    filtered_csv_path = output_dir / "detections_filtered_for_overlay.csv"
    output_video_path = output_dir / "annotated_tracking.mp4"

    candidates = read_detection_candidates(detections_csv_path)
    real_trajectory_points = build_best_path(candidates)
    trajectory_points = add_visual_estimates(real_trajectory_points)

    save_points_csv(
        output_path=filtered_csv_path,
        points=trajectory_points,
    )

    print(f"Video: {VIDEO_NAME}")
    print(f"Detection candidates: {len(candidates)}")
    print(f"Real trajectory points: {len(real_trajectory_points)}")
    print(f"Trajectory points including estimates: {len(trajectory_points)}")
    print(f"Saved filtered trajectory CSV: {filtered_csv_path}")

    if len(real_trajectory_points) < 2:
        print("Warning: not enough real trajectory points.")
        print("Try lowering MIN_CONFIDENCE or increasing MAX_BOX_WIDTH/MAX_BOX_HEIGHT.")

    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0].name}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video_path}")

    written_frames = 0

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"Warning: could not read frame {frame_path.name}")
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        actual_frame_idx = extract_frame_idx(frame_path.name)

        visible_points = get_visible_points_for_frame(
            all_points=trajectory_points,
            current_frame_idx=actual_frame_idx,
        )

        draw_final_trajectory(
            frame=frame,
            points=visible_points,
        )

        cv2.putText(
            frame,
            f"trajectory points: {len(visible_points)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        written_frames += 1

    writer.release()

    print("Overlay video export complete.")
    print(f"Frames written: {written_frames}")
    print(f"FPS: {FPS}")
    print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    main()