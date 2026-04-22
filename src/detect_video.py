from pathlib import Path
import csv
import math

import cv2
from ultralytics import YOLO

from tracker import BallTracker, Detection, yolo_result_to_detections


TEE_ROI_SIZE = 250
FLIGHT_ROI_SIZE = 400

FULL_FRAME_CONF = 0.08
LAUNCH_CONF = 0.20

LAUNCH_FRAMES = 12
FORCED_FULL_FRAME_FRAMES = 10
TEE_IGNORE_RADIUS = 50
DETECTED_STREAK_REQUIRED = 8
EARLY_FLIGHT_Y_MARGIN = 8.0
FORCE_FLIGHT_DISTANCE = 40.0

MIN_BALL_SIZE = 2.0
MAX_BALL_SIZE = 25.0


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


def convert_roi_result_to_full_frame_detections(
    result,
    offset_x: int,
    offset_y: int,
) -> list[Detection]:
    detections: list[Detection] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = box.tolist()
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
) -> None:
    for detection in detections:
        x1_i = int(detection.x1)
        y1_i = int(detection.y1)
        x2_i = int(detection.x2)
        y2_i = int(detection.y2)

        center_x, center_y = detection.center

        cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        label = f"{prefix} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1_i, max(y1_i - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            (int(center_x), int(center_y)),
            4,
            (0, 0, 255),
            -1,
        )


def filter_detections_near_tee(
    detections: list[Detection],
    tee_position: tuple[float, float] | None,
    ignore_radius: float,
) -> list[Detection]:
    if tee_position is None:
        return detections

    filtered: list[Detection] = []

    for detection in detections:
        if distance_between_points(detection.center, tee_position) > ignore_radius:
            filtered.append(detection)

    return filtered


def filter_detections_below_tee(
    detections: list[Detection],
    tee_position: tuple[float, float] | None,
    y_margin: float,
) -> list[Detection]:
    if tee_position is None:
        return detections

    filtered: list[Detection] = []

    for detection in detections:
        _, detection_y = detection.center
        tee_y = tee_position[1]

        if detection_y <= tee_y + y_margin:
            filtered.append(detection)

    return filtered


def filter_detections_by_size(
    detections: list[Detection],
    min_size: float,
    max_size: float,
) -> list[Detection]:
    filtered: list[Detection] = []

    for detection in detections:
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1

        if min_size <= width <= max_size and min_size <= height <= max_size:
            filtered.append(detection)

    return filtered


def choose_highest_confidence_detection(
    detections: list[Detection],
) -> list[Detection]:
    if not detections:
        return []

    best = max(detections, key=lambda det: det.confidence)
    return [best]


def run_full_frame_detection(
    model: YOLO,
    frame_path: Path,
    conf: float,
) -> list[Detection]:
    full_results = model.predict(
        source=str(frame_path),
        conf=conf,
        iou=0.7,
        save=False,
        verbose=False,
        device=0,
    )
    full_result = full_results[0]
    return yolo_result_to_detections(full_result)


def run_roi_detection(
    model: YOLO,
    image,
    predicted_position: tuple[float, float],
    roi_size: int,
    conf: float,
) -> tuple[list[Detection], tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_bounds(
        image_width=image_width,
        image_height=image_height,
        center_x=predicted_position[0],
        center_y=predicted_position[1],
        roi_size=roi_size,
    )

    roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi_image.size == 0:
        return [], (roi_x1, roi_y1, roi_x2, roi_y2)

    roi_results = model.predict(
        source=roi_image,
        conf=conf,
        iou=0.7,
        save=False,
        verbose=False,
        device=0,
    )
    roi_result = roi_results[0]

    detections = convert_roi_result_to_full_frame_detections(
        roi_result,
        offset_x=roi_x1,
        offset_y=roi_y1,
    )

    return detections, (roi_x1, roi_y1, roi_x2, roi_y2)


def apply_airborne_filters(
    detections: list[Detection],
    tee_position: tuple[float, float] | None,
) -> list[Detection]:
    detections = filter_detections_near_tee(
        detections=detections,
        tee_position=tee_position,
        ignore_radius=TEE_IGNORE_RADIUS,
    )
    detections = filter_detections_below_tee(
        detections=detections,
        tee_position=tee_position,
        y_margin=EARLY_FLIGHT_Y_MARGIN,
    )
    detections = filter_detections_by_size(
        detections=detections,
        min_size=MIN_BALL_SIZE,
        max_size=MAX_BALL_SIZE,
    )
    return detections


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "models"
        / "experiments"
        / "yolo11n_baseline_combined_rucv_15ep"
        / "weights"
        / "best.pt"
    )
    frames_dir = project_root / "data" / "extracted_frames" / "rory_002"

    output_dir = project_root / "output" / "detections" / "rory_002"
    annotated_dir = output_dir / "annotated_frames"
    detection_csv_path = output_dir / "detections.csv"
    tracking_csv_path = output_dir / "tracking.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    tracker = BallTracker(max_missed=10, distance_threshold=120.0)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    trajectory_points = []

    state = "tee"  # "tee", "launch", "flight"
    launch_counter = 0
    forced_full_frame_counter = 0

    tee_position: tuple[float, float] | None = None
    detected_streak = 0

    with (
        detection_csv_path.open("w", newline="", encoding="utf-8") as detection_csv_file,
        tracking_csv_path.open("w", newline="", encoding="utf-8") as tracking_csv_file,
    ):
        detection_writer = csv.writer(detection_csv_file)
        tracking_writer = csv.writer(tracking_csv_file)

        detection_writer.writerow(
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

        tracking_writer.writerow(
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

            detections: list[Detection] = []
            predicted_position: tuple[float, float] | None = None
            search_mode = "full-frame"
            roi_box: tuple[int, int, int, int] | None = None

            if state == "tee":
                if tracker.is_initialized():
                    predicted_position = tracker.predict()
                    detections, roi_box = run_roi_detection(
                        model=model,
                        image=image,
                        predicted_position=predicted_position,
                        roi_size=TEE_ROI_SIZE,
                        conf=FULL_FRAME_CONF,
                    )
                    search_mode = "ROI-tee"

                if not detections:
                    detections = run_full_frame_detection(
                        model=model,
                        frame_path=frame_path,
                        conf=FULL_FRAME_CONF,
                    )
                    search_mode = "full-frame"

                detections = choose_highest_confidence_detection(detections)

            elif state == "launch":
                detections = run_full_frame_detection(
                    model=model,
                    frame_path=frame_path,
                    conf=LAUNCH_CONF,
                )
                detections = apply_airborne_filters(
                    detections=detections,
                    tee_position=tee_position,
                )
                detections = choose_highest_confidence_detection(detections)
                search_mode = "full-frame-launch"

            elif state == "flight":
                use_roi = tracker.is_initialized() and forced_full_frame_counter == 0

                if use_roi:
                    predicted_position = tracker.predict()
                    detections, roi_box = run_roi_detection(
                        model=model,
                        image=image,
                        predicted_position=predicted_position,
                        roi_size=FLIGHT_ROI_SIZE,
                        conf=FULL_FRAME_CONF,
                    )
                    detections = apply_airborne_filters(
                        detections=detections,
                        tee_position=tee_position,
                    )
                    search_mode = "ROI-flight"

                if not detections:
                    detections = run_full_frame_detection(
                        model=model,
                        frame_path=frame_path,
                        conf=FULL_FRAME_CONF,
                    )
                    detections = apply_airborne_filters(
                        detections=detections,
                        tee_position=tee_position,
                    )
                    detections = choose_highest_confidence_detection(detections)
                    search_mode = "full-frame"

            for det_id, detection in enumerate(detections):
                center_x, center_y = detection.center
                width = detection.x2 - detection.x1
                height = detection.y2 - detection.y1

                detection_writer.writerow(
                    [
                        frame_path.name,
                        det_id,
                        0,
                        "golf_ball",
                        round(detection.confidence, 6),
                        round(detection.x1, 2),
                        round(detection.y1, 2),
                        round(detection.x2, 2),
                        round(detection.y2, 2),
                        round(center_x, 2),
                        round(center_y, 2),
                        round(width, 2),
                        round(height, 2),
                    ]
                )

            draw_detection_boxes(image, detections, color=(0, 255, 0), prefix="ball")

            track_point = tracker.step(
                frame_idx=frame_idx,
                detections=detections,
                predicted_position=predicted_position,
            )

            detected_center: tuple[float, float] | None = None
            if track_point is not None and track_point.source == "detected":
                detected_center = (track_point.x, track_point.y)

            if state == "tee":
                if detected_center is not None:
                    detected_streak += 1
                    if tee_position is None:
                        tee_position = detected_center

                    if (
                        tee_position is not None
                        and distance_between_points(detected_center, tee_position) > FORCE_FLIGHT_DISTANCE
                    ):
                        state = "flight"
                        forced_full_frame_counter = FORCED_FULL_FRAME_FRAMES

                else:
                    if detected_streak >= DETECTED_STREAK_REQUIRED:
                        state = "launch"
                        launch_counter = LAUNCH_FRAMES
                        forced_full_frame_counter = FORCED_FULL_FRAME_FRAMES
                    detected_streak = 0

            elif state == "launch":
                launch_counter -= 1

                if detected_center is not None:
                    if (
                        tee_position is None
                        or distance_between_points(detected_center, tee_position) > TEE_IGNORE_RADIUS
                    ):
                        state = "flight"
                        forced_full_frame_counter = FORCED_FULL_FRAME_FRAMES
                elif launch_counter <= 0:
                    state = "flight"
                    forced_full_frame_counter = FORCED_FULL_FRAME_FRAMES

            elif state == "flight":
                if forced_full_frame_counter > 0:
                    forced_full_frame_counter -= 1

            if track_point is not None:
                tracking_writer.writerow(
                    [
                        frame_path.name,
                        track_point.frame_idx,
                        round(track_point.x, 2),
                        round(track_point.y, 2),
                        track_point.source,
                        (
                            round(track_point.confidence, 6)
                            if track_point.confidence is not None
                            else ""
                        ),
                    ]
                )

                track_x = int(track_point.x)
                track_y = int(track_point.y)
                trajectory_points.append((track_x, track_y, track_point.source))

                if track_point.source == "detected":
                    track_color = (0, 255, 0)
                else:
                    track_color = (0, 255, 255)

                cv2.circle(image, (track_x, track_y), 6, track_color, -1)
                cv2.putText(
                    image,
                    f"track: {track_point.source}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if roi_box is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
                cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            mode_text = f"mode: {search_mode}"
            cv2.putText(
                image,
                mode_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            status_text = (
                f"state: {state} | detected_streak: {detected_streak} "
                f"| launch_left: {launch_counter} | force_ff: {forced_full_frame_counter}"
            )
            cv2.putText(
                image,
                status_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for i in range(1, len(trajectory_points)):
                x1, y1, _ = trajectory_points[i - 1]
                x2, y2, source = trajectory_points[i]

                if source == "detected":
                    line_color = (0, 255, 0)
                else:
                    line_color = (0, 255, 255)

                cv2.line(image, (x1, y1), (x2, y2), line_color, 2)

            annotated_path = annotated_dir / frame_path.name
            cv2.imwrite(str(annotated_path), image)

    print("Detection + tracking run complete.")
    print(f"Frames processed: {len(frame_paths)}")
    print(f"Detection CSV saved to: {detection_csv_path}")
    print(f"Tracking CSV saved to: {tracking_csv_path}")
    print(f"Annotated frames saved to: {annotated_dir}")


if __name__ == "__main__":
    main()