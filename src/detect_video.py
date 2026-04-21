from pathlib import Path
import csv

import cv2
from ultralytics import YOLO

from tracker import BallTracker, yolo_result_to_detections

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
    frames_dir = project_root / "data" / "extracted_frames" / "rory_004"

    output_dir = project_root / "output" / "detections" / "rory_004"
    annotated_dir = output_dir / "annotated_frames"
    detection_csv_path = output_dir / "detections.csv"
    tracking_csv_path = output_dir / "tracking.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    tracker = BallTracker(max_missed=10, distance_threshold=80.0)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    trajectory_points = []

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
            results = model.predict(
                source=str(frame_path),
                conf=0.05,
                iou=0.7,
                save=False,
                verbose=False,
                device=0,
            )

            result = results[0]
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"Warning: could not read image {frame_path.name}")
                continue

            boxes = result.boxes
            names = result.names

            if boxes is not None and len(boxes) > 0:
                for det_id, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    detection_writer.writerow(
                        [
                            frame_path.name,
                            det_id,
                            cls_id,
                            names[cls_id],
                            round(conf, 6),
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                            round(center_x, 2),
                            round(center_y, 2),
                            round(width, 2),
                            round(height, 2),
                        ]
                    )

                    cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                    label = f"{names[cls_id]} {conf:.2f}"
                    cv2.putText(
                        image,
                        label,
                        (x1_i, max(y1_i - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
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

            detections = yolo_result_to_detections(result)
            track_point = tracker.step(frame_idx=frame_idx, detections=detections)

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