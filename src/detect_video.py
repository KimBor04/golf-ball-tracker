from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "models"
        / "experiments"
        / "yolo11n_baseline_15ep2"
        / "weights"
        / "best.pt"
    )
    frames_dir = project_root / "data" / "extracted_frames" / "rory_002"

    output_dir = project_root / "output" / "detections" / "rory_002"
    annotated_dir = output_dir / "annotated_frames"
    csv_path = output_dir / "detections.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
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

        for frame_path in frame_paths:
            results = model.predict(
                source=str(frame_path),
                conf=0.25,
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

                    writer.writerow(
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

            annotated_path = annotated_dir / frame_path.name
            cv2.imwrite(str(annotated_path), image)

    print("Detection run complete.")
    print(f"Frames processed: {len(frame_paths)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Annotated frames saved to: {annotated_dir}")


if __name__ == "__main__":
    main()