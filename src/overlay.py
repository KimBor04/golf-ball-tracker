from pathlib import Path

import cv2


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    frames_dir = (
        project_root
        / "output"
        / "detections"
        / "rory_002"
        / "annotated_frames"
    )
    output_video_path = (
        project_root
        / "output"
        / "detections"
        / "rory_002"
        / "annotated_tracking.mp4"
    )

    fps = 30.0

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
        fps,
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

        writer.write(frame)
        written_frames += 1

    writer.release()

    print("Overlay video export complete.")
    print(f"Frames written: {written_frames}")
    print(f"FPS: {fps}")
    print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    main()