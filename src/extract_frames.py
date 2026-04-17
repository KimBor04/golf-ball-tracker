from pathlib import Path
import argparse
import cv2


def extract_frames(video_path: Path, output_dir: Path, every_n: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    saved_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % every_n == 0:
            out_path = output_dir / f"frame_{saved_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(f"Video: {video_path.name}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Saved frames: {saved_idx}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--every-n", type=int, default=3, help="Save every n-th frame")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path("data/extracted_frames") / video_path.stem

    extract_frames(video_path, output_dir, every_n=args.every_n)