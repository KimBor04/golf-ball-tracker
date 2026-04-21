from pathlib import Path
import random
import shutil


def main() -> None:
    random.seed(42)

    project_root = Path(__file__).resolve().parents[1]

    source_images_dir = project_root / "data" / "raw" / "rucv_yolo" / "train" / "images"
    source_labels_dir = project_root / "data" / "raw" / "rucv_yolo" / "train" / "labels"

    target_root = project_root / "data" / "raw" / "rucv_split"

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    image_paths = sorted(
        [p for p in source_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    pairs = []
    for image_path in image_paths:
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise ValueError("No image-label pairs found in rucv_yolo.")

    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }

    for split_name, split_pairs in splits.items():
        images_out = target_root / split_name / "images"
        labels_out = target_root / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, images_out / image_path.name)
            shutil.copy2(label_path, labels_out / label_path.name)

    print(f"Total pairs: {total}")
    for split_name, split_pairs in splits.items():
        print(f"{split_name}: {len(split_pairs)}")


if __name__ == "__main__":
    main()