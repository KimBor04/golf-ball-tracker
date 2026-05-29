from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    data_yaml = project_root / "data" / "raw" / "rucv_split" / "data.yaml"
    model_path = project_root / "models" / "yolo11n.pt"

    model = YOLO(str(model_path))

    model.train(
        data=str(project_root / "data" / "raw" / "rucv_split" / "data.yaml"),
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        cache=True,
        workers=2,
        project=str(project_root / "models" / "experiments"),
        name="yolo11n_rucv_only_15ep",
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()