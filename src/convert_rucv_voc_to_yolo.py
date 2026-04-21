from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


CLASS_MAPPING = {
    "golfball": 0,
    "golf_ball": 0,
    "ball": 0,
}


def voc_to_yolo(size_w: int, size_h: int, xmin: float, ymin: float, xmax: float, ymax: float):
    x_center = ((xmin + xmax) / 2.0) / size_w
    y_center = ((ymin + ymax) / 2.0) / size_h
    width = (xmax - xmin) / size_w
    height = (ymax - ymin) / size_h
    return x_center, y_center, width, height


def convert_xml_file(xml_path: Path, output_txt_path: Path) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return False

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))

    if width <= 0 or height <= 0:
        return False

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip().lower()
        if class_name not in CLASS_MAPPING:
            continue

        class_id = CLASS_MAPPING[class_name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))

        x_center, y_center, box_w, box_h = voc_to_yolo(width, height, xmin, ymin, xmax, ymax)

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")
    return True


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    src_images = project_root / "data" / "raw" / "rucv_original" / "images"
    src_xml = project_root / "data" / "raw" / "rucv_original" / "annotations"

    dst_images = project_root / "data" / "raw" / "rucv_yolo" / "train" / "images"
    dst_labels = project_root / "data" / "raw" / "rucv_yolo" / "train" / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png"}
    converted = 0
    copied = 0

    for image_path in src_images.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        xml_path = src_xml / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue

        new_image_name = f"rucv_{image_path.name}"
        new_label_name = f"rucv_{image_path.stem}.txt"

        shutil.copy2(image_path, dst_images / new_image_name)
        copied += 1

        success = convert_xml_file(xml_path, dst_labels / new_label_name)
        if success:
            converted += 1

    print(f"Copied images: {copied}")
    print(f"Converted labels: {converted}")


if __name__ == "__main__":
    main()