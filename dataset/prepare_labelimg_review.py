from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Classes file does not exist: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise SystemExit(f"Classes file is empty: {path}")
    return classes


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def paired_label_path(labels_dir: Path, images_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def safe_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_classes(output_dir: Path, class_text: str, overwrite: bool) -> int:
    count = 0
    class_dirs = [p for p in output_dir.iterdir() if p.is_dir()] if output_dir.exists() else []
    for class_dir in class_dirs:
        classes_path = class_dir / "classes.txt"
        if classes_path.exists() and not overwrite:
            continue
        classes_path.write_text(class_text, encoding="utf-8")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare pseudo_workspace image/txt pairs for direct YOLO review in LabelImg."
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent / "pseudo_workspace")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--classes", type=Path, default=Path(__file__).resolve().parent / "seed_dataset" / "classes.txt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the review directory. Do not use after manual review unless intended.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Create empty txt files for images without a source label file.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    images_dir = (args.images_dir or workspace / "images").resolve()
    labels_dir = (args.labels_dir or workspace / "labels").resolve()
    output_dir = (args.output_dir or workspace / "labelimg_review").resolve()
    classes = read_classes(args.classes.resolve())
    class_text = "\n".join(classes) + "\n"

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory does not exist: {labels_dir}")

    copied_images = 0
    copied_labels = 0
    skipped_existing = 0
    missing_labels = 0
    empty_created = 0

    for image_path in iter_images(images_dir):
        rel = image_path.relative_to(images_dir)
        dst_image = output_dir / rel
        dst_label = dst_image.with_suffix(".txt")
        label_path = paired_label_path(labels_dir, images_dir, image_path)

        if safe_copy(image_path, dst_image, args.overwrite):
            copied_images += 1
        else:
            skipped_existing += 1

        if label_path.is_file():
            if safe_copy(label_path, dst_label, args.overwrite):
                copied_labels += 1
            else:
                skipped_existing += 1
        else:
            missing_labels += 1
            if args.include_empty_labels and (args.overwrite or not dst_label.exists()):
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("", encoding="utf-8")
                empty_created += 1

    classes_written = write_classes(output_dir, class_text, args.overwrite)

    print(
        {
            "copied_images": copied_images,
            "copied_labels": copied_labels,
            "classes_written": classes_written,
            "missing_labels": missing_labels,
            "empty_labels_created": empty_created,
            "skipped_existing": skipped_existing,
            "output": str(output_dir),
        }
    )
    return 0 if missing_labels == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
