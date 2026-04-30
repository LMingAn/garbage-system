from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    if image.mode != "RGB":
        return image.convert("RGB")

    return image


def clean_image(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image.load()
        rgb = to_rgb(image)
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(
            dst,
            format="JPEG",
            quality=95,
            optimize=True,
            progressive=False,
            subsampling=0,
        )


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and normalize an image dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("garbage classification"),
        help="Dataset root containing class subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("garbage classification_cleaned"),
        help="Destination root for cleaned images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)

    temp_root = output_root.with_name(f"{output_root.name}.__tmp__")
    if temp_root.exists():
        shutil.rmtree(temp_root)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    manifest_rows: list[list[str]] = [["class", "new_name", "original_path"]]
    failed_rows: list[list[str]] = [["class", "original_path", "reason"]]
    class_counts: list[tuple[str, int, int]] = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = collect_images(class_dir)
        ok_count = 0
        fail_count = 0

        for src in images:
            new_name = f"{class_name}_{ok_count + 1:04d}.jpg"
            dst = temp_root / class_name / new_name

            try:
                clean_image(src, dst)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                fail_count += 1
                failed_rows.append([class_name, str(src), str(exc)])
                continue

            ok_count += 1
            manifest_rows.append([class_name, new_name, str(src)])

        class_counts.append((class_name, ok_count, fail_count))

    temp_root.mkdir(parents=True, exist_ok=True)
    with (temp_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(manifest_rows)

    with (temp_root / "failed.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(failed_rows)

    temp_root.rename(output_root)

    total_ok = sum(ok for _, ok, _ in class_counts)
    total_failed = sum(failed for _, _, failed in class_counts)
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Cleaned images: {total_ok}")
    print(f"Failed images: {total_failed}")
    for class_name, ok_count, fail_count in class_counts:
        print(f"{class_name}: {ok_count} cleaned, {fail_count} failed")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
