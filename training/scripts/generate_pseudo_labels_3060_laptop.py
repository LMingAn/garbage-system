from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "backend" / "model" / "seed_best_3060_laptop.pt"
DEFAULT_SOURCE = ROOT / "dataset" / "unlabeled_pool"
DEFAULT_OUTPUT = ROOT / "dataset" / "pseudo_workspace"
LOCAL_ULTRALYTICS_DIR = ROOT / "training" / ".ultralytics"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_yolo_txt(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def enable_cuda_performance() -> None:
    LOCAL_ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_DIR))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception as exc:
        print(f"CUDA performance knobs were skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo labels with batched RTX 3060 Laptop GPU inference. "
            "Output structure matches generate_pseudo_labels.py."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Unlabeled image directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output workspace directory.")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    enable_cuda_performance()

    source = Path(args.source)
    output = Path(args.output)
    images_out = output / "images"
    labels_out = output / "labels"
    vis_out = output / "preview"
    vis_out.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(source))
    if not image_paths:
        print({"images": 0, "boxes": 0, "output": str(output)})
        return

    print(
        "Pseudo-label config: "
        f"weights={args.weights}, source={source}, output={output}, "
        f"imgsz={args.imgsz}, batch={args.batch}, conf={args.conf}, "
        f"iou={args.iou}, half={args.half}, device={args.device}"
    )

    from ultralytics import YOLO

    model = YOLO(args.weights)
    total_boxes = 0

    for batch_paths in batched(image_paths, max(args.batch, 1)):
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=len(batch_paths),
            workers=args.workers,
            half=args.half,
            max_det=args.max_det,
            verbose=False,
        )

        for result in results:
            img_path = Path(result.path)
            rows: list[str] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xywhn = result.boxes.xywhn.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xywhn)):
                    cls_id = int(clses[i])
                    x, y, w, h = xywhn[i].tolist()
                    rows.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    total_boxes += 1

            rel = img_path.relative_to(source)
            label_path = (labels_out / rel).with_suffix(".txt")
            save_yolo_txt(label_path, rows)

            if args.copy_images:
                dst_img = images_out / rel
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_img)

            plotted = result.plot()
            vis_path = vis_out / rel.name
            PILImage.fromarray(plotted[..., ::-1]).save(vis_path)

    print({"images": len(image_paths), "boxes": total_boxes, "output": str(output)})
    print("Please review output/labels manually before using them for final training.")


if __name__ == "__main__":
    main()
