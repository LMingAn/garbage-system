from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "training" / "runs"
BACKEND_MODEL_DIR = ROOT / "backend" / "model"
LOCAL_ULTRALYTICS_DIR = ROOT / "training" / ".ultralytics"
DEFAULT_SEED_WEIGHT = BACKEND_MODEL_DIR / "seed_best_3060_laptop.pt"


def copy_best_weight(src: Path, dst_name: str = "best.pt") -> Path:
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKEND_MODEL_DIR / dst_name
    shutil.copy2(src, dst)
    return dst


def normalize_batch(batch: int | float) -> int | float:
    if isinstance(batch, int):
        return batch
    if batch.is_integer():
        return int(batch)
    return batch


def normalize_cache(cache: str) -> bool | str:
    if cache == "false":
        return False
    return cache


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


def resolve_model(model_arg: str) -> str:
    if model_arg != "auto":
        return model_arg
    if DEFAULT_SEED_WEIGHT.exists():
        return str(DEFAULT_SEED_WEIGHT)
    return "yolov8s.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the final garbage detection model tuned for RTX 3060 Laptop GPU. "
            "Defaults favor final model quality while keeping Windows memory use stable."
        )
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "data.yaml"),
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Use auto to start from backend/model/seed_best_3060_laptop.pt when it exists; "
            "otherwise fall back to yolov8s.pt."
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Use -1 for Ultralytics auto-batch. Use an integer if auto-batch is unstable.",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Windows multiprocessing can run out of RAM with many workers. "
            "Use 2 by default; try 4 only if system memory is enough."
        ),
    )
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="disk",
        help=(
            "Use disk by default on Windows to avoid worker-spawn MemoryError. "
            "Use ram only when system memory is enough."
        ),
    )
    parser.add_argument("--name", default="final_model")
    parser.add_argument("--copy-name", default="best.pt")
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.00045)
    parser.add_argument("--lrf", type=float, default=0.005)
    parser.add_argument("--close-mosaic", type=int, default=25)
    args = parser.parse_args()

    enable_cuda_performance()

    from ultralytics import YOLO

    model_path = resolve_model(args.model)
    batch = normalize_batch(args.batch)
    cache = normalize_cache(args.cache)
    print(
        "Training config: "
        f"model={model_path}, data={args.data}, imgsz={args.imgsz}, batch={batch}, "
        f"epochs={args.epochs}, patience={args.patience}, workers={args.workers}, "
        f"cache={args.cache}, device={args.device}"
    )

    model = YOLO(model_path)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=4.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        freeze=args.freeze,
        amp=True,
        cache=cache,
        close_mosaic=args.close_mosaic,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.08,
        scale=0.3,
        shear=1.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.45,
        mixup=0.03,
        copy_paste=0.0,
        erasing=0.2,
        box=7.5,
        cls=0.6,
        dfl=1.5,
        nbs=64,
        val=True,
        plots=True,
        seed=0,
        deterministic=False,
        project=str(RUNS_ROOT),
        name=args.name,
        exist_ok=True,
    )

    best = RUNS_ROOT / args.name / "weights" / "best.pt"
    if best.exists():
        deployed = copy_best_weight(best, args.copy_name)
        print(f"Final model training finished. Copied best weight to: {deployed}")
    else:
        raise FileNotFoundError(f"Best final model weight was not found: {best}")


if __name__ == "__main__":
    main()
