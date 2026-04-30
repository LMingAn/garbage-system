from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "training" / "runs"
BACKEND_MODEL_DIR = ROOT / "backend" / "model"


def copy_best_weight(src: Path, dst_name: str) -> Path:
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


def train_stage(
    *,
    model_path: str | Path,
    data: str,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int | float,
    device: str,
    workers: int,
    patience: int,
    freeze: int,
    cache: bool | str,
    lr0: float,
    lrf: float,
    close_mosaic: int,
    mosaic: float,
    scale: float,
    hsv_s: float,
    hsv_v: float,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        freeze=freeze,
        amp=True,
        cache=cache,
        close_mosaic=close_mosaic,
        hsv_h=0.015,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=3.0,
        translate=0.08,
        scale=scale,
        shear=1.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=mosaic,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.25,
        box=7.5,
        cls=0.65,
        dfl=1.5,
        nbs=64,
        val=True,
        plots=True,
        seed=0,
        deterministic=False,
        project=str(RUNS_ROOT),
        name=name,
        exist_ok=True,
    )

    best = RUNS_ROOT / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Best weight was not found: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a higher-accuracy seed model tuned for RTX 3060 Laptop GPU. "
            "Defaults favor annotation-review quality over the fastest runtime model."
        )
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "data.yaml"),
    )
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Use -1 for Ultralytics auto-batch. Use an integer if auto-batch is unstable.",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--warmup-stage-epochs", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Windows multiprocessing can run out of RAM with many workers. "
            "Use 2 by default; try 4 only if system memory is enough."
        ),
    )
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--name", default="seed_model_3060_laptop")
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="disk",
        help=(
            "Use disk by default on Windows to avoid worker-spawn MemoryError. "
            "Use ram only when system memory is enough."
        ),
    )
    parser.add_argument(
        "--copy-name",
        default="seed_best_3060_laptop.pt",
        help="Backend model filename for the final best weight.",
    )
    parser.add_argument(
        "--also-deploy",
        action="store_true",
        help="Also overwrite backend/model/seed_best.pt after training.",
    )
    args = parser.parse_args()

    enable_cuda_performance()

    batch = normalize_batch(args.batch)
    cache = normalize_cache(args.cache)
    print(
        "Training config: "
        f"model={args.model}, imgsz={args.imgsz}, batch={batch}, "
        f"workers={args.workers}, cache={args.cache}, device={args.device}"
    )

    warmup_name = f"{args.name}_warmup"
    warmup_best = train_stage(
        model_path=args.model,
        data=args.data,
        name=warmup_name,
        epochs=args.warmup_stage_epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=max(10, args.warmup_stage_epochs // 2),
        freeze=args.freeze,
        cache=cache,
        lr0=0.001,
        lrf=0.01,
        close_mosaic=5,
        mosaic=0.7,
        scale=0.35,
        hsv_s=0.55,
        hsv_v=0.3,
    )

    final_best = train_stage(
        model_path=warmup_best,
        data=args.data,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        freeze=0,
        cache=cache,
        lr0=0.0006,
        lrf=0.005,
        close_mosaic=15,
        mosaic=0.45,
        scale=0.25,
        hsv_s=0.45,
        hsv_v=0.25,
    )

    deployed = copy_best_weight(final_best, args.copy_name)
    print(f"3060 Laptop seed model training finished. Copied best weight to: {deployed}")

    if args.also_deploy:
        default_deployed = copy_best_weight(final_best, "seed_best.pt")
        print(f"Also deployed to default seed model path: {default_deployed}")


if __name__ == "__main__":
    main()
