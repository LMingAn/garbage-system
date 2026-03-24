from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO

from common import RUNS_ROOT, copy_best_weight


def stage1(data: str, device: str, batch: int):
    model = YOLO("yolov8s.pt")
    model.train(
        data=data,
        epochs=40,
        imgsz=640,
        batch=batch,
        device=device,
        workers=8,
        patience=30,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        multi_scale=0.25,
        freeze=10,
        cache=False,
        amp=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        project=str(RUNS_ROOT),
        name="garbage_stage1_head_warmup",
        exist_ok=True,
    )
    return RUNS_ROOT / "garbage_stage1_head_warmup" / "weights" / "best.pt"


def stage2(data: str, device: str, batch: int, weight: Path):
    model = YOLO(str(weight))
    model.train(
        data=data,
        epochs=120,
        imgsz=640,
        batch=batch,
        device=device,
        workers=8,
        patience=40,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        cos_lr=True,
        multi_scale=0.25,
        freeze=0,
        cache=False,
        amp=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.25,
        translate=0.05,
        scale=0.25,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        project=str(RUNS_ROOT),
        name="garbage_stage2_full_finetune",
        exist_ok=True,
    )
    return RUNS_ROOT / "garbage_stage2_full_finetune" / "weights" / "best.pt"


def validate(weight: Path, data: str):
    model = YOLO(str(weight))
    normal = model.val(data=data, imgsz=640, batch=8, augment=False, verbose=False)
    tta = model.val(data=data, imgsz=640, batch=8, augment=True, verbose=False)
    normal_map50 = float(getattr(normal.box, "map50", 0.0))
    tta_map50 = float(getattr(tta.box, "map50", 0.0))
    print({"normal_map50": normal_map50, "tta_map50": tta_map50})
    deployed = copy_best_weight(weight)
    print("部署权重已更新到:", deployed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "configs" / "data.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    s1 = stage1(args.data, args.device, args.batch)
    if not s1.exists():
        raise FileNotFoundError(f"Stage1 权重不存在: {s1}")
    s2 = stage2(args.data, args.device, args.batch, s1)
    if not s2.exists():
        raise FileNotFoundError(f"Stage2 权重不存在: {s2}")
    validate(s2, args.data)


if __name__ == "__main__":
    main()
