from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "training" / "runs"
BACKEND_MODEL_DIR = ROOT / "backend" / "model"


def copy_best_weight(src: Path, dst_name: str = "best.pt") -> Path:
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKEND_MODEL_DIR / dst_name
    shutil.copy2(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description='使用人工精标核心集训练种子模型。')
    parser.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'data.yaml'))
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--freeze', type=int, default=10)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=20,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        freeze=args.freeze,
        amp=True,
        cache=False,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.25,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        project=str(RUNS_ROOT),
        name='seed_model',
        exist_ok=True,
    )

    best = RUNS_ROOT / 'seed_model' / 'weights' / 'best.pt'
    if best.exists():
        deployed = copy_best_weight(best, 'seed_best.pt')
        print('种子模型训练完成，已复制到:', deployed)
    else:
        raise FileNotFoundError(f'未找到种子模型权重: {best}')


if __name__ == '__main__':
    main()
