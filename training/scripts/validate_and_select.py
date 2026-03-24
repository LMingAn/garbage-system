from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import yaml

from common import CONFIG_ROOT, RUNS_ROOT, copy_best_weight


def metric_value(metrics, name: str) -> float:
    return float(getattr(metrics.box, name, 0.0))


def validate_model(weight_path: Path, data_yaml: Path, imgsz: int = 640, batch: int = 8, augment: bool = False):
    model = YOLO(str(weight_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, augment=augment, verbose=False)
    return {
        "weight": str(weight_path),
        "augment": augment,
        "map50": metric_value(metrics, "map50"),
        "map": metric_value(metrics, "map"),
        "mp": metric_value(metrics, "mp"),
        "mr": metric_value(metrics, "mr"),
    }


def main():
    data_yaml = CONFIG_ROOT / "data.yaml"
    candidates = [
        RUNS_ROOT / "garbage_stage1_head_warmup" / "weights" / "best.pt",
        RUNS_ROOT / "garbage_stage2_full_finetune" / "weights" / "best.pt",
    ]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError("未找到待验证权重，请先运行 Stage1 / Stage2 训练。")

    reports = []
    for weight in candidates:
        reports.append(validate_model(weight, data_yaml, imgsz=640, batch=8, augment=False))
        reports.append(validate_model(weight, data_yaml, imgsz=640, batch=8, augment=True))

    reports.sort(key=lambda x: (x["map50"], x["map"]), reverse=True)
    best = reports[0]
    deployed = copy_best_weight(Path(best["weight"]))

    print("\n验证结果（按 map50 排序）:")
    for row in reports:
        print(row)

    print("\n已复制最优权重到部署目录:", deployed)
    print("最佳方案:", best)


if __name__ == "__main__":
    main()
