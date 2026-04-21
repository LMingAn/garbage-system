from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get('YOLO_MODEL_PATH', ROOT / 'model' / 'best.pt'))


def resolve_device() -> str | int:
    target = str(os.environ.get('YOLO_DEVICE', 'auto')).strip().lower()
    if target in {'cpu', 'mps'}:
        return target
    if target in {'0', '1', '2', '3'}:
        return int(target)
    return 0


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'未找到模型权重: {MODEL_PATH}')
    return YOLO(str(MODEL_PATH))
