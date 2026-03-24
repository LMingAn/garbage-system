from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"
CONFIG_ROOT = TRAINING_ROOT / "configs"
RUNS_ROOT = TRAINING_ROOT / "runs"
BACKEND_MODEL_DIR = ROOT / "backend" / "model"


def copy_best_weight(src: Path, dst_name: str = "best.pt") -> Path:
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKEND_MODEL_DIR / dst_name
    shutil.copy2(src, dst)
    return dst
