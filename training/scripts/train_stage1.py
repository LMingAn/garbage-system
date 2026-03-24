from pathlib import Path
from ultralytics import YOLO
import yaml

from common import CONFIG_ROOT


def main():
    cfg_path = CONFIG_ROOT / "stage1_head_warmup.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_path = cfg.pop("model")
    model = YOLO(model_path)
    model.train(**cfg)


if __name__ == "__main__":
    main()
