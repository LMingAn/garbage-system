"""基于真实框标注整理 YOLO 数据集。

输入目录要求：
source_root/
  images/
  labels/
其中 labels 中每张图片都应已有真实框 YOLO txt 标签。

本脚本只做：
1. 校验图片-标签对应关系
2. 划分 train/val/test
3. 复制到标准 dataset 目录

注意：本脚本不再生成整图伪标注。
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get('SOURCE_ROOT', '../raw_dataset')).resolve()
TARGET_ROOT = Path(os.environ.get('TARGET_ROOT', '../dataset')).resolve()
TRAIN_RATIO = float(os.environ.get('TRAIN_RATIO', '0.8'))
VAL_RATIO = float(os.environ.get('VAL_RATIO', '0.1'))
SEED = int(os.environ.get('SEED', '42'))
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

random.seed(SEED)

for sub in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']:
    (TARGET_ROOT / sub).mkdir(parents=True, exist_ok=True)

image_dir = SOURCE_ROOT / 'images'
label_dir = SOURCE_ROOT / 'labels'
if not image_dir.exists() or not label_dir.exists():
    raise SystemExit(f'源目录格式错误，请确保存在: {image_dir} 与 {label_dir}')

image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS]
random.shuffle(image_files)

valid_pairs = []
for img_path in image_files:
    txt_path = label_dir / f'{img_path.stem}.txt'
    if not txt_path.exists():
        print(f'跳过无标签图片: {img_path.name}')
        continue
    valid_pairs.append((img_path, txt_path))

n = len(valid_pairs)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

splits = {
    'train': valid_pairs[:train_end],
    'val': valid_pairs[train_end:val_end],
    'test': valid_pairs[val_end:]
}

for split, pairs in splits.items():
    for img_path, txt_path in pairs:
        shutil.copy2(img_path, TARGET_ROOT / 'images' / split / img_path.name)
        shutil.copy2(txt_path, TARGET_ROOT / 'labels' / split / txt_path.name)

print(f'完成数据集整理: {TARGET_ROOT}')
print({k: len(v) for k, v in splits.items()})
