from __future__ import annotations

"""
安全版数据划分脚本。

作用：
1. 从已经具备真实 YOLO 标签的数据集中划分 train/val/test；
2. 绝不再生成“整图框”伪标注；
3. 适合作为人工精标核心集或人工校正后的数据集整理脚本。
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_pairs(source_images: Path, source_labels: Path):
    pairs = []
    for img in source_images.rglob('*'):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(source_images)
        label = (source_labels / rel).with_suffix('.txt')
        if not label.exists():
            continue
        pairs.append((img, label, rel.name))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='划分已具备真实标签的数据集，不生成伪标注。')
    parser.add_argument('--source-images', required=True)
    parser.add_argument('--source-labels', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    src_images = Path(args.source_images)
    src_labels = Path(args.source_labels)
    target = Path(args.target)
    pairs = collect_pairs(src_images, src_labels)
    if not pairs:
        raise FileNotFoundError('未找到带真实标签的图片/标签对，脚本已停止。')

    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    train_n = int(n * args.train_ratio)
    val_n = int(n * args.val_ratio)
    splits = {
        'train': pairs[:train_n],
        'val': pairs[train_n:train_n + val_n],
        'test': pairs[train_n + val_n:]
    }

    for split, items in splits.items():
        (target / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for img, label, name in items:
            shutil.copy2(img, target / 'images' / split / name)
            shutil.copy2(label, target / 'labels' / split / Path(name).with_suffix('.txt'))

    print({k: len(v) for k, v in splits.items()})
    print('已完成安全划分：仅复制真实标签，不生成任何整图伪标注。')


if __name__ == '__main__':
    main()
