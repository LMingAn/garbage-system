from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_pairs(images_root: Path, labels_root: Path):
    pairs = []
    for img in images_root.rglob('*'):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(images_root)
        label = (labels_root / rel).with_suffix('.txt')
        if label.exists():
            pairs.append((img, label, rel.name))
    return pairs


def copy_pairs(items, target: Path, split: str):
    (target / 'images' / split).mkdir(parents=True, exist_ok=True)
    (target / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for img, label, name in items:
        shutil.copy2(img, target / 'images' / split / name)
        shutil.copy2(label, target / 'labels' / split / Path(name).with_suffix('.txt'))


def main():
    parser = argparse.ArgumentParser(description='合并精标集与人工校正集，重新划分最终训练集。')
    parser.add_argument('--seed-images', required=True)
    parser.add_argument('--seed-labels', required=True)
    parser.add_argument('--review-images', required=True)
    parser.add_argument('--review-labels', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_pairs = collect_pairs(Path(args.seed_images), Path(args.seed_labels))
    review_pairs = collect_pairs(Path(args.review_images), Path(args.review_labels))
    all_pairs = seed_pairs + review_pairs
    if not all_pairs:
        raise FileNotFoundError('未找到可合并的数据。')

    random.Random(args.seed).shuffle(all_pairs)
    n = len(all_pairs)
    train_n = int(n * args.train_ratio)
    val_n = int(n * args.val_ratio)
    splits = {
        'train': all_pairs[:train_n],
        'val': all_pairs[train_n:train_n + val_n],
        'test': all_pairs[train_n + val_n:]
    }
    target = Path(args.target)
    if target.exists():
        shutil.rmtree(target)
    for split, items in splits.items():
        copy_pairs(items, target, split)
    print({k: len(v) for k, v in splits.items()})
    print('最终训练集已生成。')


if __name__ == '__main__':
    main()
