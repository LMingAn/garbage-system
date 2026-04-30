from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

CLASS_NAMES = ['battery', 'biological', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']


def main():
    parser = argparse.ArgumentParser(description='统计数据集框数量与类别分布。')
    parser.add_argument('--labels-dir', required=True)
    args = parser.parse_args()

    counter = Counter()
    file_count = 0
    box_count = 0
    for txt in Path(args.labels_dir).rglob('*.txt'):
        file_count += 1
        for line in txt.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            counter[cls] += 1
            box_count += 1

    print({'files': file_count, 'boxes': box_count})
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f'{cls_id:02d} {name}: {counter.get(cls_id, 0)}')


if __name__ == '__main__':
    main()
