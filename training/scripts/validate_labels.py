from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='校验 YOLO 标签的类号与坐标范围。')
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--num-classes', type=int, default=9)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    bad = []
    for txt in labels_dir.rglob('*.txt'):
        for idx, line in enumerate(txt.read_text(encoding='utf-8').splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                bad.append((str(txt), idx, '字段数错误'))
                continue
            try:
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
            except Exception:
                bad.append((str(txt), idx, '类型错误'))
                continue
            if not (0 <= cls < args.num_classes):
                bad.append((str(txt), idx, f'类号越界: {cls}'))
            x, y, w, h = vals
            if not all(0.0 <= v <= 1.0 for v in vals):
                bad.append((str(txt), idx, '坐标超出 0~1'))
            if w <= 0 or h <= 0:
                bad.append((str(txt), idx, '宽高必须大于 0'))
    if bad:
        print('发现标签问题:')
        for row in bad[:200]:
            print(row)
        raise SystemExit(1)
    print('标签校验通过。')


if __name__ == '__main__':
    main()
