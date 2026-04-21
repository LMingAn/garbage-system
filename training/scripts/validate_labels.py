from __future__ import annotations

import os
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get('SOURCE_ROOT', '../raw_dataset')).resolve()
LABEL_DIR = SOURCE_ROOT / 'labels'
IMAGE_DIR = SOURCE_ROOT / 'images'
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_COUNT = int(os.environ.get('CLASS_COUNT', '9'))

bad_files = []
missing_labels = []
missing_images = []
valid_labels = 0
box_count = 0

if not LABEL_DIR.exists():
    raise SystemExit(f'标签目录不存在: {LABEL_DIR}')

for image in IMAGE_DIR.iterdir() if IMAGE_DIR.exists() else []:
    if image.suffix.lower() in VALID_IMAGE_EXTS:
        txt = LABEL_DIR / f'{image.stem}.txt'
        if not txt.exists():
            missing_labels.append(image.name)

for txt in LABEL_DIR.rglob('*.txt'):
    image_exists = any((IMAGE_DIR / f'{txt.stem}{ext}').exists() for ext in VALID_IMAGE_EXTS)
    if not image_exists:
        missing_images.append(txt.name)
    lines = txt.read_text(encoding='utf-8').splitlines()
    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            bad_files.append((txt.name, idx, '字段数不是5'))
            continue
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except Exception:
            bad_files.append((txt.name, idx, '无法解析为数字'))
            continue
        if not (0 <= cls < CLASS_COUNT):
            bad_files.append((txt.name, idx, f'类别越界: {cls}'))
            continue
        if not all(0 <= v <= 1 for v in coords):
            bad_files.append((txt.name, idx, f'坐标不在0~1范围: {coords}'))
            continue
        if coords[2] <= 0 or coords[3] <= 0:
            bad_files.append((txt.name, idx, '宽高必须大于0'))
            continue
        valid_labels += 1
        box_count += 1

print('=== 标签校验结果 ===')
print(f'标签目录: {LABEL_DIR}')
print(f'有效标注行数: {valid_labels}')
print(f'目标框总数: {box_count}')
print(f'缺少标签图片数: {len(missing_labels)}')
print(f'缺少图片标签数: {len(missing_images)}')
print(f'错误标注行数: {len(bad_files)}')

if missing_labels:
    print('\n缺少标签的图片（前20条）:')
    for item in missing_labels[:20]:
        print(' -', item)

if missing_images:
    print('\n缺少图片的标签（前20条）:')
    for item in missing_images[:20]:
        print(' -', item)

if bad_files:
    print('\n错误标注（前30条）:')
    for name, line_no, reason in bad_files[:30]:
        print(f' - {name} line {line_no}: {reason}')
