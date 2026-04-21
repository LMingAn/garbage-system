from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get('SOURCE_ROOT', '../raw_dataset')).resolve()
LABEL_DIR = SOURCE_ROOT / 'labels'
OUTPUT_PATH = Path(os.environ.get('STATS_OUTPUT', '../dataset_stats.json')).resolve()

CLASS_NAMES = [
    'battery', 'biological', 'clothes', 'glass', 'metal',
    'paper', 'plastic', 'shoes', 'trash'
]

image_to_boxes = Counter()
class_counter = Counter()

for txt in LABEL_DIR.rglob('*.txt'):
    count = 0
    for line in txt.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        if 0 <= cls < len(CLASS_NAMES):
            class_counter[CLASS_NAMES[cls]] += 1
            count += 1
    image_to_boxes[txt.stem] = count

stats = {
    'label_root': str(LABEL_DIR),
    'image_count_with_labels': len(image_to_boxes),
    'total_boxes': sum(class_counter.values()),
    'avg_boxes_per_image': round(sum(image_to_boxes.values()) / max(1, len(image_to_boxes)), 3),
    'class_distribution': dict(class_counter),
    'multi_object_images': sum(1 for v in image_to_boxes.values() if v >= 2),
    'single_object_images': sum(1 for v in image_to_boxes.values() if v == 1),
    'empty_label_files': sum(1 for v in image_to_boxes.values() if v == 0)
}

OUTPUT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
print('=== 数据集统计完成 ===')
print(json.dumps(stats, ensure_ascii=False, indent=2))
print(f'统计结果已写入: {OUTPUT_PATH}')
