"""将原 10 类中的 cardboard 合并到 paper。

旧类别顺序：
0 battery
1 biological
2 cardboard
3 clothes
4 glass
5 metal
6 paper
7 plastic
8 shoes
9 trash

新类别顺序：
0 battery
1 biological
2 clothes
3 glass
4 metal
5 paper
6 plastic
7 shoes
8 trash
"""
from __future__ import annotations

import os
from pathlib import Path

LABEL_ROOT = Path(os.environ.get('LABEL_ROOT', '../dataset/labels')).resolve()
MAP = {0: 0, 1: 1, 2: 5, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

if not LABEL_ROOT.exists():
    raise SystemExit(f'标签目录不存在: {LABEL_ROOT}')

changed_files = 0
changed_rows = 0

for txt in LABEL_ROOT.rglob('*.txt'):
    rows = []
    file_changed = False
    for line in txt.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_cls = int(parts[0])
        new_cls = MAP[old_cls]
        if new_cls != old_cls:
            file_changed = True
            changed_rows += 1
        rows.append(' '.join([str(new_cls)] + parts[1:]))
    txt.write_text('\n'.join(rows), encoding='utf-8')
    if file_changed:
        changed_files += 1

print(f'cardboard 已全部合并到 paper。修改文件数: {changed_files}，修改标签行数: {changed_rows}')
