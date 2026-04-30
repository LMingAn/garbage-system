from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def save_yolo_txt(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(rows) + '\n') if rows else '', encoding='utf-8')


def iter_images(root: Path):
    for p in root.rglob('*'):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(description='使用种子模型为未标注图片生成候选伪标注。')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True, help='未标注图片目录')
    parser.add_argument('--output', required=True, help='输出 workspace 目录')
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args()

    model = YOLO(args.weights)
    source = Path(args.source)
    output = Path(args.output)
    images_out = output / 'images'
    labels_out = output / 'labels'
    vis_out = output / 'preview'
    vis_out.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0
    for img_path in iter_images(source):
        total_images += 1
        result = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        rows = []
        if result.boxes is not None and len(result.boxes) > 0:
            xywhn = result.boxes.xywhn.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xywhn)):
                cls_id = int(clses[i])
                x, y, w, h = xywhn[i].tolist()
                rows.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1
        rel = img_path.relative_to(source)
        label_path = (labels_out / rel).with_suffix('.txt')
        save_yolo_txt(label_path, rows)
        if args.copy_images:
            dst_img = images_out / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)
        plotted = result.plot()
        from PIL import Image as PILImage
        vis_path = vis_out / rel.name
        PILImage.fromarray(plotted[..., ::-1]).save(vis_path)

    print({'images': total_images, 'boxes': total_boxes, 'output': str(output)})
    print('请将 output/labels 导入标注工具进行人工校正，再进入最终训练。')


if __name__ == '__main__':
    main()
