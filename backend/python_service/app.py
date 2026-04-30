import base64
import io
import os
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

CLASS_NAMES = [
    "battery",
    "biological",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

CATEGORY_META = {
    "battery": {
        "class_name_zh": "电池",
        "category_group": "有害垃圾",
        "recycle_advice": "请投放至有害垃圾回收点，避免随意丢弃造成污染。",
        "map_keyword": "电池回收点"
    },
    "biological": {
        "class_name_zh": "厨余垃圾",
        "category_group": "厨余垃圾",
        "recycle_advice": "沥干水分后投放至厨余垃圾容器。",
        "map_keyword": "厨余垃圾投放点"
    },
    "clothes": {
        "class_name_zh": "衣物",
        "category_group": "可回收物",
        "recycle_advice": "较完整衣物可回收或捐赠，破损严重衣物按当地规则处理。",
        "map_keyword": "旧衣物回收箱"
    },
    "glass": {
        "class_name_zh": "玻璃",
        "category_group": "可回收物",
        "recycle_advice": "清空内容物后投放至可回收物容器，破碎玻璃请包裹后投放。",
        "map_keyword": "玻璃回收点"
    },
    "metal": {
        "class_name_zh": "金属",
        "category_group": "可回收物",
        "recycle_advice": "清理残留后投放至可回收物容器。",
        "map_keyword": "金属回收点"
    },
    "paper": {
        "class_name_zh": "纸类",
        "category_group": "可回收物",
        "recycle_advice": "保持干燥后投放至可回收物容器，纸箱、纸板、快递盒均归入纸类。",
        "map_keyword": "废纸回收点"
    },
    "plastic": {
        "class_name_zh": "塑料",
        "category_group": "可回收物",
        "recycle_advice": "清洗后压扁投放至可回收物容器。",
        "map_keyword": "塑料回收点"
    },
    "shoes": {
        "class_name_zh": "鞋类",
        "category_group": "可回收物",
        "recycle_advice": "可回收或按旧衣物、旧物回收渠道处理。",
        "map_keyword": "旧鞋回收点"
    },
    "trash": {
        "class_name_zh": "其他垃圾",
        "category_group": "其他垃圾",
        "recycle_advice": "无法明确回收利用的废弃物投放至其他垃圾容器。",
        "map_keyword": "其他垃圾投放点"
    }
}

def get_class_name(class_id: int, model=None) -> str:
    """
    优先使用模型自带 names；如果模型 names 不完整或异常，则使用项目固定 9 类。
    """
    try:
        if model is not None and hasattr(model, "names"):
            names = model.names

            if isinstance(names, dict):
                name = names.get(class_id)
                if name:
                    return str(name)

            if isinstance(names, list) and 0 <= class_id < len(names):
                return str(names[class_id])
    except Exception:
        pass

    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]

    return "unknown"


def build_detection_item(class_id, confidence, bbox, model=None, track_id=None):
    class_id = int(class_id)
    class_name = get_class_name(class_id, model)
    meta = CATEGORY_META.get(class_name, {})

    x1, y1, x2, y2 = bbox

    item = {
        "class_id": class_id,
        "class_name": class_name,
        "class_name_zh": meta.get("class_name_zh", class_name),
        "category_group": meta.get("category_group", "未分类"),
        "confidence": round(float(confidence), 4),
        "confidence_text": f"{float(confidence) * 100:.1f}%",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        "recycle_advice": meta.get("recycle_advice", "请按当地垃圾分类规则投放。"),
        "map_keyword": meta.get("map_keyword", "")
    }

    if track_id is not None:
        item["track_id"] = int(track_id)

    return item

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get('YOLO_MODEL_PATH', ROOT / 'model' / 'best.pt'))
CONF_THRES = float(os.environ.get('YOLO_CONF', '0.42'))
IOU_THRES = float(os.environ.get('YOLO_IOU', '0.40'))
DEFAULT_IMGSZ = int(os.environ.get('YOLO_IMGSZ', '640'))
UPLOAD_IMGSZ = int(os.environ.get('YOLO_UPLOAD_IMGSZ', '768'))
UPLOAD_TTA = os.environ.get('YOLO_UPLOAD_TTA', '1').lower() in ('1', 'true', 'yes', 'on')
MAX_TRACK_IDLE_SECONDS = float(os.environ.get('TRACK_IDLE_SECONDS', '3.0'))
CAMERA_FIXED_ROI = os.environ.get('CAMERA_FIXED_ROI', '0.28,0.18,0.44,0.62')
CAMERA_STABILITY_FRAMES = int(os.environ.get('CAMERA_STABILITY_FRAMES', '3'))
CAMERA_MIN_AREA_RATIO = float(os.environ.get('CAMERA_MIN_AREA_RATIO', '0.015'))
CAMERA_MAX_AREA_RATIO = float(os.environ.get('CAMERA_MAX_AREA_RATIO', '0.40'))
CAMERA_FOCUS_INNER_RATIO = float(os.environ.get('CAMERA_FOCUS_INNER_RATIO', '0.56'))
UPLOAD_FIXED_ROI = os.environ.get('UPLOAD_FIXED_ROI', '')

app = Flask(__name__)
MODEL = None
TRACK_STATE = {}


def load_model():
    global MODEL
    if MODEL is None:
        if YOLO is None:
            raise RuntimeError('未安装 ultralytics，请先执行 pip install ultralytics flask pillow')
        if not MODEL_PATH.exists():
            raise RuntimeError(f'未找到模型权重: {MODEL_PATH}')
        MODEL = YOLO(str(MODEL_PATH))
        print("模型类别 names:", MODEL.names)
    return MODEL


def clamp(value, low, high):
    return max(low, min(high, value))


def read_image_from_request():
    if 'image' in request.files:
        file = request.files['image']
        return Image.open(file.stream).convert('RGB')

    if 'file' in request.files:
        file = request.files['file']
        return Image.open(file.stream).convert('RGB')

    data = request.get_json(silent=True) or {}
    base64_str = data.get('base64', '')
    if not base64_str:
        raise ValueError('未提供图像数据')
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


def read_options():
    data = request.get_json(silent=True) or {}
    form = request.form or {}
    mode = str(data.get('mode') or form.get('mode') or 'upload')

    options = {
        'mode': mode,
        'conf': CONF_THRES,
        'iou': IOU_THRES,
        'imgsz': UPLOAD_IMGSZ if mode != 'camera' else DEFAULT_IMGSZ,
        'max_det': 12,
        'roi': parse_roi(UPLOAD_FIXED_ROI),
        'session_id': str(data.get('session_id') or form.get('session_id') or ''),
        'stability_frames': 1,
        'min_area_ratio': 0.008,
        'max_area_ratio': 0.75,
        'center_bias': True,
        'focus_inner_ratio': None,
        'augment': False,
    }

    if mode == 'camera':
        options.update({
            'conf': CONF_THRES,
            'iou': IOU_THRES,
            'imgsz': DEFAULT_IMGSZ,
            'max_det': 8,
            'roi': parse_roi(CAMERA_FIXED_ROI),
            'stability_frames': CAMERA_STABILITY_FRAMES,
            'min_area_ratio': CAMERA_MIN_AREA_RATIO,
            'max_area_ratio': CAMERA_MAX_AREA_RATIO,
            'focus_inner_ratio': clamp(CAMERA_FOCUS_INNER_RATIO, 0.2, 0.9),
            'augment': False,
        })

    options['conf'] = clamp(options['conf'], 0.05, 0.95)
    options['iou'] = clamp(options['iou'], 0.1, 0.95)
    options['imgsz'] = int(clamp(options['imgsz'], 320, 1280))
    if mode != 'camera':
        options['augment'] = UPLOAD_TTA
    options['max_det'] = int(clamp(options['max_det'], 1, 100))
    options['stability_frames'] = int(clamp(options['stability_frames'], 1, 6))
    options['min_area_ratio'] = clamp(options['min_area_ratio'], 0.0005, 0.3)
    options['max_area_ratio'] = clamp(options['max_area_ratio'], 0.1, 0.98)
    if not options['session_id']:
        options['session_id'] = str(uuid.uuid4())
    return options


def parse_roi(raw_roi):
    if not raw_roi:
        return None
    if isinstance(raw_roi, str):
        parts = [x.strip() for x in raw_roi.split(',')]
        if len(parts) != 4:
            return None
        try:
            raw_roi = [float(x) for x in parts]
        except ValueError:
            return None
    if isinstance(raw_roi, dict):
        raw_roi = [raw_roi.get('x', 0), raw_roi.get('y', 0), raw_roi.get('w', 1), raw_roi.get('h', 1)]
    if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in raw_roi]
    except Exception:
        return None
    x = clamp(x, 0.0, 0.95)
    y = clamp(y, 0.0, 0.95)
    w = clamp(w, 0.05, 1.0 - x)
    h = clamp(h, 0.05, 1.0 - y)
    return [x, y, w, h]


def crop_by_roi(image: Image.Image, roi):
    if not roi:
        return image, None
    width, height = image.size
    x, y, w, h = roi
    left = int(round(x * width))
    top = int(round(y * height))
    right = int(round((x + w) * width))
    bottom = int(round((y + h) * height))
    left = clamp(left, 0, width - 1)
    top = clamp(top, 0, height - 1)
    right = clamp(right, left + 1, width)
    bottom = clamp(bottom, top + 1, height)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def point_in_box(x, y, box):
    if not box:
        return True
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def edge_margin_ratio(item, target_box):
    if not target_box:
        return 0.0
    x1, y1, x2, y2 = item['bbox']
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    tw = max(1.0, target_box[2] - target_box[0])
    th = max(1.0, target_box[3] - target_box[1])
    dx = abs(cx - (target_box[0] + target_box[2]) / 2.0) / (tw / 2.0)
    dy = abs(cy - (target_box[1] + target_box[3]) / 2.0) / (th / 2.0)
    dist = min(1.4, (dx * dx + dy * dy) ** 0.5)
    return dist


def box_iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def score_candidate(item, image_size, center_bias=True, target_box=None):
    width, height = image_size
    x1, y1, x2, y2 = item['bbox']
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = width * height
    area_ratio = box_area / frame_area if frame_area else 0.0
    score = item['confidence']
    if center_bias:
        target = target_box or [0, 0, width, height]
        tcx = (target[0] + target[2]) / 2.0
        tcy = (target[1] + target[3]) / 2.0
        tw = max(1.0, target[2] - target[0])
        th = max(1.0, target[3] - target[1])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(cx - tcx) / (tw / 2.0)
        dy = abs(cy - tcy) / (th / 2.0)
        center_score = 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5)
        score += center_score * (0.16 if target_box else 0.08)
    if 0.02 <= area_ratio <= 0.28:
        score += 0.05
    return score


def filter_predictions(predictions, image_size, options, allowed_zone=None, target_box=None):
    width, height = image_size
    frame_area = width * height
    kept = []
    for item in predictions:
        x1, y1, x2, y2 = item['bbox']
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_ratio = box_area / frame_area if frame_area else 0.0
        if area_ratio < options['min_area_ratio']:
            continue
        if area_ratio > options['max_area_ratio']:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if options['mode'] == 'camera' and allowed_zone and not point_in_box(cx, cy, allowed_zone):
            continue
        item['area_ratio'] = round(area_ratio, 6)
        item['center_distance'] = round(edge_margin_ratio(item, target_box or allowed_zone), 6) if (target_box or allowed_zone) else 0.0
        item['score'] = round(score_candidate(item, image_size, options['center_bias'], target_box or allowed_zone), 6)
        kept.append(item)
    kept.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
    return kept


def apply_track_stability(predictions, options):
    now = time.time()
    expired = [k for k, v in TRACK_STATE.items() if now - v.get('ts', 0) > MAX_TRACK_IDLE_SECONDS]
    for k in expired:
        TRACK_STATE.pop(k, None)

    if options['mode'] != 'camera':
        return predictions

    session_id = options['session_id']
    previous = TRACK_STATE.get(session_id)
    if not predictions:
        if previous:
            previous['miss'] = previous.get('miss', 0) + 1
            previous['ts'] = now
        return predictions

    best = predictions[0]
    stable_count = 1
    if previous and previous.get('class_name') == best['class_name'] and box_iou(previous.get('bbox', [0,0,0,0]), best['bbox']) >= 0.3:
        stable_count = previous.get('stable_count', 1) + 1
        alpha = 0.6
        pb = previous['bbox']
        nb = best['bbox']
        best['bbox'] = [round(pb[i] * (1 - alpha) + nb[i] * alpha, 2) for i in range(4)]
        best['confidence'] = round(best['confidence'] * 0.7 + previous.get('confidence', best['confidence']) * 0.3, 6)
    TRACK_STATE[session_id] = {
        'class_name': best['class_name'],
        'bbox': best['bbox'],
        'confidence': best['confidence'],
        'stable_count': stable_count,
        'ts': now,
        'miss': 0,
    }
    best['stable_count'] = stable_count
    if stable_count < options['stability_frames'] and previous:
        prev_proxy = {
            'class_name': previous['class_name'],
            'confidence': previous['confidence'],
            'bbox': previous['bbox'],
            'stable_count': stable_count,
            'score': previous.get('confidence', 0),
            'suppressed': True,
        }
        return [prev_proxy] + predictions
    return predictions


def predict_image(image: Image.Image, options):
    model = load_model()
    original_size = image.size
    roi_image, roi_box = crop_by_roi(image, options['roi'])
    results = model.predict(
        source=roi_image,
        conf=options['conf'],
        iou=options['iou'],
        imgsz=options['imgsz'],
        max_det=options['max_det'],
        verbose=False,
        augment=options.get('augment', False),
    )
    result = results[0]

    predictions = []
    best_item = None
    best_score = -1.0
    print("模型类别 names:", model.names)
    print("YOLO 原始检测框数量:", 0 if result.boxes is None else len(result.boxes))

    offset_x = roi_box[0] if roi_box else 0
    offset_y = roi_box[1] if roi_box else 0

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()

        xyxy = [
            xyxy[0] + offset_x,
            xyxy[1] + offset_y,
            xyxy[2] + offset_x,
            xyxy[3] + offset_y
        ]

        item = build_detection_item(
            class_id=cls_id,
            confidence=conf,
            bbox=xyxy,
            model=model
        )

        print("原始检测:", cls_id, conf, xyxy)

        predictions.append(item)


    allowed_zone = roi_box if options['mode'] == 'camera' else None
    target_box = roi_box if options['mode'] == 'camera' else None
    predictions = filter_predictions(
        predictions,
        original_size,
        options,
        allowed_zone=allowed_zone,
        target_box=target_box
    )
    predictions = apply_track_stability(predictions, options)
    predictions = filter_predictions(predictions, original_size, options, allowed_zone=allowed_zone,
                                     target_box=target_box)
    print("过滤后检测框数量:", len(predictions))
    print("过滤后结果:", predictions)

    for item in predictions:
        score = item.get('score', item['confidence'])
        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None:
        return {
            'class_name': 'unknown',
            'class_name_zh': '未识别',
            'category_group': '未分类',
            'confidence': 0,
            'confidence_text': '0.0%',
            'bbox': None,
            'predictions': [],
            'detections': [],
            'summary': {
                'detected_count': 0,
                'top_class_name': 'unknown',
                'top_class_name_zh': '未识别',
                'top_confidence': 0,
                'top_confidence_text': '0.0%',
                'category_group': '未分类',
                'recycle_advice': '未检测到可识别目标。'
            },
            'roi': options['roi'],
            'roi_box': roi_box,
            'focus_zone': None,
            'source_size': {'width': original_size[0], 'height': original_size[1]},
            'stable': False,
            'stable_count': 0,
            'mode': options['mode']
        }

    stable_count = int(best_item.get('stable_count', 1))

    return {
        'class_name': best_item['class_name'],
        'class_name_zh': best_item.get('class_name_zh', best_item['class_name']),
        'category_group': best_item.get('category_group', '未分类'),
        'confidence': best_item['confidence'],
        'confidence_text': best_item.get(
            'confidence_text',
            f"{best_item['confidence'] * 100:.1f}%"
        ),
        'bbox': best_item['bbox'],

        # 多目标结果：前端应优先使用 detections / predictions
        'predictions': predictions,
        'detections': predictions,

        'summary': {
            'detected_count': len(predictions),
            'top_class_name': best_item['class_name'],
            'top_class_name_zh': best_item.get('class_name_zh', best_item['class_name']),
            'top_confidence': best_item['confidence'],
            'top_confidence_text': best_item.get(
                'confidence_text',
                f"{best_item['confidence'] * 100:.1f}%"
            ),
            'category_group': best_item.get('category_group', '未分类'),
            'recycle_advice': best_item.get(
                'recycle_advice',
                '请按当地垃圾分类规则投放。'
            )
        },

        'recycle_advice': best_item.get('recycle_advice', '请按当地垃圾分类规则投放。'),
        'map_keyword': best_item.get('map_keyword', ''),

        'roi': options['roi'],
        'roi_box': roi_box,
        'focus_zone': None,
        'source_size': {'width': original_size[0], 'height': original_size[1]},
        'stable': stable_count >= options['stability_frames'],
        'stable_count': stable_count,
        'mode': options['mode'],
        'params': {
            'conf': options['conf'],
            'iou': options['iou'],
            'imgsz': options['imgsz'],
            'stability_frames': options['stability_frames'],
            'min_area_ratio': options['min_area_ratio'],
            'max_area_ratio': options['max_area_ratio'],
            'focus_inner_ratio': options.get('focus_inner_ratio'),
            'augment': options.get('augment', False),
        }
    }


@app.route('/health', methods=['GET'])
def health():
    ok = YOLO is not None and MODEL_PATH.exists()
    return jsonify({'code': 200, 'data': {'ready': ok, 'model_path': str(MODEL_PATH)}})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = read_image_from_request()
        options = read_options()
        data = predict_image(image, options)
        return jsonify({'code': 200, 'msg': 'success', 'data': data})
    except Exception as exc:
        return jsonify({'code': 500, 'msg': str(exc)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
