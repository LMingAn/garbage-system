from __future__ import annotations

import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from services.model_manager import get_model, MODEL_PATH, resolve_device
from services.postprocess import SimpleFrameTracker

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'uploads'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONF = float(os.environ.get('YOLO_CONF', '0.35'))
DEFAULT_IOU = float(os.environ.get('YOLO_IOU', '0.45'))
DEFAULT_IMGSZ = int(os.environ.get('YOLO_IMGSZ', '640'))
FRAME_STRIDE = int(os.environ.get('VIDEO_FRAME_STRIDE', '2'))
SAVE_FRAME_VISUAL = os.environ.get('SAVE_FRAME_VISUAL', '0') == '1'

CLASS_NAMES = [
    'battery', 'biological', 'clothes', 'glass', 'metal',
    'paper', 'plastic', 'shoes', 'trash'
]

app = Flask(__name__)
FRAME_TRACKERS: Dict[str, SimpleFrameTracker] = {}


def _success(data: Dict[str, Any], msg: str = 'ok'):
    return jsonify({'code': 200, 'msg': msg, 'data': data})


def _parse_body() -> Dict[str, Any]:
    if request.is_json:
        return request.get_json(force=True, silent=True) or {}
    return request.form.to_dict()


def _load_image_from_form() -> Image.Image:
    if 'file' in request.files:
        return Image.open(request.files['file'].stream).convert('RGB')
    raise ValueError('未上传图像文件')


def _load_image_from_base64(base64_str: str) -> Image.Image:
    if not base64_str:
        raise ValueError('缺少 base64 图像数据')
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


def _parse_number(payload: Dict[str, Any], name: str, default: float) -> float:
    value = payload.get(name)
    if value in (None, ''):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_int(payload: Dict[str, Any], name: str, default: int) -> int:
    value = payload.get(name)
    if value in (None, ''):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _extract_detections(result) -> List[dict]:
    names = result.names if hasattr(result, 'names') else {idx: name for idx, name in enumerate(CLASS_NAMES)}
    boxes = result.boxes
    detections: List[dict] = []
    if boxes is None:
        return detections
    for xyxy, conf, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        label = names.get(int(cls), str(int(cls)))
        detections.append({
            'class_name': label,
            'confidence': round(float(conf), 4),
            'bbox': [round(float(v), 2) for v in xyxy]
        })
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections


def _render_image(np_image: np.ndarray, detections: List[dict]) -> np.ndarray:
    canvas = np_image.copy()
    palette = [
        (0, 255, 0), (0, 180, 255), (255, 180, 0), (255, 80, 80),
        (180, 0, 255), (255, 255, 0), (0, 255, 180), (255, 0, 180), (200, 200, 255)
    ]
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        track_id = det.get('track_id') or 0
        color = palette[(track_id or abs(hash(det['class_name']))) % len(palette)]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if track_id:
            label += f" ID:{track_id}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - 24)), (min(canvas.shape[1] - 1, x1 + max(90, len(label) * 10)), y1), color, -1)
        cv2.putText(canvas, label, (x1 + 4, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    return canvas


def _save_image(np_image: np.ndarray, prefix: str = 'image') -> str:
    out_name = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    return f"/uploads/{out_name}"


def _run_predict(image: Image.Image, conf: float, iou: float, imgsz: int):
    model = get_model()
    return model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        device=resolve_device()
    )[0]


def _predict_pil(image: Image.Image, conf: float, iou: float, imgsz: int, save_visual: bool = True) -> Tuple[List[dict], str | None, float]:
    begin = time.time()
    result = _run_predict(image, conf, iou, imgsz)
    detections = _extract_detections(result)
    saved_path = None
    if save_visual:
        visual = _render_image(np.array(image), detections)
        saved_path = _save_image(visual)
    elapsed_ms = round((time.time() - begin) * 1000, 2)
    return detections, saved_path, elapsed_ms




@app.get('/')
def home():
    return _success({
        'service': 'garbage detection python service',
        'status': 'running',
        'routes': [
            '/health',
            '/predict/image',
            '/predict/frame',
            '/predict/video',
            '/tracker/reset',
            '/uploads/<filename>'
        ]
    }, 'Python 识别服务运行正常')


@app.get('/favicon.ico')
def favicon():
    return '', 204


@app.get('/uploads/<path:filename>')
def serve_upload(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)

@app.get('/health')
def health():
    return _success({
        'model_path': str(MODEL_PATH),
        'default_conf': DEFAULT_CONF,
        'default_iou': DEFAULT_IOU,
        'default_imgsz': DEFAULT_IMGSZ,
        'device': str(resolve_device())
    })


@app.post('/tracker/reset')
def reset_tracker():
    payload = _parse_body()
    session_id = str(payload.get('session_id') or 'default')
    tracker = FRAME_TRACKERS.get(session_id)
    if tracker:
        tracker.reset()
    return _success({'session_id': session_id}, 'tracker 已重置')


@app.post('/predict/image')
def predict_image():
    try:
        payload = _parse_body()
        image = _load_image_from_form()
        conf = _parse_number(payload, 'conf', DEFAULT_CONF)
        iou = _parse_number(payload, 'iou', DEFAULT_IOU)
        imgsz = _parse_int(payload, 'imgsz', DEFAULT_IMGSZ)
        save_visual = str(payload.get('save_visual', '1')) != '0'
        detections, saved_path, elapsed_ms = _predict_pil(image, conf, iou, imgsz, save_visual=save_visual)
        return _success({
            'detections': detections,
            'source_size': [image.width, image.height],
            'saved_path': saved_path,
            'stable': False,
            'stable_count': 0,
            'elapsed_ms': elapsed_ms
        }, '图片识别成功')
    except Exception as exc:
        return jsonify({'code': 500, 'msg': str(exc)}), 500


@app.post('/predict/frame')
def predict_frame():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        session_id = str(payload.get('session_id') or 'default')
        image = _load_image_from_base64(str(payload.get('base64', '')))
        conf = _parse_number(payload, 'conf', DEFAULT_CONF)
        iou = _parse_number(payload, 'iou', DEFAULT_IOU)
        imgsz = _parse_int(payload, 'imgsz', DEFAULT_IMGSZ)
        detections, saved_path, elapsed_ms = _predict_pil(image, conf, iou, imgsz, save_visual=SAVE_FRAME_VISUAL)
        tracker = FRAME_TRACKERS.setdefault(session_id, SimpleFrameTracker())
        detections, stable, stable_count = tracker.update(detections)
        return _success({
            'detections': detections,
            'source_size': [image.width, image.height],
            'saved_path': saved_path,
            'stable': stable,
            'stable_count': stable_count,
            'elapsed_ms': elapsed_ms
        }, '视频帧识别成功')
    except Exception as exc:
        return jsonify({'code': 500, 'msg': str(exc)}), 500


@app.post('/predict/video')
def predict_video():
    try:
        if 'file' not in request.files:
            raise ValueError('未上传视频文件')
        file = request.files['file']
        temp_video_name = f"upload_{uuid.uuid4().hex[:10]}{Path(file.filename).suffix or '.mp4'}"
        temp_video_path = OUTPUT_DIR / temp_video_name
        file.save(temp_video_path)

        payload = _parse_body()
        conf = _parse_number(payload, 'conf', DEFAULT_CONF)
        iou = _parse_number(payload, 'iou', DEFAULT_IOU)
        imgsz = _parse_int(payload, 'imgsz', DEFAULT_IMGSZ)
        frame_stride = max(1, _parse_int(payload, 'frame_stride', FRAME_STRIDE))

        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            raise RuntimeError('视频文件无法打开')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        output_name = f"result_{uuid.uuid4().hex[:10]}.mp4"
        output_path = OUTPUT_DIR / output_name
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        model = get_model()
        tracker = SimpleFrameTracker()
        frame_index = 0
        latest_detections: List[dict] = []
        class_summary: Dict[str, int] = {}
        started = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_index % frame_stride == 0:
                result = model.predict(source=rgb, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device=resolve_device())[0]
                latest_detections = _extract_detections(result)
                latest_detections, _, _ = tracker.update(latest_detections)
                for det in latest_detections:
                    class_summary[det['class_name']] = class_summary.get(det['class_name'], 0) + 1
            visual_rgb = _render_image(rgb, latest_detections)
            writer.write(cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR))
            frame_index += 1

        cap.release()
        writer.release()

        elapsed = max(time.time() - started, 0.001)
        summary_rows = [
            {'class_name': k, 'confidence': 1.0, 'bbox': [], 'track_id': None, 'hit_count': v}
            for k, v in sorted(class_summary.items(), key=lambda item: item[1], reverse=True)
        ]
        return _success({
            'detections': summary_rows,
            'fps': round(float(fps), 2),
            'source_size': [width, height],
            'saved_path': f'/uploads/{output_name}',
            'stable': False,
            'stable_count': 0,
            'frame_count': frame_index,
            'total_frames': total_frames,
            'elapsed_ms': round(elapsed * 1000, 2),
            'avg_process_fps': round(frame_index / elapsed, 2)
        }, '视频分析完成')
    except Exception as exc:
        return jsonify({'code': 500, 'msg': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PYTHON_PORT', '5000')), debug=False)
