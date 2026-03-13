import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get('YOLO_MODEL_PATH', ROOT / 'model' / 'best.pt'))
CONF_THRES = float(os.environ.get('YOLO_CONF', '0.35'))

app = Flask(__name__)
MODEL = None


def load_model():
    global MODEL
    if MODEL is None:
        if YOLO is None:
            raise RuntimeError('未安装 ultralytics，请先执行 pip install ultralytics flask pillow')
        if not MODEL_PATH.exists():
            raise RuntimeError(f'未找到模型权重: {MODEL_PATH}')
        MODEL = YOLO(str(MODEL_PATH))
    return MODEL


def read_image_from_request():
    if 'image' in request.files:
        file = request.files['image']
        return Image.open(file.stream).convert('RGB')

    data = request.get_json(silent=True) or {}
    base64_str = data.get('base64', '')
    if not base64_str:
        raise ValueError('未提供图像数据')
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


def predict_image(image: Image.Image):
    model = load_model()
    results = model.predict(source=image, conf=CONF_THRES, verbose=False)
    result = results[0]

    predictions = []
    best_item = None
    best_conf = -1.0

    names = result.names if hasattr(result, 'names') else {}
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        item = {
            'class_name': names.get(cls_id, str(cls_id)),
            'confidence': conf,
            'bbox': [round(v, 2) for v in xyxy]
        }
        predictions.append(item)
        if conf > best_conf:
            best_conf = conf
            best_item = item

    if best_item is None:
        return {
            'class_name': 'unknown',
            'confidence': 0,
            'predictions': []
        }

    return {
        'class_name': best_item['class_name'],
        'confidence': best_item['confidence'],
        'bbox': best_item['bbox'],
        'predictions': predictions
    }


@app.route('/health', methods=['GET'])
def health():
    ok = YOLO is not None and MODEL_PATH.exists()
    return jsonify({'code': 200, 'data': {'ready': ok, 'model_path': str(MODEL_PATH)}})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = read_image_from_request()
        data = predict_image(image)
        return jsonify({'code': 200, 'msg': 'success', 'data': data})
    except Exception as exc:
        return jsonify({'code': 500, 'msg': str(exc)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
