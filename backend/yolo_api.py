from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os

app = Flask(__name__)

# 模型只加载一次
model = YOLO("model/best.pt", verbose=False)

# 加载回收建议
advice_path = "config/recycle_advice.json"
if os.path.exists(advice_path):
    with open(advice_path, "r", encoding="utf-8") as f:
        recycle_advice = json.load(f)
else:
    recycle_advice = {}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = None

        # 上传文件模式
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # base64 模式（摄像头实时识别）
        elif request.json and 'base64' in request.json:
            import base64
            b64_data = request.json['base64'].split(",")[-1]
            img_data = base64.b64decode(b64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        else:
            return jsonify({"code": 400, "msg": "缺少图片参数"})

        # 模型推理
        results = model(img, verbose=False)
        img_h, img_w = img.shape[:2]
        pred_results = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                img_area = float(img_w * img_h)
                if conf < 0.25 or (conf < 0.6 and img_area > 0 and box_area / img_area > 0.95):
                    continue
                pred_results.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "advice": recycle_advice.get(cls_name, "暂无回收建议")
                })

        if pred_results:
            best_result = max(pred_results, key=lambda x: x["confidence"])
            best_result = dict(best_result)
            best_result["predictions"] = pred_results
            return jsonify({"code": 200, "data": best_result, "msg": "识别成功"})
        else:
            return jsonify({"code": 400, "data": None, "msg": "未识别到目标"})

    except Exception as e:
        return jsonify({"code": 500, "data": None, "msg": f"识别失败：{str(e)}"})


if __name__ == "__main__":
    app.run(port=5000)
