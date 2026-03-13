import sys
import json
import cv2
import base64
import numpy as np
import os
from ultralytics import YOLO

os.environ["YOLO_VERBOSE"] = "False"

# 加载模型（路径对应后端model文件夹）
model = YOLO("model/best.pt", verbose=False)


# 接收参数：图片路径或 base64 编码（摄像头识别用）
def predict(image_input, is_base64=False):
    try:
        if is_base64:
            # 处理base64编码的图片（摄像头识别）
            img_data = base64.b64decode(image_input.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # 处理本地图片（上传识别）
            img = cv2.imread(image_input)

        # 模型推理
        results = model(img, verbose=False)
        img_h, img_w = img.shape[:2]
        # 解析结果
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
                    "bbox": [x1, y1, x2, y2]  # 检测框坐标
                })

        # 读取回收建议
        with open("config/recycle_advice.json", "r", encoding="utf-8") as f:
            advice = json.load(f)

        # 返回结果（取置信度最高的结果）
        if pred_results:
            best_result = max(pred_results, key=lambda x: x["confidence"])
            best_result = dict(best_result)
            best_result["predictions"] = pred_results
            best_result["advice"] = advice.get(best_result["class_name"], "暂无回收建议")
            output = {
                "code": 200,
                "data": best_result,
                "msg": "识别成功"
            }
        else:
            output = {
                "code": 400,
                "data": None,
                "msg": "未识别到垃圾类别"
            }
    except Exception as e:
        output = {
            "code": 500,
            "data": None,
            "msg": f"识别失败：{str(e)}"
        }

    sys.stdout.write(json.dumps(output, ensure_ascii=False))
    sys.stdout.flush()


# 接收后端传参
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"code": 400, "msg": "缺少图片参数"}, ensure_ascii=False))
        sys.exit(0)

    image_input = sys.argv[1]
    is_base64 = sys.argv[2] == "True" if len(sys.argv) > 2 else False

    # 直接调用 predict，不要再打印返回值
    predict(image_input, is_base64)

    os._exit(0)
