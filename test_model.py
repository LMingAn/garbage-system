from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("backend/model/best.pt")

# 测试图片路径（替换为你的测试图片）
test_img_path = "test.jpg"

# 推理
results = model(test_img_path)

# 解析结果
for r in results:
    boxes = r.boxes  # 检测框
    for box in boxes:
        cls_id = int(box.cls[0])  # 类别ID
        cls_name = model.names[cls_id]  # 类别名称
        conf = float(box.conf[0])  # 置信度
        print(f"识别结果：{cls_name}，置信度：{conf:.2f}")

# 可视化结果
annotated_img = results[0].plot()
cv2.imwrite("result.jpg", annotated_img)
cv2.imshow("识别结果", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()