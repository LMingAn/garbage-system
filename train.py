from ultralytics import YOLO

# 加载YOLOv8n预训练模型
model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    results = model.train(
        data="dataset/data.yaml",  # 数据集配置文件路径
        epochs=60,                 # 训练轮次
        imgsz=640,                  # 输入图片尺寸
        batch=8,                    # 批次大小
        device="0",                 # 0为显卡模式
        workers=2,                  # 数据加载线程数
        patience=10,                # 早停机制，10轮无提升自动停止，避免过拟合
        save=True,                  # 自动保存模型
        project="runs/train",      # 模型保存路径
        name="garbage_model",       # 训练项目名称
        exist_ok=True               # 覆盖已有结果
    )

    # 验证模型
    val_results = model.val()

    # 导出模型（方便后端调用，导出为pt格式）
    print("模型训练完成！最佳模型路径：runs/train/garbage_model/weights/best.pt")