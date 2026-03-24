import os
import random
import shutil
from PIL import Image

# 原始数据集路径（替换为解压后的路径）
original_data_path = "D:/Download/garbage_classification-Simplify"
# 目标路径
target_path = "D:/Download/garbage-system-training-optimized/garbage_proj/dataset"

# 类别映射
class_mapping = {
    "battery": 0, "biological": 1, "cardboard": 2, "clothes": 3, "glass": 4,
    "metal": 5, "paper": 6, "plastic": 7, "shoes": 8, "trash": 9
}

# 创建文件夹
os.makedirs(f"{target_path}/images/train", exist_ok=True)
os.makedirs(f"{target_path}/images/val", exist_ok=True)
os.makedirs(f"{target_path}/labels/train", exist_ok=True)
os.makedirs(f"{target_path}/labels/val", exist_ok=True)

# 遍历所有类别文件夹
for class_name, class_id in class_mapping.items():
    class_folder = os.path.join(original_data_path, class_name)
    if not os.path.exists(class_folder):
        print(f"跳过不存在的类别：{class_name}")
        continue

    # 获取该类别下所有图片
    image_files = [f for f in os.listdir(class_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    # 随机划分训练/验证集（8:2）
    random.shuffle(image_files)
    train_size = int(len(image_files) * 0.8)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # 处理训练集
    for img_file in train_files:
        img_path = os.path.join(class_folder, img_file)
        # 复制图片到train目录
        shutil.copy(img_path, os.path.join(target_path, "images/train", img_file))
        # 生成YOLO格式标签（假设每个图片只有一个垃圾目标，占满画面）
        img = Image.open(img_path)
        w, h = img.size
        # YOLO格式：class_id x_center y_center width height（归一化到0-1）
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0
        # 保存标签文件
        label_file = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(target_path, "labels/train", label_file), "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}")

    # 处理验证集（逻辑同训练集）
    for img_file in val_files:
        img_path = os.path.join(class_folder, img_file)
        shutil.copy(img_path, os.path.join(target_path, "images/val", img_file))
        img = Image.open(img_path)
        w, h = img.size
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0
        label_file = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(target_path, "labels/val", label_file), "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}")

# 生成data.yaml文件
yaml_content = f"""
# 数据集配置
path: {os.path.abspath(target_path)}  # 数据集根路径
train: images/train  # 训练图片路径
val: images/val      # 验证图片路径

# 类别
nc: {len(class_mapping)}  # 类别数
names: {list(class_mapping.keys())}  # 类别名称
"""
with open(os.path.join(target_path, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("数据集整理完成！")