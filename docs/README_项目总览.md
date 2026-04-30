# 项目总览

本项目用于垃圾图片和摄像头画面的目标检测识别。系统由三部分组成：

- `frontend/`：前端页面，采用 Element Plus 组件风格，按功能分页展示。
- `backend/`：Node.js API 服务，负责静态页面、上传、历史记录和地图查询。
- `backend/python_service/`：Python YOLO 推理服务，负责图片和摄像头帧识别。
- `training/`：数据处理、训练配置和模型训练脚本。
- `dataset/`：训练数据、标注数据和划分后的数据集。

## 识别类别

当前模型按 9 类垃圾目标检测：

- `battery`
- `biological`
- `clothes`
- `glass`
- `metal`
- `paper`
- `plastic`
- `shoes`
- `trash`

## 启动服务

Python 推理服务：

```bash
cd backend/python_service
pip install -r requirements.txt
python app.py
```

Node 服务：

```bash
cd backend
npm install
node app.js
```

浏览器访问：

```text
http://127.0.0.1:3000/
```

## AnyLabeling 环境

项目根目录已创建专用虚拟环境：

```bash
anylabeling_env\Scripts\activate
anylabeling
```

该环境只用于标注工具，避免和推理、训练环境依赖混用。

## 训练路线

推荐流程：

1. 整理原始分类图片到 `dataset/raw_classified/`。
2. 清洗低质量图片。
3. 抽取核心样本并进行人工框标注，形成 `dataset/seed_dataset/`。
4. 划分 `dataset/seed_dataset_split/`。
5. 训练 `YOLOv8n` 种子模型。
6. 使用种子模型生成候选伪标注。
7. 人工校正候选标注。
8. 合并生成 `dataset/final_dataset/`。
9. 训练正式检测模型。

## 文档索引

- `docs/数据清洗划分与训练详细步骤.md`：数据集整理、标注、划分和训练流程。
- `docs/伪标注优化训练方案.md`：半自动标注方案。
- `docs/README_实时识别优化说明.md`：摄像头实时识别策略。
- `docs/README_扩展功能说明.md`：历史记录、知识库和回收导航说明。
- `docs/项目清理说明.md`：项目清理记录。
