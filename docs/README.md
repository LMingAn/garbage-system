# garbage-system-optimized

基于 **Python + YOLOv8 + Node.js + 原生前端** 的智能垃圾分类与回收处理系统优化版。

本版以 **GTX 1650 可落地训练与部署** 为约束，按你的最终毕设目标重构：

- `cardboard` 已并入 `paper`
- 类别调整为 **9 类**
- 任务从“演示型单主体识别”升级为 **多目标检测 + 视频稳定识别 + 回收处理闭环**
- 停止使用整图伪标注，改为 **真实框数据集流程**
- 增加 **数据校验、标签合并、数据集统计、两阶段训练、视频平滑跟踪、回收点检索**

---

## 1. 当前类别（9类）

1. battery
2. biological
3. clothes
4. glass
5. metal
6. paper
7. plastic
8. shoes
9. trash

其中：`cardboard -> paper`

---

## 2. 推荐运行环境

### 训练端（GTX 1650）
- Python 3.10 / 3.11
- PyTorch + CUDA（建议按本机环境单独安装）
- ultralytics
- 4GB 显存场景优先使用 `YOLOv8n`

### 服务端
- Node.js 18+
- Python Flask 服务

---

## 3. 项目结构

```text
backend/                 Node.js 网关后端
backend/python_service/  Python 推理服务
frontend/                图片/视频/摄像头/知识库/历史记录前端
training/                数据集准备、校验、统计、训练脚本
docs/                    重构说明、项目顺序、毕设写作参考
```

---

## 4. 启动顺序

### 第一步：启动 Python 识别服务

```bash
cd backend/python_service
pip install -r requirements.txt
python app.py
```

### 第二步：启动 Node.js 后端

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

### 第三步：访问系统

```text
http://127.0.0.1:3000
```

---

## 5. 训练顺序（推荐）

### 5.1 如原始标签仍有 cardboard 类，先合并标签

```bash
cd training/scripts
python merge_cardboard_into_paper.py
```

### 5.2 校验 YOLO 标签是否规范

```bash
python validate_labels.py
```

### 5.3 生成数据集统计信息

```bash
python dataset_stats.py
```

### 5.4 整理标准 YOLO 数据集目录

```bash
python prepare_yolo_dataset.py
```

### 5.5 先做 GTX 1650 轻量训练

```bash
python train_pipeline.py --profile gtx1650_n
```

### 5.6 若轻量训练稳定，再尝试更高精度版本

```bash
python train_pipeline.py --profile gtx1650_s
```

---

## 6. 训练建议（GTX 1650）

### 推荐主方案
- 模型：`YOLOv8n`
- 图片尺寸：`640`
- batch：`4~8`
- 两阶段训练：开启
- 先图片识别跑通，再上视频识别

### 进阶方案
- 模型：`YOLOv8s`
- batch：`2~4`
- 用于图片精度提升版、论文对比版

---

## 7. 当前系统功能

### 图片识别
- 上传图片
- 多目标垃圾检测
- 输出检测框、类别、置信度、垃圾大类、回收建议

### 视频识别
- 上传本地视频
- 间隔帧检测
- 简单 IoU 时序跟踪
- 框平滑与类别投票
- 输出分析后视频

### 摄像头识别
- 浏览器端摄像头采集
- 连续帧检测
- 跟踪 ID、稳定计数
- 前端实时框显示

### 回收处理闭环
- 分类知识库
- 历史记录
- 回收建议
- 回收点检索（配置高德 Web Key 时可用）

---

## 8. 注意事项

1. **请将你训练得到的 `best.pt` 放入：**

```text
backend/model/best.pt
```

2. 默认不附带最终大模型权重与完整数据集。
3. 视频稳定部分采用 **本科毕设可解释实现**：
   - 间隔帧检测
   - IoU 匹配
   - bbox 平滑
   - 类别投票
   - 连续帧稳定确认
4. 当前方案优先保证：
   - 可训练
   - 可部署
   - 可答辩讲清楚

---

## 9. 优先阅读文件

- `docs/完整项目构建顺序.md`
- `docs/项目重构说明.md`
- `docs/毕设论文可直接使用表述.md`
- `backend/app.js`
- `backend/python_service/app.py`
- `training/scripts/train_pipeline.py`
- `training/scripts/validate_labels.py`
- `training/scripts/dataset_stats.py`

