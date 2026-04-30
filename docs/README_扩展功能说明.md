# 扩展功能说明

## 历史记录

接口：

- `GET /api/history`
- `DELETE /api/history/:id`
- `DELETE /api/history`

记录字段：

- 垃圾类别
- 归属分类
- 置信度
- 投放建议
- 识别方式
- 识别时间

图片上传会自动写入历史记录。摄像头识别需要点击“保存当前结果”后写入。

## 分类知识

接口：

```text
GET /api/knowledge
```

知识内容来自：

```text
backend/config/category_meta.json
backend/config/recycle_advice.json
```

前端支持按类别、示例和定义搜索。

## 回收导航

接口：

```text
GET /api/recycle-points?class_name=plastic&lat=...&lng=...
```

未配置高德 Key 时，接口返回高德和百度地图搜索链接。配置 `AMAP_WEB_KEY` 后，可返回附近点位列表。

PowerShell 配置示例：

```powershell
$env:AMAP_WEB_KEY="your_key"
node backend/app.js
```

## 上传文件清理

后端会定时清理 `backend/uploads` 中的过期文件。

可选环境变量：

- `UPLOAD_CLEANUP_MAX_AGE_HOURS`
- `UPLOAD_CLEANUP_INTERVAL_MINUTES`
- `UPLOAD_CLEANUP_MIN_FILE_AGE_MINUTES`

默认保留 24 小时内文件，并跳过 10 分钟内的新文件。

## 前端分页

前端按功能分为：

- 图片识别
- 实时识别
- 历史记录
- 分类知识
- 回收导航

页面不再通过长滚动承载所有功能，导航项切换当前功能页。
