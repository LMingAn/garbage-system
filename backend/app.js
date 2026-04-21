const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://127.0.0.1:5000';
const ROOT = __dirname;
const FRONTEND_DIR = path.resolve(ROOT, '../frontend');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const DATA_DIR = path.join(ROOT, 'data');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');
const CATEGORY_META_FILE = path.join(ROOT, 'config', 'category_meta.json');
const ADVICE_FILE = path.join(ROOT, 'config', 'recycle_advice.json');
const AMAP_WEB_KEY = process.env.AMAP_WEB_KEY || '';
const CLEANUP_MAX_AGE_HOURS = Number(process.env.UPLOAD_CLEANUP_MAX_AGE_HOURS || 24);
const CLEANUP_INTERVAL_MINUTES = Number(process.env.UPLOAD_CLEANUP_INTERVAL_MINUTES || 60);

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(HISTORY_FILE)) fs.writeFileSync(HISTORY_FILE, '[]', 'utf-8');

const categoryMeta = JSON.parse(fs.readFileSync(CATEGORY_META_FILE, 'utf-8'));
const recycleAdvice = JSON.parse(fs.readFileSync(ADVICE_FILE, 'utf-8'));

app.use(cors());
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/static', express.static(FRONTEND_DIR));
app.use(express.static(FRONTEND_DIR));

const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname || '.dat');
    cb(null, `${Date.now()}-${Math.random().toString(36).slice(2, 10)}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 200 * 1024 * 1024 }
});

function readHistory() {
  try {
    return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
  } catch (_) {
    return [];
  }
}

function writeHistory(rows) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(rows, null, 2), 'utf-8');
}

function getCategoryInfo(slug) {
  const meta = categoryMeta[slug] || {};
  return {
    slug,
    name: meta.name || slug,
    group: meta.group || '未分类',
    definition: meta.definition || '',
    examples: meta.examples || [],
    hazard: meta.hazard || '',
    value: meta.value || '',
    keyword: meta.mapKeyword || meta.name || slug,
    advice: recycleAdvice[slug] || '请按当地生活垃圾分类规范投放。'
  };
}

function normalizeDetection(item) {
  const slug = item.class_name || item.slug || 'unknown';
  const info = getCategoryInfo(slug);
  const confidence = Number(item.confidence || 0);
  return {
    class_name: slug,
    class_name_zh: info.name,
    category_group: info.group,
    confidence,
    confidence_text: `${(confidence * 100).toFixed(2)}%`,
    bbox: item.bbox || item.box || [],
    track_id: item.track_id ?? null,
    stable_count: Number(item.stable_count || 0),
    hit_count: Number(item.hit_count || 0),
    advice: info.advice,
    map_keyword: info.keyword
  };
}

function normalizePayload(raw, mode) {
  const data = raw?.data || raw || {};
  const detections = Array.isArray(data.detections) ? data.detections.map(normalizeDetection) : [];
  const primary = detections[0] || null;
  return {
    mode,
    source_size: data.source_size || null,
    roi_box: data.roi_box || null,
    stable: Boolean(data.stable),
    stable_count: Number(data.stable_count || 0),
    fps: data.fps || null,
    avg_process_fps: data.avg_process_fps || null,
    frame_count: data.frame_count || null,
    total_frames: data.total_frames || null,
    elapsed_ms: data.elapsed_ms || null,
    saved_path: data.saved_path || null,
    detections,
    summary: primary ? {
      class_name: primary.class_name,
      class_name_zh: primary.class_name_zh,
      confidence: primary.confidence,
      confidence_text: primary.confidence_text,
      category_group: primary.category_group,
      advice: primary.advice,
      stable_count: primary.stable_count,
      map_keyword: primary.map_keyword
    } : null
  };
}

function appendHistory(entry) {
  const rows = readHistory();
  rows.unshift(entry);
  writeHistory(rows.slice(0, 1000));
  return entry;
}

function createHistoryRecords(mode, result, extra = {}) {
  const now = new Date().toISOString();
  if (!Array.isArray(result.detections) || result.detections.length === 0) return [];
  return result.detections.map((item) => appendHistory({
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    identify_time: now,
    detect_mode: mode,
    class_name: item.class_name,
    display_name: item.class_name_zh,
    category_group: item.category_group,
    confidence: item.confidence,
    confidence_text: item.confidence_text,
    advice: item.advice,
    track_id: item.track_id,
    stable_count: item.stable_count,
    hit_count: item.hit_count,
    ...extra
  }));
}

function cleanupUploads() {
  const now = Date.now();
  const maxAgeMs = CLEANUP_MAX_AGE_HOURS * 60 * 60 * 1000;
  for (const name of fs.readdirSync(UPLOAD_DIR)) {
    const filePath = path.join(UPLOAD_DIR, name);
    try {
      const stat = fs.statSync(filePath);
      if (stat.isFile() && now - stat.mtimeMs > maxAgeMs) fs.unlinkSync(filePath);
    } catch (error) {
      console.warn('清理上传文件失败:', error.message);
    }
  }
}

async function postToPython(endpoint, { filePath, fields = {}, timeout = 120000, json = false }) {
  if (json) {
    const response = await axios.post(`${PYTHON_API_BASE}${endpoint}`, fields, { timeout });
    return response.data;
  }
  const form = new FormData();
  if (filePath) form.append('file', fs.createReadStream(filePath));
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null || value === '') continue;
    form.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
  }
  const response = await axios.post(`${PYTHON_API_BASE}${endpoint}`, form, {
    headers: form.getHeaders(),
    timeout
  });
  return response.data;
}


app.get('/favicon.ico', (_, res) => res.status(204).end());

app.get('/api/health', async (_, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_BASE}/health`, { timeout: 8000 });
    res.json({ code: 200, msg: 'ok', data: { node: 'ready', python: response.data?.data || response.data, amapConfigured: Boolean(AMAP_WEB_KEY) } });
  } catch (error) {
    res.status(500).json({ code: 500, msg: `Python 服务不可用: ${error.message}` });
  }
});

app.get('/api/categories', (_, res) => {
  const rows = Object.keys(categoryMeta).map(getCategoryInfo);
  res.json({ code: 200, data: rows });
});

app.get('/api/knowledge', (req, res) => {
  const q = String(req.query.q || '').trim().toLowerCase();
  const rows = Object.keys(categoryMeta)
    .map(getCategoryInfo)
    .filter((item) => !q || [item.slug, item.name, item.group, item.definition, item.examples.join(' ')].join(' ').toLowerCase().includes(q));
  res.json({ code: 200, data: rows });
});

app.get('/api/history', (req, res) => {
  const keyword = String(req.query.keyword || '').trim().toLowerCase();
  const mode = String(req.query.mode || '').trim();
  let rows = readHistory();
  if (keyword) rows = rows.filter((item) => JSON.stringify(item).toLowerCase().includes(keyword));
  if (mode) rows = rows.filter((item) => item.detect_mode === mode);
  res.json({ code: 200, data: rows });
});

app.delete('/api/history/:id', (req, res) => {
  const rows = readHistory().filter((item) => item.id !== req.params.id);
  writeHistory(rows);
  res.json({ code: 200, msg: '删除成功' });
});

app.delete('/api/history', (_, res) => {
  writeHistory([]);
  res.json({ code: 200, msg: '已清空历史记录' });
});

app.post('/api/tracker/reset', async (req, res) => {
  try {
    const raw = await postToPython('/tracker/reset', { fields: { session_id: req.body?.session_id || 'default' }, timeout: 10000, json: true });
    res.json({ code: 200, msg: raw.msg || 'tracker 已重置', data: raw.data || null });
  } catch (error) {
    res.status(500).json({ code: 500, msg: error.response?.data?.msg || error.message });
  }
});

app.post('/api/predict/image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ code: 400, msg: '未上传图片文件' });
    const raw = await postToPython('/predict/image', {
      filePath: req.file.path,
      fields: { conf: req.body.conf, iou: req.body.iou, imgsz: req.body.imgsz, save_visual: '1' }
    });
    const result = normalizePayload(raw, 'image');
    createHistoryRecords('image', result, { source_file: req.file.filename });
    res.json({ code: 200, msg: '识别成功', data: result });
  } catch (error) {
    res.status(500).json({ code: 500, msg: error.response?.data?.msg || error.message });
  }
});

app.post('/api/predict/frame', async (req, res) => {
  try {
    const { base64, frame_id, session_id } = req.body || {};
    if (!base64) return res.status(400).json({ code: 400, msg: '缺少 base64 图像数据' });
    const raw = await postToPython('/predict/frame', {
      fields: { base64, frame_id, session_id, conf: req.body.conf, iou: req.body.iou, imgsz: req.body.imgsz },
      timeout: 30000,
      json: true
    });
    const result = normalizePayload(raw, 'camera');
    res.json({ code: 200, msg: '识别成功', data: result });
  } catch (error) {
    res.status(500).json({ code: 500, msg: error.response?.data?.msg || error.message });
  }
});

app.post('/api/predict/video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ code: 400, msg: '未上传视频文件' });
    const raw = await postToPython('/predict/video', {
      filePath: req.file.path,
      fields: { frame_stride: req.body.frame_stride || 2, conf: req.body.conf, iou: req.body.iou, imgsz: req.body.imgsz || 640, save_visual: '1' },
      timeout: 30 * 60 * 1000
    });
    const result = normalizePayload(raw, 'video');
    createHistoryRecords('video', result, { source_file: req.file.filename });
    res.json({ code: 200, msg: '视频分析完成', data: result });
  } catch (error) {
    res.status(500).json({ code: 500, msg: error.response?.data?.msg || error.message });
  }
});

app.get('/api/recycle-points', async (req, res) => {
  const keyword = String(req.query.keyword || '').trim();
  const city = String(req.query.city || '').trim();
  if (!AMAP_WEB_KEY) return res.json({ code: 200, data: [], msg: '未配置高德 Web Key，当前仅返回空列表。' });
  try {
    const response = await axios.get('https://restapi.amap.com/v3/place/text', {
      params: { key: AMAP_WEB_KEY, keywords: keyword, city, offset: 8, page: 1, extensions: 'base' },
      timeout: 10000
    });
    const pois = Array.isArray(response.data?.pois) ? response.data.pois : [];
    res.json({ code: 200, data: pois });
  } catch (error) {
    res.status(500).json({ code: 500, msg: `回收点查询失败: ${error.message}` });
  }
});

const frontendPages = new Set(['/','/index.html','/image.html','/video.html','/camera.html','/recycle.html','/knowledge.html','/history.html']);

app.use((req, res, next) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({
      code: 404,
      msg: '接口不存在',
      data: null
    });
  }
  next();
});

app.get('/', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

app.get('/index.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

app.get('/image.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'image.html'));
});

app.get('/video.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'video.html'));
});

app.get('/camera.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'camera.html'));
});

app.get('/recycle.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'recycle.html'));
});

app.get('/knowledge.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'knowledge.html'));
});

app.get('/history.html', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'history.html'));
});

app.use((req, res) => {
  res.status(404).sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

setInterval(cleanupUploads, CLEANUP_INTERVAL_MINUTES * 60 * 1000);
cleanupUploads();

app.listen(PORT, () => console.log(`Node backend running at http://127.0.0.1:${PORT}`));
