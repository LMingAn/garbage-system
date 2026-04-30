const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const port = process.env.PORT || 3000;
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://127.0.0.1:5000';
const FRONTEND_DIR = path.resolve(__dirname, '../frontend');
const UPLOAD_DIR = path.resolve(__dirname, 'uploads');
const DATA_DIR = path.resolve(__dirname, 'data');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');
const CATEGORY_META_FILE = path.resolve(__dirname, 'config/category_meta.json');
const ADVICE_FILE = path.resolve(__dirname, 'config/recycle_advice.json');
const AMAP_KEY = process.env.AMAP_WEB_KEY || '';
const UPLOAD_CLEANUP_MAX_AGE_HOURS = Number(process.env.UPLOAD_CLEANUP_MAX_AGE_HOURS || 24);
const UPLOAD_CLEANUP_INTERVAL_MINUTES = Number(process.env.UPLOAD_CLEANUP_INTERVAL_MINUTES || 60);
const UPLOAD_CLEANUP_MIN_FILE_AGE_MINUTES = Number(process.env.UPLOAD_CLEANUP_MIN_FILE_AGE_MINUTES || 10);

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(HISTORY_FILE)) fs.writeFileSync(HISTORY_FILE, '[]', 'utf-8');

const recycleAdvice = JSON.parse(fs.readFileSync(ADVICE_FILE, 'utf-8'));
const categoryMeta = JSON.parse(fs.readFileSync(CATEGORY_META_FILE, 'utf-8'));

app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/static', express.static(FRONTEND_DIR));
app.use(express.static(FRONTEND_DIR));

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const safe = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}${path.extname(file.originalname || '.jpg')}`;
    cb(null, safe);
  }
});
const upload = multer({ storage });

function readHistory() {
  try {
    return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
  } catch (error) {
    return [];
  }
}

function writeHistory(records) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(records, null, 2), 'utf-8');
}


function cleanupUploads() {
  const now = Date.now();
  const maxAgeMs = Math.max(1, UPLOAD_CLEANUP_MAX_AGE_HOURS) * 60 * 60 * 1000;
  const minFileAgeMs = Math.max(1, UPLOAD_CLEANUP_MIN_FILE_AGE_MINUTES) * 60 * 1000;
  let removedCount = 0;

  try {
    const files = fs.readdirSync(UPLOAD_DIR);
    for (const file of files) {
      const filePath = path.join(UPLOAD_DIR, file);
      let stat;
      try {
        stat = fs.statSync(filePath);
      } catch (error) {
        continue;
      }

      if (!stat.isFile()) continue;

      const ageMs = now - stat.mtimeMs;
      if (ageMs < minFileAgeMs) continue;
      if (ageMs < maxAgeMs) continue;

      try {
        fs.unlinkSync(filePath);
        removedCount += 1;
      } catch (error) {
        console.warn(`清理上传文件失败: ${filePath}`, error.message);
      }
    }
  } catch (error) {
    console.warn('扫描 uploads 目录失败:', error.message);
  }

  if (removedCount > 0) {
    console.log(`uploads 自动清理完成，共删除 ${removedCount} 个过期文件`);
  }

  return removedCount;
}

function getCategoryInfo(slug) {
  const meta = categoryMeta[slug] || {};
  return {
    slug,
    title: meta.name || slug,
    group: meta.group || '未分类',
    definition: meta.definition || '',
    examples: meta.examples || [],
    hazard: meta.hazard || '',
    value: meta.value || '',
    advice: recycleAdvice[slug] || '请按当地垃圾分类要求投放。'
  };
}

function normalizePredictionPayload(payload, mode) {
  const data = payload?.data || payload;
  const predictions = Array.isArray(data?.predictions) ? data.predictions : [];
  const primary = data?.class_name ? data : (predictions[0] || {});
  const slug = primary.class_name || primary.slug || 'unknown';
  const info = getCategoryInfo(slug);
  const confidence = typeof primary.confidence === 'number' ? Number(primary.confidence.toFixed(4)) : Number(primary.confidence || 0);

  return {
    class_name: slug,
    display_name: info.title,
    confidence,
    confidence_text: `${(confidence * 100).toFixed(2)}%`,
    advice: info.advice,
    category_group: info.group,
    science: info,
    predictions: predictions.map(item => {
      const itemSlug = item.class_name || item.slug || 'unknown';
      const itemInfo = getCategoryInfo(itemSlug);
      const conf = typeof item.confidence === 'number' ? Number(item.confidence.toFixed(4)) : Number(item.confidence || 0);
      return {
        class_name: itemSlug,
        display_name: itemInfo.title,
        category_group: itemInfo.group,
        confidence: conf,
        confidence_text: `${(conf * 100).toFixed(2)}%`,
        bbox: item.bbox || item.box || [],
        advice: itemInfo.advice,
        score: item.score || null
      };
    }),
    raw_image: data?.image || null,
    detect_mode: mode,
    stable: Boolean(data?.stable),
    stable_count: Number(data?.stable_count || 0),
    roi: data?.roi || null,
    roi_box: data?.roi_box || null,
    focus_zone: data?.focus_zone || null,
    source_size: data?.source_size || null,
    params: data?.params || {}
  };
}

function appendHistory(record) {
  const all = readHistory();
  all.unshift(record);
  writeHistory(all.slice(0, 500));
  return record;
}

function createHistoryRecord(result, mode, extra = {}) {
  return appendHistory({
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    class_name: result.class_name,
    display_name: result.display_name,
    confidence: result.confidence,
    confidence_text: result.confidence_text,
    advice: result.advice,
    category_group: result.category_group,
    detect_mode: mode,
    identify_time: new Date().toISOString(),
    ...extra
  });
}

async function requestPythonPrediction({ filePath, base64, mode, options = {} }) {
  if (filePath) {
    const form = new FormData();
    form.append('image', fs.createReadStream(filePath));
    Object.entries(options).forEach(([key, value]) => {
      if (value === undefined || value === null || value === '') return;
      form.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
    });
    form.append('mode', mode);
    const response = await axios.post(`${PYTHON_API_BASE}/predict`, form, { headers: form.getHeaders(), timeout: 30000 });
    return normalizePredictionPayload(response.data, mode);
  }
  const response = await axios.post(`${PYTHON_API_BASE}/predict`, { base64, mode, ...options }, { timeout: 30000 });
  return normalizePredictionPayload(response.data, mode);
}

app.get('/api/health', (req, res) => {
  res.json({
    code: 200,
    msg: 'ok',
    data: {
      pythonApi: PYTHON_API_BASE,
      amapConfigured: Boolean(AMAP_KEY),
      uploadCleanup: {
        enabled: true,
        maxAgeHours: UPLOAD_CLEANUP_MAX_AGE_HOURS,
        intervalMinutes: UPLOAD_CLEANUP_INTERVAL_MINUTES,
        minFileAgeMinutes: UPLOAD_CLEANUP_MIN_FILE_AGE_MINUTES
      }
    }
  });
});

app.get('/api/categories', (req, res) => {
  const rows = Object.keys(categoryMeta).map(getCategoryInfo);
  res.json({ code: 200, data: rows });
});

app.get('/api/knowledge', (req, res) => {
  const q = String(req.query.q || '').trim().toLowerCase();
  const rows = Object.keys(categoryMeta).map(getCategoryInfo).filter(item => {
    if (!q) return true;
    const hay = [item.slug, item.title, item.group, item.definition, ...(item.examples || [])].join(' ').toLowerCase();
    return hay.includes(q);
  });
  res.json({ code: 200, data: rows });
});

app.get('/api/history', (req, res) => {
  const q = String(req.query.q || '').trim().toLowerCase();
  const mode = String(req.query.mode || '').trim();
  let rows = readHistory();
  if (q) {
    rows = rows.filter(item => [item.class_name, item.display_name, item.advice, item.category_group, item.detect_mode].join(' ').toLowerCase().includes(q));
  }
  if (mode) rows = rows.filter(item => item.detect_mode === mode);
  res.json({ code: 200, data: rows });
});

app.delete('/api/history/:id', (req, res) => {
  const rows = readHistory();
  const next = rows.filter(item => item.id !== req.params.id);
  writeHistory(next);
  res.json({ code: 200, msg: '删除成功' });
});

app.delete('/api/history', (req, res) => {
  writeHistory([]);
  res.json({ code: 200, msg: '历史记录已清空' });
});

app.post('/api/predict/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ code: 400, msg: '请上传图片文件' });
    const options = {};
    const result = await requestPythonPrediction({ filePath: req.file.path, mode: 'upload', options });
    const record = createHistoryRecord(result, 'upload', { image_path: `/uploads/${path.basename(req.file.path)}` });
    res.json({ code: 200, msg: '识别成功', data: { ...result, history_id: record.id } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ code: 500, msg: `识别失败: ${err.response?.data?.msg || err.message}` });
  }
});

app.post('/api/predict/camera', async (req, res) => {
  try {
    const { base64, saveRecord = false, session_id } = req.body;
    if (!base64) return res.status(400).json({ code: 400, msg: '缺少 base64 图片' });
    const options = { session_id };
    const result = await requestPythonPrediction({ base64, mode: 'camera', options });
    let historyId = null;
    if (saveRecord) {
      const record = createHistoryRecord(result, 'camera');
      historyId = record.id;
    }
    res.json({ code: 200, msg: '识别成功', data: { ...result, history_id: historyId } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ code: 500, msg: `识别失败: ${err.response?.data?.msg || err.message}` });
  }
});

app.get('/api/recycle-points', async (req, res) => {
  const { lat, lng, class_name = '' } = req.query;
  const info = getCategoryInfo(class_name);
  const keywords = encodeURIComponent(`${info.group} 回收点`);
  const amapNav = lat && lng
    ? `https://uri.amap.com/search?keyword=${keywords}&center=${lng},${lat}`
    : `https://uri.amap.com/search?keyword=${keywords}`;
  const baiduNav = lat && lng
    ? `https://map.baidu.com/search/${keywords}/@${lng},${lat},17z`
    : `https://map.baidu.com/search/${keywords}`;

  if (!lat || !lng) {
    return res.json({ code: 200, data: { provider: 'link', points: [], amapNav, baiduNav, tips: '未获取定位，已提供地图搜索与导航入口。' } });
  }

  if (!AMAP_KEY) {
    return res.json({
      code: 200,
      data: {
        provider: 'link',
        points: [
          { id: 'fallback-1', name: `${info.group}投放点搜索`, address: '请点击下方导航按钮在高德/百度地图中查看周边结果', location: { lat: Number(lat), lng: Number(lng) } }
        ],
        amapNav,
        baiduNav,
        tips: '未配置高德 Web API Key，当前返回导航链接方案。配置后可展示真实周边回收点列表。'
      }
    });
  }

  try {
    const radius = 3000;
    const amapResp = await axios.get('https://restapi.amap.com/v5/place/around', {
      params: {
        key: AMAP_KEY,
        location: `${lng},${lat}`,
        keywords: `${info.group} 回收`,
        radius,
        page_size: 5,
        sortrule: 'distance'
      },
      timeout: 15000
    });
    const pois = Array.isArray(amapResp.data?.pois) ? amapResp.data.pois : [];
    const points = pois.map((poi, index) => {
      const [poiLng, poiLat] = String(poi.location || ',').split(',');
      return {
        id: poi.id || `poi-${index}`,
        name: poi.name,
        address: poi.address || poi.cityname || '暂无地址',
        distance: poi.distance,
        location: { lat: Number(poiLat), lng: Number(poiLng) },
        amapUri: `https://uri.amap.com/marker?position=${poiLng},${poiLat}&name=${encodeURIComponent(poi.name)}`
      };
    });
    res.json({ code: 200, data: { provider: 'amap', points, amapNav, baiduNav, tips: points.length ? '已按距离排序展示周边回收点。' : '未搜索到附近回收点，请尝试扩大范围或切换地图导航。' } });
  } catch (error) {
    console.error('recycle point search failed', error.message);
    res.json({ code: 200, data: { provider: 'link', points: [], amapNav, baiduNav, tips: '地图服务查询失败，已退化为地图搜索链接。' } });
  }
});

app.use((req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

cleanupUploads();
const cleanupTimer = setInterval(cleanupUploads, Math.max(1, UPLOAD_CLEANUP_INTERVAL_MINUTES) * 60 * 1000);
cleanupTimer.unref?.();

app.listen(port, () => {
  console.log(`Node服务器运行在 http://localhost:${port}`);
  console.log(`uploads 自动清理已启用：保留 ${UPLOAD_CLEANUP_MAX_AGE_HOURS} 小时内文件，每 ${UPLOAD_CLEANUP_INTERVAL_MINUTES} 分钟清理一次`);
});
