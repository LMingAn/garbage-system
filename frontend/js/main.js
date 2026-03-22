const API_BASE = 'http://localhost:3000/api';

const FIXED_CAMERA_ROI = { x: 0.28, y: 0.18, w: 0.44, h: 0.62 };

const state = {
  lastResult: null,
  cameraLastResult: null,
  stream: null,
  detectInterval: null,
  detectBusy: false,
  lastSavedSignature: '',
  scienceRows: [],
  smoothedBox: null,
  geo: null,
  cameraSessionId: `cam_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  roi: FIXED_CAMERA_ROI,
  roiBox: null,
  sourceSize: null,
  settings: {}
};

const imageInput = document.getElementById('imageInput');
const pickImageBtn = document.getElementById('pickImageBtn');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const resultContainer = document.getElementById('resultContainer');
const classResultDisplay = document.getElementById('classResultDisplay');
const confResult = document.getElementById('confResult');
const adviceResult = document.getElementById('adviceResult');
const resultGroupBadge = document.getElementById('resultGroupBadge');
const errorContainer = document.getElementById('errorContainer');
const jumpScienceBtn = document.getElementById('jumpScienceBtn');
const loadPointBtn = document.getElementById('loadPointBtn');
const mapResult = document.getElementById('mapResult');

const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const snapshotRecordBtn = document.getElementById('snapshotRecordBtn');
const cameraVideo = document.getElementById('cameraVideo');
const overlayCanvas = document.getElementById('overlayCanvas');
const cameraResultContainer = document.getElementById('cameraResultContainer');
const cameraClassResult = document.getElementById('cameraClassResult');
const cameraGroupResult = document.getElementById('cameraGroupResult');
const cameraConfResult = document.getElementById('cameraConfResult');
const cameraAdviceResult = document.getElementById('cameraAdviceResult');
const cameraStableState = document.getElementById('cameraStableState');
const cameraScienceBtn = document.getElementById('cameraScienceBtn');
const cameraPointBtn = document.getElementById('cameraPointBtn');
const cameraMapResult = document.getElementById('cameraMapResult');

const historySearchInput = document.getElementById('historySearchInput');
const historyModeFilter = document.getElementById('historyModeFilter');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyTableBody = document.getElementById('historyTableBody');
const historyEmpty = document.getElementById('historyEmpty');
const scienceSearchInput = document.getElementById('scienceSearchInput');
const scienceGrid = document.getElementById('scienceGrid');


pickImageBtn.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files?.[0];
  if (file && file.type.startsWith('image/')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    imageInput.files = dt.files;
    handleImageChange();
  }
});
imageInput.addEventListener('change', handleImageChange);

function handleImageChange() {
  const file = imageInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewContainer.classList.remove('d-none');
    predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}


predictBtn.addEventListener('click', async () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  setButtonLoading(predictBtn, true, '识别中...', 'bi-hourglass-split');
  hideError();
  try {
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch(`${API_BASE}/predict/upload`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.code !== 200) throw new Error(data.msg || '识别失败');
    state.lastResult = data.data;
    renderUploadResult(data.data);
    await loadHistory();
  } catch (error) {
    showError(error.message || '网络异常');
  } finally {
    setButtonLoading(predictBtn, false, '开始识别', 'bi-search');
  }
});

function renderUploadResult(result) {
  resultContainer.classList.remove('d-none');
  classResultDisplay.textContent = result.display_name;
  confResult.textContent = result.confidence_text;
  adviceResult.textContent = result.advice;
  resultGroupBadge.textContent = result.category_group;
  mapResult.classList.add('d-none');
}

function renderCameraResult(result) {
  cameraResultContainer.classList.remove('d-none');
  cameraClassResult.textContent = result.display_name;
  cameraGroupResult.textContent = result.category_group;
  cameraConfResult.textContent = result.confidence_text;
  cameraAdviceResult.textContent = result.advice;
  cameraStableState.textContent = result.stable ? `已稳定（连续 ${result.stable_count || 1} 帧）` : `观察中（连续 ${result.stable_count || 1} 帧）`;
  state.roiBox = Array.isArray(result.roi_box) ? result.roi_box : null;
  state.sourceSize = result.source_size || null;
  cameraMapResult.classList.add('d-none');
}

function showError(msg) { errorContainer.textContent = msg; errorContainer.classList.remove('d-none'); }
function hideError() { errorContainer.classList.add('d-none'); }
function setButtonLoading(btn, loading, text, icon = 'bi-hourglass-split') {
  btn.disabled = loading;
  btn.innerHTML = `<i class="bi ${icon} me-2"></i>${text}`;
}

startCameraBtn.addEventListener('click', async () => {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    cameraVideo.srcObject = state.stream;
    await cameraVideo.play();
    syncOverlaySize();
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    snapshotRecordBtn.disabled = false;
    state.detectInterval = setInterval(runCameraDetect, 950);
  } catch (error) {
    alert('摄像头启动失败，请检查浏览器权限。');
  }
});

stopCameraBtn.addEventListener('click', stopCamera);
window.addEventListener('resize', syncOverlaySize);
window.addEventListener('beforeunload', stopCamera);

function stopCamera() {
  if (state.detectInterval) clearInterval(state.detectInterval);
  state.detectInterval = null;
  state.detectBusy = false;
  state.smoothedBox = null;
  state.cameraLastResult = null;
  if (state.stream) state.stream.getTracks().forEach(track => track.stop());
  state.stream = null;
  cameraVideo.srcObject = null;
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  state.roiBox = null;
  state.sourceSize = null;
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  snapshotRecordBtn.disabled = true;
}

function syncOverlaySize() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) return;
  overlayCanvas.width = cameraVideo.videoWidth;
  overlayCanvas.height = cameraVideo.videoHeight;
  drawPredictions(state.cameraLastResult?.predictions || []);
}

async function runCameraDetect() {
  if (state.detectBusy || !state.stream || !cameraVideo.videoWidth) return;
  state.detectBusy = true;
  try {
    syncOverlaySize();
    const captureCanvas = document.createElement('canvas');
    const maxW = 640;
    const ratio = cameraVideo.videoWidth / cameraVideo.videoHeight;
    captureCanvas.width = Math.min(maxW, cameraVideo.videoWidth);
    captureCanvas.height = Math.round(captureCanvas.width / ratio);
    const ctx = captureCanvas.getContext('2d', { willReadFrequently: false });
    ctx.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    const base64 = captureCanvas.toDataURL('image/jpeg', 0.76);
    const res = await fetch(`${API_BASE}/predict/camera`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base64,
        saveRecord: false,
        session_id: state.cameraSessionId,
      })
    });
    const data = await res.json();
    if (data.code === 200 && data.data) {
      state.cameraLastResult = data.data;
      renderCameraResult(data.data);
      drawPredictions(data.data.predictions || []);
    }
  } catch (error) {
    console.error('camera detect failed', error);
  } finally {
    state.detectBusy = false;
  }
}

function drawPredictions(predictions) {
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  drawROI(ctx);
  if (!predictions.length) return;

  const candidates = predictions
    .filter(item => Array.isArray(item.bbox) && item.bbox.length === 4)
    .map(item => normalizeBox(item.bbox, overlayCanvas.width, overlayCanvas.height, item))
    .filter(Boolean)
    .sort((a, b) => (b.item.score || b.item.confidence || 0) - (a.item.score || a.item.confidence || 0));

  if (!candidates.length) return;
  const best = candidates[0];
  state.smoothedBox = smoothBox(best);
  const box = state.smoothedBox;

  ctx.lineWidth = 4;
  ctx.strokeStyle = best.item.suppressed ? '#f59e0b' : '#22c55e';
  ctx.fillStyle = best.item.suppressed ? 'rgba(245,158,11,.12)' : 'rgba(34,197,94,.15)';
  ctx.strokeRect(box.x1, box.y1, box.w, box.h);
  ctx.fillRect(box.x1, box.y1, box.w, box.h);

  const label = `${best.item.display_name || best.item.class_name} ${best.item.confidence_text || ((best.item.confidence * 100).toFixed(1) + '%')}`;
  ctx.font = 'bold 20px Microsoft YaHei';
  const tw = ctx.measureText(label).width + 24;
  const th = 34;
  const tx = box.x1;
  const ty = Math.max(0, box.y1 - th - 6);
  ctx.fillStyle = best.item.suppressed ? '#f59e0b' : '#22c55e';
  ctx.fillRect(tx, ty, tw, th);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, tx + 12, ty + 23);
}

function drawROI(ctx) {
  if (!overlayCanvas.width || !overlayCanvas.height) return;
  const roi = state.roi || FIXED_CAMERA_ROI;
  const x = roi.x * overlayCanvas.width;
  const y = roi.y * overlayCanvas.height;
  const w = roi.w * overlayCanvas.width;
  const h = roi.h * overlayCanvas.height;
  ctx.save();
  ctx.fillStyle = 'rgba(4,10,23,.20)';
  ctx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  ctx.clearRect(x, y, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.90)';
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 8]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,255,255,.96)';
  ctx.font = 'bold 16px Microsoft YaHei';
  ctx.fillText('推荐识别区域', x + 8, Math.max(20, y - 12));
  ctx.restore();
}


function normalizeBox(bbox, canvasW, canvasH, item) {
  let [x1, y1, x2, y2] = bbox.map(Number);
  const sourceW = Number(state.sourceSize?.width || canvasW);
  const sourceH = Number(state.sourceSize?.height || canvasH);
  const scaleX = canvasW / sourceW;
  const scaleY = canvasH / sourceH;

  if (x2 <= 1 && y2 <= 1) {
    x1 *= canvasW; x2 *= canvasW; y1 *= canvasH; y2 *= canvasH;
  } else {
    x1 *= scaleX; x2 *= scaleX; y1 *= scaleY; y2 *= scaleY;
  }

  x1 = Math.max(0, Math.min(canvasW, x1));
  x2 = Math.max(0, Math.min(canvasW, x2));
  y1 = Math.max(0, Math.min(canvasH, y1));
  y2 = Math.max(0, Math.min(canvasH, y2));
  const w = x2 - x1, h = y2 - y1;
  if (w <= 0 || h <= 0) return null;
  const area = w * h;
  if (area >= canvasW * canvasH * 0.94) return null;

  const roiPixels = getCurrentRoiPixels(canvasW, canvasH);
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  if (roiPixels && !(cx >= roiPixels.x1 && cx <= roiPixels.x2 && cy >= roiPixels.y1 && cy <= roiPixels.y2)) {
    return null;
  }

  return { x1, y1, x2, y2, w, h, item };
}

function getCurrentRoiPixels(canvasW, canvasH) {
  if (Array.isArray(state.roiBox) && state.roiBox.length === 4 && state.sourceSize?.width && state.sourceSize?.height) {
    const [rx1, ry1, rx2, ry2] = state.roiBox.map(Number);
    const sx = canvasW / Number(state.sourceSize.width);
    const sy = canvasH / Number(state.sourceSize.height);
    return { x1: rx1 * sx, y1: ry1 * sy, x2: rx2 * sx, y2: ry2 * sy };
  }
  const roi = state.roi || FIXED_CAMERA_ROI;
  return { x1: roi.x * canvasW, y1: roi.y * canvasH, x2: (roi.x + roi.w) * canvasW, y2: (roi.y + roi.h) * canvasH };
}

function smoothBox(next) {
  if (!state.smoothedBox) return next;
  const alpha = 0.5;
  const prev = state.smoothedBox;
  const x1 = prev.x1 * (1 - alpha) + next.x1 * alpha;
  const y1 = prev.y1 * (1 - alpha) + next.y1 * alpha;
  const w = prev.w * (1 - alpha) + next.w * alpha;
  const h = prev.h * (1 - alpha) + next.h * alpha;
  return { ...next, x1, y1, w, h };
}

snapshotRecordBtn.addEventListener('click', async () => {
  if (!state.cameraLastResult) return;
  const signature = `${state.cameraLastResult.class_name}_${state.cameraLastResult.confidence_text}`;
  if (signature === state.lastSavedSignature) return alert('当前识别结果已保存过一次。');
  try {
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = Math.min(640, cameraVideo.videoWidth);
    captureCanvas.height = Math.round(captureCanvas.width * cameraVideo.videoHeight / cameraVideo.videoWidth);
    captureCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const base64 = captureCanvas.toDataURL('image/jpeg', 0.76);
    const res = await fetch(`${API_BASE}/predict/camera`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base64,
        saveRecord: true,
        session_id: state.cameraSessionId,
      })
    });
    const data = await res.json();
    if (data.code !== 200) throw new Error(data.msg || '保存失败');
    state.lastSavedSignature = signature;
    await loadHistory();
    alert('当前识别结果已保存到历史记录。');
  } catch (error) {
    alert(error.message || '保存失败');
  }
});

async function loadHistory() {
  const q = encodeURIComponent(historySearchInput.value.trim());
  const mode = encodeURIComponent(historyModeFilter.value);
  const res = await fetch(`${API_BASE}/history?q=${q}&mode=${mode}`);
  const data = await res.json();
  const rows = data.data || [];
  historyTableBody.innerHTML = rows.map(item => `
    <tr>
      <td><button class="btn btn-link p-0 text-decoration-none history-link" data-slug="${item.class_name}">${item.display_name || item.class_name}</button></td>
      <td><span class="badge text-bg-light">${item.category_group || '-'}</span></td>
      <td>${item.confidence_text || ((item.confidence || 0) * 100).toFixed(2) + '%'}</td>
      <td>${item.detect_mode === 'upload' ? '图片上传' : '摄像头识别'}</td>
      <td>${formatDate(item.identify_time)}</td>
      <td class="text-secondary">${item.advice || '-'}</td>
      <td><button class="btn btn-sm btn-outline-danger delete-history-btn" data-id="${item.id}">删除</button></td>
    </tr>
  `).join('');
  historyEmpty.classList.toggle('d-none', rows.length > 0);

  document.querySelectorAll('.delete-history-btn').forEach(btn => btn.addEventListener('click', () => deleteHistory(btn.dataset.id)));
  document.querySelectorAll('.history-link').forEach(btn => btn.addEventListener('click', () => scrollToScience(btn.dataset.slug)));
}

async function deleteHistory(id) {
  await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
  await loadHistory();
}

clearHistoryBtn.addEventListener('click', async () => {
  if (!confirm('确定清空全部历史记录吗？')) return;
  await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  await loadHistory();
});
historySearchInput.addEventListener('input', debounce(loadHistory, 250));
historyModeFilter.addEventListener('change', loadHistory);

async function loadScience() {
  const res = await fetch(`${API_BASE}/knowledge`);
  const data = await res.json();
  state.scienceRows = data.data || [];
  renderScience(state.scienceRows);
}

function renderScience(rows) {
  const q = scienceSearchInput.value.trim().toLowerCase();
  const filtered = rows.filter(item => {
    if (!q) return true;
    return [item.slug, item.title, item.group, item.definition, ...(item.examples || [])].join(' ').toLowerCase().includes(q);
  });
  scienceGrid.innerHTML = filtered.map(item => `
    <div class="col-md-6 col-xl-4" id="science-${item.slug}">
      <div class="science-card p-4">
        <div class="d-flex justify-content-between align-items-start gap-3">
          <div>
            <span class="badge text-bg-primary-subtle text-primary-emphasis">${item.group}</span>
            <h4 class="mt-3 mb-1">${item.title}</h4>
            <div class="text-muted small">分类标识：${item.slug}</div>
          </div>
          <i class="bi bi-journal-richtext fs-2 text-primary"></i>
        </div>
        <div class="mt-3"><strong>定义：</strong><p class="text-secondary mt-2">${item.definition}</p></div>
        <div><strong>常见示例：</strong><ul>${(item.examples || []).map(ex => `<li>${ex}</li>`).join('')}</ul></div>
        <div><strong>危害说明：</strong><p class="text-secondary">${item.hazard || '-'}</p></div>
        <div><strong>回收价值：</strong><p class="text-secondary">${item.value || '-'}</p></div>
      </div>
    </div>
  `).join('');
}

scienceSearchInput.addEventListener('input', () => renderScience(state.scienceRows));

function scrollToScience(slug) {
  const item = document.getElementById(`science-${slug}`);
  document.getElementById('science').scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(() => {
    item?.classList.add('highlight');
    item?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => item?.classList.remove('highlight'), 1600);
  }, 300);
}

jumpScienceBtn.addEventListener('click', () => state.lastResult && scrollToScience(state.lastResult.class_name));
cameraScienceBtn.addEventListener('click', () => state.cameraLastResult && scrollToScience(state.cameraLastResult.class_name));
loadPointBtn.addEventListener('click', () => state.lastResult && loadRecyclePoints(state.lastResult.class_name, mapResult));
cameraPointBtn.addEventListener('click', () => state.cameraLastResult && loadRecyclePoints(state.cameraLastResult.class_name, cameraMapResult));

async function ensureGeo() {
  if (state.geo) return state.geo;
  if (!navigator.geolocation) return null;
  return new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        state.geo = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        resolve(state.geo);
      },
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
    );
  });
}

async function loadRecyclePoints(className, container) {
  container.classList.remove('d-none');
  container.innerHTML = '正在查询附近回收点...';
  const geo = await ensureGeo();
  const params = new URLSearchParams({ class_name: className });
  if (geo) { params.set('lat', geo.lat); params.set('lng', geo.lng); }
  const res = await fetch(`${API_BASE}/recycle-points?${params.toString()}`);
  const data = await res.json();
  const info = data.data || {};
  const points = info.points || [];
  container.innerHTML = `
    <div class="d-flex justify-content-between align-items-start gap-3 flex-wrap">
      <div>
        <h5 class="mb-1">周边回收点 / 导航建议</h5>
        <p class="text-secondary mb-2">${info.tips || '已查询附近投放点。'}</p>
      </div>
      <div class="d-flex gap-2 flex-wrap">
        <a class="btn btn-outline-primary btn-sm" target="_blank" href="${info.amapNav || '#'}">高德地图导航</a>
        <a class="btn btn-outline-secondary btn-sm" target="_blank" href="${info.baiduNav || '#'}">百度地图导航</a>
      </div>
    </div>
    <div class="mt-3">${points.length ? points.map(p => `
      <div class="border rounded-4 p-3 mb-2 bg-white">
        <div class="fw-semibold">${p.name}</div>
        <div class="text-secondary small mt-1">${p.address || '暂无地址'} ${p.distance ? `· 距离约 ${p.distance} 米` : ''}</div>
        ${p.amapUri ? `<a target="_blank" class="btn btn-sm btn-link px-0 mt-1" href="${p.amapUri}">地图预览 / 定位</a>` : ''}
      </div>
    `).join('') : '<div class="text-secondary">未获取到真实点位列表，可直接点击上方导航按钮查看周边结果。</div>'}</div>
  `;
}

function formatDate(str) {
  const date = new Date(str);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function debounce(fn, wait = 300) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

(async function init() {
  await Promise.all([loadScience(), loadHistory()]);
})();
