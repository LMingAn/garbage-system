const API_BASE = 'http://localhost:3000/api';
const FIXED_CAMERA_ROI = { x: 0.28, y: 0.18, w: 0.44, h: 0.62 };

const state = {
  uploadResult: null,
  uploadPredictions: [],
  selectedFile: null,
  stream: null,
  detectTimer: null,
  detectBusy: false,
  cameraLastResult: null,
  cameraSessionId: `cam_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  cameraSourceSize: null,
  geo: null,
  scienceRows: [],
};

const $ = (id) => document.getElementById(id);
const page = document.body.dataset.page;

document.addEventListener('DOMContentLoaded', () => {
  if (page === 'image') initImagePage();
  if (page === 'camera') initCameraPage();
  if (page === 'history') initHistoryPage();
  if (page === 'knowledge') initKnowledgePage();
  if (page === 'recycle') initRecyclePage();
});

function show(el, visible) {
  if (!el) return;
  el.classList.toggle('hidden', !visible);
}

async function api(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const data = await res.json();
  if (data.code && data.code !== 200) throw new Error(data.msg || '请求失败');
  return data.data;
}

function initImagePage() {
  const uploadArea = $('uploadArea');
  const imageInput = $('imageInput');
  const predictBtn = $('predictBtn');

  uploadArea.addEventListener('click', () => imageInput.click());
  uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) setSelectedImage(file);
  });
  imageInput.addEventListener('change', () => {
    const file = imageInput.files?.[0];
    if (file) setSelectedImage(file);
  });
  $('previewImage').addEventListener('load', drawUploadBoxes);
  predictBtn.addEventListener('click', predictUpload);
  $('scienceBtn').addEventListener('click', () => {
    if (state.uploadResult) location.href = `knowledge.html?q=${encodeURIComponent(state.uploadResult.class_name)}`;
  });
  $('recycleBtn').addEventListener('click', () => {
    if (state.uploadResult) location.href = `recycle.html?class_name=${encodeURIComponent(state.uploadResult.class_name)}`;
  });
}

function setSelectedImage(file) {
  state.selectedFile = file;
  state.uploadResult = null;
  state.uploadPredictions = [];
  $('previewImage').src = URL.createObjectURL(file);
  show($('previewBox'), true);
  show($('uploadError'), false);
  show($('uploadEmpty'), true);
  show($('uploadResult'), false);
  $('predictBtn').disabled = false;
}

async function predictUpload() {
  if (!state.selectedFile) return;
  const btn = $('predictBtn');
  btn.disabled = true;
  btn.textContent = '识别中...';
  show($('uploadError'), false);
  try {
    const form = new FormData();
    form.append('image', state.selectedFile);
    const data = await api('/predict/upload', { method: 'POST', body: form });
    state.uploadResult = data;
    state.uploadPredictions = Array.isArray(data.predictions) ? data.predictions : [];
    renderUploadResult(data);
    drawUploadBoxes();
  } catch (error) {
    $('uploadError').textContent = error.message || '识别失败';
    show($('uploadError'), true);
  } finally {
    btn.disabled = false;
    btn.textContent = '开始识别';
  }
}

function renderUploadResult(result) {
  show($('uploadEmpty'), false);
  show($('uploadResult'), true);
  $('classResult').textContent = result.display_name || result.class_name || '-';
  $('confResult').textContent = result.confidence_text || percent(result.confidence);
  $('adviceResult').textContent = result.advice || '-';
  $('resultGroup').textContent = result.category_group || '-';
  $('boxCount').textContent = state.uploadPredictions.length;
  $('boxList').innerHTML = state.uploadPredictions.map(item => `
    <div class="box-row">
      <span>${escapeHtml(item.display_name || item.class_name || '-')}</span>
      <strong>${escapeHtml(item.confidence_text || percent(item.confidence))}</strong>
    </div>
  `).join('');
}

function drawUploadBoxes() {
  const img = $('previewImage');
  const canvas = $('uploadCanvas');
  if (!img || !canvas || !img.clientWidth || !img.clientHeight) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(img.clientWidth * dpr);
  canvas.height = Math.round(img.clientHeight * dpr);
  canvas.style.width = `${img.clientWidth}px`;
  canvas.style.height = `${img.clientHeight}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, img.clientWidth, img.clientHeight);
  if (!state.uploadPredictions.length) return;
  const sourceW = Number(state.uploadResult?.source_size?.width || img.naturalWidth);
  const sourceH = Number(state.uploadResult?.source_size?.height || img.naturalHeight);
  const fit = containRect(img.clientWidth, img.clientHeight, sourceW, sourceH);
  state.uploadPredictions.forEach((item, index) => drawDetectionBox(ctx, item, fit, sourceW, sourceH, index));
}

function initCameraPage() {
  $('startCameraBtn').addEventListener('click', startCamera);
  $('stopCameraBtn').addEventListener('click', stopCamera);
  $('snapshotRecordBtn').addEventListener('click', saveCameraRecord);
  $('cameraScienceBtn').addEventListener('click', () => {
    if (state.cameraLastResult) location.href = `knowledge.html?q=${encodeURIComponent(state.cameraLastResult.class_name)}`;
  });
  $('cameraRecycleBtn').addEventListener('click', () => {
    if (state.cameraLastResult) location.href = `recycle.html?class_name=${encodeURIComponent(state.cameraLastResult.class_name)}`;
  });
  window.addEventListener('resize', syncCameraOverlay);
  window.addEventListener('beforeunload', stopCamera);
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = $('cameraVideo');
    video.srcObject = state.stream;
    await video.play();
    $('startCameraBtn').disabled = true;
    $('stopCameraBtn').disabled = false;
    $('snapshotRecordBtn').disabled = false;
    syncCameraOverlay();
    state.detectTimer = setInterval(runCameraDetect, 950);
  } catch (error) {
    alert('摄像头启动失败，请检查浏览器权限。');
  }
}

function stopCamera() {
  if (state.detectTimer) clearInterval(state.detectTimer);
  state.detectTimer = null;
  state.detectBusy = false;
  if (state.stream) state.stream.getTracks().forEach(track => track.stop());
  state.stream = null;
  const video = $('cameraVideo');
  if (video) video.srcObject = null;
  const canvas = $('overlayCanvas');
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  if ($('startCameraBtn')) $('startCameraBtn').disabled = false;
  if ($('stopCameraBtn')) $('stopCameraBtn').disabled = true;
  if ($('snapshotRecordBtn')) $('snapshotRecordBtn').disabled = true;
}

function syncCameraOverlay() {
  const video = $('cameraVideo');
  const canvas = $('overlayCanvas');
  if (!video || !canvas || !video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  drawCameraPredictions(state.cameraLastResult?.predictions || []);
}

async function runCameraDetect() {
  const video = $('cameraVideo');
  if (state.detectBusy || !state.stream || !video.videoWidth) return;
  state.detectBusy = true;
  try {
    const capture = document.createElement('canvas');
    capture.width = Math.min(640, video.videoWidth);
    capture.height = Math.round(capture.width * video.videoHeight / video.videoWidth);
    capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
    const data = await api('/predict/camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base64: capture.toDataURL('image/jpeg', 0.76),
        saveRecord: false,
        session_id: state.cameraSessionId,
      }),
    });
    state.cameraLastResult = data;
    state.cameraSourceSize = data.source_size || null;
    renderCameraResult(data);
    syncCameraOverlay();
  } catch (error) {
    console.error('camera detect failed', error);
  } finally {
    state.detectBusy = false;
  }
}

function renderCameraResult(result) {
  show($('cameraEmpty'), false);
  show($('cameraResult'), true);
  $('cameraClassResult').textContent = result.display_name || result.class_name || '-';
  $('cameraConfResult').textContent = result.confidence_text || percent(result.confidence);
  $('cameraGroupResult').textContent = result.category_group || '-';
  $('cameraAdviceResult').textContent = result.advice || '-';
  $('cameraStableState').textContent = `${result.stable ? '已稳定' : '观察中'}，连续 ${result.stable_count || 1} 帧`;
}

function drawCameraPredictions(predictions) {
  const canvas = $('overlayCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawRoi(ctx, canvas.width, canvas.height);
  const sourceW = Number(state.cameraSourceSize?.width || canvas.width);
  const sourceH = Number(state.cameraSourceSize?.height || canvas.height);
  const fit = { x: 0, y: 0, scale: canvas.width / sourceW, scaleY: canvas.height / sourceH };
  predictions.slice(0, 3).forEach((item, index) => drawDetectionBox(ctx, item, fit, sourceW, sourceH, index));
}

function drawRoi(ctx, width, height) {
  const x = FIXED_CAMERA_ROI.x * width;
  const y = FIXED_CAMERA_ROI.y * height;
  const w = FIXED_CAMERA_ROI.w * width;
  const h = FIXED_CAMERA_ROI.h * height;
  ctx.save();
  ctx.fillStyle = 'rgba(15,23,42,.22)';
  ctx.fillRect(0, 0, width, height);
  ctx.clearRect(x, y, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.9)';
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 8]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 16px Microsoft YaHei';
  ctx.fillText('推荐识别区域', x + 8, Math.max(22, y - 10));
  ctx.restore();
}

async function saveCameraRecord() {
  const video = $('cameraVideo');
  if (!state.cameraLastResult || !video.videoWidth) return;
  const capture = document.createElement('canvas');
  capture.width = Math.min(640, video.videoWidth);
  capture.height = Math.round(capture.width * video.videoHeight / video.videoWidth);
  capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
  await api('/predict/camera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      base64: capture.toDataURL('image/jpeg', 0.76),
      saveRecord: true,
      session_id: state.cameraSessionId,
    }),
  });
  alert('已保存到历史记录。');
}

function initHistoryPage() {
  $('historySearchInput').addEventListener('input', debounce(loadHistory, 250));
  $('historyModeFilter').addEventListener('change', loadHistory);
  $('clearHistoryBtn').addEventListener('click', clearHistory);
  loadHistory();
}

async function loadHistory() {
  const params = new URLSearchParams();
  if ($('historySearchInput').value.trim()) params.set('q', $('historySearchInput').value.trim());
  if ($('historyModeFilter').value) params.set('mode', $('historyModeFilter').value);
  const rows = await api(`/history?${params.toString()}`);
  $('historyTableBody').innerHTML = rows.map(item => `
    <tr>
      <td>${escapeHtml(item.display_name || item.class_name || '-')}</td>
      <td>${escapeHtml(item.category_group || '-')}</td>
      <td>${escapeHtml(item.confidence_text || percent(item.confidence))}</td>
      <td>${item.detect_mode === 'upload' ? '图片上传' : '摄像头识别'}</td>
      <td>${formatDate(item.identify_time)}</td>
      <td>${escapeHtml(item.advice || '-')}</td>
      <td><button class="danger" data-delete-id="${escapeHtml(item.id)}">删除</button></td>
    </tr>
  `).join('');
  show($('historyEmpty'), rows.length === 0);
  document.querySelectorAll('[data-delete-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      await api(`/history/${encodeURIComponent(btn.dataset.deleteId)}`, { method: 'DELETE' });
      loadHistory();
    });
  });
}

async function clearHistory() {
  if (!confirm('确定清空全部历史记录吗？')) return;
  await api('/history', { method: 'DELETE' });
  loadHistory();
}

function initKnowledgePage() {
  const q = new URLSearchParams(location.search).get('q') || '';
  $('scienceSearchInput').value = q;
  $('scienceSearchInput').addEventListener('input', () => renderScience(state.scienceRows));
  loadScience().then(rows => renderScience(rows));
}

async function loadScience() {
  state.scienceRows = await api('/knowledge');
  return state.scienceRows;
}

function renderScience(rows) {
  const q = ($('scienceSearchInput')?.value || '').trim().toLowerCase();
  const filtered = rows.filter(item => {
    if (!q) return true;
    return [item.slug, item.title, item.group, item.definition, ...(item.examples || [])].join(' ').toLowerCase().includes(q);
  });
  $('scienceGrid').innerHTML = filtered.map(item => `
    <article class="card knowledge-card">
      <div class="card-title"><h2>${escapeHtml(item.title)}</h2><span>${escapeHtml(item.group || '-')}</span></div>
      <p><strong>定义：</strong>${escapeHtml(item.definition || '-')}</p>
      <p><strong>示例：</strong>${escapeHtml((item.examples || []).join('、') || '-')}</p>
      <p><strong>危害：</strong>${escapeHtml(item.hazard || '-')}</p>
      <p><strong>价值：</strong>${escapeHtml(item.value || '-')}</p>
    </article>
  `).join('');
}

function initRecyclePage() {
  const params = new URLSearchParams(location.search);
  const selected = params.get('class_name') || '';
  loadScience().then(rows => {
    $('recycleClassSelect').innerHTML = rows.map(item => `
      <option value="${escapeHtml(item.slug)}">${escapeHtml(item.title)}</option>
    `).join('');
    if (selected) $('recycleClassSelect').value = selected;
  });
  $('searchRecycleBtn').addEventListener('click', () => loadRecyclePoints($('recycleClassSelect').value));
  if (selected) setTimeout(() => loadRecyclePoints(selected), 300);
}

async function ensureGeo() {
  if (state.geo) return state.geo;
  if (!navigator.geolocation) return null;
  return new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(
      pos => {
        state.geo = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        resolve(state.geo);
      },
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
    );
  });
}

async function loadRecyclePoints(className) {
  if (!className) return;
  $('recycleStatus').textContent = '查询中';
  $('recycleResult').className = 'info-list';
  $('recycleResult').innerHTML = '<div class="info-row">正在查询附近回收点...</div>';
  const geo = await ensureGeo();
  const params = new URLSearchParams({ class_name: className });
  if (geo) {
    params.set('lat', geo.lat);
    params.set('lng', geo.lng);
  }
  const info = await api(`/recycle-points?${params.toString()}`);
  $('recycleStatus').textContent = info.provider || '查询完成';
  const points = info.points || [];
  $('recycleResult').innerHTML = `
    <div class="info-row">${escapeHtml(info.tips || '已生成地图查询入口')}</div>
    <div class="actions">
      <a class="btn primary" target="_blank" href="${escapeAttr(info.amapNav || '#')}">高德地图</a>
      <a class="btn" target="_blank" href="${escapeAttr(info.baiduNav || '#')}">百度地图</a>
    </div>
    <div class="point-list mt">
      ${points.length ? points.map(point => `
        <div class="info-row">
          <strong>${escapeHtml(point.name || '-')}</strong><br>
          <span class="muted">${escapeHtml(point.address || '暂无地址')} ${point.distance ? `，约 ${escapeHtml(point.distance)} 米` : ''}</span>
        </div>
      `).join('') : '<div class="info-row muted">暂无真实点位列表，可使用上方地图入口继续查询。</div>'}
    </div>
  `;
}

function containRect(viewW, viewH, sourceW, sourceH) {
  const scale = Math.min(viewW / sourceW, viewH / sourceH);
  return {
    x: (viewW - sourceW * scale) / 2,
    y: (viewH - sourceH * scale) / 2,
    scale,
    scaleY: scale,
  };
}

function drawDetectionBox(ctx, item, fit, sourceW, sourceH, index = 0) {
  if (!Array.isArray(item.bbox) || item.bbox.length !== 4) return;
  let [x1, y1, x2, y2] = item.bbox.map(Number);
  if (x2 <= 1 && y2 <= 1) {
    x1 *= sourceW; x2 *= sourceW; y1 *= sourceH; y2 *= sourceH;
  }
  const sx = fit.scale;
  const sy = fit.scaleY || fit.scale;
  const x = fit.x + x1 * sx;
  const y = fit.y + y1 * sy;
  const w = (x2 - x1) * sx;
  const h = (y2 - y1) * sy;
  if (w <= 0 || h <= 0) return;
  const colors = ['#16a34a', '#2563eb', '#d97706', '#7c3aed'];
  const color = colors[index % colors.length];
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}22`;
  ctx.strokeRect(x, y, w, h);
  ctx.fillRect(x, y, w, h);
  const label = `${item.display_name || item.class_name || '-'} ${item.confidence_text || percent(item.confidence)}`;
  ctx.font = 'bold 14px Microsoft YaHei';
  const tw = ctx.measureText(label).width + 14;
  const ty = Math.max(0, y - 26);
  ctx.fillStyle = color;
  ctx.fillRect(x, ty, tw, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, x + 7, ty + 16);
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatDate(str) {
  const date = new Date(str);
  if (Number.isNaN(date.getTime())) return '-';
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#96;');
}
