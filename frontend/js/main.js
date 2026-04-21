
const API = '/api';
let currentImageFile = null;
let currentVideoFile = null;
let cameraStream = null;
let cameraTimer = null;
let frameBusy = false;
const sessionId = `cam_${Date.now()}`;
const overlayColors = ['#19c37d', '#ff7a59', '#6f5cff', '#f7b500', '#ef476f', '#06d6a0', '#118ab2'];

const el = (id) => document.getElementById(id);
const has = (id) => Boolean(el(id));

function renderDetections(containerId, detections = []) {
  const box = el(containerId);
  if (!box) return;
  box.innerHTML = '';
  if (!detections.length) {
    box.innerHTML = '<div class="detection-item">未检测到目标。</div>';
    return;
  }
  detections.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'detection-item';
    div.innerHTML = `
      <strong>${item.class_name_zh || item.class_name}</strong>
      <div>垃圾大类：${item.category_group || '--'}</div>
      <div>置信度：${item.confidence_text || ((item.confidence || 0) * 100).toFixed(2) + '%'}</div>
      <div>建议：${item.advice || '--'}</div>
      <div>Track ID：${item.track_id ?? '--'} / 稳定计数：${item.stable_count ?? '--'}</div>
      ${item.hit_count ? `<div>视频累计出现：${item.hit_count} 次</div>` : ''}
    `;
    box.appendChild(div);
  });
}

function setSummary(id, text) {
  if (!has(id)) return;
  el(id).textContent = text || '暂无结果';
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';
  const text = await response.text();

  if (!contentType.includes('application/json')) {
    throw new Error(`接口未返回 JSON，请确认前端是通过 Node 后端访问，当前返回：${text.slice(0, 120)}`);
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new Error(`JSON 解析失败：${text.slice(0, 120)}`);
  }

  if (!response.ok || data.code !== 200) {
    throw new Error(data.msg || '请求失败');
  }
  return data.data ?? null;
}

function drawOverlay(detections = []) {
  const video = el('cameraVideo');
  const canvas = el('overlayCanvas');
  if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;

  detections.forEach((det, index) => {
    const color = overlayColors[(det.track_id || index) % overlayColors.length];
    const [x1, y1, x2, y2] = det.bbox || [];
    if ([x1, y1, x2, y2].some((v) => Number.isNaN(Number(v)))) return;
    const sx = x1 * scaleX;
    const sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, sy, sw, sh);
    ctx.fillStyle = color;
    const label = `${det.class_name_zh || det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
    ctx.fillRect(sx, Math.max(0, sy - 22), Math.max(110, label.length * 8), 22);
    ctx.fillStyle = '#111';
    ctx.font = '12px sans-serif';
    ctx.fillText(label, sx + 6, Math.max(14, sy - 7));
  });
}

if (has('imageInput')) {
  el('imageInput').addEventListener('change', (e) => {
    currentImageFile = e.target.files?.[0] || null;
    if (!currentImageFile) return;
    const reader = new FileReader();
    reader.onload = () => { if (has('imagePreview')) el('imagePreview').src = reader.result; };
    reader.readAsDataURL(currentImageFile);
  });
}

if (has('imageDetectBtn')) {
  el('imageDetectBtn').addEventListener('click', async () => {
    if (!currentImageFile) return alert('请先选择图片');
    const form = new FormData();
    form.append('image', currentImageFile);
    setSummary('imageSummary', '图片识别中...');
    try {
      const data = await api(`${API}/predict/image`, { method: 'POST', body: form });
      setSummary('imageSummary', data.summary ? `主目标：${data.summary.class_name_zh} / ${data.summary.confidence_text} / ${data.summary.category_group}` : '未检测到目标');
      if (data.saved_path && has('imageResult')) el('imageResult').src = data.saved_path;
      renderDetections('imageDetections', data.detections);
      if (data.summary?.map_keyword && has('recycleKeyword')) el('recycleKeyword').value = data.summary.map_keyword;
    } catch (error) {
      setSummary('imageSummary', error.message);
    }
  });
}

if (has('videoInput')) {
  el('videoInput').addEventListener('change', (e) => {
    currentVideoFile = e.target.files?.[0] || null;
  });
}

if (has('videoDetectBtn')) {
  el('videoDetectBtn').addEventListener('click', async () => {
    if (!currentVideoFile) return alert('请先选择视频');
    const form = new FormData();
    form.append('video', currentVideoFile);
    form.append('frame_stride', has('videoStride') ? (el('videoStride').value || '2') : '2');
    setSummary('videoSummary', '视频分析中，可能需要几分钟...');
    try {
      const data = await api(`${API}/predict/video`, { method: 'POST', body: form });
      setSummary('videoSummary', `视频分析完成，原视频FPS=${data.fps || '--'}，处理速度=${data.avg_process_fps || '--'} FPS`);
      if (data.saved_path && has('videoResult')) el('videoResult').src = data.saved_path;
      renderDetections('videoDetections', data.detections);
    } catch (error) {
      setSummary('videoSummary', error.message);
    }
  });
}

async function startCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  const video = el('cameraVideo');
  video.srcObject = cameraStream;
  await video.play();
  await api(`${API}/tracker/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  });
  if (has('startCameraBtn')) el('startCameraBtn').disabled = true;
  if (has('stopCameraBtn')) el('stopCameraBtn').disabled = false;
  if (has('resetTrackerBtn')) el('resetTrackerBtn').disabled = false;
  cameraTimer = setInterval(sendFrameForDetect, 600);
}

async function stopCamera() {
  if (cameraTimer) clearInterval(cameraTimer);
  cameraTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  drawOverlay([]);
  if (has('startCameraBtn')) el('startCameraBtn').disabled = false;
  if (has('stopCameraBtn')) el('stopCameraBtn').disabled = true;
  if (has('resetTrackerBtn')) el('resetTrackerBtn').disabled = true;
  setSummary('cameraSummary', '摄像头已关闭');
}

async function sendFrameForDetect() {
  if (frameBusy || !cameraStream) return;
  frameBusy = true;
  try {
    const video = el('cameraVideo');
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL('image/jpeg', 0.75);
    const data = await api(`${API}/predict/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base64, session_id: sessionId })
    });
    drawOverlay(data.detections || []);
    setSummary('cameraSummary', data.summary ? `主目标：${data.summary.class_name_zh || data.summary.class_name} / ${data.summary.confidence_text} / 稳定计数 ${data.summary.stable_count || 0}` : `稳定目标数：${data.stable_count || 0}`);
    renderDetections('cameraDetections', data.detections);
  } catch (error) {
    setSummary('cameraSummary', error.message);
  } finally {
    frameBusy = false;
  }
}

if (has('startCameraBtn')) el('startCameraBtn').addEventListener('click', () => startCamera().catch((err) => alert(err.message)));
if (has('stopCameraBtn')) el('stopCameraBtn').addEventListener('click', stopCamera);
if (has('resetTrackerBtn')) {
  el('resetTrackerBtn').addEventListener('click', async () => {
    await api(`${API}/tracker/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    setSummary('cameraSummary', '跟踪状态已重置');
  });
}

async function loadKnowledge() {
  if (!has('knowledgeGrid')) return;
  const q = has('knowledgeSearch') ? el('knowledgeSearch').value.trim() : '';
  const data = await api(`${API}/knowledge?q=${encodeURIComponent(q)}`);
  const grid = el('knowledgeGrid');
  grid.innerHTML = '';
  if (!data.length) {
    grid.innerHTML = '<div class="empty-state">没有匹配到相关科普内容。</div>';
    return;
  }
  data.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'knowledge-card';
    div.innerHTML = `
      <h3>${item.name}</h3>
      <div>大类：${item.group}</div>
      <p>${item.definition}</p>
      <p><strong>示例：</strong>${(item.examples || []).join('、')}</p>
      <p><strong>危害：</strong>${item.hazard || '--'}</p>
      <p><strong>价值：</strong>${item.value || '--'}</p>
      <p><strong>回收建议：</strong>${item.advice}</p>
    `;
    grid.appendChild(div);
  });
}

async function searchRecyclePoints() {
  if (!has('recycleKeyword') || !has('recyclePoints')) return;
  const keyword = el('recycleKeyword').value.trim();
  const city = has('recycleCity') ? el('recycleCity').value.trim() : '';
  if (!keyword) return alert('请输入查询关键词');
  if (has('recyclePointTips')) el('recyclePointTips').textContent = '查询中...';
  try {
    const rows = await api(`${API}/recycle-points?keyword=${encodeURIComponent(keyword)}&city=${encodeURIComponent(city)}`);
    const box = el('recyclePoints');
    box.innerHTML = '';
    if (!rows.length) {
      box.innerHTML = '<div class="knowledge-card">当前未查询到结果。若已配置高德 Web Key，请检查关键词或城市。</div>';
    } else {
      rows.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'knowledge-card';
        div.innerHTML = `
          <h3>${item.name || '未知点位'}</h3>
          <p><strong>地址：</strong>${item.address || '--'}</p>
          <p><strong>类型：</strong>${item.type || '--'}</p>
          <p><strong>位置：</strong>${item.location || '--'}</p>
        `;
        box.appendChild(div);
      });
    }
    if (has('recyclePointTips')) el('recyclePointTips').textContent = `共返回 ${rows.length} 个结果`;
  } catch (error) {
    if (has('recyclePointTips')) el('recyclePointTips').textContent = error.message;
  }
}

async function loadHistory() {
  if (!has('historyTableBody')) return;
  const keyword = has('historyKeyword') ? el('historyKeyword').value.trim() : '';
  const rows = await api(`${API}/history?keyword=${encodeURIComponent(keyword)}`);
  const tbody = el('historyTableBody');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7">暂无历史记录</td></tr>';
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.identify_time || '--'}</td>
      <td>${row.display_name || row.class_name}</td>
      <td>${row.category_group || '--'}</td>
      <td>${row.confidence_text || '--'}</td>
      <td>${row.detect_mode || '--'}</td>
      <td>${row.advice || '--'}</td>
      <td><button class="btn btn-danger" data-id="${row.id}">删除</button></td>
    `;
    tr.querySelector('button').addEventListener('click', async () => {
      try {
        await api(`${API}/history/${row.id}`, { method: 'DELETE' });
        await loadHistory();
      } catch (error) {
        alert(error.message);
      }
    });
    tbody.appendChild(tr);
  });
}

if (has('knowledgeSearch')) el('knowledgeSearch').addEventListener('input', () => loadKnowledge().catch(console.error));
if (has('searchRecycleBtn')) el('searchRecycleBtn').addEventListener('click', () => searchRecyclePoints().catch(console.error));
if (has('refreshHistoryBtn')) el('refreshHistoryBtn').addEventListener('click', () => loadHistory().catch(console.error));
if (has('clearHistoryBtn')) {
  el('clearHistoryBtn').addEventListener('click', async () => {
    try {
      await api(`${API}/history`, { method: 'DELETE' });
      await loadHistory();
    } catch (error) {
      alert(error.message);
    }
  });
}

loadKnowledge().catch(() => {});
loadHistory().catch(() => {});
