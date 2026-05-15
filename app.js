/**
 * RoboVision — Camera-based Object Detection System
 * Uses TensorFlow.js + COCO-SSD model to identify objects via webcam
 */

const video      = document.getElementById('webcam');
const canvas     = document.getElementById('overlay');
const ctx        = canvas.getContext('2d');
const startBtn   = document.getElementById('startBtn');
const pauseBtn   = document.getElementById('pauseBtn');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const fpsDisplay = document.getElementById('fpsDisplay');

// Stats elements
const detectedList  = document.getElementById('detectedList');
const objCount      = document.getElementById('objCount');
const avgConf       = document.getElementById('avgConf');
const inferenceTime = document.getElementById('inferenceTime');
const totalSeen     = document.getElementById('totalSeen');
const logList       = document.getElementById('logList');

// Config
const slider        = document.getElementById('confidenceSlider');
const confValueSpan = document.getElementById('confidenceValue');

let model       = null;
let stream      = null;
let isPaused    = false;
let isDetecting = false;
let totalCount  = 0;

// FPS tracking
let frameCount  = 0;
let lastFpsTime = performance.now();

// Detection history for log
const detectionHistory = new Set();

// Color palette for bounding boxes
const COLORS = ['#00e5ff', '#39ff14', '#ffe600', '#ff3d71', '#bf5af2', '#ff9f0a', '#30d158'];
const objColorMap = {};

function getColor(label) {
  if (!objColorMap[label]) {
    objColorMap[label] = COLORS[Object.keys(objColorMap).length % COLORS.length];
  }
  return objColorMap[label];
}

// ─── Status helpers ─────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusText.textContent = text;
  statusDot.className = 'status-indicator ' + state;
}

// ─── Load model ─────────────────────────────────────────────────────────────
async function loadModel() {
  setStatus('loading', 'LOADING MODEL...');
  try {
    model = await cocoSsd.load();
    setStatus('active', 'MODEL READY');
  } catch (e) {
    setStatus('error', 'MODEL LOAD FAILED');
    console.error(e);
  }
}

// ─── Start camera ────────────────────────────────────────────────────────────
async function startCamera() {
  setStatus('loading', 'REQUESTING CAMERA...');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(res => { video.onloadedmetadata = res; });
    video.play();

    // Sync canvas size to video
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    setStatus('active', 'CAMERA LIVE');
    startBtn.disabled = true;
    pauseBtn.disabled = false;

    if (!model) await loadModel();
    isDetecting = true;
    detect();
  } catch (err) {
    setStatus('error', 'CAMERA ACCESS DENIED');
    console.error(err);
    alert('Camera access was denied or is unavailable.\n\nPlease allow camera permission and try again.');
  }
}

// ─── Main detection loop ─────────────────────────────────────────────────────
async function detect() {
  if (!isDetecting) return;
  if (isPaused) { requestAnimationFrame(detect); return; }

  const t0 = performance.now();

  // Resize canvas to match video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  let predictions = [];
  if (model && video.readyState === 4) {
    try {
      const minScore = slider.value / 100;
      predictions = await model.detect(video, undefined, minScore);
    } catch (e) { /* skip frame */ }
  }

  const elapsed = Math.round(performance.now() - t0);
  inferenceTime.textContent = elapsed + ' ms';

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw predictions
  predictions.forEach(pred => {
    const [x, y, w, h] = pred.bbox;
    const label = pred.class;
    const conf  = Math.round(pred.score * 100);
    const color = getColor(label);

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur  = 0;

    // Corner accents
    drawCornerAccents(ctx, x, y, w, h, color);

    // Label background
    const text   = `${label.toUpperCase()}  ${conf}%`;
    ctx.font     = 'bold 13px "Share Tech Mono", monospace';
    const tw     = ctx.measureText(text).width;
    const lx     = x;
    const ly     = y > 22 ? y - 6 : y + h + 20;

    ctx.fillStyle = 'rgba(5,10,14,0.85)';
    ctx.fillRect(lx, ly - 16, tw + 12, 20);

    // Label border
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1;
    ctx.strokeRect(lx, ly - 16, tw + 12, 20);

    // Label text
    ctx.fillStyle = color;
    ctx.fillText(text, lx + 6, ly - 2);

    // Confidence bar at bottom of box
    ctx.fillStyle = 'rgba(5,10,14,0.6)';
    ctx.fillRect(x, y + h - 4, w, 4);
    ctx.fillStyle = color;
    ctx.fillRect(x, y + h - 4, w * pred.score, 4);
  });

  // Update UI
  updateDetectedList(predictions);
  updateStats(predictions);
  updateLog(predictions);
  updateFPS();

  requestAnimationFrame(detect);
}

// ─── Corner accent drawing ────────────────────────────────────────────────────
function drawCornerAccents(ctx, x, y, w, h, color) {
  const s = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2.5;

  // Top-left
  ctx.beginPath(); ctx.moveTo(x, y + s); ctx.lineTo(x, y); ctx.lineTo(x + s, y); ctx.stroke();
  // Top-right
  ctx.beginPath(); ctx.moveTo(x + w - s, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + s); ctx.stroke();
  // Bottom-left
  ctx.beginPath(); ctx.moveTo(x, y + h - s); ctx.lineTo(x, y + h); ctx.lineTo(x + s, y + h); ctx.stroke();
  // Bottom-right
  ctx.beginPath(); ctx.moveTo(x + w - s, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - s); ctx.stroke();
}

// ─── Update detected objects list ────────────────────────────────────────────
function updateDetectedList(predictions) {
  if (predictions.length === 0) {
    detectedList.innerHTML = '<p class="empty-state">No objects detected</p>';
    return;
  }

  // Group by class and take highest confidence
  const grouped = {};
  predictions.forEach(p => {
    if (!grouped[p.class] || p.score > grouped[p.class]) {
      grouped[p.class] = p.score;
    }
  });

  detectedList.innerHTML = '';
  Object.entries(grouped)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, score]) => {
      const conf = Math.round(score * 100);
      const confClass = conf >= 75 ? 'conf-high' : conf >= 50 ? 'conf-med' : 'conf-low';
      const tag = document.createElement('div');
      tag.className = 'obj-tag';
      tag.innerHTML = `
        <span class="obj-name">${label}</span>
        <span class="obj-conf ${confClass}">${conf}%</span>
      `;
      detectedList.appendChild(tag);
    });
}

// ─── Update stats panel ───────────────────────────────────────────────────────
function updateStats(predictions) {
  objCount.textContent = predictions.length;

  if (predictions.length > 0) {
    const avg = predictions.reduce((s, p) => s + p.score, 0) / predictions.length;
    avgConf.textContent = Math.round(avg * 100) + '%';
  } else {
    avgConf.textContent = '--%';
  }
}

// ─── Detection log ────────────────────────────────────────────────────────────
function updateLog(predictions) {
  predictions.forEach(p => {
    const key = p.class;
    if (!detectionHistory.has(key)) {
      detectionHistory.add(key);
      totalCount++;
      totalSeen.textContent = totalCount;

      const entry   = document.createElement('div');
      entry.className = 'log-entry';
      const now     = new Date();
      const time    = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
      entry.innerHTML = `<span class="log-time">[${time}]</span><span class="log-obj"> NEW: ${p.class.toUpperCase()}</span>`;
      logList.prepend(entry);

      // Keep log manageable
      while (logList.children.length > 30) logList.removeChild(logList.lastChild);

      // Reset after a while so it can be detected again
      setTimeout(() => detectionHistory.delete(key), 5000);
    }
  });
}

// ─── FPS counter ──────────────────────────────────────────────────────────────
function updateFPS() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsDisplay.textContent = frameCount;
    frameCount  = 0;
    lastFpsTime = now;
  }
}

// ─── Event listeners ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', startCamera);

pauseBtn.addEventListener('click', () => {
  isPaused = !isPaused;
  pauseBtn.textContent = isPaused ? 'RESUME' : 'PAUSE';
  setStatus(isPaused ? 'loading' : 'active', isPaused ? 'PAUSED' : 'CAMERA LIVE');
});

slider.addEventListener('input', () => {
  confValueSpan.textContent = slider.value + '%';
});

// ─── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  setStatus('loading', 'LOADING MODEL...');
  try {
    model = await cocoSsd.load();
    setStatus('active', 'MODEL READY — CLICK START');
    document.getElementById('modelName').textContent = 'COCO-SSD v2';
  } catch (e) {
    setStatus('error', 'MODEL FAILED TO LOAD');
    console.error(e);
  }
})();
