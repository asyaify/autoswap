/* ── State ──────────────────────────────────────── */
let currentMode = 'outfit';
let slots = { 1: null, 2: null };
let activePickerSlot = null;
let polling = new Map(); // prompt_id -> card element

const modeConfig = {
  outfit: {
    label: 'Замена одежды',
    slot1: 'Персона',
    slot2: 'Одежда',
    defaultPrompt: 'The person from <image1> wearing the complete outfit from <image2>. Replace ALL of the original clothing with the outfit shown in <image2>. Keep the EXACT same face, facial features, expression, hairstyle, and pose from <image1>. Same background. Full body shot, natural studio lighting, photorealistic.',
    defaultNegative: 'original clothing, different face, altered face, changed expression, different hairstyle, blurry, distorted, artifacts, cartoon, painting, different person'
  },
  background: {
    label: 'Замена фона',
    slot1: 'Персона',
    slot2: 'Фон',
    defaultPrompt: 'Place the person from <image1> naturally into the scene from <image2>. Keep the EXACT same face, facial features, expression, hairstyle, clothing and pose from <image1>. The person should be naturally integrated into the environment, correct proportions and scale. Professional photo, natural lighting, photorealistic.',
    defaultNegative: 'different face, altered face, changed expression, different hairstyle, different clothing, floating, wrong proportions, too small, too large, blurry, distorted, artifacts, cartoon, painting, different person'
  },
  facefix: {
    label: 'Коррекция лица',
    slot1: 'Результат (тело)',
    slot2: 'Оригинал (лицо)',
    defaultPrompt: 'head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2. ensure the head and body have correct anatomical proportions, and blend the skin tones, shadows, and lighting naturally so the final result appears as one coherent, realistic person.',
    defaultNegative: 'different body, different clothing, different background, different lighting, blurry, distorted, artifacts, wrong proportions'
  }
};

/* ── Init ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkComfyUI();
  setInterval(checkComfyUI, 15000);
  loadGallery();

  // Textarea auto-resize
  const ta = document.getElementById('promptInput');
  ta.addEventListener('input', () => autoResizeTextarea(ta));

  // Enter to send (Shift+Enter for newline)
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      generate();
    }
  });

  // Sidebar nav
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  // Upload zone
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.style.borderColor = 'var(--accent)'; });
  uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor = ''; });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = '';
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
    fileInput.value = '';
  });
});

/* ── Mode switching ──────────────────────────────── */
function setMode(mode) {
  currentMode = mode;
  const cfg = modeConfig[mode];

  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  document.getElementById('slot1label').textContent = cfg.slot1;
  document.getElementById('slot2label').textContent = cfg.slot2;

  // Set default prompt: replace if empty or if it matches another mode's default
  const ta = document.getElementById('promptInput');
  const otherDefaults = Object.values(modeConfig).map(c => c.defaultPrompt);
  if (!ta.value.trim() || otherDefaults.includes(ta.value.trim())) {
    ta.value = cfg.defaultPrompt;
  }
  autoResizeTextarea(ta);

  // Update negative prompt default per mode
  const negInput = document.getElementById('negativeInput');
  if (negInput) negInput.value = cfg.defaultNegative || 'blurry, distorted, artifacts';

  // Hide welcome
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
}

/* ── Image Slots ─────────────────────────────────── */
function setSlot(num, filename) {
  slots[num] = filename;
  const img = document.getElementById(`slot${num}img`);
  const ph = document.getElementById(`slot${num}placeholder`);
  const rm = document.getElementById(`slot${num}remove`);
  const slot = document.getElementById(`slot${num}`);

  img.src = `/api/image/input/${encodeURIComponent(filename)}`;
  img.style.display = 'block';
  ph.style.display = 'none';
  rm.style.display = 'flex';
  slot.classList.add('filled');
}

function clearSlot(num) {
  slots[num] = null;
  const img = document.getElementById(`slot${num}img`);
  const ph = document.getElementById(`slot${num}placeholder`);
  const rm = document.getElementById(`slot${num}remove`);
  const slot = document.getElementById(`slot${num}`);

  img.style.display = 'none';
  ph.style.display = 'flex';
  rm.style.display = 'none';
  slot.classList.remove('filled');
}

/* ── Image Picker ────────────────────────────────── */
function openPicker(slot) {
  activePickerSlot = slot;
  document.getElementById('pickerModal').style.display = 'flex';
  loadLibrary();
}

function closePicker() {
  document.getElementById('pickerModal').style.display = 'none';
  activePickerSlot = null;
}

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.getElementById('tabLibrary').style.display = tab === 'library' ? 'block' : 'none';
  document.getElementById('tabOutputs').style.display = tab === 'outputs' ? 'block' : 'none';
  document.getElementById('tabUpload').style.display = tab === 'upload' ? 'block' : 'none';
  if (tab === 'outputs') loadOutputPicker();
}

async function loadLibrary() {
  const grid = document.getElementById('libraryGrid');
  grid.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Загрузка…</p>';
  try {
    const resp = await fetch('/api/images/input');
    const files = await resp.json();
    grid.innerHTML = '';
    files.forEach(f => {
      const div = document.createElement('div');
      div.className = 'image-grid-item';
      div.innerHTML = `<img src="/api/image/input/${encodeURIComponent(f)}" loading="lazy"><div class="img-name">${f}</div>`;
      div.addEventListener('click', () => {
        setSlot(activePickerSlot, f);
        closePicker();
      });
      grid.appendChild(div);
    });
  } catch (e) {
    grid.innerHTML = '<p style="color:var(--error);font-size:13px">Не удалось загрузить</p>';
  }
}

async function loadOutputPicker() {
  const grid = document.getElementById('outputsGrid');
  grid.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Загрузка…</p>';
  try {
    const resp = await fetch('/api/images/output');
    const files = await resp.json();
    grid.innerHTML = '';
    files.forEach(f => {
      const div = document.createElement('div');
      div.className = 'image-grid-item';
      div.innerHTML = `<img src="/api/image/output/${encodeURIComponent(f)}" loading="lazy"><div class="img-name">${f}</div>`;
      div.addEventListener('click', () => useOutputAsInput(f));
      grid.appendChild(div);
    });
  } catch (e) {
    grid.innerHTML = '<p style="color:var(--error);font-size:13px">Не удалось загрузить</p>';
  }
}

async function useOutputAsInput(filename) {
  try {
    const resp = await fetch('/api/use_output', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename })
    });
    const data = await resp.json();
    if (data.filename) {
      setSlot(activePickerSlot, data.filename);
      closePicker();
    }
  } catch (e) {
    alert('Ошибка копирования файла');
  }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.filename) {
      setSlot(activePickerSlot, data.filename);
      closePicker();
    }
  } catch (e) {
    alert('Ошибка загрузки файла');
  }
}

/* ── Advanced Panel ──────────────────────────────── */
function toggleAdvanced() {
  const panel = document.getElementById('advancedPanel');
  const icon = document.getElementById('advToggleIcon');
  const visible = panel.style.display !== 'none';
  panel.style.display = visible ? 'none' : 'block';
  icon.textContent = visible ? '▸' : '▾';
}

/* ── Generate ────────────────────────────────────── */
async function generate() {
  const prompt = document.getElementById('promptInput').value.trim();
  if (!slots[1] || !slots[2]) {
    shakeSlots();
    return;
  }
  if (!prompt) {
    document.getElementById('promptInput').focus();
    return;
  }

  // Hide welcome
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  const negative = document.getElementById('negativeInput').value;
  const seed = document.getElementById('seedInput').value;
  const prefix = document.getElementById('prefixInput').value || 'web_ui';
  const quality = document.getElementById('qualitySelect').value;
  const megapixels = document.getElementById('megapixelsSelect').value;
  const stepsMap = { fast: 4, balanced: 8, hq: 20 };

  const payload = {
    mode: currentMode,
    image1: slots[1],
    image2: slots[2],
    prompt,
    negative,
    seed: seed || null,
    prefix,
    quality,
    megapixels: parseFloat(megapixels),
    steps: stepsMap[quality] || 4
  };

  // Create result card
  const card = createResultCard(payload);
  const feed = document.getElementById('resultsFeed');
  feed.insertBefore(card, feed.firstChild);

  const btn = document.getElementById('sendBtn');
  btn.disabled = true;

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (data.error) {
      showCardError(card, data.error);
      btn.disabled = false;
      return;
    }
    card.dataset.promptId = data.prompt_id;
    card.dataset.seed = data.seed;
    startPolling(data.prompt_id, card);
  } catch (e) {
    showCardError(card, 'Нет связи с сервером');
    btn.disabled = false;
  }
}

function createResultCard(payload) {
  const cfg = modeConfig[payload.mode];
  const card = document.createElement('div');
  card.className = 'result-card';

  const now = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
  card.innerHTML = `
    <div class="result-header">
      <span class="result-mode">${cfg.label}</span>
      <span class="result-time">${now}</span>
    </div>
    <div class="result-prompt">${escapeHtml(payload.prompt).substring(0, 200)}${payload.prompt.length > 200 ? '…' : ''}</div>
    <div class="result-images-row">
      <img class="result-input-thumb" src="/api/image/input/${encodeURIComponent(payload.image1)}">
      <img class="result-input-thumb" src="/api/image/input/${encodeURIComponent(payload.image2)}">
    </div>
    <div class="result-body">
      <div class="result-status">
        <div class="spinner"></div>
        <span class="result-status-text">Генерация…</span>
      </div>
    </div>
  `;
  return card;
}

function showCardError(card, msg) {
  card.querySelector('.result-body').innerHTML = `<div class="result-error">⚠ ${escapeHtml(msg)}</div>`;
}

function showCardResult(card, images) {
  const body = card.querySelector('.result-body');
  body.innerHTML = '';
  images.forEach(img => {
    const el = document.createElement('img');
    el.className = 'result-output';
    el.src = `/api/image/output/${encodeURIComponent(img)}`;
    el.addEventListener('click', () => openLightbox(el.src));
    body.appendChild(el);

    // Add "Fix face" button under each result
    const actions = document.createElement('div');
    actions.className = 'result-actions';
    actions.innerHTML = `<button class="action-btn" title="Использовать для коррекции лица">👤 Исправить лицо</button>`;
    actions.querySelector('.action-btn').addEventListener('click', () => startFaceFix(img));
    body.appendChild(actions);
  });
  loadGallery();
}

async function startFaceFix(outputFilename) {
  // Switch to facefix mode and set slot1 to this output image
  setMode('facefix');
  try {
    const resp = await fetch('/api/use_output', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: outputFilename })
    });
    const data = await resp.json();
    if (data.filename) {
      setSlot(1, data.filename);
    }
  } catch (e) {
    alert('Ошибка');
  }
}

/* ── Polling ─────────────────────────────────────── */
function startPolling(promptId, card) {
  const statusText = card.querySelector('.result-status-text');
  let seconds = 0;

  const interval = setInterval(async () => {
    seconds += 3;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    statusText.textContent = `Генерация… ${mins > 0 ? mins + 'м ' : ''}${secs}с`;

    try {
      const resp = await fetch(`/api/status/${promptId}`);
      const data = await resp.json();

      if (data.status === 'done') {
        clearInterval(interval);
        document.getElementById('sendBtn').disabled = false;
        showCardResult(card, data.images);
      } else if (data.status === 'error') {
        clearInterval(interval);
        document.getElementById('sendBtn').disabled = false;
        showCardError(card, data.error || 'Ошибка генерации');
      }
    } catch (e) {
      // Network error, keep polling
    }
  }, 3000);
}

/* ── Gallery ─────────────────────────────────────── */
async function loadGallery() {
  try {
    const resp = await fetch('/api/images/output');
    const files = await resp.json();
    const grid = document.getElementById('gallery');
    grid.innerHTML = '';
    files.slice(0, 12).forEach(f => {
      const img = document.createElement('img');
      img.src = `/api/image/output/${encodeURIComponent(f)}`;
      img.loading = 'lazy';
      img.title = f;
      img.addEventListener('click', () => openLightbox(img.src));
      grid.appendChild(img);
    });
  } catch (e) {}
}

/* ── ComfyUI Status ──────────────────────────────── */
async function checkComfyUI() {
  try {
    const resp = await fetch('/api/comfyui/status');
    const data = await resp.json();
    const dot = document.querySelector('.status-dot');
    dot.classList.toggle('offline', !data.online);
  } catch (e) {
    document.querySelector('.status-dot').classList.add('offline');
  }
}

/* ── Lightbox ────────────────────────────────────── */
function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
  document.getElementById('lightbox').style.display = 'none';
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeLightbox();
    closePicker();
  }
});

/* ── Helpers ─────────────────────────────────────── */
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function autoResizeTextarea(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 240) + 'px';
}

function shakeSlots() {
  [1, 2].forEach(n => {
    if (!slots[n]) {
      const el = document.getElementById(`slot${n}`);
      el.style.animation = 'none';
      el.offsetHeight;
      el.style.animation = 'shake 0.4s ease';
    }
  });
}

// Shake keyframes (added dynamically)
const style = document.createElement('style');
style.textContent = `@keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-4px)} 75%{transform:translateX(4px)} }`;
document.head.appendChild(style);
