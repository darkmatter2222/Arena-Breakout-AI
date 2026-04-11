const express = require('express');
const cors = require('cors');
const http = require('http');
const { WebSocketServer, WebSocket } = require('ws');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ── Config ──────────────────────────────────────────────────────────────
const PORT = 3001;
const VISION_ROOT = path.resolve(__dirname, '..', '..');
const DATASET_DIR = path.join(VISION_ROOT, 'dataset');
const RUNS_DIR = path.join(VISION_ROOT, 'runs');
const VENV_PYTHON = path.join(VISION_ROOT, 'venv', 'Scripts', 'python.exe');
const WORKER_SCRIPT = path.join(__dirname, 'inference_worker.py');

// ── Express + WebSocket ─────────────────────────────────────────────────
const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ── Utilities ───────────────────────────────────────────────────────────
function boxIou(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter === 0) return 0;
  const union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return union > 0 ? inter / union : 0;
}

function hasOverlaps(boxes) {
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++)
      if (boxIou(boxes[i], boxes[j]) > 0) return true;
  return false;
}

function parseLabels(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, 'utf8').trim();
  if (!text) return [];
  return text.split('\n').map(line => {
    const p = line.trim().split(/\s+/);
    return { cls: +p[0], cx: +p[1], cy: +p[2], w: +p[3], h: +p[4] };
  }).filter(b => !isNaN(b.cx));
}

function labelsToString(labels) {
  return labels.map(b => `${b.cls} ${b.cx} ${b.cy} ${b.w} ${b.h}`).join('\n');
}

function parseResultsCsv(csvPath) {
  const lines = fs.readFileSync(csvPath, 'utf8').split('\n').filter(l => l.trim());
  if (lines.length < 2) return null;
  const headers = lines[0].split(',').map(h => h.trim());
  const mi = headers.indexOf('metrics/mAP50(B)');
  const pi = headers.indexOf('metrics/precision(B)');
  const ri = headers.indexOf('metrics/recall(B)');
  const m95i = headers.indexOf('metrics/mAP50-95(B)');
  let best = { epoch: 0, mAP50: 0, precision: 0, recall: 0, mAP50_95: 0 };
  const history = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    const entry = {
      epoch: parseInt(cols[0]),
      precision: parseFloat(cols[pi]),
      recall: parseFloat(cols[ri]),
      mAP50: parseFloat(cols[mi]),
      mAP50_95: parseFloat(cols[m95i]),
    };
    history.push(entry);
    if (entry.mAP50 > best.mAP50) best = { ...entry };
  }
  return { totalEpochs: history.length, best, last: history[history.length - 1], history };
}

// ── Image Index (cached) ────────────────────────────────────────────────
let imageIndex = null;
let indexTimestamp = 0;

function buildImageIndex() {
  imageIndex = [];
  for (const split of ['train', 'val']) {
    const imgDir = path.join(DATASET_DIR, 'images', split);
    const lblDir = path.join(DATASET_DIR, 'labels', split);
    if (!fs.existsSync(imgDir)) continue;
    for (const file of fs.readdirSync(imgDir).sort()) {
      if (!/\.(jpg|jpeg|png)$/i.test(file)) continue;
      const stem = path.parse(file).name;
      const lblPath = path.join(lblDir, stem + '.txt');
      const boxes = parseLabels(lblPath);
      imageIndex.push({
        filename: file, split, boxCount: boxes.length,
        hasLabel: fs.existsSync(lblPath),
        hasOverlaps: boxes.length >= 2 && hasOverlaps(boxes),
        hasTiny: boxes.some(b => b.w < 0.02 || b.h < 0.02),
      });
    }
  }
  indexTimestamp = Date.now();
  return imageIndex;
}

function getIndex() {
  if (!imageIndex || Date.now() - indexTimestamp > 30000) buildImageIndex();
  return imageIndex;
}

// ── Inference Bridge ────────────────────────────────────────────────────
class InferenceBridge {
  constructor() { this.proc = null; this.pending = new Map(); this.nextId = 1; this.ready = false; }

  start() {
    return new Promise((resolve) => {
      this.proc = spawn(VENV_PYTHON, [WORKER_SCRIPT], { stdio: ['pipe', 'pipe', 'pipe'], cwd: VISION_ROOT });
      const rl = readline.createInterface({ input: this.proc.stdout });
      rl.on('line', (line) => {
        try {
          const msg = JSON.parse(line);
          if (msg.ready) { this.ready = true; resolve(); return; }
          const p = this.pending.get(msg.id);
          if (p) { this.pending.delete(msg.id); msg.error ? p.reject(new Error(msg.error)) : p.resolve(msg.result); }
        } catch (e) { /* ignore parse errors from YOLO output */ }
      });
      this.proc.stderr.on('data', () => {}); // suppress YOLO warnings
      this.proc.on('exit', () => {
        this.ready = false;
        for (const [, p] of this.pending) p.reject(new Error('Worker exited'));
        this.pending.clear();
        setTimeout(() => this.start(), 3000);
      });
      setTimeout(() => { if (!this.ready) resolve(); }, 15000); // timeout
    });
  }

  predict(modelName, modelPath, imagePath, conf = 0.5) {
    if (!this.ready) return Promise.reject(new Error('Worker not ready'));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { this.pending.delete(id); reject(new Error('Timeout')); }, 60000);
      this.pending.set(id, {
        resolve: (r) => { clearTimeout(timeout); resolve(r); },
        reject: (e) => { clearTimeout(timeout); reject(e); },
      });
      this.proc.stdin.write(JSON.stringify({ id, action: 'predict', model: modelName, modelPath, imagePath, conf }) + '\n');
    });
  }
}

const bridge = new InferenceBridge();

// ── Training Manager ────────────────────────────────────────────────────
class TrainingManager {
  constructor() { this.proc = null; this.status = 'idle'; this.timer = null; this.crashCount = 0; }

  getStatus() {
    const csvPath = path.join(RUNS_DIR, 'yolo11n_target', 'results.csv');
    let progress = null;
    if (fs.existsSync(csvPath)) progress = parseResultsCsv(csvPath);
    return { status: this.status, progress, crashCount: this.crashCount };
  }

  start() {
    if (this.status === 'training') return { error: 'Already training' };
    this.status = 'training'; this.crashCount = 0;
    this._spawn();
    this._startMonitor();
    return { status: 'started' };
  }

  _spawn() {
    this.proc = spawn(VENV_PYTHON, [path.join(VISION_ROOT, 'train.py')], { cwd: VISION_ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    this.proc.stdout.on('data', () => {});
    this.proc.stderr.on('data', () => {});
    this.proc.on('exit', (code) => {
      if (this.status === 'stopping') { this.status = 'idle'; this._stopMonitor(); return; }
      if (code !== 0) {
        this.crashCount++;
        this._broadcast({ type: 'training_crash', crashCount: this.crashCount });
        setTimeout(() => { if (this.status === 'training') this._spawn(); }, 3000);
      } else {
        this.status = 'completed';
        this._stopMonitor();
        this._broadcast({ type: 'training_complete', ...this.getStatus() });
      }
    });
  }

  stop() { this.status = 'stopping'; if (this.proc) this.proc.kill(); this._stopMonitor(); }

  _startMonitor() {
    this.timer = setInterval(() => {
      this._broadcast({ type: 'training_progress', ...this.getStatus() });
    }, 5000);
  }

  _stopMonitor() { if (this.timer) { clearInterval(this.timer); this.timer = null; } }

  _broadcast(msg) {
    const data = JSON.stringify(msg);
    wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(data); });
  }
}

const trainer = new TrainingManager();

// ═══════════════════════════════════════════════════════════════════════
//  ROUTES: Images
// ═══════════════════════════════════════════════════════════════════════

app.get('/api/images', (req, res) => {
  const { page = 1, perPage = 60, split = 'all', labeled = 'all', overlaps = 'all', tiny = 'all', sort = 'name', search = '' } = req.query;
  let items = [...getIndex()];

  if (split !== 'all') items = items.filter(i => i.split === split);
  if (labeled === 'true') items = items.filter(i => i.hasLabel);
  if (labeled === 'false') items = items.filter(i => !i.hasLabel);
  if (overlaps === 'true') items = items.filter(i => i.hasOverlaps);
  if (tiny === 'true') items = items.filter(i => i.hasTiny);
  if (search) items = items.filter(i => i.filename.toLowerCase().includes(search.toLowerCase()));

  if (sort === 'boxes') items.sort((a, b) => b.boxCount - a.boxCount);
  else items.sort((a, b) => a.filename.localeCompare(b.filename));

  const total = items.length;
  const p = Math.max(1, parseInt(page));
  const pp = Math.max(1, Math.min(200, parseInt(perPage)));
  const start = (p - 1) * pp;
  const pageItems = items.slice(start, start + pp);

  res.json({ items: pageItems, total, page: p, perPage: pp, pages: Math.ceil(total / pp) });
});

app.get('/api/images/:split/:filename', (req, res) => {
  const { split, filename } = req.params;
  const imgPath = path.join(DATASET_DIR, 'images', split, filename);
  if (!fs.existsSync(imgPath)) return res.status(404).json({ error: 'Not found' });
  const stem = path.parse(filename).name;
  const lblPath = path.join(DATASET_DIR, 'labels', split, stem + '.txt');
  const labels = parseLabels(lblPath);

  // Find neighbors in the full index
  const idx = getIndex();
  const pos = idx.findIndex(i => i.split === split && i.filename === filename);
  const prev = pos > 0 ? { split: idx[pos - 1].split, filename: idx[pos - 1].filename } : null;
  const next = pos < idx.length - 1 ? { split: idx[pos + 1].split, filename: idx[pos + 1].filename } : null;

  res.json({ filename, split, labels, prev, next });
});

app.get('/api/images/:split/:filename/file', (req, res) => {
  const filePath = path.join(DATASET_DIR, 'images', req.params.split, req.params.filename);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Not found' });
  res.sendFile(filePath);
});

app.put('/api/images/:split/:filename/labels', (req, res) => {
  const { split, filename } = req.params;
  const { labels } = req.body;
  const stem = path.parse(filename).name;
  const lblDir = path.join(DATASET_DIR, 'labels', split);
  if (!fs.existsSync(lblDir)) fs.mkdirSync(lblDir, { recursive: true });
  const lblPath = path.join(lblDir, stem + '.txt');
  fs.writeFileSync(lblPath, labelsToString(labels));
  imageIndex = null; // invalidate cache
  res.json({ saved: true });
});

app.delete('/api/images/:split/:filename', (req, res) => {
  const { split, filename } = req.params;
  const imgPath = path.join(DATASET_DIR, 'images', split, filename);
  const stem = path.parse(filename).name;
  const lblPath = path.join(DATASET_DIR, 'labels', split, stem + '.txt');
  if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
  if (fs.existsSync(lblPath)) fs.unlinkSync(lblPath);
  imageIndex = null;
  res.json({ deleted: true });
});

app.post('/api/images/bulk-delete', (req, res) => {
  const { items } = req.body;
  let deleted = 0;
  for (const { split, filename } of items) {
    const imgPath = path.join(DATASET_DIR, 'images', split, filename);
    const stem = path.parse(filename).name;
    const lblPath = path.join(DATASET_DIR, 'labels', split, stem + '.txt');
    if (fs.existsSync(imgPath)) { fs.unlinkSync(imgPath); deleted++; }
    if (fs.existsSync(lblPath)) fs.unlinkSync(lblPath);
  }
  imageIndex = null;
  res.json({ deleted });
});

app.post('/api/images/bulk-move', (req, res) => {
  const { items, targetSplit } = req.body;
  let moved = 0;
  const targetImgDir = path.join(DATASET_DIR, 'images', targetSplit);
  const targetLblDir = path.join(DATASET_DIR, 'labels', targetSplit);
  if (!fs.existsSync(targetImgDir)) fs.mkdirSync(targetImgDir, { recursive: true });
  if (!fs.existsSync(targetLblDir)) fs.mkdirSync(targetLblDir, { recursive: true });
  for (const { split, filename } of items) {
    if (split === targetSplit) continue;
    const stem = path.parse(filename).name;
    const srcImg = path.join(DATASET_DIR, 'images', split, filename);
    const srcLbl = path.join(DATASET_DIR, 'labels', split, stem + '.txt');
    if (fs.existsSync(srcImg)) { fs.renameSync(srcImg, path.join(targetImgDir, filename)); moved++; }
    if (fs.existsSync(srcLbl)) fs.renameSync(srcLbl, path.join(targetLblDir, stem + '.txt'));
  }
  imageIndex = null;
  res.json({ moved });
});

// ═══════════════════════════════════════════════════════════════════════
//  ROUTES: Models
// ═══════════════════════════════════════════════════════════════════════

app.get('/api/models', (req, res) => {
  const models = [];
  if (!fs.existsSync(RUNS_DIR)) return res.json(models);
  for (const dir of fs.readdirSync(RUNS_DIR).sort()) {
    const bestPt = path.join(RUNS_DIR, dir, 'weights', 'best.pt');
    if (!fs.existsSync(bestPt)) continue;
    const csvPath = path.join(RUNS_DIR, dir, 'results.csv');
    let metrics = null;
    if (fs.existsSync(csvPath)) metrics = parseResultsCsv(csvPath);
    models.push({ name: dir, weightsPath: bestPt, metrics });
  }
  res.json(models);
});

app.post('/api/models/:name/predict', async (req, res) => {
  try {
    const modelName = req.params.name;
    const { split, filename, conf = 0.5 } = req.body;
    const modelsDir = fs.readdirSync(RUNS_DIR);
    const modelDir = modelsDir.find(d => d === modelName);
    if (!modelDir) return res.status(404).json({ error: 'Model not found' });
    const weightsPath = path.join(RUNS_DIR, modelDir, 'weights', 'best.pt');
    const imagePath = path.join(DATASET_DIR, 'images', split, filename);
    if (!fs.existsSync(imagePath)) return res.status(404).json({ error: 'Image not found' });
    const detections = await bridge.predict(modelName, weightsPath, imagePath, conf);
    res.json({ detections });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/models/:name/batch-predict', async (req, res) => {
  try {
    const modelName = req.params.name;
    const { items, conf = 0.5 } = req.body;
    const weightsPath = path.join(RUNS_DIR, modelName, 'weights', 'best.pt');
    if (!fs.existsSync(weightsPath)) return res.status(404).json({ error: 'Model not found' });
    const results = [];
    for (const { split, filename } of items) {
      const imagePath = path.join(DATASET_DIR, 'images', split, filename);
      if (!fs.existsSync(imagePath)) { results.push({ split, filename, detections: [], error: 'not found' }); continue; }
      const detections = await bridge.predict(modelName, weightsPath, imagePath, conf);
      results.push({ split, filename, detections });
    }
    res.json({ results });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════
//  ROUTES: Dataset
// ═══════════════════════════════════════════════════════════════════════

app.get('/api/dataset/stats', (req, res) => {
  const idx = getIndex();
  const trainCount = idx.filter(i => i.split === 'train').length;
  const valCount = idx.filter(i => i.split === 'val').length;
  const labeled = idx.filter(i => i.hasLabel).length;
  const unlabeled = idx.filter(i => !i.hasLabel).length;
  const totalBoxes = idx.reduce((s, i) => s + i.boxCount, 0);
  const withOverlaps = idx.filter(i => i.hasOverlaps).length;
  const withTiny = idx.filter(i => i.hasTiny).length;
  res.json({ total: idx.length, trainCount, valCount, labeled, unlabeled, totalBoxes, avgBoxes: labeled ? (totalBoxes / labeled).toFixed(1) : 0, withOverlaps, withTiny });
});

app.get('/api/dataset/quality', (req, res) => {
  const idx = getIndex();
  res.json({
    overlapping: idx.filter(i => i.hasOverlaps),
    tiny: idx.filter(i => i.hasTiny),
    unlabeled: idx.filter(i => !i.hasLabel),
  });
});

app.post('/api/dataset/scrub', (req, res) => {
  const { action } = req.body;
  let removed = 0;
  if (action === 'delete-overlaps') {
    const idx = getIndex().filter(i => i.hasOverlaps);
    for (const item of idx) {
      const imgPath = path.join(DATASET_DIR, 'images', item.split, item.filename);
      const stem = path.parse(item.filename).name;
      const lblPath = path.join(DATASET_DIR, 'labels', item.split, stem + '.txt');
      if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
      if (fs.existsSync(lblPath)) fs.unlinkSync(lblPath);
      removed++;
    }
  } else if (action === 'delete-tiny') {
    const idx = getIndex().filter(i => i.hasTiny);
    for (const item of idx) {
      const imgPath = path.join(DATASET_DIR, 'images', item.split, item.filename);
      const stem = path.parse(item.filename).name;
      const lblPath = path.join(DATASET_DIR, 'labels', item.split, stem + '.txt');
      if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
      if (fs.existsSync(lblPath)) fs.unlinkSync(lblPath);
      removed++;
    }
  }
  imageIndex = null;
  res.json({ removed });
});

// ═══════════════════════════════════════════════════════════════════════
//  ROUTES: Training
// ═══════════════════════════════════════════════════════════════════════

app.get('/api/training/status', (req, res) => res.json(trainer.getStatus()));
app.post('/api/training/start', (req, res) => res.json(trainer.start()));
app.post('/api/training/stop', (req, res) => { trainer.stop(); res.json({ status: 'stopping' }); });

// ── Refresh index endpoint ──────────────────────────────────────────────
app.post('/api/refresh', (req, res) => { imageIndex = null; buildImageIndex(); res.json({ count: imageIndex.length }); });

// ── Start ───────────────────────────────────────────────────────────────
(async () => {
  console.log('Starting inference worker...');
  await bridge.start();
  console.log(`Inference worker: ${bridge.ready ? 'ready' : 'starting in background'}`);
  buildImageIndex();
  console.log(`Indexed ${imageIndex.length} images`);
  server.listen(PORT, () => console.log(`\nAnnotation server → http://localhost:${PORT}`));
})();
