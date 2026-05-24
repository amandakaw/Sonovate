from flask import Flask, request, jsonify
import librosa
import numpy as np
import scipy.fft
import os
import uuid
import webbrowser
import threading

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

APP_HOST = "127.0.0.1"
APP_PORT = 5000
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
DISPLAY_URL = "http://Sonovate.com"

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Sonovate.com</title>

<style>
* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  color: #e2e8f0;
  text-align: center;
  font-family: "Segoe UI", system-ui, sans-serif;
  padding: 0 20px 48px;
  background:
    radial-gradient(ellipse 80% 60% at 15% 20%, rgba(56, 189, 248, 0.12) 0%, transparent 55%),
    radial-gradient(ellipse 70% 50% at 85% 75%, rgba(99, 102, 241, 0.14) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 50% 100%, rgba(34, 211, 238, 0.08) 0%, transparent 50%),
    linear-gradient(160deg, #060d18 0%, #0a1628 35%, #0f1f35 65%, #081420 100%);
  background-attachment: fixed;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

.container {
  position: relative;
  z-index: 1;
  max-width: 980px;
  margin: auto;
  padding: 36px 32px 40px;
  background: rgba(10, 22, 40, 0.55);
  border: 1px solid rgba(56, 189, 248, 0.18);
  border-radius: 20px;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.site-header {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 6px;
  padding: 22px 24px 20px;
  margin-bottom: 28px;
  background: rgba(6, 13, 24, 0.75);
  border-bottom: 1px solid rgba(56, 189, 248, 0.22);
  backdrop-filter: blur(10px);
}

.header-main {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 10px 20px;
}

.site-url {
  font-size: 0.95rem;
  font-weight: 400;
  letter-spacing: 0.03em;
  color: #38bdf8;
  text-decoration: none;
  opacity: 0.9;
}

.site-url:hover {
  color: #7dd3fc;
  text-decoration: underline;
}

.brand {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: linear-gradient(135deg, #e0f2fe 0%, #38bdf8 45%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.slogan {
  font-size: 1rem;
  font-style: italic;
  font-weight: 300;
  letter-spacing: 0.04em;
  color: #7dd3fc;
  opacity: 0.92;
  display: inline-block;
}

.slogan::before {
  content: "";
  margin-right: 12px;
  opacity: 0.5;
  font-style: normal;
}

.page-title {
  margin: 0 0 8px;
  font-size: 1.35rem;
  font-weight: 500;
  color: #cbd5e1;
}

.subtitle {
  color: #64748b;
  font-size: 0.9rem;
  margin-bottom: 28px;
}

.upload-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 12px;
}

input[type="file"] {
  color: #94a3b8;
  font-size: 0.9rem;
}

input[type="file"]::file-selector-button {
  padding: 9px 16px;
  margin-right: 10px;
  border: 1px solid rgba(56, 189, 248, 0.35);
  border-radius: 8px;
  background: rgba(15, 30, 55, 0.8);
  color: #bae6fd;
  cursor: pointer;
}

button.primary {
  padding: 10px 22px;
  background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
  border: none;
  color: white;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  box-shadow: 0 4px 16px rgba(56, 189, 248, 0.25);
  transition: transform 0.15s, box-shadow 0.15s;
}

button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35);
}

button.primary:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}

.box {
  margin-top: 14px;
  padding: 12px 16px;
  background: rgba(15, 30, 55, 0.6);
  border: 1px solid rgba(56, 189, 248, 0.12);
  border-radius: 10px;
  color: #cbd5e1;
  font-size: 0.95rem;
}

.loading-container {
  margin-top: 18px;
  padding: 16px 20px;
  background: rgba(15, 30, 55, 0.7);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 12px;
}

.loading-container.hidden { display: none; }

.loading-label {
  margin: 0 0 10px;
  font-size: 0.9rem;
  color: #7dd3fc;
}

.loading-track {
  height: 8px;
  background: rgba(15, 40, 70, 0.9);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(56, 189, 248, 0.15);
}

.loading-fill {
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0284c7, #22d3ee, #818cf8);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease infinite;
  transition: width 0.25s ease;
}

@keyframes shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}

.dish-comparison {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 40px;
  margin-top: 28px;
  flex-wrap: wrap;
}

.dish-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.dish-panel h3 {
  font-size: 0.95rem;
  font-weight: 500;
  color: #94a3b8;
  margin: 0 0 6px;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.dish-panel .rate {
  font-size: 0.85rem;
  color: #38bdf8;
  margin-bottom: 10px;
}

canvas {
  background: radial-gradient(circle at 40% 35%, #152238 0%, #0a1220 72%);
  border-radius: 50%;
  border: 2px solid rgba(56, 189, 248, 0.45);
  box-shadow: 0 0 32px rgba(56, 189, 248, 0.12), inset 0 0 40px rgba(0, 0, 0, 0.3);
}

.results-summary {
  margin: 28px auto 0;
  max-width: 520px;
  padding: 14px 20px;
  background: rgba(15, 30, 55, 0.65);
  border: 1px solid rgba(129, 140, 248, 0.25);
  border-radius: 12px;
  color: #c7d2fe;
  font-size: 0.95rem;
  line-height: 1.6;
  min-height: 0;
}

.results-summary:empty { display: none; }

.results-summary .change-value {
  font-size: 1.15rem;
  font-weight: 600;
  color: #38bdf8;
}

.notification-panel {
  position: fixed;
  top: 24px;
  right: 24px;
  width: 320px;
  background: rgba(10, 22, 42, 0.92);
  border: 1px solid rgba(56, 189, 248, 0.35);
  border-radius: 14px;
  backdrop-filter: blur(16px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45), 0 0 24px rgba(56, 189, 248, 0.1);
  z-index: 100;
  overflow: hidden;
  animation: slideIn 0.35s ease;
}

.notification-panel.hidden { display: none; }

@keyframes slideIn {
  from { opacity: 0; transform: translateX(24px); }
  to { opacity: 1; transform: translateX(0); }
}

.notification-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  background: linear-gradient(135deg, rgba(2, 132, 199, 0.35) 0%, rgba(99, 102, 241, 0.25) 100%);
  border-bottom: 1px solid rgba(56, 189, 248, 0.2);
  font-weight: 600;
  color: #e0f2fe;
}

.notification-close {
  background: none;
  border: none;
  color: #94a3b8;
  font-size: 1.4rem;
  line-height: 1;
  cursor: pointer;
  padding: 0 4px;
  border-radius: 4px;
  transition: color 0.15s;
}

.notification-close:hover { color: #e2e8f0; }

.notification-body {
  padding: 16px;
  text-align: left;
  font-size: 0.9rem;
  line-height: 1.65;
  color: #cbd5e1;
}

.notification-body strong { color: #38bdf8; }
</style>
</head>

<body>

<header class="site-header">
  <div class="brand">Sonovate</div>
  <div class="slogan">THE INTERSECTION OF SCIENCE AND MUSIC</div>
</header>

<div class="container">
  <h1 class="page-title">Audio → Bacterial Petri Dish Simulation</h1>
  <p class="subtitle">VISUALIZING HOW MUSIC SHAPES BACTERIAL GROWTH</p>

  <div class="upload-row">
    <input type="file" id="fileInput" accept=".mp3,.wav" />
    <button class="primary" id="analyzeBtn" onclick="uploadFile()">Analyze & Simulate</button>
  </div>

  <div id="loadingBar" class="loading-container hidden">
    <p class="loading-label">Analyzing audio…</p>
    <div class="loading-track">
      <div class="loading-fill" id="loadingFill"></div>
    </div>
  </div>

  <div id="info" class="box"></div>

  <div class="dish-comparison">
    <div class="dish-panel">
      <h3>Without Sound</h3>
      <div class="rate">Growth rate: 0.02</div>
      <canvas id="petriBaseline" width="400" height="400"></canvas>
    </div>
    <div class="dish-panel">
      <h3>With Sound</h3>
      <div class="rate" id="affectedRate">Growth rate: —</div>
      <canvas id="petriAffected" width="400" height="400"></canvas>
    </div>
  </div>

  <div id="resultsSummary" class="results-summary"></div>

  <div id="timers" class="box">
    Real World Elapsed Time: 0.0 seconds<br>
    Simulated Time: 0.0 seconds
  </div>
</div>

<div id="notificationPanel" class="notification-panel hidden">
  <div class="notification-header">
    <span>Incubation Complete!</span>
    <button class="notification-close" onclick="minimizeNotification()" title="Minimize">×</button>
  </div>
  <div class="notification-body" id="notificationBody"></div>
</div>

<audio id="simAudio" style="display:none"></audio>

<script>
const canvasBaseline = document.getElementById("petriBaseline");
const canvasAffected = document.getElementById("petriAffected");
const ctxBaseline = canvasBaseline.getContext("2d");
const ctxAffected = canvasAffected.getContext("2d");

const DISH_SIZE = 400;
const DISH_CENTER = DISH_SIZE / 2;
const DISH_RADIUS = 190;
const SPAWN_RADIUS = 150;

const BASELINE_RATE = 0.02;
const INITIAL_COLONIES = 10;
const BASELINE_ADDED_COVERAGE = 8500;

let baselineColonies = [];
let affectedColonies = [];
let animationRunning = false;
let initialCoverage = 0;
let baselineFinalCoverage = 0;
let affectedFinalCoverage = 0;

let startTime = null;

const REAL_DURATION = 10.0;
const SIM_SCALE = 120;

let effectiveGrowthRate = BASELINE_RATE;

let audioObjectUrl = null;
let loadingInterval = null;

function randomGreenShade() {
  const hue = 95 + Math.random() * 45;
  const sat = 55 + Math.random() * 35;
  const light = 30 + Math.random() * 28;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function fitsInDish(x, y, r) {
  return Math.hypot(x - DISH_CENTER, y - DISH_CENTER) + r <= DISH_RADIUS - 2;
}

function hasAcceptableSpacing(x, y, r, colonies) {
  for (const c of colonies) {
    const dist = Math.hypot(x - c.x, y - c.y);
    if (dist < 0.9) return false;
    const minDist = (c.r + r) * 0.62;
    if (dist < minDist) return false;
  }
  return true;
}

function tryPlaceColony(colonies, generator, attempts = 22) {
  for (let i = 0; i < attempts; i++) {
    const candidate = generator();
    if (
      fitsInDish(candidate.x, candidate.y, candidate.r) &&
      hasAcceptableSpacing(candidate.x, candidate.y, candidate.r, colonies)
    ) {
      return candidate;
    }
  }
  return null;
}

function randomColonyCandidate() {
  const angle = Math.random() * Math.PI * 2;
  const dist = Math.sqrt(Math.random()) * SPAWN_RADIUS;
  const r = 2.2 + Math.random() * 3.3;
  return {
    x: DISH_CENTER + Math.cos(angle) * dist,
    y: DISH_CENTER + Math.sin(angle) * dist,
    r,
    color: randomGreenShade()
  };
}

function createColony(colonies = []) {
  return tryPlaceColony(colonies, randomColonyCandidate);
}

function createColonyNear(colonies) {
  const near = tryPlaceColony(colonies, () => {
    const r = 2.2 + Math.random() * 3.3;
    const parent = colonies[Math.floor(Math.random() * colonies.length)];
    const angle = Math.random() * Math.PI * 2;
    const gap = 0.5 + Math.random() * 1.5;
    const offset = parent.r + r + gap;
    return {
      x: parent.x + Math.cos(angle) * offset,
      y: parent.y + Math.sin(angle) * offset,
      r,
      color: randomGreenShade()
    };
  });

  if (near) return near;
  return tryPlaceColony(colonies, randomColonyCandidate, 14);
}

function targetCoverageAtProgress(finalCoverage, progress) {
  return initialCoverage + (finalCoverage - initialCoverage) * progress;
}

function initColonies() {
  baselineColonies = [];
  affectedColonies = [];
  for (let i = 0; i < INITIAL_COLONIES; i++) {
    const baseline = createColony(baselineColonies);
    if (baseline) baselineColonies.push(baseline);
    const affected = createColony(affectedColonies);
    if (affected) affectedColonies.push(affected);
  }

  initialCoverage = dishCoverage(baselineColonies);
  baselineFinalCoverage = initialCoverage + BASELINE_ADDED_COVERAGE;
  affectedFinalCoverage = baselineFinalCoverage * (effectiveGrowthRate / BASELINE_RATE);
}

function spawnToCoverage(colonies, targetCoverage) {
  const current = dishCoverage(colonies);
  const deficit = targetCoverage - current;
  if (deficit <= 0) return;

  const avgColonyArea = Math.PI * 3.35 * 3.35;
  const maxAttempts = Math.min(Math.max(Math.ceil(deficit / avgColonyArea), 4), 12);

  for (let i = 0; i < maxAttempts; i++) {
    if (dishCoverage(colonies) >= targetCoverage) break;
    const colony = createColonyNear(colonies);
    if (!colony) break;
    colonies.push(colony);
  }
}

function dishCoverage(colonies) {
  return colonies.reduce((sum, c) => sum + Math.PI * c.r * c.r, 0);
}

function drawPetriDish(ctx, colonies) {
  ctx.clearRect(0, 0, DISH_SIZE, DISH_SIZE);

  ctx.beginPath();
  ctx.arc(DISH_CENTER, DISH_CENTER, DISH_RADIUS, 0, Math.PI * 2);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  for (let c of colonies) {
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.fillStyle = c.color;
    ctx.fill();
  }
}

function updateColonies(colonies, finalCoverage, progress) {
  const target = targetCoverageAtProgress(finalCoverage, progress);
  spawnToCoverage(colonies, target);
}

function stopSimulationAudio() {
  const audioEl = document.getElementById("simAudio");
  audioEl.pause();
  audioEl.currentTime = 0;
  audioEl.removeAttribute("src");
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
    audioObjectUrl = null;
  }
}

function startSimulationAudio(file) {
  stopSimulationAudio();
  const audioEl = document.getElementById("simAudio");
  audioObjectUrl = URL.createObjectURL(file);
  audioEl.src = audioObjectUrl;
  audioEl.loop = true;
  audioEl.play().catch(() => {});
}

function showLoadingBar() {
  const loadingBar = document.getElementById("loadingBar");
  const loadingFill = document.getElementById("loadingFill");
  const analyzeBtn = document.getElementById("analyzeBtn");

  loadingBar.classList.remove("hidden");
  loadingFill.style.width = "0%";
  analyzeBtn.disabled = true;

  let progress = 0;
  clearInterval(loadingInterval);
  loadingInterval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 8 + 2, 88);
    loadingFill.style.width = progress + "%";
  }, 180);
}

function hideLoadingBar() {
  const loadingBar = document.getElementById("loadingBar");
  const loadingFill = document.getElementById("loadingFill");
  const analyzeBtn = document.getElementById("analyzeBtn");

  clearInterval(loadingInterval);
  loadingFill.style.width = "100%";

  setTimeout(() => {
    loadingBar.classList.add("hidden");
    loadingFill.style.width = "0%";
    analyzeBtn.disabled = false;
  }, 350);
}

function buildResultsHtml(percentChange, sign) {
  return `
    <strong>Incubation Complete!</strong><br>
    Growth Rate Change: <span class="change-value">${sign}${percentChange.toFixed(1)}%</span> vs baseline
  `;
}

function showIncubationResults(percentChange, sign) {
  const panel = document.getElementById("notificationPanel");
  const body = document.getElementById("notificationBody");
  const summary = document.getElementById("resultsSummary");

  const html = buildResultsHtml(percentChange, sign);
  body.innerHTML = html;
  summary.innerHTML = html;
  panel.classList.remove("hidden");
}

function minimizeNotification() {
  document.getElementById("notificationPanel").classList.add("hidden");
}

function animate() {
  if (!animationRunning) return;

  const now = performance.now();
  const realElapsed = (now - startTime) / 1000;

  const timers = document.getElementById("timers");

  const clampedReal = Math.min(realElapsed, REAL_DURATION);
  const simTime = clampedReal * SIM_SCALE;
  const progress = clampedReal / REAL_DURATION;

  timers.innerHTML = `
    Real World Elapsed Time: ${clampedReal.toFixed(1)} seconds<br>
    Simulated Time: ${simTime.toFixed(1)} seconds
  `;

  updateColonies(baselineColonies, baselineFinalCoverage, progress);
  updateColonies(affectedColonies, affectedFinalCoverage, progress);

  drawPetriDish(ctxBaseline, baselineColonies);
  drawPetriDish(ctxAffected, affectedColonies);

  if (realElapsed >= REAL_DURATION) {
    animationRunning = false;
    stopSimulationAudio();

    const percentChange = ((effectiveGrowthRate - BASELINE_RATE) / BASELINE_RATE) * 100;
    const sign = percentChange >= 0 ? "+" : "";
    showIncubationResults(percentChange, sign);

    return;
  }

  requestAnimationFrame(animate);
}

async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const info = document.getElementById("info");
  const summary = document.getElementById("resultsSummary");
  const affectedRateEl = document.getElementById("affectedRate");

  if (!fileInput.files.length) {
    alert("Upload audio file");
    return;
  }

  summary.innerHTML = "";
  minimizeNotification();
  info.innerHTML = "";
  stopSimulationAudio();

  const audioFile = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", audioFile);

  showLoadingBar();

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Analysis failed");
    }

    effectiveGrowthRate = data.r;
    affectedRateEl.textContent = `Growth rate: ${effectiveGrowthRate.toFixed(4)}`;

    info.innerHTML = `
      Tempo: ${data.tempo} BPM <br>
      Frequency: ${data.frequency} Hz
    `;

    startTime = performance.now();

    initColonies();
    animationRunning = true;
    startSimulationAudio(audioFile);
    animate();
  } catch (err) {
    info.innerHTML = `<span style="color:#f87171">${err.message || "Analysis failed. Please try again."}</span>`;
  } finally {
    hideLoadingBar();
  }
}

initColonies();
drawPetriDish(ctxBaseline, baselineColonies);
drawPetriDish(ctxAffected, affectedColonies);
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return INDEX_HTML


def analyze_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050, mono=True)

    fft = np.abs(scipy.fft.rfft(audio))
    freqs = scipy.fft.rfftfreq(len(audio), d=1 / sr)

    dominant_freq = float(freqs[np.argmax(fft)])

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    return tempo, dominant_freq


def save_upload(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    original_name = file.filename or "upload.mp3"
    extension = os.path.splitext(original_name)[1].lower() or ".mp3"
    filename = f"{uuid.uuid4().hex}{extension}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path


def compute_growth_curve(tempo, frequency):
    frequency = max(frequency, 1e-3)

    growth_rate = 0.02 * (
        1 +
        0.3 * ((tempo - 120) / 120) +
        0.15 * np.log(frequency / 440)
    )

    curve = []
    N = 1

    for _ in range(50):
        N = N + growth_rate * N * (1 - N / 100)
        curve.append(N)

    return curve, growth_rate


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No audio file uploaded."}), 400

    path = None
    try:
        path = save_upload(file)
        tempo, freq = analyze_audio(path)
        curve, r = compute_growth_curve(tempo, freq)

        return jsonify({
            "tempo": round(tempo, 2),
            "frequency": round(freq, 2),
            "growth_curve": curve,
            "r": float(r)
        })

    except Exception as exc:
        return jsonify({"error": f"Could not analyze audio: {exc}"}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    app.run(debug=True)
