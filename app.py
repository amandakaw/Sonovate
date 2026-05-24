from flask import Flask, request, jsonify
import librosa
import numpy as np
import scipy.fft
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# FRONTEND (PETRI DISH UI)
# -------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Audio → Petri Dish Simulation</title>

<style>
body {
  background: #0b0f14;
  color: white;
  text-align: center;
  font-family: Arial;
  padding: 20px;
}

.container {
  max-width: 800px;
  margin: auto;
}

canvas {
  background: #111827;
  border-radius: 50%;
  margin-top: 20px;
  border: 2px solid #2dd4bf;
}

button {
  padding: 10px 20px;
  background: #3b82f6;
  border: none;
  color: white;
  border-radius: 8px;
  cursor: pointer;
}

.box {
  margin-top: 10px;
}
</style>
</head>

<body>

<div class="container">
  <h1>🧫 Audio → Bacterial Petri Dish Simulation</h1>

  <input type="file" id="fileInput" accept=".mp3,.wav" />
  <br><br>

  <button onclick="uploadFile()">Analyze & Simulate</button>

  <div id="info" class="box"></div>

  <canvas id="petri" width="600" height="600"></canvas>
</div>

<script>
const canvas = document.getElementById("petri");
const ctx = canvas.getContext("2d");

let colonies = [];
let animationRunning = false;
let growthRate = 0;

function initColonies() {
  colonies = [];
  for (let i = 0; i < 25; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 200;

    colonies.push({
      x: 300 + Math.cos(angle) * radius,
      y: 300 + Math.sin(angle) * radius,
      r: 2
    });
  }
}

function drawPetriDish() {
  ctx.clearRect(0, 0, 600, 600);

  // dish border glow
  ctx.beginPath();
  ctx.arc(300, 300, 290, 0, Math.PI * 2);
  ctx.strokeStyle = "#2dd4bf";
  ctx.lineWidth = 3;
  ctx.stroke();

  // colonies
  for (let c of colonies) {
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.fillStyle = "#22c55e";
    ctx.fill();
  }
}

function animate() {
  if (!animationRunning) return;

  for (let c of colonies) {
    c.r += growthRate;
    if (c.r > 60) c.r = 60; // cap growth
  }

  drawPetriDish();
  requestAnimationFrame(animate);
}

async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const info = document.getElementById("info");

  if (!fileInput.files.length) {
    alert("Upload audio file");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  info.innerHTML = "🧬 Analyzing audio...";

  const res = await fetch("/analyze", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  info.innerHTML = `
    🎶 Tempo: ${data.tempo} BPM <br>
    📡 Frequency: ${data.frequency} Hz
  `;

  // convert biological growth curve into animation speed
  const avgGrowth = data.growth_curve.reduce((a,b)=>a+b,0) / data.growth_curve.length;
  growthRate = avgGrowth / 200;

  initColonies();

  animationRunning = true;
  animate();
}

// initial render
initColonies();
drawPetriDish();
</script>

</body>
</html>
"""

# -------------------------
# HOME ROUTE
# -------------------------
@app.route("/")
def home():
    return INDEX_HTML

# -------------------------
# AUDIO ANALYSIS
# -------------------------
def analyze_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050, mono=True)

    fft = np.abs(scipy.fft.rfft(audio))
    freqs = scipy.fft.rfftfreq(len(audio), d=1/sr)

    dominant_freq = freqs[np.argmax(fft)]

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    return tempo, float(dominant_freq)

# -------------------------
# GROWTH MODEL
# -------------------------
def compute_growth_curve(tempo, frequency):
    frequency = max(frequency, 1e-3)

    r = 0.02 * (
        1 +
        0.3 * ((tempo - 120) / 120) +
        0.15 * np.log(frequency / 440)
    )

    curve = []
    N = 1

    for _ in range(50):
        N = N + r * N * (1 - N / 100)
        curve.append(N)

    return curve

# -------------------------
# API
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]

    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        tempo, freq = analyze_audio(path)
        curve = compute_growth_curve(tempo, freq)

        return jsonify({
            "tempo": round(tempo, 2),
            "frequency": round(freq, 2),
            "growth_curve": curve
        })

    finally:
        os.remove(path)

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)