import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from PIL import Image
import io
import os
import random
import time

app = Flask(__name__)

modelo = tf.keras.models.load_model("modelo_tomates.h5")
CLASES = ["Tomate_Deteriorado", "Tomate_Maduro"]

# ─────────────────────────────────────────────
# HTML MEJORADO
# ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clasificador de Tomates</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0f172a;
      --surface:   #1e293b;
      --surface2:  #273449;
      --border:    rgba(255,255,255,0.08);
      --text:      #f1f5f9;
      --muted:     #94a3b8;
      --green:     #22c55e;
      --green-bg:  rgba(34,197,94,0.12);
      --red:       #f87171;
      --red-bg:    rgba(248,113,113,0.12);
      --accent:    #1D9E75;
      --radius-md: 10px;
      --radius-lg: 14px;
    }

    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 1.5rem 1rem;
    }

    /* ── HEADER ── */
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    header h1 {
      font-size: 22px;
      font-weight: 500;
      margin-bottom: 4px;
    }
    header p { font-size: 14px; color: var(--muted); }

    /* ── GRID ── */
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      max-width: 920px;
      margin: 0 auto;
    }
    @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } }

    /* ── CARD ── */
    .card {
      background: var(--surface);
      border: 0.5px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
    }

    /* ── SECTION LABEL ── */
    .label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 500;
      margin-bottom: 12px;
    }

    /* ── UPLOAD ZONE ── */
    .upload-zone {
      border: 1.5px dashed rgba(255,255,255,0.18);
      border-radius: var(--radius-lg);
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      background: var(--surface2);
      transition: border-color 0.2s;
    }
    .upload-zone:hover { border-color: var(--accent); }
    .upload-zone i { font-size: 32px; color: var(--muted); display: block; margin-bottom: 8px; }
    .upload-zone p { font-size: 14px; color: var(--muted); }
    .upload-zone small { font-size: 12px; color: rgba(148,163,184,0.6); }

    #preview {
      width: 100%;
      height: 200px;
      object-fit: cover;
      border-radius: var(--radius-md);
      display: none;
      margin-bottom: 1rem;
    }

    /* ── BUTTON ── */
    .btn {
      width: 100%;
      margin-top: 1rem;
      padding: 10px;
      font-size: 14px;
      font-weight: 500;
      background: var(--surface2);
      border: 0.5px solid rgba(255,255,255,0.12);
      border-radius: var(--radius-md);
      cursor: pointer;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn:hover { background: rgba(255,255,255,0.08); }
    .btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .btn.primary:hover { background: #0F6E56; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* ── SENSOR ── */
    .sensor-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 0;
      border-bottom: 0.5px solid var(--border);
      font-size: 13px;
    }
    .sensor-row:last-child { border-bottom: none; }
    .sensor-name {
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .sensor-val { font-weight: 500; }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      display: inline-block;
      margin-right: 6px;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    /* ── TABS ── */
    .tab-row { display: flex; gap: 4px; margin-bottom: 1rem; }
    .tab {
      font-size: 13px;
      padding: 5px 12px;
      border-radius: var(--radius-md);
      border: 0.5px solid transparent;
      cursor: pointer;
      color: var(--muted);
      background: none;
    }
    .tab.active {
      background: var(--surface2);
      border-color: var(--border);
      color: var(--text);
      font-weight: 500;
    }

    /* ── RESULT ── */
    .result-badge {
      padding: 6px 14px;
      border-radius: 99px;
      font-size: 13px;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 14px;
    }
    .badge-maduro   { background: var(--green-bg); color: var(--green); }
    .badge-deteriorado { background: var(--red-bg);   color: var(--red); }

    .bar-track {
      height: 6px;
      background: var(--surface2);
      border-radius: 99px;
      overflow: hidden;
      margin: 5px 0 3px;
    }
    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.8s ease;
    }
    .bar-green { background: var(--green); }
    .bar-red   { background: var(--red); }

    /* ── METRIC CARDS ── */
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 1rem;
    }
    .metric-card {
      background: var(--surface2);
      border-radius: var(--radius-md);
      padding: 12px;
    }
    .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 500; }
    .metric-unit  { font-size: 13px; color: var(--muted); font-weight: 400; }

    /* ── HISTORY ── */
    .history-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 10px;
      background: var(--surface2);
      border-radius: var(--radius-md);
      font-size: 13px;
      margin-bottom: 6px;
    }
    .history-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot-green { background: var(--green); }
    .dot-red   { background: var(--red); }

    /* ── SPINNER ── */
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinning { animation: spin 0.8s linear infinite; }
  </style>
</head>
<body>

<header>
  <h1><i class="ti ti-plant-2"
     style="font-size:20px;vertical-align:-2px;margin-right:6px;color:#1D9E75"></i>
    Clasificador de tomates
  </h1>
  <p>Modelo CNN &middot; ESP32-CAM + DHT22 &middot; Tiempo real</p>
</header>

<div class="grid">

  <!-- ── COLUMNA IZQUIERDA ── -->
  <div>
    <div class="card">
      <div class="label">Imagen de entrada</div>

      <div class="upload-zone" id="dropzone"
           onclick="document.getElementById('fileInput').click()">
        <i class="ti ti-photo-up"></i>
        <p>Arrastra una imagen o haz clic</p>
        <small>JPG, PNG &middot; máx. 5 MB</small>
      </div>

      <img id="preview" alt="Vista previa del tomate">

      <input type="file" id="fileInput" accept="image/*" style="display:none"
             onchange="handleFile(this)">

      <div id="changeBtn" style="display:none">
        <button class="btn" onclick="cambiarFoto()" style="margin-top:0.75rem">
          <i class="ti ti-photo-edit"></i> Cambiar foto
        </button>
      </div>

      <button class="btn primary" id="predictBtn" onclick="runPrediction()">
        <i class="ti ti-scan"></i> Analizar imagen
      </button>
    </div>

    <!-- SENSOR -->
    <div class="card" style="margin-top:1rem">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <div class="label" style="margin-bottom:0">Sensor IoT</div>
        <span style="font-size:12px;color:var(--muted)">
          <span class="status-dot"></span>ESP32 &middot; en línea
        </span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-temperature"></i>Temperatura</span>
        <span class="sensor-val" id="sTemp">— °C</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-droplet"></i>Humedad relativa</span>
        <span class="sensor-val" id="sHum">— %</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-eye"></i>Condición estimada</span>
        <span class="sensor-val" id="sCond" style="color:var(--green)">—</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-clock"></i>Última lectura</span>
        <span class="sensor-val" id="sTime"
              style="font-weight:400;color:var(--muted)">—</span>
      </div>
    </div>
  </div>

  <!-- ── COLUMNA DERECHA ── -->
  <div>
    <div class="card">
      <div class="tab-row">
        <button class="tab active" onclick="setTab('result',this)">Resultado</button>
        <button class="tab" onclick="setTab('history',this)">Historial</button>
        <button class="tab" onclick="setTab('stats',this)">Estadísticas</button>
      </div>

      <!-- RESULTADO -->
      <div id="tab-result">
        <div id="no-result" style="padding:2.5rem 0;text-align:center">
          <i class="ti ti-scan"
             style="font-size:36px;color:var(--muted);display:block;margin-bottom:8px"></i>
          <p style="font-size:14px;color:var(--muted)">
            Sube una imagen para clasificar
          </p>
        </div>
        <div id="has-result" style="display:none">
          <div id="resultBadge" class="result-badge badge-maduro">
            <i class="ti ti-circle-check"></i>
            <span id="resultLabel">Tomate maduro</span>
          </div>
          <div class="label" style="margin-top:12px">Confianza del modelo</div>
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
            <span>Tomate maduro</span><span id="conf1">—</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-green" id="bar1" style="width:0%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:13px;margin:8px 0 3px">
            <span>Tomate deteriorado</span><span id="conf2">—</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill bar-red" id="bar2" style="width:0%"></div>
          </div>

          <div class="metric-grid" style="margin-top:1.25rem">
            <div class="metric-card">
              <div class="metric-label">Inferencia</div>
              <div class="metric-value" id="mInfer">—</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Modelo</div>
              <div class="metric-value" style="font-size:14px;padding-top:4px">
                CNN<span class="metric-unit"> 224×224</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- HISTORIAL -->
      <div id="tab-history" style="display:none">
        <div class="label">Últimas clasificaciones</div>
        <div id="historyList"></div>
        <p id="histEmpty"
           style="font-size:13px;color:var(--muted);text-align:center;padding:1rem 0">
          Sin clasificaciones aún
        </p>
      </div>

      <!-- ESTADÍSTICAS -->
      <div id="tab-stats" style="display:none">
        <div class="label">Sesión actual</div>
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Total analizados</div>
            <div class="metric-value" id="stTotal">0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Tasa de maduros</div>
            <div class="metric-value" id="stRate">
              —<span class="metric-unit">%</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Confianza media</div>
            <div class="metric-value" id="stConf">
              —<span class="metric-unit">%</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Tiempo medio</div>
            <div class="metric-value" id="stTime">
              —<span class="metric-unit">ms</span>
            </div>
          </div>
        </div>
      </div>

    </div><!-- .card -->
  </div>

</div><!-- .grid -->

<script>
/* ── ESTADO ── */
let total = 0, maduroCount = 0, confSum = 0, timeSum = 0;
let sElapsed = 0, sTimer = null;

/* ── TABS ── */
function setTab(name, el) {
  ['result','history','stats'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===name ? 'block' : 'none';
  });
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
}

/* ── UPLOAD ── */
function handleFile(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('dropzone').style.display = 'none';
    document.getElementById('changeBtn').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function cambiarFoto() {
  const input = document.getElementById('fileInput');
  input.value = '';
  input.click();
}

/* ── PREDICCIÓN (envía al servidor) ── */
function runPrediction() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) {
    alert('Sube una imagen primero'); return;
  }

  const btn = document.getElementById('predictBtn');
  btn.innerHTML = '<i class="ti ti-loader-2 spinning"></i> Analizando...';
  btn.disabled = true;

  const t0 = performance.now();
  const fd = new FormData();
  fd.append('imagen', fileInput.files[0]);

  fetch('/', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      const elapsed = Math.round(performance.now() - t0);
      showResult(data.clase, data.confianza, elapsed);
      updateSensorUI(data.temp, data.hum);
    })
    .catch(() => alert('Error al contactar el servidor'))
    .finally(() => {
      btn.innerHTML = '<i class="ti ti-scan"></i> Analizar imagen';
      btn.disabled = false;
    });
}

/* ── MOSTRAR RESULTADO ── */
function showResult(clase, conf, ms) {
  const isMaduro = clase === 'Tomate_Maduro';
  const c1 = isMaduro ? conf : Math.round(100 - conf);
  const c2 = 100 - c1;

  document.getElementById('no-result').style.display = 'none';
  document.getElementById('has-result').style.display = 'block';

  const badge = document.getElementById('resultBadge');
  badge.className = 'result-badge ' + (isMaduro ? 'badge-maduro' : 'badge-deteriorado');
  badge.querySelector('i').className = 'ti ' + (isMaduro ? 'ti-circle-check' : 'ti-circle-x');
  document.getElementById('resultLabel').textContent =
    isMaduro ? 'Tomate maduro' : 'Tomate deteriorado';

  document.getElementById('conf1').textContent = c1 + '%';
  document.getElementById('conf2').textContent = c2 + '%';
  document.getElementById('bar1').style.width = c1 + '%';
  document.getElementById('bar2').style.width = c2 + '%';
  document.getElementById('mInfer').innerHTML = ms + ' <span class="metric-unit">ms</span>';

  // Historial
  total++; if (isMaduro) maduroCount++;
  confSum += conf; timeSum += ms;

  const list = document.getElementById('historyList');
  document.getElementById('histEmpty').style.display = 'none';
  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `
    <span style="display:flex;align-items:center;gap:8px">
      <span class="history-dot ${isMaduro?'dot-green':'dot-red'}"></span>
      ${isMaduro?'Tomate maduro':'Deteriorado'}
    </span>
    <span style="color:var(--muted);font-size:12px">${conf}% &middot; ahora</span>`;
  list.prepend(item);

  // Estadísticas
  document.getElementById('stTotal').textContent = total;
  document.getElementById('stRate').innerHTML =
    Math.round(maduroCount/total*100) + '<span class="metric-unit">%</span>';
  document.getElementById('stConf').innerHTML =
    Math.round(confSum/total) + '<span class="metric-unit">%</span>';
  document.getElementById('stTime').innerHTML =
    Math.round(timeSum/total) + '<span class="metric-unit">ms</span>';
}

/* ── SENSOR ── */
function updateSensorUI(temp, hum) {
  document.getElementById('sTemp').textContent = temp + ' °C';
  document.getElementById('sHum').textContent  = hum + ' %';
  const good = temp >= 20 && temp <= 28 && hum >= 50 && hum <= 80;
  const cond = document.getElementById('sCond');
  cond.textContent = good ? 'Óptima'
    : (temp > 28 ? 'Temperatura alta' : 'Humedad fuera de rango');
  cond.style.color = good ? 'var(--green)' : 'var(--red)';

  if (sTimer) clearInterval(sTimer);
  sElapsed = 0;
  sTimer = setInterval(() => {
    sElapsed++;
    document.getElementById('sTime').textContent = 'hace ' + sElapsed + ' s';
  }, 1000);
}

// Lectura inicial del sensor al cargar
fetch('/sensor').then(r=>r.json()).then(d => updateSensorUI(d.temp, d.hum));
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────
def predecir_imagen(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = modelo.predict(arr)
    clase = CLASES[np.argmax(pred)]
    confianza = round(float(np.max(pred)) * 100, 1)
    return clase, confianza


def leer_sensor():
    """Simula DHT22 — reemplazar por lectura real de ESP32."""
    return round(random.uniform(18, 35), 1), round(random.uniform(40, 90), 1)


# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
@app.after_request
def no_cache(r):
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("imagen")
        if not file:
            return jsonify({"error": "No se recibió imagen"}), 400

        t0 = time.time()
        clase, confianza = predecir_imagen(file.read())
        temp, hum = leer_sensor()

        return jsonify({
            "clase":     clase,
            "confianza": confianza,
            "temp":      temp,
            "hum":       hum,
            "ms":        round((time.time() - t0) * 1000)
        })

    return render_template_string(HTML)


@app.route("/sensor")
def sensor():
    temp, hum = leer_sensor()
    return jsonify({"temp": temp, "hum": hum})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
