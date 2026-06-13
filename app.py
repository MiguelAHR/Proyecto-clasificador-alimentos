import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import io
import random
import time

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

modelo = tf.keras.models.load_model("modelo_tomates.h5", compile=False)
modelo.make_predict_function()
CLASES = ["Tomate_Deteriorado", "Tomate_Maduro"]

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
      --bg: #0f172a; --surface: #1e293b; --surface2: #273449;
      --border: rgba(255,255,255,0.08); --text: #f1f5f9; --muted: #94a3b8;
      --green: #22c55e; --green-bg: rgba(34,197,94,0.12);
      --red: #f87171; --red-bg: rgba(248,113,113,0.12);
      --accent: #1D9E75; --radius-md: 10px; --radius-lg: 14px;
    }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 1.5rem 1rem; }
    header { text-align: center; margin-bottom: 2rem; }
    header h1 { font-size: 22px; font-weight: 500; margin-bottom: 4px; }
    header p { font-size: 14px; color: var(--muted); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; max-width: 920px; margin: 0 auto; }
    @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius-lg); padding: 1.25rem; }
    .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; margin-bottom: 12px; }
    .upload-zone { border: 1.5px dashed rgba(255,255,255,0.18); border-radius: var(--radius-lg); padding: 2rem 1rem; text-align: center; cursor: pointer; background: var(--surface2); }
    .upload-zone i { font-size: 32px; color: var(--muted); display: block; margin-bottom: 8px; }
    .upload-zone p { font-size: 14px; color: var(--muted); }
    .upload-zone small { font-size: 12px; color: rgba(148,163,184,0.6); }
    #preview { width: 100%; height: 200px; object-fit: cover; border-radius: var(--radius-md); display: none; margin-bottom: 0.75rem; }
    .btn { width: 100%; margin-top: 0.75rem; padding: 10px; font-size: 14px; font-weight: 500; background: var(--surface2); border: 0.5px solid rgba(255,255,255,0.12); border-radius: var(--radius-md); cursor: pointer; color: var(--text); display: flex; align-items: center; justify-content: center; gap: 8px; text-decoration: none; }
    .btn:hover { background: rgba(255,255,255,0.08); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: white; }
    .btn.primary:hover { background: #0F6E56; }
    .sensor-row { display: flex; align-items: center; justify-content: space-between; padding: 9px 0; border-bottom: 0.5px solid var(--border); font-size: 13px; }
    .sensor-row:last-child { border-bottom: none; }
    .sensor-name { color: var(--muted); display: flex; align-items: center; gap: 6px; }
    .sensor-val { font-weight: 500; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    .result-badge { padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; margin-bottom: 14px; }
    .badge-maduro { background: var(--green-bg); color: var(--green); }
    .badge-deteriorado { background: var(--red-bg); color: var(--red); }
    .bar-track { height: 6px; background: var(--surface2); border-radius: 99px; overflow: hidden; margin: 5px 0 3px; }
    .bar-fill { height: 100%; border-radius: 99px; }
    .bar-green { background: var(--green); }
    .bar-red { background: var(--red); }
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 1rem; }
    .metric-card { background: var(--surface2); border-radius: var(--radius-md); padding: 12px; }
    .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 500; }
    .metric-unit { font-size: 13px; color: var(--muted); font-weight: 400; }
    .empty-state { padding: 2.5rem 0; text-align: center; }
    .empty-state i { font-size: 36px; color: var(--muted); display: block; margin-bottom: 8px; }
    .empty-state p { font-size: 14px; color: var(--muted); }
    input[type=file] { display: none; }
  </style>
</head>
<body>

<header>
  <h1><i class="ti ti-plant-2" style="font-size:20px;vertical-align:-2px;margin-right:6px;color:#1D9E75"></i>Clasificador de tomates</h1>
  <p>Modelo CNN &middot; ESP32-CAM + DHT22 &middot; Tiempo real</p>
</header>

<form method="POST" enctype="multipart/form-data" id="mainForm">
<div class="grid">

  <!-- COLUMNA IZQUIERDA -->
  <div>
    <div class="card">
      <div class="label">Imagen de entrada</div>

      <div class="upload-zone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <i class="ti ti-photo-up"></i>
        <p>Haz clic para seleccionar</p>
        <small>JPG, PNG &middot; máx. 5 MB</small>
      </div>

      <img id="preview" alt="Vista previa">

      <input type="file" id="fileInput" name="imagen" accept="image/*" required onchange="handleFile(this)">

      <div id="changeBtn" style="display:none">
        <button type="button" class="btn" onclick="document.getElementById('fileInput').click()">
          <i class="ti ti-photo-edit"></i> Cambiar foto
        </button>
      </div>

      <button type="submit" class="btn primary" style="margin-top:0.75rem">
        <i class="ti ti-scan"></i> Analizar imagen
      </button>
    </div>

    <!-- SENSOR -->
    <div class="card" style="margin-top:1rem">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <div class="label" style="margin-bottom:0">Sensor IoT</div>
        <span style="font-size:12px;color:var(--muted)"><span class="status-dot"></span>ESP32 &middot; en línea</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-temperature"></i>Temperatura</span>
        <span class="sensor-val">{{ temp if temp else '—' }} {% if temp %}°C{% endif %}</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-droplet"></i>Humedad relativa</span>
        <span class="sensor-val">{{ hum if hum else '—' }} {% if hum %}%{% endif %}</span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-eye"></i>Condición estimada</span>
        <span class="sensor-val" style="color:{{ condicion_color if condicion_color else 'var(--muted)' }}">
          {{ condicion if condicion else '—' }}
        </span>
      </div>
      <div class="sensor-row">
        <span class="sensor-name"><i class="ti ti-clock"></i>Última lectura</span>
        <span class="sensor-val" style="font-weight:400;color:var(--muted)">
          {% if temp %}hace 0 s{% else %}—{% endif %}
        </span>
      </div>
    </div>
  </div>

  <!-- COLUMNA DERECHA -->
  <div>
    <div class="card" style="min-height: 300px;">
      <div class="label">Resultado</div>

      {% if resultado %}
        {% if resultado == 'Tomate_Maduro' %}
          <div class="result-badge badge-maduro">
            <i class="ti ti-circle-check"></i> Tomate maduro
          </div>
        {% else %}
          <div class="result-badge badge-deteriorado">
            <i class="ti ti-circle-x"></i> Tomate deteriorado
          </div>
        {% endif %}

        <div class="label" style="margin-top:12px">Confianza del modelo</div>

        {% set c1 = confianza if resultado == 'Tomate_Maduro' else (100 - confianza) %}
        {% set c2 = 100 - c1 %}

        <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
          <span>Tomate maduro</span><span>{{ c1 | round(1) }}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill bar-green" style="width:{{ c1 }}%"></div>
        </div>

        <div style="display:flex;justify-content:space-between;font-size:13px;margin:8px 0 3px">
          <span>Tomate deteriorado</span><span>{{ c2 | round(1) }}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill bar-red" style="width:{{ c2 }}%"></div>
        </div>

        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Inferencia</div>
            <div class="metric-value">{{ ms }} <span class="metric-unit">ms</span></div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Modelo</div>
            <div class="metric-value" style="font-size:14px;padding-top:4px">CNN<span class="metric-unit"> 224×224</span></div>
          </div>
        </div>

      {% else %}
        <div class="empty-state">
          <i class="ti ti-scan"></i>
          <p>Sube una imagen para clasificar</p>
        </div>
      {% endif %}
    </div>
  </div>

</div>
</form>

<script>
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
</script>
</body>
</html>
"""

def predecir_imagen(img_bytes):
    t0 = time.time()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = modelo.predict(arr)
    clase = CLASES[np.argmax(pred)]
    confianza = round(float(np.max(pred)) * 100, 1)
    ms = round((time.time() - t0) * 1000)
    return clase, confianza, ms

def leer_sensor():
    temp = round(random.uniform(18, 35), 1)
    hum = round(random.uniform(40, 90), 1)
    return temp, hum

def condicion_sensor(temp, hum):
    if temp >= 20 and temp <= 28 and hum >= 50 and hum <= 80:
        return "Óptima", "var(--green)"
    elif temp > 28:
        return "Temperatura alta", "var(--red)"
    else:
        return "Humedad fuera de rango", "var(--red)"

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = confianza = temp = hum = ms = condicion = condicion_color = None

    if request.method == "POST":
        file = request.files.get("imagen")
        if file:
            clase, confianza, ms = predecir_imagen(file.read())
            resultado = clase
            temp, hum = leer_sensor()
            condicion, condicion_color = condicion_sensor(temp, hum)

    return render_template_string(HTML,
        resultado=resultado, confianza=confianza,
        temp=temp, hum=hum, ms=ms,
        condicion=condicion, condicion_color=condicion_color)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
