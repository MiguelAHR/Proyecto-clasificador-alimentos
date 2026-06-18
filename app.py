import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import io
import os

app = Flask(__name__)

modelo = None
CLASES = [
    "Lechuga_Deteriorada",
    "Lechuga_Fresca",
    "Tomate_Deteriorado",
    "Tomate_Maduro",
    "Zanahoria_Deteriorada",
    "Zanahoria_Fresca",
]

INFO_CLASES = {
    "Lechuga_Fresca":        {"label": "Lechuga fresca",        "estado": "fresco",      "contenedor": "lechuga"},
    "Lechuga_Deteriorada":   {"label": "Lechuga deteriorada",   "estado": "deteriorado", "contenedor": "lechuga"},
    "Tomate_Maduro":         {"label": "Tomate maduro",         "estado": "fresco",      "contenedor": "tomate"},
    "Tomate_Deteriorado":    {"label": "Tomate deteriorado",    "estado": "deteriorado", "contenedor": "tomate"},
    "Zanahoria_Fresca":      {"label": "Zanahoria fresca",      "estado": "fresco",      "contenedor": "zanahoria"},
    "Zanahoria_Deteriorada": {"label": "Zanahoria deteriorada", "estado": "deteriorado", "contenedor": "zanahoria"},
}

# Rangos óptimos por alimento
RANGOS = {
    "tomate":    {"t_min": 20, "t_max": 28, "h_min": 50, "h_max": 80},
    "lechuga":   {"t_min":  2, "t_max":  8, "h_min": 90, "h_max": 98},
    "zanahoria": {"t_min":  0, "t_max":  5, "h_min": 90, "h_max": 95},
}

def get_modelo():
    global modelo
    if modelo is None:
        modelo = tf.keras.models.load_model("modelo_tomates.h5")
    return modelo

def generar_recomendaciones(resultado, temp_t, hum_t, temp_l, hum_l, temp_z, hum_z):
    if not resultado:
        return []
    info = INFO_CLASES[resultado]
    estado = info["estado"]
    cont = info["contenedor"]
    label = info["label"]

    if estado == "deteriorado":
        return [{"tipo": "peligro", "icono": "ti-trash", "texto": f"El alimento detectado ({label}) está en mal estado. Se recomienda desecharlo inmediatamente para evitar contaminación."}]

    temp_map = {"tomate": temp_t, "lechuga": temp_l, "zanahoria": temp_z}
    hum_map  = {"tomate": hum_t,  "lechuga": hum_l,  "zanahoria": hum_z}
    r = RANGOS[cont]
    temp = temp_map[cont]
    hum  = hum_map[cont]
    recs = []

    if temp < r["t_min"]:
        diff = round(r["t_min"] - temp, 1)
        recs.append({"tipo": "warn", "icono": "ti-temperature-plus", "texto": f"Temperatura del contenedor de {cont} muy baja ({temp}°C). Sube {diff}°C para alcanzar el mínimo óptimo de {r['t_min']}°C."})
    elif temp > r["t_max"]:
        diff = round(temp - r["t_max"], 1)
        recs.append({"tipo": "warn", "icono": "ti-temperature-minus", "texto": f"Temperatura del contenedor de {cont} muy alta ({temp}°C). Baja {diff}°C para no superar el máximo de {r['t_max']}°C."})
    else:
        recs.append({"tipo": "ok", "icono": "ti-circle-check", "texto": f"Temperatura del contenedor de {cont} en rango óptimo ({temp}°C). Mantener entre {r['t_min']}–{r['t_max']}°C."})

    if hum < r["h_min"]:
        diff = round(r["h_min"] - hum, 1)
        recs.append({"tipo": "warn", "icono": "ti-droplet-plus", "texto": f"Humedad del contenedor de {cont} muy baja ({hum}%). Aumenta {diff}% para alcanzar el mínimo de {r['h_min']}%."})
    elif hum > r["h_max"]:
        diff = round(hum - r["h_max"], 1)
        recs.append({"tipo": "warn", "icono": "ti-droplet-minus", "texto": f"Humedad del contenedor de {cont} muy alta ({hum}%). Reduce {diff}% para no superar el máximo de {r['h_max']}%."})
    else:
        recs.append({"tipo": "ok", "icono": "ti-circle-check", "texto": f"Humedad del contenedor de {cont} en rango óptimo ({hum}%). Mantener entre {r['h_min']}–{r['h_max']}%."})

    return recs

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clasificador de Alimentos</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0f172a; --surface: #1e293b; --surface2: #273449;
      --border: rgba(255,255,255,0.08); --text: #f1f5f9; --muted: #94a3b8;
      --green: #22c55e; --green-bg: rgba(34,197,94,0.12);
      --red: #f87171; --red-bg: rgba(248,113,113,0.12);
      --yellow: #facc15; --yellow-bg: rgba(250,204,21,0.12);
      --accent: #1D9E75; --radius-md: 10px; --radius-lg: 14px;
    }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 1.5rem 1rem; }
    header { text-align: center; margin-bottom: 2rem; }
    header h1 { font-size: 22px; font-weight: 500; margin-bottom: 4px; }
    header p { font-size: 14px; color: var(--muted); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; max-width: 960px; margin: 0 auto; }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--surface); border: 0.5px solid var(--border); border-radius: var(--radius-lg); padding: 1.25rem; }
    .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; margin-bottom: 12px; }
    .upload-zone { border: 1.5px dashed rgba(255,255,255,0.18); border-radius: var(--radius-lg); padding: 2rem 1rem; text-align: center; cursor: pointer; background: var(--surface2); transition: border-color 0.2s; }
    .upload-zone:hover { border-color: var(--accent); }
    .upload-zone i { font-size: 32px; color: var(--muted); display: block; margin-bottom: 8px; }
    .upload-zone p { font-size: 14px; color: var(--muted); }
    .upload-zone small { font-size: 12px; color: rgba(148,163,184,0.5); }
    #preview { width: 100%; height: 200px; object-fit: cover; border-radius: var(--radius-md); display: none; margin-bottom: 0.75rem; }
    #changeBtn { display: none; }
    .btn { width: 100%; margin-top: 0.75rem; padding: 10px; font-size: 14px; font-weight: 500; background: var(--surface2); border: 0.5px solid rgba(255,255,255,0.12); border-radius: var(--radius-md); cursor: pointer; color: var(--text); display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn:hover { background: rgba(255,255,255,0.08); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: white; }
    .btn.primary:hover { background: #0F6E56; }
    input[type=file] { display: none; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    /* contenedores */
    .contenedores { display: flex; flex-direction: column; gap: 0.75rem; }
    .cont-box { border: 1.5px solid rgba(255,255,255,0.10); border-radius: var(--radius-lg); padding: 0.75rem 1rem; background: var(--surface2); position: relative; }
    .cont-box.activo { border-color: var(--accent); }
    .cont-tag { position: absolute; top: -9px; left: 10px; background: var(--surface); padding: 0 6px; font-size: 10px; color: var(--muted); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
    .cont-box.activo .cont-tag { color: var(--accent); }
    .cont-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .cont-info { display: flex; align-items: center; gap: 8px; }
    .cont-emoji { font-size: 22px; }
    .cont-name { font-size: 13px; font-weight: 500; }
    .cont-sub  { font-size: 10px; color: var(--muted); }
    .cond-badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 99px; font-size: 10px; font-weight: 500; }
    .cond-ok   { background: rgba(34,197,94,0.15);  color: var(--green); }
    .cond-warn { background: rgba(250,204,21,0.15); color: var(--yellow); }
    .cond-bad  { background: rgba(248,113,113,0.15); color: var(--red); }
    .cont-knobs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
    .knob-card { display: flex; flex-direction: column; align-items: center; gap: 4px; }
    .knob-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
    .knob-wrap { position: relative; width: 72px; height: 72px; }
    .knob-svg { width: 72px; height: 72px; }
    .knob-val { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; pointer-events: none; }
    .knob-num { font-size: 14px; font-weight: 500; }
    .knob-unit { font-size: 9px; color: var(--muted); }

    /* resultado */
    .result-badge { padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; margin-bottom: 14px; }
    .badge-fresco { background: var(--green-bg); color: var(--green); }
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

    /* recomendaciones */
    .rec-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: var(--radius-md); margin-bottom: 8px; font-size: 13px; line-height: 1.5; }
    .rec-ok   { background: var(--green-bg); color: var(--green); }
    .rec-warn { background: var(--yellow-bg); color: var(--yellow); }
    .rec-peligro { background: var(--red-bg); color: var(--red); }
    .rec-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
    .full-row { grid-column: 1 / -1; }
  </style>
</head>
<body>

<header>
  <h1><i class="ti ti-plant-2" style="font-size:20px;vertical-align:-2px;margin-right:6px;color:#1D9E75"></i>Clasificador de alimentos</h1>
  <p>Modelo CNN &middot; ESP32-CAM + DHT22 &middot; Tiempo real</p>
</header>

<form method="POST" enctype="multipart/form-data">
<div class="grid">

  <!-- COLUMNA IZQUIERDA -->
  <div>
    <div class="card">
      <div class="lbl">Imagen de entrada</div>
      <div class="upload-zone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <i class="ti ti-photo-up"></i>
        <p>Haz clic para seleccionar</p>
        <small>JPG, PNG &middot; máx. 5 MB</small>
      </div>
      <img id="preview" alt="Vista previa">
      <input type="file" id="fileInput" name="imagen" accept="image/*" required onchange="handleFile(this)">
      <button type="button" class="btn" id="changeBtn" onclick="cambiarFoto()" style="display:none">
        <i class="ti ti-photo-edit"></i> Cambiar foto
      </button>
      <button type="submit" class="btn primary">
        <i class="ti ti-scan"></i> Analizar imagen
      </button>
    </div>

    <!-- CONTENEDORES IoT -->
    <div class="card" style="margin-top:1rem">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
        <div class="lbl" style="margin-bottom:0">Contenedores de almacenamiento</div>
        <span style="font-size:12px;color:var(--muted)"><span class="status-dot"></span>ESP32 en línea</span>
      </div>
      <div class="contenedores">

        <!-- TOMATE -->
        <div class="cont-box {% if resultado and info[resultado].contenedor == 'tomate' %}activo{% endif %}" id="box-tomate">
          <span class="cont-tag">A-1 · Tomates</span>
          <div class="cont-head">
            <div class="cont-info">
              <span class="cont-emoji">🍅</span>
              <div>
                <div class="cont-name">Tomates</div>
                <div class="cont-sub">Óptimo: 20–28°C · 50–80%</div>
              </div>
            </div>
            <div class="cond-badge" id="cond-tomate"><i class="ti ti-circle-check" style="font-size:11px"></i><span id="condTxt-tomate">—</span></div>
          </div>
          <div class="cont-knobs">
            <div class="knob-card">
              <div class="knob-lbl">Temp.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#f87171" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="110" transform="rotate(135 36 36)" id="arc-t-tomate"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-t-tomate">24</span><span class="knob-unit">°C</span></div>
              </div>
            </div>
            <div class="knob-card">
              <div class="knob-lbl">Hum.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#60a5fa" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="73" transform="rotate(135 36 36)" id="arc-h-tomate"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-h-tomate">60</span><span class="knob-unit">%</span></div>
              </div>
            </div>
          </div>
        </div>

        <!-- LECHUGA -->
        <div class="cont-box {% if resultado and info[resultado].contenedor == 'lechuga' %}activo{% endif %}" id="box-lechuga">
          <span class="cont-tag">B-1 · Lechugas</span>
          <div class="cont-head">
            <div class="cont-info">
              <span class="cont-emoji">🥬</span>
              <div>
                <div class="cont-name">Lechugas</div>
                <div class="cont-sub">Óptimo: 2–8°C · 90–98%</div>
              </div>
            </div>
            <div class="cond-badge" id="cond-lechuga"><i class="ti ti-circle-check" style="font-size:11px"></i><span id="condTxt-lechuga">—</span></div>
          </div>
          <div class="cont-knobs">
            <div class="knob-card">
              <div class="knob-lbl">Temp.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#f87171" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="120" transform="rotate(135 36 36)" id="arc-t-lechuga"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-t-lechuga">5</span><span class="knob-unit">°C</span></div>
              </div>
            </div>
            <div class="knob-card">
              <div class="knob-lbl">Hum.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#60a5fa" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="50" transform="rotate(135 36 36)" id="arc-h-lechuga"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-h-lechuga">93</span><span class="knob-unit">%</span></div>
              </div>
            </div>
          </div>
        </div>

        <!-- ZANAHORIA -->
        <div class="cont-box {% if resultado and info[resultado].contenedor == 'zanahoria' %}activo{% endif %}" id="box-zanahoria">
          <span class="cont-tag">C-1 · Zanahorias</span>
          <div class="cont-head">
            <div class="cont-info">
              <span class="cont-emoji">🥕</span>
              <div>
                <div class="cont-name">Zanahorias</div>
                <div class="cont-sub">Óptimo: 0–5°C · 90–95%</div>
              </div>
            </div>
            <div class="cond-badge" id="cond-zanahoria"><i class="ti ti-circle-check" style="font-size:11px"></i><span id="condTxt-zanahoria">—</span></div>
          </div>
          <div class="cont-knobs">
            <div class="knob-card">
              <div class="knob-lbl">Temp.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#f87171" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="126" transform="rotate(135 36 36)" id="arc-t-zanahoria"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-t-zanahoria">3</span><span class="knob-unit">°C</span></div>
              </div>
            </div>
            <div class="knob-card">
              <div class="knob-lbl">Hum.</div>
              <div class="knob-wrap">
                <svg class="knob-svg" viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="36" transform="rotate(135 36 36)"/>
                  <circle cx="36" cy="36" r="28" fill="none" stroke="#60a5fa" stroke-width="5" stroke-linecap="round" stroke-dasharray="146" stroke-dashoffset="51" transform="rotate(135 36 36)" id="arc-h-zanahoria"/>
                </svg>
                <div class="knob-val"><span class="knob-num" id="num-h-zanahoria">92</span><span class="knob-unit">%</span></div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- COLUMNA DERECHA -->
  <div style="display:flex;flex-direction:column;gap:1rem">

    <!-- RESULTADO -->
    <div class="card" style="min-height:300px">
      <div class="lbl">Resultado</div>
      {% if resultado %}
        {% set estado = info[resultado].estado %}
        {% set label  = info[resultado].label %}
        <div class="result-badge {{ 'badge-fresco' if estado == 'fresco' else 'badge-deteriorado' }}">
          <i class="ti {{ 'ti-circle-check' if estado == 'fresco' else 'ti-circle-x' }}"></i>
          {{ label }}
        </div>
        <div class="lbl" style="margin-top:12px">Probabilidades por categoría</div>
        {% for clase, prob in probs.items() | sort(attribute='1', reverse=True) %}
          {% set es = info[clase].estado %}
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;margin-top:6px">
            <span style="{{ 'font-weight:600' if clase == resultado else '' }}">{{ info[clase].label }}</span>
            <span>{{ prob }}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill {{ 'bar-green' if es == 'fresco' else 'bar-red' }}"
                 style="width:{{ prob }}%;{{ 'opacity:1' if clase == resultado else 'opacity:0.4' }}"></div>
          </div>
        {% endfor %}
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Confianza</div>
            <div class="metric-value">{{ confianza }}<span class="metric-unit">%</span></div>
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

    <!-- RECOMENDACIONES -->
    {% if recomendaciones %}
    <div class="card">
      <div class="lbl">Recomendaciones</div>
      {% for rec in recomendaciones %}
        <div class="rec-item rec-{{ rec.tipo }}">
          <i class="ti {{ rec.icono }} rec-icon"></i>
          <span>{{ rec.texto }}</span>
        </div>
      {% endfor %}
    </div>
    {% endif %}

  </div>

</div>

<input type="hidden" id="h-t-tomate"    name="t_tomate"    value="24">
<input type="hidden" id="h-h-tomate"    name="h_tomate"    value="60">
<input type="hidden" id="h-t-lechuga"   name="t_lechuga"   value="5">
<input type="hidden" id="h-h-lechuga"   name="h_lechuga"   value="93">
<input type="hidden" id="h-t-zanahoria" name="t_zanahoria" value="3">
<input type="hidden" id="h-h-zanahoria" name="h_zanahoria" value="92">
</form>

<script>
const ARC = 146, GAP = 36;

const RANGOS = {
  tomate:    {tMin:20, tMax:28, hMin:50, hMax:80},
  lechuga:   {tMin:2,  tMax:8,  hMin:90, hMax:98},
  zanahoria: {tMin:0,  tMax:5,  hMin:90, hMax:95},
};

const LIMITES = {
  tomate:    {tRange:50, hRange:100, tInit:24, hInit:60},
  lechuga:   {tRange:20, hRange:100, tInit:5,  hInit:93},
  zanahoria: {tRange:15, hRange:100, tInit:3,  hInit:92},
};

// Estado actual de cada contenedor (fluctúa automáticamente)
const state = {};
for (const k of ['tomate','lechuga','zanahoria']) {
  state[k] = {
    t: LIMITES[k].tInit,
    h: LIMITES[k].hInit,
    tDir: 0.3,
    hDir: 0.4,
  };
}

function setArc(id, val, maxVal) {
  const el = document.getElementById(id);
  if (!el) return;
  const used = Math.max(0, Math.min(1, val / maxVal)) * (ARC - GAP);
  el.setAttribute('stroke-dashoffset', Math.round(ARC - used));
}

function getCondClass(t, h, rango) {
  const tOk = t >= rango.tMin && t <= rango.tMax;
  const hOk = h >= rango.hMin && h <= rango.hMax;
  if (tOk && hOk) return {cls:'cond-ok', icon:'ti-circle-check', txt:'Óptimo'};
  if (!tOk && !hOk) return {cls:'cond-bad', icon:'ti-circle-x', txt:'Fuera de rango'};
  if (!tOk) return {cls:'cond-warn', icon:'ti-alert-triangle', txt: t > rango.tMax ? 'Temp. alta' : 'Temp. baja'};
  return {cls:'cond-warn', icon:'ti-alert-triangle', txt: h > rango.hMax ? 'Hum. alta' : 'Hum. baja'};
}

function renderContenedor(nombre) {
  const s = state[nombre];
  const lim = LIMITES[nombre];
  const r = RANGOS[nombre];
  const t = parseFloat(s.t.toFixed(1));
  const h = parseFloat(s.h.toFixed(1));

  document.getElementById('num-t-' + nombre).textContent = t;
  document.getElementById('num-h-' + nombre).textContent = h;
  setArc('arc-t-' + nombre, t, lim.tRange);
  setArc('arc-h-' + nombre, h, lim.hRange);

  const c = getCondClass(t, h, r);
  const badge = document.getElementById('cond-' + nombre);
  badge.className = 'cond-badge ' + c.cls;
  badge.querySelector('i').className = 'ti ' + c.icon + ' ' + 'font-size:11px';
  document.getElementById('condTxt-' + nombre).textContent = c.txt;

  document.getElementById('h-t-' + nombre).value = t;
  document.getElementById('h-h-' + nombre).value = h;
}

function fluctuar() {
  for (const k of ['tomate','lechuga','zanahoria']) {
    const s = state[k];
    const lim = LIMITES[k];
    s.t += s.tDir + (Math.random() - 0.5) * 0.3;
    s.h += s.hDir + (Math.random() - 0.5) * 0.5;
    if (s.t >= lim.tRange * 0.9 || s.t < 0)   s.tDir *= -1;
    if (s.h >= 99 || s.h < 1) s.hDir *= -1;
    s.t = Math.max(0, Math.min(lim.tRange, s.t));
    s.h = Math.max(1, Math.min(99, s.h));
    renderContenedor(k);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  {% if temp_t %}
  state.tomate.t    = {{ temp_t }};
  state.tomate.h    = {{ hum_t }};
  state.lechuga.t   = {{ temp_l }};
  state.lechuga.h   = {{ hum_l }};
  state.zanahoria.t = {{ temp_z }};
  state.zanahoria.h = {{ hum_z }};
  {% endif %}
  for (const k of ['tomate','lechuga','zanahoria']) renderContenedor(k);
  setInterval(fluctuar, 1500);

  const saved = sessionStorage.getItem('tomato_img');
  {% if resultado %}
  if (saved) {
    document.getElementById('preview').src = saved;
    document.getElementById('preview').style.display = 'block';
    document.getElementById('dropzone').style.display = 'none';
    document.getElementById('changeBtn').style.display = 'flex';
  }
  {% else %}
  sessionStorage.removeItem('tomato_img');
  {% endif %}
});

function handleFile(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    const canvas = document.createElement('canvas');
    canvas.width = 224; canvas.height = 224;
    const ctx = canvas.getContext('2d');
    const tmpImg = new Image();
    tmpImg.onload = () => {
      ctx.drawImage(tmpImg, 0, 0, 224, 224);
      sessionStorage.setItem('tomato_img', canvas.toDataURL('image/jpeg', 0.7));
    };
    tmpImg.src = dataUrl;
    document.getElementById('preview').src = dataUrl;
    document.getElementById('preview').style.display = 'block';
    document.getElementById('dropzone').style.display = 'none';
    document.getElementById('changeBtn').style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function cambiarFoto() {
  sessionStorage.removeItem('tomato_img');
  document.getElementById('fileInput').click();
}
</script>
</body>
</html>
"""

def predecir_imagen(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = get_modelo().predict(arr)[0]
    idx = int(np.argmax(pred))
    clase = CLASES[idx]
    confianza = float(pred[idx]) * 100
    probs = {CLASES[i]: round(float(pred[i]) * 100, 1) for i in range(len(CLASES))}
    return clase, round(confianza, 2), probs

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = confianza = probs = None
    temp_t = hum_t = temp_l = hum_l = temp_z = hum_z = None
    recomendaciones = []

    if request.method == "POST":
        file = request.files["imagen"]
        resultado, confianza, probs = predecir_imagen(file.read())
        temp_t = float(request.form.get("t_tomate",    24))
        hum_t  = float(request.form.get("h_tomate",    60))
        temp_l = float(request.form.get("t_lechuga",    5))
        hum_l  = float(request.form.get("h_lechuga",   93))
        temp_z = float(request.form.get("t_zanahoria",  3))
        hum_z  = float(request.form.get("h_zanahoria", 92))
        recomendaciones = generar_recomendaciones(resultado, temp_t, hum_t, temp_l, hum_l, temp_z, hum_z)

    return render_template_string(HTML,
        resultado=resultado, confianza=confianza, probs=probs, info=INFO_CLASES,
        temp_t=temp_t, hum_t=hum_t, temp_l=temp_l, hum_l=hum_l,
        temp_z=temp_z, hum_z=hum_z, recomendaciones=recomendaciones)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
