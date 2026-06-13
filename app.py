import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import io
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# MODELO (LAZY LOADING PARA EVITAR CRASH EN RENDER)
# ─────────────────────────────────────────────
modelo = None

def cargar_modelo():
    global modelo
    if modelo is None:
        modelo = tf.keras.models.load_model("modelo_tomates.h5")
    return modelo

CLASES = ["Tomate_Deteriorado", "Tomate_Maduro"]

# ─────────────────────────────────────────────
# INTERFAZ WEB
# ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Clasificador de Tomates</title>
</head>
<body style="font-family:Arial; text-align:center; background:#0f172a; color:white;">

    <h1>Clasificador de Alimentos</h1>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="imagen" required>
        <br><br>
        <button type="submit">Predecir</button>
    </form>

    {% if resultado %}
        <h2>Resultado: {{ resultado }}</h2>
        <h3>Confianza: {{ confianza }}%</h3>

        <h3>📊 Datos del sensor DHT22 (IoT simulado)</h3>
        <p>🌡 Temperatura: {{ temp }} °C</p>
        <p>💧 Humedad: {{ hum }} %</p>
    {% endif %}

</body>
</html>
"""

# ─────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
def predecir_imagen(img_bytes):
    modelo = cargar_modelo()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr, verbose=0)

    clase = CLASES[np.argmax(pred)]
    confianza = float(np.max(pred)) * 100

    return clase, round(confianza, 2)


# ─────────────────────────────────────────────
# RUTA PRINCIPAL
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    confianza = None
    temp = None
    hum = None

    if request.method == "POST":
        file = request.files["imagen"]
        img_bytes = file.read()

        resultado, confianza = predecir_imagen(img_bytes)

        # 🔧 DATOS SIMULADOS (ESP32-CAM futuro)
        import random
        temp = round(random.uniform(18, 35), 1)
        hum = round(random.uniform(40, 90), 1)

    return render_template_string(
        HTML,
        resultado=resultado,
        confianza=confianza,
        temp=temp,
        hum=hum
    )


# ─────────────────────────────────────────────
# INICIO (RENDER COMPATIBLE)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
