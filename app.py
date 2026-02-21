from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import json
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64
import io
import uuid
from pyngrok import ngrok

# ================== CONFIG ==================
# Mengarah ke folder 'model' yang baru kamu upload
MODEL_PATH = "model/songke_final.keras" 
CLASS_INDICES_PATH = "model/class_indices.json"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 0.4
IMG_SIZE = 224

# ================== INIT FLASK ==================
app = Flask(__name__)
app.secret_key = "songke_secret"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== LOAD MODEL ==================
print("⏳ Memuat model dari folder model/...")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model berhasil dimuat")
else:
    print(f"❌ ERROR: File {MODEL_PATH} tidak ditemukan!")

# ================== LOAD CLASS LABEL ==================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_label = {int(v): k for k, v in class_indices.items()}
else:
    index_to_label = {
        0: "Motif Jok", 1: "Motif Mata Manuk", 2: "Motif Ntala",
        3: "Motif Ranggong", 4: "Motif Rempa Teke", 5: "Motif Wela Kaweng"
    }

# ================== DESKRIPSI ==================
descriptions = {
    "Motif Jok": "Simbol Persatuan dan Kesatuan. Merefleksikan kehidupan sosial dan hierarki masyarakat Manggarai.",
    "Motif Mata Manuk": "Simbol Keramahan dan Kehormatan. Cerminan dari keikhlasan dalam menyambut tamu.",
    "Motif Ntala": "Melambangkan harapan dan cita-cita yang tinggi setinggi bintang di langit.",
    "Motif Ranggong": "Simbol Keuletan dan Kerja Keras, terinspirasi dari jaring laba-laba di sawah Lodok.",
    "Motif Rempa Teke": "Simbol Kesetiaan Tak Tergoyahkan, melambangkan kesetiaan dalam pernikahan.",
    "Motif Wela Kaweng": "Simbol Kehidupan dan Ketahanan. Mengajarkan harmoni dengan alam sekitar."
}

# ================== FUNGSI PEMBANTU ==================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])
    all_preds = [{"label": index_to_label[i], "value": float(preds[i])} for i in range(len(preds))]
    all_preds.sort(key=lambda x: x["value"], reverse=True)
    return index_to_label[idx], conf, all_preds

# ================== ROUTES ==================
@app.route("/")
def index():
    # Akan mencari file index.html di dalam folder 'templates' yang kamu upload
    return render_template("index.html")

@app.route("/predict_file_ajax", methods=["POST"])
def predict_file_ajax():
    if "file" not in request.files: return jsonify({"error": "File tidak ditemukan"})
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung"})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename}")
    file.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB")
        label, conf, all_preds = predict_image(img)
        os.remove(filepath)
        if conf < CONFIDENCE_THRESHOLD: return jsonify({"error": "Motif tidak dikenali"})
        return jsonify({"class": label, "desc": descriptions.get(label, "-"), "confidence": f"{conf * 100:.2f}%", "all_predictions": all_preds})
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"})

# ================== RUN DENGAN NGROK ==================
if __name__ == "__main__":
    # Menutup koneksi lama jika ada
    ngrok.kill() 
    
    # Membuka tunnel baru di port 5000
    public_url = ngrok.connect(5000).public_url
    print(f"\n✨ LINK WEBSITE KAMU: {public_url} ✨\n")
    
    # Jalankan Flask
    app.run(port=5000)