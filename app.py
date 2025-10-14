import os
import uuid
import numpy as np
import pandas as pd
import requests
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# ============== CONFIG ==============
MODEL_LOCAL_PATH = "model.h5"
MODEL_DRIVE_ID = "1Fo7f37x3ALDjxEqp-M55fVBrMjKdLr-L"  # from your Google Drive link
UPLOAD_FOLDER = "static/uploads"
REPORTS_DIR = "model/reports"
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_MAPPING = {
    0: "fake_1000",
    1: "fake_500",
    2: "genuine_1000",
    3: "genuine_500"
}

# ============== FLASK ==============
app = Flask(__name__)

# ============== MODEL LOADING ==============
def download_from_drive_if_missing(drive_id, out_path):
    """Downloads the .h5 model from Google Drive if not present."""
    if os.path.exists(out_path):
        print("✅ Model already exists locally.")
        return

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={drive_id}"
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(url, out_path, quiet=False)
        print("✅ Download complete.")
    except Exception as e:
        print("❌ Failed to download model from Drive:", e)

# Ensure model exists
if not os.path.exists(MODEL_LOCAL_PATH):
    download_from_drive_if_missing(MODEL_DRIVE_ID, MODEL_LOCAL_PATH)

# Load model
model = None
try:
    model = load_model(MODEL_LOCAL_PATH)
    print("✅ .h5 model loaded successfully.")
except Exception as e:
    print("❌ Failed to load .h5 model:", e)

# ============== UTILITIES ==============
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def preprocess_image(img_path):
    """Prepares image for model prediction (resizing, normalization)."""
    img = Image.open(img_path).convert("RGB").resize((224, 224))  # adjust to match training
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    return arr

def predict_image(filepath):
    """Predicts class for a single image using .h5 model."""
    if model is None:
        raise RuntimeError("Model not loaded.")

    img_array = preprocess_image(filepath)
    preds = model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = CLASS_MAPPING.get(class_index, f"class_{class_index}")
    return label, confidence

# ============== ROUTES ==============
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/")
def index_redirect():
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "images" not in request.files:
            return render_template("result.html", message="❌ No file uploaded", results=[], report_filename="N/A")

        file = request.files["images"]
        if file.filename == "":
            return render_template("result.html", message="❌ Empty filename", results=[], report_filename="N/A")

        if not allowed_file(file.filename):
            return render_template("result.html", message="❌ Unsupported file extension", results=[], report_filename="N/A")

        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(saved_path)

        label, confidence = predict_image(saved_path)
        result = [{
            "filename": f"uploads/{unique_name}",
            "label": label,
            "confidence": round(confidence, 2)
        }]

        os.remove(saved_path)
        return render_template("result.html", message="✅ Single Prediction Complete", results=result, report_filename="N/A")

    except Exception as e:
        print("ERROR in /predict:", e)
        return render_template("result.html", message=f"Error: {e}", results=[], report_filename="N/A")

@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    try:
        if "images" not in request.files:
            return render_template("result.html", message="❌ No files uploaded", results=[], report_filename="N/A")

        files = request.files.getlist("images")
        results = []

        for file in files:
            if not file or file.filename == "" or not allowed_file(file.filename):
                continue

            ext = os.path.splitext(secure_filename(file.filename))[1].lower()
            unique_name = f"{uuid.uuid4().hex}{ext}"
            saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(saved_path)

            try:
                label, confidence = predict_image(saved_path)
                results.append({
                    "filename": f"uploads/{unique_name}",
                    "label": label,
                    "confidence": round(confidence, 2)
                })
            except Exception as sub_e:
                print(f"Prediction error for {file.filename}:", sub_e)
                results.append({
                    "filename": f"uploads/{unique_name}",
                    "label": "ERROR",
                    "confidence": 0.0
                })
            finally:
                try:
                    os.remove(saved_path)
                except Exception:
                    pass

        if not results:
            return render_template("result.html", message="⚠️ No valid images processed", results=[], report_filename="N/A")

        # Save Excel report
        df = pd.DataFrame(results)
        report_name = f"bulk_prediction_report_{uuid.uuid4().hex[:8]}.xlsx"
        report_path = os.path.join(REPORTS_DIR, report_name)
        df.to_excel(report_path, index=False)

        return render_template("result.html", message="📦 Bulk Prediction Complete", results=results, report_filename=report_name)

    except Exception as e:
        print("ERROR in /predict_bulk:", e)
        return render_template("result.html", message=f"Error: {e}", results=[], report_filename="N/A")

@app.route("/download_report/<filename>")
def download_report(filename):
    safe = secure_filename(filename)
    full = os.path.join(REPORTS_DIR, safe)
    if not os.path.exists(full):
        return "Report not found", 404
    return send_from_directory(REPORTS_DIR, safe, as_attachment=True)

# ============== RUN ==============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
