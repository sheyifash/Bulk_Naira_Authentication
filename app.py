import os
import numpy as np
import gdown
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tflite_support.task import core

# -------------------------
# CONFIG
# -------------------------
MODEL_DRIVE_ID = "1NZZw7mgTFUYb5QpYxGMIpT3iHwi0541e"   # Google Drive file ID for your TFLite model
TFLITE_PATH = "model.tflite"
UPLOAD_FOLDER = "static/uploads"
REPORTS_DIR = "model/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_MAPPING = {
    0: "fake_1000",
    1: "fake_500",
    2: "genuine_1000",
    3: "genuine_500"
}

# -------------------------
# FLASK
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# TFLite model handling
# -------------------------
def download_tflite_if_needed():
    """Download model from Google Drive if not found locally."""
    if not os.path.exists(TFLITE_PATH):
        print("📥 Downloading TFLite model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, TFLITE_PATH, quiet=False)


def load_interpreter(path):
    """Load TensorFlow Lite interpreter."""
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        print("✅ TensorFlow Lite interpreter loaded successfully")
        return interpreter
    except Exception as e:
        print("❌ Failed to load TFLite interpreter:", e)
        return None


# Download and load once at startup
download_tflite_if_needed()
interpreter = load_interpreter(TFLITE_PATH)


def softmax(x):
    """Softmax utility."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_image_tflite(image_path, interpreter):
    """Predict class of a single image."""
    if interpreter is None:
        raise RuntimeError("Model interpreter not loaded")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]['shape']
    h, w = int(in_shape[1]), int(in_shape[2])

    img = Image.open(image_path).convert("RGB").resize((w, h))
    arr = np.array(img)

    input_dtype = input_details[0]['dtype']
    if np.issubdtype(input_dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)

    input_data = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]['index'])
    scale, zero_point = output_details[0].get('quantization', (0.0, 0))

    if scale and scale != 0:
        logits = (raw_output.astype(np.float32) - zero_point) * scale
    else:
        logits = raw_output.astype(np.float32)

    probs = logits[0]
    if not (probs.sum() <= 1.0 + 1e-6 and probs.max() <= 1.0):
        probs = softmax(probs)

    class_index = int(np.argmax(probs))
    confidence = float(np.max(probs))
    class_name = CLASS_MAPPING.get(class_index, f"class_{class_index}")

    return {"predicted_class": class_name, "confidence": confidence}


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load your model once (outside routes)
MODEL_PATH = "model/outputs/naira_auth_model.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully for prediction!")

# ----------------------------------------------------
# SINGLE PREDICTION
# ----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "images" not in request.files:
            return render_template("result.html", message="❌ No image uploaded", results=[], report_filename="N/A")

        file = request.files["images"]
        if file.filename == "":
            return render_template("result.html", message="⚠️ Empty file name", results=[], report_filename="N/A")

        filename = secure_filename(file.filename)
        filepath = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        img = load_img(filepath, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        label = CLASS_MAPPING[class_index]

        result = [{
            "filename": f"uploads/{filename}",
            "label": label,
            "confidence": confidence
        }]

        return render_template(
            "result.html",
            message="✅ Single Prediction Complete",
            results=result,
            report_filename="N/A"
        )

    except Exception as e:
        print("❌ Error in /predict:", e)
        return render_template("result.html", message=f"Error: {e}", results=[], report_filename="N/A")


# ----------------------------------------------------
# BULK PREDICTION
# ----------------------------------------------------
@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    try:
        if "images" not in request.files:
            return render_template("result.html", message="❌ No files uploaded", results=[], report_filename="N/A")

        files = request.files.getlist("images")
        results = []
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        for file in files:
            if file.filename == "":
                continue

            filepath = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(filepath)

            img = load_img(filepath, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array, verbose=0)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            label = CLASS_MAPPING[class_index]

            results.append({
                "filename": f"uploads/{file.filename}",
                "label": label,
                "confidence": confidence
            })

        if not results:
            return render_template("result.html", message="⚠️ No valid images found", results=[], report_filename="N/A")

        df = pd.DataFrame(results)
        report_folder = os.path.join("model", "reports")
        os.makedirs(report_folder, exist_ok=True)
        report_filename = "bulk_prediction_report.xlsx"
        df.to_excel(os.path.join(report_folder, report_filename), index=False)

        return render_template(
            "result.html",
            message="📦 Bulk Prediction Complete!",
            results=results,
            report_filename=report_filename
        )

    except Exception as e:
        print("❌ Error in /predict_bulk:", e)
        return render_template("result.html", message=f"Error: {e}", results=[], report_filename="N/A")


@app.route("/download_report/<filename>")
def download_report(filename):
    safe_filename = secure_filename(filename)
    try:
        return send_from_directory(REPORTS_DIR, safe_filename, as_attachment=True)
    except FileNotFoundError:
        return "Report not found.", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
