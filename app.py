import os
import numpy as np
import gdown
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf

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


@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return redirect(url_for("index"))

    file = request.files["images"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        result = predict_image_tflite(file_path, interpreter)
        image_url = os.path.join("uploads", filename)
        return render_template(
            "result.html",
            message="Single Prediction Complete",
            results=[{
                "filename": image_url,
                "label": result["predicted_class"],
                "confidence": round(result["confidence"], 2)
            }],
            report_filename="N/A"
        )
    except Exception as e:
        print("Predict error:", e)
        return redirect(url_for("index"))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for file in files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            res = predict_image_tflite(file_path, interpreter)
            results.append({
                "filename": os.path.join("uploads", filename),
                "label": res["predicted_class"],
                "confidence": round(res["confidence"], 2)
            })
        except Exception as e:
            print(f"Error on {filename}:", e)
            results.append({
                "filename": os.path.join("uploads", filename),
                "label": "ERROR",
                "confidence": 0.0
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    if not results:
        return jsonify({"error": "No valid files processed"}), 400

    df = pd.DataFrame(results)
    report_path = os.path.join(REPORTS_DIR, "bulk_prediction_report.xlsx")
    df.to_excel(report_path, index=False)
    report_filename = os.path.basename(report_path)

    return render_template(
        "result.html",
        message="Bulk Prediction Complete!",
        results=results,
        report_filename=report_filename
    )


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
