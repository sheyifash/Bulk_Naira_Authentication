import os
import numpy as np
import gdown
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import tensorflow.lite as tflite

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.tflite"
MODEL_DRIVE_ID = "1NZZw7mgTFUYb5QpYxGMIpT3iHwi0541e"  # your tflite file ID
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

# ============================================================
# INITIALIZE FLASK
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# MODEL HANDLER (TFLITE)
# ============================================================

interpreter = None
input_details = None
output_details = None

def download_model_if_needed():
    """Download model from Google Drive if missing."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading TFLite model from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}",
                MODEL_PATH,
                quiet=False
            )
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

def load_tflite_model():
    """Load TensorFlow Lite model once."""
    global interpreter, input_details, output_details
    if interpreter is None:
        download_model_if_needed()
        try:
            print(f"🔍 Loading TFLite model from: {MODEL_PATH}")
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("✅ TFLite model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            interpreter = None

# Load model once
load_tflite_model()

# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):
    if interpreter is None:
        raise Exception("Model not loaded.")

    # Determine input shape (e.g. (1, 224, 224, 3))
    input_shape = input_details[0]['shape'][1:3]
    img = load_img(image_path, target_size=input_shape)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    class_name = CLASS_MAPPING.get(class_index, "Unknown")

    return {
        "predicted_class": class_name,
        "confidence": confidence
    }

# ============================================================
# ROUTES
# ============================================================

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
        result = predict_image(file_path)
        image_url = os.path.join("uploads", filename)
        return render_template(
            "result.html",
            message="Single Prediction Complete",
            results=[{
                "filename": image_url,
                "label": result["predicted_class"],
                "confidence": round(result["confidence"], 2),
            }],
            report_filename="N/A"
        )
    except Exception as e:
        print(f"❌ Error predicting single file: {e}")
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
            raw_result = predict_image(file_path)
            results.append({
                "filename": os.path.join("uploads", filename),
                "label": raw_result["predicted_class"],
                "confidence": round(raw_result["confidence"], 2)
            })
        except Exception as e:
            print(f"❌ Error predicting file {filename}: {e}")
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

    # Save results to Excel
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

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
