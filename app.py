import os
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = "model/outputs/naira_auth_model.h5"
TRAIN_DIR = "model/dataset/train"
UPLOAD_FOLDER = "static/uploads"
REPORTS_DIR = "model/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.6  # You can adjust this


# ----------------------------
# FLASK APP SETUP
# ----------------------------
app = Flask(__name__)
CORS(app)


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_class_mapping():
    """Auto-detects class mapping from train directory"""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=1,
        class_mode="categorical"
    )
    mapping = {v: k for k, v in generator.class_indices.items()}
    print(f"🧩 Detected class mapping: {mapping}")
    return mapping


def load_trained_model():
    """Loads the trained Keras model"""
    print(f"🔍 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return model


def preprocess_image(image_path, target_size):
    """Loads and preprocesses an image"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
    return img_array


def predict_image(image_path, model, class_mapping):
    """Predicts class of a single image"""
    input_shape = model.input_shape[1:3]
    img_array = preprocess_image(image_path, input_shape)
    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    class_name = class_mapping.get(class_index, "Unknown")

    # Apply threshold
    if confidence < CONFIDENCE_THRESHOLD:
        label = "Uncertain"
    else:
        label = class_name

    return label, confidence


# ----------------------------
# LOAD MODEL AND CLASSES ON STARTUP
# ----------------------------
model = load_trained_model()
class_mapping = get_class_mapping()


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles single image upload and prediction"""
    if "images" not in request.files:
        return redirect(url_for("index"))

    file = request.files["images"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        label, confidence = predict_image(file_path, model, class_mapping)
        image_url = os.path.join("uploads", filename)
        results = [{
            "filename": image_url,
            "label": label,
            "confidence": round(confidence, 2)
        }]
        return render_template("result.html", message="Prediction Complete", results=results, report_filename="N/A")
    except Exception as e:
        print("❌ Prediction Error:", e)
        return redirect(url_for("index"))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    """Handles multiple image uploads and prediction"""
    if "images" not in request.files:
        return redirect(url_for("index"))

    files = request.files.getlist("images")
    results = []

    for file in files:
        if file.filename == "":
            continue

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            label, confidence = predict_image(file_path, model, class_mapping)
            results.append({
                "filename": os.path.join("uploads", filename),
                "label": label,
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            print(f"Error predicting {filename}:", e)
            results.append({
                "filename": os.path.join("uploads", filename),
                "label": "ERROR",
                "confidence": 0.0
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    if results:
        df = pd.DataFrame(results)
        report_path = os.path.join(REPORTS_DIR, "bulk_prediction_report.xlsx")
        df.to_excel(report_path, index=False)
        report_filename = os.path.basename(report_path)
    else:
        report_filename = "N/A"

    return render_template("result.html", message="Bulk Prediction Complete", results=results, report_filename=report_filename)


@app.route("/download_report/<filename>")
def download_report(filename):
    safe_filename = secure_filename(filename)
    try:
        return send_from_directory(REPORTS_DIR, safe_filename, as_attachment=True)
    except FileNotFoundError:
        return "Report not found.", 404


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
