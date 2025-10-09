# app.py (TFLite-ready Flask app)
import os
import uuid
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# ============== CONFIG ==============
MODEL_LOCAL_PATH = "model.tflite"   # local path to tflite file
MODEL_DRIVE_ID = "1NZZw7mgTFUYb5QpYxGMIpT3iHwi0541e"  # optional: Drive file id to auto-download
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

# ============== TFLITE INTERPRETER LOADER ==============
def download_from_drive_if_missing(drive_id, out_path):
    try:
        import gdown
    except Exception:
        print("gdown not available; skipping auto-download.")
        return False
    url = f"https://drive.google.com/uc?id={drive_id}"
    try:
        gdown.download(url, out_path, quiet=False)
        return True
    except Exception as e:
        print("Failed to download model from Drive:", e)
        return False

def get_tflite_interpreter(model_path):
    Interpreter = None
    # Try tflite_runtime first (lightweight)
    try:
        import tflite_runtime.interpreter as tflite_rt
        Interpreter = tflite_rt.Interpreter
        print("Using tflite_runtime.Interpreter")
    except Exception:
        # Try tensorflow fallback
        try:
            from tensorflow.lite import Interpreter as TFInterpreter
            Interpreter = TFInterpreter
            print("Using tensorflow.lite.Interpreter")
        except Exception:
            try:
                # alternate import path
                from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter2
                Interpreter = TFInterpreter2
                print("Using tensorflow.lite.python.interpreter.Interpreter")
            except Exception as e:
                print("No TFLite interpreter available (tflite_runtime or tensorflow.lite).", e)
                Interpreter = None

    if Interpreter is None:
        return None

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Ensure model is present; attempt download if missing and Drive ID provided
if not os.path.exists(MODEL_LOCAL_PATH) and MODEL_DRIVE_ID:
    print("Model file missing locally. Attempting to download from Google Drive...")
    download_from_drive_if_missing(MODEL_DRIVE_ID, MODEL_LOCAL_PATH)

interpreter = None
if os.path.exists(MODEL_LOCAL_PATH):
    try:
        interpreter = get_tflite_interpreter(MODEL_LOCAL_PATH)
        if interpreter:
            print("✅ TFLite interpreter loaded.")
        else:
            print("❌ Interpreter not available; predictions will fail.")
    except Exception as e:
        print("Failed to init tflite interpreter:", e)
else:
    print(f"Model file not found at {MODEL_LOCAL_PATH}. Place your .tflite there or set MODEL_DRIVE_ID to auto-download.")

# ============== UTILITIES ==============
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess_for_interpreter(img_path, input_details):
    """
    Returns numpy array ready to set_tensor for the interpreter.
    Handles float and quantized inputs.
    """
    # Determine required input height/width
    shape = input_details[0]['shape']  # e.g., [1, h, w, 3]
    # Some tflite builds put dynamic dims (-1 or 0); fallback to shape_signature or default
    try:
        h, w = int(shape[1]), int(shape[2])
        if h <= 0 or w <= 0:
            raise Exception("invalid dims")
    except Exception:
        # try shape_signature
        ss = input_details[0].get('shape_signature')
        if ss is not None and len(ss) >= 3:
            h, w = int(ss[1]), int(ss[2])
        else:
            # fallback
            h, w = 224, 224

    img = Image.open(img_path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(img)  # uint8 0-255 by default

    # Input dtype from model
    input_dtype = input_details[0]['dtype']
    quant = input_details[0].get('quantization', (0.0, 0))
    scale, zero_point = quant if quant is not None else (0.0, 0)

    if np.issubdtype(input_dtype, np.floating):
        # float model — normalize 0..1
        arr_f = arr.astype(np.float32) / 255.0
        input_data = np.expand_dims(arr_f, axis=0).astype(np.float32)
    else:
        # integer (likely uint8) quantized model
        # Many quantized models were quantized from normalized floats (0..1)
        # So convert to float 0..1 first, then quantize with scale/zero_point
        if scale and scale != 0:
            arr_f = arr.astype(np.float32) / 255.0
            q = np.round(arr_f / scale + zero_point).astype(input_dtype)
            input_data = np.expand_dims(q, axis=0)
        else:
            # no quant params; give raw uint8 0-255
            input_data = np.expand_dims(arr.astype(input_dtype), axis=0)

    return input_data, (h, w)

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])

    # Handle quantized outputs if present
    out_quant = output_details[0].get('quantization', (0.0, 0))
    out_scale, out_zero_point = out_quant if out_quant is not None else (0.0, 0)
    if out_scale and out_scale != 0:
        logits = (raw_output.astype(np.float32) - out_zero_point) * out_scale
    else:
        logits = raw_output.astype(np.float32)

    # If logits appear not normalized, apply softmax
    probs = logits[0]
    # If sum approx 1 and values within [0,1], assume already probs
    if not (0.99 <= probs.sum() <= 1.01 and probs.min() >= -1e-6 and probs.max() <= 1.0 + 1e-6):
        probs = softmax(probs)

    class_index = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return class_index, confidence, probs

# ============== PREDICTION WRAPPERS ==============
def predict_single_file_local(filepath):
    if interpreter is None:
        raise RuntimeError("TFLite interpreter not loaded")

    input_details = interpreter.get_input_details()
    input_data, _ = preprocess_for_interpreter(filepath, input_details)
    class_index, confidence, probs = run_tflite_inference(interpreter, input_data)
    label = CLASS_MAPPING.get(class_index, f"class_{class_index}")
    return label, confidence

# ============== ROUTES ==============
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/")
def index_redirect():
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    # Single file endpoint used by your index.html form (name="images")
    try:
        if "images" not in request.files:
            return render_template("result.html", message="❌ No file uploaded", results=[], report_filename="N/A")

        file = request.files["images"]
        if file.filename == "":
            return render_template("result.html", message="❌ Empty filename", results=[], report_filename="N/A")

        if not allowed_file(file.filename):
            return render_template("result.html", message="❌ Unsupported file extension", results=[], report_filename="N/A")

        # make unique filename to avoid collisions
        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(saved_path)

        label, confidence = predict_single_file_local(saved_path)

        result = [{
            "filename": f"uploads/{unique_name}",
            "label": label,
            "confidence": round(confidence, 2)
        }]

        # remove file after predicting
        try:
            os.remove(saved_path)
        except Exception:
            pass

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
                label, confidence = predict_single_file_local(saved_path)
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

        # Save to excel report
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
    # debug True for local testing; set False in production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
