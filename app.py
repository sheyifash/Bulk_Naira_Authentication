import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename 

# ==============================
# Configuration
# ==============================

MODEL_PATH = "model/outputs/naira_auth_model.h5"
CLASS_MAPPING = {
    0: "fake_1000",
    1: "fake_500",
    2: "genuine_1000",
    3: "genuine_500"
}

# UPLOAD_FOLDER is set inside 'static' so the browser can display the images
UPLOAD_FOLDER = "static/uploads"
# REPORTS_DIR is kept private, requiring the custom download route
REPORTS_DIR = "model/reports"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==============================
# Initialize Flask App
# ==============================
# Flask will serve files from the 'static' folder automatically
app = Flask(__name__)
CORS(app)

# ==============================
# Load Model
# ==============================
print(f"🔍 Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None # Set model to None if loading fails

# ==============================
# Helper Function - Predict Image
# ==============================
def predict_image(image_path):
    """Loads, preprocesses, and predicts the class of a single image."""
    if model is None:
        raise Exception("Model not loaded.")
        
    input_shape = model.input_shape[1:3]
    
    # Image loading and preprocessing
    img = load_img(image_path, target_size=input_shape)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    class_name = CLASS_MAPPING[class_index]
    confidence = float(np.max(prediction))

    return {
        "predicted_class": class_name,
        "confidence": confidence
    }

# ==============================
# API Route - Serve Templates (The 'index' endpoint FIX)
# ==============================
@app.route("/")
def index():
    """Serves the main file upload page (index.html)."""
    return render_template("index.html")

# ==============================
# API Route - Single Prediction (Using 'images' key as requested)
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return redirect(url_for('index'))

    file = request.files["images"]
    if file.filename == "":
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    # File path for saving in static/uploads
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save file temporarily
    file.save(file_path)

    try:
        # Get prediction result
        result = predict_image(file_path)
        
        # Prepare data for the HTML template. 
        # The path needs to be relative to the static folder ('uploads/filename.jpg')
        image_url = os.path.join("uploads", filename) 
        
        # NOTE: This route is primarily for single-file tests.
        return render_template(
            "result.html", 
            message="Single Prediction Complete",
            results=[{ # Wrap single result in a list for consistent template use
                "filename": image_url,
                "label": result['predicted_class'],
                "confidence": round(result['confidence'], 2), 
            }],
            # Dummy report filename is needed for the download button to render
            report_filename="N/A" 
        )

    except Exception as e:
        print(f"❌ FATAL ERROR during single prediction: {e}")
        # Clean up must happen in finally block regardless of error
        return redirect(url_for('index')) 
        
    finally:
        # Cleanup: Delete the file immediately after use
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"DEBUG: Cleaned up file {file_path}")


# ==============================
# API Route - Bulk Folder Prediction (Targeted by frontend)
# ==============================
@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    # Check for the 'images' key from the HTML form
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    results = []
    
    for file in files:
        if file.filename == "":
            continue # Skip empty files

        filename = secure_filename(file.filename)
        # File path for saving in static/uploads
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            raw_result = predict_image(file_path)
            
            # The 'filename' for the template is relative to static folder ('uploads/filename.jpg')
            image_url_for_template = os.path.join("uploads", filename)

            formatted_result = {
                "filename": image_url_for_template,
                "label": raw_result["predicted_class"],
                "confidence": round(raw_result["confidence"], 2)
            }
            results.append(formatted_result)

        except Exception as e:
            print(f"❌ Error predicting file {filename}: {e}")
            error_result = {"filename": os.path.join("uploads", filename), "label": "ERROR", "confidence": 0.0, "error_message": str(e)}
            results.append(error_result)
            
        finally:
            # Cleanup: Delete the file immediately after use
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if not results:
        return jsonify({"error": "No valid files were processed"}), 400

    # 1. Save results to Excel
    df = pd.DataFrame(results)
    # The full path to the saved report
    report_path = os.path.join(REPORTS_DIR, "bulk_prediction_report.xlsx")
    df.to_excel(report_path, index=False)
    
    # 2. Extract only the filename for the download link
    report_filename = os.path.basename(report_path)

    # 3. Return the HTML template
    return render_template(
        "result.html",
        message="Bulk Prediction Complete!",
        results=results, 
        report_filename=report_filename # Passed to url_for('download_report', filename=...)
    )

# ==============================
# API Route - Download Report (FIX for 404 Error)
# ==============================
@app.route("/download_report/<filename>")
def download_report(filename):
    """Safely serves the prediction report file for download."""
    safe_filename = secure_filename(filename)
    
    try:
        # Use send_from_directory to serve the file from the REPORTS_DIR
        return send_from_directory(
            REPORTS_DIR, 
            safe_filename, 
            as_attachment=True # Forces download
        )
    except FileNotFoundError:
        return "Report not found.", 404


# ==============================
# Run Flask App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
