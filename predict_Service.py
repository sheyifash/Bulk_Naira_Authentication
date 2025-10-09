import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Model path
MODEL_PATH = "model/outputs/naira_auth_model.h5"
TRAIN_DIR = "model/dataset/train"
REPORTS_DIR = "model/reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ✅ Automatically detect class mapping from training folder
def get_class_mapping():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),  # Doesn’t affect anything, just for scanning
        batch_size=1,
        class_mode="categorical"
    )
    mapping = {v: k for k, v in generator.class_indices.items()}
    print(f"🧩 Detected class mapping: {mapping}")
    return mapping


# ✅ Load trained model
def load_trained_model():
    print(f"🔍 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return model


# ✅ Predict a single image with thresholding
def predict_single(image_path, model, class_mapping, threshold=0.6):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None

    input_shape = model.input_shape[1:3]
    print(f"📏 Model expects input size: {input_shape}")

    # Preprocess
    img = load_img(image_path, target_size=input_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    class_name = class_mapping.get(class_index, "Unknown")

    # Thresholding
    if confidence < threshold:
        result = "Uncertain"
    else:
        result = class_name

    print(f"🖼️ Image: {os.path.basename(image_path)}")
    print(f"🔮 Prediction: {result} ({confidence:.2f} confidence)")
    print("─────────────────────────────")

    return result, confidence


# ✅ Predict all images in a folder
def predict_bulk(folder_path, model, class_mapping, threshold=0.6):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return None

    results = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            result = predict_single(img_path, model, class_mapping, threshold)
            if result:
                results.append((img_name, *result))

    if results:
        df = pd.DataFrame(results, columns=["Image Name", "Predicted Class", "Confidence"])
        report_path = os.path.join(REPORTS_DIR, "bulk_prediction_report.xlsx")
        df.to_excel(report_path, index=False)
        print(f"📊 Report saved to: {report_path}")
    else:
        print("⚠️ No valid image files found.")


# ✅ Main logic
if __name__ == "__main__":
    model = load_trained_model()
    class_mapping = get_class_mapping()

    if len(sys.argv) < 2:
        print("❌ Usage: python model/predict_service.py <image_or_folder_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        predict_bulk(input_path, model, class_mapping)
    else:
        predict_single(input_path, model, class_mapping)
