

import os
import io
import numpy as np
from flask import Flask, raequest, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ── Config 
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "pneumonia_model.h5")
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]   # index 0 = normal, 1 = pneumonia
MAX_BYTES    = 10 * 1024 * 1024          # 10 MB upload limit

# ── App setup 
app = Flask(__name__)
CORS(app)                                # allow requests from the HTML frontend
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

# ── Load model once at startup 
print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
model.predict(np.zeros((1, 224, 224, 3)))  # warm-up pass
print("Model ready ✓")


# ── Helper 
def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw image bytes → model-ready tensor.
    Matches MobileNetV2 preprocessing: pixels in [-1, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)   # → [-1, 1]
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "pneumonia_mobilenetv2"})


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate request ──────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file in request (field name: 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        return jsonify({"error": f"Unsupported content type: {file.content_type}"}), 415

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        tensor    = preprocess(img_bytes)
        preds     = model.predict(tensor, verbose=0)[0]   # shape: (2,)

        pred_idx     = int(np.argmax(preds))
        prediction   = CLASS_NAMES[pred_idx].lower()      # "normal" | "pneumonia"
        probability  = float(preds[pred_idx])
        confidence_pct = round(probability * 100, 2)

        return jsonify({
            "prediction"  : prediction,           # "normal" or "pneumonia"
            "probability" : probability,           # float 0-1  (used by Diagnosis.py)
            "confidence"  : confidence_pct,        # % for display
            "label"       : CLASS_NAMES[pred_idx], # "NORMAL" or "PNEUMONIA"
            "scores": {
                "normal"    : round(float(preds[0]), 4),
                "pneumonia" : round(float(preds[1]), 4),
            }
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
