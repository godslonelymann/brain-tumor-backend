from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import uvicorn
import time
import os
import requests

app = FastAPI()

# -------------------------
# CORS (open for dev; restrict in prod)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production: replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model Setup
# -------------------------
MODEL_PATH = "modelres50.h5"
MODEL_URL = "https://huggingface.co/anuuuuuragggggg/brain-tumor-resnet50/blob/main/modelres50.h5"

# 🔺 Replace <username>/<repo-name> with your Hugging Face repo details

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match training order)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# Preprocessing (match training: 200x200 RGB normalized)
# -------------------------
TARGET_SIZE = (200, 200)

def preprocess_pil_to_model_input(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")                        # 3 channels
    img = img.resize(TARGET_SIZE, Image.BILINEAR)   # resize
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize
    arr = np.expand_dims(arr, axis=0)               # shape (1, 200, 200, 3)
    return arr

# -------------------------
# Inference Endpoint
# -------------------------
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    orig_w, orig_h = img.size

    # Preprocess
    x = preprocess_pil_to_model_input(img)

    # Predict
    t0 = time.perf_counter()
    preds = model.predict(x)
    infer_ms = int((time.perf_counter() - t0) * 1000)

    # Format predictions
    preds = np.squeeze(preds)  # (4,)
    probs = preds.astype(float).tolist()

    top_idx = int(np.argmax(preds))
    label = CLASS_NAMES[top_idx]
    prob = float(preds[top_idx])

    topk = [{"label": cls, "prob": float(p)} for cls, p in zip(CLASS_NAMES, probs)]

    return {
        "model": {
            "name": "BrainTumorCNN",
            "file": MODEL_PATH,
            "input_shape": [1, 200, 200, 3],
            "inference_ms": infer_ms
        },
        "prediction": {
            "label": label,
            "probability": prob,
            "topk": topk
        },
        "image_meta": {
            "width": orig_w,
            "height": orig_h,
            "channels": 3
        }
    }

# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -------------------------
# Run (local + Render)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT, fallback = 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
