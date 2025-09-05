from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import uvicorn
import time

app = FastAPI()

# -------------------------
# CORS (open for local Next.js dev)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Keras model
# -------------------------
MODEL_PATH = "modelres50.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Make sure class order matches your training
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# Preprocessing (match training: 200x200 RGB, normalized)
# -------------------------
TARGET_SIZE = (200, 200)

def preprocess_pil_to_model_input(img: Image.Image) -> np.ndarray:
    """
    Match model input: (200, 200, 3) normalized to [0,1]
    """
    img = img.convert("RGB")                        # 3 channels
    img = img.resize(TARGET_SIZE, Image.BILINEAR)   # resize to 200x200
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize to [0,1]
    arr = np.expand_dims(arr, axis=0)               # shape: (1, 200, 200, 3)
    return arr

# -------------------------
# Inference Endpoint
# -------------------------
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read uploaded image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    orig_w, orig_h = img.size

    # Preprocess
    x = preprocess_pil_to_model_input(img)

    # Predict
    t0 = time.perf_counter()
    preds = model.predict(x)
    infer_ms = int((time.perf_counter() - t0) * 1000)

    # Ensure shape (1, 4)
    preds = np.squeeze(preds)      # (4,)
    probs = preds.astype(float).tolist()

    # Top-1 prediction
    top_idx = int(np.argmax(preds))
    label = CLASS_NAMES[top_idx]
    prob  = float(preds[top_idx])

    # Build top-k list in CLASS_NAMES order
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
# Run (for local testing)
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
