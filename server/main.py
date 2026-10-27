"""FastAPI server exposing digit prediction over HTTP."""

import base64
import io
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import MODEL_PATH
from ml.predict import predict_image

app = FastAPI(title="Digit Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64-encoded image, optionally a data URL (e.g. canvas.toDataURL())


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def decode_base64_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python -m ml.train` first.")

    try:
        image = decode_base64_image(request.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    digit, confidence, probabilities = predict_image(image)
    return PredictResponse(digit=digit, confidence=confidence, probabilities=probabilities)
