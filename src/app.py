import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Classifier API",
    description="ML pipeline sentiment analysis endpoint",
    version="1.0.0"
)

# Load model at startup
try:
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prediction = model.predict([text])[0]
    confidence = round(float(model.predict_proba([text]).max()), 4)
    sentiment  = "positive" if prediction == 1 else "negative"

    logger.info(f"Prediction: {sentiment} ({confidence}) for: {text[:50]}")

    return PredictResponse(
        text=text,
        sentiment=sentiment,
        confidence=confidence
    )


@app.get("/")
def root():
    return {"message": "Sentiment Classifier API", "docs": "/docs"}