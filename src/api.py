from fastapi import FastAPI
from pydantic import BaseModel
from .inference import inference

# --------------------
# FastAPI application
# --------------------

app = FastAPI(
    title="BERT Customer Support Intent Classifier",
    description="API for classifying customer support messages.",
    version="1.0.0"
)


# --------------------
# Request schema
# --------------------

class PredictionRequest(BaseModel):

    text: str

# --------------------
# Prediction endpoint
# --------------------

@app.post("/predict")
def predict_intent(request: PredictionRequest):

    label, confidence = inference(request.text)

    return {
        "intent": label,
        "confidence": confidence
    }