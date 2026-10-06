from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------
# Device
# --------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------
# Model path
# --------------------

project_dir = Path(__file__).resolve().parent.parent
model_dir = project_dir / "bert_clinc_model"


# --------------------
# Load tokenizer
# --------------------

tokenizer = AutoTokenizer.from_pretrained(model_dir)


# --------------------
# Load trained model
# --------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_dir
)

model = model.to(device)
model.eval()


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
# Prediction function
# --------------------

def predict(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()

    label = model.config.id2label[prediction]

    return label, confidence


# --------------------
# Prediction endpoint
# --------------------

@app.post("/predict")
def predict_intent(request: PredictionRequest):

    label, confidence = predict(request.text)

    return {
        "intent": label,
        "confidence": confidence
    }