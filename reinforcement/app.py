from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json

from functions import predict_llm
LOG_FILE = "predictions_log.json"

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

class FeedbackInput(BaseModel):
    prompt_features: dict
    correct_llm: str  # User-provided correct answer

@app.post("/predict/")
def predict(prompt: PromptInput):
    prompt_data = prompt.dict()
    predicted_llm, confidence = predict_llm(prompt_data)
    return {"predicted_llm": predicted_llm, "confidence": confidence}

@app.post("/feedback/")
def collect_feedback(feedback: FeedbackInput):
    # Load existing logs
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    # Find matching entry and update feedback
    for entry in reversed(logs):  # Search from the latest entries
        if entry["features"] == feedback.prompt_features and entry["correct_llm"] is None:
            entry["correct_llm"] = feedback.correct_llm
            break

    # Save updated logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    return {"message": "Feedback recorded!"}
