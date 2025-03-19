from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime

from functions import predict_llm, evaluate_llm, extract_prompt_features, log_feedback, retrain_model_with_metrics, retrain_model_without_metrics


LOG_FILE = "predictions_log.json"

app = FastAPI()

class PromptInput(BaseModel):
    prompt: str

class FeedbackInput(BaseModel):
    prompt: str
    response: str
    correct_llm: str  
    ground_truth: Optional[str]


class LLLSelected(BaseModel):
    prompt: str


@app.post("/predict/")
def predict(prompt: PromptInput):
    prompt_data = prompt.dict()
    predicted_llm, confidence = predict_llm(prompt_data)
    return {"predicted_llm": predicted_llm, "confidence": confidence}


@app.post("/call_llm/")
def call_llm(llm: LLLSelected):
    prompt_data = llm
    predicted_llm, confidence = predict_llm(prompt_data)
    return {"predicted_llm": predicted_llm, "confidence": confidence}


def log_prediction(prompt_features, predicted_llm, prediction_proba):
    """
    Logs the model's predictions and confidence scores for future training.
    """
    log_entry = {
        "timestamp": str(datetime.now()),
        "features": prompt_features,
        "predicted_llm": predicted_llm,
        "prediction_proba": prediction_proba,
    }

    # Append log entry to JSON file
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    
@app.post("/feedback/")
def collect_feedback(feedback: FeedbackInput):
    # Load existing logs
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    prompt = feedback.prompt
    response = feedback.response
    correct_llm = feedback.correct_llm
    ground_truth = feedback.ground_truth

    faithfulness, bleu, rouge_score =  evaluate_llm(response, ground_truth)
    prompt_features = extract_prompt_features(prompt)
    log_feedback(prompt_features,faithfulness, bleu, rouge_score, correct_llm)


    # Save updated logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    return {"message": "Feedback recorded!"}


@app.post("/retrain/")
def retrain(retrain_type: str):
    if retrain_type == "with_metrics":
        retrain_model_with_metrics()
    elif retrain_type == "without_metrics":
        retrain_model_without_metrics()

    return {"message": "Model retrained and saved!"}
