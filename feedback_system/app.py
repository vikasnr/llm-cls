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
    task_details: str

class FeedbackInput(BaseModel):
    prompt: str
    response: str
    correct_llm: str  
    ground_truth: Optional[str]

class LLLSelected(BaseModel):
    prompt: str


@app.post("/predict/")
def predict(prompt: PromptInput):
    '''predict llm model'''
    prompt_data = prompt.prompt
    task_details = prompt.task_details
    predicted_llm, confidence = predict_llm(prompt_data,task_details)
    return {"predicted_llm": predicted_llm, "confidence": confidence}


@app.post("/call_llm/")
def call_llm(llm: LLLSelected):
    '''
    placeholder function is used to call the LLM model
    '''
    return {"response": "response from LLM model"}

    
@app.post("/feedback/")
def collect_feedback(feedback: FeedbackInput):
    '''collect feedback'''

    prompt = feedback.prompt
    response = feedback.response
    correct_llm = feedback.correct_llm
    ground_truth = feedback.ground_truth

    faithfulness, bleu, rouge_score =  evaluate_llm(response, ground_truth)
    prompt_features = extract_prompt_features(prompt)
    log_feedback(prompt_features,faithfulness, bleu, rouge_score, correct_llm)

    return {"message": "Feedback logged successfully!"}



@app.post("/retrain/")
    
def retrain(retrain_type: str):
    '''retrain model'''
    if retrain_type == "with_metrics":
        retrain_model_with_metrics()
    elif retrain_type == "without_metrics":
        retrain_model_without_metrics()

    return {"message": "Model retrained and saved!"}



