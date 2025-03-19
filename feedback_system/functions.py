import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime


    
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



from extract_features import extract_prompt_features
LOG_FILE = "predictions_log.json"
FEEDBACK_LOG_FILE = "feedback_log.json"
# Load trained models
clf = joblib.load("llm_classifier.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

X_train_subset = pd.read_csv("X_train_subset.csv")


LOG_FILE = "predictions_log.json"

def log_prediction(prompt_features, predicted_llm, prediction_proba):
    """
    Logs the model's predictions and confidence scores for future training.
    """
    log_entry = {
        "timestamp": str(datetime.now()),
        "features": prompt_features,
        "predicted_llm": predicted_llm,
        "prediction_proba": prediction_proba,
        "correct_llm": None  # Placeholder for user feedback
    }
    # append to llm_feedback_scores.csv file
    log_df = pd.DataFrame([log_entry])
    try:
        existing_logs = pd.read_csv("logs/llm_prediction.csv")
        updated_logs = pd.concat([existing_logs, log_df], ignore_index=True)
    except FileNotFoundError:
        updated_logs = log_df

    updated_logs.to_csv("logs/llm_prediction.csv", index=False)



# Initialize evaluation tools
nltk.download('punkt')
rouge = Rouge()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Model for semantic similarity


def evaluate_llm(response, ground_truth):
    
    # Faithfulness Score (Semantic Similarity)
    resp_embedding = model.encode([response])
    gt_embedding = model.encode([ground_truth])
    faithfulness = cosine_similarity(resp_embedding, gt_embedding)[0][0]

    # BLEU Score (Word Overlap)
    reference_tokens = [nltk.word_tokenize(ground_truth.lower())]
    response_tokens = nltk.word_tokenize(response.lower())
    bleu = sentence_bleu(reference_tokens, response_tokens)

    # ROUGE Score (Recall-Based)
    rouge_score = rouge.get_scores(response, ground_truth)[0]['rouge-l']['f']

    return faithfulness, bleu, rouge_score


def log_feedback(prompt_features,faithfulness, bleu, rouge_score, correct_llm):
    """
    Logs the model's predictions and confidence scores for future training.
    """
    log_entry = {
        "timestamp": str(datetime.now()),
        "features": prompt_features,
        "faithfulness_score": faithfulness,
        "bleu_score": bleu,
        "rouge_score": rouge_score,
        "correct_llm": correct_llm
    }

    # Append log entry to JSON file
    log_df = pd.DataFrame([log_entry])
    try:
        existing_logs = pd.read_csv("logs/llm_feedback_scores.csv")
        updated_logs = pd.concat([existing_logs, log_df], ignore_index=True)
    except FileNotFoundError:
        updated_logs = log_df

    updated_logs.to_csv("logs/llm_feedback_scores.csv", index=False)
    

def classify_complexity():
    """
    Classifies the complexity of a given prompt.
    """
    # Placeholder for complexity classification logic
    return "Medium"

def predict_llm(prompt,task_details):
    """
    Predicts the best LLM and logs the prediction.
    """
    features = extract_prompt_features(prompt)


    complexity = classify_complexity(prompt)

    features.update({"complexity": complexity, "module": "document_extraction","prompt_length": len(prompt),"module":task_details})

    print(features)
    new_df = pd.DataFrame([features])

    # Encode categorical features
    cat_features = ohe.transform(new_df[["complexity", "module"]])
    cat_df = pd.DataFrame(cat_features, columns=ohe.get_feature_names_out())

    
    # Drop original categorical features and merge transformed ones
    new_df = pd.concat([new_df, cat_df], axis=1)
    new_df = new_df.drop(columns=["complexity", "module"])


    # Ensure column order matches training set
    new_df = new_df[X_train_subset.columns]

    # Scale features
    new_df_scaled = scaler.transform(new_df)

    # Predict LLM
    y_pred_proba = clf.predict_proba(new_df_scaled)[0]  # Get confidence scores
    y_pred = clf.predict(new_df_scaled)
    predicted_llm = le.inverse_transform(y_pred)[0]

    # Log the prediction
    log_prediction(features, predicted_llm, max(y_pred_proba))

    return predicted_llm, max(y_pred_proba)
