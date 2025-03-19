import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from extract_features import extract_prompt_features
LOG_FILE = "predictions_log.json"

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

    # Append log entry to JSON file
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

def classify_complexity():
    """
    Classifies the complexity of a given prompt.
    """
    # Placeholder for complexity classification logic
    return "Medium"

def predict_llm(prompt):
    """
    Predicts the best LLM and logs the prediction.
    """
    features = extract_prompt_features(new_prompt)


    complexity = classify_complexity(prompt)
    features.update({"complexity": complexity, "module": "document_extraction","prompt_length": len(prompt)})
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



# predicted_llm, confidence = predict_llm(prompt)

# print(f"Recommended LLM: {predicted_llm} (Confidence: {confidence:.2f})")
