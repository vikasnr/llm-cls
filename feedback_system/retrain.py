import numpy as np
import joblib
import json
from datetime import datetime


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


LOG_FILE = "predictions_log.json"
FEEDBACK_LOG_FILE = "feedback_log.json"


# Load trained models
model = joblib.load("files/best_xgboost.pkl")
scaler = joblib.load("files/v5_random_forest_scaler.pkl")
ohe = joblib.load("files/v5_random_forest_onehot_encoder.pkl")
le = joblib.load("files/v5_random_forest_label_encoder.pkl")

X_train_subset = pd.read_csv("files/X_train_subset.csv")


def retrain_model_without_metrics():
    # Load logs
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    # Convert logs to DataFrame
    df = pd.DataFrame(logs)

    # Extract features
    df["correct_llm"].fillna(df["predicted_llm"], inplace=True)
    df["was_correct"] = df["predicted_llm"] == df["correct_llm"]
    df["confidence_score"] = df["prediction_proba"]

    # One-hot encode categorical features again
    df_encoded = ohe.fit_transform(df[["complexity", "data_type", "module"]])
    df_encoded = pd.DataFrame(df_encoded, columns=ohe.get_feature_names_out())

    df = df.drop(
        columns=[
            "complexity",
            "data_type",
            "module",
            "predicted_llm",
            "prediction_proba",
        ]
    )
    df = pd.concat([df, df_encoded], axis=1)

    # Re-train model
    X_new = df.drop(columns=["correct_llm"])
    y_new = le.fit_transform(df["correct_llm"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
    )

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    new_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    new_clf.fit(X_train_scaled, y_train)

    # Save updated models
    joblib.dump(new_clf, "llm_classifier.pkl")
    joblib.dump(ohe, "onehot_encoder.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model retrained and saved!")


def retrain_model_with_metrics(
    previous_file="logs/llm_selection_dataset.csv",
    feedback_file="logs/llm_feedback_scores.csv",
):

    # Load previous LLM selection data
    df = pd.read_csv(previous_file)

    # Load post-hoc evaluation scores
    feedback_scores = pd.read_csv(feedback_file)

    # Compute Moving Average Scores for Each LLM
    llm_performance = (
        feedback_scores.groupby("predicted_llm")[
            ["faithfulness_score", "bleu_score", "rouge_score"]
        ]
        .mean()
        .reset_index()
    )

    # Merge LLM performance data with training set
    df = df.merge(llm_performance, left_on="LLM", right_on="predicted_llm", how="left")

    # Fill missing values (new models may not have historical scores)
    df[["faithfulness_score", "bleu_score", "rouge_score"]] = df[
        ["faithfulness_score", "bleu_score", "rouge_score"]
    ].fillna(df[["faithfulness_score", "bleu_score", "rouge_score"]].mean())

    # Retrain the LLM classifier using updated data

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df["LLM"] = label_encoder.fit_transform(df["LLM"])

    # Prepare features and target
    X = df.drop(columns=["LLM", "predicted_llm"])  # Remove unnecessary columns
    y = df["LLM"]

    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost Model
    clf = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Updated Model Accuracy: {accuracy:.4f}")
