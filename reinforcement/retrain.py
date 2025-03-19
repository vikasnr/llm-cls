def retrain_model():
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
    
    df = df.drop(columns=["complexity", "data_type", "module", "predicted_llm", "prediction_proba"])
    df = pd.concat([df, df_encoded], axis=1)

    # Re-train model
    X_new = df.drop(columns=["correct_llm"])
    y_new = le.fit_transform(df["correct_llm"])

    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42, stratify=y_new)

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

retrain_model()
