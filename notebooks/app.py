import spacy
import textstat
import re
from collections import Counter

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_prompt_features(prompt):
    doc = nlp(prompt)
    
    #  Linguistic Features
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    unique_words = len(set([token.text.lower() for token in doc]))
    lexical_diversity = unique_words / num_tokens if num_tokens > 0 else 0
    readability = textstat.flesch_kincaid_grade(prompt)
    
    # Part-of-Speech (POS) Distribution
    pos_counts = Counter([token.pos_ for token in doc])
    noun_ratio = pos_counts.get("NOUN", 0) / num_tokens
    verb_ratio = pos_counts.get("VERB", 0) / num_tokens
    adjective_ratio = pos_counts.get("ADJ", 0) / num_tokens

    #  Structural Features
    num_named_entities = len(doc.ents)
    contains_table = bool(re.search(r"(\||\+\-+)|Table \d+", prompt))  # Detect tables
    contains_list = bool(re.search(r"(\d+\.)|(- )|(\* )", prompt))  # Detect bullet points or numbered lists

    # Content-Specific Features
    keyword_density = {word: prompt.lower().count(word) for word in ["summary", "analyze", "data", "report"]}
    redundancy_score = len(re.findall(r"(\b\w+\b).*\1", prompt))  # Count repeated words
    compression_ratio = len(prompt) / num_tokens if num_tokens > 0 else 0

    #  Task-Specific Complexity Features
    contains_numbers = bool(re.search(r"\d+", prompt))  # Presence of numerical data
    contains_chain_of_thought = bool(re.search(r"step-by-step|explain your reasoning", prompt.lower()))
    contains_output_constraints = bool(re.search(r"limit to|output in|return a json", prompt.lower()))
    is_multi_turn = "previous response" in prompt.lower() or "as mentioned before" in prompt.lower()

    return {
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "lexical_diversity": lexical_diversity,
        "readability": readability,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adjective_ratio": adjective_ratio,
        "num_named_entities": num_named_entities,
        "contains_table": contains_table,
        "contains_list": contains_list,
        "redundancy_score": redundancy_score,
        "compression_ratio": compression_ratio,
        "contains_numbers": contains_numbers,
        "contains_chain_of_thought": contains_chain_of_thought,
        "contains_output_constraints": contains_output_constraints,
        "is_multi_turn": is_multi_turn
    }

# Example Prompt
prompt = """
You are an AI assistant. Summarize the research paper and provide key insights. 
Ensure the response is formatted as JSON and limited to 200 words. 
Step-by-step analysis is preferred. Example: "In this study, the authors examined..."
"""

# Extract Features
features = extract_prompt_features(prompt)
print(features)



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the trained model and encoders
model = joblib.load("llm_classifier.pkl")
ohe = joblib.load("ohe_encoder.pkl")  # Load OneHotEncoder
scaler = joblib.load("scaler.pkl")  # Load Scaler (if used)
le = joblib.load("label_encoder.pkl")  # Load LabelEncoder for LLM







df = pd.DataFrame(features,index=[0])
df



# Initialize FastAPI app
app = FastAPI()

# Define request model
class PromptRequest(BaseModel):
    prompt: str
# Define API route for prediction
@app.post("/predict_llm")
def predict_llm(prompt_data: PromptRequest):
    try:


        # Extract features from the prompt
        prompt_features = extract_prompt_features(prompt_data.prompt)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([prompt_data.dict()])

        # One-hot encode categorical features
        categorical_cols = ["complexity", "data_type", "module"]
        categorical_encoded = ohe.transform(input_df[categorical_cols])
        categorical_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out())

        # Drop original categorical columns and merge encoded data
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df, categorical_df], axis=1)

        # Scale numerical features
        numerical_cols = input_df.columns  # Assuming all are now numerical
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict the LLM class
        prediction = model.predict(input_df)

        # Convert prediction back to LLM label
        predicted_llm = le.inverse_transform(prediction)[0]

        return {"predicted_llm": predicted_llm}
    
    except Exception as e:
        return {"error": str(e)}

# Run the app using: uvicorn filename:app --reload
