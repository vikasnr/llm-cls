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


