## A LLM classifer with Feedback System 
#### An LLM classifier with a feedback system continuously improves its predictions by incorporating user feedback, refining its model based on real-world performance. This adaptive approach helps enhance accuracy, reduce biases, and optimize LLM selection for various tasks over time.

## Summary

 - [Why Synthetic Data?](#Why-I-Decided-to-Generate-Features-Instead-of-Relying-on-Public-Datasets)
 - [Data Preprocessing](#Data-Preprocessing)
 - [Model Training and Inference ](#Model-Training-and-Inference)
 - [Feedback System](#Feedback-System)
 - [API Endpoints](#API-Endpoints)
 - [Limitation and Scope for Improvement](#Limitation-and-Scope-for-Improvement)
 - [Resources](#Resources)



------------------------------------

### Why I Decided to Generate Features Instead of Relying on Public Datasets


Before diving into the other sections, I’d like to explain why I chose to generate the data myself.
I lacked programmatic access to LLM models, which restricted my ability to create features for this assignment. Preprocessing publicly available benchmark datasets like "grade-school-math," "hellaswag," "mmlu," "arc-challenge," "mbpp," "winogrande," and "mtbench" presents several challenges in terms of time and data integration. These datasets vary significantly in structure, format, and granularity, making it difficult to standardize them into a unified feature set.  

Aggregating data from Hugging Face and other benchmark sources requires extensive cleaning, normalization, and alignment of missing attributes. Many datasets lack key metadata which are essential for my classifier. Additionally, these benchmarks often focus on specific LLM capabilities rather than providing a balanced distribution across all prompt types. 

-----------------------------


### Data Preprocessing

Refer notebooks/data_preprocessing.ipynb for more details on preprocessing.

-------------------------------------------------

### Model Training and Inference 

- Refer notebooks/Modelling_Final.ipynb
- Refer model architecture notebooks/files/xgboost_arch.png

--------------------------


### Feedback System

<img src="notebooks/files/feedback_system.png" width="760" class="center">



### API Endpoints

##### `/predict`
- Endpoint that takes a prompt, task description and returns the best LLM model based on the features of the prompt and the confidence score.
- Calls **`predict_llm()`**
  - Extracts features from the prompt, predicts the best LLM model, and logs the prediction.

##### `/feedback`
- Endpoint that takes a prompt, response, correct LLM model, and ground truth response, and logs the feedback scores.
- Calls **`evaluate_llm()`**  
  - Computes the faithfulness, BLEU, and ROUGE scores using the response and ground truth response.
- Calls **`log_feedback()`**  
  - Logs the feedback scores and the correct LLM model for future training.

##### `/retrain`
- Endpoint that retrains the LLM model based on the feedback logs.
- Calls **`retrain_model_with_metrics()`**  
  - Loads previous LLM selection data, faithfulness, BLEU, and ROUGE scores, and retrains the model.
- Calls **`retrain_model_without_metrics()`**  
  - Loads logs, converts them to a DataFrame, extracts features, retrains the model, and saves the updated models.
  
##### `/llm_call`
- **(Dummy API)**
- Takes a prompt and the LLM model name and returns the generated response.

##### `monitor_metric()`
- **(Dummy function)**
- This could be a seperate service that monitors log and triggers retrain based on set rules

#### Refer feedback_system folder for more details

-------------------------------------------------

### Limitation and Scope for Improvement

- ###### Limitations of Synthetic Dataset
    Synthetic dataset doesn’t closely resemble real user queries and responses. This is one area which will need greater amount of time - fetching from varies source and aggregating them for modules.
    
- ###### Feature selection
    I wouldn't say the features choosen initially are the best. All though I have added `BLEU`, `ROUGE`, `Hallucination` score scores etc in feedback system, it could be helpful to have these scores during intial modelling.

- ###### Increasing Training Data
    Increasing the number of training examples by collecting from varies sources and transforming will definitely be beneficial.
  
- ###### Trying Other Models
    With the increase in train samples using complex models like deep learning models could yield better results.

-------------------------------------------------

### Resources

[stanford helm](https://crfm.stanford.edu/helm/)\

[LMSYS](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)\

[RouterBench](https://github.com/withmartian/routerbench)\











