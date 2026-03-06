# Online Recruitment Fraud Detection: Project Explanation Guide

This document is designed to help you explain every component of your final year project to your panel members, step-by-step. It breaks down the code logic and the "why" behind each step.

---

## 1. Data Preprocessing (`src/preprocess.py`)
**Goal:** Clean the raw text data and prepare it for machine learning models. Raw data is messy and needs to be standardized.

**Implementation Details:**
- **Loading Data (`load_data`):** The dataset (`fake_job_postings.csv`) is loaded. We filter it to include rows with the correct structure and a limit of 18,000 rows.
- **Text Cleaning (`preprocess_text`):**
  - **Lowercasing:** Converts all text to lowercase so "Job" and "job" are treated equally.
  - **Removing Special Characters:** Uses Regular Expressions (`re.sub`) to remove punctuation and numbers, keeping only alphabetic characters.
  - **Stopwords Removal:** Uses NLTK to remove common English words (like "the", "and", "is") that don't add meaningful predictive value.
  - **Lemmatization:** Reduces words to their base form (e.g., "running" becomes "run") using NLTK's `WordNetLemmatizer`.
- **Feature Engineering (`preprocess_data`):** 
  - Combines multiple text fields (`title`, `location`, `description`, `requirements`, etc.) into a single `text` column because contextual models like BERT learn better from rich, combined text.
  - Categorical columns (like `employment_type`) are encoded into numbers using `LabelEncoder`.
  - Binary columns (like `telecommuting`) are converted directly to 0 or 1.
- **Splitting Data (`split_data`):** Splits the dataset into 80% Training Data and 20% Testing Data using `train_test_split`. We use `stratify=y` to ensure the 80/20 split maintains the same ratio of fraudulent to legitimate jobs in both sets.

---

## 2. Exploratory Data Analysis (EDA) (`src/eda.py`)
**Goal:** Understand the characteristics of the dataset, visualize patterns, and find differences between fraudulent and legitimate job postings.

**Implementation Details:**
- **Class Distribution (`plot_class_distribution`):** Visualizes the heavy imbalance in the dataset (mostly legitimate jobs, very few fraudulent ones) using Donut Charts. This justifies the need for SMOTE/balancing later.
- **Text Length Analysis (`analyze_text_length`):** Analyzes the word count of job postings. Fraudulent postings often have different length distributions (e.g., shorter descriptions) compared to legitimate ones. We plot this using an advanced gradient histogram.
- **Keyword Extraction (`most_common_words`):** Uses Python's `Counter` to find the most frequent words in both fraudulent and legitimate texts. Fraudulent texts might overuse urgent or overly appealing words.
- **Correlation Analysis (`correlation_analysis`):** Creates a Heatmap using `seaborn` to check if numerical features (like `has_company_logo` or `has_questions`) correlate strongly with the `fraudulent` label.

---

## 3. Handling Data Imbalance (SMOTE / Oversampling) (`src/smote_balance.py`)
**Goal:** Fix the dataset imbalance. If we train on 95% legitimate and 5% fraudulent data, the model will just guess "legitimate" every time and achieve 95% accuracy without actually learning to detect fraud.

**Implementation Details:**
- **The Challenge:** Traditional SMOTE creates synthetic numerical data points by interpolating between existing points. However, our primary feature is **Text**, and you cannot easily "interpolate" text embeddings without destroying grammar and meaning.
- **The Solution (`apply_smote`):** Instead of standard SMOTE, the code uses `RandomOverSampler` from the `imblearn` library. This duplicates the minority class (fraudulent postings) until it matches the majority class count. 
- **Result:** The models are fine-tuned on a perfectly balanced training set, forcing them to learn the characteristics of fraud rather than relying on class probability.

---

## 4. BERT Model (`src/bert_model.py`)
**Goal:** Use a state-of-the-art Transformer language model to understand the deep context of the job postings.

**Implementation Details:**
- **Architecture:** We use `BertForSequenceClassification` from the Hugging Face `transformers` library, which adds a classification head (with 2 outputs: Legit vs Fraud) on top of the pre-trained BERT base model.
- **Tokenization:** Text is converted to tokens and padded/truncated to a maximum length of 128 tokens using `BertTokenizer`. We "pre-tokenize" the entire dataset in batches to drastically speed up training.
- **Dataset Class (`JobDataset`):** A custom PyTorch Dataset class that feeds the tokenized `input_ids`, `attention_mask`, and `labels` to the model.
- **Training Loop (`train_bert`):** 
  - Uses the `AdamW` optimizer (Adam with weight decay) with an optimal learning rate of `2e-5`.
  - Uses PyTorch Native Mixed Precision (`torch.cuda.amp`) if a GPU is available, which speeds up training by using 16-bit floats where possible without losing accuracy.
  - Computes the Loss (Cross-Entropy Loss internally by the Hugging Face model) and backpropagates to update weights.

---

## 5. RoBERTa Model (`src/roberta_model.py`)
**Goal:** Train a more robust, heavily optimized version of BERT. RoBERTa (Robustly Optimized BERT Pretraining Approach) removes Next Sentence Prediction and trains on much more data for longer.

**Implementation Details:**
- **Architecture:** Uses `RobertaForSequenceClassification` and `RobertaTokenizer`.
- **Differences from BERT:** RoBERTa typically yields slightly better contextual embeddings because of its larger pre-training corpus and dynamic masking. We train it similarly to BERT (using AdamW, PyTorch Dataloaders).
- **Justification to Panel:** By implementing both BERT and RoBERTa, the project demonstrates a comparative analysis of two leading NLP architectures to see which performs better on the specific task of recruitment fraud.

---

## 6. Evaluation (`evaluate_models.py`)
**Goal:** Rigorously test the trained models on the unseen 20% test data.

**Implementation Details:**
- **Metrics Calculated:**
  - **Accuracy:** Overall correctness, but misleading due to class imbalance.
  - **Balanced Accuracy:** Average of recall obtained on each class (crucial for imbalanced data).
  - **Precision:** Out of all jobs predicted as fraud, how many were actually fraud? (Minimizes false alarms).
  - **Recall:** Out of all actual fraudulent jobs, how many did we catch? (Crucial for fraud detection).
  - **F1-Score:** The harmonic mean of Precision and Recall.
- **Confusion Matrices (`plot_advanced_confusion_matrix`):** Visualizes True Positives, True Negatives, False Positives, and False Negatives.
- **Comparison Plot (`plot_model_comparison`):** Automatically generates side-by-side bar charts comparing BERT and RoBERTa across all aforementioned metrics to declare a definitive "best model".

---

## 7. Web Application (`app.py` & `src/routes.py`)
**Goal:** Deploy the trained models into a user-friendly interface so end-users (job seekers or HR portals) can practically use the fraud detection system.

**Implementation Details:**
- **Framework:** Uses **Flask**, a lightweight Python web framework.
- **Endpoints (`src/routes.py`):**
  - **`GET /` & HTML pages:** Renders the frontend UI templates (Dashboard, Single Prediction, Batch Analysis).
  - **`POST /api/predict`:** Accepts a single job text, chooses the selected model (BERT or RoBERTa), tokenizes the text, runs it through the PyTorch model, and returns the prediction and a confidence score.
  - **`POST /api/compare`:** Runs the text through *both* models simultaneously and returns a comparative analysis.
  - **`POST /api/batch-predict`:** Allows users to upload a CSV file of multiple job postings. The app iterates through the CSV, runs inference on each row, and returns aggregated results.
- **Real-time processing:** Connects the Deep Learning logic (`src.services.predict_fraud`) directly to API calls, making the machine learning models actionable via HTTP requests.

---

## Tip for your Presentation:
1. **Start with the Problem:** "Online recruitment fraud is rising, causing financial and identity loss."
2. **Explain the Solution:** "We use NLP to detect fraudulent patterns in job descriptions."
3. **Highlight the Tech:** "We chose BERT and RoBERTa because traditional machine learning (like Naive Bayes) struggles with complex sentence contexts. Transformers understand the *meaning* behind the text."
4. **Emphasize SMOTE/Balancing:** Panelists love it when you address class imbalance. Explain that without `RandomOverSampler`, the model would be biased towards legitimate jobs.
5. **Show the App:** Conclude by demonstrating the Flask app. A working UI proves that your project goes beyond just theory.
