# 🚀 Job Fraud Detection Pipeline Guide

This guide explains the correct order to run the scripts in this project to achieve the best results.

---

## 🏗️ Phase 1: Data Preparation

### 1. Preprocessing
**Script:** `python src/preprocess.py`
- **What it does:** Cleans the raw job data, handles missing values, and prepares text for the AI models.
- **Output:** Saves processed CSV files in the `data/` folder.

### 2. Exploratory Data Analysis (EDA)
**Script:** `python src/eda.py`
- **What it does:** Generates charts for word counts, class distribution, and correlation.
- **Output:** Saves visualizations in the `results/` folder.

### 3. Class Balancing (SMOTE)
**Script:** `python src/smote_balance.py`
- **What it does:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the number of fraud vs. legitimate jobs. This is **CRITICAL** for high accuracy.
- **Output:** Saves `X_train_balanced.csv` and `y_train_balanced.csv`.

---

## 🧠 Phase 2: Model Training

### 4. BERT Model Training
**Script:** `python src/bert_model.py`
- **What it does:** Trains the BERT-base model on the balanced data.
- **Status:** **Currently running** (with 3 epochs).
- **Time:** ~2-4 hours on CPU.

### 5. RoBERTa Model Training
**Script:** `python src/roberta_model.py`
- **What it does:** Trains the RoBERTa model on the balanced data.
- **When to run:** After BERT finishes.

---

## 📊 Phase 3: Evaluation & Deployment

### 6. Final Evaluation
**Script:** `python evaluate_models.py`
- **What it does:** Compares both models on the test set and generates a unified performance report.
- **Output:** Saves comparison charts in the `results/` folder.

### 7. Launch Web Application
**Script:** `python app.py`
- **What it does:** Starts the Flask web dashboard for single and batch predictions.
- **URL:** [http://localhost:5000](http://localhost:5000)

---

## 📂 Project Structure Note
- **`src/`**: All core source code.
- **`models/`**: Saved model weights (BERT/RoBERTa).
- **`results/`**: All generated PNG charts and logs.
- **`legacy_scripts/`**: Old experiment files and testing scripts (safe to ignore).
