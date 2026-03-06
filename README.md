# 🛡️ Online Recruitment Fraud (ORF) Detection System

Welcome to the Online Recruitment Fraud Detection System! This project leverages state-of-the-art Natural Language Processing (NLP) architectures—specifically **BERT** and **RoBERTa**—to accurately detect and flag fraudulent job postings.

## 🎯 Project Overview

Online recruitment fraud is a growing concern where fake job postings are used to extract personal information or money from legitimate job seekers. This system analyzes the text of job postings and predicts whether they are **REAL** or **FRAUDULENT** with high accuracy. 

It includes:
- Data Preprocessing and Exploratory Data Analysis (EDA)
- Dataset balancing using SMOTE
- Model training capabilities for both BERT and RoBERTa
- A user-friendly web dashboard to test individual postings or run batch checks via CSV

---

## 🚀 Quick Start Guide

To run this project locally, simply follow these steps in your command line:

### 1. Set Up the Environment
```powershell
# Navigate to the project directory
cd final_year_project

# Activate the virtual environment
.\my_venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
You can run the individual components one by one:
```powershell
python src/preprocess.py
python src/eda.py
python src/smote_balance.py
python src/bert_model.py
python src/roberta_model.py
```

### 3. Launch the Dashboard
Once the models are ready, start the web application:
```powershell
python app.py
```
Then, open your web browser and go to **[http://localhost:5000](http://localhost:5000)**.

---

## 📚 Documentation
For an in-depth, line-by-line explanation of the code, methodology, and how the models work, please refer to the `PROJECT_EXPLANATION.md` file included in this repository. 
For detailed execution instructions, see `MASTER_EXECUTION_GUIDE.md`.
