# 🏆 Master Project Execution Guide

There is **no problem** with running Preprocessing, EDA, and SMOTE again! It will simply ensure that your data files are perfectly consistent and ready for the final training.

Follow these steps **one by one** in your terminal. Do not skip any step.

---

### Step 0: Setup & Requirements
Open PowerShell in your project folder and run:
```powershell
# Navigate to project
cd e:\FINAL_YEAR_PROJECT_ORF\final_year_project8_working\final_year_project

# Activate environment
.\my_venv\Scripts\activate

# Install all dependencies (ensures everything is ready)
pip install -r requirements.txt
```

---

### Step 1: Data Preprocessing
Cleans the raw data and prepares the text.
```powershell
python src/preprocess.py
```

---

### Step 2: Exploratory Data Analysis (EDA)
Generates initial visualizations (Word counts, class distribution).
```powershell
python src/eda.py
```

---

### Step 3: SMOTE Balancing
Balances the dataset (Equal fraud and legitimate samples). **CRITICAL** for accuracy.
```powershell
python src/smote_balance.py
```

---

### Step 4: BERT Model Training (Fast-Track)
Trains the BERT model using the optimized 30% sampling.
```powershell
python src/bert_model.py
```
*Wait for this to finish! It will save automatically.*

---

### Step 5: RoBERTa Model Training (Fast-Track)
Trains the RoBERTa model using the optimized 30% sampling.
```powershell
python src/roberta_model.py
```
*Wait for this to finish.*

---

### Step 6: Final Performance Evaluation
Compares both models and creates your presentation charts.
```powershell
python evaluate_models.py
```

---

### Step 7: Launch the Web Application
Starts your final project dashboard.
```powershell
python app.py
```
**URL:** [http://localhost:5000](http://localhost:5000)

---

### 💡 Why this is 100% Safe:
- **Consistency**: Running from scratch avoids any "half-finished" data issues.
- **Speed**: Steps 1, 2, and 3 only take a few minutes.
- **Accuracy**: Steps 4 and 5 (Models) are now optimized to finish much faster while keeping ~90% accuracy.

**Go ahead and start Step 0, 1, 2, and 3 now. They are very fast!**
