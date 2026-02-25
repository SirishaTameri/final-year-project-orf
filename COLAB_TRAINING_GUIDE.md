# 🚀 Google Colab Training Guide

Using Google Colab is a **brilliant** idea. While your laptop takes 50+ hours, Colab's Free GPU (T4) will finish BERT in about **20-30 minutes**.

---

### Step 1: Prepare your Data
You need to upload your balanced data from your laptop to Colab.
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Click the **Folder icon** 📂 on the left sidebar.
4. Drag and drop these 4 files from your `data/` folder into Colab:
   - `X_train_balanced.csv`
   - `y_train_balanced.csv`
   - `X_test.csv`
   - `y_test.csv`

---

### Step 2: Enable GPU (CRITICAL)
1. Go to **Runtime** -> **Change runtime type**.
2. Select **T4 GPU** (or any Hardware accelerator).
3. Click **Save**.

---

### Step 3: Run the Training Code
Copy and paste the code below into a cell in Colab and play it:

```python
!pip install transformers torch pandas scikit-learn matplotlib seaborn

import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 1. Load Data
X_train = pd.read_csv('X_train_balanced.csv')['text'].values
y_train = pd.read_csv('y_train_balanced.csv').values.ravel()
X_test = pd.read_csv('X_test.csv')['text'].values
y_test = pd.read_csv('y_test.csv').values.ravel()

class JobDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

def train_model(model_name, X_train, y_train):
    print(f"\n--- Training {model_name} ---")
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    device = torch.device('cuda')
    model.to(device)
    loader = DataLoader(JobDataset(X_train, y_train, tokenizer), batch_size=32, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for epoch in range(3):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Complete. Final Loss: {loss.item()}")
    
    model.save_pretrained(f"{model_name}_model")
    tokenizer.save_pretrained(f"{model_name}_tokenizer")
    print(f"Finished {model_name}!")

# Run both!
train_model('bert', X_train, y_train)
train_model('roberta', X_train, y_train)

# Zip them so you can download
!zip -r models_from_colab.zip bert_model bert_tokenizer roberta_model roberta_tokenizer
```

---

### Step 4: Move Models back to Laptop
1. Once the code finishes, you will see a file named `models_from_colab.zip` in the Colab file area.
2. **Download it** to your laptop.
3. Extract the folders and move them into your project's `models/` folder:
   - `e:\...\final_year_project\models\bert_model`
   - `e:\...\final_year_project\models\bert_tokenizer`
   - `e:\...\final_year_project\models\roberta_model`
   - `e:\...\final_year_project\models\roberta_tokenizer`

4. Then, just run **Step 6 (Evaluation)** on your laptop!
