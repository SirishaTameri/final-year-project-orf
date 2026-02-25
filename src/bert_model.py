import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import traceback
import os

class JobDataset(Dataset):
    """Dataset that can use precomputed tokenization outputs to avoid per-sample tokenization."""
    def __init__(self, texts, labels, tokenizer, max_len=128, encodings=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.encodings is not None:
            input_ids = self.encodings['input_ids'][idx]
            attention_mask = self.encodings['attention_mask'][idx]
            text = str(self.texts[idx]) if self.texts is not None else ''
        else:
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(sample_fraction=1.0):
    """Load balanced training data and test data with optional sub-sampling."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data'

    X_train = pd.read_csv(data_dir / 'X_train_balanced.csv')
    y_train = pd.read_csv(data_dir / 'y_train_balanced.csv')
    
    # Optional sub-sampling to speed up training on CPU
    if sample_fraction < 1.0:
        print(f"Sub-sampling training data to {sample_fraction*100}%...")
        X_train = X_train.sample(frac=sample_fraction, random_state=42)
        y_train = y_train.loc[X_train.index]
        
    y_train = y_train.values.ravel()
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').values.ravel()

    return X_train['text'].values, y_train, X_test['text'].values, y_test

def train_bert(X_train, y_train, epochs=1, batch_size=16, max_len=128, num_workers=0):
    """Train BERT model with optimizations: pretokenize, multiple workers, mixed precision."""
    # Try to load from local saved model first, otherwise download
    if os.path.exists('models/bert_model'):
        print("Loading BERT model from local cache...")
        tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
        model = BertForSequenceClassification.from_pretrained('models/bert_model')
    else:
        print("Downloading BERT model from HuggingFace...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Pre-tokenize training data in batch to avoid per-sample tokenization cost
    print("Tokenizing training data in batch...")
    try:
        train_encodings = tokenizer(list(X_train), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    except Exception:
        # Fallback to safe tokenization without return_tensors
        enc = tokenizer(list(X_train), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_len)
        # convert to tensors
        train_encodings = {k: torch.tensor(v) for k, v in enc.items()}

    train_dataset = JobDataset(X_train, y_train, tokenizer, max_len=max_len, encodings=train_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.to(device)

    use_amp = device.type == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model.train()
    for epoch in range(epochs):
        last_loss = None
        # enumerate batches so we can print progress per N batches
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            last_loss = loss.item()

            # Print batch-level progress every 10 batches to show liveliness
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {last_loss}")

    return model, tokenizer

def evaluate_model(model, tokenizer, X_test, y_test, batch_size=16, max_len=128, num_workers=2):
    """Evaluate the model. Uses pretokenization to speed up evaluation."""
    print("Tokenizing test data in batch...")
    try:
        test_encodings = tokenizer(list(X_test), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    except Exception:
        enc = tokenizer(list(X_test), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_len)
        test_encodings = {k: torch.tensor(v) for k, v in enc.items()}

    test_dataset = JobDataset(X_test, y_test, tokenizer, max_len=max_len, encodings=test_encodings)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_bert.png')
    plt.close()  # Close instead of show to avoid blocking

    return accuracy, balanced_acc, precision, recall, f1

if __name__ == "__main__":
    try:
        # User requested 100% data
        X_train, y_train, X_test, y_test = load_data(sample_fraction=1.0)
        model, tokenizer = train_bert(X_train, y_train, epochs=3, batch_size=16, max_len=128, num_workers=0)
        evaluate_model(model, tokenizer, X_test, y_test, batch_size=32, max_len=128, num_workers=2)

        # Save model
        os.makedirs('models/bert_model', exist_ok=True)
        os.makedirs('models/bert_tokenizer', exist_ok=True)
        model.save_pretrained('models/bert_model')
        tokenizer.save_pretrained('models/bert_tokenizer')
        print("BERT model and tokenizer saved successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        print("An error occurred during run. Traceback written to 'bert_run_error.log'")
        with open('bert_run_error.log', 'w', encoding='utf-8') as f:
            f.write(tb)
        raise