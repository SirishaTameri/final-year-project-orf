import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class JobDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(sample_fraction=1.0):
    """Load balanced training data and test data with optional sub-sampling."""
    X_train = pd.read_csv('data/X_train_balanced.csv')
    y_train = pd.read_csv('data/y_train_balanced.csv')
    
    # Optional sub-sampling to speed up training on CPU
    if sample_fraction < 1.0:
        print(f"Sub-sampling training data to {sample_fraction*100}%...")
        X_train = X_train.sample(frac=sample_fraction, random_state=42)
        y_train = y_train.loc[X_train.index]
        
    y_train = y_train.values.ravel()
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    return X_train['text'].values, y_train, X_test['text'].values, y_test

def train_roberta(X_train, y_train, epochs=3, batch_size=16):
    """Train RoBERTa model."""
    import os
    # Try to load from local saved model first, otherwise download
    if os.path.exists('models/roberta_model'):
        print("Loading RoBERTa model from local cache...")
        tokenizer = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
        model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')
    else:
        print("Downloading RoBERTa model from HuggingFace...")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    train_dataset = JobDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model, tokenizer

def evaluate_model(model, tokenizer, X_test, y_test):
    """Evaluate the model."""
    test_dataset = JobDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

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
    plt.savefig('confusion_matrix_roberta.png')
    plt.close()  # Close instead of show to avoid blocking

    return accuracy, balanced_acc, precision, recall, f1

if __name__ == "__main__":
    # User requested 100% data
    X_train, y_train, X_test, y_test = load_data(sample_fraction=1.0)
    model, tokenizer = train_roberta(X_train, y_train)
    evaluate_model(model, tokenizer, X_test, y_test)

    # Save model
    model.save_pretrained('models/roberta_model')
    tokenizer.save_pretrained('models/roberta_tokenizer')
    print("RoBERTa model and tokenizer saved successfully.")
    tokenizer.save_pretrained('models/roberta_tokenizer')