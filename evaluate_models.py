#!/usr/bin/env python3
"""
Evaluate pre-trained BERT and RoBERTa models on test data.
This script loads saved models and performs evaluation only.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Advanced styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for fraud detection
FRAUD_COLOR = '#FF6B6B'  # Vibrant red
LEGIT_COLOR = '#4ECDC4'  # Teal/Cyan
ACCENT_COLOR = '#FFE66D'  # Golden yellow
BERT_COLOR = '#6C5CE7'   # Purple for BERT
ROBERTA_COLOR = '#00B894' # Green for RoBERTa

class JobDataset(torch.utils.data.Dataset):
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

def load_test_data():
    """Load test data."""
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    return X_test['text'].values, y_test

def plot_advanced_confusion_matrix(cm, model_name, color):
    """Create an advanced, modern confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Create heatmap with custom styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=True,
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                ax=ax, vmin=0)
    
    ax.set_title(f'{model_name}: Fraud Detection Performance\nConfusion Matrix Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add custom border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor(color)
    
    plt.tight_layout()
    return fig

def plot_performance_metrics(metrics_dict, model_name, color):
    """Create a modern metrics visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Balanced Accuracy']
    values = [
        metrics_dict['accuracy'],
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1'],
        metrics_dict['balanced_acc']
    ]
    
    # Create bar chart with gradient
    bars = ax.barh(metrics, values, color=color, edgecolor='white', linewidth=2, height=0.6)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f} ({val*100:.1f}%)',
                va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlim([0, 1.15])
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Performance Metrics Dashboard', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def evaluate_bert():
    """Evaluate BERT model with advanced visualizations."""
    print("\n" + "="*70)
    print("🧠 EVALUATING BERT MODEL")
    print("="*70)
    
    if not os.path.exists('models/bert_model'):
        print("⚠ BERT model not found. Skipping evaluation.")
        return None
    
    X_test, y_test = load_test_data()
    
    tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
    model = BertForSequenceClassification.from_pretrained('models/bert_model')
    
    test_dataset = JobDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    probabilities = []
    
    print("Running inference on test data...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 50 == 0:
                print(f"  Processed {(i + 1) * 16} samples...")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().detach().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'balanced_acc': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n✨ BERT MODEL RESULTS:")
    print(f"  ├─ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ├─ Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"  ├─ Precision: {precision:.4f}")
    print(f"  ├─ Recall: {recall:.4f}")
    print(f"  └─ F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    fig = plot_advanced_confusion_matrix(cm, 'BERT', BERT_COLOR)
    plt.savefig('confusion_matrix_bert_eval.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("\n  📊 Confusion matrix saved: confusion_matrix_bert_eval.png")
    
    # Performance Metrics
    fig = plot_performance_metrics(metrics, 'BERT', BERT_COLOR)
    plt.savefig('bert_performance_metrics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  📈 Performance metrics saved: bert_performance_metrics.png")
    
    return {
        'model': 'BERT',
        'metrics': metrics,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }

def evaluate_roberta():
    """Evaluate RoBERTa model with advanced visualizations."""
    print("\n" + "="*70)
    print("🤖 EVALUATING ROBERTA MODEL")
    print("="*70)
    
    if not os.path.exists('models/roberta_model'):
        print("⚠ RoBERTa model not found. Skipping evaluation.")
        return None
    
    X_test, y_test = load_test_data()
    
    tokenizer = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
    model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')
    
    test_dataset = JobDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    probabilities = []
    
    print("Running inference on test data...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 50 == 0:
                print(f"  Processed {(i + 1) * 16} samples...")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().detach().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'balanced_acc': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n✨ ROBERTA MODEL RESULTS:")
    print(f"  ├─ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ├─ Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"  ├─ Precision: {precision:.4f}")
    print(f"  ├─ Recall: {recall:.4f}")
    print(f"  └─ F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    fig = plot_advanced_confusion_matrix(cm, 'RoBERTa', ROBERTA_COLOR)
    plt.savefig('confusion_matrix_roberta_eval.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("\n  📊 Confusion matrix saved: confusion_matrix_roberta_eval.png")
    
    # Performance Metrics
    fig = plot_performance_metrics(metrics, 'RoBERTa', ROBERTA_COLOR)
    plt.savefig('roberta_performance_metrics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  📈 Performance metrics saved: roberta_performance_metrics.png")
    
    return {
        'model': 'RoBERTa',
        'metrics': metrics,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }

def plot_model_comparison(bert_results, roberta_results):
    """Create a comparative visualization of both models."""
    if bert_results is None or roberta_results is None:
        print("⚠ Cannot create comparison plot - both models need to be evaluated.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BERT vs RoBERTa: Comprehensive Model Comparison', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bert_values = [
        bert_results['metrics']['accuracy'],
        bert_results['metrics']['precision'],
        bert_results['metrics']['recall'],
        bert_results['metrics']['f1']
    ]
    roberta_values = [
        roberta_results['metrics']['accuracy'],
        roberta_results['metrics']['precision'],
        roberta_results['metrics']['recall'],
        roberta_results['metrics']['f1']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, bert_values, width, label='BERT', color=BERT_COLOR, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, roberta_values, width, label='RoBERTa', color=ROBERTA_COLOR, edgecolor='white', linewidth=1.5)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0.7, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Balanced Accuracy Comparison
    ax = axes[0, 1]
    balanced_values = [bert_results['metrics']['balanced_acc'], roberta_results['metrics']['balanced_acc']]
    colors = [BERT_COLOR, ROBERTA_COLOR]
    bars = ax.bar(['BERT', 'RoBERTa'], balanced_values, color=colors, edgecolor='white', linewidth=2, width=0.5)
    ax.set_ylabel('Balanced Accuracy', fontweight='bold')
    ax.set_title('Balanced Accuracy (Class-Weighted)', fontweight='bold')
    ax.set_ylim([0.7, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, balanced_values):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Confusion matrices side by side (simplified)
    ax = axes[1, 0]
    bert_cm = confusion_matrix(bert_results['true_labels'], bert_results['predictions'])
    bert_accuracy = np.trace(bert_cm) / np.sum(bert_cm)
    ax.text(0.5, 0.6, 'BERT', ha='center', fontsize=14, fontweight='bold', color=BERT_COLOR)
    ax.text(0.5, 0.4, f'Accuracy: {bert_accuracy:.4f}\nRecall: {bert_results["metrics"]["recall"]:.4f}',
           ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor=BERT_COLOR, alpha=0.2))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    ax = axes[1, 1]
    roberta_cm = confusion_matrix(roberta_results['true_labels'], roberta_results['predictions'])
    roberta_accuracy = np.trace(roberta_cm) / np.sum(roberta_cm)
    ax.text(0.5, 0.6, 'RoBERTa', ha='center', fontsize=14, fontweight='bold', color=ROBERTA_COLOR)
    ax.text(0.5, 0.4, f'Accuracy: {roberta_accuracy:.4f}\nRecall: {roberta_results["metrics"]["recall"]:.4f}',
           ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor=ROBERTA_COLOR, alpha=0.2))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("\n  🏆 Model comparison saved: model_comparison.png")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING ADVANCED MODEL EVALUATION")
    print("="*70)
    
    bert_results = evaluate_bert()
    roberta_results = evaluate_roberta()
    
    if bert_results and roberta_results:
        plot_model_comparison(bert_results, roberta_results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE! All advanced visualizations have been saved.")
    print("="*70 + "\n")
