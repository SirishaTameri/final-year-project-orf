import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

def test_model():
    model_path = 'models/bert_model'
    tokenizer_path = 'models/bert_tokenizer'
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    print("Loading model for diagnostics...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Test cases
    test_cases = [
        ("Real Job: Looking for a software engineer with 5 years of experience in Python and Django. Full-time position in Bangalore.", "REAL"),
        ("FAKE JOB: $$$ EASY MONEY $$$ WORK FROM HOME EARN 5000 DOLLARS A DAY NO EXPERIENCE NEEDED SEND YOUR BANK DETAILS TO WIN!!!", "FRAUD")
    ]

    for text, expected in test_cases:
        print(f"\nTesting {expected} case...")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            
            print(f"Logits: {logits.numpy()}")
            print(f"Probabilities: {probs.numpy()}")
            print(f"Predicted Class: {'FRAUD' if prediction == 1 else 'REAL'}")
            print(f"Confidence: {probs[0][prediction].item():.4%}")

if __name__ == "__main__":
    test_model()
