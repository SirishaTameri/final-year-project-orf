import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

# Download NLTK data (run once)
nltk.download('stopwords')                                                                       
nltk.download('wordnet')

def load_data(filepath='data/fake_job_postings.csv'):
    """Load the dataset."""
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
        header = next(reader)  # Read header first
        rows = []
        for row in reader:
            if len(row) == 18 and row[-2] in ['f', 't']:
                rows.append(row)
            if len(rows) >= 18000:  # Limit to 18000 data rows
                break
    df = pd.DataFrame(rows, columns=header)
    print(f"Dataset shape: {df.shape}")
    print(df.info())
    return df

def preprocess_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def preprocess_data(df):
    """Preprocess the entire dataset."""
    # Combine text columns for input
    text_columns = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
    df['text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    df['text'] = df['text'].apply(preprocess_text)

    # Handle categorical columns
    categorical_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    # Encode categorical variables (simple label encoding for now)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Handle binary columns
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Drop unnecessary columns
    df = df.drop(columns=['salary_range'] + text_columns)

    # Target variable
    df['fraudulent'] = df['fraudulent'].map({'f': 0, 't': 1}).fillna(0).astype(int)

    print(f"Preprocessed dataset shape: {df.shape}")
    print(df.head())
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    # Save preprocessed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)