import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load preprocessed data."""
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    return X_train, y_train

def apply_smote(X_train, y_train, smote_variant='borderline'):
    """
    Apply balancing techniques. 
    Note: For raw text data, standard SMOTE (interpolation) is not applicable. 
    We use RandomOverSampler which is the standard equivalent for text fine-tuning.
    """
    from imblearn.over_sampling import RandomOverSampler
    
    # For text data, we use RandomOverSampler to replicate minority class samples
    # This aligns with how "SMOTE" is often adapted for NLP (oversampling)
    print(f"Applying RandomOverSampler (Text-compatible balancing)...")
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
    print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

def plot_resampled_distribution(y_original, y_resampled):
    """Plot before and after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pd.Series(y_original).value_counts().plot(kind='bar', ax=axes[0], title='Original Class Distribution')
    pd.Series(y_resampled).value_counts().plot(kind='bar', ax=axes[1], title='Resampled Class Distribution')
    import os
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/smote_distribution.png')
    plt.close()

if __name__ == "__main__":
    X_train, y_train = load_data()
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    plot_resampled_distribution(y_train, y_resampled)

    # Save resampled data
    X_resampled.to_csv('data/X_train_balanced.csv', index=False)
    pd.DataFrame(y_resampled, columns=['fraudulent']).to_csv('data/y_train_balanced.csv', index=False)