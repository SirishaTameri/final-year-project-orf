import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for headless/crash-prone environments
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Advanced styling for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for fraud detection (red = danger/fraud, green = safe/legitimate)
FRAUD_COLOR = '#FF6B6B'  # Vibrant red
LEGIT_COLOR = '#4ECDC4'  # Teal/Cyan
ACCENT_COLOR = '#FFE66D'  # Golden yellow
SECONDARY_COLOR = '#95E1D3'  # Mint

def load_preprocessed_data():
    """Load preprocessed data."""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('data/y_test.csv').iloc[:, 0]
    return X_train, X_test, y_train, y_test

def plot_class_distribution(y_train, y_test):
    """Plot sophisticated class distribution with donut charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Job Posting Authenticity Distribution Across Datasets', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Train set donut chart
    train_counts = y_train.value_counts()
    colors_train = [LEGIT_COLOR, FRAUD_COLOR]
    explode = (0.05, 0.05)
    
    axes[0].pie(train_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%',
                colors=colors_train, explode=explode, shadow=True, startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0].set_title('Training Set (n={:,})'.format(len(y_train)), 
                      fontsize=12, fontweight='bold', pad=20)
    
    # Draw circle for donut effect in train
    centre_circle_train = plt.Circle((0, 0), 0.70, fc='white')
    axes[0].add_artist(centre_circle_train)
    
    # Test set donut chart
    test_counts = y_test.value_counts()
    colors_test = [LEGIT_COLOR, FRAUD_COLOR]
    
    axes[1].pie(test_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%',
                colors=colors_test, explode=explode, shadow=True, startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1].set_title('Test Set (n={:,})'.format(len(y_test)), 
                      fontsize=12, fontweight='bold', pad=20)
    
    # Draw circle for donut effect in test
    centre_circle_test = plt.Circle((0, 0), 0.70, fc='white')
    axes[1].add_artist(centre_circle_test)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("OK Class distribution visualization saved: class_distribution.png")

def analyze_text_length(X_train):
    """Analyze text length distribution with advanced histogram."""
    X_train['text_length'] = X_train['text'].apply(lambda x: len(str(x).split()))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create gradient histogram
    n, bins, patches = ax.hist(X_train['text_length'], bins=60, edgecolor='white', linewidth=1.5)
    
    # Apply gradient coloring
    cm = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(patches))
    for i, patch in enumerate(patches):
        patch.set_facecolor(cm(norm(i)))
    
    ax.set_xlabel('Word Count per Job Posting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Job Posting Text Length Distribution: A Fraud Detection Perspective', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics
    mean_length = X_train['text_length'].mean()
    median_length = X_train['text_length'].median()
    ax.axvline(mean_length, color=FRAUD_COLOR, linestyle='--', linewidth=2.5, label=f'Mean: {mean_length:.0f}')
    ax.axvline(median_length, color=LEGIT_COLOR, linestyle='--', linewidth=2.5, label=f'Median: {median_length:.0f}')
    
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('text_length_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("OK Text length distribution visualization saved: text_length_distribution.png")

def most_common_words(X_train, top_n=20):
    """Visualize and print most common words in fraudulent vs legitimate posts."""
    fraud_text = ' '.join(X_train[X_train['fraudulent'] == 1]['text'].fillna(''))
    non_fraud_text = ' '.join(X_train[X_train['fraudulent'] == 0]['text'].fillna(''))

    fraud_words = Counter(fraud_text.split()).most_common(top_n)
    non_fraud_words = Counter(non_fraud_text.split()).most_common(top_n)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Most Frequent Keywords: Fraudulent vs. Legitimate Job Postings', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    # Fraudulent posts
    fraud_df = pd.DataFrame(fraud_words, columns=['word', 'count'])
    axes[0].barh(range(len(fraud_df)), fraud_df['count'], color=FRAUD_COLOR, edgecolor='white', linewidth=1.5)
    axes[0].set_yticks(range(len(fraud_df)))
    axes[0].set_yticklabels(fraud_df['word'], fontsize=10)
    axes[0].set_xlabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Fraudulent Postings', fontsize=12, fontweight='bold', color=FRAUD_COLOR)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legitimate posts
    legit_df = pd.DataFrame(non_fraud_words, columns=['word', 'count'])
    axes[1].barh(range(len(legit_df)), legit_df['count'], color=LEGIT_COLOR, edgecolor='white', linewidth=1.5)
    axes[1].set_yticks(range(len(legit_df)))
    axes[1].set_yticklabels(legit_df['word'], fontsize=10)
    axes[1].set_xlabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Legitimate Postings', fontsize=12, fontweight='bold', color=LEGIT_COLOR)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('most_common_words.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("OK Most common words visualization saved: most_common_words.png")
    
    print("\n" + "="*70)
    print("TEXTUAL ANALYSIS: KEYWORD PATTERNS IN FRAUD DETECTION")
    print("="*70)
    print("\nTOP 20 KEYWORDS IN FRAUDULENT POSTINGS:")
    print("-" * 70)
    for i, (word, count) in enumerate(fraud_words, 1):
        print(f"  {i:2d}. {word:20s} → {count:6d} occurrences")
    
    print("\nTOP 20 KEYWORDS IN LEGITIMATE POSTINGS:")
    print("-" * 70)
    for i, (word, count) in enumerate(non_fraud_words, 1):
        print(f"  {i:2d}. {word:20s} → {count:6d} occurrences")
    print("="*70)

def correlation_analysis(X_train):
    """Analyze correlations with advanced heatmap."""
    # Select only numeric columns for correlation
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("⚠ No numeric columns found for correlation analysis.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = X_train[numeric_cols].corr()
    
    # Advanced heatmap with mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, square=True, linewidths=1.5, cbar_kws={'label': 'Correlation'},
                ax=ax, vmin=-1, vmax=1, annot_kws={'size': 9, 'weight': 'bold'})
    
    ax.set_title('Feature Correlation Matrix: Uncovering Fraud Indicators', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("OK Feature correlation visualization saved: feature_correlation.png")

def data_quality_summary(X_train, X_test, y_train, y_test):
    """Display comprehensive data quality report."""
    print("\n" + "="*70)
    print("DATA QUALITY & DISTRIBUTION SUMMARY")
    print("="*70)
    
    print(f"\nDATASET DIMENSIONS:")
    print(f"  • Training Samples: {len(X_train):,}")
    print(f"  • Test Samples: {len(X_test):,}")
    print(f"  • Total Samples: {len(X_train) + len(X_test):,}")
    
    print(f"\nCLASS DISTRIBUTION (Training Set):")
    train_counts = y_train.value_counts()
    for label, count in train_counts.items():
        pct = 100 * count / len(y_train)
        status = "Legitimate" if label == 0 else "Fraudulent"
        print(f"  • {status}: {count:,} samples ({pct:.1f}%)")
    
    print(f"\nCLASS DISTRIBUTION (Test Set):")
    test_counts = y_test.value_counts()
    for label, count in test_counts.items():
        pct = 100 * count / len(y_test)
        status = "Legitimate" if label == 0 else "Fraudulent"
        print(f"  • {status}: {count:,} samples ({pct:.1f}%)")
    
    print(f"\nFEATURES & DATA TYPES:")
    print(f"  • Total Features: {len(X_train.columns)}")
    print(f"  • Numeric Features: {len(X_train.select_dtypes(include=[np.number]).columns)}")
    print(f"  • Categorical/Text Features: {len(X_train.select_dtypes(include=['object']).columns)}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS: ONLINE RECRUITMENT FRAUD DETECTION")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    # Add fraudulent column to X_train for analysis
    X_train['fraudulent'] = y_train

    data_quality_summary(X_train, X_test, y_train, y_test)
    
    print("\nGENERATING VISUALIZATIONS...")
    plot_class_distribution(y_train, y_test)
    analyze_text_length(X_train)
    most_common_words(X_train)
    correlation_analysis(X_train)
    
    print("\nEDA COMPLETE! All visualizations have been saved.")
    print("="*70 + "\n")