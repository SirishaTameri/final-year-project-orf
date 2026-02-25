#!/usr/bin/env python3
"""
Main script to run the Online Recruitment Fraud Detection project.
"""

import os
import subprocess
import sys

def run_script(script_name):
    """Run a Python script."""
    try:
        # Force unbuffered output from child Python processes so prints appear immediately
        subprocess.run([sys.executable, '-u', f'src/{script_name}'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False
    return True

def main():
    print("Starting Online Recruitment Fraud Detection Project...")

    # Check if data exists
    if not os.path.exists('data/fake_job_postings.csv'):
        print("Please download the dataset first. See README.md")
        return

    # Step 1: Preprocess (skip if already preprocessed)
    if not os.path.exists('data/X_train_balanced.csv'):
        print("Step 1: Preprocessing data...")
        if not run_script('preprocess.py'):
            return
    else:
        print("Step 1: Preprocessed data already exists, skipping...")

    # Step 2: EDA
    print("Step 2: Exploratory Data Analysis...")
    if not run_script('eda.py'):
        return

    # Step 3: SMOTE Balancing (skip if already balanced)
    if not os.path.exists('data/X_train_balanced.csv'):
        print("Step 3: Balancing data with SMOTE...")
        if not run_script('smote_balance.py'):
            return
    else:
        print("Step 3: Data already balanced, skipping...")

    # Step 4: Train BERT
    print("Step 4: Training BERT model...")
    if not run_script('bert_model.py'):
        return

    # Step 5: Train RoBERTa
    print("Step 5: Training RoBERTa model...")
    if not run_script('roberta_model.py'):
        return

    print("Project completed! Check results in console and saved models.")

if __name__ == "__main__":
    main()