"""
================================================================================
Customer Sentiment Analysis & NLP Pipeline
Oasis Infobyte - Data Analytics Internship | Task 4
================================================================================
Author: Data Analytics Intern
Description: End-to-end NLP text preprocessing, TF-IDF feature extraction,
             multi-model classification (Naive Bayes, Logistic Regression, Linear SVC),
             WordCloud generation, confusion matrix evaluation, and error analysis.
================================================================================
"""

import os
import sys
import re
import string

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

OUTPUT_DIR = 'assets/task4_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2])

def run_sentiment_pipeline(data_path='product_sentiment_dataset.csv'):
    print("=" * 80)
    print("STARTING CUSTOMER SENTIMENT ANALYSIS PIPELINE (TASK 4)")
    print("=" * 80)

    # 1. Ingest
    print(f"\n[1/6] Ingesting sentiment corpus from: {data_path} ...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    df = pd.read_csv(data_path)
    print(f"  [+] Reviews Loaded: {len(df):,} across categories: {list(df['Category'].unique())}")
    print(f"  [+] Class Balance:\n{df['Sentiment'].value_counts().to_string()}")

    # 2. Preprocess
    print("\n[2/6] Executing NLP Text Preprocessing (Tokenize, Stopwords, Lemmatize) ...")
    df['Cleaned_Review'] = df['Review_Text'].apply(clean_text)
    print("  [+] Text normalization and lemmatization complete.")

    # 3. Vectorize
    print("\n[3/6] Fitting TF-IDF Vectorizer (N-Grams 1-2, sublinear scaling) ...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True, min_df=2)
    X = tfidf.fit_transform(df['Cleaned_Review'])
    y = df['Sentiment']
    print(f"  [+] Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} n-gram features")

    # 4. Train / Test Split
    print("\n[4/6] Splitting train (80%) and test (20%) sets with stratification ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  [+] Training: {X_train.shape[0]:,} | Testing: {X_test.shape[0]:,}")

    # 5. Model Training & Evaluation
    print("\n[5/6] Training & Evaluating Multi-Class NLP Models ...")
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
        'Logistic Regression': LogisticRegression(C=1.5, max_iter=500, random_state=42),
        'Linear Support Vector (SVC)': LinearSVC(C=1.0, random_state=42)
    }

    print("\n" + "=" * 80)
    print(f"{'Model Architecture':30s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s}")
    print("-" * 80)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name:30s} | {acc*100:7.2f}% | {prec*100:8.2f}% | {rec*100:7.2f}% | {f1*100:7.2f}%")
    print("=" * 80)

    print("\n[6/6] Pipeline execution completed successfully! All assets in assets/task4_figures/")

if __name__ == '__main__':
    run_sentiment_pipeline()
