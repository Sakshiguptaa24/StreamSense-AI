import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import joblib

import os, sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_full_pipeline

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'imdb_cleaned.csv'))
    X = df['review']
    y = df['sentiment']
    return X, y

def train_and_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = get_full_pipeline()
    start = time.time()
    X_train_vec = pipeline.fit_transform(X_train)
    train_time = time.time() - start
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train_vec, y_train)
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    # Save models and pipeline
    _ensure_model_dir()
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'tfidf_pipeline.joblib'))
    joblib.dump(lr, os.path.join(MODEL_DIR, 'logreg_model.joblib'))
    joblib.dump(nb, os.path.join(MODEL_DIR, 'nb_model.joblib'))
    # Save test set for evaluation
    joblib.dump((X_test, y_test), os.path.join(MODEL_DIR, 'test_set.joblib'))
    return train_time
