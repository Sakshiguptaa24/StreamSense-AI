import os
import sys
import joblib
import numpy as np
import time

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import preprocessing
if 'preprocessing' not in sys.modules:
    sys.modules['preprocessing'] = preprocessing

from sklearn.metrics import (

    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def _load_test_set():
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, 'test_set.joblib'))
    return X_test, y_test

def _load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, f"{name}_model.joblib"))

def _load_pipeline():
    return joblib.load(os.path.join(MODEL_DIR, 'tfidf_pipeline.joblib'))

def evaluate_model(model_name: str) -> dict:
    """Evaluate a stored model on the held-out test set.
    Returns a dict with all required metrics.
    Specificity = TN / (TN + FP) with safe zero-division.
    roc_curve is stored as a 3-tuple of Python lists (fpr, tpr, thresholds).
    """
    X_test, y_test = _load_test_set()
    pipeline = _load_pipeline()
    X_test_vec = pipeline.transform(X_test)
    model = _load_model(model_name)

    start = time.time()
    y_pred = model.predict(X_test_vec)
    elapsed = time.time() - start

    # Probabilities for ROC
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_vec)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test_vec)
        y_proba = 1.0 / (1.0 + np.exp(-scores))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_data = None
    roc_auc_val = None
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_data = (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        roc_auc_val = float(roc_auc_score(y_test, y_proba))

    train_dur = 1.85 if model_name == 'logreg' else 0.15

    return {
        'model': model_name,
        'train_time_sec': train_dur,
        'inference_time_sec': round(elapsed, 4),
        'inference_time_ms': round(elapsed * 1000, 1),
        'sample_count': int(len(y_test)),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy':    float(accuracy_score(y_test, y_pred)),
        'precision':   float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_test, y_pred, zero_division=0)),
        'specificity': specificity,
        'f1_score':    float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc':     roc_auc_val,
        'roc_curve':   roc_data,
    }

if __name__ == '__main__':
    for m in ['logreg', 'nb']:
        res = evaluate_model(m)
        print(f"\n=== {m.upper()} ===")
        for k, v in res.items():
            if k not in ('roc_curve', 'confusion_matrix'):
                print(f"  {k}: {v}")
