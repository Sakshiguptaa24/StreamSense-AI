import os
import joblib
import pandas as pd
import sys as _sys, os as _os
_SRC_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import preprocessing as _preprocessing
if 'preprocessing' not in _sys.modules:
    _sys.modules['preprocessing'] = _preprocessing


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}_model.joblib")
    return joblib.load(path)

def load_pipeline():
    return joblib.load(os.path.join(MODEL_DIR, 'tfidf_pipeline.joblib'))

def predict_review(review: str, model_name='logreg'):
    """Predict sentiment (0/1) for a single review string.
    Returns a dict with prediction and probability (if available)."""
    pipeline = load_pipeline()
    model = load_model(model_name)
    X_vec = pipeline.transform([review])
    pred = model.predict(X_vec)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_vec)[0, 1]
    elif hasattr(model, 'decision_function'):
        # sigmoid conversion
        score = model.decision_function(X_vec)[0]
        prob = 1 / (1 + __import__('math').exp(-score))
    sentiment = 'Positive' if int(pred) == 1 else 'Negative'
    return {'prediction': int(pred), 'probability': prob, 'sentiment': sentiment, 'model': model_name}


