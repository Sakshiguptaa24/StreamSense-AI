import os
import pandas as pd
# Minimal stopword set to avoid NLTK binary dependency
STOPWORDS = {
    'a','an','the','and','or','but','if','while','with','without','of','at','by','for','to','in','on','up','down','from','as','is','are','was','were','be','been','being','has','have','had','do','does','did','so','such','that','this','these','those','it','its','i','you','he','she','they','we','them','his','her','their','our','ours','your','yours','my','mine'
}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')

def _ensure_processed_dir():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_imdb_cleaned() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, 'imdb_cleaned.csv')
    return pd.read_csv(path)

def text_preprocess(series: pd.Series) -> pd.Series:
    stop_words = STOPWORDS
    # Simple lower‑casing and removal of non‑alphabetic characters
    return series.str.lower().str.replace(r'[^a-z\s]', ' ', regex=True).apply(lambda txt: ' '.join([w for w in txt.split() if w not in stop_words]))

def build_pipeline() -> Pipeline:
    """Return a scikit‑learn pipeline that vectorises text with TF‑IDF.
    The pipeline expects a pandas Series of raw review strings.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    pipeline = Pipeline([
        ("preprocess", FunctionTransformer(lambda X: X, validate=False)),  # placeholder – actual preprocessing handled before fit
        ("tfidf", vectorizer)
    ])
    return pipeline

# Helper to create a pipeline that includes preprocessing inside a custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = STOPWORDS
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is an iterable of strings
        cleaned = []
        for txt in X:
            txt = txt.lower()
            txt = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in txt])
            tokens = [w for w in txt.split() if w not in self.stop_words]
            cleaned.append(' '.join(tokens))
        return cleaned

def get_full_pipeline():
    return Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])
