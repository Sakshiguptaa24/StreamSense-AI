import os
import sqlite3
from datetime import datetime

# Database located inside the models folder (same as other model artefacts)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'history.db')

def _init_db():
    """Create the history database and required tables if they do not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Table for model evaluation runs
    cur.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            timestamp TEXT,
            train_time_sec REAL,
            sample_count INTEGER,
            accuracy REAL,
            precision REAL,
            recall REAL,
            specificity REAL,
            f1_score REAL,
            roc_auc REAL
        )
    ''')
    # Table for live predictions
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            review TEXT,
            prediction INTEGER,
            probability REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_run(metrics: dict):
    """Insert a row into the runs table.
    Expected keys in *metrics* match those returned by src/evaluation.evaluate_model.
    """
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs (model, timestamp, train_time_sec, sample_count,
                          accuracy, precision, recall, specificity, f1_score, roc_auc)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            metrics.get('model'),
            datetime.utcnow().isoformat(),
            metrics.get('train_time_sec'),
            metrics.get('sample_count'),
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('specificity'),
            metrics.get('f1_score'),
            metrics.get('roc_auc'),
        ),
    )
    conn.commit()
    conn.close()

def log_prediction(model: str, review: str, prediction: int, probability: float | None):
    """Insert a prediction record into the predictions table."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (model, review, prediction, probability, timestamp)
        VALUES (?,?,?,?,?)
        """,
        (model, review, prediction, probability, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

def get_prediction_history(limit: int = 10):
    """Retrieve recent live predictions from the database as a list of dicts."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, model, review, prediction, probability, timestamp FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),))
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "model", "review", "prediction", "probability", "timestamp"]
    return [dict(zip(cols, r)) for r in rows]

def get_runs_history(limit: int = 10):
    """Retrieve recent model runs from the database as a list of dicts."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),))
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


