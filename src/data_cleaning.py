import os
import pandas as pd
from typing import Tuple

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

def _ensure_processed_dir():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the raw data directory, or fallback to processed directory."""
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        fallback_map = {
            "netflix_titles.csv": "netflix_cleaned.csv",
            "IMDB Dataset.csv": "imdb_cleaned.csv",
        }
        alt_name = fallback_map.get(filename)
        if alt_name:
            alt_path = os.path.join(PROCESSED_DIR, alt_name)
            if os.path.exists(alt_path):
                return pd.read_csv(alt_path)
        raise FileNotFoundError(f"Dataset {filename} not found in {RAW_DIR} or {PROCESSED_DIR}")
    return pd.read_csv(path)


def clean_netflix(df: pd.DataFrame) -> pd.DataFrame:
    """Perform cleaning specific to the Netflix titles dataset.
    Steps:
    * Strip whitespace from string columns
    * Parse dates in "date_added"
    * Standardise "duration" to integer minutes (for movies) or seasons (for TV shows)
    * Remove duplicates
    * Handle missing values (drop rows where essential fields are missing)
    """
    df = df.copy()
    # Trim whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Parse date_added if present
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    # Standardise duration
    if "duration" in df.columns:
        def parse_duration(val):
            if pd.isna(val):
                return None
            parts = val.split()
            if len(parts) >= 2:
                try:
                    num = int(parts[0])
                except ValueError:
                    return None
                unit = parts[1].lower()
                if unit.startswith('min'):
                    return num  # minutes
                elif unit.startswith('season'):
                    return num  # seasons
            return None
        df["duration_parsed"] = df["duration"].apply(parse_duration)

    # Drop duplicates based on unique identifier (show_id)
    if "show_id" in df.columns:
        df = df.drop_duplicates(subset=["show_id"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    # Drop rows with all NaNs in critical columns
    critical = ["title", "type", "release_year"]
    df = df.dropna(subset=critical, how="any")
    return df

def clean_imdb(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the IMDb reviews dataset.
    Steps:
    * Rename columns to lower case, strip whitespace
    * Map sentiment strings to binary (positive=1, negative=0)
    * Drop rows with missing review text or sentiment label
    * Remove duplicates based on review text
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Expected columns: "review", "sentiment" (text labels)
    # Convert sentiment to binary
    sentiment_map = {'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0}
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].str.lower().map(sentiment_map)
    # Strip whitespace in text columns
    if 'review' in df.columns:
        df['review'] = df['review'].astype(str).str.strip()
        df['review'] = df['review'].replace({'nan': None, 'None': None, '': None})
    # Drop rows with missing values in essentials
    df = df.dropna(subset=['review', 'sentiment'])
    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['review'], keep='first')
    return df

def save_processed(df: pd.DataFrame, filename: str) -> str:
    """Save a cleaned DataFrame to the processed data directory.
    Returns the absolute path of the saved file.
    """
    _ensure_processed_dir()
    out_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out_path, index=False)
    return out_path

def run_all_cleaning() -> Tuple[str, str]:
    """Convenience function that cleans both datasets and returns paths to processed CSV files."""
    netflix_raw_path = os.path.join(RAW_DIR, "netflix_titles.csv")
    imdb_raw_path = os.path.join(RAW_DIR, "IMDB Dataset.csv")

    if os.path.exists(netflix_raw_path):
        netflix_clean = clean_netflix(pd.read_csv(netflix_raw_path))
        netflix_path = save_processed(netflix_clean, "netflix_cleaned.csv")
    else:
        netflix_path = os.path.join(PROCESSED_DIR, "netflix_cleaned.csv")

    if os.path.exists(imdb_raw_path):
        imdb_clean = clean_imdb(pd.read_csv(imdb_raw_path))
        imdb_path = save_processed(imdb_clean, "imdb_cleaned.csv")
    else:
        imdb_path = os.path.join(PROCESSED_DIR, "imdb_cleaned.csv")

    return netflix_path, imdb_path


