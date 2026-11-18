import os
import pandas as pd
import plotly.express as px
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

def load_processed(filename: str) -> pd.DataFrame:
    """Load a cleaned CSV from the processed directory."""
    path = PROCESSED_DIR / filename
    return pd.read_csv(path)

def genre_distribution(df: pd.DataFrame):
    """Return a Plotly pie chart of genre frequencies."""
    # The Netflix dataset stores genres as a comma‑separated string
    genre_series = df["listed_in"].dropna().str.split(",")
    all_genres = [g.strip() for sublist in genre_series for g in sublist]
    genre_counts = pd.Series(all_genres).value_counts().nlargest(15)
    fig = px.pie(values=genre_counts.values, names=genre_counts.index,
                 title="Top 15 Genres (Netflix)",
                 color_discrete_sequence=px.colors.sequential.Plasma)
    return fig

def country_distribution(df: pd.DataFrame):
    """Bar chart of the most common production countries."""
    if "country" not in df.columns:
        return None
    country_counts = df["country"].dropna().value_counts().nlargest(15)
    fig = px.bar(x=country_counts.index, y=country_counts.values,
                 title="Top 15 Countries (Netflix)",
                 labels={"x": "Country", "y": "Count"},
                 color_discrete_sequence=["#A78BFA"])
    return fig

def release_year_trends(df: pd.DataFrame):
    """Line chart showing number of titles released each year, split by type."""
    if "release_year" not in df.columns:
        return None
    yearly = df.groupby(["release_year", "type"]).size().reset_index(name="count")
    fig = px.line(yearly, x="release_year", y="count", color="type",
                  title="Release‑Year Trends (Movies vs TV Shows)",
                  labels={"release_year": "Year", "count": "Titles"})
    return fig

def duration_analysis(df: pd.DataFrame):
    """Box plot of duration (minutes for movies, seasons for TV shows)."""
    if "duration_parsed" not in df.columns:
        return None
    fig = px.box(df, x="type", y="duration_parsed",
                 title="Duration Distribution by Type",
                 labels={"type": "Content Type", "duration_parsed": "Duration (min or seasons)"},
                 color="type")
    return fig

