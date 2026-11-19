"""
StreamSense AI — Movie & TV Show Recommendation System
Content-based filtering using TF-IDF vectorization and cosine similarity over Netflix metadata.
"""

import os
import re
import html
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _clean_str(val: Any) -> str:
    """Helper to clean string metadata and handle NaNs."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ["unknown director", "unknown cast", "nan", "none"]:
        return ""
    return s


def build_metadata_soup(df: pd.DataFrame) -> pd.Series:
    """Combine relevant metadata attributes into a single textual representation (soup).
    Weights genres and format slightly higher by repetition.
    """
    def row_to_soup(row):
        mtype = _clean_str(row.get("type", ""))
        genres = _clean_str(row.get("listed_in", ""))
        # Repeat genres 2x to reinforce topical clustering in TF-IDF
        genre_tokens = f"{genres} {genres}" if genres else ""
        director = _clean_str(row.get("director", ""))
        cast = _clean_str(row.get("cast", ""))
        country = _clean_str(row.get("country", ""))
        description = _clean_str(row.get("description", ""))
        
        soup_parts = [mtype, genre_tokens, director, cast, country, description]
        return " ".join([p for p in soup_parts if p]).lower()

    return df.apply(row_to_soup, axis=1)


def build_tfidf_matrix(df: pd.DataFrame):
    """Fit TfidfVectorizer over metadata soups and return the sparse feature matrix."""
    soups = build_metadata_soup(df)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(soups)
    return tfidf_matrix, vectorizer


def parse_items(val: Any) -> set:
    """Parse comma-separated metadata into a normalized set of lowercased tokens."""
    cleaned = _clean_str(val)
    if not cleaned:
        return set()
    return {x.strip().lower() for x in cleaned.split(",") if x.strip()}


def generate_explanation(target: pd.Series, candidate: pd.Series) -> str:
    """Generate an evidence-grounded textual explanation for why candidate was recommended."""
    reasons = []

    # 1. Check Shared Genres
    t_genres = parse_items(target.get("listed_in"))
    c_genres = parse_items(candidate.get("listed_in"))
    shared_g = t_genres & c_genres
    if shared_g:
        genre_str = ", ".join([g.title() for g in sorted(shared_g)[:3]])
        reasons.append(f"shared genre classification (<strong>{html.escape(genre_str)}</strong>)")

    # 2. Check Shared Director
    t_dir = _clean_str(target.get("director"))
    c_dir = _clean_str(candidate.get("director"))
    if t_dir and c_dir and t_dir.lower() == c_dir.lower():
        reasons.append(f"directed by the same filmmaker (<strong>{html.escape(t_dir)}</strong>)")

    # 3. Check Shared Cast
    t_cast = parse_items(target.get("cast"))
    c_cast = parse_items(candidate.get("cast"))
    shared_c = t_cast & c_cast
    if shared_c:
        cast_str = ", ".join([c.title() for c in sorted(shared_c)[:2]])
        reasons.append(f"features shared lead cast (<strong>{html.escape(cast_str)}</strong>)")

    # 4. Check Shared Country
    t_country = parse_items(target.get("country"))
    c_country = parse_items(candidate.get("country"))
    shared_country = t_country & c_country
    if shared_country and "united states" not in [x.lower() for x in shared_country]:
        c_str = ", ".join([c.title() for c in sorted(shared_country)[:2]])
        reasons.append(f"originates from the same international cinema market (<strong>{html.escape(c_str)}</strong>)")

    # 5. Format Alignment
    same_format = (str(target.get("type", "")).strip().lower() == str(candidate.get("type", "")).strip().lower())

    if reasons:
        explanation = "Recommended because it shares " + "; ".join(reasons) + "."
    elif same_format:
        fmt = str(target.get("type", "title"))
        explanation = f"Recommended based on strong narrative thematic alignment and shared {fmt.lower()} storytelling tone."
    else:
        explanation = "Recommended based on cross-format narrative similarities in plot synopsis and thematic tone."

    return explanation


def get_recommendations(
    df: pd.DataFrame,
    tfidf_matrix,
    selected_title: str,
    top_n: int = 5,
    format_filter: str = "All"
) -> List[Dict[str, Any]]:
    """Retrieve top N similar movies or TV shows for a selected Netflix title.
    
    Parameters:
        df: Cleaned Netflix DataFrame.
        tfidf_matrix: Pre-computed sparse TF-IDF matrix.
        selected_title: The movie or TV show title selected by the user.
        top_n: Number of recommendations to return (e.g. 5 or 10).
        format_filter: 'All', 'Movie', or 'TV Show'.
        
    Returns:
        List of dictionaries containing title metadata, similarity score, and explanation.
    """
    if df is None or len(df) == 0 or tfidf_matrix is None:
        return []

    # Find the title index case-insensitively
    matches = df[df["title"].astype(str).str.strip().str.lower() == selected_title.strip().lower()]
    if matches.empty:
        return []

    target_idx = matches.index[0]
    target_row = df.loc[target_idx]

    # Calculate cosine similarity of target against all titles using linear_kernel
    # linear_kernel on L2-normalized TF-IDF is identical to cosine_similarity but faster
    cosine_sim_vector = linear_kernel(tfidf_matrix[target_idx], tfidf_matrix).flatten()

    # Rank indices from highest to lowest similarity
    ranked_indices = cosine_sim_vector.argsort()[::-1]

    recommendations = []
    for idx in ranked_indices:
        # 1. Never recommend the exact same title selected by the user
        if idx == target_idx:
            continue

        candidate_row = df.loc[idx]

        # 2. Apply format filter if user requested
        if format_filter in ["Movie", "TV Show"]:
            if str(candidate_row.get("type", "")).strip() != format_filter:
                continue

        score = float(cosine_sim_vector[idx])
        # Skip titles with 0 similarity score
        if score <= 0.0001 and len(recommendations) >= 3:
            break

        explanation = generate_explanation(target_row, candidate_row)

        recommendations.append({
            "show_id": candidate_row.get("show_id", ""),
            "title": candidate_row.get("title", "Unknown Title"),
            "type": candidate_row.get("type", "Movie"),
            "release_year": int(candidate_row["release_year"]) if pd.notna(candidate_row.get("release_year")) else "N/A",
            "rating": candidate_row.get("rating", "Not Rated") if pd.notna(candidate_row.get("rating")) else "Not Rated",
            "duration": candidate_row.get("duration", "") if pd.notna(candidate_row.get("duration")) else "",
            "listed_in": candidate_row.get("listed_in", "") if pd.notna(candidate_row.get("listed_in")) else "",
            "country": candidate_row.get("country", "") if pd.notna(candidate_row.get("country")) else "",
            "director": _clean_str(candidate_row.get("director", "")) or "Unknown",
            "cast": _clean_str(candidate_row.get("cast", "")) or "Unknown",
            "description": candidate_row.get("description", "No synopsis available.") if pd.notna(candidate_row.get("description")) else "No synopsis available.",
            "similarity_score": round(score, 4),
            "similarity_pct": f"{score * 100:.1f}%",
            "explanation": explanation
        })

        if len(recommendations) >= top_n:
            break

    return recommendations

