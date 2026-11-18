# StreamSense AI ✦
### Streaming Platform Content & Audience Sentiment Analytics
**Enterprise AI Analytics & Natural Language Processing Platform**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%201.38-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn%201.5-F7931E.svg)](https://scikit-learn.org/)
[![Visualizations](https://img.shields.io/badge/charts-Plotly-8B5CF6.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

StreamSense AI is an enterprise-grade AI analytics and streaming intelligence SaaS platform built for high-performance exploratory content analytics and binary sentiment classification. It synthesizes metadata across **8,807 curated Netflix titles** with **49,582 deduplicated audience reviews** from the Stanford IMDb corpus, powering interactive exploratory data analysis, pipeline-hardened machine learning, and real-time sentiment inference.

---

## 1. Quick Deploy to Streamlit Community Cloud

This repository is pre-configured and fully verified for one-click deployment on **Streamlit Community Cloud**:

1. **Fork or Push** this repository to your personal GitHub account.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **"New app"**.
4. Select:
   - **Repository:** `your-username/StreamSense-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**
   - The application will boot automatically within 60–90 seconds.
   - All pre-trained models (`logreg_model.joblib`, `nb_model.joblib`, `tfidf_pipeline.joblib`) and curated datasets (`netflix_cleaned.csv`, `imdb_cleaned.csv`) are bundled inside the repository, so the application runs immediately without requiring manual training.

---

## 2. Platform Architecture & Modules

The platform is organized into four hierarchical navigation clusters comprising nine specialized analytics modules:

| Cluster | Page Name | Core Functionality & Deliverables |
|---|---|---|
| **Overview** | **Executive Dashboard** | 4 primary KPI cards, catalogue format distribution (69.6% Movies vs 30.4% TV Shows), top genre concentration, and sentiment polarity breakdown. |
| **Data Intelligence** | **Data Upload & Wrangling** | Drag-and-drop CSV ingestion, missing-value audit, duplicate detection, and automated cleaning routines. |
| | **Content Intelligence** | In-depth Netflix analysis: runtime histograms, TV season counts, international production hubs, and multi-year release velocity. |
| | **Sentiment Analysis** | IMDb review analytics: polarity distribution (24,884 positive vs 24,698 negative), word count histograms, and interactive corpus explorer. |
| **Machine Learning** | **Model Training Studio** | Interactive pipeline trainer with customizable TF-IDF feature vocabulary (1,000–10,000) and n-gram ranges. |
| | **Model Evaluation Lab** | Empirical benchmark matrix, ROC curves, latency comparisons, and fully labeled $2 \times 2$ confusion matrices. |
| | **Live Review Analyzer** | Real-time sentiment prediction engine with probability gauges, confidence meters, and SQLite transaction logging. |
| **Insights & Architecture** | **Research Findings** | Strategic synthesis of catalogue trends, commercial content licensing suggestions, and algorithmic sentiment takeaways. |
| | **About & Methodology** | System architecture diagrams, mathematical formulation of metrics, software dependencies, and academic citations. |

---

## 3. Empirical Machine Learning Benchmarks

Evaluated on the exact held-out 20% test partition (**$N = 9,917$ reviews**, stratified split, `random_state=42`):

| Evaluation Metric | Logistic Regression | Multinomial Naive Bayes | Benchmark Definition / Formula |
|:---|:---:|:---:|:---|
| **Accuracy** | **89.41%** (0.8941) | 86.38% (0.8638) | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | **88.61%** (0.8861) | 85.22% (0.8522) | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | **90.54%** (0.9054) | 88.15% (0.8815) | $\frac{TP}{TP + FN}$ |
| **Specificity** | **88.28%** (0.8828) | 84.60% (0.8460) | $\frac{TN}{TN + FP}$ *(safeguarded against zero-division)* |
| **F1-Score** | **89.56%** (0.8956) | 86.66% (0.8666) | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC-AUC** | **0.9614** | 0.9386 | Trapezoidal area under ROC curve |
| **Confusion Matrix** | **TP=4506, TN=4361, FP=579, FN=471** | **TP=4387, TN=4179, FP=761, FN=590** | Empirical $2 \times 2$ matrix |
| **Model Fit Time** | ~1.85 seconds | ~0.15 seconds | Training time on 39,665 training samples |
| **Inference Latency** | ~35 ms (3.5 µs/review) | ~8 ms (0.8 µs/review) | Batch throughput on 9,917 test samples |
| **Test Set Count** | 9,917 reviews | 9,917 reviews | Stratified held-out test split |

---

## 4. Local Development & Execution

```bash
# 1. Clone the repository
git clone https://github.com/your-username/StreamSense-AI.git
cd StreamSense-AI

# 2. (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```
Open **`http://localhost:8501`** in any modern web browser.

---

## 5. Cloud Architecture & Storage Notice

- **Memory Efficiency**: Datasets and evaluation pipelines utilize `@st.cache_data` caching to operate within Streamlit Community Cloud's standard 1 GB container RAM ceiling.
- **Ephemeral Storage**: In hosted container environments (such as Streamlit Community Cloud), container filesystem storage is ephemeral. Inferences logged to `models/history.db` persist across user interactions within the active session, but will reset upon container cold restarts or redeployments. For persistent database logging across container lifecycles in enterprise environments, connection to an external managed database (e.g. Supabase, PostgreSQL, or Google Cloud SQL) is recommended.

---

## 6. Academic References & Citations

1. **IMDb Dataset**: Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142–150.
2. **Scikit-learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
3. **Streamlit**: Streamlit Inc. (2024). *Streamlit: The fastest way to build and share data apps*.
