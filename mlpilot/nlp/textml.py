"""
mlpilot/nlp/textml.py
TextML — The NLP toolkit for mlpilot.
One-liners for sentiment, classification, topics, and embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlpilot.nlp.embedders import SimpleSentiment, TextEmbedder
from mlpilot.utils.display import print_step, print_success, print_warning
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# NLP Result Objects
# ---------------------------------------------------------------------------

class TextClassificationResult(BaseResult):
    def __init__(self, model, metrics, target):
        self.model = model
        self.metrics = metrics
        self.target = target

    def print_report(self):
        print_success(f"Text Classification complete for '{self.target}'")
        for k, v in self.metrics.items():
            print_step(f"{k}: {v:.4f}", "📊")


class TopicResult(BaseResult):
    def __init__(self, topics: List[List[str]]):
        self.topics = topics

    def print_topics(self):
        print_success(f"Discovered {len(self.topics)} topics:")
        for i, words in enumerate(self.topics):
            print_step(f"Topic {i+1}: {', '.join(words[:5])}", "🏷️")

    def __repr__(self) -> str:
        lines = [f"TopicResult(n_topics={len(self.topics)}):"]
        for i, words in enumerate(self.topics):
            lines.append(f"  Topic {i+1}: {', '.join(words[:8])}")
        return "\n".join(lines)

    def __iter__(self):
        return iter(self.topics)

    def __getitem__(self, idx):
        return self.topics[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_classify(
    df: pd.DataFrame,
    text_col: str,
    target: str,
    verbose: bool = True,
) -> TextClassificationResult:
    """
    Automated Text Classification.
    Vectorizes text and trains a baseline model.
    """
    if verbose:
        print_step(f"TextML: Training classifier for '{target}'...", "📝")

    X = df[text_col].fillna("")
    y = df[target]

    pipeline = Pipeline([
        ("vect", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X, y)

    # In a real tool, we'd do a split and evaluation here
    metrics = {"accuracy_estimate": 0.0}  # Placeholder

    res = TextClassificationResult(pipeline, metrics, target)
    if verbose:
        res.print_report()
    return res


def sentiment(series: pd.Series, verbose: bool = True) -> pd.Series:
    """
    Analyze sentiment of a text series.
    Returns a series from -1 (negative) to 1 (positive).
    """
    try:
        from textblob import TextBlob
        if verbose:
            print_step("TextML: Analyzing sentiment via TextBlob...", "🎭")
        return series.fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
    except ImportError:
        if verbose:
            print_warning("textblob not installed — using fallback rule-based analyzer.")
        analyzer = SimpleSentiment()
        return series.fillna("").apply(analyzer.score)


def topics(series: pd.Series, n_topics: int = 5, verbose: bool = True) -> TopicResult:
    """
    Discover hidden topics in a text collection.
    """
    if verbose:
        print_step(f"TextML: Extracting {n_topics} topics...", "🔍")

    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    X = tfidf.fit_transform(series.fillna(""))

    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(X)

    feature_names = tfidf.get_feature_names_out()
    discovered_topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        discovered_topics.append(top_words)

    res = TopicResult(discovered_topics)
    if verbose:
        res.print_topics()
    return res


def embed(series: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """
    Convert text series to numerical embeddings.
    """
    if verbose:
        print_step("TextML: Vectorizing text...", "🔢")
    embedder = TextEmbedder()
    vecs = embedder.fit_transform(series.fillna(""))
    return pd.DataFrame(vecs, columns=[f"dim_{i}" for i in range(vecs.shape[1])])
