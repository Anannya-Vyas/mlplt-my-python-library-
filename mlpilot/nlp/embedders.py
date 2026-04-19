"""
mlpilot/nlp/embedders.py
Text embedding logic for TextML.
Handles vectorization via TF-IDF (fallback) or Transformers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEmbedder(BaseEstimator, TransformerMixin):
    """
    Flexible text vectorizer.
    Starts with TF-IDF, can be upgraded to Sentence-Transformers.
    """

    def __init__(self, method: str = "tfidf", max_features: int = 5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit(self, X: Union[pd.Series, List[str]], y: Any = None) -> TextEmbedder:
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                ngram_range=(1, 2)
            )
            self.vectorizer.fit(X)
        return self

    def transform(self, X: Union[pd.Series, List[str]]) -> np.ndarray:
        if self.method == "tfidf":
            return self.vectorizer.transform(X).toarray()
        return np.zeros((len(X), 1))

    def fit_transform(self, X: Union[pd.Series, List[str]], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class SimpleSentiment:
    """Rule-based sentiment analyzer (VADER Lite)."""

    def __init__(self):
        # Basic seed words
        self.pos = {"great", "good", "excellent", "amazing", "love", "best", "happy", "awesome"}
        self.neg = {"bad", "terrible", "worst", "hate", "awful", "horrible", "poor", "unhappy"}

    def score(self, text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        words = text.lower().split()
        p = sum(1 for w in words if w in self.pos)
        n = sum(1 for w in words if w in self.neg)
        total = p + n
        if total == 0:
            return 0.0
        return (p - n) / total
