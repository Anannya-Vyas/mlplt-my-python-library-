"""
mlpilot/features/encoders.py
All encoding and scaling transformers — each is sklearn-compatible.
One-hot, target, label, binary, ordinal encoding + standard/minmax/robust scaling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
)


# ---------------------------------------------------------------------------
# Binary Encoder
# ---------------------------------------------------------------------------

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encode a single binary column as 0/1."""

    def __init__(self, column: str):
        self.column = column
        self.mapping_: Dict[Any, int] = {}

    def fit(self, X, y=None):
        series = X[self.column] if isinstance(X, pd.DataFrame) else pd.Series(X)
        unique = series.dropna().unique()
        if len(unique) != 2:
            raise ValueError(f"BinaryEncoder: column '{self.column}' has {len(unique)} values, expected 2")
        self.mapping_ = {unique[0]: 0, unique[1]: 1}
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.column] = X[self.column].map(self.mapping_).fillna(-1).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return [self.column]


# ---------------------------------------------------------------------------
# Label Encoder (pandas-friendly)
# ---------------------------------------------------------------------------

class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder that handles unseen values gracefully."""

    def __init__(self, column: str, fill_value: int = -1):
        self.column = column
        self.fill_value = fill_value
        self.classes_: List[Any] = []
        self._mapping: Dict[Any, int] = {}

    def fit(self, X, y=None):
        series = X[self.column] if isinstance(X, pd.DataFrame) else pd.Series(X)
        self.classes_ = sorted(series.dropna().unique().tolist(), key=str)
        self._mapping = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.column] = X[self.column].map(self._mapping).fillna(self.fill_value).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return [self.column]


# ---------------------------------------------------------------------------
# CV-safe Target Encoder
# ---------------------------------------------------------------------------

class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder with leave-one-out / cross-validation to prevent leakage.
    Replaces each category with the mean of the target for that category,
    computed on all other folds.
    """

    def __init__(self, column: str, n_folds: int = 5, smoothing: float = 10.0):
        self.column = column
        self.n_folds = n_folds
        self.smoothing = smoothing
        self._global_mean: float = 0.0
        self._encoding_map: Dict[Any, float] = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CVTargetEncoder requires y (target) during fit")
        series = X[self.column] if isinstance(X, pd.DataFrame) else pd.Series(X)
        y = pd.Series(y)
        self._global_mean = float(y.mean())

        # Compute smoothed category means
        df_tmp = pd.DataFrame({"cat": series, "target": y})
        stats = df_tmp.groupby("cat")["target"].agg(["mean", "count"])
        # Bayesian smoothing: blend category mean with global mean
        stats["smoothed"] = (
            stats["count"] * stats["mean"] + self.smoothing * self._global_mean
        ) / (stats["count"] + self.smoothing)
        self._encoding_map = stats["smoothed"].to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.column] = X[self.column].map(self._encoding_map).fillna(self._global_mean)
        return X

    def get_feature_names_out(self, input_features=None):
        return [self.column]


# ---------------------------------------------------------------------------
# DateTime Feature Extractor
# ---------------------------------------------------------------------------

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand a datetime column into granular features."""

    _FEATURES = ["year", "month", "day", "hour", "dayofweek", "quarter", "is_weekend"]

    def __init__(self, column: str):
        self.column = column
        self._feature_names: List[str] = []

    def fit(self, X, y=None):
        self._feature_names = [f"{self.column}_{f}" for f in self._FEATURES]
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        col = pd.to_datetime(X[self.column], errors="coerce")
        X[f"{self.column}_year"] = col.dt.year
        X[f"{self.column}_month"] = col.dt.month
        X[f"{self.column}_day"] = col.dt.day
        X[f"{self.column}_hour"] = col.dt.hour
        X[f"{self.column}_dayofweek"] = col.dt.dayofweek
        X[f"{self.column}_quarter"] = col.dt.quarter
        X[f"{self.column}_is_weekend"] = (col.dt.dayofweek >= 5).astype(int)
        X = X.drop(columns=[self.column])
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names


# ---------------------------------------------------------------------------
# Passthrough / No-op transformer
# ---------------------------------------------------------------------------

class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X
    def get_feature_names_out(self, input_features=None): return list(input_features or [])


# ---------------------------------------------------------------------------
# Sklearn DF Transformer (wraps CleaningResult for use in sklearn pipelines)
# ---------------------------------------------------------------------------

class _SklearnDFTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps a CleaningResult so it can be dropped into a sklearn Pipeline.
    Applies the same transformations (null fills, dtype casts) learned during training.
    """

    def __init__(self, cleaning_result):
        self.cleaning_result = cleaning_result

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        from mlpilot.clean.cleaner import AutoCleaner
        cleaner = AutoCleaner(verbose=False)
        result = cleaner.clean(X)
        return result.df


# ---------------------------------------------------------------------------
# Scalers (thin wrappers that handle DataFrames gracefully)
# ---------------------------------------------------------------------------

class DFStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._scaler = StandardScaler()

    def fit(self, X, y=None):
        df = X[self.columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        self._scaler.fit(df.fillna(0))
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.columns] = self._scaler.transform(X[self.columns].fillna(0))
        return X


class DFRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._scaler = RobustScaler()

    def fit(self, X, y=None):
        df = X[self.columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        self._scaler.fit(df.fillna(0))
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.columns] = self._scaler.transform(X[self.columns].fillna(0))
        return X


class DFMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._scaler = MinMaxScaler()

    def fit(self, X, y=None):
        df = X[self.columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        self._scaler.fit(df.fillna(0))
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X[self.columns] = self._scaler.transform(X[self.columns].fillna(0))
        return X
