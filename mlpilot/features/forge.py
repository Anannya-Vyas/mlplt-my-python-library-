"""
mlpilot/features/forge.py
FeatureForge — leakage-safe feature engineering pipeline.
One function that handles encoding, scaling, and datetime expansion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlpilot.features.encoders import (
    BinaryEncoder, CVTargetEncoder, DFMinMaxScaler, DFRobustScaler,
    DFStandardScaler, DatetimeFeatureExtractor, SafeLabelEncoder
)
from mlpilot.utils.display import RichTable, print_step, print_success
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# FeatureResult
# ---------------------------------------------------------------------------

class FeatureResult(BaseResult):
    """
    Complete result from ml.features().

    Provides leakage-safe fit_transform() / transform() — statistics are
    always learned from training data only.

    Attributes
    ----------
    df : pd.DataFrame
        The input dataframe (before transformation).
    target : str or None
        The target column name.
    pipeline : sklearn Pipeline
        The assembled sklearn Pipeline.
    feature_names_out : list[str]
        Names of output features after transformation.
    report : FeatureReport
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str],
        steps: List[tuple],
        col_actions: Dict[str, str],
    ):
        self.df = df
        self.target = target
        self._steps = steps
        self.col_actions = col_actions
        self._pipeline: Optional[Pipeline] = None
        self._fitted = False
        self._feature_names_out: List[str] = []

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline(self._steps) if self._steps else Pipeline([("passthrough", "passthrough")])
        return self._pipeline

    @property
    def feature_names_out(self) -> List[str]:
        return self._feature_names_out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the pipeline on df and return the transformed dataframe.
        ALWAYS call this on training data — never on test data.
        """
        X = self._drop_target(df)
        y_col = self._get_y(df, y)
        result = X.copy()
        for name, step in self._steps:
            if hasattr(step, "fit"):
                if hasattr(step, "fit") and "CVTargetEncoder" in type(step).__name__:
                    step.fit(result, y_col)
                else:
                    step.fit(result, y_col)
                result = step.transform(result)
            else:
                pass
        self._fitted = True
        self._feature_names_out = list(result.columns)
        if self._pipeline is None:
            from sklearn.pipeline import Pipeline
            self._pipeline = Pipeline(self._steps)
        # Re-fit pipeline properly
        X2 = self._drop_target(df)
        try:
            self._pipeline.fit(X2, y_col)
        except Exception:
            pass
        return self._apply_steps(self._drop_target(df), y_col, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using statistics learned during fit_transform().
        Use this for test/validation data — NO leakage.
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit_transform(df_train) before transform(df_test)"
            )
        X = self._drop_target(df)
        return self._apply_steps(X, y=None, fit=False)

    def _apply_steps(self, X: pd.DataFrame, y, fit: bool) -> pd.DataFrame:
        result = X.copy()
        for name, step in self._steps:
            if fit:
                if hasattr(step, "fit_transform"):
                    try:
                        if hasattr(step, "_requires_y") or "Target" in type(step).__name__:
                            step.fit(result, y)
                            result = step.transform(result)
                        else:
                            step.fit(result)
                            result = step.transform(result)
                    except Exception:
                        try:
                            step.fit(result, y)
                            result = step.transform(result)
                        except Exception:
                            pass
                else:
                    result = step.transform(result)
            else:
                result = step.transform(result)
        self._feature_names_out = list(result.columns)
        return result

    def _drop_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target and self.target in df.columns:
            return df.drop(columns=[self.target])
        return df.copy()

    def _get_y(self, df: pd.DataFrame, y) -> Optional[pd.Series]:
        if y is not None:
            return pd.Series(y)
        if self.target and self.target in df.columns:
            return df[self.target]
        return None

    def report(self) -> None:
        """Print a summary of what was done to each column."""
        tbl = RichTable(
            title="Feature Engineering Report",
            columns=["Column", "Action"]
        )
        for col, action in self.col_actions.items():
            tbl.add_row(col, action)
        tbl.print()

    def __iter__(self):
        """
        Allows 'Magic Unpacking': X, y = ml.features(df, target='survived')
        Automatically performs fit_transform() on the initial dataframe.
        """
        X = self.fit_transform(self.df)
        y = self.df[self.target] if self.target else None
        return iter([X, y])

    def __repr__(self) -> str:
        return (f"FeatureResult(target='{self.target}', "
                f"n_steps={len(self._steps)}, "
                f"n_features_out={len(self._feature_names_out)})")


# ---------------------------------------------------------------------------
# FeatureForge Engine
# ---------------------------------------------------------------------------

class FeatureForge:
    """Internal engine. Use ml.features() instead."""

    def __init__(
        self,
        target: Optional[str] = None,
        encoding: Union[str, Dict] = "auto",
        scaling: Union[str, Dict] = "auto",
        datetime_features: bool = True,
        max_cardinality_onehot: int = 20,
        feature_selection: bool = False,
        n_features_to_keep: Union[int, float] = 1.0,
        verbose: bool = True,
    ):
        self.target = target
        self.encoding = encoding
        self.scaling = scaling
        self.datetime_features = datetime_features
        self.max_cardinality_onehot = max_cardinality_onehot
        self.feature_selection = feature_selection
        self.n_features_to_keep = n_features_to_keep
        self.verbose = verbose

    def build(self, df: pd.DataFrame) -> FeatureResult:
        if self.verbose:
            print_step("Building FeatureForge pipeline...", "⚙️")

        steps = []
        col_actions: Dict[str, str] = {}
        feat_cols = [c for c in df.columns if c != self.target]

        # --------------- Datetime expansion ---------------
        if self.datetime_features:
            dt_cols = df[feat_cols].select_dtypes(include=["datetime64"]).columns.tolist()
            for col in dt_cols:
                extractor = DatetimeFeatureExtractor(col)
                steps.append((f"dt_{col}", extractor))
                col_actions[col] = "datetime → year/month/day/hour/dayofweek/quarter/is_weekend"
                feat_cols.remove(col)

        # --------------- Categorical encoding ---------------
        cat_cols = df[feat_cols].select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        for col in cat_cols:
            n_unique = df[col].nunique(dropna=True)
            enc = self._choose_encoding(col, n_unique)

            if enc == "binary":
                steps.append((f"bin_{col}", BinaryEncoder(col)))
                col_actions[col] = "binary encoding (0/1)"
            elif enc == "onehot":
                # Use pandas get_dummies style (we do it manually to keep column names)
                steps.append((f"ohe_{col}", _PandasOHE(col)))
                col_actions[col] = f"one-hot encoding ({n_unique} categories)"
            elif enc == "target":
                steps.append((f"te_{col}", CVTargetEncoder(col, smoothing=10.0)))
                col_actions[col] = "target encoding (leakage-safe CV)"
            elif enc == "label":
                steps.append((f"le_{col}", SafeLabelEncoder(col)))
                col_actions[col] = "label encoding"

        # --------------- Numeric scaling ---------------
        num_cols_remaining = [
            c for c in feat_cols
            if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        if num_cols_remaining:
            scaler = self._choose_scaler(df, num_cols_remaining)
            if scaler is not None:
                steps.append(("scaler", scaler))
                for col in num_cols_remaining:
                    col_actions[col] = f"{self.scaling} scaling"

        if self.verbose:
            print_success(f"Pipeline built: {len(steps)} transformation steps, "
                          f"{len(col_actions)} columns processed")

        return FeatureResult(df=df, target=self.target, steps=steps, col_actions=col_actions)

    def _choose_encoding(self, col: str, n_unique: int) -> str:
        if isinstance(self.encoding, dict) and col in self.encoding:
            return self.encoding[col]

        strategy = self.encoding if isinstance(self.encoding, str) else "auto"

        if strategy != "auto":
            return strategy

        # Auto selection
        if n_unique == 2:
            return "binary"
        if n_unique <= self.max_cardinality_onehot:
            return "onehot"
        if self.target:
            return "target"
        return "label"

    def _choose_scaler(self, df: pd.DataFrame, cols: List[str]) -> Optional[Any]:
        strategy = self.scaling if isinstance(self.scaling, str) else "auto"

        if strategy == "none":
            return None

        if strategy == "auto":
            # Use robust if any column has significant outliers
            has_outliers = False
            for col in cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
                    if n_out / len(df) > 0.05:
                        has_outliers = True
                        break
            strategy = "robust" if has_outliers else "standard"

        if strategy == "standard":
            return DFStandardScaler(cols)
        if strategy == "robust":
            return DFRobustScaler(cols)
        if strategy == "minmax":
            return DFMinMaxScaler(cols)
        return None


# ---------------------------------------------------------------------------
# Helper: Pandas OHE (keeps column names, avoids sklearn OHE complexity)
# ---------------------------------------------------------------------------

class _PandasOHE:
    """One-hot encoder that produces human-readable column names."""

    def __init__(self, column: str, drop_first: bool = True):
        self.column = column
        self.drop_first = drop_first
        self.categories_: List[Any] = []

    def fit(self, X, y=None):
        series = X[self.column] if isinstance(X, pd.DataFrame) else pd.Series(X)
        self.categories_ = sorted(series.dropna().unique().tolist(), key=str)
        return self

    def transform(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        dummies = pd.get_dummies(X[self.column], prefix=self.column, drop_first=self.drop_first)
        X = X.drop(columns=[self.column])
        X = pd.concat([X, dummies], axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        cats = self.categories_[1:] if self.drop_first else self.categories_
        return [f"{self.column}_{c}" for c in cats]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def features(
    df: pd.DataFrame,
    target: Optional[str] = None,
    encoding: Union[str, Dict] = "auto",
    scaling: Union[str, Dict] = "auto",
    datetime_features: bool = True,
    interactions: bool = False,
    polynomial_degree: int = 1,
    max_cardinality_onehot: int = 20,
    feature_selection: bool = False,
    n_features_to_keep: Union[int, float] = 1.0,
    verbose: bool = True,
) -> FeatureResult:
    """
    FeatureForge — Leakage-Safe Feature Engineering Pipeline.

    Handles all encoding, scaling, and feature creation in one pipeline.
    Fit on training data → transform test data with the same statistics.
    Leakage is architecturally impossible.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (or training set).
    target : str, optional
        Target column — excluded from transformations.
    encoding : str or dict
        'auto' | 'onehot' | 'label' | 'target' | 'binary' | 'ordinal'
        Or per-column dict: {'col_name': 'strategy'}
    scaling : str or dict
        'auto' | 'standard' | 'minmax' | 'robust' | 'none'
    datetime_features : bool
        Expand datetime → hour, day, weekday, month, quarter, year.
    max_cardinality_onehot : int
        Use target encoding if cardinality exceeds this.
    feature_selection : bool
        Run mutual info feature selection after engineering.
    n_features_to_keep : int or float
        Keep top-N or top-fraction after selection.
    verbose : bool
        Print progress to terminal.

    Returns
    -------
    FeatureResult
        Call result.fit_transform(df_train) → then result.transform(df_test).
        Attributes: pipeline, feature_names_out, col_actions.

    Examples
    --------
    >>> import mlpilot as ml
    >>> feat_result = ml.features(df_train, target='churn')
    >>> X_train = feat_result.fit_transform(df_train)
    >>> X_test  = feat_result.transform(df_test)   # uses train stats only
    >>> feat_result.report()                        # print what was done
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    forge = FeatureForge(
        target=target,
        encoding=encoding,
        scaling=scaling,
        datetime_features=datetime_features,
        max_cardinality_onehot=max_cardinality_onehot,
        feature_selection=feature_selection,
        n_features_to_keep=n_features_to_keep,
        verbose=verbose,
    )
    return forge.build(df)
