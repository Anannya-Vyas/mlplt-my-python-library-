"""
mlpilot/clean/strategies.py
Null imputation, outlier handling, dtype fixing, category unification,
and duplicate removal strategies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Null Handling
# ---------------------------------------------------------------------------

class NullStrategy:
    """
    Decides and applies null-filling strategy per column.
    All statistics (medians, modes, KNN fits) are computed on train data only.
    """

    def __init__(
        self,
        strategy: str = "auto",
        null_threshold: float = 0.5,
    ):
        self.strategy = strategy
        self.null_threshold = null_threshold
        self._fill_values: Dict[str, Any] = {}
        self._dropped_cols: List[str] = []
        self._knn_imputer = None

    def fit(self, df: pd.DataFrame, protect_cols: Optional[List[str]] = None) -> "NullStrategy":
        protect_cols = protect_cols or []
        cols_to_process = [c for c in df.columns if df[c].isnull().any() and c not in protect_cols]

        for col in cols_to_process:
            pct = df[col].isnull().mean()
            series = df[col]

            if pct > self.null_threshold:
                self._dropped_cols.append(col)
                continue

            chosen = self._choose_strategy(series, pct)

            if chosen in ("median", "auto_numeric_low"):
                self._fill_values[col] = ("median", float(series.median()))
            elif chosen == "mean":
                self._fill_values[col] = ("mean", float(series.mean()))
            elif chosen in ("mode", "auto_cat_low"):
                mode_val = series.mode()
                self._fill_values[col] = ("mode", mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN")
            elif chosen == "knn":
                # KNN is handled separately below
                self._fill_values[col] = ("knn", None)
            elif chosen == "zero":
                self._fill_values[col] = ("zero", 0)

        # Fit KNN imputer if any column needs it
        knn_cols = [c for c, (method, _) in self._fill_values.items() if method == "knn"]
        if knn_cols:
            try:
                from sklearn.impute import KNNImputer
                numeric_knn = [c for c in knn_cols
                               if pd.api.types.is_numeric_dtype(df[c])]
                if numeric_knn:
                    self._knn_imputer = KNNImputer(n_neighbors=5)
                    self._knn_imputer.fit(df[numeric_knn])
                    self._knn_cols = numeric_knn
                else:
                    # Fallback to median for non-numeric
                    for col in knn_cols:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            mode_val = df[col].mode()
                            self._fill_values[col] = ("mode", mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN")
            except ImportError:
                # Fallback to median
                for col in knn_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        self._fill_values[col] = ("median", float(df[col].median()))

        return self

    def _choose_strategy(self, series: pd.Series, pct_missing: float) -> str:
        if self.strategy != "auto":
            # For non-auto, check if applying median/mean makes sense
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if self.strategy in ("median", "mean") and not is_numeric:
                return "mode"  # fallback for non-numeric
            return self.strategy

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if pct_missing < 0.05:
            return "auto_numeric_low" if is_numeric else "auto_cat_low"
        elif pct_missing <= 0.20:
            return "knn" if is_numeric else "auto_cat_low"
        else:
            return "median" if is_numeric else "mode"

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Returns (cleaned_df, list_of_changes)."""
        df = df.copy()
        changes = []

        # Drop high-missing columns
        if self._dropped_cols:
            df = df.drop(columns=[c for c in self._dropped_cols if c in df.columns])
            for col in self._dropped_cols:
                changes.append({"column": col, "action": "drop_column",
                                "detail": f"Dropped: >{self.null_threshold*100:.0f}% missing"})

        # Apply KNN
        if self._knn_imputer is not None:
            knn_cols = [c for c in getattr(self, "_knn_cols", []) if c in df.columns]
            if knn_cols:
                before_null = df[knn_cols].isnull().sum().sum()
                df[knn_cols] = self._knn_imputer.transform(df[knn_cols])
                for col in knn_cols:
                    changes.append({"column": col, "action": "impute",
                                    "detail": "KNN imputation (5 neighbors)", "n_affected": 0})

        # Apply fill values
        for col, (method, value) in self._fill_values.items():
            if col not in df.columns or method == "knn":
                continue
            n_null = int(df[col].isnull().sum())
            if n_null == 0:
                continue
            df[col] = df[col].fillna(value)
            changes.append({
                "column": col, "action": "impute",
                "detail": f"{method} imputation (value={value!r})",
                "n_affected": n_null,
            })

        return df, changes


# ---------------------------------------------------------------------------
# Outlier Handling
# ---------------------------------------------------------------------------

class OutlierStrategy:
    """IQR, Z-score, or Isolation Forest outlier detection and handling."""

    def __init__(self, strategy: str = "auto", action: str = "clip"):
        self.strategy = strategy
        self.action = action
        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._iso_forest = None

    def fit(self, df: pd.DataFrame, protect_cols: Optional[List[str]] = None) -> "OutlierStrategy":
        if self.strategy == "none":
            return self
        protect_cols = protect_cols or []
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])  # skip boolean columns
            and c not in protect_cols
        ]

        if self.strategy in ("auto", "iqr"):
            for col in numeric_cols:
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                self._bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        elif self.strategy == "zscore":
            for col in numeric_cols:
                mu = float(df[col].mean())
                sigma = float(df[col].std())
                if sigma > 0:
                    self._bounds[col] = (mu - 3 * sigma, mu + 3 * sigma)

        elif self.strategy == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                clean = df[numeric_cols].dropna()
                self._iso_forest = IsolationForest(contamination=0.05, random_state=42)
                self._iso_forest.fit(clean)
                self._iso_cols = numeric_cols
                # Also compute IQR bounds for clipping
                for col in numeric_cols:
                    q1 = float(df[col].quantile(0.25))
                    q3 = float(df[col].quantile(0.75))
                    iqr = q3 - q1
                    self._bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            except ImportError:
                # Fallback to IQR
                for col in numeric_cols:
                    q1 = float(df[col].quantile(0.25))
                    q3 = float(df[col].quantile(0.75))
                    iqr = q3 - q1
                    self._bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], int]:
        if self.strategy == "none" or not self._bounds:
            return df, [], 0

        df = df.copy()
        changes = []
        total_handled = 0

        for col, (lo, hi) in self._bounds.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            mask = (df[col] < lo) | (df[col] > hi)
            n_out = int(mask.sum())
            if n_out == 0:
                continue

            if self.action == "clip":
                df[col] = df[col].clip(lower=lo, upper=hi)
                changes.append({"column": col, "action": "clip_outliers",
                                "detail": f"Clipped {n_out} outliers to [{lo:.3g}, {hi:.3g}]",
                                "n_affected": n_out})
            elif self.action == "remove":
                df = df[~mask]
                changes.append({"column": col, "action": "remove_outlier_rows",
                                "detail": f"Removed {n_out} outlier rows",
                                "n_affected": n_out})
            elif self.action == "flag":
                df[f"{col}_is_outlier"] = mask.astype(int)
                changes.append({"column": col, "action": "flag_outliers",
                                "detail": f"Added '{col}_is_outlier' flag column",
                                "n_affected": n_out})

            total_handled += n_out

        return df, changes, total_handled


# ---------------------------------------------------------------------------
# Dtype Fixer
# ---------------------------------------------------------------------------

class DtypeFixer:
    """Detects and fixes common dtype problems."""

    def fit_transform(self, df: pd.DataFrame,
                      protect_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        protect_cols = protect_cols or []
        df = df.copy()
        changes = []

        for col in df.columns:
            if col in protect_cols:
                continue
            series = df[col]
            original_dtype = str(series.dtype)

            # object → numeric attempt
            if series.dtype == object:
                converted = pd.to_numeric(series, errors="coerce")
                non_null_orig = series.notna().sum()
                non_null_conv = converted.notna().sum()
                # Accept if we didn't lose too many values
                if non_null_orig > 0 and non_null_conv / non_null_orig >= 0.95:
                    df[col] = converted
                    changes.append({
                        "column": col, "action": "cast_dtype",
                        "detail": f"object → float64 (numeric conversion)",
                        "n_affected": int(series.notna().sum()),
                    })
                    continue

            # integer 0/1 columns that look like booleans
            if pd.api.types.is_integer_dtype(series):
                unique_vals = set(series.dropna().unique())
                if unique_vals <= {0, 1}:
                    df[col] = series.astype(bool)
                    changes.append({
                        "column": col, "action": "cast_dtype",
                        "detail": "int → bool (binary column)",
                        "n_affected": int(series.notna().sum()),
                    })
                    continue

            # object date columns
            if series.dtype == object and col.lower() in (
                "date", "datetime", "timestamp", "created_at", "updated_at"
            ):
                try:
                    df[col] = pd.to_datetime(series, errors="coerce")
                    changes.append({
                        "column": col, "action": "cast_dtype",
                        "detail": "object → datetime64",
                        "n_affected": int(series.notna().sum()),
                    })
                except Exception:
                    pass

        return df, changes


# ---------------------------------------------------------------------------
# Category Unifier
# ---------------------------------------------------------------------------

class CategoryUnifier:
    """
    Unify inconsistent categorical values:
    'Yes', 'YES', 'yes', 'Y', 'y' → 'Yes'
    'No', 'NO', 'no', 'N', 'n' → 'No'
    Strips whitespace, normalizes case within groups.
    """

    _BOOL_MAP = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "t": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No", "f": "No",
    }

    def fit_transform(
        self,
        df: pd.DataFrame,
        custom_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        protect_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        protect_cols = protect_cols or []
        custom_mapping = custom_mapping or {}
        df = df.copy()
        changes = []

        for col in df.columns:
            if col in protect_cols or df[col].dtype not in (object, "category"):
                continue

            series = df[col]
            # Strip whitespace
            cleaned = series.str.strip() if hasattr(series, "str") else series
            n_changed = int((cleaned != series).sum())

            if col in custom_mapping:
                df[col] = cleaned.map(custom_mapping[col]).fillna(cleaned)
                changes.append({
                    "column": col, "action": "unify_categories",
                    "detail": f"Applied custom mapping ({len(custom_mapping[col])} values)",
                })
            else:
                # Apply bool unification
                unique_lower = set(cleaned.dropna().str.lower())
                if unique_lower.issubset(set(self._BOOL_MAP.keys())):
                    df[col] = cleaned.str.lower().map(self._BOOL_MAP).fillna(cleaned)
                    changes.append({
                        "column": col, "action": "unify_categories",
                        "detail": "Unified Yes/No variants",
                    })
                elif n_changed > 0:
                    df[col] = cleaned
                    changes.append({
                        "column": col, "action": "strip_whitespace",
                        "detail": f"Stripped whitespace ({n_changed} cells)",
                        "n_affected": n_changed,
                    })

        return df, changes

# ---------------------------------------------------------------------------
# Leakage Guard
# ---------------------------------------------------------------------------

class LeakageGuard:
    """
    Detects and removes variables that are perfect or near-perfect proxies 
    for the target (Data Leakage).
    
    Calculates correlation between all features and the target.
    Drops any feature with abs(correlation) > 0.98.
    """

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self._leaky_cols: List[str] = []

    def fit(self, df: pd.DataFrame, target: str, protect_cols: Optional[List[str]] = None) -> "LeakageGuard":
        if not target or target not in df.columns:
            return self
        
        protect_cols = protect_cols or []
        feat_cols = [c for c in df.columns if c != target and c not in protect_cols]
        
        # Prepare data for correlation (numeric only)
        # We temporarily encode categoricals to catch leakage in strings (like 'alive')
        temp_df = df[[target] + feat_cols].copy()
        for col in temp_df.columns:
            if not pd.api.types.is_numeric_dtype(temp_df[col]):
                temp_df[col] = temp_df[col].astype('category').cat.codes
        
        # Calculate correlation matrix
        corr_matrix = temp_df.corr().abs()
        target_corr = corr_matrix[target].drop(labels=[target])
        
        self._leaky_cols = target_corr[target_corr > self.threshold].index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        if not self._leaky_cols:
            return df, []
        
        df = df.copy()
        changes = []
        for col in self._leaky_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                changes.append({
                    "column": col, "action": "drop_leaky_column",
                    "detail": "Data Leakage detected (Near-perfect correlation to target)",
                    "n_affected": len(df)
                })
        
        return df, changes
