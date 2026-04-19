"""
mlpilot/validate/checks.py
Individual check functions for DataValidator.
Each function returns a list of ValidationIssue objects.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from mlpilot.utils.types import ValidationIssue


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

def check_dtypes(df: pd.DataFrame, schema: Optional[dict] = None) -> List[ValidationIssue]:
    issues = []
    if schema is None:
        return issues
    expected_dtypes = schema.get("dtypes", {})
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            issues.append(ValidationIssue(
                severity="critical", check="schema_missing_column",
                column=col, message=f"Expected column '{col}' is missing",
            ))
        elif not pd.api.types.is_dtype_equal(df[col].dtype, expected):
            issues.append(ValidationIssue(
                severity="warning", check="schema_dtype_mismatch",
                column=col,
                message=f"Expected dtype '{expected}', got '{df[col].dtype}'",
                statistic=str(df[col].dtype),
            ))
    return issues


def check_unexpected_columns(df: pd.DataFrame, schema: Optional[dict] = None) -> List[ValidationIssue]:
    issues = []
    if schema is None:
        return issues
    expected_cols = set(schema.get("columns", []))
    if not expected_cols:
        return issues
    for col in df.columns:
        if col not in expected_cols:
            issues.append(ValidationIssue(
                severity="warning", check="schema_unexpected_column",
                column=col, message=f"Unexpected column '{col}' — not in schema",
            ))
    return issues


# ---------------------------------------------------------------------------
# Constant and near-constant checks
# ---------------------------------------------------------------------------

def check_constant_columns(df: pd.DataFrame) -> List[ValidationIssue]:
    issues = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique == 0:
            issues.append(ValidationIssue(
                severity="warning", check="constant_col",
                column=col, message=f"Column '{col}' is entirely null",
            ))
        elif n_unique == 1:
            issues.append(ValidationIssue(
                severity="warning", check="constant_col",
                column=col, message=f"Column '{col}' has only one unique value — zero variance",
                statistic=df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None,
            ))
    return issues


def check_near_constant(df: pd.DataFrame, threshold: float = 0.99) -> List[ValidationIssue]:
    issues = []
    for col in df.columns:
        if df[col].empty:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=True).iloc[0] \
            if df[col].nunique(dropna=True) > 0 else 0
        if top_freq >= threshold:
            issues.append(ValidationIssue(
                severity="warning", check="near_constant",
                column=col,
                message=f"Column '{col}' is {top_freq*100:.1f}% a single value — likely useless",
                statistic=top_freq,
            ))
    return issues


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def check_data_leakage(df: pd.DataFrame, target: Optional[str] = None) -> List[ValidationIssue]:
    issues = []
    if target is None or target not in df.columns:
        return issues

    target_series = df[target]

    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target_series):
            try:
                # Near-perfect correlation is a leakage signal
                non_null = df[[col, target]].dropna()
                if len(non_null) > 10:
                    corr = abs(float(non_null[col].corr(non_null[target])))
                    if corr > 0.995:
                        issues.append(ValidationIssue(
                            severity="critical", check="leakage_correlation",
                            column=col,
                            message=f"Column '{col}' has near-perfect correlation ({corr:.4f}) "
                                    f"with target '{target}' — possible leakage",
                            statistic=corr,
                        ))
            except Exception:
                pass
    return issues


def check_id_columns(df: pd.DataFrame) -> List[ValidationIssue]:
    """Detect likely ID columns that should not be features."""
    issues = []
    n = len(df)
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / max(n, 1)
        # Very high cardinality + integer or string type suggests an ID
        if ratio > 0.9 and n_unique > 100:
            if pd.api.types.is_integer_dtype(df[col]) or df[col].dtype == object:
                issues.append(ValidationIssue(
                    severity="warning", check="id_column",
                    column=col,
                    message=f"Column '{col}' has {n_unique:,} unique values ({ratio*100:.0f}%) "
                            f"— likely an ID column that should be excluded from features",
                    statistic=ratio,
                ))
    return issues


# ---------------------------------------------------------------------------
# Duplicate checks
# ---------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame) -> List[ValidationIssue]:
    issues = []
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        issues.append(ValidationIssue(
            severity="warning", check="duplicates",
            column=None,
            message=f"{n_dups:,} exact duplicate rows detected ({n_dups/len(df)*100:.1f}%)",
            statistic=n_dups,
        ))
    return issues


# ---------------------------------------------------------------------------
# Distribution drift (train vs test)
# ---------------------------------------------------------------------------

def check_distribution_drift(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    alpha: float = 0.05,
) -> List[ValidationIssue]:
    """
    KS test for numeric columns, chi-squared for categorical.
    Flags any column where p < alpha.
    """
    issues = []
    common_cols = set(df_train.columns) & set(df_test.columns)

    for col in common_cols:
        col_train = df_train[col].dropna()
        col_test = df_test[col].dropna()
        if len(col_train) == 0 or len(col_test) == 0:
            continue

        if pd.api.types.is_numeric_dtype(df_train[col]):
            try:
                from scipy import stats
                stat, p = stats.ks_2samp(col_train.values, col_test.values)
                if p < alpha:
                    issues.append(ValidationIssue(
                        severity="warning", check="distribution_drift",
                        column=col,
                        message=f"Distribution drift detected in '{col}' "
                                f"(KS stat={stat:.3f}, p={p:.4f})",
                        statistic=p,
                    ))
            except ImportError:
                pass  # scipy not available
        else:
            try:
                from scipy.stats import chi2_contingency
                cats = set(col_train.unique()) | set(col_test.unique())
                train_counts = col_train.value_counts()
                test_counts = col_test.value_counts()
                contingency = pd.DataFrame({
                    "train": [train_counts.get(c, 0) for c in cats],
                    "test": [test_counts.get(c, 0) for c in cats],
                })
                if contingency.values.sum() > 0:
                    chi2, p, _, _ = chi2_contingency(contingency.values)
                    if p < alpha:
                        issues.append(ValidationIssue(
                            severity="warning", check="distribution_drift",
                            column=col,
                            message=f"Categorical drift detected in '{col}' "
                                    f"(chi2={chi2:.2f}, p={p:.4f})",
                            statistic=p,
                        ))
            except ImportError:
                pass

    return issues


# ---------------------------------------------------------------------------
# Missing value checks
# ---------------------------------------------------------------------------

def check_missing(df: pd.DataFrame, threshold: float = 0.3) -> List[ValidationIssue]:
    issues = []
    for col in df.columns:
        pct = df[col].isnull().mean()
        if pct > threshold:
            issues.append(ValidationIssue(
                severity="warning" if pct < 0.7 else "critical",
                check="high_missing",
                column=col,
                message=f"Column '{col}' has {pct*100:.1f}% missing values",
                statistic=pct,
            ))
    return issues


# ---------------------------------------------------------------------------
# Time series ordering
# ---------------------------------------------------------------------------

def check_time_ordering(df: pd.DataFrame, date_col: Optional[str] = None) -> List[ValidationIssue]:
    issues = []
    if date_col is None:
        # Try to find datetime column
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if not dt_cols:
            return issues
        date_col = dt_cols[0]

    if date_col not in df.columns:
        return issues

    dates = df[date_col].dropna()
    if not pd.api.types.is_datetime64_any_dtype(dates):
        return issues

    is_monotonic = dates.is_monotonic_increasing or dates.is_monotonic_decreasing
    if not is_monotonic:
        issues.append(ValidationIssue(
            severity="warning", check="time_ordering",
            column=date_col,
            message=f"Column '{date_col}' is not monotonically ordered — "
                    "this may cause issues with time-series splits",
        ))
    return issues
