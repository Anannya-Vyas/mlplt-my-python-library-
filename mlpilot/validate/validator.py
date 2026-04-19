"""
mlpilot/validate/validator.py
DataValidator — schema & quality validation, drift detection, leakage checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from mlpilot.utils.display import RichTable, print_step, print_success, print_warning
from mlpilot.utils.types import BaseResult, ValidationIssue
from mlpilot.validate.checks import (
    check_constant_columns, check_data_leakage, check_distribution_drift,
    check_dtypes, check_duplicates, check_id_columns, check_missing,
    check_near_constant, check_time_ordering, check_unexpected_columns,
)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class ValidationResult(BaseResult):
    """
    Complete result from ml.validate().

    Attributes
    ----------
    passed : bool
        True if no critical issues found.
    issues : list[ValidationIssue]
        All issues (warnings + critical).
    critical : list[ValidationIssue]
        Issues that will likely break training.
    warnings : list[ValidationIssue]
        Issues that may degrade performance.
    """

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = sorted(
            issues, key=lambda x: {"critical": 0, "warning": 1, "info": 2}.get(x.severity, 2)
        )
        self.critical = [i for i in self.issues if i.severity == "critical"]
        self.warnings = [i for i in self.issues if i.severity == "warning"]
        self.passed = len(self.critical) == 0

    def print_report(self) -> None:
        """Print a rich formatted validation report."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        style = "bold green" if self.passed else "bold red"

        try:
            from rich.console import Console
            Console().print(f"\n  Validation: {status}", style=style)
        except ImportError:
            print(f"\n  Validation: {status}")

        if self.critical:
            tbl = RichTable(title="Critical Issues", columns=["Check", "Column", "Message"])
            for issue in self.critical:
                tbl.add_row(issue.check, issue.column or "—", issue.message)
            tbl.print()

        if self.warnings:
            tbl = RichTable(title="Warnings", columns=["Check", "Column", "Message"])
            for issue in self.warnings:
                tbl.add_row(issue.check, issue.column or "—", issue.message)
            tbl.print()

        if not self.issues:
            print_success("All checks passed — no issues found")

    def raise_if_critical(self) -> None:
        """Raise ValueError if any critical issues exist."""
        if self.critical:
            msg = "\n".join(str(i) for i in self.critical)
            raise ValueError(f"Critical validation failures:\n{msg}")

    def __repr__(self) -> str:
        return (f"ValidationResult(passed={self.passed}, "
                f"critical={len(self.critical)}, warnings={len(self.warnings)})")


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Learn the schema from a training dataframe.
    Returns a schema dict that can be passed to validate().

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset.

    Returns
    -------
    dict with keys 'columns', 'dtypes', 'stats'
    """
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "stats": {},
    }
    for col in df.columns:
        s = df[col]
        schema["stats"][col] = {
            "n_unique": int(s.nunique()),
            "pct_missing": float(s.isnull().mean()),
        }
        if pd.api.types.is_numeric_dtype(s):
            schema["stats"][col].update({
                "min": float(s.min()) if not s.isna().all() else None,
                "max": float(s.max()) if not s.isna().all() else None,
                "mean": float(s.mean()) if not s.isna().all() else None,
            })
    return schema


# ---------------------------------------------------------------------------
# DataValidator Engine
# ---------------------------------------------------------------------------

class DataValidator:
    """Internal validator engine. Use ml.validate() instead."""

    def __init__(
        self,
        df_test: Optional[pd.DataFrame] = None,
        schema: Optional[Dict] = None,
        target: Optional[str] = None,
        date_col: Optional[str] = None,
        drift_alpha: float = 0.05,
        missing_threshold: float = 0.3,
        near_constant_threshold: float = 0.99,
        verbose: bool = True,
    ):
        self.df_test = df_test
        self.schema = schema
        self.target = target
        self.date_col = date_col
        self.drift_alpha = drift_alpha
        self.missing_threshold = missing_threshold
        self.near_constant_threshold = near_constant_threshold
        self.verbose = verbose

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        issues: List[ValidationIssue] = []

        if self.verbose:
            print_step("Running DataValidator checks...", "🔍")

        # Basic checks
        issues += check_constant_columns(df)
        issues += check_near_constant(df, self.near_constant_threshold)
        issues += check_missing(df, self.missing_threshold)
        issues += check_duplicates(df)
        issues += check_id_columns(df)
        issues += check_data_leakage(df, self.target)
        issues += check_time_ordering(df, self.date_col)

        # Schema checks
        if self.schema:
            issues += check_dtypes(df, self.schema)
            issues += check_unexpected_columns(df, self.schema)

        # Drift checks (train vs test)
        if self.df_test is not None:
            issues += check_distribution_drift(df, self.df_test, self.drift_alpha)

        result = ValidationResult(issues)

        if self.verbose:
            n_crit = len(result.critical)
            n_warn = len(result.warnings)
            if result.passed:
                print_success(f"Validation passed — {n_warn} warnings")
            else:
                print_warning(f"Validation failed — {n_crit} critical, {n_warn} warnings")

        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    df: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    schema: Optional[Dict] = None,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    drift_alpha: float = 0.05,
    missing_threshold: float = 0.3,
    near_constant_threshold: float = 0.99,
    verbose: bool = True,
) -> ValidationResult:
    """
    DataValidator — Schema & Quality Validation.

    Catches data quality issues that silently destroy model performance:
    leakage, distribution drift, schema violations, constant columns, and more.

    Parameters
    ----------
    df : pd.DataFrame
        Primary (training) dataset to validate.
    df_test : pd.DataFrame, optional
        Test dataset — triggers distribution drift checks.
    schema : dict, optional
        Schema from ml.infer_schema(df_train) to enforce on new data.
    target : str, optional
        Target column — used for leakage detection.
    date_col : str, optional
        Date column to check for monotonic ordering.
    drift_alpha : float
        Significance level for drift detection (default 0.05).
    missing_threshold : float
        Flag columns with more missing data than this (0–1).
    verbose : bool
        Print progress to terminal.

    Returns
    -------
    ValidationResult
        Attributes: passed, issues, critical, warnings.
        Methods: print_report(), raise_if_critical().

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.validate(df_train, df_test=df_test, target='churn')
    >>> result.print_report()
    >>> result.raise_if_critical()  # Raises ValueError if any critical issues

    >>> # Schema enforcement
    >>> schema = ml.infer_schema(df_train)
    >>> result = ml.validate(df_new, schema=schema)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    # Handle common user error: passing schema as second argument (documented as df_test)
    if isinstance(df_test, dict) and schema is None:
        schema = df_test
        df_test = None

    validator = DataValidator(
        df_test=df_test,
        schema=schema,
        target=target,
        date_col=date_col,
        drift_alpha=drift_alpha,
        missing_threshold=missing_threshold,
        near_constant_threshold=near_constant_threshold,
        verbose=verbose,
    )
    return validator.validate(df)
