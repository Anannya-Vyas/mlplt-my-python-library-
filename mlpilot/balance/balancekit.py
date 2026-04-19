"""
mlpilot/balance/balancekit.py
BalanceKit — intelligent imbalanced data handling.
Auto-selects SMOTE, ADASYN, class weighting, or undersampling based on severity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from mlpilot.utils.display import print_step, print_success, print_warning, RichTable
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# BalanceResult
# ---------------------------------------------------------------------------

class BalanceResult(BaseResult):
    def __init__(
        self,
        X_resampled: pd.DataFrame,
        y_resampled: pd.Series,
        X_original: pd.DataFrame,
        y_original: pd.Series,
        strategy_used: str,
        class_weights: Dict[Any, float],
        severity: str,
        imbalance_ratio: float,
    ):
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        self.X_original = X_original
        self.y_original = y_original
        self.strategy_used = strategy_used
        self.class_weights = class_weights
        self.severity = severity
        self.imbalance_ratio = imbalance_ratio

    def print_report(self) -> None:
        tbl = RichTable(title="⚖️ BalanceKit Report", columns=["Metric", "Before", "After"])
        before_counts = self.y_original.value_counts()
        after_counts = self.y_resampled.value_counts()
        for cls in sorted(before_counts.index.tolist(), key=str):
            tbl.add_row(
                f"Class {cls}",
                str(before_counts.get(cls, 0)),
                str(after_counts.get(cls, 0)),
            )
        tbl.print()
        print_success(
            f"Strategy: {self.strategy_used} | Severity: {self.severity} | "
            f"Ratio: {self.imbalance_ratio:.1f}:1"
        )

    def __repr__(self) -> str:
        return (f"BalanceResult(strategy={self.strategy_used}, "
                f"severity={self.severity}, "
                f"samples_before={len(self.y_original)}, "
                f"samples_after={len(self.y_resampled)})")


# ---------------------------------------------------------------------------
# BalanceKit Engine
# ---------------------------------------------------------------------------

class BalanceKit:
    """Internal engine. Use ml.balance() instead."""

    def __init__(
        self,
        strategy: str = "auto",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.strategy = strategy
        self.random_state = random_state
        self.verbose = verbose

    def balance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> BalanceResult:
        if self.verbose:
            print_step("BalanceKit: analyzing class distribution...", "⚖️")

        vc = y.value_counts()
        if len(vc) < 2:
            raise ValueError("Need at least 2 classes to balance")

        majority = vc.iloc[0]
        minority = vc.iloc[-1]
        ratio = majority / max(minority, 1)

        # Determine severity
        if ratio < 2:
            severity = "none"
        elif ratio < 5:
            severity = "mild"
        elif ratio < 20:
            severity = "moderate"
        elif ratio < 100:
            severity = "severe"
        else:
            severity = "critical"

        # Choose strategy
        if self.strategy == "auto":
            chosen_strategy = self._auto_strategy(severity)
        else:
            chosen_strategy = self.strategy

        if self.verbose:
            print_step(
                f"Severity: {severity} ({ratio:.1f}:1) → strategy: {chosen_strategy}", "📊"
            )

        # Compute class weights always (useful even if not resampling)
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array(sorted(y.unique().tolist(), key=str))
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weights = dict(zip(classes.tolist(), weights.tolist()))

        X_res, y_res = self._apply_strategy(X, y, chosen_strategy, severity)

        if self.verbose:
            print_success(
                f"Balanced: {len(y)} → {len(y_res)} samples using {chosen_strategy}"
            )

        return BalanceResult(
            X_resampled=X_res,
            y_resampled=y_res,
            X_original=X.copy(),
            y_original=y.copy(),
            strategy_used=chosen_strategy,
            class_weights=class_weights,
            severity=severity,
            imbalance_ratio=ratio,
        )

    def _auto_strategy(self, severity: str) -> str:
        strategies = {
            "none": "class_weight",
            "mild": "class_weight",
            "moderate": "smote",
            "severe": "adasyn",
            "critical": "adasyn_undersample",
        }
        return strategies[severity]

    def _apply_strategy(
        self, X: pd.DataFrame, y: pd.Series, strategy: str, severity: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if strategy == "class_weight":
            # No resampling — return original (weights are in BalanceResult)
            return X.copy(), y.copy()

        elif strategy == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=self.random_state)
                X_res, y_res = sm.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                print_warning(
                    "imbalanced-learn not installed — install with: pip install mlpilot[imb]. "
                    "Falling back to random oversampling."
                )
                return self._random_oversample(X, y)

        elif strategy == "adasyn":
            try:
                from imblearn.over_sampling import ADASYN
                adasyn = ADASYN(random_state=self.random_state)
                X_res, y_res = adasyn.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                print_warning(
                    "imbalanced-learn not installed — falling back to SMOTE simulation."
                )
                return self._random_oversample(X, y)

        elif strategy in ("adasyn_undersample", "combined"):
            try:
                from imblearn.combine import SMOTETomek
                sm = SMOTETomek(random_state=self.random_state)
                X_res, y_res = sm.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                return self._random_oversample(X, y)

        elif strategy == "random_undersample":
            try:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=self.random_state)
                X_res, y_res = rus.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                return self._random_undersample(X, y)

        else:
            print_warning(f"Unknown strategy '{strategy}' — returning original data")
            return X.copy(), y.copy()

    def _random_oversample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Simple random oversampling fallback."""
        vc = y.value_counts()
        majority_count = vc.iloc[0]
        X_parts = [X]
        y_parts = [y]
        for cls in vc.index[1:]:
            mask = y == cls
            n_needed = majority_count - mask.sum()
            if n_needed > 0:
                oversampled = X[mask].sample(n_needed, replace=True, random_state=self.random_state)
                X_parts.append(oversampled)
                y_parts.append(pd.Series([cls] * n_needed))
        X_res = pd.concat(X_parts, ignore_index=True)
        y_res = pd.concat(y_parts, ignore_index=True)
        return X_res, y_res

    def _random_undersample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Simple random undersampling fallback."""
        vc = y.value_counts()
        minority_count = vc.iloc[-1]
        X_parts = []
        y_parts = []
        for cls in vc.index:
            mask = y == cls
            X_cls = X[mask].sample(min(minority_count, mask.sum()),
                                    random_state=self.random_state)
            X_parts.append(X_cls)
            y_parts.append(y[X_cls.index])
        X_res = pd.concat(X_parts, ignore_index=True)
        y_res = pd.concat(y_parts, ignore_index=True)
        return X_res, y_res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def balance(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray, str]] = None,
    strategy: str = "auto",
    random_state: int = 42,
    verbose: bool = True,
    target: Optional[str] = None,
) -> BalanceResult:
    """
    BalanceKit — Intelligent Imbalanced Data Handling.
    
    Supports both (X, y) and (df, target='col') signatures.
    """
    # 1. Handle (df, target='col') signature
    if isinstance(X, pd.DataFrame) and (isinstance(y, str) or target is not None):
        target_col = y if isinstance(y, str) else target
        if target_col in X.columns:
            y_final = X[target_col]
            X_final = X.drop(columns=[target_col])
        else:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    else:
        # 2. Handle standard (X, y) signature
        if y is None:
            raise ValueError("Must provide y (target series) or target column name")
        X_final = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_final = pd.Series(y) if not isinstance(y, pd.Series) else y

    kit = BalanceKit(strategy=strategy, random_state=random_state, verbose=verbose)
    return kit.balance(X_final, y_final)
