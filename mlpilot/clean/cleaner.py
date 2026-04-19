"""
mlpilot/clean/cleaner.py
AutoCleaner — the intelligent data cleaning engine.
One function call handles nulls, outliers, dtypes, duplicates, and category issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from mlpilot.clean.diff import CleaningReport
from mlpilot.clean.strategies import CategoryUnifier, DtypeFixer, NullStrategy, OutlierStrategy, LeakageGuard
from mlpilot.utils.display import print_step, print_success, print_warning
from mlpilot.utils.report_base import HTMLReportBuilder
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# Custom Rule
# ---------------------------------------------------------------------------

@dataclass
class CleaningRule:
    """
    A user-defined cleaning rule applied after auto-cleaning.

    Parameters
    ----------
    column : str
        Name of the column to apply the rule to.
    strategy : str
        'clip', 'impute', 'unify', 'validate', 'drop'
    min, max : float
        For 'clip' strategy — clamp values to [min, max]
    method : str
        For 'impute' — 'median', 'mean', 'mode', 'knn', 'value'
    value : Any
        For 'impute' with method='value' — the fill value
    n_neighbors : int
        For 'impute' with method='knn'
    mapping : dict
        For 'unify' — {bad_value: good_value}
    pattern : str
        For 'validate' — regex; rows that don't match are flagged
    """
    column: str
    strategy: str
    min: Optional[float] = None
    max: Optional[float] = None
    method: Optional[str] = None
    value: Any = None
    n_neighbors: int = 5
    mapping: Optional[Dict[str, str]] = None
    pattern: Optional[str] = None

    def apply(self, df: pd.DataFrame) -> tuple:
        """Apply this rule to df. Returns (modified_df, change_dict)."""
        if self.column not in df.columns:
            print_warning(f"CleaningRule: column '{self.column}' not found — skipping")
            return df, {}

        df = df.copy()
        change = {"column": self.column, "action": f"custom_{self.strategy}",
                  "detail": "", "n_affected": 0}

        if self.strategy == "clip":
            lo = self.min if self.min is not None else df[self.column].min()
            hi = self.max if self.max is not None else df[self.column].max()
            before = df[self.column].copy()
            df[self.column] = df[self.column].clip(lower=lo, upper=hi)
            change["n_affected"] = int((df[self.column] != before).sum())
            change["detail"] = f"Clipped to [{lo}, {hi}]"

        elif self.strategy == "impute":
            n_null = int(df[self.column].isnull().sum())
            if n_null > 0:
                method = self.method or "median"
                if method == "median":
                    fill = df[self.column].median()
                elif method == "mean":
                    fill = df[self.column].mean()
                elif method == "mode":
                    fill = df[self.column].mode().iloc[0]
                elif method == "value":
                    fill = self.value
                elif method == "knn":
                    try:
                        from sklearn.impute import KNNImputer
                        imp = KNNImputer(n_neighbors=self.n_neighbors)
                        df[[self.column]] = imp.fit_transform(df[[self.column]])
                        change["detail"] = f"KNN imputation ({self.n_neighbors} neighbors)"
                        change["n_affected"] = n_null
                        return df, change
                    except Exception:
                        fill = df[self.column].median()
                else:
                    fill = df[self.column].median()
                df[self.column] = df[self.column].fillna(fill)
                change["n_affected"] = n_null
                change["detail"] = f"Custom impute: {method} = {fill!r}"

        elif self.strategy == "unify":
            if self.mapping:
                df[self.column] = df[self.column].map(self.mapping).fillna(df[self.column])
                change["detail"] = f"Custom mapping ({len(self.mapping)} values)"
                change["n_affected"] = len(self.mapping)

        elif self.strategy == "drop":
            df = df.drop(columns=[self.column])
            change["detail"] = "Column dropped by custom rule"

        elif self.strategy == "validate":
            if self.pattern:
                import re
                mask = df[self.column].astype(str).str.match(self.pattern, na=False)
                n_invalid = int((~mask).sum())
                change["detail"] = f"Validation: {n_invalid} rows failed pattern '{self.pattern}'"
                change["n_affected"] = n_invalid

        return df, change


# ---------------------------------------------------------------------------
# CleaningResult
# ---------------------------------------------------------------------------

class CleaningResult(BaseResult):
    """
    Complete result from ml.clean().

    Attributes
    ----------
    df : pd.DataFrame
        The cleaned dataframe.
    original_df : pd.DataFrame
        The original (unchanged) dataframe.
    report : CleaningReport
        Full diff report of every change made.
    n_nulls_filled : int
    n_outliers_handled : int
    n_duplicates_removed : int
    n_dtypes_fixed : int
    quality_before : float
    quality_after : float
    """

    def __init__(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame,
        report: CleaningReport,
        quality_before: float = 0.0,
        quality_after: float = 0.0,
    ):
        self.df = df
        self.original_df = original_df
        self.report = report
        self.n_nulls_filled = report.n_nulls_filled
        self.n_outliers_handled = report.n_outliers_handled
        self.n_duplicates_removed = report.n_duplicates_removed
        self.n_dtypes_fixed = report.n_dtypes_fixed
        self.quality_before = quality_before
        self.quality_after = quality_after

    def undo(self) -> pd.DataFrame:
        """Return the original uncleaned dataframe."""
        return self.original_df.copy()

    def to_html(self, path: Optional[str] = None) -> str:
        builder = HTMLReportBuilder(
            title="Data Cleaning Report",
            subtitle=f"{self.report.original_shape[0]:,} rows × {self.report.original_shape[1]} columns",
        )
        # Summary stats
        stats = [
            ("Nulls Filled", self.n_nulls_filled, ""),
            ("Outliers Handled", self.n_outliers_handled, ""),
            ("Dtypes Fixed", self.n_dtypes_fixed, ""),
            ("Duplicates Removed", self.n_duplicates_removed, ""),
            ("Quality Before", f"{self.quality_before:.0f}/100", ""),
            ("Quality After", f"{self.quality_after:.0f}/100", ""),
        ]
        content = builder.stat_grid(stats, cols=3)
        content += builder.score_bar(self.quality_after)

        # Changes table
        if self.report.changes:
            rows = [[c.column, c.action, c.detail, str(c.n_affected or "—")]
                    for c in self.report.changes]
            content += builder.table(["Column", "Action", "Detail", "Cells Affected"], rows)
        else:
            content += '<p style="color:var(--success)">✓ No changes needed</p>'

        builder.add_section(builder.section("🧹 Cleaning Report", content))

        # Reproducible code
        code = self.code
        if code:
            builder.add_section(builder.section("💻 Reproducible Code",
                                               builder.code_block(code)))

        out_path = path or "mlpilot_cleaning_report.html"
        return builder.save(out_path)

    @property
    def code(self) -> str:
        """Python code that reproduces the cleaning."""
        lines = [
            "import mlpilot as ml\n",
            "# AutoCleaner — reproduce these changes",
            "result = ml.clean(",
            "    df,",
        ]
        if self.n_nulls_filled > 0:
            lines.append("    null_strategy='auto',")
        if self.n_outliers_handled > 0:
            lines.append("    outlier_strategy='auto',")
        if self.n_dtypes_fixed > 0:
            lines.append("    fix_dtypes=True,")
        if self.n_duplicates_removed > 0:
            lines.append("    remove_duplicates=True,")
        lines.append(")")
        lines.append("df_clean = result.df")
        return "\n".join(lines)

    @property
    def pipeline_step(self):
        """Return an sklearn-compatible transformer for this cleaning."""
        # Lazy import to avoid circular dependency
        from mlpilot.features.encoders import _SklearnDFTransformer
        return _SklearnDFTransformer(self)

    def __repr__(self) -> str:
        return (
            f"CleaningResult(rows={len(self.df):,}, "
            f"nulls_filled={self.n_nulls_filled}, "
            f"quality={self.quality_before:.0f}→{self.quality_after:.0f})"
        )


# ---------------------------------------------------------------------------
# AutoCleaner Engine
# ---------------------------------------------------------------------------

class AutoCleaner:
    """
    Internal cleaning engine. Use the top-level clean() function instead.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        null_strategy: str = "auto",
        null_threshold: float = 0.5,
        outlier_strategy: str = "auto",
        outlier_action: str = "clip",
        fix_dtypes: bool = True,
        unify_categories: bool = True,
        remove_duplicates: bool = True,
        custom_rules: Optional[List[CleaningRule]] = None,
        protect_cols: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.target = target
        self.null_strategy = null_strategy
        self.null_threshold = null_threshold
        self.outlier_strategy = outlier_strategy
        self.outlier_action = outlier_action
        self.fix_dtypes = fix_dtypes
        self.unify_categories_flag = unify_categories
        self.remove_duplicates = remove_duplicates
        self.custom_rules = custom_rules or []
        self.protect_cols = list(protect_cols or [])
        self.verbose = verbose

        if target and target not in self.protect_cols:
            self.protect_cols.append(target)

    def clean(self, df: pd.DataFrame) -> CleaningResult:
        original_df = df.copy()
        all_changes: List[Dict] = []
        quality_before = self._quality_score(df)

        if self.verbose:
            print_step("Starting AutoCleaner...", "* ")

        # 1. Fix dtypes first (makes null/outlier detection more accurate)
        if self.fix_dtypes:
            fixer = DtypeFixer()
            df, dtype_changes = fixer.fit_transform(df, protect_cols=self.protect_cols)
            all_changes.extend(dtype_changes)
            if self.verbose and dtype_changes:
                print_step(f"Fixed {len(dtype_changes)} dtype issues", "> ")

        # 2. Remove duplicates
        if self.remove_duplicates:
            try:
                # Use astype(object) to avoid boolean subtraction issues in latest pandas/numpy
                n_dups = int(df.astype(object).duplicated().sum())
                if n_dups > 0:
                    df = df.drop_duplicates().reset_index(drop=True)
                    all_changes.append({
                        "column": "—", "action": "remove_duplicates",
                        "detail": f"Removed {n_dups} duplicate rows",
                        "n_affected": n_dups,
                    })
                    if self.verbose:
                        print_step(f"Removed {n_dups} duplicate rows", "x ")
            except Exception as e:
                if self.verbose:
                    print_warning(f"Could not remove duplicates: {e}")

        # 3. Unify categories
        if self.unify_categories_flag:
            unifier = CategoryUnifier()
            df, unify_changes = unifier.fit_transform(
                df, protect_cols=self.protect_cols
            )
            all_changes.extend(unify_changes)
            if self.verbose and unify_changes:
                print_step(f"Unified {len(unify_changes)} categorical inconsistencies", "- ")

        # 4. Handle nulls
        null_strat = NullStrategy(
            strategy=self.null_strategy,
            null_threshold=self.null_threshold
        )
        null_strat.fit(df, protect_cols=self.protect_cols)
        df, null_changes = null_strat.transform(df)
        all_changes.extend(null_changes)
        n_filled = sum(c.get("n_affected", 0) for c in null_changes if c.get("action") == "impute")
        if self.verbose and null_changes:
            print_step(f"Filled {n_filled} null values", "+ ")

        # 5. Handle outliers
        outlier_strat = OutlierStrategy(
            strategy=self.outlier_strategy,
            action=self.outlier_action
        )
        outlier_strat.fit(df, protect_cols=self.protect_cols)
        df, outlier_changes, n_out = outlier_strat.transform(df)
        all_changes.extend(outlier_changes)
        if self.verbose and outlier_changes:
            print_step(f"Handled {n_out} outliers ({self.outlier_action})", "! ")

        # 6. Leakage Guard (Anti-Cheating Check)
        if self.target:
            guard = LeakageGuard()
            guard.fit(df, target=self.target, protect_cols=self.protect_cols)
            df, leakage_changes = guard.transform(df)
            all_changes.extend(leakage_changes)
            if self.verbose and leakage_changes:
                print_warning(f"Dropped {len(leakage_changes)} leaky proxy columns")

        # 7. Custom rules
        for rule in self.custom_rules:
            df, change = rule.apply(df)
            if change:
                all_changes.append(change)

        quality_after = self._quality_score(df)
        report = CleaningReport(all_changes, original_df, df)

        if self.verbose:
            improvement = quality_after - quality_before
            sign = "+" if improvement >= 0 else ""
            print_success(
                f"Cleaning complete — quality: {quality_before:.0f} → {quality_after:.0f} "
                f"({sign}{improvement:.0f} pts)"
            )

        return CleaningResult(
            df=df,
            original_df=original_df,
            report=report,
            quality_before=quality_before,
            quality_after=quality_after,
        )

    def _quality_score(self, df: pd.DataFrame) -> float:
        score = 100.0
        if df.empty:
            return 0.0
        n_cells = df.size
        n_missing = int(df.isnull().sum().sum())
        score -= min(40, n_missing / n_cells * 100 * 2)
        try:
            n_dups = int(df.astype(object).duplicated().sum())
        except Exception:
            n_dups = 0
        dup_pct = n_dups / max(len(df), 1) * 100
        score -= min(20, dup_pct)
        return max(0.0, min(100.0, round(score, 1)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    target: Optional[str] = None,
    null_strategy: str = "auto",
    null_threshold: float = 0.5,
    outlier_strategy: str = "auto",
    outlier_action: str = "clip",
    fix_dtypes: bool = True,
    unify_categories: bool = True,
    remove_duplicates: bool = True,
    custom_rules: Optional[List[CleaningRule]] = None,
    protect_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> CleaningResult:
    """
    AutoCleaner — Intelligent Data Cleaning.

    Replaces 10–15 hours of manual data cleaning per project.
    Handles nulls, outliers, dtype errors, duplicates, and category unification
    — all automatically with sensible defaults.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to clean.
    target : str, optional
        Target column — will never be modified.
    null_strategy : str
        'auto' | 'median' | 'mean' | 'mode' | 'knn' | 'drop_rows' | 'drop_cols'
    null_threshold : float
        Drop column if pct_missing > this (0–1). Default 0.5.
    outlier_strategy : str
        'auto' | 'iqr' | 'zscore' | 'isolation_forest' | 'none'
    outlier_action : str
        'clip' | 'remove' | 'flag' | 'none'
    fix_dtypes : bool
        Auto-fix object→numeric, int→bool, etc.
    unify_categories : bool
        Unify 'yes'/'Yes'/'YES'→'Yes' etc.
    remove_duplicates : bool
        Drop exact duplicate rows.
    custom_rules : list[CleaningRule], optional
        User-defined rules applied after auto-cleaning.
    protect_cols : list[str], optional
        Additional columns to never modify.
    verbose : bool
        Print progress to terminal.

    Returns
    -------
    CleaningResult
        Attributes: df, original_df, report, n_nulls_filled,
        n_outliers_handled, n_dtypes_fixed, n_duplicates_removed,
        quality_before, quality_after.
        Methods: undo(), to_html(), to_pdf(). Property: code, pipeline_step.

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.clean(df, target='churn')
    >>> df_clean = result.df
    >>> result.report.print()
    >>> df_original = result.undo()
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to clean")

    cleaner = AutoCleaner(
        target=target,
        null_strategy=null_strategy,
        null_threshold=null_threshold,
        outlier_strategy=outlier_strategy,
        outlier_action=outlier_action,
        fix_dtypes=fix_dtypes,
        unify_categories=unify_categories,
        remove_duplicates=remove_duplicates,
        custom_rules=custom_rules,
        protect_cols=protect_cols,
        verbose=verbose,
    )
    return cleaner.clean(df)
