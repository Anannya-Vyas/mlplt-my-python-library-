"""
mlpilot/eda/analyzer.py
SmartEDA — the flagship intelligent EDA module.
Produces a complete 12-section analysis in one function call.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlpilot.utils.types import (
    BaseResult, ColumnProfile, DataIssue, DatasetSummary, Recommendation
)
from mlpilot.utils.display import (
    RichTable, print_banner, print_step, print_success, print_warning, ProgressBar
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelationMatrix:
    matrix: pd.DataFrame
    method: str
    threshold: float
    high_pairs: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class TargetAnalysis:
    column: str
    task: str                            # 'binary', 'multiclass', 'regression'
    class_distribution: Dict[Any, int] = field(default_factory=dict)
    imbalance_score: float = 0.0
    imbalance_severity: str = "none"
    recommended_strategy: str = "none"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_correlated: List[Tuple[str, float]] = field(default_factory=list)


class EDAResult(BaseResult):
    """
    Complete result from ml.analyze().
    Contains every stat, plot, issue, and recommendation produced by SmartEDA.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        summary: DatasetSummary,
        column_profiles: Dict[str, ColumnProfile],
        correlations: Optional[CorrelationMatrix],
        target_analysis: Optional[TargetAnalysis],
        quality_score: float,
        issues: List[DataIssue],
        recommendations: List[Recommendation],
        plots: Dict[str, Any],
        report_path: Optional[str] = None,
    ):
        self.df = df
        self.summary = summary
        self.column_profiles = column_profiles
        self.correlations = correlations
        self.target_analysis = target_analysis
        self.quality_score = quality_score
        self.issues = issues
        self.recommendations = recommendations
        self.plots = plots
        self.report_path = report_path

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def to_html(self, path: Optional[str] = None) -> str:
        from mlpilot.eda.report import EDAReportBuilder
        builder = EDAReportBuilder(self).build_full_report()
        out_path = path or "mlpilot_eda_report.html"
        saved = builder.save(out_path)
        print_success(f"EDA report saved → {saved}")
        return saved

    def to_pdf(self, path: Optional[str] = None) -> None:
        print_warning("PDF export requires 'weasyprint'. Install it with: pip install weasyprint")

    def to_notebook(self, path: Optional[str] = None) -> None:
        print_warning("Notebook export coming in v0.2.0")

    def show(self) -> None:
        """Display the report inline in a Jupyter notebook."""
        try:
            from IPython.display import HTML, display
            from mlpilot.eda.report import EDAReportBuilder
            html = EDAReportBuilder(self).build_full_report().build()
            display(HTML(html))
        except ImportError:
            print_warning("Not in a Jupyter notebook. Use result.to_html() instead.")

    def print_summary(self) -> None:
        """Print a rich terminal summary table."""
        s = self.summary
        print_banner(f"EDA Summary — Quality Score: {self.quality_score:.0f}/100")

        # Dataset overview table
        overview = RichTable(title="Dataset Overview", columns=["Property", "Value"])
        overview.add_row("Shape", f"{s.n_rows:,} rows × {s.n_cols} columns")
        overview.add_row("Numeric columns", str(s.n_numeric))
        overview.add_row("Categorical columns", str(s.n_categorical))
        overview.add_row("Datetime columns", str(s.n_datetime))
        overview.add_row("Missing cells", f"{s.n_missing_cells:,} ({s.pct_missing_cells:.1f}%)")
        overview.add_row("Duplicate rows", str(s.n_duplicate_rows))
        overview.add_row("Memory", f"{s.memory_mb:.1f} MB")
        overview.print()

        # Column profiles
        col_tbl = RichTable(
            title="Column Profiles",
            columns=["Column", "Type", "Missing%", "Unique", "Cardinality", "Warnings"]
        )
        for col, prof in self.column_profiles.items():
            warn_str = ", ".join(prof.warnings[:2]) if prof.warnings else "—"
            col_tbl.add_row(
                col, prof.dtype_label,
                f"{prof.pct_missing:.1f}%",
                str(prof.n_unique), prof.cardinality, warn_str
            )
        col_tbl.print()

        # Issues
        if self.issues:
            issue_tbl = RichTable(title="Issues", columns=["Severity", "Column", "Message"])
            for issue in self.issues:
                issue_tbl.add_row(issue.severity.upper(), issue.column or "—", issue.message)
            issue_tbl.print()

        # Recommendations
        if self.recommendations:
            rec_tbl = RichTable(
                title="Recommendations",
                columns=["Priority", "Action", "Column", "Reason"]
            )
            for rec in sorted(self.recommendations, key=lambda x: x.priority):
                rec_tbl.add_row(str(rec.priority), rec.action, rec.column or "—", rec.reason)
            rec_tbl.print()

        print_success(f"Quality score: {self.quality_score:.0f}/100")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

class SmartEDA:
    """
    Internal class that performs all EDA computations.
    Use the top-level analyze() function instead.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        plot_backend: str = "plotly",
        max_categories: int = 50,
        sample_size: Optional[int] = None,
        corr_method: str = "pearson",
        corr_threshold: float = 0.8,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.df_original = df
        self.target = target
        self.plot_backend = plot_backend
        self.max_categories = max_categories
        self.sample_size = sample_size
        self.corr_method = corr_method
        self.corr_threshold = corr_threshold
        self.include_sections = include_sections
        self.exclude_sections = exclude_sections or []
        self.verbose = verbose

        # Sample if requested
        if sample_size and len(df) > sample_size:
            self.df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            if verbose:
                print_warning(f"Large dataset: sampling {sample_size:,} rows for EDA speed")
        else:
            self.df = df.copy()

    def run(self) -> EDAResult:
        from mlpilot.eda.plots import PlotEngine
        engine = PlotEngine(self.plot_backend)

        if self.verbose:
            print_step("Profiling dataset...", "| ")

        summary = self._compute_summary()
        column_profiles = self._profile_columns()
        correlations = self._compute_correlations(column_profiles)
        target_analysis = self._analyze_target(column_profiles) if self.target else None
        plots = self._generate_plots(engine, column_profiles, correlations, target_analysis)
        issues = self._detect_issues(summary, column_profiles, correlations, target_analysis)
        recommendations = self._build_recommendations(issues, column_profiles, target_analysis)
        quality_score = self._compute_quality_score(summary, column_profiles, issues)

        if self.verbose:
            print_success(f"EDA complete — quality score: {quality_score:.0f}/100")

        return EDAResult(
            df=self.df_original,
            summary=summary,
            column_profiles=column_profiles,
            correlations=correlations,
            target_analysis=target_analysis,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations,
            plots=plots,
        )

    # ------------------------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------------------------

    def _compute_summary(self) -> DatasetSummary:
        df = self.df
        n_rows, n_cols = df.shape
        missing_cells = int(df.isnull().sum().sum())
        pct_missing = missing_cells / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0.0

        type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0, "boolean": 0}
        for col in df.columns:
            lbl = self._dtype_label(df[col])
            type_counts[lbl] = type_counts.get(lbl, 0) + 1

        return DatasetSummary(
            n_rows=n_rows,
            n_cols=n_cols,
            n_numeric=type_counts["numeric"],
            n_categorical=type_counts["categorical"],
            n_datetime=type_counts["datetime"],
            n_text=type_counts["text"],
            n_boolean=type_counts["boolean"],
            n_missing_cells=missing_cells,
            pct_missing_cells=round(pct_missing, 2),
            n_duplicate_rows=int(df.duplicated().sum()),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 2),
        )

    # ------------------------------------------------------------------
    # Column Profiling
    # ------------------------------------------------------------------

    def _dtype_label(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            # High-cardinality integers might be IDs, but count as numeric
            return "numeric"
        # Object / string
        n_unique = series.nunique(dropna=True)
        avg_len = series.dropna().astype(str).str.len().mean() if len(series.dropna()) > 0 else 0
        if avg_len > 50 and n_unique > 100:
            return "text"
        return "categorical"

    def _cardinality_label(self, n_unique: int, n_rows: int) -> str:
        ratio = n_unique / max(n_rows, 1)
        if n_unique <= 2:
            return "low"
        if ratio > 0.9:
            return "unique"
        if n_unique > self.max_categories:
            return "high"
        if n_unique > 10:
            return "medium"
        return "low"

    def _profile_columns(self) -> Dict[str, ColumnProfile]:
        df = self.df
        profiles: Dict[str, ColumnProfile] = {}

        for col in df.columns:
            series = df[col]
            lbl = self._dtype_label(series)
            n_missing = int(series.isnull().sum())
            n_unique = int(series.nunique(dropna=True))
            cardinality = self._cardinality_label(n_unique, len(df))

            profile = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                dtype_label=lbl,
                n_missing=n_missing,
                pct_missing=round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0.0,
                n_unique=n_unique,
                cardinality=cardinality,
            )

            if lbl == "numeric":
                clean = series.dropna()
                if len(clean) > 0:
                    profile.mean = float(clean.mean())
                    profile.median = float(clean.median())
                    profile.std = float(clean.std())
                    profile.min = float(clean.min())
                    profile.max = float(clean.max())
                    profile.q1 = float(clean.quantile(0.25))
                    profile.q3 = float(clean.quantile(0.75))
                    try:
                        profile.skewness = float(clean.skew())
                        profile.kurtosis = float(clean.kurtosis())
                    except Exception:
                        pass
                    # IQR outliers
                    iqr = profile.q3 - profile.q1
                    if iqr > 0:
                        lo, hi = profile.q1 - 1.5 * iqr, profile.q3 + 1.5 * iqr
                        profile.n_outliers_iqr = int(((clean < lo) | (clean > hi)).sum())
                    # Z-score outliers
                    if profile.std and profile.std > 0:
                        z = np.abs((clean - profile.mean) / profile.std)
                        profile.n_outliers_zscore = int((z > 3).sum())

            elif lbl in ("categorical", "boolean"):
                vc = series.value_counts(dropna=True).head(10)
                profile.top_values = list(zip(vc.index.tolist(), vc.values.tolist()))
                profile.mode = vc.index[0] if len(vc) > 0 else None

            # Build warnings
            profile.warnings = self._column_warnings(profile)
            profile.recommended_transform = self._recommend_transform(profile)

            profiles[col] = profile

        return profiles

    def _column_warnings(self, prof: ColumnProfile) -> List[str]:
        w = []
        if prof.pct_missing > 50:
            w.append("high_missing")
        elif prof.pct_missing > 20:
            w.append("moderate_missing")
        if prof.cardinality == "high":
            w.append("high_cardinality")
        if prof.cardinality == "unique":
            w.append("likely_id_column")
        if prof.is_numeric and prof.std == 0:
            w.append("constant_column")
        if prof.is_numeric and prof.skewness and abs(prof.skewness) > 2:
            w.append("high_skewness")
        return w

    def _recommend_transform(self, prof: ColumnProfile) -> str:
        if "constant_column" in prof.warnings:
            return "drop"
        if "likely_id_column" in prof.warnings:
            return "drop (ID column)"
        if "high_missing" in prof.warnings:
            return "consider dropping"
        if prof.is_numeric:
            if prof.pct_missing > 0:
                return "impute (KNN or median)"
            if "high_skewness" in prof.warnings:
                return "log transform or robust scaling"
            return "standard or robust scaling"
        if prof.is_categorical:
            if prof.cardinality == "high":
                return "target encoding or hashing"
            return "one-hot encoding"
        if prof.is_datetime:
            return "extract features (hour, day, month, year)"
        return "review"

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    def _compute_correlations(
        self, profiles: Dict[str, ColumnProfile]
    ) -> Optional[CorrelationMatrix]:
        numeric_cols = [c for c, p in profiles.items() if p.is_numeric]
        if len(numeric_cols) < 2:
            return None
        try:
            sub = self.df[numeric_cols].dropna()
            if len(sub) < 10:
                return None
            corr = sub.corr(method=self.corr_method)
            high_pairs = []
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr.iloc[i, j]
                    if abs(val) >= self.corr_threshold and not np.isnan(val):
                        high_pairs.append((cols[i], cols[j], round(val, 4)))
            return CorrelationMatrix(
                matrix=corr,
                method=self.corr_method,
                threshold=self.corr_threshold,
                high_pairs=sorted(high_pairs, key=lambda x: -abs(x[2])),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Target Analysis
    # ------------------------------------------------------------------

    def _analyze_target(
        self, profiles: Dict[str, ColumnProfile]
    ) -> Optional[TargetAnalysis]:
        if self.target not in self.df.columns:
            print_warning(f"Target column '{self.target}' not found — skipping target analysis")
            return None

        target_series = self.df[self.target].dropna()
        prof = profiles.get(self.target)

        if prof and prof.is_numeric and prof.n_unique > 20:
            task = "regression"
        elif target_series.nunique() == 2:
            task = "binary"
        else:
            task = "multiclass"

        # Class distribution
        if task == "regression":
            class_dist = {}
            imbalance_score = 0.0
            severity = "none"
            strategy = "none"
        else:
            vc = target_series.value_counts()
            class_dist = vc.to_dict()
            if len(vc) >= 2:
                ratio = vc.iloc[0] / vc.iloc[-1]
                imbalance_score = min(100.0, (ratio - 1) / (ratio + 1) * 100)
                if ratio < 2:
                    severity = "none"
                    strategy = "none"
                elif ratio < 5:
                    severity = "mild"
                    strategy = "class_weight"
                elif ratio < 20:
                    severity = "moderate"
                    strategy = "smote"
                elif ratio < 100:
                    severity = "severe"
                    strategy = "adasyn"
                else:
                    severity = "critical"
                    strategy = "adasyn + undersampling"
            else:
                imbalance_score, severity, strategy = 0.0, "none", "none"

        # Top correlated features (numeric only)
        top_corr = []
        if task != "regression":
            # Point-biserial or correlation with label-encoded target
            try:
                y_encoded = pd.factorize(target_series)[0]
                for col, p in profiles.items():
                    if col == self.target or not p.is_numeric:
                        continue
                    try:
                        col_clean = self.df[col].fillna(self.df[col].median())
                        corr_val = float(np.corrcoef(col_clean.values, y_encoded)[0, 1])
                        if not np.isnan(corr_val):
                            top_corr.append((col, round(corr_val, 4)))
                    except Exception:
                        pass
                top_corr = sorted(top_corr, key=lambda x: -abs(x[1]))[:10]
            except Exception:
                pass

        # Mutual info feature importance
        feature_importance = {}
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            feat_cols = [c for c, p in profiles.items()
                         if c != self.target and p.is_numeric]
            if feat_cols:
                X_sub = self.df[feat_cols].fillna(0)
                y_sub = self.df[self.target]
                if task == "regression":
                    mi = mutual_info_regression(X_sub, y_sub, random_state=42)
                else:
                    y_enc = pd.factorize(y_sub)[0]
                    mi = mutual_info_classif(X_sub, y_enc, random_state=42)
                feature_importance = dict(zip(feat_cols, mi.tolist()))
        except Exception:
            pass

        return TargetAnalysis(
            column=self.target,
            task=task,
            class_distribution=class_dist,
            imbalance_score=round(imbalance_score, 1),
            imbalance_severity=severity,
            recommended_strategy=strategy,
            feature_importance=feature_importance,
            top_correlated=top_corr,
        )

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plots(
        self,
        engine: Any,
        profiles: Dict[str, ColumnProfile],
        correlations: Optional[CorrelationMatrix],
        target_analysis: Optional[TargetAnalysis],
    ) -> Dict[str, Any]:
        plots: Dict[str, Any] = {}

        # Histograms for numeric columns (up to 12)
        numeric_cols = [c for c, p in profiles.items() if p.is_numeric and c != self.target]
        for col in numeric_cols[:12]:
            fig = engine.histogram(self.df[col])
            if fig is not None:
                plots[f"hist_{col}"] = fig

        # Bar charts for categorical columns (up to 8)
        cat_cols = [c for c, p in profiles.items() if p.is_categorical and c != self.target]
        for col in cat_cols[:8]:
            fig = engine.bar_chart(self.df[col])
            if fig is not None:
                plots[f"bar_{col}"] = fig

        # Correlation heatmap
        if correlations is not None:
            fig = engine.correlation_heatmap(correlations.matrix)
            if fig is not None:
                plots["correlation_heatmap"] = fig

        # Missing heatmap (only if there are missing values)
        if self.df.isnull().any().any():
            fig = engine.missing_heatmap(self.df)
            if fig is not None:
                plots["missing_heatmap"] = fig

        # Target distribution
        if target_analysis is not None:
            task = target_analysis.task
            fig = engine.target_distribution(self.df[self.target], task)
            if fig is not None:
                plots["target_distribution"] = fig

        return plots

    # ------------------------------------------------------------------
    # Issue Detection
    # ------------------------------------------------------------------

    def _detect_issues(
        self,
        summary: DatasetSummary,
        profiles: Dict[str, ColumnProfile],
        correlations: Optional[CorrelationMatrix],
        target_analysis: Optional[TargetAnalysis],
    ) -> List[DataIssue]:
        issues: List[DataIssue] = []

        # Per-column issues
        for col, prof in profiles.items():
            if "high_missing" in prof.warnings:
                issues.append(DataIssue("warning", col, "high_missing",
                    f"{prof.pct_missing:.1f}% missing — consider dropping or imputing carefully",
                    prof.pct_missing))
            if "constant_column" in prof.warnings:
                issues.append(DataIssue("warning", col, "constant_col",
                    "Column is constant — will not help any model"))
            if "likely_id_column" in prof.warnings:
                issues.append(DataIssue("warning", col, "id_column",
                    "Looks like an ID column — should be excluded from features",
                    prof.n_unique))
            if "high_skewness" in prof.warnings:
                issues.append(DataIssue("info", col, "high_skewness",
                    f"Skewness = {prof.skewness:.2f} — consider log transform",
                    prof.skewness))

        # Dataset-level issues
        if summary.n_duplicate_rows > 0:
            issues.append(DataIssue("warning", None, "duplicates",
                f"{summary.n_duplicate_rows:,} duplicate rows detected",
                summary.n_duplicate_rows))

        if correlations and correlations.high_pairs:
            for a, b, v in correlations.high_pairs[:3]:
                issues.append(DataIssue("info", None, "high_correlation",
                    f"High correlation between '{a}' and '{b}' ({v:.2f}) — may cause multicollinearity",
                    v))

        if target_analysis and target_analysis.imbalance_severity in ("severe", "critical"):
            issues.append(DataIssue("warning", self.target, "class_imbalance",
                f"Target imbalance is {target_analysis.imbalance_severity} "
                f"(score={target_analysis.imbalance_score:.0f}/100). "
                f"Recommended: {target_analysis.recommended_strategy}",
                target_analysis.imbalance_score))

        return sorted(issues, key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x.severity])

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        issues: List[DataIssue],
        profiles: Dict[str, ColumnProfile],
        target_analysis: Optional[TargetAnalysis],
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []
        priority = 1

        for issue in issues:
            if issue.code == "high_missing":
                recs.append(Recommendation(
                    priority=priority, action="impute_or_drop", column=issue.column,
                    reason=f"{issue.value:.1f}% missing",
                    suggested_code=f"result = ml.clean(df, target='{self.target}')",
                ))
                priority += 1

            elif issue.code == "constant_col":
                recs.append(Recommendation(
                    priority=priority, action="drop_column", column=issue.column,
                    reason="Zero variance — no predictive value",
                    suggested_code=f"df = df.drop(columns=['{issue.column}'])",
                ))
                priority += 1

            elif issue.code == "id_column":
                recs.append(Recommendation(
                    priority=priority, action="exclude_from_features", column=issue.column,
                    reason="Likely a row identifier",
                    suggested_code=f"df = df.drop(columns=['{issue.column}'])",
                ))
                priority += 1

            elif issue.code == "class_imbalance":
                recs.append(Recommendation(
                    priority=priority, action="balance_classes", column=issue.column,
                    reason=f"Severe imbalance (score={issue.value:.0f}/100)",
                    suggested_code="result = ml.balance(X_train, y_train)",
                ))
                priority += 1

        # Always recommend feature engineering
        recs.append(Recommendation(
            priority=priority + 10, action="feature_engineering", column=None,
            reason="Apply leakage-safe encoding and scaling",
            suggested_code=f"result = ml.features(df, target='{self.target or 'target'}')",
        ))

        return recs

    # ------------------------------------------------------------------
    # Quality Score
    # ------------------------------------------------------------------

    def _compute_quality_score(
        self,
        summary: DatasetSummary,
        profiles: Dict[str, ColumnProfile],
        issues: List[DataIssue],
    ) -> float:
        score = 100.0
        # Penalize for missing values
        score -= min(30, summary.pct_missing_cells * 2)
        # Penalize for duplicates
        dup_pct = summary.n_duplicate_rows / max(summary.n_rows, 1) * 100
        score -= min(15, dup_pct)
        # Penalize per warning issue
        for issue in issues:
            if issue.severity == "critical":
                score -= 15
            elif issue.severity == "warning":
                score -= 5
            elif issue.severity == "info":
                score -= 1
        # Bonus for decent size
        if summary.n_rows >= 1000:
            score += 5
        return max(0.0, min(100.0, round(score, 1)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    df: pd.DataFrame,
    target: Optional[str] = None,
    report_format: Optional[str] = "html",
    output_path: str = "./mlpilot_eda",
    plot_backend: str = "plotly",
    max_categories: int = 50,
    sample_size: Optional[int] = None,
    corr_method: str = "pearson",
    corr_threshold: float = 0.8,
    include_sections: Optional[List[str]] = None,
    exclude_sections: Optional[List[str]] = None,
    color_palette: str = "mlpilot",
    verbose: bool = True,
) -> EDAResult:
    """
    SmartEDA — Intelligent Exploratory Data Analysis.

    Produces a complete 12-section analysis in one call.
    Replaces 8–12 hours of manual EDA per project.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    target : str, optional
        Target column name. Unlocks imbalance detection, feature importance, etc.
    report_format : str or None
        'html' to save report, None for in-memory only.
    output_path : str
        Directory to save the report.
    plot_backend : str
        'plotly' (default), 'matplotlib', or 'seaborn'.
    max_categories : int
        Skip categorical analysis for columns with more unique values than this.
    sample_size : int, optional
        Sample this many rows for speed on large datasets.
    corr_method : str
        'pearson', 'spearman', or 'kendall'.
    corr_threshold : float
        Flag correlations above this absolute value as high.
    verbose : bool
        Print progress to terminal.

    Returns
    -------
    EDAResult
        Contains summary, column_profiles, correlations, target_analysis,
        quality_score, issues, recommendations, and plots.

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.analyze(df, target='churn')
    >>> result.print_summary()
    >>> result.to_html('eda_report.html')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to analyze")

    eda = SmartEDA(
        df=df,
        target=target,
        plot_backend=plot_backend,
        max_categories=max_categories,
        sample_size=sample_size,
        corr_method=corr_method,
        corr_threshold=corr_threshold,
        include_sections=include_sections,
        exclude_sections=exclude_sections or [],
        verbose=verbose,
    )
    result = eda.run()

    if report_format == "html":
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = str((out_dir / "eda_report.html").resolve())
        result.to_html(report_path)
        result.report_path = report_path

    return result
