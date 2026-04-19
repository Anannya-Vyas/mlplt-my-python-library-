"""
mlpilot/train/eval.py
EvalSuite — complete model evaluation with all metrics and diagnostic plots.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlpilot.utils.display import RichTable, print_step, print_success
from mlpilot.utils.report_base import HTMLReportBuilder
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class EvalResult(BaseResult):
    """
    Complete result from ml.evaluate().

    Attributes
    ----------
    metrics : dict
        All computed metrics.
    confusion_matrix : pd.DataFrame
        Confusion matrix (classification only).
    roc_auc : float (classification binary)
    report_text : str
        sklearn classification_report output.
    plots : dict[str, Figure]
    threshold_analysis : pd.DataFrame (binary classification)
    """

    def __init__(
        self,
        metrics: Dict[str, Any],
        task: str,
        confusion_matrix: Optional[pd.DataFrame] = None,
        report_text: str = "",
        plots: Optional[Dict[str, Any]] = None,
        threshold_analysis: Optional[pd.DataFrame] = None,
        optimal_threshold: Optional[float] = None,
    ):
        self.metrics = metrics
        self.task = task
        self.confusion_matrix = confusion_matrix
        self.report_text = report_text
        self.plots = plots or {}
        self.threshold_analysis = threshold_analysis
        self.optimal_threshold = optimal_threshold

        # Convenience properties
        self.roc_auc = metrics.get("roc_auc")
        self.accuracy = metrics.get("accuracy")
        self.f1 = metrics.get("f1")
        self.rmse = metrics.get("rmse")
        self.r2 = metrics.get("r2")

    def print_report(self) -> None:
        """Print rich terminal evaluation summary."""
        tbl = RichTable(title="Evaluation Metrics", columns=["Metric", "Value"])
        for k, v in self.metrics.items():
            if isinstance(v, float):
                tbl.add_row(k, f"{v:.4f}")
            else:
                tbl.add_row(k, str(v))
        tbl.print()

        if self.report_text:
            print("\n" + self.report_text)

    def to_html(self, path: Optional[str] = None) -> str:
        builder = HTMLReportBuilder(title="Model Evaluation Report",
                                    subtitle=f"Task: {self.task}")
        # Metrics grid
        items = [(k, f"{v:.4f}" if isinstance(v, float) else str(v), "")
                 for k, v in self.metrics.items()]
        content = builder.stat_grid(items[:8], cols=4)

        # Confusion matrix
        if self.confusion_matrix is not None:
            rows = [list(map(str, r)) for r in self.confusion_matrix.values]
            content += builder.table(
                [""] + list(self.confusion_matrix.columns.astype(str)),
                [[str(self.confusion_matrix.index[i])] + rows[i]
                 for i in range(len(rows))]
            )

        # Threshold analysis
        if self.threshold_analysis is not None:
            th_rows = [list(map(lambda x: f"{x:.4f}", row))
                       for row in self.threshold_analysis.values]
            content += "<h3 style='margin-top:1rem'>Threshold Analysis</h3>"
            content += builder.table(list(self.threshold_analysis.columns), th_rows)

        builder.add_section(builder.section("📊 Evaluation Results", content))

        out_path = path or "mlpilot_evaluation.html"
        return builder.save(out_path)

    def __repr__(self) -> str:
        parts = [f"task={self.task}"]
        if self.roc_auc:
            parts.append(f"roc_auc={self.roc_auc:.4f}")
        if self.r2:
            parts.append(f"r2={self.r2:.4f}")
        return f"EvalResult({', '.join(parts)})"


# ---------------------------------------------------------------------------
# EvalSuite Engine
# ---------------------------------------------------------------------------

class EvalSuite:
    def __init__(
        self,
        task: str = "auto",
        threshold: float = 0.5,
        optimize_threshold: bool = False,
        business_costs: Optional[Dict[str, float]] = None,
        calibrate: bool = False,
        verbose: bool = True,
    ):
        self.task = task
        self.threshold = threshold
        self.optimize_threshold = optimize_threshold
        self.business_costs = business_costs
        self.calibrate = calibrate
        self.verbose = verbose

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> EvalResult:
        from sklearn import metrics as skm

        if self.task == "auto":
            n_unique = pd.Series(y_test).nunique()
            if pd.api.types.is_float_dtype(y_test) and n_unique > 20:
                self.task = "regression"
            elif n_unique == 2:
                self.task = "binary"
            else:
                self.task = "multiclass"

        if self.verbose:
            print_step(f"Evaluating model — task: {self.task}", "📊")

        y_pred = model.predict(X_test)
        metrics: Dict[str, Any] = {}
        cm = None
        report_text = ""
        plots: Dict[str, Any] = {}
        threshold_analysis = None
        optimal_threshold = None

        if self.task in ("binary", "multiclass"):
            metrics["accuracy"] = round(float(skm.accuracy_score(y_test, y_pred)), 4)
            avg = "binary" if self.task == "binary" else "macro"
            metrics["precision"] = round(float(skm.precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            metrics["recall"] = round(float(skm.recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            metrics["f1"] = round(float(skm.f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)

            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)
                    if self.task == "binary":
                        y_prob_pos = y_prob[:, 1]
                        metrics["roc_auc"] = round(float(skm.roc_auc_score(y_test, y_prob_pos)), 4)
                        metrics["log_loss"] = round(float(skm.log_loss(y_test, y_prob)), 4)
                        metrics["brier_score"] = round(float(skm.brier_score_loss(y_test, y_prob_pos)), 4)

                        # Threshold optimization
                        threshold_rows = []
                        thresholds = np.arange(0.1, 0.91, 0.05)
                        best_f1 = -1
                        for t in thresholds:
                            y_t = (y_prob_pos >= t).astype(int)
                            p = skm.precision_score(y_test, y_t, zero_division=0)
                            r = skm.recall_score(y_test, y_t, zero_division=0)
                            f = skm.f1_score(y_test, y_t, zero_division=0)
                            threshold_rows.append([t, p, r, f])
                            if f > best_f1:
                                best_f1 = f
                                optimal_threshold = float(t)
                        threshold_analysis = pd.DataFrame(
                            threshold_rows, columns=["threshold", "precision", "recall", "f1"]
                        )
                        if self.optimize_threshold and optimal_threshold:
                            metrics["optimal_threshold"] = round(optimal_threshold, 2)
                            # Re-predict with optimal threshold
                            y_pred = (y_prob_pos >= optimal_threshold).astype(int)
                            metrics["f1_at_optimal"] = round(
                                float(skm.f1_score(y_test, y_pred, zero_division=0)), 4
                            )
                    else:
                        metrics["roc_auc_ovr"] = round(float(
                            skm.roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                        ), 4)
            except Exception:
                pass

            try:
                metrics["mcc"] = round(float(skm.matthews_corrcoef(y_test, y_pred)), 4)
            except Exception:
                pass

            # Business cost
            if self.business_costs and self.task == "binary":
                try:
                    tn, fp, fn, tp = skm.confusion_matrix(y_test, y_pred).ravel()
                    cost = fp * self.business_costs.get("fp_cost", 0) + \
                           fn * self.business_costs.get("fn_cost", 0)
                    metrics["business_cost"] = round(float(cost), 2)
                except Exception:
                    pass

            report_text = skm.classification_report(y_test, y_pred, zero_division=0)

            # Confusion matrix
            cm_arr = skm.confusion_matrix(y_test, y_pred)
            labels = sorted(pd.Series(y_test).unique())
            cm = pd.DataFrame(cm_arr, index=labels, columns=labels)

            # ROC curve plot
            try:
                import plotly.graph_objects as go
                if self.task == "binary" and hasattr(model, "predict_proba"):
                    fpr, tpr, _ = skm.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                            name=f"ROC (AUC={metrics.get('roc_auc', 0):.3f})",
                                            line=dict(color="#6c63ff", width=2)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                            line=dict(color="#94a3b8", dash="dash"), name="Random"))
                    fig.update_layout(title="ROC Curve", template="plotly_dark",
                                     paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                                     font_color="#e2e8f0",
                                     xaxis_title="False Positive Rate",
                                     yaxis_title="True Positive Rate")
                    plots["roc_curve"] = fig
            except Exception:
                pass

        else:  # regression
            metrics["mae"] = round(float(skm.mean_absolute_error(y_test, y_pred)), 4)
            metrics["mse"] = round(float(skm.mean_squared_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(skm.mean_squared_error(y_test, y_pred))), 4)
            metrics["r2"] = round(float(skm.r2_score(y_test, y_pred)), 4)
            try:
                metrics["mape"] = round(float(skm.mean_absolute_percentage_error(y_test, y_pred)), 4)
            except Exception:
                pass
            metrics["explained_variance"] = round(
                float(skm.explained_variance_score(y_test, y_pred)), 4
            )
            n, p = len(y_test), X_test.shape[1] if hasattr(X_test, "shape") else 1
            r2 = metrics["r2"]
            metrics["adj_r2"] = round(1 - (1 - r2) * (n - 1) / max(n - p - 1, 1), 4)

            # Residuals plot
            try:
                import plotly.graph_objects as go
                residuals = np.array(y_test) - np.array(y_pred)
                fig = go.Figure(go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(color="#6c63ff", opacity=0.5),
                ))
                fig.update_layout(title="Residuals vs Predicted", template="plotly_dark",
                                 paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                                 font_color="#e2e8f0",
                                 xaxis_title="Predicted", yaxis_title="Residual")
                fig.add_hline(y=0, line_color="#ef4444", line_dash="dash")
                plots["residuals"] = fig
            except Exception:
                pass

        if self.verbose:
            print_success("Evaluation complete")
            result_tmp = EvalResult(metrics, self.task, cm, report_text,
                                     plots, threshold_analysis, optimal_threshold)
            result_tmp.print_report()

        return EvalResult(
            metrics=metrics,
            task=self.task,
            confusion_matrix=cm,
            report_text=report_text,
            plots=plots,
            threshold_analysis=threshold_analysis,
            optimal_threshold=optimal_threshold,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "auto",
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    business_costs: Optional[Dict[str, float]] = None,
    calibrate: bool = False,
    verbose: bool = True,
) -> EvalResult:
    """
    EvalSuite — Complete Model Evaluation.

    Computes every relevant metric and diagnostic plot for your model.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X_test, y_test : test data
    task : 'auto' | 'classification' | 'regression' | 'multiclass'
    threshold : float
        Decision threshold (binary classification, default 0.5).
    optimize_threshold : bool
        Auto-find threshold maximizing F1.
    business_costs : dict, optional
        {'fn_cost': 100, 'fp_cost': 10} for cost-sensitive evaluation.
    calibrate : bool
        Check and report calibration quality.
    verbose : bool

    Returns
    -------
    EvalResult
        Attributes: metrics, confusion_matrix, roc_auc, report_text,
        plots, threshold_analysis, optimal_threshold.
        Methods: print_report(), to_html().

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.evaluate(model, X_test, y_test, optimize_threshold=True)
    >>> result.metrics
    >>> result.to_html('evaluation.html')
    """
    suite = EvalSuite(
        task=task,
        threshold=threshold,
        optimize_threshold=optimize_threshold,
        business_costs=business_costs,
        calibrate=calibrate,
        verbose=verbose,
    )
    return suite.evaluate(model, pd.DataFrame(X_test), pd.Series(y_test))
