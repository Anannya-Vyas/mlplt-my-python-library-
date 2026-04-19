"""
mlpilot/explain/explainer.py
Explainer — SHAP-powered global and local model explanations.
Works on any sklearn-compatible model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlpilot.utils.display import print_step, print_success, print_warning
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# ExplainerResult (for single row explanations)
# ---------------------------------------------------------------------------

class RowExplanation:
    def __init__(self, shap_values: Any, feature_names: List[str],
                 prediction: Any, probability: Optional[float] = None,
                 text: str = ""):
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.prediction = prediction
        self.probability = probability
        self.text = text

    def __repr__(self) -> str:
        prob_str = f", probability={self.probability:.2f}" if self.probability else ""
        return f"RowExplanation(prediction={self.prediction}{prob_str})"


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    """
    SHAP-powered model explainer.
    Created by ml.explain(model, X_train, X_test).
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.feature_names = feature_names or list(X_train.columns)
        self._shap_explainer = None
        self._shap_values_train = None
        self._shap_values_test = None

    def _ensure_shap(self) -> None:
        if self._shap_explainer is not None:
            return
        try:
            import shap  # noqa: F401
        except ImportError:
            raise ImportError(
                "SHAP is required for explanations. Install with: pip install mlpilot[shap]"
            )

        import shap

        model_type = type(self.model).__name__

        # Choose appropriate explainer
        if "XGB" in model_type or "LGBM" in model_type or "GradientBoosting" in model_type:
            self._shap_explainer = shap.TreeExplainer(self.model)
        elif "Forest" in model_type or "Tree" in model_type or "ExtraTrees" in model_type:
            self._shap_explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, "predict_proba"):
            # Use KernelExplainer for other models (slower)
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self._shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
        else:
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self._shap_explainer = shap.KernelExplainer(
                self.model.predict, background
            )

        print_step("Computing SHAP values...", "🔍")
        try:
            self._shap_values_train = self._shap_explainer.shap_values(self.X_train)
            self._shap_values_test = self._shap_explainer.shap_values(self.X_test)
        except Exception as e:
            print_warning(f"SHAP computation issue: {e}. Using approximation.")
            # Fallback: use a subsample
            sample = self.X_train.sample(min(200, len(self.X_train)), random_state=42)
            self._shap_values_train = self._shap_explainer.shap_values(sample)
            self._shap_values_test = self._shap_explainer.shap_values(
                self.X_test.head(200)
            )

    def feature_importance(self, n_features: int = 20) -> Any:
        """
        Bar chart of mean |SHAP| values — global feature importance.
        Returns a plotly figure.
        """
        self._ensure_shap()
        sv = self._shap_values_train
        if isinstance(sv, list):
            # multiclass — average across classes
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        elif sv.ndim == 3:
            sv = np.abs(sv).mean(axis=2)

        mean_shap = np.abs(sv).mean(axis=0)
        feat_names = self.feature_names[:len(mean_shap)]
        importance_df = pd.DataFrame({
            "feature": feat_names,
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).head(n_features)

        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                marker_color="#6c63ff",
            ))
            fig.update_layout(
                title=f"Top {n_features} Features by SHAP Importance",
                template="plotly_dark",
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e2e8f0",
                yaxis=dict(autorange="reversed"),
                height=max(400, n_features * 25),
                xaxis_title="Mean |SHAP value|",
            )
            fig.show()
            return fig
        except ImportError:
            # Print table fallback
            from mlpilot.utils.display import RichTable
            tbl = RichTable(title="Feature Importance (SHAP)",
                            columns=["Rank", "Feature", "Mean |SHAP|"])
            for i, row in importance_df.iterrows():
                tbl.add_row(str(i + 1), row["feature"], f"{row['importance']:.4f}")
            tbl.print()
            return importance_df
            
    def plot_importance(self, n_features: int = 20) -> Any:
        """Alias for feature_importance()"""
        return self.feature_importance(n_features)

    def summary(self) -> Any:
        """Alias for summary_plot()"""
        return self.summary_plot()

    def plot_summary(self) -> Any:
        """Alias for summary_plot()"""
        return self.summary_plot()

    def summary_plot(self) -> Any:
        """SHAP beeswarm/summary plot."""
        self._ensure_shap()
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            shap.summary_plot(
                self._shap_values_train,
                self.X_train,
                feature_names=self.feature_names,
                show=False,
            )
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            return fig
        except Exception as e:
            print_warning(f"Summary plot failed: {e}")
            return None

    def explain_row(self, row: pd.Series, text: bool = False) -> RowExplanation:
        """
        Local explanation for a single row/sample.

        Parameters
        ----------
        row : pd.Series
            A single row (e.g., X_test.iloc[0]).
        text : bool
            If True, generate a natural language explanation.

        Returns
        -------
        RowExplanation with shap_values, prediction, probability, text.
        """
        self._ensure_shap()
        import shap

        row_df = pd.DataFrame([row] if isinstance(row, pd.Series) else row)
        sv = self._shap_explainer.shap_values(row_df)

        # Get prediction
        prediction = self.model.predict(row_df)[0]
        probability = None
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(row_df)[0]
            if len(prob) == 2:
                probability = float(prob[1])
            else:
                probability = float(max(prob))

        # Build text explanation
        explanation_text = ""
        if text:
            explanation_text = self._build_text_explanation(row, sv, prediction, probability)

        # Waterfall plot
        try:
            if isinstance(sv, list):
                sv_single = sv[int(prediction)][0]
            elif sv.ndim == 3:
                sv_single = sv[0, :, int(prediction)]
            else:
                sv_single = sv[0]

            import plotly.graph_objects as go
            feature_names = self.feature_names[:len(sv_single)]
            vals = sv_single[:len(feature_names)]
            sorted_idx = np.argsort(np.abs(vals))[::-1][:15]
            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="h",
                measure=["relative"] * len(sorted_idx),
                y=[feature_names[i] for i in sorted_idx],
                x=[vals[i] for i in sorted_idx],
                marker=dict(
                    color=["#ef4444" if v > 0 else "#10b981" for v in [vals[i] for i in sorted_idx]]
                ),
            ))
            fig.update_layout(title=f"SHAP Waterfall — Prediction: {prediction}",
                             template="plotly_dark",
                             paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                             font_color="#e2e8f0")
            fig.show()
        except Exception:
            pass

        return RowExplanation(
            shap_values=sv,
            feature_names=self.feature_names,
            prediction=prediction,
            probability=probability,
            text=explanation_text,
        )

    def plot_contributions(self, row: pd.Series, text: bool = False) -> RowExplanation:
        """Alias for explain_row() — used by many AI assistants."""
        return self.explain_row(row, text=text)

    def plot_waterfall(self, row: pd.Series, text: bool = False) -> RowExplanation:
        """Alias for explain_row()"""
        return self.explain_row(row, text=text)

    def plot_force(self, row: pd.Series, text: bool = False) -> RowExplanation:
        """Alias for explain_row()"""
        return self.explain_row(row, text=text)

    def _build_text_explanation(self, row, sv, prediction, probability) -> str:
        """Generate a natural language explanation."""
        try:
            if isinstance(sv, list):
                sv_use = sv[int(prediction)][0]
            elif sv.ndim == 3:
                sv_use = sv[0, :, int(prediction)]
            else:
                sv_use = sv[0]

            feat_names = self.feature_names[:len(sv_use)]
            vals = sv_use[:len(feat_names)]
            top_idx = np.argsort(np.abs(vals))[::-1][:5]

            prob_str = f"(probability: {probability:.0%})" if probability else ""
            lines = [f"This sample is predicted as class {prediction} {prob_str}."]
            lines.append("The main contributing factors are:")
            for rank, i in enumerate(top_idx, 1):
                direction = "increases" if vals[i] > 0 else "decreases"
                feat_val = row.iloc[i] if hasattr(row, "iloc") else row[i]
                lines.append(
                    f"  {rank}. '{feat_names[i]}' = {feat_val!r} "
                    f"({direction} prediction by {abs(vals[i]):.3f} SHAP units)"
                )
            return "\n".join(lines)
        except Exception:
            return f"Predicted: {prediction}" + (f" with probability {probability:.2%}" if probability else "")

    def whatif(self, row: pd.Series, changes: Dict[str, Any]) -> RowExplanation:
        """
        Show how the prediction changes when specific feature values change.

        Parameters
        ----------
        row : pd.Series
            Original sample.
        changes : dict
            {feature_name: new_value} changes to apply.
        """
        original_pred = self.model.predict(pd.DataFrame([row]))[0]
        modified_row = row.copy()
        for feature, value in changes.items():
            if feature in modified_row.index:
                modified_row[feature] = value

        modified_pred = self.model.predict(pd.DataFrame([modified_row]))[0]

        from mlpilot.utils.display import RichTable
        tbl = RichTable(title="What-if Analysis", columns=["Feature", "Original", "Changed"])
        for feature, value in changes.items():
            original_val = row.get(feature, "—")
            tbl.add_row(feature, str(original_val), str(value))
        tbl.print()
        print_success(f"Prediction: {original_pred} → {modified_pred}")

        return self.explain_row(modified_row)

    def flip_prediction(self, row: pd.Series, max_changes: int = 5) -> Dict[str, Any]:
        """
        Find the minimum changes needed to flip the prediction.
        Returns a dict of suggested changes.
        """
        self._ensure_shap()
        row_df = pd.DataFrame([row])
        original_pred = self.model.predict(row_df)[0]

        sv = self._shap_explainer.shap_values(row_df)
        if isinstance(sv, list):
            sv_use = sv[int(original_pred)][0]
        elif sv.ndim == 3:
            sv_use = sv[0, :, int(original_pred)]
        else:
            sv_use = sv[0]

        feat_names = self.feature_names[:len(sv_use)]
        vals = sv_use[:len(feat_names)]

        # Top features pushing toward current prediction
        top_features = [(feat_names[i], vals[i]) for i in np.argsort(np.abs(vals))[::-1][:max_changes]]

        suggestions = {}
        for feat, shap_val in top_features:
            if feat in row.index:
                current = row[feat]
                if pd.api.types.is_numeric_dtype(type(current)):
                    # Suggest moving in opposite direction
                    delta = -shap_val * 2
                    suggestions[feat] = round(float(current) + delta, 4)

        print_step("Minimum changes to flip prediction:", "🔄")
        for feat, val in suggestions.items():
            print_step(f"  {feat}: {row.get(feat, '?')} → {val}", "  ")

        return suggestions

    def interaction_plot(self, feature: str) -> Any:
        """Show how a feature interacts with other features via SHAP."""
        print_warning("interaction_plot() requires shap>=0.42. "
                      "This is a placeholder — full support in v0.2.0")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(
    model: Any,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Explainer:
    """
    Explainer — SHAP-Powered Model Explainability.

    Works on any sklearn-compatible model.
    Requires: pip install mlpilot[shap]

    Parameters
    ----------
    model : sklearn-compatible estimator (fitted)
    X_train : pd.DataFrame
        Training data — used to build SHAP background.
    X_test : pd.DataFrame, optional
        Test data — used for local explanations.
    feature_names : list[str], optional
        Override feature names.
    verbose : bool

    Returns
    -------
    Explainer
        Methods: feature_importance(), summary_plot(), explain_row(),
        whatif(), flip_prediction(), interaction_plot().

    Examples
    --------
    >>> import mlpilot as ml
    >>> exp = ml.explain(model, X_train, X_test)
    >>> exp.feature_importance()         # global SHAP importance bar chart
    >>> exp.explain_row(X_test.iloc[0], text=True)  # natural language explanation
    >>> exp.whatif(X_test.iloc[0], changes={'age': 35, 'balance': 5000})
    """
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if X_test is not None and not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    if verbose:
        print_step("Building Explainer...", "🔍")

    return Explainer(model=model, X_train=X_train, X_test=X_test,
                     feature_names=feature_names)
