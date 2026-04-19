"""
mlpilot/eda/plots.py
All plot-generation functions for SmartEDA.
Supports plotly (default), matplotlib, and seaborn backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


def _get_mlpilot_colors() -> List[str]:
    return [
        "#6c63ff", "#00d9c6", "#f59e0b", "#ef4444", "#10b981",
        "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
    ]


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------

def _plotly_histogram(series: pd.Series, title: str, color: str = "#6c63ff") -> Any:
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series.dropna(),
            nbinsx=30,
            marker_color=color,
            opacity=0.85,
            name=series.name,
        ))
        fig.update_layout(
            title=title, template="plotly_dark",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig
    except ImportError:
        return None


def _plotly_bar(series: pd.Series, title: str, top_n: int = 20) -> Any:
    try:
        import plotly.graph_objects as go
        vc = series.value_counts().head(top_n)
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str),
            y=vc.values,
            marker_color=_get_mlpilot_colors()[0],
        ))
        fig.update_layout(
            title=title, template="plotly_dark",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(l=40, r=20, t=40, b=80),
        )
        return fig
    except ImportError:
        return None


def _plotly_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> Any:
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont_size=9,
        ))
        fig.update_layout(
            title=title, template="plotly_dark",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e2e8f0",
            margin=dict(l=120, r=20, t=60, b=120),
            height=max(400, len(corr_matrix) * 30 + 100),
        )
        return fig
    except ImportError:
        return None


def _plotly_missing_heatmap(df: pd.DataFrame) -> Any:
    try:
        import plotly.graph_objects as go
        miss = df.isnull().astype(int)
        fig = go.Figure(go.Heatmap(
            z=miss.values.T,
            x=list(range(len(df))),
            y=miss.columns.tolist(),
            colorscale=[[0, "#1a1d2e"], [1, "#ef4444"]],
            showscale=False,
        ))
        fig.update_layout(
            title="Missing Value Map",
            template="plotly_dark",
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font_color="#e2e8f0",
            height=max(200, len(df.columns) * 20 + 80),
            margin=dict(l=120, r=20, t=40, b=40),
        )
        return fig
    except ImportError:
        return None


def _plotly_target_distribution(series: pd.Series, task: str = "classification") -> Any:
    try:
        import plotly.graph_objects as go
        colors = _get_mlpilot_colors()
        if task == "classification":
            vc = series.value_counts()
            fig = go.Figure(go.Bar(
                x=vc.index.astype(str), y=vc.values,
                marker_color=colors[:len(vc)],
            ))
            fig.update_layout(title=f"Target Distribution: {series.name}",
                              template="plotly_dark",
                              paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                              font_color="#e2e8f0", showlegend=False)
        else:
            fig = go.Figure(go.Histogram(
                x=series.dropna(), marker_color=colors[0], nbinsx=30,
            ))
            fig.update_layout(title=f"Target Distribution: {series.name}",
                              template="plotly_dark",
                              paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                              font_color="#e2e8f0", showlegend=False)
        return fig
    except ImportError:
        return None


def _plotly_box(series: pd.Series, title: str) -> Any:
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Box(
            y=series.dropna(),
            name=series.name,
            marker_color="#6c63ff",
            line_color="#00d9c6",
        ))
        fig.update_layout(title=title, template="plotly_dark",
                          paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                          font_color="#e2e8f0", showlegend=False)
        return fig
    except ImportError:
        return None


def _plotly_scatter(x: pd.Series, y: pd.Series, color: pd.Series = None,
                    title: str = "") -> Any:
    try:
        import plotly.express as px
        df_plot = pd.DataFrame({"x": x, "y": y})
        if color is not None:
            df_plot["color"] = color.astype(str)
            fig = px.scatter(df_plot, x="x", y="y", color="color",
                            color_discrete_sequence=_get_mlpilot_colors())
        else:
            fig = px.scatter(df_plot, x="x", y="y",
                            color_discrete_sequence=_get_mlpilot_colors())
        fig.update_layout(title=title, template="plotly_dark",
                          paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                          font_color="#e2e8f0")
        return fig
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Matplotlib fallback
# ---------------------------------------------------------------------------

def _mpl_histogram(series: pd.Series, title: str) -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1d2e")
        ax.set_facecolor("#1a1d2e")
        ax.hist(series.dropna(), bins=30, color="#6c63ff", alpha=0.85, edgecolor="none")
        ax.set_title(title, color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3150")
        plt.tight_layout()
        return fig
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

class PlotEngine:
    """
    Selects the right backend and produces the right charts.
    Returns whatever the backend returns (plotly Figure, mpl Figure, etc.).
    Callers should handle the type when embedding.
    """

    def __init__(self, backend: str = "plotly"):
        self.backend = backend

    def histogram(self, series: pd.Series, title: str = "") -> Any:
        t = title or f"Distribution: {series.name}"
        if self.backend == "plotly":
            return _plotly_histogram(series, t)
        return _mpl_histogram(series, t)

    def bar_chart(self, series: pd.Series, title: str = "") -> Any:
        t = title or f"Value Counts: {series.name}"
        if self.backend == "plotly":
            return _plotly_bar(series, t)
        return None

    def correlation_heatmap(self, corr: pd.DataFrame) -> Any:
        if self.backend == "plotly":
            return _plotly_heatmap(corr)
        return None

    def missing_heatmap(self, df: pd.DataFrame) -> Any:
        if self.backend == "plotly":
            return _plotly_missing_heatmap(df)
        return None

    def target_distribution(self, series: pd.Series, task: str = "classification") -> Any:
        if self.backend == "plotly":
            return _plotly_target_distribution(series, task)
        return None

    def box_plot(self, series: pd.Series, title: str = "") -> Any:
        t = title or f"Box Plot: {series.name}"
        if self.backend == "plotly":
            return _plotly_box(series, t)
        return None

    def scatter(self, x: pd.Series, y: pd.Series, color: pd.Series = None,
                title: str = "") -> Any:
        if self.backend == "plotly":
            return _plotly_scatter(x, y, color, title)
        return None

    def feature_vs_target(self, feature: pd.Series, target: pd.Series,
                          title: str = "") -> Any:
        """Plot a feature against the target variable."""
        t = title or f"{feature.name} vs {target.name}"
        if pd.api.types.is_numeric_dtype(feature):
            return self.box_plot(feature, t)
        else:
            return self.bar_chart(feature, t)
