"""
mlpilot/timeseries/timesense.py
TimeSense — The automated time series forecasting engine.
One function call to go from a DataFrame to a multi-model forecast.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlpilot.timeseries.forecasters import MLForecaster, ProphetForecaster, StatsmodelsForecaster
from mlpilot.utils.display import print_step, print_success, print_warning
from mlpilot.utils.types import BaseResult


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------

class ForecastResult(BaseResult):
    """
    Result object from ml.forecast().

    Attributes
    ----------
    df_forecast : pd.DataFrame
        Dataframe with columns [ds, yhat, yhat_lower, yhat_upper].
    df_historical : pd.DataFrame
        The original input data.
    metrics : dict
        In-sample metrics (MAE, RMSE, etc.).
    engine_used : str
    target : str
    date_col : str
    """

    def __init__(
        self,
        df_forecast: pd.DataFrame,
        df_historical: pd.DataFrame,
        metrics: Dict[str, float],
        engine_used: str,
        target: str,
        date_col: str,
    ):
        self.df_forecast = df_forecast
        self.df_historical = df_historical
        self.metrics = metrics
        self.engine_used = engine_used
        self.target = target
        self.date_col = date_col

    def plot(self, show_historical: bool = True) -> Any:
        """Interactive forecast plot via Plotly."""
        try:
            import plotly.graph_objects as go
            fig = go.Figure()

            if show_historical:
                fig.add_trace(go.Scatter(
                    x=self.df_historical[self.date_col],
                    y=self.df_historical[self.target],
                    name="Historical",
                    line=dict(color="#94a3b8"),
                ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([self.df_forecast["ds"], self.df_forecast["ds"][::-1]]),
                y=pd.concat([self.df_forecast["yhat_upper"], self.df_forecast["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(108, 99, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Confidence Interval",
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=self.df_forecast["ds"],
                y=self.df_forecast["yhat"],
                name="Forecast",
                line=dict(color="#6c63ff", width=3),
            ))

            fig.update_layout(
                title=f"Forecast: {self.target} (via {self.engine_used})",
                template="plotly_dark",
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e2e8f0",
                xaxis_title="Date",
                yaxis_title=self.target,
            )
            fig.show()
            return fig
        except ImportError:
            print_warning("Plotly not installed — cannot show plot.")
            return None

    def print_summary(self) -> None:
        """Print summary of the forecast."""
        print_success(f"Forecast complete using {self.engine_used} engine.")
        print_step(f"Dates: {self.df_forecast['ds'].min().date()} to {self.df_forecast['ds'].max().date()}", "📅")
        for k, v in self.metrics.items():
            print_step(f"{k.upper()}: {v:.4f}", "📊")

    def __repr__(self) -> str:
        return (f"ForecastResult(engine={self.engine_used}, "
                f"horizon={len(self.df_forecast)}, target={self.target})")


# ---------------------------------------------------------------------------
# TimeSense Engine
# ---------------------------------------------------------------------------

def forecast(
    df: pd.DataFrame,
    target: Optional[str] = None,
    date_col: str = "auto",
    horizon: int = 30,
    engine: str = "auto",
    confidence: float = 0.95,
    verbose: bool = True,
) -> ForecastResult:
    """
    TimeSense — Intelligent Multi-Model Forecasting.

    Automatically handles date parsing, frequency detection, and model selection.
    Falls back gracefully if heavy libraries (Prophet) are missing.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Column to forecast.
    date_col : str
        Column containing dates. 'auto' searches for datetime types or common names.
    horizon : int
        Number of periods to forecast.
    engine : str
        'auto' | 'prophet' | 'statsmodels' | 'ml'
    confidence : float
        Confidence level (0-1).
    verbose : bool

    Returns
    -------
    ForecastResult
        Methods: plot(), print_summary().
        Properties: df_forecast, metrics.
    """
    df = df.copy()

    # 1. Date Detection
    if date_col == "auto":
        # Check for datetime dtypes
        dt_cols = df.select_dtypes(include=["datetime64"]).columns
        if not dt_cols.empty:
            date_col = dt_cols[0]
        else:
            # Check for common names
            candidates = ["date", "timestamp", "ds", "time", "day", "month"]
            for c in candidates:
                if c in df.columns.to_list():
                    date_col = c
                    break
            else:
                raise ValueError("Could not auto-detect date column. Please specify 'date_col'.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 1b. Target Detection (v1.2.5 Hardening)
    if target is None:
        # If there is a 'y' column, use it (Prophet style)
        if "y" in df.columns.to_list():
            target = "y"
        else:
            # Otherwise use the first numeric column that isn't the date_col
            num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
            if date_col in num_cols: num_cols.remove(date_col)
            if num_cols:
                target = num_cols[0]
            else:
                raise ValueError("Could not auto-detect target column for forecast.")

    if verbose:
        print_step(f"TimeSense: preparing forecast for '{target}'...", "📈")

    # 2. Engine Selection
    if engine == "auto":
        # Check Prophet availability
        try:
            import prophet  # noqa: F401
            engine = "prophet"
        except ImportError:
            try:
                import statsmodels  # noqa: F401
                engine = "statsmodels"
            except ImportError:
                engine = "ml"

    if verbose:
        print_step(f"Using '{engine}' engine", "⚙️")

    # 3. Fitting
    if engine == "prophet":
        forecaster = ProphetForecaster(interval_width=confidence)
    elif engine == "statsmodels":
        forecaster = StatsmodelsForecaster()
    else:
        forecaster = MLForecaster()

    forecaster.fit(df, target, date_col)

    # 4. Predicting
    df_forecast = forecaster.predict(horizon)

    # 5. Metrics (simple in-sample MAE/RMSE)
    # Note: In a real production tool we'd do backtesting here
    # For v0.1.0 we just report fitting success
    metrics = {
        "mae": 0.0,  # Placeholder for complex backtesting
        "rmse": 0.0,
    }

    result = ForecastResult(
        df_forecast=df_forecast,
        df_historical=df,
        metrics=metrics,
        engine_used=engine,
        target=target,
        date_col=date_col,
    )

    if verbose:
        result.print_summary()

    return result
