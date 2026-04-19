"""
mlpilot/timeseries/forecasters.py
Internal forecaster wrappers for TimeSense.
Supports Prophet, Statsmodels, and ML-based (XGBoost/LGBM) forecasting.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Base interface for all TimeSense forecasters."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str, date_col: str) -> BaseForecaster:
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Returns df with [ds, yhat, yhat_lower, yhat_upper]."""
        pass


# ---------------------------------------------------------------------------
# ML-Based Forecaster (The default fallback)
# ---------------------------------------------------------------------------

class MLForecaster(BaseForecaster):
    """
    Forecasting via Gradient Boosting (XGBoost/LGBM) with windowing.
    No heavy dependencies required beyond scikit-learn/xgboost.
    """

    def __init__(self, model_type: str = "xgboost", lags: int = 7,
                 rolling: List[int] = [7, 30], random_state: int = 42):
        self.model_type = model_type
        self.lags = lags
        self.rolling = rolling
        self.random_state = random_state
        self.model = None
        self.last_date = None
        self.freq = None
        self.target = None
        self.date_col = None

    def fit(self, df: pd.DataFrame, target: str, date_col: str) -> MLForecaster:
        self.target = target
        self.date_col = date_col
        df = df.sort_values(date_col).copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Detect frequency
        self.freq = pd.infer_freq(df[date_col])
        if not self.freq:
            # Fallback detection
            diffs = df[date_col].diff().dt.total_seconds().value_counts()
            if not diffs.empty:
                self.freq = pd.Timedelta(seconds=diffs.index[0])
            else:
                self.freq = "D"

        self.last_date = df[date_col].max()

        # Feature Engineering
        X, y = self._prepare_data(df)

        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                          random_state=self.random_state)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        self.model.fit(X, y)
        self._last_window = df.tail(max([self.lags] + self.rolling) + 1).copy()
        return self

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        target = self.target
        n_samples = len(df)

        # Only use rolling windows that fit
        valid_rolling = [w for w in self.rolling if w < n_samples]
        
        # Time features
        df["month"] = df[self.date_col].dt.month
        df["day"] = df[self.date_col].dt.day
        df["dayofweek"] = df[self.date_col].dt.dayofweek
        df["quarter"] = df[self.date_col].dt.quarter

        # Lags
        for i in range(1, self.lags + 1):
            df[f"lag_{i}"] = df[target].shift(i)

        # Rolling
        for window in valid_rolling:
            df[f"rolling_mean_{window}"] = df[target].shift(1).rolling(window=window).mean()
            df[f"rolling_std_{window}"] = df[target].shift(1).rolling(window=window).std()

        df = df.dropna()
        features = ["month", "day", "dayofweek", "quarter"] + \
                   [f"lag_{i}" for i in range(1, self.lags + 1)] + \
                   [f"rolling_mean_{w}" for w in valid_rolling] + \
                   [f"rolling_std_{w}" for w in valid_rolling]

        return df[features], df[target]

    def predict(self, horizon: int) -> pd.DataFrame:
        """Recursive multi-step forecasting."""
        forecast_dates = pd.date_range(start=self.last_date + pd.tseries.frequencies.to_offset(self.freq),
                                       periods=horizon, freq=self.freq)
        predictions = []
        current_df = self._last_window.copy()

        for date in forecast_dates:
            # Prepare one-row features
            row = pd.DataFrame({self.date_col: [date]})
            row["month"] = row[self.date_col].dt.month
            row["day"] = row[self.date_col].dt.day
            row["dayofweek"] = row[self.date_col].dt.dayofweek
            row["quarter"] = row[self.date_col].dt.quarter

            # Get lags from current_df
            for i in range(1, self.lags + 1):
                row[f"lag_{i}"] = current_df[self.target].iloc[-i]

            # Get rolling from current_df
            valid_rolling = [w for w in self.rolling if w < len(current_df)]
            for window in valid_rolling:
                subset = current_df[self.target].tail(window)
                row[f"rolling_mean_{window}"] = subset.mean()
                row[f"rolling_std_{window}"] = subset.std()

            features = ["month", "day", "dayofweek", "quarter"] + \
                       [f"lag_{i}" for i in range(1, self.lags + 1)] + \
                       [f"rolling_mean_{w}" for w in valid_rolling] + \
                       [f"rolling_std_{w}" for w in valid_rolling]

            pred = self.model.predict(row[features])[0]
            predictions.append(pred)

            # Append prediction to current_df for next recursive step
            new_row = pd.DataFrame({self.date_col: [date], self.target: [pred]})
            current_df = pd.concat([current_df, new_row], ignore_index=True)

        # Build output
        res = pd.DataFrame({
            "ds": forecast_dates,
            "yhat": predictions,
            "yhat_lower": [p * 0.95 for p in predictions],  # Simple heuristic for fallback
            "yhat_upper": [p * 1.05 for p in predictions],
        })
        return res


# ---------------------------------------------------------------------------
# Prophet Forecaster
# ---------------------------------------------------------------------------

class ProphetForecaster(BaseForecaster):
    """Wrapper for Facebook Prophet."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def fit(self, df: pd.DataFrame, target: str, date_col: str) -> ProphetForecaster:
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Use 'ml.forecast(..., engine=\"ml\")' "
                              "or install with 'pip install prophet'.")

        train_df = df[[date_col, target]].rename(columns={date_col: "ds", target: "y"})
        self.model = Prophet(**self.kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(train_df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        # Only return the future parts
        return forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ---------------------------------------------------------------------------
# Statsmodels Forecaster (ARIMA/ETS)
# ---------------------------------------------------------------------------

class StatsmodelsForecaster(BaseForecaster):
    """Wrapper for statsmodels ExponentialSmoothing or ARIMA."""

    def __init__(self, method: str = "ets"):
        self.method = method
        self.model_res = None
        self.last_date = None
        self.freq = None

    def fit(self, df: pd.DataFrame, target: str, date_col: str) -> StatsmodelsForecaster:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            raise ImportError("statsmodels not installed. install with 'pip install statsmodels'.")

        df = df.sort_values(date_col).copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        self.freq = df.index.inferred_freq or "D"
        df.index.freq = self.freq

        model = ExponentialSmoothing(df[target], trend="add", seasonal="add", seasonal_periods=7)
        self.model_res = model.fit()
        self.last_date = df.index.max()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        forecast = self.model_res.forecast(horizon)
        dates = pd.date_range(start=self.last_date + pd.tseries.frequencies.to_offset(self.freq),
                              periods=horizon, freq=self.freq)

        # Simple 95% CI approximation for ETS if not provided
        std = np.std(self.model_res.resid)
        return pd.DataFrame({
            "ds": dates,
            "yhat": forecast.values,
            "yhat_lower": forecast.values - 1.96 * std,
            "yhat_upper": forecast.values + 1.96 * std,
        })
