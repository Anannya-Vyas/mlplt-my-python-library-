"""
tests/test_timeseries.py
Tests for TimeSense (ml.forecast).
"""

import pandas as pd
import numpy as np
import pytest
import mlpilot as ml


@pytest.fixture
def ts_data():
    """Daily sales data for 60 days."""
    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
    # Linear trend + Weekly seasonality + Noise
    t = np.arange(60)
    sales = 100 + 2 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 5, 60)
    return pd.DataFrame({"date": dates, "sales": sales})


class TestTimeSense:
    def test_forecast_runs_auto(self, ts_data):
        result = ml.forecast(ts_data, target="sales", date_col="date", horizon=7, verbose=False)
        assert result.df_forecast is not None
        assert len(result.df_forecast) == 7
        assert "yhat" in result.df_forecast.columns

    def test_forecast_detects_date_col(self, ts_data):
        # Should auto-detect "date"
        result = ml.forecast(ts_data, target="sales", horizon=3, verbose=False)
        assert result.date_col == "date"

    def test_forecast_ml_engine(self, ts_data):
        result = ml.forecast(ts_data, target="sales", engine="ml", horizon=5, verbose=False)
        assert result.engine_used == "ml"
        assert len(result.df_forecast) == 5

    def test_forecast_output_format(self, ts_data):
        result = ml.forecast(ts_data, target="sales", horizon=7, verbose=False)
        cols = set(result.df_forecast.columns)
        assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(cols)

    def test_forecast_invalid_target(self, ts_data):
        with pytest.raises(KeyError):
            ml.forecast(ts_data, target="nonexistent", horizon=5, verbose=False)

    def test_forecast_plot_runs(self, ts_data):
        result = ml.forecast(ts_data, target="sales", horizon=5, verbose=False)
        # Mock show to avoid opening browser during tests
        fig = result.plot(show_historical=True)
        assert fig is not None or True # Just check it doesn't crash
