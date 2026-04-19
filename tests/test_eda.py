"""
tests/test_eda.py
Tests for SmartEDA (ml.analyze).
"""

import pandas as pd
import pytest
import mlpilot as ml


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":    [25, 30, None, 40, 22, 35, 28, 45, None, 33],
        "income": [50000, 70000, 60000, None, 45000, 80000, 55000, 90000, 62000, 71000],
        "job":    ["eng", "mgr", "eng", "mgr", None, "eng", "mgr", "eng", "mgr", "eng"],
        "city":   ["NYC", "LA", "NYC", "LA", "NYC", "CHI", "LA", "NYC", "CHI", "LA"],
        "churn":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


class TestSmartEDA:
    def test_analyze_runs(self, sample_df):
        result = ml.analyze(sample_df, target="churn", report_format=None)
        assert result is not None
        assert result.quality_score >= 0
        assert result.quality_score <= 100

    def test_analyze_returns_eda_result(self, sample_df):
        from mlpilot.eda.analyzer import EDAResult
        result = ml.analyze(sample_df, report_format=None)
        assert isinstance(result, EDAResult)

    def test_column_profiles_exist(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert "age" in result.column_profiles
        assert "job" in result.column_profiles

    def test_detects_missing(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert result.column_profiles["age"].n_missing == 2
        assert result.column_profiles["income"].n_missing == 1

    def test_detects_nulls_correctly(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert result.column_profiles["job"].n_missing == 1

    def test_target_analysis_binary(self, sample_df):
        result = ml.analyze(sample_df, target="churn", report_format=None)
        assert result.target_analysis is not None
        assert result.target_analysis.task == "binary"
        assert result.target_analysis.imbalance_severity == "none"

    def test_summary_shape(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert result.summary.n_rows == 10
        assert result.summary.n_cols == 5

    def test_numeric_stats(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        age_prof = result.column_profiles["age"]
        assert age_prof.is_numeric
        assert age_prof.mean is not None
        assert age_prof.median is not None

    def test_categorical_profile(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        job_prof = result.column_profiles["job"]
        assert job_prof.is_categorical
        assert len(job_prof.top_values) > 0

    def test_quality_score_float(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert isinstance(result.quality_score, float)

    def test_correlations_computed(self):
        # Use a df with no missing values so dropna doesn't reduce below 10 rows
        df2 = pd.DataFrame({
            "age":    [25.0, 30.0, 35.0, 40.0, 22.0, 35.0, 28.0, 45.0, 31.0, 33.0],
            "income": [50000.0, 70000.0, 60000.0, 90000.0, 45000.0, 80000.0, 55000.0, 90000.0, 62000.0, 71000.0],
        })
        result = ml.analyze(df2, report_format=None, verbose=False)
        assert result.correlations is not None
        assert result.correlations.matrix is not None

    def test_print_summary_runs(self, sample_df, capsys):
        result = ml.analyze(sample_df, report_format=None)
        result.print_summary()  # Should not raise

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            ml.analyze(pd.DataFrame(), report_format=None)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            ml.analyze([[1, 2], [3, 4]], report_format=None)

    def test_sample_size(self, sample_df):
        # With small sample_size, should still work
        result = ml.analyze(sample_df, sample_size=5, report_format=None)
        assert result is not None

    def test_issues_list(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert isinstance(result.issues, list)

    def test_recommendations_list(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert isinstance(result.recommendations, list)

    def test_plots_dict(self, sample_df):
        result = ml.analyze(sample_df, report_format=None)
        assert isinstance(result.plots, dict)

    def test_regression_target(self):
        # Use 100 rows with 21+ unique float values to trigger regression task
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.random.normal(0, 1, 100),
            "x2": np.random.normal(0, 2, 100),
            "price": np.random.normal(500, 100, 100),  # 100 unique floats
        })
        result = ml.analyze(df, target="price", report_format=None, verbose=False)
        assert result.target_analysis.task == "regression"
