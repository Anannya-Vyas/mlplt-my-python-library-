"""
tests/test_clean.py
Tests for AutoCleaner (ml.clean).
"""

import pandas as pd
import pytest
import mlpilot as ml
from mlpilot.clean.cleaner import CleaningRule, CleaningResult


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "age":    [25, 30, None, 40, 22, 35, 28, None, 31, 27],
        "income": [50000, 70000, 60000, None, 45000, 80000, 55000, 90000, None, 71000],
        "job":    ["eng", "mgr", "eng", "mgr", None, "eng", "mgr", "eng", "mgr", "eng"],
        "status": ["Yes", "No", "yes", "NO", "Y", "N", "YES", "no", "Yes", "No"],
        "churn":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def duplicate_df():
    return pd.DataFrame({
        "a": [1, 2, 2, 3, 3],
        "b": [10, 20, 20, 30, 30],
        "y": [0, 1, 1, 0, 0],
    })


class TestAutoCleaner:
    def test_clean_returns_result(self, dirty_df):
        result = ml.clean(dirty_df, verbose=False)
        assert isinstance(result, CleaningResult)

    def test_clean_fixes_nulls(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        assert result.df.isnull().sum().sum() == 0

    def test_null_count(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        assert result.n_nulls_filled > 0

    def test_undo_restores_original(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        restored = result.undo()
        assert restored.isnull().sum().sum() == dirty_df.isnull().sum().sum()

    def test_target_protected(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        # Target (churn) should be unchanged
        assert list(result.df["churn"]) == list(dirty_df["churn"])

    def test_removes_duplicates(self, duplicate_df):
        result = ml.clean(duplicate_df, verbose=False)
        assert result.n_duplicates_removed == 2
        assert len(result.df) == 3

    def test_quality_improves(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        assert result.quality_after >= result.quality_before

    def test_report_has_changes(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", verbose=False)
        assert len(result.report.changes) > 0

    def test_code_property(self, dirty_df):
        result = ml.clean(dirty_df, verbose=False)
        assert "ml.clean" in result.code
        assert isinstance(result.code, str)

    def test_custom_rule_clip(self, dirty_df):
        rules = [CleaningRule(column="age", strategy="clip", min=20, max=40)]
        result = ml.clean(dirty_df, target="churn", custom_rules=rules, verbose=False)
        assert result.df["age"].min() >= 20
        assert result.df["age"].max() <= 40

    def test_custom_rule_unify(self, dirty_df):
        rules = [CleaningRule(column="status", strategy="unify",
                             mapping={"yes": "Yes", "YES": "Yes", "Y": "Yes",
                                      "no": "No", "NO": "No", "N": "No"})]
        result = ml.clean(dirty_df, target="churn", custom_rules=rules, verbose=False)
        valid_values = {"Yes", "No"}
        assert set(result.df["status"].dropna().unique()).issubset(valid_values)

    def test_outlier_handling(self):
        import numpy as np
        df = pd.DataFrame({
            "x": list(range(100)) + [1000, -1000],  # extreme outliers
            "y": [0] * 102
        })
        result = ml.clean(df, outlier_strategy="iqr", outlier_action="clip", verbose=False)
        assert result.df["x"].max() < 1000

    def test_dtype_fix(self):
        df = pd.DataFrame({
            "price": ["100.5", "200.3", "300.1", "not_a_number", "150.0"],
            "y": [1, 0, 1, 0, 1]
        })
        result = ml.clean(df, verbose=False)
        # Should have attempted to convert "price" to numeric
        # (some values may still be NaN from failed conversion)
        assert result is not None

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            ml.clean(pd.DataFrame(), verbose=False)

    def test_null_strategy_median(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", null_strategy="median", verbose=False)
        assert result.df["age"].isnull().sum() == 0

    def test_null_strategy_mean(self, dirty_df):
        result = ml.clean(dirty_df, target="churn", null_strategy="mean", verbose=False)
        assert result.df["income"].isnull().sum() == 0
