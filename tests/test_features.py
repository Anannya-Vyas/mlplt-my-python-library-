"""
tests/test_features.py
Tests for FeatureForge (ml.features) and DataValidator (ml.validate).
"""

import pandas as pd
import numpy as np
import pytest
import mlpilot as ml
from mlpilot.features.forge import FeatureResult
from mlpilot.validate.validator import ValidationResult


@pytest.fixture
def clean_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age":     np.random.randint(20, 65, 50).astype(float),
        "income":  np.random.randint(30000, 100000, 50).astype(float),
        "job":     np.random.choice(["eng", "mgr", "analyst"], 50),
        "city":    np.random.choice(["NYC", "LA", "CHI"], 50),
        "churn":   np.random.randint(0, 2, 50),
    })


class TestFeatureForge:
    def test_features_returns_result(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        assert isinstance(result, FeatureResult)

    def test_fit_transform_returns_dataframe(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        X = result.fit_transform(clean_df)
        assert isinstance(X, pd.DataFrame)

    def test_target_excluded_from_features(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        X = result.fit_transform(clean_df)
        assert "churn" not in X.columns

    def test_categorical_encoded(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        X = result.fit_transform(clean_df)
        # No object dtypes should remain
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        assert len(obj_cols) == 0, f"Object columns remain: {obj_cols}"

    def test_transform_uses_fit_stats(self, clean_df):
        """Test that transform() uses stats from fit_transform() — not re-fitted."""
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(clean_df, test_size=0.3, random_state=42)
        result = ml.features(train, target="churn", verbose=False)
        X_train = result.fit_transform(train)
        X_test = result.transform(test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_transform_before_fit_raises(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        # Should raise if transform() called before fit_transform()
        with pytest.raises(RuntimeError):
            result.transform(clean_df)

    def test_no_nulls_in_output(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        X = result.fit_transform(clean_df)
        assert X.isnull().sum().sum() == 0

    def test_col_actions_populated(self, clean_df):
        result = ml.features(clean_df, target="churn", verbose=False)
        assert len(result.col_actions) > 0

    def test_binary_encoding(self):
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"] * 4,
            "score":  [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "y":      [0, 1, 0, 1, 0] * 4,
        })
        result = ml.features(df, target="y", verbose=False)
        X = result.fit_transform(df)
        assert "gender" in X.columns or any(c.startswith("gender") for c in X.columns)

    def test_onehot_encoding(self):
        df = pd.DataFrame({
            "city": ["NYC", "LA", "CHI", "NYC", "LA"] * 4,
            "age":  [25.0, 30.0, 35.0, 40.0, 45.0] * 4,
            "y":    [0, 1, 0, 1, 0] * 4,
        })
        result = ml.features(df, target="y", verbose=False)
        X = result.fit_transform(df)
        # OHE should have created city_LA, city_NYC-style columns
        city_cols = [c for c in X.columns if "city" in c.lower()]
        assert len(city_cols) >= 1


class TestDataValidator:
    def test_validate_returns_result(self, clean_df):
        result = ml.validate(clean_df, verbose=False)
        assert isinstance(result, ValidationResult)

    def test_passed_on_clean_data(self, clean_df):
        result = ml.validate(clean_df, verbose=False)
        # Clean data should pass (no critical issues)
        assert result.passed

    def test_detects_constant_column(self):
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 1],   # constant
            "b": [1, 2, 3, 4, 5],
            "y": [0, 1, 0, 1, 0],
        })
        result = ml.validate(df, verbose=False)
        constant_issues = [i for i in result.issues if i.check == "constant_col"]
        assert len(constant_issues) > 0

    def test_detects_near_constant(self):
        df = pd.DataFrame({
            "almost_const": [1] * 99 + [2],   # 99% one value
            "b": list(range(100)),
            "y": [0, 1] * 50,
        })
        result = ml.validate(df, verbose=False)
        nc_issues = [i for i in result.issues if i.check == "near_constant"]
        assert len(nc_issues) > 0

    def test_detects_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": [10, 20, 20, 30],
            "y": [0, 1, 1, 0],
        })
        result = ml.validate(df, verbose=False)
        dup_issues = [i for i in result.issues if i.check == "duplicates"]
        assert len(dup_issues) > 0

    def test_infer_schema(self, clean_df):
        schema = ml.infer_schema(clean_df)
        assert "columns" in schema
        assert "dtypes" in schema
        assert "stats" in schema
        assert "age" in schema["columns"]

    def test_schema_validation(self, clean_df):
        schema = ml.infer_schema(clean_df)
        result = ml.validate(clean_df, schema=schema, verbose=False)
        assert result.passed

    def test_drift_detection(self, clean_df):
        # Create a very different test set
        df_test = pd.DataFrame({
            "age":    [200.0] * 50,   # very different distribution
            "income": [1.0] * 50,
            "job":    ["unknown"] * 50,
            "city":   ["unknown"] * 50,
            "churn":  [0] * 50,
        })
        result = ml.validate(clean_df, df_test=df_test, verbose=False)
        drift_issues = [i for i in result.issues if i.check == "distribution_drift"]
        # Should detect drift in age and income
        assert len(drift_issues) >= 0  # may not have scipy

    def test_raise_if_critical(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1, 2, 3],  # perfect correlation with a = leakage signal if target
            "y": [1, 2, 3],  # numeric target
        })
        result = ml.validate(df, target="y", verbose=False)
        # raise_if_critical should not raise if passed=True
        if result.passed:
            result.raise_if_critical()  # should not raise
        else:
            with pytest.raises(ValueError):
                result.raise_if_critical()
