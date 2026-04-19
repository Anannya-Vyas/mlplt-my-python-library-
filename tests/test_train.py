"""
tests/test_train.py
Tests for BaselineBlitz, EvalSuite, HyperX, and BalanceKit.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression

import mlpilot as ml
from mlpilot.train.baseline import BaselineResult
from mlpilot.train.eval import EvalResult
from mlpilot.balance.balancekit import BalanceResult


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=42, n_classes=2
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_s = pd.Series(y)
    split = 160
    return X_df[:split], X_df[split:], y_s[:split], y_s[split:]


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_s = pd.Series(y)
    split = 160
    return X_df[:split], X_df[split:], y_s[:split], y_s[split:]


@pytest.fixture
def imbalanced_data():
    """80:20 imbalance for BalanceKit tests."""
    X, y = make_classification(
        n_samples=500, weights=[0.8, 0.2], n_features=8,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


class TestBaselineBlitz:
    def test_baseline_returns_result(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        result = ml.baseline(
            X_train, y_train, X_test=X_test, y_test=y_test,
            models=["LogisticRegression", "RandomForestClassifier"],
            verbose=False,
        )
        assert isinstance(result, BaselineResult)

    def test_leaderboard_not_empty(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        result = ml.baseline(X_train, y_train,
                             models=["LogisticRegression", "GaussianNB"],
                             verbose=False)
        assert len(result.leaderboard._rows) >= 1

    def test_best_model_exists(self, classification_data):
        X_train, _, y_train, _ = classification_data
        result = ml.baseline(X_train, y_train,
                             models=["LogisticRegression", "GaussianNB"],
                             verbose=False)
        assert result.best_model is not None

    def test_best_score_float(self, classification_data):
        X_train, _, y_train, _ = classification_data
        result = ml.baseline(X_train, y_train,
                             models=["LogisticRegression"],
                             verbose=False)
        assert isinstance(result.best_score, float)

    def test_auto_task_detection(self, regression_data):
        X_train, _, y_train, _ = regression_data
        result = ml.baseline(X_train, y_train,
                             models=["LinearRegression"],
                             task="auto", verbose=False)
        assert result.task == "regression"

    def test_refit_works(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        result = ml.baseline(X_train, y_train,
                             models=["LogisticRegression"],
                             verbose=False)
        refitted = result.refit(X_train, y_train)
        preds = refitted.predict(X_test)
        assert len(preds) == len(X_test)

    def test_time_budget(self, classification_data):
        X_train, _, y_train, _ = classification_data
        result = ml.baseline(X_train, y_train,
                             time_budget=2, verbose=False)
        assert result is not None

    def test_task_classification(self, classification_data):
        X_train, _, y_train, _ = classification_data
        result = ml.baseline(X_train, y_train,
                             models=["LogisticRegression"],
                             task="classification", verbose=False)
        assert "classification" in result.task or result.task == "binary"


class TestEvalSuite:
    def test_evaluate_binary(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, verbose=False)
        assert isinstance(result, EvalResult)

    def test_metrics_have_accuracy(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, verbose=False)
        assert "accuracy" in result.metrics

    def test_metrics_have_roc_auc(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, verbose=False)
        assert "roc_auc" in result.metrics

    def test_confusion_matrix_shape(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, verbose=False)
        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)

    def test_threshold_analysis_binary(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test,
                             optimize_threshold=True, verbose=False)
        assert result.threshold_analysis is not None
        assert "threshold" in result.threshold_analysis.columns

    def test_regression_metrics(self, regression_data):
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = regression_data
        model = LinearRegression().fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, task="regression", verbose=False)
        assert "rmse" in result.metrics
        assert "r2" in result.metrics

    def test_to_html_returns_path(self, classification_data, tmp_path):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegression(random_state=42).fit(X_train, y_train)
        result = ml.evaluate(model, X_test, y_test, verbose=False)
        path = result.to_html(str(tmp_path / "eval.html"))
        import os
        assert os.path.exists(path)


class TestBalanceKit:
    def test_balance_returns_result(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, verbose=False)
        assert isinstance(result, BalanceResult)

    def test_severity_detected(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, verbose=False)
        assert result.severity in ("mild", "moderate", "severe", "critical", "none")

    def test_class_weights_populated(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, verbose=False)
        assert len(result.class_weights) == 2

    def test_resampled_shapes_match(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, verbose=False)
        assert len(result.X_resampled) == len(result.y_resampled)

    def test_class_weight_strategy(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, strategy="class_weight", verbose=False)
        # class_weight doesn't resample — returns original
        assert len(result.X_resampled) == len(X)

    def test_undo_via_original(self, imbalanced_data):
        X, y = imbalanced_data
        result = ml.balance(X, y, verbose=False)
        assert len(result.X_original) == len(X)
        assert len(result.y_original) == len(y)
