"""
mlpilot/train/baseline.py
BaselineBlitz — run 15+ models in one call and get a ranked leaderboard.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from mlpilot.utils.display import RichTable, print_step, print_success, print_warning, _is_legacy_windows
from mlpilot.utils.types import BaseResult

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _get_models(task: str) -> List[Tuple[str, Any]]:
    models = []

    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import (
        Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    if task in ("classification", "binary", "multiclass"):
        models = [
            ("DummyClassifier", DummyClassifier(strategy="most_frequent")),
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
            ("RidgeClassifier", RidgeClassifier()),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42, max_depth=8)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
            ("KNeighborsClassifier", KNeighborsClassifier(n_jobs=-1)),
            ("GaussianNB", GaussianNB()),
            ("SVC", SVC(probability=True, random_state=42)),
            ("MLPClassifier", MLPClassifier(max_iter=200, random_state=42)),
        ]
        # Optional extras
        try:
            from xgboost import XGBClassifier
            models.append(("XGBClassifier", XGBClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                eval_metric="logloss", use_label_encoder=False, verbosity=0
            )))
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier
            models.append(("LGBMClassifier", LGBMClassifier(
                n_estimators=200, random_state=42, n_jobs=-1, verbose=-1
            )))
        except ImportError:
            pass

    else:  # regression
        models = [
            ("DummyRegressor", DummyRegressor(strategy="mean")),
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge()),
            ("Lasso", Lasso(max_iter=2000)),
            ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42, max_depth=8)),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42)),
            ("KNeighborsRegressor", KNeighborsRegressor(n_jobs=-1)),
            ("SVR", SVR()),
            ("MLPRegressor", MLPRegressor(max_iter=200, random_state=42)),
        ]
        try:
            from xgboost import XGBRegressor
            models.append(("XGBRegressor", XGBRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, verbosity=0
            )))
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            models.append(("LGBMRegressor", LGBMRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, verbose=-1
            )))
        except ImportError:
            pass

    return models


def _detect_task(y: pd.Series) -> str:
    n_unique = y.nunique()
    if pd.api.types.is_float_dtype(y) and n_unique > 20:
        return "regression"
    if n_unique == 2:
        return "binary"
    return "multiclass"


def _default_metric(task: str) -> str:
    if task == "binary":
        return "roc_auc"
    if task == "multiclass":
        return "f1_macro"
    return "neg_root_mean_squared_error"


def _extra_metrics(task: str, model: Any, X_test, y_test) -> Dict[str, float]:
    metrics = {}
    try:
        from sklearn import metrics as skm
        y_pred = model.predict(X_test)
        if task in ("binary", "multiclass"):
            metrics["accuracy"] = round(float(skm.accuracy_score(y_test, y_pred)), 4)
            avg = "binary" if task == "binary" else "macro"
            metrics["f1"] = round(float(skm.f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            if hasattr(model, "predict_proba") and task == "binary":
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(skm.roc_auc_score(y_test, y_prob)), 4)
        else:
            metrics["rmse"] = round(float(np.sqrt(skm.mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(skm.mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2"] = round(float(skm.r2_score(y_test, y_pred)), 4)
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class Leaderboard:
    def __init__(self, rows: List[Dict], task: str):
        self._rows = sorted(rows, key=lambda r: -r.get("cv_score", 0))
        self.task = task

    def __len__(self) -> int:
        return len(self._rows)

    def print(self) -> None:
        col_names = ["Rank", "Model", "CV Score", "Train Time"]
        if self._rows and "roc_auc" in self._rows[0]:
            col_names = ["Rank", "Model", "CV Score", "ROC-AUC", "F1", "Train Time"]
        elif self._rows and "r2" in self._rows[0]:
            col_names = ["Rank", "Model", "CV Score", "RMSE", "R²", "Train Time"]

        title = "Baseline Leaderboard"
        if not _is_legacy_windows():
            title = "-> " + title
        tbl = RichTable(title=title, columns=col_names)
        for i, row in enumerate(self._rows, 1):
            base_vals = [str(i), row["model"], f"{row['cv_score']:.4f}", f"{row['train_time']:.1f}s"]
            if "roc_auc" in row:
                base_vals = [str(i), row["model"], f"{row['cv_score']:.4f}",
                             f"{row.get('roc_auc', '—')}", f"{row.get('f1', '—')}",
                             f"{row['train_time']:.1f}s"]
            elif "r2" in row:
                base_vals = [str(i), row["model"], f"{row['cv_score']:.4f}",
                             f"{row.get('rmse', '—')}", f"{row.get('r2', '—')}",
                             f"{row['train_time']:.1f}s"]
            tbl.add_row(*base_vals)
        tbl.print()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------

class BaselineResult(BaseResult):
    def __init__(
        self,
        leaderboard: Leaderboard,
        best_model: Any,
        best_name: str,
        best_score: float,
        task: str,
        metric: str,
        all_models: Dict[str, Any],
        results: List[Dict],
    ):
        self.leaderboard = leaderboard
        self.best_model = best_model
        self.best_name = best_name
        self.best_score = best_score
        self.task = task
        self.metric = metric
        self._all_models = all_models
        self._results = results

    @property
    def best_model_name(self) -> str:
        """Alias for best_name used in some scripts."""
        return self.best_name

    def __len__(self) -> int:
        """Return the number of models in the leaderboard."""
        return len(self.leaderboard)

    def refit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Refit the best model on the full data."""
        print_step(f"Refitting {self.best_name} on full dataset...", "🔁")
        model = self._all_models[self.best_name]
        model.fit(X, y)
        print_success(f"Refitted {self.best_name}")
        return model

    def get_model(self, name: str) -> Optional[Any]:
        return self._all_models.get(name)

    def __repr__(self) -> str:
        return (f"BaselineResult(best={self.best_name}, "
                f"score={self.best_score:.4f}, task={self.task})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def baseline(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    task: str = "auto",
    cv: int = 5,
    metric: str = "auto",
    models: Optional[List[str]] = None,
    time_budget: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> BaselineResult:
    """
    BaselineBlitz — Multi-Model Comparison.

    Run 15+ models in one call. Get a ranked leaderboard in under 2 minutes.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    X_test, y_test : optional
        If provided, evaluates each model on the test set.
    task : str
        'auto' | 'classification' | 'regression' | 'multiclass'
    cv : int
        Number of cross-validation folds.
    metric : str
        'auto' selects roc_auc for binary, f1_macro for multiclass, rmse for regression.
    models : list[str], optional
        Subset of model names to run.
    time_budget : int, optional
        Seconds — stop adding models after this time.
    n_jobs : int
        Parallel jobs (default -1 = all cores).
    verbose : bool

    Returns
    -------
    BaselineResult
        Attributes: leaderboard, best_model, best_name, best_score, task, metric.
        Methods: refit(X, y).

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.baseline(X_train, y_train, X_test=X_test, y_test=y_test)
    >>> result.leaderboard.print()
    >>> best = result.best_model
    """
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train

    if task == "auto":
        task = _detect_task(y_train)

    # -----------------------------------------------------------------------
    # CATEGORICAL GUARD (v1.2.0): Auto-encode if strings found
    # -----------------------------------------------------------------------
    if X_train.select_dtypes(include=["object", "category"]).columns.any():
        if verbose:
            print_warning("Categorical data detected in features. Applying Autonomous Encoding...")
        from mlpilot.clean.cleaner import clean as _clean
        from mlpilot.features.forge import features as _feats
        
        # We use a silent clean/feature pass to harden the data
        hardened_result = _feats(X_train, target=None, verbose=False)
        X_train = hardened_result.fit_transform(X_train)
        
        # Sync X_test if provided
        if X_test is not None:
             X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
             X_test = hardened_result.transform(X_test_df)

    if metric == "auto":
        metric = _default_metric(task)

    if verbose:
        print_step(f"BaselineBlitz — task: {task}, metric: {metric}, cv={cv}", "⚡")

    all_candidate_models = _get_models(task)
    if models:
        all_candidate_models = [(n, m) for n, m in all_candidate_models if n in models]

    from sklearn.model_selection import cross_val_score

    results = []
    fitted_models: Dict[str, Any] = {}
    deadline = time.time() + time_budget if time_budget else None

    for name, model in all_candidate_models:
        if deadline and time.time() > deadline:
            if verbose:
                print_warning(f"Time budget reached — stopping after {len(results)} models")
            break

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=metric, n_jobs=n_jobs
                )
            cv_score = float(np.mean(scores))
            elapsed = time.time() - t0

            row = {"model": name, "cv_score": cv_score, "cv_std": float(np.std(scores)),
                   "train_time": elapsed}

            # Fit on full training set for test evaluation
            model.fit(X_train, y_train)
            fitted_models[name] = model

            if X_test is not None and y_test is not None:
                extra = _extra_metrics(task, model,
                                       pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test,
                                       y_test)
                row.update(extra)

            results.append(row)

            if verbose:
                sign = "+" if "+" not in metric else ""
                score_str = f"{cv_score:.4f}" if cv_score >= 0 else f"{cv_score:.4f} (neg metric)"
                print_step(f"{name:<40} CV={score_str}  {elapsed:.1f}s", "  ")

        except Exception as e:
            if verbose:
                print_warning(f"  {name} failed: {e}")
            continue

    if not results:
        raise RuntimeError("All models failed — check your data format")

    # Sort by CV score (handle negative metrics)
    results_sorted = sorted(results, key=lambda r: r["cv_score"], reverse=True)
    best_row = results_sorted[0]
    best_name = best_row["model"]
    best_score = best_row["cv_score"]
    best_model = fitted_models[best_name]

    board = Leaderboard(results, task)
    if verbose:
        board.print()
        print_success(f"Best: {best_name} (CV={best_score:.4f})")

    return BaselineResult(
        leaderboard=board,
        best_model=best_model,
        best_name=best_name,
        best_score=best_score,
        task=task,
        metric=metric,
        all_models=fitted_models,
        results=results,
    )
