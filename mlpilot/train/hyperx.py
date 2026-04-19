"""
mlpilot/train/hyperx.py
HyperX — Bayesian hyperparameter tuning with time budget.
Uses Optuna if available, falls back to sklearn RandomizedSearchCV.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlpilot.utils.display import RichTable, print_step, print_success, print_warning, _is_legacy_windows
from mlpilot.utils.types import BaseResult

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

_SEARCH_SPACES: Dict[str, Dict] = {
    "LogisticRegression": {
        "C": ("float", 1e-4, 100, "log"),
        "max_iter": ("int", 100, 2000),
    },
    "RandomForestClassifier": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "RandomForestRegressor": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
    },
    "GradientBoostingClassifier": {
        "n_estimators": ("int", 50, 400),
        "max_depth": ("int", 2, 8),
        "learning_rate": ("float", 0.01, 0.3, "log"),
        "subsample": ("float", 0.5, 1.0),
    },
    "GradientBoostingRegressor": {
        "n_estimators": ("int", 50, 400),
        "max_depth": ("int", 2, 8),
        "learning_rate": ("float", 0.01, 0.3, "log"),
    },
    "XGBClassifier": {
        "n_estimators": ("int", 50, 600),
        "max_depth": ("int", 2, 10),
        "learning_rate": ("float", 0.005, 0.5, "log"),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("float", 1e-5, 10, "log"),
        "reg_lambda": ("float", 1e-5, 10, "log"),
    },
    "XGBRegressor": {
        "n_estimators": ("int", 50, 600),
        "max_depth": ("int", 2, 10),
        "learning_rate": ("float", 0.005, 0.5, "log"),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
    "LGBMClassifier": {
        "n_estimators": ("int", 50, 600),
        "max_depth": ("int", 2, 12),
        "learning_rate": ("float", 0.005, 0.5, "log"),
        "num_leaves": ("int", 15, 200),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("float", 1e-5, 10, "log"),
        "reg_lambda": ("float", 1e-5, 10, "log"),
    },
    "LGBMRegressor": {
        "n_estimators": ("int", 50, 600),
        "max_depth": ("int", 2, 12),
        "learning_rate": ("float", 0.005, 0.5, "log"),
        "num_leaves": ("int", 15, 200),
        "subsample": ("float", 0.5, 1.0),
    },
}

_MODEL_ALIASES = {
    "xgb": "XGBClassifier",
    "xgboost": "XGBClassifier",
    "lgbm": "LGBMClassifier",
    "lightgbm": "LGBMClassifier",
    "rf": "RandomForestClassifier",
    "random_forest": "RandomForestClassifier",
    "lr": "LogisticRegression",
    "logistic": "LogisticRegression",
    "gb": "GradientBoostingClassifier",
    "xgb_reg": "XGBRegressor",
    "lgbm_reg": "LGBMRegressor",
    "rf_reg": "RandomForestRegressor",
    "gb_reg": "GradientBoostingRegressor",
}


# ---------------------------------------------------------------------------
# TuningResult
# ---------------------------------------------------------------------------

class TuningResult(BaseResult):
    def __init__(
        self,
        best_params: Dict,
        best_score: float,
        best_model: Any,
        model_name: str,
        metric: str,
        trials_df: Optional[pd.DataFrame] = None,
        sensitivity: Optional[Dict[str, float]] = None,
    ):
        self.best_params = best_params
        self.best_score = best_score
        self.best_model = best_model
        self.model_name = model_name
        self.metric = metric
        self.trials_df = trials_df
        self.sensitivity = sensitivity

    def print_report(self) -> None:
        title = f"HyperX Results — {self.model_name}"
        if not _is_legacy_windows():
            title = "-> " + title
        tbl = RichTable(title=title, columns=["Parameter", "Best Value"])
        for k, v in self.best_params.items():
            tbl.add_row(k, f"{v:.6g}" if isinstance(v, float) else str(v))
        tbl.print()
        print_success(f"Best {self.metric}: {self.best_score:.4f}")

    def plot_optimization_history(self) -> Any:
        """Plot convergence of optimization over trials."""
        if self.trials_df is None:
            print_warning("No trials data available")
            return None
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(
                x=list(range(len(self.trials_df))),
                y=self.trials_df.get("value", self.trials_df.iloc[:, 0]),
                mode="lines+markers",
                marker_color="#6c63ff",
                line_color="#6c63ff",
            ))
            fig.update_layout(title="Optimization History", template="plotly_dark",
                             paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                             font_color="#e2e8f0",
                             xaxis_title="Trial", yaxis_title=self.metric)
            return fig
        except Exception:
            return None

    def __repr__(self) -> str:
        return (f"TuningResult(model={self.model_name}, "
                f"best_score={self.best_score:.4f}, "
                f"n_trials={len(self.trials_df) if self.trials_df is not None else 0})")


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def _tune_optuna(
    model_cls: Any,
    model_name: str,
    space: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str,
    cv: int,
    n_trials: int,
    time_budget: Optional[int],
    direction: str,
    n_jobs: int,
) -> TuningResult:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import cross_val_score

    trial_records = []

    def objective(trial):
        params = {}
        for pname, spec in space.items():
            if spec[0] == "int":
                params[pname] = trial.suggest_int(pname, spec[1], spec[2])
            elif spec[0] == "float":
                log = len(spec) > 3 and spec[3] == "log"
                params[pname] = trial.suggest_float(pname, spec[1], spec[2], log=log)
            elif spec[0] == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec[1])

        model = model_cls(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring=metric, n_jobs=n_jobs)
        score = float(np.mean(scores))
        trial_records.append({"trial": trial.number, "value": score, **params})
        return score

    study = optuna.create_study(direction=direction)
    timeout = time_budget if time_budget else None

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    # Fit best model on full training set
    best_model = model_cls(**best_params)
    best_model.fit(X_train, y_train)

    # Sensitivity: param importance
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}

    trials_df = pd.DataFrame(trial_records)

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_model=best_model,
        model_name=model_name,
        metric=metric,
        trials_df=trials_df,
        sensitivity=importance,
    )


# ---------------------------------------------------------------------------
# Fallback: RandomizedSearchCV
# ---------------------------------------------------------------------------

def _tune_randomized(
    model_cls: Any,
    model_name: str,
    space: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str,
    cv: int,
    n_trials: int,
) -> TuningResult:
    from scipy.stats import loguniform, randint, uniform
    from sklearn.model_selection import RandomizedSearchCV

    scipy_space = {}
    for pname, spec in space.items():
        if spec[0] == "int":
            scipy_space[pname] = randint(spec[1], spec[2])
        elif spec[0] == "float":
            if len(spec) > 3 and spec[3] == "log":
                scipy_space[pname] = loguniform(spec[1], spec[2])
            else:
                scipy_space[pname] = uniform(spec[1], spec[2] - spec[1])
        elif spec[0] == "categorical":
            scipy_space[pname] = spec[1]

    search = RandomizedSearchCV(
        model_cls(), scipy_space,
        n_iter=n_trials, cv=cv, scoring=metric,
        n_jobs=-1, random_state=42, refit=True, error_score="raise"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    trials_rows = []
    for i, (params, score) in enumerate(
        zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
    ):
        trials_rows.append({"trial": i, "value": score, **params})

    return TuningResult(
        best_params=search.best_params_,
        best_score=search.best_score_,
        best_model=search.best_estimator_,
        model_name=model_name,
        metric=metric,
        trials_df=pd.DataFrame(trials_rows),
        sensitivity={},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune(
    model: Union[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_budget: Optional[int] = 300,
    n_trials: Optional[int] = None,
    metric: str = "auto",
    cv: int = 5,
    direction: str = "maximize",
    search_space: Optional[Dict] = None,
    n_jobs: int = -1,
    early_stopping_rounds: int = 20,
    verbose: bool = True,
) -> TuningResult:
    """
    HyperX — Intelligent Hyperparameter Tuning.

    Bayesian optimization (Optuna) with automatic search space definition.
    Falls back to RandomizedSearchCV if optuna is not installed.

    Parameters
    ----------
    model : str or sklearn estimator
        Model name string ('xgboost', 'lgbm', 'rf', etc.) or fitted estimator.
    X_train, y_train : training data
    time_budget : int, optional
        Seconds to tune (default 300 = 5 min). Alternative to n_trials.
    n_trials : int, optional
        Maximum number of trials.
    metric : str
        Scoring metric. 'auto' selects based on task.
    cv : int
        Cross-validation folds (default 5).
    direction : str
        'maximize' or 'minimize'.
    search_space : dict, optional
        Override auto search space.
    n_jobs : int
        Parallel jobs (default -1).
    verbose : bool

    Returns
    -------
    TuningResult
        Attributes: best_params, best_score, best_model, trials_df, sensitivity.
        Methods: print_report(), plot_optimization_history().

    Examples
    --------
    >>> import mlpilot as ml
    >>> result = ml.tune('lgbm', X_train, y_train, time_budget=300, metric='roc_auc')
    >>> print(result.best_params)
    >>> final_model = result.best_model
    """
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train

    # -----------------------------------------------------------------------
    # CATEGORICAL GUARD (v1.2.0): Auto-encode if strings found
    # -----------------------------------------------------------------------
    if X_train.select_dtypes(include=["object", "category"]).columns.any():
        if verbose:
            print_warning("Categorical data detected in features. Applying Autonomous Encoding...")
        from mlpilot.features.forge import features as _feats
        hardened_result = _feats(X_train, target=None, verbose=False)
        X_train = hardened_result.fit_transform(X_train)

    # Resolve model
    if isinstance(model, str):
        model_name = _MODEL_ALIASES.get(model.lower(), model)
        # Auto-detect regression if needed
        n_unique = y_train.nunique()
        is_regression = pd.api.types.is_float_dtype(y_train) and n_unique > 20
        if is_regression and "Classifier" in model_name:
            model_name = model_name.replace("Classifier", "Regressor")
        model_cls = _resolve_model_class(model_name)
    else:
        model_cls = type(model)
        model_name = type(model).__name__

    if model_cls is None:
        raise ValueError(f"Unknown model: {model}. Try 'xgboost', 'lgbm', 'rf', etc.")

    # Auto metric
    if metric == "auto":
        n_unique = y_train.nunique()
        is_regression = pd.api.types.is_float_dtype(y_train) and n_unique > 20
        if is_regression:
            metric = "neg_root_mean_squared_error"
            direction = "maximize"
        elif n_unique == 2:
            metric = "roc_auc"
        else:
            metric = "f1_macro"

    space = search_space or _SEARCH_SPACES.get(model_name, {})
    if not space:
        print_warning(f"No auto search space for {model_name} — using defaults with n_trials=20")
        n_trials = n_trials or 20

    n_trials = n_trials or 100

    if verbose:
        budget_str = f"{time_budget}s" if time_budget else f"{n_trials} trials"
        print_step(f"HyperX: {model_name} — {budget_str}, metric={metric}")

    try:
        import optuna  # noqa: F401
        result = _tune_optuna(
            model_cls, model_name, space, X_train, y_train,
            metric, cv, n_trials, time_budget, direction, n_jobs
        )
    except ImportError:
        print_warning("optuna not installed — using RandomizedSearchCV fallback. "
                      "Install with: pip install mlpilot[optuna]")
        result = _tune_randomized(
            model_cls, model_name, space, X_train, y_train, metric, cv, n_trials
        )

    if verbose:
        result.print_report()

    return result


def _resolve_model_class(name: str):
    class_map = {
        "LogisticRegression": "sklearn.linear_model",
        "RandomForestClassifier": "sklearn.ensemble",
        "RandomForestRegressor": "sklearn.ensemble",
        "GradientBoostingClassifier": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "XGBClassifier": "xgboost",
        "XGBRegressor": "xgboost",
        "LGBMClassifier": "lightgbm",
        "LGBMRegressor": "lightgbm",
    }
    module = class_map.get(name)
    if module is None:
        return None
    try:
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None
