import os
import sys
import logging
import warnings
from typing import Any, List, Optional, Union

# ===========================================================================
# GLOBAL SILENCE GUARD (v1.1.5)
# Must execute before any other imports to catch early initialization noise.
# ===========================================================================
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Silence the specific HuggingFace Hub unauthenticated warning
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Suppress all technical logs from core AI backends
for logger_name in ["huggingface_hub", "transformers", "huggingface_hub.utils._http"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

"""
mlpilot — The Complete Python ML Library
One import. Every ML tool you need.

Usage:
    import mlpilot as ml

    eda    = ml.analyze(df, target='churn')
    clean  = ml.clean(df, target='churn')
    feats  = ml.features(clean.df, target='churn')
    board  = ml.baseline(X_train, y_train)
    tuned  = ml.tune('lgbm', X_train, y_train)
    exp    = ml.explain(tuned.best_model, X_train)
    api    = ml.deploy(tuned.best_model)
"""

from mlpilot.eda.analyzer import analyze
from mlpilot.clean.cleaner import clean
from mlpilot.clean.cleaner import CleaningRule
from mlpilot.validate.validator import validate, infer_schema
from mlpilot.features.forge import features
from mlpilot.train.baseline import baseline
from mlpilot.train.eval import evaluate
from mlpilot.train.hyperx import tune
from mlpilot.explain.explainer import explain
from mlpilot.balance.balancekit import balance
from mlpilot.utils.split import split
from mlpilot.utils.setup_guide import setup

# ---------------------------------------------------------------------------
# Phase 3+ modules — lazy imported to avoid hard dependency issues
# ---------------------------------------------------------------------------

def forecast(df, **kwargs):
    """TimeSense — multi-model time series forecasting."""
    from mlpilot.timeseries.timesense import forecast as _forecast
    return _forecast(df, **kwargs)


def text_classify(df, **kwargs):
    """TextML — NLP text classification."""
    from mlpilot.nlp.textml import text_classify as _tc
    return _tc(df, **kwargs)


def sentiment(series, **kwargs):
    """TextML — zero-shot sentiment analysis."""
    from mlpilot.nlp.textml import sentiment as _sent
    return _sent(series, **kwargs)


def topics(series, **kwargs):
    """TextML — topic modeling."""
    from mlpilot.nlp.textml import topics as _topics
    return _topics(series, **kwargs)


def embed(series, **kwargs):
    """TextML — text embeddings."""
    from mlpilot.nlp.textml import embed as _embed
    return _embed(series, **kwargs)


def deploy(model, **kwargs):
    """LaunchPad — generate FastAPI + Docker deployment."""
    from mlpilot.deploy.launchpad import deploy as _deploy
    return _deploy(model, **kwargs)


def session(name: str = "mlpilot_session", **kwargs):
    """PipelineRecorder — record all operations as a reproducible pipeline."""
    from mlpilot.pipeline.recorder import session as _session
    return _session(name=name, **kwargs)


def colab_setup(**kwargs):
    """setup_ollama — automatic Ollama setup for Google Colab."""
    from mlpilot.utils.colab import setup_ollama
    return setup_ollama(**kwargs)


# ---------------------------------------------------------------------------
# AI features — requires pip install mlpilot[ai] + ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

def analyst(df, **kwargs):
    """AIAnalyst — ask questions in plain English about your data."""
    from mlpilot.ai.analyst import AIAnalyst
    return AIAnalyst(df, **kwargs)


def audit(model, X, y, **kwargs):
    """MLAudit — bias, fairness, and model card generation."""
    from mlpilot.ai.audit import audit as _audit
    return _audit(model, X, y, **kwargs)


def story(results: Optional[List[Any]] = None, **kwargs):
    """DataStory — AI-written narrative reports."""
    from mlpilot.ai.story import DataStory
    s = DataStory(**kwargs)
    if results is not None:
        return s.tell(results)
    return s


def DataStory(engine: str = "auto", **kwargs):
    """Alias for ml.story() to match common user patterns."""
    return story(**kwargs, engine=engine)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

__version__ = "2.0.4"
__author__ = "Anannya Vyas"
__license__ = "MIT"

__all__ = [
    # Phase 1
    "analyze",
    "clean",
    "CleaningRule",
    "validate",
    "infer_schema",
    "features",
    "split",
    # Phase 2
    "baseline",
    "evaluate",
    "tune",
    "explain",
    "balance",
    # Phase 3
    "forecast",
    "text_classify",
    "sentiment",
    "topics",
    "embed",
    "deploy",
    "session",
    # Phase 4
    "analyst",
    "audit",
    "story",
    # Meta
    "setup",
    "colab_setup",
    "__version__",
]
