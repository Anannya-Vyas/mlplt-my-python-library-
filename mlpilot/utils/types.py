"""
mlpilot/utils/types.py
Shared data structures and base classes used across all mlpilot modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Base Result
# ---------------------------------------------------------------------------

class BaseResult:
    """
    Base class for all mlpilot result objects.
    Provides common utility methods: to_html(), to_pdf(), repr formatting.
    """

    def to_html(self, path: Optional[str] = None) -> str:
        """Export result as an HTML report. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement to_html()")

    def to_pdf(self, path: Optional[str] = None) -> None:
        """Export result as a PDF report. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement to_pdf()")

    def __repr__(self) -> str:
        attrs = [a for a in dir(self) if not a.startswith("_") and not callable(getattr(self, a))]
        return f"{self.__class__.__name__}({', '.join(attrs[:5])}{'...' if len(attrs) > 5 else ''})"


# ---------------------------------------------------------------------------
# Change Tracking
# ---------------------------------------------------------------------------

@dataclass
class Change:
    """Represents a single transformation applied to a dataframe."""
    column: str
    action: str                  # 'impute', 'clip', 'drop', 'cast', 'encode', etc.
    detail: str                  # human-readable description
    n_affected: int = 0          # rows/cells affected
    before_value: Any = None     # sample value before
    after_value: Any = None      # sample value after

    def __str__(self) -> str:
        return f"[{self.column}] {self.action}: {self.detail} ({self.n_affected} cells)"


# ---------------------------------------------------------------------------
# Data Issues & Recommendations
# ---------------------------------------------------------------------------

@dataclass
class DataIssue:
    """A data quality problem flagged by SmartEDA or DataValidator."""
    severity: str          # 'critical', 'warning', 'info'
    column: Optional[str]  # None = dataset-level issue
    code: str              # machine-readable code: 'high_missing', 'constant_col', etc.
    message: str           # human-readable description
    value: Any = None      # the problematic value / statistic

    def __str__(self) -> str:
        loc = f"[{self.column}] " if self.column else ""
        return f"[{self.severity.upper()}] {loc}{self.message}"


@dataclass
class Recommendation:
    """A suggested next action produced by SmartEDA."""
    priority: int          # 1 = highest priority
    action: str            # short label: 'impute', 'drop_column', 'encode', etc.
    column: Optional[str]
    reason: str
    suggested_code: str = ""   # Python snippet that would address this

    def __str__(self) -> str:
        loc = f" on '{self.column}'" if self.column else ""
        return f"[P{self.priority}] {self.action}{loc}: {self.reason}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A validation problem found by DataValidator."""
    severity: str            # 'critical', 'warning'
    check: str               # which check triggered this
    column: Optional[str]
    message: str
    statistic: Any = None    # e.g., p-value, % missing

    def is_critical(self) -> bool:
        return self.severity == "critical"

    def __str__(self) -> str:
        loc = f"[{self.column}] " if self.column else ""
        return f"[{self.severity.upper()}] {loc}{self.message}"


# ---------------------------------------------------------------------------
# Column Profile (EDA per-column statistics)
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Complete statistical profile for one column."""
    name: str
    dtype: str                           # pandas dtype string
    dtype_label: str                     # 'numeric' | 'categorical' | 'datetime' | 'text' | 'boolean'
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    cardinality: str = "low"             # 'low' | 'medium' | 'high' | 'unique'

    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_outliers_iqr: int = 0
    n_outliers_zscore: int = 0

    # Categorical stats
    top_values: List[tuple] = field(default_factory=list)   # [(value, count), ...]
    mode: Any = None

    # Recommendations
    recommended_transform: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.dtype_label == "numeric"

    @property
    def is_categorical(self) -> bool:
        return self.dtype_label in ("categorical", "boolean")

    @property
    def is_datetime(self) -> bool:
        return self.dtype_label == "datetime"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------------------------

@dataclass
class DatasetSummary:
    """High-level summary of the dataset."""
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    n_boolean: int
    n_missing_cells: int
    pct_missing_cells: float
    n_duplicate_rows: int
    memory_mb: float
    dtypes_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return (self.n_rows, self.n_cols)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple context-manager timer."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"
