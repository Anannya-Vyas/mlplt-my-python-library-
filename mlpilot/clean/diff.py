"""
mlpilot/clean/diff.py
Change tracking for AutoCleaner — builds the diff/report between original and cleaned df.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlpilot.utils.display import RichTable
from mlpilot.utils.types import Change


@dataclass
class ColumnDiff:
    """Before/after comparison for a single column."""
    column: str
    action: str
    detail: str
    n_missing_before: int = 0
    n_missing_after: int = 0
    dtype_before: str = ""
    dtype_after: str = ""
    n_outliers_before: int = 0
    n_outliers_after: int = 0
    n_affected: int = 0

    def was_changed(self) -> bool:
        return self.n_affected > 0 or self.dtype_before != self.dtype_after


class CleaningReport:
    """
    Complete report of everything AutoCleaner did.
    Printable via .print() for rich terminal output.
    """

    def __init__(self, changes: List[Dict], original_df, cleaned_df):
        self.raw_changes = changes
        self._changes = self._build_changes(changes)
        self.original_shape = original_df.shape
        self.cleaned_shape = cleaned_df.shape
        self.n_nulls_filled = sum(
            c.n_affected for c in self._changes if c.action == "impute"
        )
        self.n_outliers_handled = sum(
            c.n_affected for c in self._changes if "outlier" in c.action
        )
        self.n_dtypes_fixed = sum(
            1 for c in self._changes if c.action == "cast_dtype"
        )
        self.n_duplicates_removed = next(
            (c.n_affected for c in self._changes if c.action == "remove_duplicates"), 0
        )

    def _build_changes(self, raw: List[Dict]) -> List[ColumnDiff]:
        result = []
        for r in raw:
            result.append(ColumnDiff(
                column=r.get("column", "—"),
                action=r.get("action", "unknown"),
                detail=r.get("detail", ""),
                n_affected=r.get("n_affected", 0),
                dtype_before=r.get("dtype_before", ""),
                dtype_after=r.get("dtype_after", ""),
                n_missing_before=r.get("n_missing_before", 0),
                n_missing_after=r.get("n_missing_after", 0),
            ))
        return result

    @property
    def changes(self) -> List[ColumnDiff]:
        return self._changes

    def print(self) -> None:
        """Print a rich formatted cleaning report to the terminal."""
        summary_tbl = RichTable(title="Cleaning Summary", columns=["Metric", "Value"])
        summary_tbl.add_row("Original shape", f"{self.original_shape[0]:,} × {self.original_shape[1]}")
        summary_tbl.add_row("Cleaned shape", f"{self.cleaned_shape[0]:,} × {self.cleaned_shape[1]}")
        summary_tbl.add_row("Nulls filled", str(self.n_nulls_filled))
        summary_tbl.add_row("Outliers handled", str(self.n_outliers_handled))
        summary_tbl.add_row("Dtypes fixed", str(self.n_dtypes_fixed))
        summary_tbl.add_row("Duplicates removed", str(self.n_duplicates_removed))
        summary_tbl.print()

        if self._changes:
            changes_tbl = RichTable(
                title="Changes Applied",
                columns=["Column", "Action", "Detail", "Cells Affected"]
            )
            for c in self._changes:
                changes_tbl.add_row(
                    c.column, c.action, c.detail, str(c.n_affected) if c.n_affected else "—"
                )
            changes_tbl.print()
        else:
            print("  ✓ No changes needed — dataset was already clean")

    def to_dict(self) -> Dict:
        return {
            "original_shape": self.original_shape,
            "cleaned_shape": self.cleaned_shape,
            "n_nulls_filled": self.n_nulls_filled,
            "n_outliers_handled": self.n_outliers_handled,
            "n_dtypes_fixed": self.n_dtypes_fixed,
            "n_duplicates_removed": self.n_duplicates_removed,
            "changes": [c.__dict__ for c in self._changes],
        }
