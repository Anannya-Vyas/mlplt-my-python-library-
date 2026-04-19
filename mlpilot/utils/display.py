"""
mlpilot/utils/display.py
Rich terminal output helpers used across all mlpilot modules.
Windows-safe: handles emoji encoding errors gracefully.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence


def _is_notebook() -> bool:
    """Detect whether running inside a Jupyter/IPython notebook."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except NameError:
        return False


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _is_legacy_windows() -> bool:
    """Detect Windows terminals with limited encoding (cp1252, etc)."""
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    return enc.lower() in ("cp1252", "cp850", "cp437", "ascii")


def _safe_console():
    """Return a Rich Console that handles Windows legacy encoding safely."""
    from rich.console import Console
    if _is_legacy_windows():
        # force_terminal=False makes Rich not use the legacy win32 renderer
        return Console(
            highlight=False,
            markup=True,
            safe_box=True,
            force_terminal=False,
            no_color=False,
        )
    return Console(highlight=False, markup=True, safe_box=True)


def _clean_text(text: str) -> str:
    """Remove non-ASCII characters if on a legacy Windows terminal."""
    if _is_legacy_windows():
        # Strip characters that would crash cp1252/ascii terminals
        return "".join(c for c in text if ord(c) < 128)
    return text


def _safe_print_rich(text: str, style: str = "") -> None:
    """Print using Rich, falling back to plain ASCII on Windows encoding errors."""
    clean_text = _clean_text(text)
    try:
        _safe_console().print(clean_text, style=style)
    except Exception:
        # Final safety net
        try:
            print(clean_text.encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

MLPILOT_LOGO = " mlpilot — The Complete Python ML Library"


def print_banner(subtitle: str = "The Complete Python ML Library") -> None:
    """Print the mlpilot banner."""
    if _rich_available():
        _safe_print_rich(f"\n  [bold cyan]{MLPILOT_LOGO}[/bold cyan]")
        _safe_print_rich(f"  [dim]{subtitle}[/dim]\n")
    else:
        print(f"\n  {MLPILOT_LOGO}")
        print(f"  {subtitle}\n")


# ---------------------------------------------------------------------------
# Status / progress messages
# ---------------------------------------------------------------------------

def print_step(message: str, emoji: str = "->") -> None:
    """Print a single progress step."""
    msg = _clean_text(message)
    prefix = _clean_text(emoji)
    if _rich_available():
        _safe_print_rich(f"  {prefix} {msg}", style="cyan")
    else:
        print(f"  {prefix} {msg}")


def print_success(message: str) -> None:
    msg = _clean_text(message)
    if _rich_available():
        _safe_print_rich(f"  [OK] {msg}", style="bold green")
    else:
        print(f"  [OK] {msg}")


def print_warning(message: str) -> None:
    msg = _clean_text(message)
    if _rich_available():
        _safe_print_rich(f"  [WARN] {msg}", style="bold yellow")
    else:
        print(f"  WARNING: {msg}")


def print_error(message: str) -> None:
    msg = _clean_text(message)
    if _rich_available():
        _safe_print_rich(f"  [ERROR] {msg}", style="bold red")
    else:
        print(f"  ERROR: {msg}", file=sys.stderr)


def print_line(length: int = 50) -> None:
    """Print a horizontal line, using safe ASCII on legacy Windows."""
    char = "-" if _is_legacy_windows() else "─"
    if _rich_available():
        _safe_print_rich(char * length, style="dim")
    else:
        print(char * length)


# ---------------------------------------------------------------------------
# Rich table builder
# ---------------------------------------------------------------------------

class RichTable:
    """
    Wrapper around rich.Table that falls back to plain-text printing
    when rich is not installed.
    """

    def __init__(self, title: str = "", columns: Sequence[str] = ()):
        self.title = title
        self.columns = list(columns)
        self.rows: List[List[str]] = []
        self._column_styles: Dict[int, str] = {}

    def add_column(self, name: str, style: str = "") -> None:
        self.columns.append(name)
        if style:
            self._column_styles[len(self.columns) - 1] = style

    def add_row(self, *values: Any) -> None:
        self.rows.append([str(v) for v in values])

    def print(self) -> None:
        if _rich_available():
            self._print_rich()
        else:
            self._print_plain()

    def _print_rich(self) -> None:
        from rich.table import Table

        table = Table(title=self.title, show_header=True, header_style="bold magenta",
                      safe_box=True)
        for i, col in enumerate(self.columns):
            style = self._column_styles.get(i, "")
            table.add_column(col, style=style)
        for row in self.rows:
            table.add_row(*row)
        try:
            _safe_console().print(table)
        except (UnicodeEncodeError, UnicodeDecodeError):
            self._print_plain()
        except Exception:
            self._print_plain()

    def _print_plain(self) -> None:
        if self.title:
            print(f"\n{self.title}")
            print("=" * max(len(self.title), 40))
        widths = [len(c) for c in self.columns]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*self.columns))
        print("-" * (sum(widths) + 2 * len(widths)))
        for row in self.rows:
            padded = row + [""] * (len(self.columns) - len(row))
            print(fmt.format(*padded[:len(self.columns)]))


# ---------------------------------------------------------------------------
# Progress bar (simple, no dependency)
# ---------------------------------------------------------------------------

class ProgressBar:
    """Minimal progress bar that works in terminal and notebooks."""

    def __init__(self, total: int, description: str = "", verbose: bool = True):
        self.total = total
        self.description = description
        self.verbose = verbose
        self._current = 0
        self._bar = None

        if verbose and _rich_available():
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            self._task = None

    def __enter__(self) -> "ProgressBar":
        if self.verbose and _rich_available():
            self._progress.__enter__()
            self._task = self._progress.add_task(self.description, total=self.total)
        elif self.verbose:
            print(f"  {self.description}...")
        return self

    def __exit__(self, *args) -> None:
        if self.verbose and _rich_available():
            self._progress.__exit__(*args)

    def update(self, n: int = 1, description: Optional[str] = None) -> None:
        self._current += n
        if self.verbose and _rich_available() and self._task is not None:
            self._progress.update(self._task, advance=n,
                                  description=description or self.description)
