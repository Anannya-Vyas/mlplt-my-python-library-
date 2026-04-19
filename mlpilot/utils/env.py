"""
mlpilot/utils/env.py
Environment detection and configuration helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    return "google.colab" in sys.modules


def is_notebook() -> bool:
    """Detect if running in any Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except (NameError, ImportError):
        return False


def get_platform_info() -> dict:
    import platform
    return {
        "os": platform.system(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "is_colab": is_colab(),
        "is_notebook": is_notebook(),
    }


def get_mlpilot_root() -> Path:
    """Get the absolute path to the mlpilot package root."""
    return Path(__file__).parent.parent.resolve()
