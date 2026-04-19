"""
mlpilot.utils.imports
Universal, safe importing of data science stack.
"""

import os
import sys
from typing import Any, Dict

def get_ds_workspace(df: Any = None) -> Dict[str, Any]:
    """
    Returns a dictionary of the standard DS stack, safely imported.
    """
    workspace = {
        "os": os,
        "sys": sys,
        "df": df,
        "result": None,
    }

    # Standard DS stack
    try:
        import numpy as np
        workspace["np"] = np
    except ImportError: pass

    try:
        import pandas as pd
        workspace["pd"] = pd
    except ImportError: pass

    try:
        import matplotlib.pyplot as plt
        workspace["plt"] = plt
    except ImportError: pass

    try:
        import seaborn as sns
        workspace["sns"] = sns
    except ImportError: pass

    try:
        import scipy
        workspace["scipy"] = scipy
    except ImportError: pass

    try:
        import sklearn
        workspace["sklearn"] = sklearn
    except ImportError: pass

    try:
        import math
        workspace["math"] = math
    except ImportError: pass

    return workspace
