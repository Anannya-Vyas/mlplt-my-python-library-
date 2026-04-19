"""
mlpilot/utils/split.py
Stratified and random train/test split returning (X_train, X_test, y_train, y_test).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(
    data: Union[pd.DataFrame, "FeatureResult"],  # type: ignore[name-defined]
    target: Union[str, pd.Series, None] = None,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified or random train/test split.

    Parameters
    ----------
    data : pd.DataFrame or FeatureResult
        If a FeatureResult, uses result.df automatically.
    target : str or pd.Series
        Target column name (if data is DataFrame) or Series.
    test_size : float
        Fraction for test set (default 0.2).
    stratify : bool
        Whether to stratify split on target (classification).
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Accept FeatureResult or plain DataFrame
    if hasattr(data, "df"):
        df = data.df
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected DataFrame or FeatureResult, got {type(data)}")

    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found in DataFrame")
        y = df[target]
        X = df.drop(columns=[target])
    elif isinstance(target, (pd.Series, np.ndarray)):
        y = pd.Series(target)
        X = df
    elif target is None:
        # Try to infer — look for a column named 'target' or 'label' or 'y'
        guesses = [c for c in df.columns if c.lower() in ("target", "label", "y", "class")]
        if guesses:
            y = df[guesses[0]]
            X = df.drop(columns=[guesses[0]])
        else:
            raise ValueError(
                "Could not infer target column. Pass target='column_name' explicitly."
            )
    else:
        raise TypeError(f"target must be str, pd.Series, or None, got {type(target)}")

    stratify_arr = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arr, random_state=random_state
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
