#!/usr/bin/env python3
# scripts/helpers/data_helpers.py

from typing import List, Optional, Union

import numpy as np
import pandas as pd

def is_finite(x: object) -> bool:
    """
    Check if a value is finite (not NaN, inf, or non-numeric).
    """
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division, returning 0.0 where den is zero."""
    return num.div(den.replace(0, np.nan)).fillna(0.0)

def rate(num: float, den: float) -> float:
    """
    Compute a rate (num / den), returning NaN for invalid denominators.
    """
    return (num / den) if den and den > 0 else float("nan")

def format_num(value: float, decimals: Optional[int]) -> Union[float, str]:
    """
    Format a numeric value with optional decimal places.
    """
    if decimals is None:
        return value
    return f"{value:.{decimals}f}"

def round_numeric_df(df: pd.DataFrame, decimals: int, *, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Round all numeric columns in a DataFrame to specified decimal places.
    """
    exclude = exclude or []
    num_cols = [
        c for c in df.columns
        if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if num_cols:
        df[num_cols] = df[num_cols].round(decimals)
    return df
