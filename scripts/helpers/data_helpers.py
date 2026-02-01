#!/usr/bin/env python3
# scripts/helpers/data_helpers.py
"""
Shared data processing helper functions for kwok_workload_once and kwok_trace_replayer.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd


def is_finite(x: object) -> bool:
    """
    Check if a value is finite (not NaN, inf, or non-numeric).

    Args:
        x: The value to check.

    Returns:
        True if the value can be converted to a finite float.
    """
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """
    Safely divide two pandas Series, returning 0.0 for division by zero.

    Args:
        num: Numerator Series.
        den: Denominator Series.

    Returns:
        Result of division, with 0.0 where denominator is 0 or NaN.
    """
    return num.div(den.replace(0, np.nan)).fillna(0.0)


def rate(num: float, den: float) -> float:
    """
    Compute a rate (num / den), returning NaN for invalid denominators.

    Args:
        num: Numerator value.
        den: Denominator value.

    Returns:
        Rate value, or NaN if denominator is 0 or negative.
    """
    return (num / den) if den and den > 0 else float("nan")


def format_num(value: float, decimals: Optional[int]) -> Union[float, str]:
    """
    Format a numeric value with optional decimal places.

    Args:
        value: The numeric value to format.
        decimals: Number of decimal places, or None to return raw float.

    Returns:
        Formatted string if decimals is set, otherwise the raw float value.
    """
    if decimals is None:
        return value
    return f"{value:.{decimals}f}"


def round_numeric_df(
    df: pd.DataFrame,
    decimals: int,
    *,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Round all numeric columns in a DataFrame to specified decimal places.

    Args:
        df: The DataFrame to process.
        decimals: Number of decimal places to round to.
        exclude: List of column names to exclude from rounding.

    Returns:
        The DataFrame with numeric columns rounded (modified in place).
    """
    exclude = exclude or []
    num_cols = [
        c for c in df.columns
        if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if num_cols:
        df[num_cols] = df[num_cols].round(decimals)
    return df
