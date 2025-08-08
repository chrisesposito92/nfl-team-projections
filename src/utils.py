import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_display():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

def clip_nonnegative(series: pd.Series) -> pd.Series:
    return series.clip(lower=0)

def normalize_by_group_sum(df: pd.DataFrame, group_cols: Sequence[str], value_col: str, out_col: str) -> pd.DataFrame:
    gsum = df.groupby(list(group_cols), dropna=False)[value_col].transform("sum")
    df[out_col] = 0.0
    nonzero = gsum > 0
    df.loc[nonzero, out_col] = df.loc[nonzero, value_col] / gsum[nonzero]
    if (~nonzero).any():
        # distribute evenly inside groups that are all-zero
        # count within group to split evenly
        counts = df.groupby(list(group_cols), dropna=False)[value_col].transform("count")
        df.loc[~nonzero, out_col] = 1.0 / counts[~nonzero]
    return df

def allocate_total_by_weights(total: float, weights: np.ndarray) -> np.ndarray:
    total = float(total)
    w = np.array(weights, dtype=float)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        n = len(w)
        if n == 0:
            return np.array([])
        return np.full(n, total / n)
    return total * (w / s)

def stable_sort_values(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    return df.sort_values(list(by), kind="mergesort")

def safen_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return 0.0
