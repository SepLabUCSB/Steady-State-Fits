# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:22:06 2026

@author: Ishaan
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_current_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw chronoamperometry data.

    Returns
    -------
    time_s : np.ndarray
        Time in seconds.
    current_pa : np.ndarray
        Current in pA.
    """
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", comment="#")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    required_cols = {"time/s", "<I>/mA"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    time_s = df["time/s"].to_numpy(dtype=float)
    current_pa = df["<I>/mA"].to_numpy(dtype=float) * 1e9

    return time_s, current_pa