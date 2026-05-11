# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:55:33 2026

@author: Ishaan
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_reviewed_fits_path(raw_path: Path):
    raw_path = Path(raw_path)
    return raw_path.with_name(raw_path.stem + "_reviewed_fits.xlsx")


def apply_reviewed_fits_to_results(results_df: pd.DataFrame, raw_path: Path):
    """
    If a reviewed-fits file exists, replace the automatic delta_i values
    in the main results table with the reviewed values.

    Keeps backups:
        auto_delta_i_final_pa
        auto_delta_i_from_prev_pA

    Writes final reviewed value into:
        delta_i_final_pa
        delta_i_from_prev_pA
    """
    results_df = results_df.copy()
    reviewed_path = get_reviewed_fits_path(raw_path)

    if not reviewed_path.exists():
        return results_df

    reviewed_df = pd.read_excel(reviewed_path)

    print(f"  Applying reviewed fits from {reviewed_path}")

    # Choose the reviewed delta column.
    if "delta_i_reviewed_final_pA" in reviewed_df.columns:
        reviewed_delta_col = "delta_i_reviewed_final_pA"
    elif "review_delta_i_pA" in reviewed_df.columns:
        reviewed_delta_col = "review_delta_i_pA"
    elif "delta_i_final_pa" in reviewed_df.columns:
        reviewed_delta_col = "delta_i_final_pa"
    else:
        raise KeyError(
            "Reviewed fits file does not contain a reviewed delta column. "
            f"Available columns: {list(reviewed_df.columns)}"
        )

    # Preserve original automatic values.
    if "delta_i_final_pa" in results_df.columns and "auto_delta_i_final_pa" not in results_df.columns:
        results_df["auto_delta_i_final_pa"] = results_df["delta_i_final_pa"]

    if "delta_i_from_prev_pA" in results_df.columns and "auto_delta_i_from_prev_pA" not in results_df.columns:
        results_df["auto_delta_i_from_prev_pA"] = results_df["delta_i_from_prev_pA"]

    # Try to align by impact_index first.
    if "impact_index" in results_df.columns and "impact_index" in reviewed_df.columns:
        reviewed_small = reviewed_df[
            [
                "impact_index",
                reviewed_delta_col,
                "keep_for_stats",
                "review_status",
            ]
        ].copy()

        reviewed_small = reviewed_small.rename(
            columns={reviewed_delta_col: "reviewed_delta_i_pA"}
        )

        results_df = results_df.merge(
            reviewed_small,
            on="impact_index",
            how="left",
            suffixes=("", "_reviewed"),
        )

    # Otherwise align by row order.
    else:
        n_update = min(len(results_df), len(reviewed_df))

        results_df["reviewed_delta_i_pA"] = np.nan
        results_df.loc[
            results_df.index[:n_update],
            "reviewed_delta_i_pA",
        ] = reviewed_df[reviewed_delta_col].to_numpy(dtype=float)[:n_update]

        if "keep_for_stats" in reviewed_df.columns:
            if "keep_for_stats" not in results_df.columns:
                results_df["keep_for_stats"] = True

            results_df.loc[
                results_df.index[:n_update],
                "keep_for_stats",
            ] = reviewed_df["keep_for_stats"].to_numpy(dtype=bool)[:n_update]

        if "review_status" in reviewed_df.columns:
            results_df["review_status"] = ""
            results_df.loc[
                results_df.index[:n_update],
                "review_status",
            ] = reviewed_df["review_status"].astype(str).to_numpy()[:n_update]

    # Replace final delta values where reviewed values are available.
    reviewed_delta = results_df["reviewed_delta_i_pA"].to_numpy(dtype=float)
    valid = np.isfinite(reviewed_delta)

    if "delta_i_final_pa" in results_df.columns:
        results_df.loc[valid, "delta_i_final_pa"] = reviewed_delta[valid]

    if "delta_i_from_prev_pA" in results_df.columns:
        results_df.loc[valid, "delta_i_from_prev_pA"] = reviewed_delta[valid]

    # Add a clear final column for downstream analysis.
    results_df["delta_i_reviewed_final_pA"] = results_df.get(
        "delta_i_final_pa",
        results_df["reviewed_delta_i_pA"],
    )

    return results_df