# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:25:40 2026

@author: Ishaan
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlateauResult:
    i_ss_geom_pa: np.ndarray
    i_ss_idx: np.ndarray
    keep_plateau: np.ndarray


@dataclass
class DeltaResult:
    i_ss_final_pa: np.ndarray
    delta_i_raw_pa: np.ndarray
    delta_i_final_pa: np.ndarray
    used_exp_fit_for_delta: np.ndarray


def extract_plateaus(
    time_min: np.ndarray,
    current_pa: np.ndarray,
    step_indices: list[int],
    min_plateau_pts: int = 200,
) -> PlateauResult:
    """
    Extract steady-state plateau currents between detected impact boundaries.

    Each plateau is estimated using a linear fit over the middle 40% of the
    plateau region, matching your current workflow.

    Parameters
    ----------
    time_min
        Time array in minutes.
    current_pa
        Current array in pA.
    step_indices
        Detected step boundary indices.
    min_plateau_pts
        Minimum number of points required for a plateau to be accepted.

    Returns
    -------
    PlateauResult
        i_ss_geom_pa:
            Geometric/linear-fit plateau current estimates in pA.
        i_ss_idx:
            Representative midpoint index for each accepted plateau.
        keep_plateau:
            Boolean array for interactive/manual exclusion.
    """
    n_points = len(current_pa)

    if len(step_indices) == 0:
        return PlateauResult(
            i_ss_geom_pa=np.array([], dtype=float),
            i_ss_idx=np.array([], dtype=int),
            keep_plateau=np.array([], dtype=bool),
        )

    boundaries = list(step_indices) + [n_points - 1]

    i_ss_geom_pa = []
    i_ss_idx = []

    for plateau_idx in range(len(boundaries) - 1):
        start_idx = boundaries[plateau_idx]
        end_idx = boundaries[plateau_idx + 1]
        plateau_len = end_idx - start_idx

        if plateau_len < min_plateau_pts:
            continue

        fit_start_idx = start_idx + int(0.30 * plateau_len)
        fit_end_idx = start_idx + int(0.70 * plateau_len)

        if fit_end_idx <= fit_start_idx + 2:
            continue

        time_fit_min = time_min[fit_start_idx:fit_end_idx]
        current_fit_pa = current_pa[fit_start_idx:fit_end_idx]

        slope_pa_per_min, intercept_pa = np.polyfit(time_fit_min, current_fit_pa, 1)

        time_mid_min = 0.5 * (time_fit_min[0] + time_fit_min[-1])
        current_mid_pa = slope_pa_per_min * time_mid_min + intercept_pa

        i_ss_geom_pa.append(current_mid_pa)
        i_ss_idx.append((fit_start_idx + fit_end_idx) // 2)

    i_ss_geom_pa = np.asarray(i_ss_geom_pa, dtype=float)
    i_ss_idx = np.asarray(i_ss_idx, dtype=int)
    keep_plateau = np.ones(len(i_ss_geom_pa), dtype=bool)

    return PlateauResult(
        i_ss_geom_pa=i_ss_geom_pa,
        i_ss_idx=i_ss_idx,
        keep_plateau=keep_plateau,
    )


def calculate_raw_delta_i(
    i_ss_geom_pa: np.ndarray,
    min_real_impact_pa: float = 15.0,
) -> np.ndarray:
    """
    Calculate raw step amplitudes between consecutive plateau currents.

    Negative Δi values are retained as impacts. Positive changes or small
    negative changes are converted to NaN.

    Parameters
    ----------
    i_ss_geom_pa
        Plateau current values in pA.
    min_real_impact_pa
        Minimum absolute negative impact size in pA.

    Returns
    -------
    delta_i_raw_pa
        Δi values in pA. First value is always NaN.
    """
    n_plateaus = len(i_ss_geom_pa)
    delta_i_raw_pa = np.full(n_plateaus, np.nan, dtype=float)

    if n_plateaus < 2:
        return delta_i_raw_pa

    for plateau_idx in range(1, n_plateaus):
        delta_i_raw_pa[plateau_idx] = (
            i_ss_geom_pa[plateau_idx] - i_ss_geom_pa[plateau_idx - 1]
        )

    for plateau_idx in range(1, n_plateaus):
        if (
            np.isfinite(delta_i_raw_pa[plateau_idx])
            and delta_i_raw_pa[plateau_idx] > -min_real_impact_pa
        ):
            delta_i_raw_pa[plateau_idx] = np.nan

    return delta_i_raw_pa


def apply_monoexp_overrides(
    i_ss_geom_pa: np.ndarray,
    delta_i_raw_pa: np.ndarray,
    exp_i_inf_pa: np.ndarray,
    exp_amplitude_pa: np.ndarray,
    exp_r_squared: np.ndarray,
    r_squared_threshold: float = 0.95,
    min_real_impact_pa: float = 15.0,
) -> DeltaResult:
    """
    Replace plateau-derived Δi values with monoexponential asymptote-derived
    values when the fit quality is good.

    The fitted model is assumed to be:

        i(t) = i_inf + amplitude * exp(-t / tau)

    For a step at plateau k:
        new plateau current = i_inf
        previous plateau current = i_inf + amplitude
        delta_i = new - previous

    Parameters
    ----------
    i_ss_geom_pa
        Original plateau current estimates.
    delta_i_raw_pa
        Raw Δi values from consecutive plateau differences.
    exp_i_inf_pa
        Fitted asymptotic currents from monoexponential fitting.
    exp_amplitude_pa
        Fitted amplitudes from monoexponential fitting.
    exp_r_squared
        R² values from monoexponential fitting.
    r_squared_threshold
        Minimum R² required to override plateau-derived Δi.
    min_real_impact_pa
        Minimum absolute negative impact size in pA.

    Returns
    -------
    DeltaResult
        Final plateau currents and final Δi values.
    """
    i_ss_final_pa = np.asarray(i_ss_geom_pa, dtype=float).copy()
    delta_i_final_pa = np.asarray(delta_i_raw_pa, dtype=float).copy()

    n_plateaus = len(i_ss_final_pa)
    used_exp_fit_for_delta = np.zeros(n_plateaus, dtype=bool)

    for plateau_idx in range(1, n_plateaus):
        if not np.isfinite(delta_i_raw_pa[plateau_idx]):
            continue

        if delta_i_raw_pa[plateau_idx] >= 0:
            continue

        if not (
            np.isfinite(exp_r_squared[plateau_idx])
            and exp_r_squared[plateau_idx] >= r_squared_threshold
        ):
            continue

        if not (
            np.isfinite(exp_i_inf_pa[plateau_idx])
            and np.isfinite(exp_amplitude_pa[plateau_idx])
        ):
            continue

        new_current_pa = exp_i_inf_pa[plateau_idx]
        previous_current_pa = (
            exp_i_inf_pa[plateau_idx] + exp_amplitude_pa[plateau_idx]
        )

        i_ss_final_pa[plateau_idx] = new_current_pa
        i_ss_final_pa[plateau_idx - 1] = previous_current_pa

        delta_i_final_pa[plateau_idx] = new_current_pa - previous_current_pa
        used_exp_fit_for_delta[plateau_idx] = True

    for plateau_idx in range(1, n_plateaus):
        if (
            np.isfinite(delta_i_final_pa[plateau_idx])
            and delta_i_final_pa[plateau_idx] > -min_real_impact_pa
        ):
            delta_i_final_pa[plateau_idx] = np.nan

    return DeltaResult(
        i_ss_final_pa=i_ss_final_pa,
        delta_i_raw_pa=delta_i_raw_pa,
        delta_i_final_pa=delta_i_final_pa,
        used_exp_fit_for_delta=used_exp_fit_for_delta,
    )


def calculate_global_drift(
    time_min: np.ndarray,
    i_ss_idx: np.ndarray,
    i_ss_final_pa: np.ndarray,
) -> tuple[float, float]:
    """
    Fit global drift through final plateau midpoint currents.

    Returns
    -------
    drift_slope_pa_per_min, drift_intercept_pa
    """
    if len(i_ss_idx) < 2 or len(i_ss_final_pa) < 2:
        return np.nan, np.nan

    drift_slope_pa_per_min, drift_intercept_pa = np.polyfit(
        time_min[i_ss_idx],
        i_ss_final_pa,
        1,
    )

    return float(drift_slope_pa_per_min), float(drift_intercept_pa)


def build_plateau_results_table(
    file_name: str,
    time_min: np.ndarray,
    i_ss_idx: np.ndarray,
    i_ss_geom_pa: np.ndarray,
    i_ss_final_pa: np.ndarray,
    delta_i_raw_pa: np.ndarray,
    delta_i_final_pa: np.ndarray,
    exp_tau_s: np.ndarray | None = None,
    exp_amplitude_pa: np.ndarray | None = None,
    exp_i_inf_pa: np.ndarray | None = None,
    exp_r_squared: np.ndarray | None = None,
    used_exp_fit_for_delta: np.ndarray | None = None,
    keep_plateau: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build the per-file plateau results table.

    This keeps Excel output formatting separate from the main analysis logic.
    """
    n_plateaus = len(i_ss_geom_pa)

    if exp_tau_s is None:
        exp_tau_s = np.full(n_plateaus, np.nan)

    if exp_amplitude_pa is None:
        exp_amplitude_pa = np.full(n_plateaus, np.nan)

    if exp_i_inf_pa is None:
        exp_i_inf_pa = np.full(n_plateaus, np.nan)

    if exp_r_squared is None:
        exp_r_squared = np.full(n_plateaus, np.nan)

    if used_exp_fit_for_delta is None:
        used_exp_fit_for_delta = np.full(n_plateaus, False, dtype=bool)

    if keep_plateau is None:
        keep_plateau = np.full(n_plateaus, True, dtype=bool)

    return pd.DataFrame(
        {
            "file": file_name,
            "plateau": np.arange(1, n_plateaus + 1),
            "t_mid_min": time_min[i_ss_idx],
            "i_ss_geom_pA": i_ss_geom_pa,
            "i_ss_pA": i_ss_final_pa,
            "delta_i_raw_pA": delta_i_raw_pa,
            "delta_i_from_prev_pA": delta_i_final_pa,
            "exp_tau_s": exp_tau_s,
            "exp_A_pA": exp_amplitude_pa,
            "exp_I_inf_pA": exp_i_inf_pa,
            "exp_R2": exp_r_squared,
            "used_exp_fit_for_delta": used_exp_fit_for_delta,
            "is_monoexp_R2_gt_threshold": exp_r_squared >= 0.95,
            "keep_for_stats": keep_plateau,
        }
    )