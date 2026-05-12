# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:31:15 2026

@author: Ishaan
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, lognorm, gamma

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


@dataclass
class MonoexpFitResult:
    current_inf_pa: float
    amplitude_pa: float
    tau_s: float
    r_squared: float
    predicted_current_pa: np.ndarray


@dataclass
class StepFitArrays:
    exp_tau_s: np.ndarray
    exp_amplitude_pa: np.ndarray
    exp_i_inf_pa: np.ndarray
    exp_r_squared: np.ndarray


def monoexp(time_s: np.ndarray, current_inf_pa: float, amplitude_pa: float, tau_s: float):
    """
    Monoexponential relaxation model:

        i(t) = i_inf + A exp(-t / tau)
    """
    return current_inf_pa + amplitude_pa * np.exp(-time_s / tau_s)


def calculate_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def fit_monoexp_step(
    time_s: np.ndarray,
    current_pa: np.ndarray,
    current_inf_guess_pa: float,
    amplitude_guess_pa: float,
    tau_guess_s: float = 1.0,
    tau_bounds_s: tuple[float, float] = (1e-3, 50.0),
) -> MonoexpFitResult:
    """
    Fit one post-impact monoexponential relaxation.
    """
    if curve_fit is None:
        raise ImportError("SciPy is required for monoexponential fitting.")

    time_s = np.asarray(time_s, dtype=float)
    current_pa = np.asarray(current_pa, dtype=float)

    popt, _ = curve_fit(
        monoexp,
        time_s,
        current_pa,
        p0=[current_inf_guess_pa, amplitude_guess_pa, tau_guess_s],
        bounds=(
            [-np.inf, -np.inf, tau_bounds_s[0]],
            [np.inf, np.inf, tau_bounds_s[1]],
        ),
        maxfev=10000,
    )

    current_inf_pa, amplitude_pa, tau_s = popt
    predicted_current_pa = monoexp(time_s, current_inf_pa, amplitude_pa, tau_s)
    r_squared = calculate_r_squared(current_pa, predicted_current_pa)

    return MonoexpFitResult(
        current_inf_pa=float(current_inf_pa),
        amplitude_pa=float(amplitude_pa),
        tau_s=float(tau_s),
        r_squared=r_squared,
        predicted_current_pa=predicted_current_pa,
    )

def _matches_impact_polarity(delta_i_pa: float, impact_polarity: str) -> bool:
    if not np.isfinite(delta_i_pa):
        return False

    impact_polarity = str(impact_polarity).strip().lower()

    if impact_polarity == "negative":
        return delta_i_pa < 0

    if impact_polarity == "positive":
        return delta_i_pa > 0

    if impact_polarity == "both":
        return delta_i_pa != 0

    raise ValueError(
        f"Unknown impact_polarity '{impact_polarity}'. "
        "Use 'negative', 'positive', or 'both'."
    )

def fit_monoexp_for_negative_steps(
    time_s: np.ndarray,
    current_pa: np.ndarray,
    step_indices: list[int],
    i_ss_geom_pa: np.ndarray,
    delta_i_raw_pa: np.ndarray,
    fit_window_sec: float = 3.0,
    min_plateau_pts: int = 200,
    pre_step_window_sec: float = 0.2,
    verbose: bool = True,
    impact_polarity: str = "negative",
) -> StepFitArrays:
    """
    Fit monoexponential relaxations after negative impact steps.

    This mirrors the fitting section from your original analyze_single_file()
    function but keeps the fitting logic separate from analysis orchestration.
    """
    time_s = np.asarray(time_s, dtype=float)
    current_pa = np.asarray(current_pa, dtype=float)
    i_ss_geom_pa = np.asarray(i_ss_geom_pa, dtype=float)
    delta_i_raw_pa = np.asarray(delta_i_raw_pa, dtype=float)

    n_plateaus = len(i_ss_geom_pa)

    exp_tau_s = np.full(n_plateaus, np.nan)
    exp_amplitude_pa = np.full(n_plateaus, np.nan)
    exp_i_inf_pa = np.full(n_plateaus, np.nan)
    exp_r_squared = np.full(n_plateaus, np.nan)

    if curve_fit is None:
        if verbose:
            print("  SciPy is not installed. Skipping monoexponential fits.")
        return StepFitArrays(
            exp_tau_s=exp_tau_s,
            exp_amplitude_pa=exp_amplitude_pa,
            exp_i_inf_pa=exp_i_inf_pa,
            exp_r_squared=exp_r_squared,
        )

    if n_plateaus < 2:
        return StepFitArrays(
            exp_tau_s=exp_tau_s,
            exp_amplitude_pa=exp_amplitude_pa,
            exp_i_inf_pa=exp_i_inf_pa,
            exp_r_squared=exp_r_squared,
        )

    dt_s = float(np.median(np.diff(time_s)))

    if verbose:
        print(f"  Performing monoexponential fits for {impact_polarity} steps...")

    for plateau_idx in range(1, n_plateaus):
        delta_i_pa = delta_i_raw_pa[plateau_idx]

        if not _matches_impact_polarity(delta_i_pa, impact_polarity):
            continue

        if plateau_idx >= len(step_indices):
            continue

        step_idx = int(step_indices[plateau_idx])

        pre_pts = max(1, int(pre_step_window_sec / dt_s))
        post_pts = max(int(fit_window_sec / dt_s), min_plateau_pts)

        start_idx = max(0, step_idx - pre_pts)
        end_idx = min(len(time_s), step_idx + post_pts)

        if end_idx <= start_idx + 10:
            continue

        fit_time_s = time_s[start_idx:end_idx] - time_s[start_idx]
        fit_current_pa = current_pa[start_idx:end_idx]

        if np.isfinite(i_ss_geom_pa[plateau_idx]):
            current_inf_guess_pa = i_ss_geom_pa[plateau_idx]
        else:
            current_inf_guess_pa = float(np.median(fit_current_pa[-50:]))

        if np.isfinite(i_ss_geom_pa[plateau_idx - 1]):
            amplitude_guess_pa = i_ss_geom_pa[plateau_idx - 1] - current_inf_guess_pa
        else:
            amplitude_guess_pa = float(np.median(fit_current_pa[:50]) - current_inf_guess_pa)

        try:
            fit_result = fit_monoexp_step(
                time_s=fit_time_s,
                current_pa=fit_current_pa,
                current_inf_guess_pa=current_inf_guess_pa,
                amplitude_guess_pa=amplitude_guess_pa,
                tau_guess_s=1.0,
            )

            exp_tau_s[plateau_idx] = fit_result.tau_s
            exp_amplitude_pa[plateau_idx] = fit_result.amplitude_pa
            exp_i_inf_pa[plateau_idx] = fit_result.current_inf_pa
            exp_r_squared[plateau_idx] = fit_result.r_squared

            if verbose:
                print(
                    f"    Step plateau {plateau_idx}: "
                    f"tau = {fit_result.tau_s:.3g} s, "
                    f"R² = {fit_result.r_squared:.4f}"
                )

        except Exception as exc:
            if verbose:
                print(f"    Exp fit failed for step plateau {plateau_idx}: {exc}")

    return StepFitArrays(
        exp_tau_s=exp_tau_s,
        exp_amplitude_pa=exp_amplitude_pa,
        exp_i_inf_pa=exp_i_inf_pa,
        exp_r_squared=exp_r_squared,
    )
