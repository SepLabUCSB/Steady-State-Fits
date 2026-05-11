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
        print("  Performing monoexponential fits for negative steps...")

    for plateau_idx in range(1, n_plateaus):
        delta_i_pa = delta_i_raw_pa[plateau_idx]

        if not np.isfinite(delta_i_pa) or delta_i_pa >= 0:
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


def lognormal_fit(x: np.ndarray, data: np.ndarray):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < 2:
        return np.zeros_like(x), np.nan, np.nan

    shape, loc, scale = lognorm.fit(data, floc=0)
    pdf = lognorm.pdf(x, shape, loc=0, scale=scale)

    mean_value = float(np.mean(data))
    std_value = float(np.std(data, ddof=1))

    return pdf, mean_value, std_value


def gaussian_fit(x: np.ndarray, data: np.ndarray):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 2:
        return np.zeros_like(x), np.nan, np.nan

    mean_value = float(np.mean(data))
    std_value = float(np.std(data, ddof=1))

    pdf = norm.pdf(x, mean_value, std_value)

    return pdf, mean_value, std_value


def gamma_fit(x: np.ndarray, data: np.ndarray):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < 3:
        return np.zeros_like(x), np.nan, np.nan

    shape, loc, scale = gamma.fit(data, floc=0)
    pdf = gamma.pdf(x, shape, loc=0, scale=scale)

    mean_value = float(shape * scale)
    std_value = float(np.sqrt(shape) * scale)

    return pdf, mean_value, std_value


def fit_distribution(
    x: np.ndarray,
    data: np.ndarray,
    distribution: str = "lognormal",
):
    """
    Fit a probability density to impact-size data.

    Parameters
    ----------
    distribution
        Options: "lognormal", "gaussian", or "gamma".
    """
    distribution = distribution.lower().strip()

    if distribution == "lognormal":
        return lognormal_fit(x, data)

    if distribution == "gaussian":
        return gaussian_fit(x, data)

    if distribution == "gamma":
        return gamma_fit(x, data)

    raise ValueError(
        f"Unknown distribution '{distribution}'. "
        "Use 'lognormal', 'gaussian', or 'gamma'."
    )


def fit_pdf_to_relative_histogram(
    centers: np.ndarray,
    relative_frequency: np.ndarray,
    x_fit: np.ndarray,
    distribution: str = "lognormal",
):
    """
    Fit a PDF-shaped curve directly to binned relative-frequency data.

    Returns
    -------
    y_fit
        Fitted curve in relative-frequency units, not raw PDF units.
    """
    centers = np.asarray(centers, dtype=float)
    relative_frequency = np.asarray(relative_frequency, dtype=float)
    x_fit = np.asarray(x_fit, dtype=float)

    mask = (
        np.isfinite(centers)
        & np.isfinite(relative_frequency)
        & (relative_frequency >= 0)
    )

    x = centers[mask]
    y = relative_frequency[mask]

    if len(x) < 4 or y.max() <= 0:
        return np.zeros_like(x_fit)

    distribution = distribution.lower().strip()

    if curve_fit is None:
        return np.zeros_like(x_fit)

    if distribution == "lognormal":

        def model(xx, amplitude, shape, scale):
            return amplitude * lognorm.pdf(xx, shape, loc=0, scale=scale)

        positive_mask = x > 0
        x_positive = x[positive_mask] if np.any(positive_mask) else x
        y_positive = y[positive_mask] if np.any(positive_mask) else y

        amplitude_guess = float(y_positive.max() * 5.0)
        shape_guess = 0.6
        scale_guess = (
            float(np.median(x_positive[x_positive > 0]))
            if np.any(x_positive > 0)
            else 10.0
        )

        popt, _ = curve_fit(
            model,
            x_positive,
            y_positive,
            p0=[amplitude_guess, shape_guess, scale_guess],
            bounds=([0.0, 1e-3, 1e-3], [np.inf, 5.0, np.inf]),
            maxfev=20000,
        )

        return model(x_fit, *popt)

    if distribution == "gaussian":

        def model(xx, amplitude, mean_value, std_value):
            return amplitude * norm.pdf(xx, mean_value, std_value)

        amplitude_guess = float(y.max() * 5.0)
        mean_guess = float(x[np.argmax(y)])
        std_guess = float(np.std(x)) if np.std(x) > 1e-6 else 5.0

        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=[amplitude_guess, mean_guess, std_guess],
            bounds=([0.0, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )

        return model(x_fit, *popt)

    raise ValueError(
        f"Unknown distribution '{distribution}'. "
        "Use 'lognormal' or 'gaussian'."
    )