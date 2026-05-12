# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:32:37 2026

@author: Ishaan
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.config import PipelineConfig
from modules.data_loader import load_current_trace
from modules.filters import notch_filter
from modules.detection import compute_step_magnitude, detect_step_indices
from modules.fitting import fit_monoexp_for_negative_steps
from modules.step_analysis import (
    extract_plateaus,
    calculate_raw_delta_i,
    calculate_global_drift,
    build_plateau_results_table,
)
from modules.reviewed_fits import apply_reviewed_fits_to_results


def _empty_impact_arrays():
    return np.array([], dtype=float), np.array([], dtype=float)

def _delta_matches_polarity_array(delta_i_pa: np.ndarray, impact_polarity: str):
    delta_i_pa = np.asarray(delta_i_pa, dtype=float)
    finite = np.isfinite(delta_i_pa)

    impact_polarity = str(impact_polarity).strip().lower()

    if impact_polarity == "negative":
        return finite & (delta_i_pa < 0)

    if impact_polarity == "positive":
        return finite & (delta_i_pa > 0)

    if impact_polarity == "both":
        return finite & (delta_i_pa != 0)

    raise ValueError(
        f"Unknown impact_polarity '{impact_polarity}'. "
        "Use 'negative', 'positive', or 'both'."
    )

def _save_excel_safely(df: pd.DataFrame, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_excel(out_path, index=False)
    except PermissionError:
        raise PermissionError(
            f"Could not write {out_path}. "
            "Close the file in Excel and run again."
        )

def _save_combined_workbook_safely(
    results_df: pd.DataFrame,
    monoexp_df: pd.DataFrame,
    out_path: Path,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="results", index=False)

            if monoexp_df is not None and not monoexp_df.empty:
                monoexp_df.to_excel(
                    writer,
                    sheet_name="monoexponential",
                    index=False,
                )

    except PermissionError:
        raise PermissionError(
            f"Could not write {out_path}. "
            "Close the file in Excel and run again."
        )


def _autosave_combined_rows(rows, monoexp_rows, out_xlsx: Path, verbose: bool = True):
    """
    Save combined rows in real time after each Tk-reviewed file.

    This overwrites combined_results.xlsx with all rows processed so far.
    That avoids duplicate appends while still protecting progress during
    interactive review.
    """
    if rows is None or len(rows) == 0:
        return

    combined_df = pd.DataFrame(rows)
    
    combined_monoexp_df = (
        pd.DataFrame(monoexp_rows)
        if monoexp_rows is not None and len(monoexp_rows) > 0
        else pd.DataFrame()
    )

    _save_combined_workbook_safely(
        results_df=combined_df,
        monoexp_df=combined_monoexp_df,
        out_path=out_xlsx,
    )
    
    if verbose:
       print(
           f"  [autosave] Updated {out_xlsx.name} "
           f"({len(combined_df)} result rows, "
           f"{len(combined_monoexp_df)} monoexp rows)"
       )


def _ensure_keep_for_stats(per_file_df: pd.DataFrame):
    if per_file_df is None or per_file_df.empty:
        return per_file_df

    per_file_df = per_file_df.copy()

    if "keep_for_stats" not in per_file_df.columns:
        per_file_df["keep_for_stats"] = True

    return per_file_df


def _save_kept_removed_tables(
    per_file_df: pd.DataFrame,
    path: Path,
    verbose: bool = True,
):
    """
    Save separate kept and removed plateau tables.
    """
    per_file_df = _ensure_keep_for_stats(per_file_df)

    if per_file_df is None or per_file_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep_mask = per_file_df["keep_for_stats"].astype(bool)

    per_file_keep = per_file_df[keep_mask].copy()
    per_file_removed = per_file_df[~keep_mask].copy()

    out_keep = path.with_name(path.stem + "_iss_results_kept.xlsx")
    _save_excel_safely(per_file_keep, out_keep)

    if verbose:
        print(f"  Saved KEPT per-file results to {out_keep}")

    if not per_file_removed.empty:
        out_removed = path.with_name(path.stem + "_iss_results_removed.xlsx")
        _save_excel_safely(per_file_removed, out_removed)

        if verbose:
            print(f"  Saved REMOVED per-file results to {out_removed}")
    elif verbose:
        print("  No removed plateaus for this file.")

    return per_file_keep, per_file_removed


def split_small_large_impacts(
    per_file_df: pd.DataFrame,
    small_impact_min_pa: float = -100.0,
    impact_polarity: str = "negative",
):
    """
    Split selected impacts into small/medium and large groups using |Δi|.

    negative:
        Δi < 0

    positive:
        Δi > 0

    both:
        Δi != 0

    small/medium:
        |Δi| <= |small_impact_min_pa|

    large:
        |Δi| > |small_impact_min_pa|
    """
    if per_file_df is None or per_file_df.empty:
        return _empty_impact_arrays()

    if "delta_i_from_prev_pA" not in per_file_df.columns:
        return _empty_impact_arrays()

    df = _ensure_keep_for_stats(per_file_df)

    if "keep_for_stats" in df.columns:
        df = df[df["keep_for_stats"].astype(bool)]

    delta = df["delta_i_from_prev_pA"].to_numpy(dtype=float)
    selected_mask = _delta_matches_polarity_array(delta, impact_polarity)

    selected_df = df[selected_mask].copy()

    threshold_abs_pa = abs(float(small_impact_min_pa))
    abs_delta = selected_df["delta_i_from_prev_pA"].abs()

    small_df = selected_df[abs_delta <= threshold_abs_pa]
    large_df = selected_df[abs_delta > threshold_abs_pa]

    small_impacts_pa = small_df["delta_i_from_prev_pA"].to_numpy(dtype=float)
    large_impacts_pa = large_df["delta_i_from_prev_pA"].to_numpy(dtype=float)

    return small_impacts_pa, large_impacts_pa

def _apply_notch_filter_if_requested(
    current_raw_pa: np.ndarray,
    time_min: np.ndarray,
    sample_freq_hz: float,
    config: PipelineConfig,
):
    current_pa = current_raw_pa.copy()

    if not config.filter.apply_notch_filter:
        return current_pa

    mask_after_charging = time_min > config.detection.charging_cutoff_min

    try:
        current_pa[mask_after_charging] = notch_filter(
            current_pa=current_raw_pa[mask_after_charging],
            sample_freq_hz=sample_freq_hz,
            freqs_hz=config.filter.notch_freqs_hz,
            q=config.filter.notch_q,
        )
    except TypeError:
        current_pa[mask_after_charging] = notch_filter(
            current_raw_pa[mask_after_charging],
            sample_freq_hz,
            config.filter.notch_freqs_hz,
            config.filter.notch_q,
        )

    return current_pa


def _run_interactive_review(
    path: Path,
    time_s: np.ndarray,
    time_min: np.ndarray,
    current_raw_pa: np.ndarray,
    current_pa: np.ndarray,
    step_indices,
    plateau_result,
    i_ss_final_pa: np.ndarray,
    delta_i_final_pa: np.ndarray,
    step_fit_arrays,
    drift_slope_pa_per_min: float,
    drift_intercept_pa: float,
    config: PipelineConfig,
):
    """
    Run the Tk trace reviewer and return keep_plateau.

    Importing inside this function keeps normal non-interactive runs from
    initializing Tk or GUI plotting code.
    """
    from interactive.trace_reviewer import ReviewTracePayload, review_trace

    dt_s = float(np.median(np.diff(time_s)))

    payload = ReviewTracePayload(
        path=path,
        time_s=time_s,
        time_min=time_min,
        current_raw_pa=current_raw_pa,
        current_filtered_pa=current_pa,
        step_indices=step_indices,
        i_ss_idx=plateau_result.i_ss_idx,
        i_ss_final_pa=i_ss_final_pa,
        delta_i_final_pa=delta_i_final_pa,
        exp_tau_s=step_fit_arrays.exp_tau_s,
        exp_amplitude_pa=step_fit_arrays.exp_amplitude_pa,
        exp_i_inf_pa=step_fit_arrays.exp_i_inf_pa,
        exp_r_squared=step_fit_arrays.exp_r_squared,
        monoexp_r2_threshold=config.fit.monoexp_r2_threshold,
        exp_fit_win_sec=config.fit.exp_fit_win_sec,
        min_plateau_pts=config.detection.min_plateau_pts,
        dt_s=dt_s,
        drift_slope_pa_per_min=drift_slope_pa_per_min,
        drift_intercept_pa=drift_intercept_pa,
        out_png=path.with_name(path.stem + "_trace.png"),
    )

    return review_trace(payload)


def _build_not_enough_plateaus_table(
    path: Path,
    time_min: np.ndarray,
    plateau_result,
):
    n_plateaus = len(plateau_result.i_ss_geom_pa)

    delta_nan = np.full(n_plateaus, np.nan)

    exp_tau_s = np.full(n_plateaus, np.nan)
    exp_amplitude_pa = np.full(n_plateaus, np.nan)
    exp_i_inf_pa = np.full(n_plateaus, np.nan)
    exp_r_squared = np.full(n_plateaus, np.nan)

    used_exp_fit_for_delta = np.full(n_plateaus, False, dtype=bool)
    keep_plateau = np.ones(n_plateaus, dtype=bool)

    per_file_df = build_plateau_results_table(
        file_name=path.name,
        time_min=time_min,
        i_ss_idx=plateau_result.i_ss_idx,
        i_ss_geom_pa=plateau_result.i_ss_geom_pa,
        i_ss_final_pa=plateau_result.i_ss_geom_pa,
        delta_i_raw_pa=delta_nan,
        delta_i_final_pa=delta_nan,
        exp_tau_s=exp_tau_s,
        exp_amplitude_pa=exp_amplitude_pa,
        exp_i_inf_pa=exp_i_inf_pa,
        exp_r_squared=exp_r_squared,
        used_exp_fit_for_delta=used_exp_fit_for_delta,
        keep_plateau=keep_plateau,
    )

    drift_slope_pa_per_min, drift_intercept_pa = calculate_global_drift(
        time_min=time_min,
        i_ss_idx=plateau_result.i_ss_idx,
        i_ss_final_pa=plateau_result.i_ss_geom_pa,
    )

    per_file_df["global_drift_slope_pA_per_min"] = drift_slope_pa_per_min
    per_file_df["global_drift_intercept_pA"] = drift_intercept_pa

    return _ensure_keep_for_stats(per_file_df)

def _build_monoexp_results_table(
    file_name: str,
    time_min: np.ndarray,
    plateau_result,
    delta_i_linear_pa: np.ndarray,
    keep_plateau: np.ndarray,
    step_fit_arrays,
    monoexp_user_selected: np.ndarray | None = None,
):
    n_plateaus = len(plateau_result.i_ss_geom_pa)

    exp_tau_s = np.asarray(step_fit_arrays.exp_tau_s, dtype=float)
    exp_amplitude_pa = np.asarray(step_fit_arrays.exp_amplitude_pa, dtype=float)
    exp_i_inf_pa = np.asarray(step_fit_arrays.exp_i_inf_pa, dtype=float)
    exp_r_squared = np.asarray(step_fit_arrays.exp_r_squared, dtype=float)
    monoexp_decay_rate_constant_s_inv = np.full(n_plateaus, np.nan)

    valid_tau = np.isfinite(exp_tau_s) & (exp_tau_s > 0)
    monoexp_decay_rate_constant_s_inv[valid_tau] = 1.0 / exp_tau_s[valid_tau]
    
    if monoexp_user_selected is None:
        monoexp_user_selected = np.zeros(n_plateaus, dtype=bool)
    else:
        monoexp_user_selected = np.asarray(monoexp_user_selected, dtype=bool)

    return pd.DataFrame(
        {
            "file_name": file_name,
            "plateau_index": np.arange(n_plateaus),
            "time_min": time_min[plateau_result.i_ss_idx],
    
            "monoexp_user_selected": monoexp_user_selected,
            "monoexp_i_inf_pA": exp_i_inf_pa,
            "monoexp_amplitude_pA": exp_amplitude_pa,
            "monoexp_tau_s": exp_tau_s,
            "monoexp_decay_rate_constant_s_inv": monoexp_decay_rate_constant_s_inv,
            "monoexp_r_squared": exp_r_squared,
    
            "keep_for_stats": keep_plateau,
        }
    )


def analyze_single_file(
    path,
    config: PipelineConfig,
    save_per_file: bool = True,
    verbose: bool = True,
    review_interactive: bool = False,
    impact_polarity: str = "negative",
):
    """
    Analyze one raw nanoimpact .txt file.

    Returns
    -------
    small_impacts_pa
        Signed negative Δi values for small/medium impacts.

    large_impacts_pa
        Signed negative Δi values for large impacts.

    per_file_keep_df
        Plateau-level results table used for combined_results.xlsx.
        In interactive mode, removed plateaus are excluded from this returned table.
    """
    path = Path(path)
    empty_monoexp_df = pd.DataFrame()

    if verbose:
        print(f"\n=== Processing file: {path.name} ===")

    # -----------------------------
    # Load data
    # -----------------------------
    time_s, current_raw_pa = load_current_trace(path)

    n_points = len(current_raw_pa)
    time_min = time_s / 60.0
    dt_s = float(np.median(np.diff(time_s)))
    sample_freq_hz = 1.0 / dt_s

    if verbose:
        print(f"  n = {n_points}, dt ≈ {dt_s:.4f} s, fs ≈ {sample_freq_hz:.1f} Hz")

    # -----------------------------
    # Optional notch filtering
    # -----------------------------
    current_pa = _apply_notch_filter_if_requested(
        current_raw_pa=current_raw_pa,
        time_min=time_min,
        sample_freq_hz=sample_freq_hz,
        config=config,
    )

    # -----------------------------
    # Step detection
    # -----------------------------
    step_magnitude_pa, window_pts = compute_step_magnitude(
        time_s=time_s,
        current_pa=current_pa,
        pre_post_win_sec=config.detection.pre_post_win_sec,
    )

    if verbose:
        print(f"  Detection window: {window_pts} points")

    detection_result = detect_step_indices(
        time_s=time_s,
        step_magnitude_pa=step_magnitude_pa,
        charging_cutoff_min=config.detection.charging_cutoff_min,
        cluster_gap_sec=config.detection.cluster_gap_sec,
        min_real_impact_pa=config.detection.min_real_impact_pa,
        impact_polarity=impact_polarity,
    )

    step_indices = detection_result.step_indices

    if verbose:
        print(
            f"  Detection threshold: {detection_result.threshold_pa:.2f} pA "
            f"(noise std = {detection_result.noise_std_pa:.2f} pA)"
        )
        print(f"  Cluster gap: {detection_result.cluster_gap_pts} points")
        print(f"  Detected {len(step_indices)} step boundaries.")

    if len(step_indices) == 0:
        if verbose:
            print("  No candidates found.")

        return np.array([]), np.array([]), None, empty_monoexp_df

    # -----------------------------
    # Plateau extraction
    # -----------------------------
    plateau_result = extract_plateaus(
        time_min=time_min,
        current_pa=current_pa,
        step_indices=step_indices,
        min_plateau_pts=config.detection.min_plateau_pts,
    )

    n_plateaus = len(plateau_result.i_ss_geom_pa)

    if verbose:
        print(f"  Plateaus accepted: {n_plateaus}")

    if n_plateaus == 0:
        return np.array([]), np.array([]), None, empty_monoexp_df

    # -----------------------------
    # Single-plateau / not-enough-plateaus case
    # -----------------------------
    if n_plateaus < 2:
        if verbose:
            print("  Not enough plateaus for Δi.")

        per_file_df = _build_not_enough_plateaus_table(
            path=path,
            time_min=time_min,
            plateau_result=plateau_result,
        )
        
        per_file_df = apply_reviewed_fits_to_results(
            results_df=per_file_df,
            raw_path=path,
        )
        
        per_file_df = _ensure_keep_for_stats(per_file_df)
        
        if save_per_file:
            out_xlsx = path.with_name(path.stem + "_iss_results.xlsx")
            _save_excel_safely(per_file_df, out_xlsx)

            if verbose:
                print(f"  Saved per-file results to {out_xlsx}")

            if review_interactive:
                _save_kept_removed_tables(per_file_df, path, verbose=verbose)

        return np.array([]), np.array([]), per_file_df, empty_monoexp_df

    # -----------------------------
    # Raw Δi calculation
    # -----------------------------
    delta_i_raw_pa = calculate_raw_delta_i(
        i_ss_geom_pa=plateau_result.i_ss_geom_pa,
        min_real_impact_pa=config.detection.min_real_impact_pa,
    )

    # -----------------------------
    # Monoexponential fits
    # -----------------------------
    step_fit_arrays = fit_monoexp_for_negative_steps(
        time_s=time_s,
        current_pa=current_pa,
        step_indices=step_indices,
        i_ss_geom_pa=plateau_result.i_ss_geom_pa,
        delta_i_raw_pa=delta_i_raw_pa,
        fit_window_sec=config.fit.exp_fit_win_sec,
        min_plateau_pts=config.detection.min_plateau_pts,
        pre_step_window_sec=0.2,
        verbose=verbose,
        impact_polarity=impact_polarity,
    )

    # -----------------------------
    # Linear / plateau values
    # -----------------------------
    # Monoexponential fits are saved separately in the combined workbook's
    # "monoexponential" sheet. They do not overwrite the main results sheet.
    i_ss_linear_pa = plateau_result.i_ss_geom_pa.copy()
    delta_i_linear_pa = delta_i_raw_pa.copy()
    used_exp_fit_for_delta = np.full(n_plateaus, False, dtype=bool)

    # -----------------------------
    # Drift
    # -----------------------------
    drift_slope_pa_per_min, drift_intercept_pa = calculate_global_drift(
        time_min=time_min,
        i_ss_idx=plateau_result.i_ss_idx,
        i_ss_final_pa=i_ss_linear_pa,
    )

    if verbose and np.isfinite(drift_slope_pa_per_min):
        print(f"  Global drift slope: {drift_slope_pa_per_min:.2f} pA/min")

    # -----------------------------
    # Interactive review
    # -----------------------------
    keep_plateau = np.ones(n_plateaus, dtype=bool)
    monoexp_user_selected = np.zeros(n_plateaus, dtype=bool)
    if review_interactive:
        review_result = _run_interactive_review(
            path=path,
            time_s=time_s,
            time_min=time_min,
            current_raw_pa=current_raw_pa,
            current_pa=current_pa,
            step_indices=step_indices,
            plateau_result=plateau_result,
            i_ss_final_pa=i_ss_linear_pa,
            delta_i_final_pa=delta_i_linear_pa,
            step_fit_arrays=step_fit_arrays,
            drift_slope_pa_per_min=drift_slope_pa_per_min,
            drift_intercept_pa=drift_intercept_pa,
            config=config,
        )

        keep_plateau = np.asarray(review_result.keep_plateau, dtype=bool)
    
        step_fit_arrays.exp_tau_s[:] = review_result.exp_tau_s
        step_fit_arrays.exp_amplitude_pa[:] = review_result.exp_amplitude_pa
        step_fit_arrays.exp_i_inf_pa[:] = review_result.exp_i_inf_pa
        step_fit_arrays.exp_r_squared[:] = review_result.exp_r_squared
    
        monoexp_user_selected = np.asarray(
            review_result.monoexp_user_selected,
            dtype=bool,
        )

    if keep_plateau.size != n_plateaus:
        raise ValueError(
            f"Interactive reviewer returned {keep_plateau.size} keep flags, "
            f"but there are {n_plateaus} plateaus."
        )

    delta_i_final_pa = delta_i_linear_pa.copy()
    delta_i_final_pa[~keep_plateau] = np.nan

    used_exp_fit_for_delta = np.full(n_plateaus, False, dtype=bool)

    # -----------------------------
    # Build output table
    # -----------------------------
    per_file_df = build_plateau_results_table(
        file_name=path.name,
        time_min=time_min,
        i_ss_idx=plateau_result.i_ss_idx,
        i_ss_geom_pa=plateau_result.i_ss_geom_pa,
        i_ss_final_pa=i_ss_linear_pa,
        delta_i_raw_pa=delta_i_linear_pa,
        delta_i_final_pa=delta_i_final_pa,
        exp_tau_s=step_fit_arrays.exp_tau_s,
        exp_amplitude_pa=step_fit_arrays.exp_amplitude_pa,
        exp_i_inf_pa=step_fit_arrays.exp_i_inf_pa,
        exp_r_squared=step_fit_arrays.exp_r_squared,
        used_exp_fit_for_delta=used_exp_fit_for_delta,
        keep_plateau=keep_plateau,
    )

    per_file_df = _ensure_keep_for_stats(per_file_df)

    per_file_df["global_drift_slope_pA_per_min"] = drift_slope_pa_per_min
    per_file_df["global_drift_intercept_pA"] = drift_intercept_pa
    
    # ------------------------------------------------------------
    # Apply reviewed individual-fit corrections if they exist.
    # This must happen BEFORE saving per-file, kept, combined,
    # split, and summary outputs.
    # ------------------------------------------------------------
    per_file_df = apply_reviewed_fits_to_results(
        results_df=per_file_df,
        raw_path=path,
    )
    
    per_file_df = _ensure_keep_for_stats(per_file_df)

    # -----------------------------
    # Save outputs
    # -----------------------------
    if save_per_file:
        out_xlsx = path.with_name(path.stem + "_iss_results.xlsx")
        _save_excel_safely(per_file_df, out_xlsx)

        if verbose:
            print(f"  Saved per-file results to {out_xlsx}")

    if review_interactive and save_per_file:
        per_file_keep_df, per_file_removed_df = _save_kept_removed_tables(
            per_file_df,
            path,
            verbose=verbose,
        )
    else:
        per_file_keep_df = per_file_df[
            per_file_df["keep_for_stats"].astype(bool)
        ].copy()
        
    # -----------------------------
    # Build monoexponential table for combined_results.xlsx
    # -----------------------------
    monoexp_df = _build_monoexp_results_table(
        file_name=path.name,
        time_min=time_min,
        plateau_result=plateau_result,
        delta_i_linear_pa=delta_i_raw_pa,
        keep_plateau=keep_plateau,
        step_fit_arrays=step_fit_arrays,
        monoexp_user_selected=monoexp_user_selected,
    )

    monoexp_keep_df = monoexp_df[
    monoexp_df["keep_for_stats"].astype(bool)
    & monoexp_df["monoexp_user_selected"].astype(bool)
    ].copy()

    # -----------------------------
    # Split impacts for summary
    # -----------------------------
    small_impacts_pa, large_impacts_pa = split_small_large_impacts(
        per_file_df=per_file_keep_df,
        small_impact_min_pa=config.detection.small_impact_min_pa,
        impact_polarity=impact_polarity,
    )

    if verbose:
        values = per_file_keep_df["delta_i_from_prev_pA"].to_numpy(dtype=float)
        selected_mask = _delta_matches_polarity_array(values, impact_polarity)
        values = values[selected_mask]
        abs_values = np.abs(values)

        count_0_40 = int(np.sum((abs_values >= 0) & (abs_values < 40)))
        count_40_100 = int(np.sum((abs_values >= 40) & (abs_values <= 100)))
        count_gt_100 = int(np.sum(abs_values > 100))

        print(f"  {impact_polarity} impacts 0–40 pA:   {count_0_40}")
        print(f"  {impact_polarity} impacts 40–100 pA: {count_40_100}")
        print(f"  {impact_polarity} impacts >100 pA:   {count_gt_100}")

    return small_impacts_pa, large_impacts_pa, per_file_keep_df, monoexp_keep_df


def analyze_folder(
    config: PipelineConfig,
    save_combined: bool = True,
    save_split_files: bool = True,
    save_per_file: bool = True,
    verbose: bool = True,
    review_interactive: bool = False,
    live_autosave: bool | None = None,
    impact_polarity: str = "negative",
) -> pd.DataFrame:
    """
    Analyze all raw nanoimpact files in a folder.

    Returns
    -------
    combined_df
        Combined kept plateau-level results from all files.
    """
    paths = sorted(Path(config.folder).glob(config.pattern))

    if live_autosave is None:
        live_autosave = bool(review_interactive)

    out_combined = Path(config.folder) / "combined_results.xlsx"

    if verbose:
        print(f"Found {len(paths)} files in {config.folder}")

    combined_rows = []
    all_small_impacts = []
    all_large_impacts = []
    combined_monoexp_rows = []

    for path in paths:
        small_impacts_pa, large_impacts_pa, per_file_keep_df, monoexp_keep_df = analyze_single_file(
            path=path,
            config=config,
            save_per_file=save_per_file,
            verbose=verbose,
            review_interactive=review_interactive,
            impact_polarity=impact_polarity,
        )

        if small_impacts_pa is not None and len(small_impacts_pa) > 0:
            all_small_impacts.append(small_impacts_pa)

        if large_impacts_pa is not None and len(large_impacts_pa) > 0:
            all_large_impacts.append(large_impacts_pa)

        if per_file_keep_df is not None and not per_file_keep_df.empty:
            combined_rows.extend(per_file_keep_df.to_dict("records"))
        
        if monoexp_keep_df is not None and not monoexp_keep_df.empty:
            combined_monoexp_rows.extend(monoexp_keep_df.to_dict("records"))

        if save_combined and live_autosave and len(combined_rows) > 0:
            _autosave_combined_rows(
                rows=combined_rows,
                monoexp_rows=combined_monoexp_rows,
                out_xlsx=out_combined,
                verbose=verbose,
            )

    if len(combined_rows) == 0:
        if verbose:
            print("\nNo plateau results found.")

        return pd.DataFrame()

    combined_df = pd.DataFrame(combined_rows)
    combined_df = _ensure_keep_for_stats(combined_df)
    combined_monoexp_df = (
        pd.DataFrame(combined_monoexp_rows)
        if len(combined_monoexp_rows) > 0
        else pd.DataFrame()
    )

    if save_combined:
        _save_combined_workbook_safely(
            results_df=combined_df,
            monoexp_df=combined_monoexp_df,
            out_path=out_combined,
        )

        if verbose:
            print(f"\nCombined KEPT plateau results saved to {out_combined}")

    if save_split_files and "delta_i_from_prev_pA" in combined_df.columns:
        selected_df = combined_df.copy()

    if "keep_for_stats" in selected_df.columns:
        selected_df = selected_df[selected_df["keep_for_stats"].astype(bool)]

    delta = selected_df["delta_i_from_prev_pA"].to_numpy(dtype=float)
    selected_mask = _delta_matches_polarity_array(delta, impact_polarity)
    selected_df = selected_df[selected_mask].copy()

    threshold_abs_pa = abs(float(config.detection.small_impact_min_pa))
    abs_delta = selected_df["delta_i_from_prev_pA"].abs()

    small_df = selected_df[abs_delta <= threshold_abs_pa]
    large_df = selected_df[abs_delta > threshold_abs_pa]

    if impact_polarity == "negative":
        out_small = Path(config.folder) / "combined_small_impacts.xlsx"
        out_large = Path(config.folder) / "combined_large_impacts.xlsx"
    else:
        out_small = Path(config.folder) / f"combined_{impact_polarity}_small_impacts.xlsx"
        out_large = Path(config.folder) / f"combined_{impact_polarity}_large_impacts.xlsx"

    _save_excel_safely(small_df, out_small)
    _save_excel_safely(large_df, out_large)

    if verbose:
        print(f"Small {impact_polarity} impacts saved to {out_small}")
        print(f"Large {impact_polarity} impacts saved to {out_large}")

    if verbose:
        print_impact_summary(
            all_small_impacts=all_small_impacts,
            all_large_impacts=all_large_impacts,
            impact_polarity=impact_polarity,
        )

    return combined_df


def print_impact_summary(
    all_small_impacts: list[np.ndarray],
    all_large_impacts: list[np.ndarray],
    impact_polarity: str = "negative",
):
    """
    Print simple summary statistics across all files.
    """
    if len(all_small_impacts) > 0:
        small_impacts_pa = np.concatenate(all_small_impacts)
    else:
        small_impacts_pa = np.array([], dtype=float)

    if len(all_large_impacts) > 0:
        large_impacts_pa = np.concatenate(all_large_impacts)
    else:
        large_impacts_pa = np.array([], dtype=float)

    all_impacts_pa = np.concatenate([small_impacts_pa, large_impacts_pa])

    if all_impacts_pa.size == 0:
        print(f"\nNo {impact_polarity} impacts detected in any file.")
        return

    all_abs_pa = np.abs(all_impacts_pa)

    count_0_40 = int(np.sum((all_abs_pa > 0) & (all_abs_pa < 40)))
    count_40_100 = int(np.sum((all_abs_pa >= 40) & (all_abs_pa <= 100)))
    count_gt_100 = int(np.sum(all_abs_pa > 100))

    print(f"\nImpact summary across all files ({impact_polarity}):")
    print(f"  Total selected impacts: {all_abs_pa.size}")
    print(f"  0–40 pA: {count_0_40}")
    print(f"  40–100 pA: {count_40_100}")
    print(f"  >100 pA: {count_gt_100}")
    print(f"  Mean |Δi|: {np.mean(all_abs_pa):.2f} pA")
    print(f"  Median |Δi|: {np.median(all_abs_pa):.2f} pA")
