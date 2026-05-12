# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:23:07 2026

@author: Ishaan
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    step_indices: list[int]
    threshold_pa: float
    noise_std_pa: float
    candidate_indices: np.ndarray
    cluster_gap_pts: int


def compute_step_magnitude(
    time_s,
    current_pa,
    pre_post_win_sec: float = 0.1,
    smooth_window_pts: int = 7,
    detrend: bool = True,
):
    """
    Compute the legacy linear-impact step magnitude signal.

    Workflow:
        1. Optionally linearly detrend current vs time in minutes.
        2. Smooth the detrended current with a centered rolling mean.
        3. Compute post-window mean minus pre-window mean.

    Returns
    -------
    step_magnitude_pa
        Array containing local step magnitude estimates in pA.

    window_pts
        Number of points used on each side of the candidate step.
    """
    time_s = np.asarray(time_s, dtype=float)
    current_pa = np.asarray(current_pa, dtype=float)

    if time_s.size != current_pa.size:
        raise ValueError("time_s and current_pa must have the same length.")

    if time_s.size < 5:
        return np.full(time_s.size, np.nan), 0

    dt_s = float(np.median(np.diff(time_s)))
    time_min = time_s / 60.0

    if detrend:
        drift_slope, drift_intercept = np.polyfit(time_min, current_pa, 1)
        current_for_detection = current_pa - (
            drift_slope * time_min + drift_intercept
        )
    else:
        current_for_detection = current_pa.copy()

    current_smooth = (
        pd.Series(current_for_detection)
        .rolling(smooth_window_pts, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    window_pts = max(5, int(pre_post_win_sec / dt_s))
    n_points = len(current_smooth)

    step_magnitude_pa = np.full(n_points, np.nan)

    for idx in range(window_pts, n_points - window_pts - 1):
        pre_mean = current_smooth[idx - window_pts:idx].mean()
        post_mean = current_smooth[idx + 1:idx + 1 + window_pts].mean()
        step_magnitude_pa[idx] = post_mean - pre_mean

    return step_magnitude_pa, window_pts


def _cluster_candidate_indices(
    candidate_indices,
    cluster_gap_pts: int,
    step_magnitude_pa=None,
    impact_polarity: str = "negative",
):
    candidate_indices = np.asarray(candidate_indices, dtype=int)

    if candidate_indices.size == 0:
        return []

    candidate_indices = np.sort(candidate_indices)

    impact_polarity = str(impact_polarity).strip().lower()

    clusters = []
    current_cluster = [int(candidate_indices[0])]

    for idx in candidate_indices[1:]:
        idx = int(idx)

        if idx - current_cluster[-1] <= cluster_gap_pts:
            current_cluster.append(idx)
        else:
            clusters.append(np.asarray(current_cluster, dtype=int))
            current_cluster = [idx]

    clusters.append(np.asarray(current_cluster, dtype=int))

    step_indices = []

    for cluster in clusters:
        if step_magnitude_pa is None:
            # Backward-compatible fallback.
            step_indices.append(int(cluster[-1]))
            continue

        values = np.asarray(step_magnitude_pa, dtype=float)[cluster]
        finite = np.isfinite(values)

        if not np.any(finite):
            step_indices.append(int(cluster[-1]))
            continue
        
        if impact_polarity == "negative":
            # Strongest downward step = most negative value.
            best_local_idx = int(np.nanargmin(values))

        elif impact_polarity == "positive":
            # Strongest upward step = most positive value.
            best_local_idx = int(np.nanargmax(values))

        elif impact_polarity == "both":
            # Strongest step of either sign.
            best_local_idx = int(np.nanargmax(np.abs(values)))

        else:
            raise ValueError(
                f"Unknown impact_polarity '{impact_polarity}'. "
                "Use 'negative', 'positive', or 'both'."
            )

        step_indices.append(int(cluster[best_local_idx]))

    return step_indices


def detect_step_indices(
    time_s,
    step_magnitude_pa,
    charging_cutoff_min: float = 0.8,
    cluster_gap_sec: float = 2.0,
    min_real_impact_pa: float = 15.0,
    noise_region_abs_max_pa: float = 20.0,
    default_noise_std_pa: float = 4.0,
    impact_polarity: str = "negative",
):
    """
    Detect nanoimpact steps using polarity-aware threshold logic.

    impact_polarity:
        "negative" -> detect downward/negative steps
        "positive" -> detect upward/positive steps
        "both"     -> detect both directions

    Noise region:
        step_magnitude where |step_magnitude| < noise_region_abs_max_pa

    Candidate steps are clustered, and the strongest index in each cluster is
    used as the final step boundary.
    """
    time_s = np.asarray(time_s, dtype=float)
    step_magnitude_pa = np.asarray(step_magnitude_pa, dtype=float)

    impact_polarity = str(impact_polarity).strip().lower()

    if impact_polarity not in {"negative", "positive", "both"}:
        raise ValueError(
            f"Unknown impact_polarity '{impact_polarity}'. "
            "Use 'negative', 'positive', or 'both'."
        )

    if time_s.size != step_magnitude_pa.size:
        raise ValueError("time_s and step_magnitude_pa must have the same length.")

    if time_s.size < 2:
        return DetectionResult(
            step_indices=[],
            threshold_pa=np.nan,
            noise_std_pa=np.nan,
            candidate_indices=np.array([], dtype=int),
            cluster_gap_pts=0,
        )

    time_min = time_s / 60.0
    dt_s = float(np.median(np.diff(time_s)))

    valid_mask = (time_min > charging_cutoff_min) & np.isfinite(step_magnitude_pa)

    noise_region = step_magnitude_pa[
        valid_mask & (np.abs(step_magnitude_pa) < noise_region_abs_max_pa)
    ]

    if noise_region.size == 0:
        noise_std_pa = float(default_noise_std_pa)
    else:
        noise_std_pa = float(np.nanstd(noise_region))

    threshold_abs_pa = max(3.0 * noise_std_pa, float(min_real_impact_pa))

    if impact_polarity == "negative":
        threshold_pa = -threshold_abs_pa
        candidate_mask = (
            (step_magnitude_pa <= threshold_pa)
            & (time_min > charging_cutoff_min)
            & np.isfinite(step_magnitude_pa)
        )

    elif impact_polarity == "positive":
        threshold_pa = threshold_abs_pa
        candidate_mask = (
            (step_magnitude_pa >= threshold_pa)
            & (time_min > charging_cutoff_min)
            & np.isfinite(step_magnitude_pa)
        )

    else:  # both
        threshold_pa = threshold_abs_pa
        candidate_mask = (
            (np.abs(step_magnitude_pa) >= threshold_abs_pa)
            & (time_min > charging_cutoff_min)
            & np.isfinite(step_magnitude_pa)
        )

    candidate_indices = np.where(candidate_mask)[0]

    cluster_gap_pts = max(5, int(cluster_gap_sec / dt_s))

    step_indices = _cluster_candidate_indices(
        candidate_indices=candidate_indices,
        cluster_gap_pts=cluster_gap_pts,
        step_magnitude_pa=step_magnitude_pa,
        impact_polarity=impact_polarity,
    )

    return DetectionResult(
        step_indices=step_indices,
        threshold_pa=float(threshold_pa),
        noise_std_pa=float(noise_std_pa),
        candidate_indices=np.asarray(candidate_indices, dtype=int),
        cluster_gap_pts=int(cluster_gap_pts),
    )