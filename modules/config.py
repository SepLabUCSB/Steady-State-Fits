# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:14:28 2026

@author: Ishaan
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class FilterConfig:
    apply_notch_filter: bool = True
    notch_freqs_hz: list[float] = field(default_factory=lambda: [17, 18, 19, 20, 21, 22, 40])
    notch_q: float = 30.0


@dataclass
class DetectionConfig:
    charging_cutoff_min: float = 0.8
    pre_post_win_sec: float = 0.1
    cluster_gap_sec: float = 2.0
    min_plateau_pts: int = 200
    min_real_impact_pa: float = 15.0
    small_impact_min_pa: float = -100.0


@dataclass
class FitConfig:
    exp_fit_win_sec: float = 3.0
    monoexp_r2_threshold: float = 0.95


@dataclass
class PlotConfig:
    plot_mode: Literal["inline", "tk"] = "inline"
    bin_width_pa: float = 5.0
    show_filtered_trace: bool = True


@dataclass
class PipelineConfig:
    folder: Path
    pattern: str = "*.txt"
    filter: FilterConfig = field(default_factory=FilterConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)