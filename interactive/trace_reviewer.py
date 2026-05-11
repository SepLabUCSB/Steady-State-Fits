# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:59:46 2026

@author: Ishaan
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button

from interactive.tk_backend import show_blocking_safely, tk_style_context
from interactive.impact_fit_adjuster import open_impact_fit_adjuster


@dataclass
class ReviewTracePayload:
    path: Path
    time_s: np.ndarray
    time_min: np.ndarray
    current_raw_pa: np.ndarray
    current_filtered_pa: np.ndarray
    step_indices: list[int] | np.ndarray
    i_ss_idx: np.ndarray
    i_ss_final_pa: np.ndarray
    delta_i_final_pa: np.ndarray
    exp_tau_s: np.ndarray
    exp_amplitude_pa: np.ndarray
    exp_i_inf_pa: np.ndarray
    exp_r_squared: np.ndarray
    monoexp_r2_threshold: float
    exp_fit_win_sec: float
    min_plateau_pts: int
    dt_s: float
    drift_slope_pa_per_min: float
    drift_intercept_pa: float
    out_png: Path | None = None


def _midpoint_colors(delta_i_final_pa):
    colors = []

    if len(delta_i_final_pa) > 0:
        colors.append("green")

    for idx in range(1, len(delta_i_final_pa)):
        delta_i_pa = delta_i_final_pa[idx]

        if not np.isfinite(delta_i_pa):
            colors.append("gray")
            continue

        amplitude_pa = abs(delta_i_pa)

        if amplitude_pa < 40:
            colors.append("red")
        elif amplitude_pa <= 100:
            colors.append("green")
        else:
            colors.append("gray")

    return colors


def review_trace(payload: ReviewTracePayload):
    """
    Interactive Tk reviewer.

    Controls
    --------
    Left click:
        Show monoexponential fit if available, otherwise show linear baseline.

    Right click:
        Toggle plateau keep/remove.

    Toggle filter:
        Switch raw/filtered trace.

    Next file:
        Save trace PNG, close window, and return keep mask.
    """
    path = Path(payload.path)

    time_s = np.asarray(payload.time_s, dtype=float)
    time_min = np.asarray(payload.time_min, dtype=float)
    current_raw_pa = np.asarray(payload.current_raw_pa, dtype=float)
    current_filtered_pa = np.asarray(payload.current_filtered_pa, dtype=float)
    step_indices = np.asarray(payload.step_indices, dtype=int)
    i_ss_idx = np.asarray(payload.i_ss_idx, dtype=int)
    i_ss_final_pa = np.asarray(payload.i_ss_final_pa, dtype=float)
    delta_i_final_pa = np.asarray(payload.delta_i_final_pa, dtype=float)

    exp_tau_s = np.asarray(payload.exp_tau_s, dtype=float)
    exp_amplitude_pa = np.asarray(payload.exp_amplitude_pa, dtype=float)
    exp_i_inf_pa = np.asarray(payload.exp_i_inf_pa, dtype=float)
    exp_r_squared = np.asarray(payload.exp_r_squared, dtype=float)

    n_plateaus = len(i_ss_final_pa)
    keep_plateau = np.ones(n_plateaus, dtype=bool)

    if payload.out_png is None:
        out_png = path.with_name(path.stem + "_trace.png")
    else:
        out_png = Path(payload.out_png)

    with tk_style_context():
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(left=0.08, right=0.70, bottom=0.22, top=0.90)

        line_raw, = ax.plot(
            time_min,
            current_raw_pa,
            color="purple",
            linewidth=1.0,
            label="Raw current",
        )

        line_filtered, = ax.plot(
            time_min,
            current_filtered_pa,
            color="black",
            linewidth=1.0,
            label="Notch-filtered",
        )

        line_raw.set_visible(True)
        line_filtered.set_visible(False)

        if len(step_indices) > 0:
            ax.scatter(
                time_min[step_indices],
                current_filtered_pa[step_indices],
                s=18,
                color="cornflowerblue",
                zorder=3,
            )

        midpoint_colors = _midpoint_colors(delta_i_final_pa)

        scatter_mid = ax.scatter(
            time_min[i_ss_idx],
            i_ss_final_pa,
            s=60,
            c=midpoint_colors,
            edgecolor="black",
            zorder=4,
        )

        time_plot = np.linspace(time_min.min(), time_min.max(), 200)
        ax.plot(
            time_plot,
            payload.drift_slope_pa_per_min * time_plot + payload.drift_intercept_pa,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Current (pA)")
        ax.set_title(path.name, fontsize=10)

        legend_handles = [
            Line2D([], [], color="purple", linewidth=1.5, label="Raw current"),
            Line2D([], [], color="black", linewidth=1.5, label="Notch-filtered"),
            Line2D(
                [],
                [],
                marker="o",
                color="cornflowerblue",
                markersize=7,
                linestyle="None",
                label="Step boundaries",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="green",
                markeredgecolor="black",
                markersize=9,
                linestyle="None",
                label="i_ss midpoints",
            ),
            Line2D(
                [],
                [],
                color="black",
                linewidth=2,
                linestyle="--",
                label=(
                    "Drift fit "
                    f"({payload.drift_slope_pa_per_min:.1f} pA/min)"
                ),
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            fancybox=True,
            borderpad=0.6,
            handlelength=2.0,
        )

        xlim_initial = ax.get_xlim()
        ylim_initial = ax.get_ylim()

        overlay_artists = []
        show_filtered = False

        facecolors = scatter_mid.get_facecolors()
        edgecolors = scatter_mid.get_edgecolors()

        if facecolors.shape[0] == 1 and n_plateaus > 1:
            facecolors = np.tile(facecolors[0], (n_plateaus, 1))

        if edgecolors.shape[0] == 1 and n_plateaus > 1:
            edgecolors = np.tile(edgecolors[0], (n_plateaus, 1))

        scatter_mid.set_facecolors(facecolors)
        scatter_mid.set_edgecolors(edgecolors)
        original_facecolors = scatter_mid.get_facecolors().copy()
        def build_trace_df():
            return pd.DataFrame(
                {
                    "time_s": time_s,
                    "time_min": time_min,
                    "current_raw_pa": current_raw_pa,
                    "current_filtered_pa": current_filtered_pa,
                }
            )
        
        def build_impacts_df():
            rows = []
        
            for plateau_idx in range(n_plateaus):
                step_idx = (
                    int(step_indices[plateau_idx])
                    if plateau_idx < len(step_indices)
                    else np.nan
                )
        
                rows.append(
                    {
                        "impact_index": plateau_idx,
                        "source_file": str(path),
                        "step_index": step_idx,
                        "i_ss_index": int(i_ss_idx[plateau_idx]),
                        "time_s": float(time_s[i_ss_idx[plateau_idx]]),
                        "time_min": float(time_min[i_ss_idx[plateau_idx]]),
                        "i_ss_final_pa": float(i_ss_final_pa[plateau_idx]),
                        "delta_i_final_pa": float(delta_i_final_pa[plateau_idx])
                        if plateau_idx < len(delta_i_final_pa)
                        else np.nan,
                        "exp_tau_s": float(exp_tau_s[plateau_idx])
                        if plateau_idx < len(exp_tau_s)
                        else np.nan,
                        "exp_amplitude_pa": float(exp_amplitude_pa[plateau_idx])
                        if plateau_idx < len(exp_amplitude_pa)
                        else np.nan,
                        "exp_i_inf_pa": float(exp_i_inf_pa[plateau_idx])
                        if plateau_idx < len(exp_i_inf_pa)
                        else np.nan,
                        "exp_r_squared": float(exp_r_squared[plateau_idx])
                        if plateau_idx < len(exp_r_squared)
                        else np.nan,
                        "keep_for_stats": bool(keep_plateau[plateau_idx]),
                        "review_status": "unreviewed",
                    }
                )
        
            return pd.DataFrame(rows)
        
        def refresh_keep_colors():
            local_facecolors = scatter_mid.get_facecolors()
            local_edgecolors = scatter_mid.get_edgecolors()
        
            if local_facecolors.shape[0] == 1 and n_plateaus > 1:
                local_facecolors = np.tile(local_facecolors[0], (n_plateaus, 1))
        
            if local_edgecolors.shape[0] == 1 and n_plateaus > 1:
                local_edgecolors = np.tile(local_edgecolors[0], (n_plateaus, 1))
        
            for plateau_idx in range(n_plateaus):
                if keep_plateau[plateau_idx]:
                    delta_value = delta_i_final_pa[plateau_idx]
                
                    if not np.isfinite(delta_value):
                        color = [0.5, 0.5, 0.5, 1.0]
                    else:
                        amplitude = abs(delta_value)
                
                        if amplitude < 40:
                            color = [1.0, 0.0, 0.0, 1.0]      # red
                        elif amplitude <= 100:
                            color = [0.0, 0.5, 0.0, 1.0]      # green
                        else:
                            color = [0.5, 0.5, 0.5, 1.0]      # gray
                
                    local_facecolors[plateau_idx] = color
                    local_edgecolors[plateau_idx] = [0, 0, 0, 1]
                
                else:
                    local_facecolors[plateau_idx] = [0.7, 0.7, 0.7, 0.2]
                    local_edgecolors[plateau_idx] = [0.3, 0.3, 0.3, 1.0]
        
            scatter_mid.set_facecolors(local_facecolors)
            scatter_mid.set_edgecolors(local_edgecolors)
            fig.canvas.draw_idle()
        
        def update_from_fit_adjuster(reviewed_impacts_df):
            if reviewed_impacts_df is None:
                return
        
            # Update keep mask.
            if "keep_for_stats" in reviewed_impacts_df.columns:
                keep_values = reviewed_impacts_df["keep_for_stats"].to_numpy(dtype=bool)
                n_update = min(len(keep_values), len(keep_plateau))
                keep_plateau[:n_update] = keep_values[:n_update]
        
            # Update reviewed delta values locally.
            if "delta_i_reviewed_final_pA" in reviewed_impacts_df.columns:
                reviewed_delta = reviewed_impacts_df["delta_i_reviewed_final_pA"].to_numpy(dtype=float)
            elif "review_delta_i_pA" in reviewed_impacts_df.columns:
                reviewed_delta = reviewed_impacts_df["review_delta_i_pA"].to_numpy(dtype=float)
            else:
                reviewed_delta = None
        
            if reviewed_delta is not None:
                n_update = min(len(reviewed_delta), len(delta_i_final_pa))
        
                reviewed_slice = reviewed_delta[:n_update]
                valid = np.isfinite(reviewed_slice)
                update_idx = np.arange(n_update)[valid]
        
                # Important: direct indexing, not chained slicing.
                delta_i_final_pa[update_idx] = reviewed_slice[valid]
        
                try:
                    payload.delta_i_final_pa[update_idx] = reviewed_slice[valid]
                except Exception:
                    pass
        
            refresh_keep_colors()
            fig.canvas.draw_idle()
        
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
        
            print(f"    Live-updated reviewed fits for {path.name}")
        fit_adjuster_apps = []
        def on_open_fit_adjuster(event):
            if n_plateaus == 0:
                print("    No impacts available for fit review.")
                return
        
            trace_df = build_trace_df()
            impacts_df = build_impacts_df()
        
            try:
                parent = fig.canvas.manager.window
            except Exception:
                parent = None
        
            def restore_parent_window():
                try:
                    if parent is not None:
                        parent.deiconify()
                        parent.lift()
                except Exception:
                    pass
        
            def on_adjuster_save(reviewed_impacts_df):
                update_from_fit_adjuster(reviewed_impacts_df)
        
            app = open_impact_fit_adjuster(
                parent=None,
                trace_df=trace_df,
                impacts_df=impacts_df,
                source_path=path,
                on_save=on_adjuster_save,
                on_update=update_from_fit_adjuster,
                on_close=restore_parent_window,
            )
            
            # Keep a reference so the Tk callbacks/window do not get orphaned.
            fit_adjuster_apps.append(app)

        def clear_overlays():
            nonlocal overlay_artists

            for artist in overlay_artists:
                try:
                    artist.remove()
                except Exception:
                    pass

            overlay_artists = []

        def on_click(event):
            nonlocal overlay_artists

            if event.inaxes is None or event.inaxes != ax:
                return

            if event.xdata is None:
                return

            clicked_time_s = event.xdata * 60.0
            midpoint_times_s = time_min[i_ss_idx] * 60.0
            plateau_idx = int(np.argmin(np.abs(midpoint_times_s - clicked_time_s)))

            if plateau_idx < 0 or plateau_idx >= n_plateaus:
                return

            # Right click: keep/remove
            if event.button == 3:
                keep_plateau[plateau_idx] = not keep_plateau[plateau_idx]

                if keep_plateau[plateau_idx]:
                    state = "KEPT"
                else:
                    state = "REMOVED"

                refresh_keep_colors()

                print(f"    Plateau {plateau_idx} ({path.name}): {state}")
                return

            # Left click: overlay fit or linear baseline
            clear_overlays()

            use_exp = (
                plateau_idx >= 1
                and plateau_idx < len(step_indices)
                and np.isfinite(exp_r_squared[plateau_idx])
                and exp_r_squared[plateau_idx] >= payload.monoexp_r2_threshold
                and np.isfinite(exp_i_inf_pa[plateau_idx])
                and np.isfinite(exp_amplitude_pa[plateau_idx])
                and np.isfinite(exp_tau_s[plateau_idx])
                and exp_tau_s[plateau_idx] > 0
            )

            if use_exp:
                step_idx = int(step_indices[plateau_idx])

                pre_pts = int(0.2 / payload.dt_s)
                post_pts = max(
                    int(payload.exp_fit_win_sec / payload.dt_s),
                    int(payload.min_plateau_pts),
                )

                start_idx = max(0, step_idx - pre_pts)
                end_idx = min(len(time_s), step_idx + post_pts)

                fit_time_s = time_s[start_idx:end_idx] - time_s[start_idx]
                fit_current_pa = current_filtered_pa[start_idx:end_idx]

                current_inf_pa = exp_i_inf_pa[plateau_idx]
                amplitude_pa = exp_amplitude_pa[plateau_idx]
                tau_s = exp_tau_s[plateau_idx]

                model_current_pa = current_inf_pa + amplitude_pa * np.exp(
                    -fit_time_s / tau_s
                )

                line_data, = ax.plot(
                    (fit_time_s + time_s[start_idx]) / 60.0,
                    fit_current_pa,
                    color="orange",
                    linewidth=1.2,
                )

                line_model, = ax.plot(
                    (fit_time_s + time_s[start_idx]) / 60.0,
                    model_current_pa,
                    color="black",
                    linewidth=2.0,
                )

                overlay_artists.extend([line_data, line_model])
                fig.canvas.draw_idle()

                print(
                    f"    Plateau {plateau_idx}: "
                    f"exp fit R² = {exp_r_squared[plateau_idx]:.4f}"
                )
                return

            # Linear fallback
            center_idx = int(i_ss_idx[plateau_idx])
            start_idx = max(0, center_idx - int(2.0 / payload.dt_s))
            end_idx = min(len(time_s) - 1, center_idx + int(2.0 / payload.dt_s))

            zoom_time_s = time_s[start_idx:end_idx]
            zoom_current_pa = current_filtered_pa[start_idx:end_idx]
            level_pa = i_ss_final_pa[plateau_idx]

            line_data, = ax.plot(
                zoom_time_s / 60.0,
                zoom_current_pa,
                color="orange",
                linewidth=1.2,
            )

            line_level, = ax.plot(
                zoom_time_s[[0, -1]] / 60.0,
                [level_pa, level_pa],
                color="black",
                linewidth=2.0,
            )

            overlay_artists.extend([line_data, line_level])
            fig.canvas.draw_idle()

            print(f"    Plateau {plateau_idx}: linear baseline")

        fig.canvas.mpl_connect("button_press_event", on_click)
        
        ax_fit_adjuster = fig.add_axes([0.56, 0.06, 0.14, 0.10])
        button_fit_adjuster = Button(ax_fit_adjuster, "Review fits")
        button_fit_adjuster.on_clicked(on_open_fit_adjuster)

        ax_toggle = fig.add_axes([0.72, 0.06, 0.12, 0.10])
        button_toggle = Button(ax_toggle, "Toggle filter")

        def on_toggle(event):
            nonlocal show_filtered

            show_filtered = not show_filtered

            if show_filtered:
                line_raw.set_visible(False)
                line_filtered.set_visible(True)
            else:
                line_raw.set_visible(True)
                line_filtered.set_visible(False)

            fig.canvas.draw_idle()

        button_toggle.on_clicked(on_toggle)

        ax_next = fig.add_axes([0.86, 0.06, 0.12, 0.10])
        button_next = Button(ax_next, "Next file")

        close_timers = []

        def on_next(event):
            try:
                ax.set_xlim(xlim_initial)
                ax.set_ylim(ylim_initial)
                clear_overlays()
                fig.canvas.draw_idle()

                out_png.parent.mkdir(parents=True, exist_ok=True)

                # Hide buttons from saved trace image.
                ax_next.set_visible(False)
                ax_toggle.set_visible(False)
                ax_fit_adjuster.set_visible(False)

                fig.savefig(out_png, dpi=400, bbox_inches="tight")

                ax_next.set_visible(True)
                ax_toggle.set_visible(True)
                ax_fit_adjuster.set_visible(True)

                print(f"  Saved trace figure to {out_png}")

            except Exception as exc:
                print(f"  Error while saving trace figure: {exc}")

            def close_figure():
                try:
                    plt.close(fig)
                except Exception as exc:
                    print(f"  Warning: could not close figure cleanly: {exc}")
                return False

            timer = fig.canvas.new_timer(interval=100)
            timer.add_callback(close_figure)
            close_timers.append(timer)
            timer.start()

        button_next.on_clicked(on_next)

        # This is required. Without it, the Tk window never opens.
        show_blocking_safely(fig)

    
    return keep_plateau