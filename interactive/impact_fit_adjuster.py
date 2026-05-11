# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:50:26 2026

@author: Ishaan
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def open_impact_fit_adjuster(
    parent,
    trace_df,
    impacts_df,
    source_path=None,
    on_save=None,
    on_update=None,
    on_close=None,
):
    # Use the existing Tk parent if available.
    # Do not withdraw or hide the parent.
    if parent is not None:
        window = tk.Toplevel(parent)
    else:
        window = tk.Toplevel()

    window.title("Individual Impact Fit Adjuster")
    window.geometry("1150x800")

    app = ImpactFitAdjuster(
        window=window,
        trace_df=trace_df,
        impacts_df=impacts_df,
        source_path=source_path,
        on_save=on_save,
        on_update=on_update,
        on_close=on_close,
    )

    # Important: keep the app object alive.
    window._impact_fit_adjuster_app = app

    # Bring adjuster to front without hiding the parent.
    try:
        window.lift()
        window.focus_force()
        window.attributes("-topmost", True)
        window.after(300, lambda: window.attributes("-topmost", False))
    except Exception:
        pass

    return app


class ImpactFitAdjuster:
    def __init__(
        self,
        window,
        trace_df,
        impacts_df,
        source_path=None,
        on_save=None,
        on_update=None,
        on_close=None,
    ):
        self.window = window
        self.trace_df = trace_df.copy()
        self.impacts_df = impacts_df.copy()
        self.source_path = Path(source_path) if source_path is not None else None
        self.on_save = on_save
        self.on_close = on_close
        self.on_update = on_update
        self.current_index = 0
        self.mode = tk.StringVar(value="pre_start")
        self.status_var = tk.StringVar()
        self.fit_info_var = tk.StringVar()

        self.time_s = self.trace_df["time_s"].to_numpy(dtype=float)

        if "current_filtered_pa" in self.trace_df.columns:
            self.current_pa = self.trace_df["current_filtered_pa"].to_numpy(dtype=float)
        elif "current_raw_pa" in self.trace_df.columns:
            self.current_pa = self.trace_df["current_raw_pa"].to_numpy(dtype=float)
        else:
            raise KeyError(
                "trace_df must contain 'current_filtered_pa' or 'current_raw_pa'."
            )

        self._ensure_review_columns()
        self._initialize_default_windows()

        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        self._build_layout()
        self._draw_current_impact()

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    def _ensure_review_columns(self):
        defaults = {
            "review_status": "unreviewed",
            "keep_for_stats": True,
            "review_pre_start_s": np.nan,
            "review_pre_end_s": np.nan,
            "review_post_start_s": np.nan,
            "review_post_end_s": np.nan,
            "review_eval_time_s": np.nan,
            "review_pre_slope_pa_per_s": np.nan,
            "review_pre_intercept_pa": np.nan,
            "review_post_slope_pa_per_s": np.nan,
            "review_post_intercept_pa": np.nan,
            "review_pre_r2": np.nan,
            "review_post_r2": np.nan,
            "review_pre_n": np.nan,
            "review_post_n": np.nan,
            "review_delta_i_pA": np.nan,
            "review_pre_value_pA": np.nan,
            "review_post_value_pA": np.nan,
        }

        for col, default in defaults.items():
            if col not in self.impacts_df.columns:
                self.impacts_df[col] = default
    
    def _update_final_delta_columns(self):
        """
        Keep original automatic delta as backup, and make reviewed delta
        the final delta used by parent/downstream code.
        """
        if "auto_delta_i_final_pa" not in self.impacts_df.columns:
            if "delta_i_final_pa" in self.impacts_df.columns:
                self.impacts_df["auto_delta_i_final_pa"] = self.impacts_df[
                    "delta_i_final_pa"
                ]
    
        if "review_delta_i_pA" in self.impacts_df.columns:
            reviewed_delta = pd.to_numeric(
                self.impacts_df["review_delta_i_pA"],
                errors="coerce",
            )
        else:
            reviewed_delta = pd.Series(np.nan, index=self.impacts_df.index)
    
        if "auto_delta_i_final_pa" in self.impacts_df.columns:
            auto_delta = pd.to_numeric(
                self.impacts_df["auto_delta_i_final_pa"],
                errors="coerce",
            )
        elif "delta_i_final_pa" in self.impacts_df.columns:
            auto_delta = pd.to_numeric(
                self.impacts_df["delta_i_final_pa"],
                errors="coerce",
            )
        else:
            auto_delta = pd.Series(np.nan, index=self.impacts_df.index)
    
        self.impacts_df["delta_i_reviewed_final_pA"] = reviewed_delta.where(
            np.isfinite(reviewed_delta),
            auto_delta,
        )
    
        # Compatibility with old downstream code.
        self.impacts_df["delta_i_final_pa"] = self.impacts_df[
            "delta_i_reviewed_final_pA"
        ]
    
        if "delta_i_from_prev_pA" in self.impacts_df.columns:
            self.impacts_df["delta_i_from_prev_pA"] = self.impacts_df[
                "delta_i_reviewed_final_pA"
            ]
    def notify_parent_update(self):
        """
        Push current reviewed values back to parent trace reviewer.
        """
        idx = self._get_row_index()
        self._refit_row(idx)
        self._update_final_delta_columns()

        if self.on_update is None:
            print("DEBUG: no on_update callback connected")
            return

        try:
            print("DEBUG: notifying parent from fit adjuster")
            self.on_update(self.impacts_df.copy())
        except Exception as exc:
            print(f"Warning: could not update parent trace reviewer: {exc}")

    def _get_row_index(self):
        return self.impacts_df.index[self.current_index]

    def _get_impact_time_s(self, row):
        if "step_index" in row and np.isfinite(row["step_index"]):
            step_idx = int(row["step_index"])

            if 0 <= step_idx < len(self.time_s):
                return float(self.time_s[step_idx])

        if "time_s" in row and np.isfinite(row["time_s"]):
            return float(row["time_s"])

        if "i_ss_index" in row and np.isfinite(row["i_ss_index"]):
            i_ss_idx = int(row["i_ss_index"])

            if 0 <= i_ss_idx < len(self.time_s):
                return float(self.time_s[i_ss_idx])

        raise ValueError("Could not determine impact time.")

    def _get_eval_time_s(self, row):
        if "i_ss_index" in row and np.isfinite(row["i_ss_index"]):
            i_ss_idx = int(row["i_ss_index"])

            if 0 <= i_ss_idx < len(self.time_s):
                return float(self.time_s[i_ss_idx])

        if "time_s" in row and np.isfinite(row["time_s"]):
            return float(row["time_s"])

        return self._get_impact_time_s(row)

    def _initialize_default_windows(self):
        """
        Set initial pre/post regression windows.

        These are intentionally only starting guesses. The user can adjust them
        one impact at a time.
        """
        trace_start = float(np.nanmin(self.time_s))
        trace_end = float(np.nanmax(self.time_s))

        default_pre_window_s = 20.0
        default_post_window_s = 20.0
        impact_gap_s = 0.30

        for idx in self.impacts_df.index:
            row = self.impacts_df.loc[idx]

            impact_time_s = self._get_impact_time_s(row)
            eval_time_s = self._get_eval_time_s(row)

            pre_end_s = max(trace_start, impact_time_s - impact_gap_s)
            pre_start_s = max(trace_start, pre_end_s - default_pre_window_s)

            post_start_s = max(trace_start, eval_time_s - default_post_window_s / 2.0)
            post_end_s = min(trace_end, eval_time_s + default_post_window_s / 2.0)

            if not np.isfinite(self.impacts_df.loc[idx, "review_pre_start_s"]):
                self.impacts_df.loc[idx, "review_pre_start_s"] = pre_start_s

            if not np.isfinite(self.impacts_df.loc[idx, "review_pre_end_s"]):
                self.impacts_df.loc[idx, "review_pre_end_s"] = pre_end_s

            if not np.isfinite(self.impacts_df.loc[idx, "review_post_start_s"]):
                self.impacts_df.loc[idx, "review_post_start_s"] = post_start_s

            if not np.isfinite(self.impacts_df.loc[idx, "review_post_end_s"]):
                self.impacts_df.loc[idx, "review_post_end_s"] = post_end_s

            if not np.isfinite(self.impacts_df.loc[idx, "review_eval_time_s"]):
                self.impacts_df.loc[idx, "review_eval_time_s"] = eval_time_s

        for idx in self.impacts_df.index:
            self._refit_row(idx)

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------
    def _build_layout(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Button(
            control_frame,
            text="Previous",
            command=self.previous_impact,
        ).pack(side="left", padx=3)

        ttk.Button(
            control_frame,
            text="Next",
            command=self.next_impact,
        ).pack(side="left", padx=3)

        ttk.Button(
            control_frame,
            text="Accept",
            command=self.accept_current_fit,
        ).pack(side="left", padx=3)

        ttk.Button(
            control_frame,
            text="Reject",
            command=self.reject_current_impact,
        ).pack(side="left", padx=3)

        ttk.Button(
            control_frame,
            text="Reset auto window",
            command=self.reset_current_window,
        ).pack(side="left", padx=3)

        ttk.Button(
            control_frame,
            text="Save + close",
            command=self.save_and_close,
        ).pack(side="left", padx=3)
        
        ttk.Button(
            control_frame,
            text="Close",
            command=self.close_window,
        ).pack(side="left", padx=3)

        mode_frame = ttk.LabelFrame(main_frame, text="Click mode: choose boundary, then click on trace")
        mode_frame.pack(side="top", fill="x", padx=8, pady=4)

        modes = [
            ("Pre start", "pre_start"),
            ("Pre end", "pre_end"),
            ("Post start", "post_start"),
            ("Post end", "post_end"),
            ("Eval time", "eval_time"),
        ]

        for text, value in modes:
            ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode,
                value=value,
            ).pack(side="left", padx=8)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(side="top", fill="x", padx=8, pady=3)

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        ttk.Label(
            status_frame,
            textvariable=self.fit_info_var,
            anchor="e",
        ).pack(side="right")

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.window.bind("<Key>", self.on_keypress)

    # ------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------
    def _fit_line(self, start_s, end_s):
        lo = min(start_s, end_s)
        hi = max(start_s, end_s)

        mask = (
            np.isfinite(self.time_s)
            & np.isfinite(self.current_pa)
            & (self.time_s >= lo)
            & (self.time_s <= hi)
        )

        x = self.time_s[mask]
        y = self.current_pa[mask]

        if x.size < 2:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "n": int(x.size),
                "x": x,
                "y": y,
            }

        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "n": int(x.size),
            "x": x,
            "y": y,
        }

    def _refit_row(self, idx):
        row = self.impacts_df.loc[idx]

        pre_start = float(row["review_pre_start_s"])
        pre_end = float(row["review_pre_end_s"])
        post_start = float(row["review_post_start_s"])
        post_end = float(row["review_post_end_s"])
        eval_time = float(row["review_eval_time_s"])

        pre_fit = self._fit_line(pre_start, pre_end)
        post_fit = self._fit_line(post_start, post_end)

        pre_value = pre_fit["slope"] * eval_time + pre_fit["intercept"]
        post_value = post_fit["slope"] * eval_time + post_fit["intercept"]
        delta = post_value - pre_value

        self.impacts_df.loc[idx, "review_pre_slope_pa_per_s"] = pre_fit["slope"]
        self.impacts_df.loc[idx, "review_pre_intercept_pa"] = pre_fit["intercept"]
        self.impacts_df.loc[idx, "review_post_slope_pa_per_s"] = post_fit["slope"]
        self.impacts_df.loc[idx, "review_post_intercept_pa"] = post_fit["intercept"]
        self.impacts_df.loc[idx, "review_pre_r2"] = pre_fit["r2"]
        self.impacts_df.loc[idx, "review_post_r2"] = post_fit["r2"]
        self.impacts_df.loc[idx, "review_pre_n"] = pre_fit["n"]
        self.impacts_df.loc[idx, "review_post_n"] = post_fit["n"]
        self.impacts_df.loc[idx, "review_pre_value_pA"] = pre_value
        self.impacts_df.loc[idx, "review_post_value_pA"] = post_value
        self.impacts_df.loc[idx, "review_delta_i_pA"] = delta

        return pre_fit, post_fit

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def _draw_current_impact(self):
        self.ax.clear()

        idx = self._get_row_index()
        row = self.impacts_df.loc[idx]

        pre_fit, post_fit = self._refit_row(idx)

        pre_start = float(row["review_pre_start_s"])
        pre_end = float(row["review_pre_end_s"])
        post_start = float(row["review_post_start_s"])
        post_end = float(row["review_post_end_s"])
        eval_time = float(row["review_eval_time_s"])

        impact_time = self._get_impact_time_s(row)

        left = max(float(np.nanmin(self.time_s)), pre_start - 5.0)
        right = min(float(np.nanmax(self.time_s)), post_end + 5.0)

        mask = (
            np.isfinite(self.time_s)
            & np.isfinite(self.current_pa)
            & (self.time_s >= left)
            & (self.time_s <= right)
        )

        self.ax.plot(
            self.time_s[mask] / 60.0,
            self.current_pa[mask],
            color="purple",
            linewidth=1.0,
            label="Filtered trace",
        )

        # Highlight pre/post points used in fits.
        self.ax.scatter(
            pre_fit["x"] / 60.0,
            pre_fit["y"],
            s=12,
            color="orange",
            alpha=0.8,
            label="Pre-fit points",
        )

        self.ax.scatter(
            post_fit["x"] / 60.0,
            post_fit["y"],
            s=12,
            color="gold",
            alpha=0.8,
            label="Post-fit points",
        )

        # Red linear fits.
        if np.isfinite(pre_fit["slope"]):
            x_pre = np.linspace(pre_start, pre_end, 50)
            y_pre = pre_fit["slope"] * x_pre + pre_fit["intercept"]

            self.ax.plot(
                x_pre / 60.0,
                y_pre,
                color="red",
                linewidth=3.0,
                label="Linear fits",
            )

        if np.isfinite(post_fit["slope"]):
            x_post = np.linspace(post_start, post_end, 50)
            y_post = post_fit["slope"] * x_post + post_fit["intercept"]

            self.ax.plot(
                x_post / 60.0,
                y_post,
                color="red",
                linewidth=3.0,
            )

        # Boundary lines.
        boundaries = [
            (pre_start, "pre start"),
            (pre_end, "pre end"),
            (post_start, "post start"),
            (post_end, "post end"),
        ]

        for boundary_s, name in boundaries:
            self.ax.axvline(
                boundary_s / 60.0,
                color="black",
                linestyle=":",
                linewidth=1.4,
            )

        self.ax.axvline(
            impact_time / 60.0,
            color="cornflowerblue",
            linestyle="--",
            linewidth=1.4,
            label="impact time",
        )

        self.ax.axvline(
            eval_time / 60.0,
            color="green",
            linestyle="--",
            linewidth=1.4,
            label="eval time",
        )

        delta = self.impacts_df.loc[idx, "review_delta_i_pA"]
        keep = bool(self.impacts_df.loc[idx, "keep_for_stats"])
        status = self.impacts_df.loc[idx, "review_status"]

        self.ax.set_title(
            f"Impact {self.current_index + 1} of {len(self.impacts_df)} | "
            f"Δi_review = {delta:.2f} pA | keep={keep} | {status}"
        )

        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Current (pA)")
        self.ax.legend(frameon=True, fontsize=8, loc="best")
        self.ax.grid(alpha=0.25)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.status_var.set(
            "Click mode: "
            f"{self.mode.get()} | "
            "keys: 1 pre-start, 2 pre-end, 3 post-start, 4 post-end, "
            "5 eval, A accept, R reject, N next, P previous, S save"
        )

        self.fit_info_var.set(
            f"pre n={pre_fit['n']}, R²={pre_fit['r2']:.3f} | "
            f"post n={post_fit['n']}, R²={post_fit['r2']:.3f}"
        )

    # ------------------------------------------------------------
    # User interactions
    # ------------------------------------------------------------
    def on_plot_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.xdata is None:
            return

        clicked_s = float(event.xdata) * 60.0
        idx = self._get_row_index()
        mode = self.mode.get()

        if mode == "pre_start":
            self.impacts_df.loc[idx, "review_pre_start_s"] = clicked_s

        elif mode == "pre_end":
            self.impacts_df.loc[idx, "review_pre_end_s"] = clicked_s

        elif mode == "post_start":
            self.impacts_df.loc[idx, "review_post_start_s"] = clicked_s

        elif mode == "post_end":
            self.impacts_df.loc[idx, "review_post_end_s"] = clicked_s

        elif mode == "eval_time":
            self.impacts_df.loc[idx, "review_eval_time_s"] = clicked_s

        self.impacts_df.loc[idx, "review_status"] = "edited"
        self._draw_current_impact()
        self.notify_parent_update()

    def on_keypress(self, event):
        key = event.char.lower()

        if key == "1":
            self.mode.set("pre_start")
        elif key == "2":
            self.mode.set("pre_end")
        elif key == "3":
            self.mode.set("post_start")
        elif key == "4":
            self.mode.set("post_end")
        elif key == "5":
            self.mode.set("eval_time")
        elif key == "a":
            self.accept_current_fit()
            return
        elif key == "r":
            self.reject_current_impact()
            return
        elif key == "n":
            self.next_impact()
            return
        elif key == "p":
            self.previous_impact()
            return
        elif key == "s":
            self.save_reviewed_fits()
            return

        self._draw_current_impact()

    def previous_impact(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._draw_current_impact()

    def next_impact(self):
        if self.current_index < len(self.impacts_df) - 1:
            self.current_index += 1
            self._draw_current_impact()

    def accept_current_fit(self):
        idx = self._get_row_index()
        self.impacts_df.loc[idx, "review_status"] = "accepted"
        self.impacts_df.loc[idx, "keep_for_stats"] = True
        self._draw_current_impact()
        self.notify_parent_update()

    def reject_current_impact(self):
        idx = self._get_row_index()
        self.impacts_df.loc[idx, "review_status"] = "rejected"
        self.impacts_df.loc[idx, "keep_for_stats"] = False
        self._draw_current_impact()
        self.notify_parent_update()

    def reset_current_window(self):
        idx = self._get_row_index()

        for col in [
            "review_pre_start_s",
            "review_pre_end_s",
            "review_post_start_s",
            "review_post_end_s",
            "review_eval_time_s",
        ]:
            self.impacts_df.loc[idx, col] = np.nan

        # Reinitialize only this row.
        row = self.impacts_df.loc[idx]
        trace_start = float(np.nanmin(self.time_s))
        trace_end = float(np.nanmax(self.time_s))

        impact_time_s = self._get_impact_time_s(row)
        eval_time_s = self._get_eval_time_s(row)

        pre_end_s = max(trace_start, impact_time_s - 0.30)
        pre_start_s = max(trace_start, pre_end_s - 20.0)

        post_start_s = max(trace_start, eval_time_s - 10.0)
        post_end_s = min(trace_end, eval_time_s + 10.0)

        self.impacts_df.loc[idx, "review_pre_start_s"] = pre_start_s
        self.impacts_df.loc[idx, "review_pre_end_s"] = pre_end_s
        self.impacts_df.loc[idx, "review_post_start_s"] = post_start_s
        self.impacts_df.loc[idx, "review_post_end_s"] = post_end_s
        self.impacts_df.loc[idx, "review_eval_time_s"] = eval_time_s
        self.impacts_df.loc[idx, "review_status"] = "reset"

        self._draw_current_impact()
        self.notify_parent_update()
        
    def save_and_close(self):
        self.save_reviewed_fits(close_after=True)

    def save_reviewed_fits(self, close_after=False):
        # Refit all rows before saving.
        for idx in self.impacts_df.index:
            self._refit_row(idx)
            
        self._update_final_delta_columns()
    
        if self.source_path is not None:
            out_path = self.source_path.with_name(
                self.source_path.stem + "_reviewed_fits.xlsx"
            )
        else:
            out_path = Path("reviewed_fits.xlsx")
    
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.impacts_df.to_excel(out_path, index=False)
    
        if self.on_save is not None:
            self.on_save(self.impacts_df)
        
        self.notify_parent_update()
    
        print(f"Saved reviewed fits to {out_path}")
    
        if close_after:
            self.close_window()
        else:
            messagebox.showinfo(
                "Saved",
                f"Reviewed fits saved to:\n{out_path}",
            )

    def close_window(self):
        try:
            if self.on_close is not None:
                self.on_close()
        except Exception as exc:
            print(f"Warning: could not restore parent window: {exc}")
    
        try:
            self.window.destroy()
        except Exception as exc:
            print(f"Warning: could not close fit-adjuster window: {exc}")