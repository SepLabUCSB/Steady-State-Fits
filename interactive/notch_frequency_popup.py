# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:45:59 2026

@author: Ishaan
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

from modules.filters import notch_filter


class NotchFrequencyPopup:
    """
    Popup that computes FFT peaks and lets user apply custom notch frequencies
    back to the parent trace reviewer.
    """

    def __init__(
        self,
        parent_window,
        time_s,
        time_min,
        current_raw_pa,
        sample_freq_hz,
        default_freqs_hz,
        default_q,
        charging_cutoff_min,
        on_apply,
        initial_freqs_hz=None,
        initial_q=None,
    ):
        self.parent_window = parent_window
        self.time_s = np.asarray(time_s, dtype=float)
        self.time_min = np.asarray(time_min, dtype=float)
        self.current_raw_pa = np.asarray(current_raw_pa, dtype=float)
        self.sample_freq_hz = float(sample_freq_hz)
        self.default_freqs_hz = tuple(float(f) for f in default_freqs_hz)
        self.default_q = float(default_q)
        
        if initial_freqs_hz is None:
            self.initial_freqs_hz = self.default_freqs_hz
        else:
            self.initial_freqs_hz = tuple(float(f) for f in initial_freqs_hz)
        
        if initial_q is None:
            self.initial_q = self.default_q
        else:
            self.initial_q = float(initial_q)
        self.charging_cutoff_min = float(charging_cutoff_min)
        self.on_apply = on_apply

        self.detected_freqs_hz = []

        self.prominence = tk.StringVar(value="0.1")
        self.distance = tk.StringVar(value="10")
        self.min_freq = tk.StringVar(value="0")
        self.max_freq = tk.StringVar(value="")
        self.q_value = tk.StringVar(value=str(self.initial_q))
        self.freqs_text = tk.StringVar(
            value=", ".join(f"{f:g}" for f in self.initial_freqs_hz)
        )
        
        if self.initial_freqs_hz == self.default_freqs_hz and self.initial_q == self.default_q:
            status = "Using default config frequencies."
        else:
            status = "Using previously applied custom frequencies."
        
        self.status_text = tk.StringVar(value=status)

        self.fig = Figure(figsize=(5, 4), dpi=100, constrained_layout=True)
        self.ax_fft = self.fig.add_subplot(111)

        self.make_popup()
        self.draw_fft()

    def make_popup(self):
        self.popup = tk.Toplevel(self.parent_window)
        self.popup.title("Custom FFT Notch Frequencies")

        self.leftframe = ttk.Frame(self.popup, padding=8)
        self.rightframe = ttk.Frame(self.popup, padding=8)

        self.leftframe.grid(row=0, column=0, sticky="nsew")
        self.rightframe.grid(row=0, column=1, sticky="nsew")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.rightframe)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        row = 0

        ttk.Label(self.leftframe, text="Find Peak Parameters").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        row += 1

        ttk.Label(self.leftframe, text="Prominence:").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.leftframe, textvariable=self.prominence, width=10).grid(
            row=row, column=1, sticky="ew"
        )
        row += 1

        ttk.Label(self.leftframe, text="Distance:").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.leftframe, textvariable=self.distance, width=10).grid(
            row=row, column=1, sticky="ew"
        )
        row += 1

        ttk.Label(self.leftframe, text="Min Hz:").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.leftframe, textvariable=self.min_freq, width=10).grid(
            row=row, column=1, sticky="ew"
        )
        row += 1

        ttk.Label(self.leftframe, text="Max Hz:").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.leftframe, textvariable=self.max_freq, width=10).grid(
            row=row, column=1, sticky="ew"
        )
        row += 1

        ttk.Button(self.leftframe, text="Find peaks", command=self.find_and_fill_peaks).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )
        row += 1

        ttk.Label(self.leftframe, text="Frequencies to notch, Hz:").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1

        ttk.Entry(self.leftframe, textvariable=self.freqs_text, width=28).grid(
            row=row, column=0, columnspan=2, sticky="ew"
        )
        row += 1

        ttk.Label(self.leftframe, text="Q:").grid(row=row, column=0, sticky="e")
        ttk.Entry(self.leftframe, textvariable=self.q_value, width=10).grid(
            row=row, column=1, sticky="ew"
        )
        row += 1

        ttk.Button(self.leftframe, text="Apply custom to parent trace", command=self.apply_custom).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 4)
        )
        row += 1

        ttk.Button(self.leftframe, text="Reset to config defaults", command=self.apply_default).grid(
            row=row, column=0, columnspan=2, sticky="ew"
        )
        row += 1

        ttk.Label(
            self.leftframe,
            textvariable=self.status_text,
            wraplength=220,
            foreground="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _parse_float_or_none(self, value):
        value = value.strip()
        if value == "":
            return None
        return float(value)

    def _parse_freqs(self):
        raw = self.freqs_text.get().strip()

        if raw == "":
            return []

        freqs = []
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            freqs.append(float(part))

        nyquist = self.sample_freq_hz / 2.0
        freqs = sorted(
            {
                float(f)
                for f in freqs
                if np.isfinite(f) and 0 < float(f) < nyquist
            }
        )

        return freqs

    def _fft_data(self):
        mask = (
            np.isfinite(self.time_s)
            & np.isfinite(self.time_min)
            & np.isfinite(self.current_raw_pa)
            & (self.time_min > self.charging_cutoff_min)
        )

        t = self.time_s[mask]
        y = self.current_raw_pa[mask]

        if len(y) < 4:
            raise ValueError("Not enough post-charging data points for FFT.")

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        y = y - np.nanmean(y)

        n = len(y)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_freq_hz)
        amp = np.abs(np.fft.rfft(y))

        return freqs, amp

    def draw_fft(self):
        self.ax_fft.clear()
    
        try:
            freqs, amp = self._fft_data()
        except Exception as exc:
            self.status_text.set(f"Could not compute FFT: {exc}")
            self.canvas.draw()
            return
    
        self.ax_fft.plot(freqs, amp, linewidth=1.2)
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        self.ax_fft.set_title("FFT of raw post-charging trace")
    
        # Zoom into the user-selected frequency window.
        try:
            min_f = self._parse_float_or_none(self.min_freq.get())
            max_f = self._parse_float_or_none(self.max_freq.get())
    
            if min_f is not None or max_f is not None:
                if min_f is None:
                    min_f = 0.0
                if max_f is None:
                    max_f = self.sample_freq_hz / 2.0
    
                self.ax_fft.set_xlim(min_f, max_f)
    
                window_mask = (freqs >= min_f) & (freqs <= max_f)
                if np.any(window_mask):
                    ymax = np.nanmax(amp[window_mask])
                    if np.isfinite(ymax) and ymax > 0:
                        self.ax_fft.set_ylim(0, ymax * 1.15)
    
        except Exception as exc:
            print(f"Could not update FFT axis window: {exc}")
    
        # Mark currently selected notch frequencies.
        try:
            selected = self._parse_freqs()
    
            if selected:
                y_vals = np.interp(selected, freqs, amp)
                self.ax_fft.plot(selected, y_vals, "o", markersize=4)
    
                for f, a in zip(selected, y_vals):
                    self.ax_fft.text(f, a, f"{f:.2f} Hz", fontsize=8)
    
        except Exception as exc:
            print(f"Could not mark selected notch frequencies: {exc}")
    
        self.canvas.draw()

    def find_and_fill_peaks(self):
        try:
            freqs, amp = self._fft_data()
    
            min_f = self._parse_float_or_none(self.min_freq.get())
            max_f = self._parse_float_or_none(self.max_freq.get())
    
            if min_f is None:
                min_f = 1.0
            if max_f is None:
                max_f = self.sample_freq_hz / 2.0
    
            mask = (freqs >= min_f) & (freqs <= max_f)
    
            freqs_m = freqs[mask]
            amp_m = amp[mask]
    
            if len(freqs_m) < 3:
                raise ValueError("Selected frequency window is too small.")
    
            # Remove tiny numerical baseline offsets.
            amp_m = amp_m - np.nanmin(amp_m)
    
            # Prominence is now interpreted as a fraction of the max amplitude
            # inside the selected frequency window.
            prominence_fraction = float(self.prominence.get())
            if prominence_fraction <= 0:
                prominence_fraction = 0.05
    
            prominence_abs = prominence_fraction * np.nanmax(amp_m)
    
            # Distance is now interpreted in Hz, not FFT bins.
            distance_hz = float(self.distance.get())
            df = np.nanmedian(np.diff(freqs_m))
    
            if not np.isfinite(df) or df <= 0:
                raise ValueError("Could not determine FFT frequency spacing.")
    
            distance_bins = max(1, int(distance_hz / df))
    
            peaks, props = find_peaks(
                amp_m,
                prominence=prominence_abs,
                distance=distance_bins,
            )
    
            if len(peaks) == 0:
                self.detected_freqs_hz = []
                self.freqs_text.set("")
                self.status_text.set(f"No peaks found from {min_f:g}–{max_f:g} Hz.")
                self.draw_fft()
                return
    
            peak_freqs = freqs_m[peaks]
            peak_proms = props["prominences"]
    
            # Keep only the strongest peaks so the textbox does not get flooded.
            max_peaks = 12
            strongest_order = np.argsort(peak_proms)[::-1][:max_peaks]
            peak_freqs = peak_freqs[strongest_order]
    
            # Round for notch filtering and deduplicate.
            found = sorted({round(float(f), 2) for f in peak_freqs if f > 0})
    
            self.detected_freqs_hz = found
            self.freqs_text.set(", ".join(f"{f:g}" for f in found))
    
            self.status_text.set(
                f"Found {len(found)} peak(s) from {min_f:g}–{max_f:g} Hz."
            )
    
            self.draw_fft()
    
        except Exception as exc:
            messagebox.showerror("FFT peak detection failed", str(exc))
            self.status_text.set(f"Peak detection failed: {exc}")

    def apply_custom(self):
        try:
            freqs = self._parse_freqs()
            q = float(self.q_value.get())

            self.on_apply(freqs, q, "custom")
            self.status_text.set(
                "Applied custom notch frequencies: "
                + (", ".join(f"{f:g}" for f in freqs) if freqs else "none")
            )
            self.draw_fft()

        except Exception as exc:
            messagebox.showerror("Could not apply custom notch", str(exc))
            self.status_text.set(f"Could not apply custom notch: {exc}")

    def apply_default(self):
        try:
            self.freqs_text.set(", ".join(f"{f:g}" for f in self.default_freqs_hz))
            self.q_value.set(str(self.default_q))

            self.on_apply(list(self.default_freqs_hz), self.default_q, "default")
            self.status_text.set("Reverted to config default notch frequencies.")
            self.draw_fft()

        except Exception as exc:
            messagebox.showerror("Could not apply defaults", str(exc))
            self.status_text.set(f"Could not apply defaults: {exc}")