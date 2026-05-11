# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:22:30 2026

@author: Ishaan
"""

import numpy as np
from scipy import signal


def notch_filter(current_pa, sample_freq_hz, freqs_hz, q=30):
    filtered = np.asarray(current_pa, dtype=float).copy()

    for freq_hz in freqs_hz:
        if freq_hz <= 0 or freq_hz >= sample_freq_hz / 2:
            continue

        b, a = signal.iirnotch(w0=freq_hz, Q=q, fs=sample_freq_hz)
        filtered = signal.filtfilt(b, a, filtered)

    return filtered