# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:00:21 2026

@author: Ishaan
"""

from pathlib import Path

from interactive.tk_backend import force_tk_backend

force_tk_backend()

from modules.config import PipelineConfig
from modules.analysis import analyze_folder


def run_review_impacts_tk(folder: Path):
    """
    Run raw processing with interactive Tk review enabled.

    This requires modules.analysis.analyze_folder to accept:
        review_interactive=True
    """
    folder = Path(folder)

    config = PipelineConfig(
        folder=folder,
        pattern="*.txt",
    )

    return analyze_folder(
        config=config,
        save_combined=True,
        save_split_files=True,
        save_per_file=True,
        verbose=True,
        review_interactive=True,
    )


if __name__ == "__main__":
    run_review_impacts_tk(
        # Input filepath here
        Path(r"")
    )
