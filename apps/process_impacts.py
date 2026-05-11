# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:54:36 2026

@author: Ishaan
"""

from pathlib import Path

from nanoimpact.config import PipelineConfig
from nanoimpact.analysis import analyze_folder


def run_process_impacts():
    config = PipelineConfig(
        folder=Path(r"Z:\Projects\Ishaan\Raw Data\4NP Nanoimpacts pH 4\Thiol + No Thiol Overlayed Histogram"),
        pattern="*.txt",
    )

    combined_df = analyze_folder(config)

    out_xlsx = config.folder / "combined_results.xlsx"
    combined_df.to_excel(out_xlsx, index=False)

    print(f"Saved combined results to {out_xlsx}")


if __name__ == "__main__":
    run_process_impacts()