# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:06:45 2026

@author: Ishaan
"""

from pathlib import Path
import argparse

from interactive.tk_backend import force_tk_backend

# Force Tk before any interactive plotting code is imported.
force_tk_backend()

from modules.config import PipelineConfig
from modules.analysis import analyze_folder

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Simple nanoimpact pipeline with Tk interactive impact review. "
            "This runs impact detection, plateau fitting, monoexponential correction, "
            "manual keep/remove review, and Excel export only."
        )
    )

    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("data"),
        help="Folder containing raw nanoimpact .txt files.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="File pattern for raw data files.",
    )

    parser.add_argument(
        "--no-per-file",
        action="store_true",
        help="Do not save per-file *_iss_results.xlsx files.",
    )

    parser.add_argument(
        "--no-split",
        action="store_true",
        help=(
            "Do not save combined_small_impacts.xlsx and "
            "combined_large_impacts.xlsx."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce printed output.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        folder=args.folder,
        pattern=args.pattern,
    )

    analyze_folder(
        config=config,
        save_combined=True,
        save_split_files=not args.no_split,
        save_per_file=not args.no_per_file,
        verbose=not args.quiet,
        review_interactive=True,
    )


if __name__ == "__main__":
    main()