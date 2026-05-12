# Steady-State-Fits
This repository contains a Python program designed to analyze steady state chronoamperometric traces obtained from nanoimpact electrochemistry.

The program detects negative current impacts in chronoamperometry traces, fits local steady state plateaus before and after each impact, and calculates the current change, $\Delta i_{ss}$.

After automatic detection and fitting, the user can review impacts in an interactive Tk window, adjust fitting regions, keep or remove events, and export the final results to Excel.

## Installation
Clone the repo with GitHub.
```
git clone https://github.com/SepLabUCSB/Steady-State-Fits
```

## Usage
Open main_simple.py in your favorite Python environment

### Option 1: Run in Python IDE (e.g., Spyder)

Open [`main_simple.py`](main_simple.py) in IDE and run the file.

By default, the program uses:

```python
folder = Path("data")
pattern = "*.txt"
```

This means it will process all `.txt` files in the `data/` folder.

To analyze a different folder or file pattern in Spyder, change the default values in `build_parser()` inside [`main_simple.py`](main_simple.py).

## Main Entry Point
The main analysis workflow is in [`main_simple.py`](main_simple.py).

<img src="https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/main_entry.png" alt="Main entry" width="80%">

This script builds a command-line interface, prepares a [`PipelineConfig`](modules/config.py), and then calls [`analyze_folder()`](modules/analysis.py) to process all matching raw data files in the selected folder.

The script is intentionally minimal. Most of the analysis logic is kept inside the [`modules/`](modules/) and [`interactive/`](interactive/) packages. This makes the command-line script easier to read and makes the analysis code easier to maintain.

## Example

Place raw chronoamperometry `.txt` files in the [`data/`](data/) folder.

Example:

```text
data/
├── trace_001.txt
├── trace_002.txt
└── trace_003.txt
```

By default, the program analyzes all `.txt` files in the `data/` folder.

Run the program in the environment.

An interactive Matplotlib window opens showing the current–time trace with detected step boundaries, midpoint markers, a drift-fit line, and allows users control to review fits, toggle filtering, or move to the next file.

![App Screenshot](https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/Fitting1.gif?raw=true)

Example current–time trace showing large periodic noise that can be removed using Fourier notch filtering, revealing the underlying step-like current response:

![App Screenshot](https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/Fitting3.gif?raw=true)

An interactive fit-adjustment window lets users review each detected impact, adjust pre/post fitting boundaries and evaluation time, then accept, reject, or save the fit to an excel file generated within the data folder.

![App Screenshot](https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/Fitting2.gif?raw=true)

## Folder Structure
<img src="https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/folder_list.png" alt="Folder structure" width="80%">

## What each file does 

| File | Main purpose |
|---|---|
| [`main_simple.py`](main_simple.py) | Main command-line entry point. Parses user options, sets up the pipeline configuration, forces the Tk backend, and runs the analysis. |
| [`modules/config.py`](modules/config.py) | Defines `PipelineConfig`, which stores the main analysis settings such as folder path and file pattern. |
| [`modules/analysis.py`](modules/analysis.py) | Coordinates the full analysis workflow, including loading files, detecting impacts, fitting plateaus, launching review, and exporting results. |
| [`modules/data_loader.py`](modules/data_loader.py) | Loads raw chronoamperometry `.txt` files and prepares the time/current data for analysis. |
| [`modules/filters.py`](modules/filters.py) | Contains filtering tools used to reduce noise before impact detection. |
| [`modules/detection.py`](modules/detection.py) | Detects candidate negative current impacts in the chronoamperometry trace. |
| [`modules/fitting.py`](modules/fitting.py) | Fits steady state current plateaus before and after each impact to calculate Δi<sub>ss</sub>. |
| [`modules/step_analysis.py`](modules/step_analysis.py) | Provides helper tools for analyzing current step size, timing, and quality. |
| [`interactive/tk_backend.py`](interactive/tk_backend.py) | Forces Matplotlib to use the Tk backend for interactive review windows. |
| [`interactive/trace_reviewer.py`](interactive/trace_reviewer.py) | Provides the interactive review interface for keeping, removing, or adjusting detected impacts. |
| [`apps/process_impacts.py`](apps/process_impacts.py) | Alternate script for running impact processing outside the main entry point. |
| [`apps/review_impacts_tk.py`](apps/review_impacts_tk.py) | Alternate script for launching or testing the Tk-based review interface. |

## Packages used
This project uses both Python standard-library modules and external scientific Python packages.

### Standard-library packages
`pathlib` is used to handle file and folder paths in a clean, operating-system-independent way.
In [`main_simple.py`](main_simple.py), the input folder is stored as a `Path` object:

<img src="https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/pathlib.png" alt="Pathlib" width="80%">

Using `Path` makes it easier to combine folders, check whether files exist, and work across Windows, macOS, and Linux.

`argparse` is used to create the command-line interface. It allows the user to run the program with options such as --folder, --pattern, --no-per-file, --no-split, and --quiet.

This makes the pipeline flexible without requiring the user to edit the Python source code each time they want to analyze a different folder or change an output option.

### Interactive plotting packages
`tkinter` is Python’s standard GUI toolkit. The interactive reviewer uses Tk so the user can inspect impacts in a graphical window.

Most standard Python installations include `tkinter`, but some minimal installations may not. If the interactive window does not open, check that Tk support is installed for your Python environment.

`matplotlib` is used for plotting chronoamperometry traces, detecting nanoimpacts, and fitting current vs. time regions. The Tk backend allows `Matplotlib` figures to be displayed interactively inside a Tk-compatible window.

The backend is forced near the top of main_simple.py:

<img src="https://github.com/SepLabUCSB/Steady-State-Fits/blob/trunk/docs/tkbackend.png" width="80%">

This should happen before importing modules that create plots.

## Scientific computing packages
Listed in [`requirements.txt`](requirements.txt)

## Installation

Use a Python environment with the required packages installed:

```text
numpy
scipy
pandas
matplotlib
openpyxl
```

## Alternate Usage

### Option 2: Run from the command line

From the repository folder, run:

```bash
python main_simple.py
```

This runs the full pipeline using the default `data/` folder.

To choose a different folder:

```bash
python main_simple.py --folder "C:/Users/YourName/Documents/nanoimpact_data"
```

To analyze only selected files:

```bash
python main_simple.py --folder data --pattern "Pt_*.txt"
```

## Command-Line Options

| Option | Description |
|---|---|
| `--folder` | Folder containing raw `.txt` files. Default: `data`. |
| `--pattern` | File pattern for selecting raw data files. Default: `*.txt`. |
| `--no-per-file` | Do not save individual Excel files for each trace. |
| `--no-split` | Do not save separate small and large impact Excel files. |
| `--quiet` | Reduce printed terminal output. |

## Example Commands

Analyze all `.txt` files in the default `data/` folder:

```bash
python main_simple.py
```

Analyze one specific file:

```bash
python main_simple.py --folder data --pattern "example_trace.txt"
```

Analyze files beginning with `Pt_`:

```bash
python main_simple.py --folder data --pattern "Pt_*.txt"
```

Run with less terminal output:

```bash
python main_simple.py --quiet
```

Run without per-file Excel outputs:

```bash
python main_simple.py --no-per-file
```

## Analysis Workflow

The pipeline performs the following steps:

1. Load raw chronoamperometry `.txt` files.
2. Detect candidate negative current impacts.
3. Fit local steady state plateaus before and after each impact.
4. Calculate Δi<sub>ss</sub>.
5. Apply monoexponential correction when appropriate.
6. Open the interactive Tk review window.
7. Export accepted and removed impacts to Excel.

## Output Files

Depending on the selected options, the pipeline may generate:

```text
sample_001_iss_results.xlsx
combined_impacts.xlsx
combined_small_impacts.xlsx
combined_large_impacts.xlsx
```

The per-file Excel files contain results from individual traces.

The combined Excel files summarize impacts across all analyzed files.

If your data has a specific cutoff, add it:

Small and large impacts are separated based on the magnitude of Δi<sub>ss</sub>. In this pipeline, impacts with \|Δi<sub>ss</sub>\| < 100 pA are classified as small, while impacts with \|Δi<sub>ss</sub>\| ≥ 100 pA are classified as large.

## Troubleshooting

### The interactive window does not open

Make sure `tkinter` is available in your Python environment.

You can test it with:

```bash
python -m tkinter
```

### No files are found

Check that the input folder exists and contains `.txt` files.

Also check that the file pattern is correct.

Example:

```bash
python main_simple.py --folder data --pattern "*.txt"
```

### Excel files are not created

Make sure `pandas` and `openpyxl` are installed:

```bash
pip install pandas openpyxl
```

Also make sure the Excel file is not already open.
