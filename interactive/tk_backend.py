# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:58:52 2026

@author: Ishaan
"""

import contextlib
import warnings

def force_tk_backend():
    """
    Force Matplotlib to use TkAgg before pyplot figures are created.
    This is needed for interactive trace review.
    """
    import matplotlib

    try:
        matplotlib.use("TkAgg", force=True)
    except Exception as exc:
        print(f"Warning: could not force TkAgg backend: {exc}")

    try:
        import matplotlib.pyplot as plt

        if "TkAgg" not in matplotlib.get_backend():
            plt.switch_backend("TkAgg")

        import matplotlib.backends._backend_tk as backend_tk

        if not hasattr(backend_tk.FigureManagerTk, "_owns_mainloop"):
            backend_tk.FigureManagerTk._owns_mainloop = False

    except Exception as exc:
        print(f"Warning: could not fully initialize TkAgg backend: {exc}")


def tk_rcparams():
    """
    Smaller GUI-friendly plotting parameters for interactive windows.
    """
    return {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "axes.linewidth": 1.2,
        "figure.autolayout": False,
    }


def show_blocking_safely(fig):
    """
    Robust blocking show for Spyder/Windows/Tk.
    """
    import matplotlib.pyplot as plt

    try:
        plt.show(block=True)
    except AttributeError as exc:
        print(
            "Warning: plt.show(block=True) failed; "
            f"using pause-loop fallback: {exc}"
        )

        try:
            while plt.fignum_exists(fig.number):

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="FigureCanvasAgg is non-interactive*",
                        category=UserWarning,
                    )
                    plt.pause(0.05)
        except Exception:
            plt.close(fig)


@contextlib.contextmanager
def tk_style_context():
    import matplotlib as mpl

    with mpl.rc_context(tk_rcparams()):
        yield