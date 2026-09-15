#!/usr/bin/env python3
"""
Representative signal-level figures for the report (ICLR style):
    plots/signals_by_class.{pdf,png}  one preprocessed tri-axial trial per class
    plots/spectra_by_class.{pdf,png}  FFT amplitude spectrum of |a| per class

These show, at the raw-signal and frequency level, why the four classes look
different - the input to the whole feature/classification pipeline.

Run from the repo root:  python3 scripts/signal_plots.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "iclr_style"))
import iclrplot as ip  # noqa: E402

PRE = os.path.join(REPO, "data", "preprocessed")
PLOTS = os.path.join(REPO, "results", "plots")
ORDER = ["piston_only", "servo_only", "both_simultaneous", "both_alternating"]
TITLE = {"piston_only": "Piston only", "servo_only": "Servo only",
         "both_simultaneous": "Both simultaneous", "both_alternating": "Both alternating"}
TRIAL = "trial01"
ip.setup()


def load(cls):
    f = os.path.join(PRE, cls, f"{cls}_{TRIAL}.csv")
    d = pd.read_csv(f)
    return (d["Time (s)"].values,
            d["Acceleration x (m/s^2)"].values,
            d["Acceleration y (m/s^2)"].values,
            d["Acceleration z (m/s^2)"].values)


def signals():
    axcols = ip.palette("categorical", 3)
    fig, axes = ip.figure(width="full", nrows=2, ncols=2, aspect=0.62, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, ORDER):
        t, x, y, z = load(cls)
        for sig, lab, c in [(x, "x", axcols[0]), (y, "y", axcols[1]), (z, "z", axcols[2])]:
            ax.plot(t, sig, lw=0.5, color=c, label=lab)
        ax.set_ylim(-12, 12); ax.set_ylabel("accel (m/s$^2$)")
        ax.legend(loc="upper right", ncol=3, fontsize=7, handlelength=1.0, columnspacing=0.8)
    for ax in axes[2:]:
        ax.set_xlabel("time (s)")
    ip.label_panels(axes, titles=[TITLE[c] for c in ORDER])
    ip.finish(fig, os.path.join(PLOTS, "signals_by_class"))


def spectra():
    fig, axes = ip.figure(width="full", nrows=2, ncols=2, aspect=0.62, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, ORDER):
        t, x, y, z = load(cls)
        a = np.sqrt(x**2 + y**2 + z**2); a = a - a.mean()
        fs = 1.0 / np.median(np.diff(t))
        n = len(a)
        amp = np.abs(np.fft.rfft(a)) / n
        amp[1:] *= 2
        f = np.fft.rfftfreq(n, 1 / fs)
        m = f <= 30
        ax.plot(f[m], amp[m], lw=0.8, color=ip.CATEGORICAL[0])
        ax.set_xlim(0, 30); ax.set_ylabel("amplitude (m/s$^2$)")
    for ax in axes[2:]:
        ax.set_xlabel("frequency (Hz)")
    ip.label_panels(axes, titles=[TITLE[c] for c in ORDER])
    ip.finish(fig, os.path.join(PLOTS, "spectra_by_class"))


if __name__ == "__main__":
    signals()
    spectra()
    print("Wrote signals_by_class and spectra_by_class (pdf + png).")
