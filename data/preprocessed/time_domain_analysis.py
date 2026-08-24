#!/usr/bin/env python3
"""
Time-domain analysis: RMS, peak-to-peak, crest factor, kurtosis.


OUTPUT STRUCTURE

    time_domain_analysis/
        servo_only/
            servo_only_trial01_time_domain.csv   <- one report per trial
            servo_only_trial02_time_domain.csv
            ...
        piston_only/
            ...
        both_simultaneous/
            ...
        both_alternating/
            ...
        summary_servo_only.csv                    <- one summary per class
        summary_piston_only.csv                   <- (mean/std/min/max across
        summary_both_simultaneous.csv                that class's trials)
        summary_both_alternating.csv
        all_trials_combined.csv                   <- every trial's metrics
                                                       in one table, for
                                                       quick comparison/plots
        stationary/
            <name>_time_domain.csv                <- top-level CSV gets its
                                                       own individual report

METRICS (computed per axis: ax, ay, az, and the absolute/magnitude signal):
    RMS           - sqrt(mean(x^2)), overall vibration energy level
    P2P           - max(x) - min(x), peak-to-peak swing
    Crest factor  - max(|x|) / RMS, how "spiky" the signal is relative to
                    its average energy (high crest factor = sharp
                    transients on a quiet background, e.g. the piston
                    shock events; low crest factor = more continuous
                    vibration)
    Kurtosis      - scipy's excess kurtosis (Gaussian = 0). Higher values
                    mean heavier tails / more outlier-like spikes than a
                    normal distribution.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis

CLASS_DIRS = ["servo_only", "piston_only", "both_simultaneous", "both_alternating"]
AXES = ["ax", "ay", "az", "aabs"]
SKIP_KEYWORDS = ["label", "manifest"]


def load_csv(path):
    df = pd.read_csv(path)
    n_cols = df.shape[1]
    base_names = ["t", "ax", "ay", "az", "aabs"]
    df.columns = base_names[:n_cols] + list(df.columns[n_cols:]) if n_cols <= 5 else list(df.columns)
    if "aabs" not in df.columns:
        df["aabs"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    return df


def compute_metrics(x):
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    p2p = float(np.max(x) - np.min(x))
    peak = float(np.max(np.abs(x)))
    crest = float(peak / rms) if rms > 0 else np.nan
    kurt = float(sp_kurtosis(x, fisher=True, bias=False))
    return rms, p2p, crest, kurt


def analyze_file(path):
    df = load_csv(path)
    rows = []
    for axis in AXES:
        rms, p2p, crest, kurt = compute_metrics(df[axis].values)
        rows.append({"axis": axis, "RMS": rms, "P2P": p2p, "CrestFactor": crest, "Kurtosis": kurt})
    return pd.DataFrame(rows)


def main(root="."):
    root = os.path.abspath(root)
    out_root = os.path.join(root, "time_domain_analysis")
    os.makedirs(out_root, exist_ok=True)

    combined_rows = []

    for cls in CLASS_DIRS:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            print(f"Skipping missing folder: {cls}")
            continue

        out_cls_dir = os.path.join(out_root, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(cls_dir, "*.csv")))
        class_rows = []

        for f in csv_files:
            fname = os.path.splitext(os.path.basename(f))[0]
            report_df = analyze_file(f)
            report_df.to_csv(os.path.join(out_cls_dir, f"{fname}_time_domain.csv"), index=False)

            for _, row in report_df.iterrows():
                rec = {"class": cls, "file": os.path.basename(f), **row.to_dict()}
                combined_rows.append(rec)
                class_rows.append(row.to_dict())

        if class_rows:
            cls_df = pd.DataFrame(class_rows)
            summary = cls_df.groupby("axis")[["RMS", "P2P", "CrestFactor", "Kurtosis"]].agg(
                ["mean", "std", "min", "max"]
            )
            summary.columns = ["_".join(c) for c in summary.columns]
            summary = summary.reset_index()
            summary.to_csv(os.path.join(out_root, f"summary_{cls}.csv"), index=False)
            print(f"Analyzed {len(csv_files)} trials in {cls}/ -> summary_{cls}.csv")

    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(os.path.join(out_root, "all_trials_combined.csv"), index=False)
        print("Wrote all_trials_combined.csv")

    # top-level CSVs (e.g. the stationary benchmark) - individual reports only
    top_csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    stationary_files = [f for f in top_csvs if not any(k in os.path.basename(f).lower() for k in SKIP_KEYWORDS)]
    if stationary_files:
        stat_out_dir = os.path.join(out_root, "stationary")
        os.makedirs(stat_out_dir, exist_ok=True)
        for f in stationary_files:
            try:
                report_df = analyze_file(f)
            except Exception as e:
                print(f"Skipping {os.path.basename(f)} - not sensor data ({e})")
                continue
            fname = os.path.splitext(os.path.basename(f))[0]
            report_df.to_csv(os.path.join(stat_out_dir, f"{fname}_time_domain.csv"), index=False)
            print(f"Analyzed top-level file: {os.path.basename(f)}")

    print(f"\nDone. Reports written under {out_root}")


if __name__ == "__main__":
    root_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root_arg)
