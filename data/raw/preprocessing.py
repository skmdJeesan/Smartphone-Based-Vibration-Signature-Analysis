#!/usr/bin/env python3
"""
Preprocessing step 1: gravity removal + uniform time-base resampling.

To be saved in the folder that contains:
    servo_only/
    piston_only/
    both_simultaneous/
    both_alternating/
    Stationary_benchmark.csv

It creates a sibling "preprocessed/" folder with the same 4 subfolders
(same filenames inside) plus the preprocessed stationary file(s), so the
downstream time/frequency-domain analysis can just point at "preprocessed/"
instead of the raw data.

WHAT EACH STEP DOES AND WHY, IN THIS ORDER:

1. Uniform time-base resampling (done FIRST)
   Real device timestamps are never perfectly evenly spaced. FFT/CWT and
   filters below both assume a fixed sample interval, so every trial is
   linearly interpolated onto a uniform time grid at FS_TARGET (402.1 Hz,
   matched to the phone's measured native rate) before anything else
   touches it.

2. Gravity removal (done SECOND, after resampling, since it uses a filter
   that assumes uniform sampling)
   Gravity sits almost entirely at 0 Hz / very low frequency, while the
   piston/servo vibration signal sits at several Hz and up. A 4th-order
   Butterworth low-pass at 0.3 Hz tracks the slow gravity component per
   axis; that estimate is then subtracted from the raw signal, leaving
   just the linear (motion) acceleration. This is the standard approach
   used in human-activity-recognition preprocessing pipelines, and it
   also handles the case where the phone's orientation drifts slowly
   during a trial (not just a fixed offset).

Change GRAVITY_CUTOFF_HZ below if 0.3 Hz doesn't work.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
 
FS_TARGET = 402.1          # Hz - matched to the phone's measured native rate
GRAVITY_CUTOFF_HZ = 0.3    # Hz - low-pass cutoff used to estimate gravity
FILTER_ORDER = 4
 
CLASS_DIRS = ["servo_only", "piston_only", "both_simultaneous", "both_alternating"]
 
 
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = ["t", "ax", "ay", "az", "aabs"][:len(df.columns)]
    return df
 
 
def uniform_resample(df, fs_target=FS_TARGET):
    t = df["t"].values
    t0, t1 = t[0], t[-1]
    n_new = int(round((t1 - t0) * fs_target)) + 1
    t_uniform = np.linspace(t0, t1, n_new)
    out = {"t": t_uniform}
    for col in ["ax", "ay", "az"]:
        out[col] = np.interp(t_uniform, t, df[col].values)
    return pd.DataFrame(out)
 
 
def remove_gravity(df, fs, cutoff=GRAVITY_CUTOFF_HZ, order=FILTER_ORDER):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    out = df.copy()
    for col in ["ax", "ay", "az"]:
        gravity_est = filtfilt(b, a, df[col].values)
        out[col] = df[col].values - gravity_est
    return out
 
 
def process_file(path, fs_target=FS_TARGET):
    df = load_csv(path)
    df = uniform_resample(df, fs_target)
    df = remove_gravity(df, fs_target)
    df["aabs"] = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
    df = df.rename(columns={
        "t": "Time (s)",
        "ax": "Acceleration x (m/s^2)",
        "ay": "Acceleration y (m/s^2)",
        "az": "Acceleration z (m/s^2)",
        "aabs": "Absolute acceleration (m/s^2)",
    })
    return df
 
 
def main(root="."):
    root = os.path.abspath(root)
    out_root = os.path.join(root, "preprocessed")
    os.makedirs(out_root, exist_ok=True)
 
    total = 0
    for cls in CLASS_DIRS:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            print(f"Skipping missing folder: {cls}")
            continue
        out_cls_dir = os.path.join(out_root, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        csv_files = sorted(glob.glob(os.path.join(cls_dir, "*.csv")))
        for f in csv_files:
            df_out = process_file(f)
            df_out.to_csv(os.path.join(out_cls_dir, os.path.basename(f)), index=False)
        print(f"Processed {len(csv_files)} files in {cls}/")
        total += len(csv_files)
 
    top_csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    for f in top_csvs:
        name_lower = os.path.basename(f).lower()
        if "label" in name_lower or "manifest" in name_lower:
            print(f"Skipping non-sensor file: {os.path.basename(f)}")
            continue
        try:
            df_out = process_file(f)
        except (ValueError, TypeError) as e:
            print(f"Skipping {os.path.basename(f)} - doesn't look like sensor data ({e})")
            continue
        df_out.to_csv(os.path.join(out_root, os.path.basename(f)), index=False)
        print(f"Processed stationary/top-level file: {os.path.basename(f)}")
        total += 1
 
    print(f"\nDone. {total} files written to {out_root}")
 
 
if __name__ == "__main__":
    root_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root_arg)
 
