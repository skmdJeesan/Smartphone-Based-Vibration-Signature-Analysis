#!/usr/bin/env python3
"""
Verifies that preprocessing (gravity removal + uniform time-base resampling)
actually worked, on every file in this folder and its subfolders.

"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS_EXPECTED = 402.1          # Hz - expected uniform sample rate
DT_JITTER_LIMIT = 1e-6       # s - max allowed std of the time step
FS_TOLERANCE = 1.0           # Hz - how far the measured rate may drift from FS_EXPECTED
GRAVITY_CUTOFF_HZ = 0.3      # Hz - must match the cutoff used during preprocessing
GRAVITY_RESIDUAL_LIMIT = 5.0  # m/s^2 - max allowed low-pass-band energy after removal.


SKIP_KEYWORDS = ["label", "manifest"]


def find_csv_files(root):
    all_csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    this_file = os.path.abspath(__file__)
    out = []
    for f in all_csvs:
        name_lower = os.path.basename(f).lower()
        if any(k in name_lower for k in SKIP_KEYWORDS):
            continue
        if os.path.abspath(f) == this_file:
            continue
        out.append(f)
    return sorted(out)


def check_file(path):
    result = {"file": path, "pass": True, "issues": []}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        result["pass"] = False
        result["issues"].append(f"could not read file: {e}")
        return result

    if df.shape[1] < 4:
        result["pass"] = False
        result["issues"].append(f"expected >=4 columns, found {df.shape[1]}")
        return result

    df.columns = ["t", "ax", "ay", "az"] + list(df.columns[4:])

    for col in ["t", "ax", "ay", "az"]:
        if not np.issubdtype(df[col].dtype, np.number):
            result["pass"] = False
            result["issues"].append(f"column '{col}' is not numeric")
            return result

    if df[["t", "ax", "ay", "az"]].isna().any().any():
        result["pass"] = False
        result["issues"].append("contains NaN values")

    # --- uniform time base check ---
    t = df["t"].values
    if len(t) < 3:
        result["pass"] = False
        result["issues"].append("too few samples to check timing")
        return result

    dt = np.diff(t)
    dt_std = dt.std()
    measured_fs = 1.0 / dt.mean()

    if dt_std > DT_JITTER_LIMIT:
        result["pass"] = False
        result["issues"].append(
            f"time base not uniform (dt std = {dt_std:.2e} s, expected < {DT_JITTER_LIMIT:.0e} s)"
        )

    if abs(measured_fs - FS_EXPECTED) > FS_TOLERANCE:
        result["pass"] = False
        result["issues"].append(
            f"sample rate {measured_fs:.2f} Hz is far from expected {FS_EXPECTED} Hz"
        )

    # --- gravity removal check (low-pass band, not raw mean - see docstring) ---
    nyq = measured_fs / 2
    b, a = butter(4, GRAVITY_CUTOFF_HZ / nyq, btype="low")
    lowpass = np.stack([filtfilt(b, a, df[c].values) for c in ["ax", "ay", "az"]], axis=1)
    residual = np.sqrt(np.mean(np.sum(lowpass ** 2, axis=1)))

    if residual > GRAVITY_RESIDUAL_LIMIT:
        result["pass"] = False
        result["issues"].append(
            f"gravity residual too large (||mean|| = {residual:.3f} m/s^2, "
            f"expected < {GRAVITY_RESIDUAL_LIMIT} m/s^2)"
        )

    result["dt_std"] = dt_std
    result["measured_fs"] = measured_fs
    result["gravity_residual"] = residual
    return result


def main(root="."):
    root = os.path.abspath(root)
    files = find_csv_files(root)

    if not files:
        print(f"No CSV files found under {root} (excluding label/manifest files).")
        return

    print(f"Checking {len(files)} file(s) under {root}\n")

    n_pass = 0
    n_fail = 0

    for f in files:
        rel = os.path.relpath(f, root)
        r = check_file(f)
        if r["pass"]:
            n_pass += 1
            print(f"[OK]   {rel}   (Fs={r['measured_fs']:.2f} Hz, dt_std={r['dt_std']:.2e} s, "
                  f"gravity_residual={r['gravity_residual']:.3f} m/s^2)")
        else:
            n_fail += 1
            print(f"[FAIL] {rel}")
            for issue in r["issues"]:
                print(f"        - {issue}")

    print(f"\n{n_pass}/{len(files)} files passed all checks.")
    if n_fail:
        print(f"{n_fail} file(s) need attention - see [FAIL] lines above.")


if __name__ == "__main__":
    import sys
    root_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root_arg)
