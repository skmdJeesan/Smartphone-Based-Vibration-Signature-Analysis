#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch


# ============================================================
# SETTINGS
# ============================================================

FS_TARGET = 402.1          # Expected sampling frequency [Hz]

# Frequency below which we consider the signal to be
# "very low frequency" for gravity-removal verification.
LOW_FREQ_LIMIT = 0.3       # Hz

# Tolerance for sampling-frequency verification.
# 0.5% of 402.1 Hz ≈ 2 Hz.
FS_TOLERANCE_PERCENT = 0.5

# Tolerance for time-step uniformity.
# 0.1% of the nominal sample interval.
DT_TOLERANCE_PERCENT = 0.1

CLASS_DIRS = [
    "servo_only",
    "piston_only",
    "both_simultaneous",
    "both_alternating"
]


# ============================================================
# COLUMN HANDLING
# ============================================================

TIME_COL = "Time (s)"
AX_COL = "Acceleration x (m/s^2)"
AY_COL = "Acceleration y (m/s^2)"
AZ_COL = "Acceleration z (m/s^2)"
ABS_COL = "Absolute acceleration (m/s^2)"


# ============================================================
# LOAD FILE
# ============================================================

def load_file(path):

    df = pd.read_csv(path)

    required = [
        TIME_COL,
        AX_COL,
        AY_COL,
        AZ_COL
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# SAMPLING VERIFICATION
# ============================================================

def verify_sampling(df):

    t = df[TIME_COL].to_numpy(dtype=float)

    dt = np.diff(t)

    result = {}

    result["Number of samples"] = len(t)

    result["Duration (s)"] = t[-1] - t[0]

    # --------------------------------------------------------
    # Basic timestamp checks
    # --------------------------------------------------------

    result["Non-monotonic samples"] = np.sum(dt <= 0)

    # Expected dt
    expected_dt = 1.0 / FS_TARGET

    result["Expected dt (s)"] = expected_dt

    result["Mean dt (s)"] = np.mean(dt)

    result["Std dt (s)"] = np.std(dt)

    result["Min dt (s)"] = np.min(dt)

    result["Max dt (s)"] = np.max(dt)

    # Sampling frequency from timestamps
    fs_measured = 1.0 / np.mean(dt)

    result["Measured Fs (Hz)"] = fs_measured

    fs_error_percent = (
        (fs_measured - FS_TARGET)
        / FS_TARGET
        * 100
    )

    result["Fs error (%)"] = fs_error_percent

    # --------------------------------------------------------
    # Uniformity
    # --------------------------------------------------------

    dt_error_percent = (
        np.abs(dt - expected_dt)
        / expected_dt
        * 100
    )

    result["Maximum dt deviation (%)"] = np.max(
        dt_error_percent
    )

    result["RMS dt deviation (%)"] = np.sqrt(
        np.mean(dt_error_percent ** 2)
    )

    result["Large dt deviations"] = np.sum(
        dt_error_percent > DT_TOLERANCE_PERCENT
    )

    # --------------------------------------------------------
    # Pass/fail
    # --------------------------------------------------------

    fs_ok = (
        abs(fs_error_percent)
        <= FS_TOLERANCE_PERCENT
    )

    uniform_ok = (
        np.max(dt_error_percent)
        <= DT_TOLERANCE_PERCENT
    )

    monotonic_ok = (
        result["Non-monotonic samples"] == 0
    )

    result["Sampling PASS"] = (
        fs_ok and uniform_ok and monotonic_ok
    )

    return result


# ============================================================
# GRAVITY / LOW FREQUENCY VERIFICATION
# ============================================================

def verify_gravity_removal(df):

    result = {}

    axes = {
        "X": df[AX_COL].to_numpy(dtype=float),
        "Y": df[AY_COL].to_numpy(dtype=float),
        "Z": df[AZ_COL].to_numpy(dtype=float)
    }

    # --------------------------------------------------------
    # Mean value
    #
    # After gravity removal, the mean should generally be
    # close to zero for a stationary / steady portion.
    # --------------------------------------------------------

    for axis, signal in axes.items():

        result[f"{axis} mean (m/s^2)"] = np.mean(signal)

        result[f"{axis} RMS (m/s^2)"] = np.sqrt(
            np.mean(signal ** 2)
        )

    # --------------------------------------------------------
    # Combined magnitude
    # --------------------------------------------------------

    magnitude = np.sqrt(
        axes["X"]**2 +
        axes["Y"]**2 +
        axes["Z"]**2
    )

    result["Mean magnitude (m/s^2)"] = np.mean(
        magnitude
    )

    result["RMS magnitude (m/s^2)"] = np.sqrt(
        np.mean(magnitude ** 2)
    )

    # --------------------------------------------------------
    # Low-frequency power
    # --------------------------------------------------------

    fs = FS_TARGET

    low_freq_ratios = []

    for axis, signal in axes.items():

        # Welch PSD
        f, Pxx = welch(
            signal,
            fs=fs,
            nperseg=min(4096, len(signal))
        )

        total_power = np.trapz(Pxx, f)

        low_mask = f <= LOW_FREQ_LIMIT

        low_power = np.trapz(
            Pxx[low_mask],
            f[low_mask]
        )

        if total_power > 0:
            low_ratio = (
                low_power / total_power * 100
            )
        else:
            low_ratio = 0.0

        result[
            f"{axis} low-frequency power < {LOW_FREQ_LIMIT} Hz (%)"
        ] = low_ratio

        low_freq_ratios.append(low_ratio)

    result[
        f"Average low-frequency power < {LOW_FREQ_LIMIT} Hz (%)"
    ] = np.mean(low_freq_ratios)

    return result


# ============================================================
# PROCESS ALL FILES
# ============================================================

def process_all(root):

    rows = []

    preprocessed_root = root

    if not os.path.isdir(preprocessed_root):

        raise FileNotFoundError(
            "Could not find 'preprocessed/' folder."
        )

    for cls in CLASS_DIRS:

        class_dir = os.path.join(
            preprocessed_root,
            cls
        )

        if not os.path.isdir(class_dir):

            print(
                f"WARNING: Missing folder: {cls}"
            )

            continue

        files = sorted(
            glob.glob(
                os.path.join(
                    class_dir,
                    "*.csv"
                )
            )
        )

        print(
            f"\nChecking {cls}: {len(files)} files"
        )

        for path in files:

            filename = os.path.basename(path)

            try:

                df = load_file(path)

                sampling = verify_sampling(df)

                gravity = verify_gravity_removal(df)

                row = {
                    "Class": cls,
                    "File": filename
                }

                row.update(sampling)
                row.update(gravity)

                rows.append(row)

                print(
                    f"  {filename}: "
                    f"{'PASS' if sampling['Sampling PASS'] else 'CHECK'}"
                )

            except Exception as e:

                print(
                    f"  ERROR: {filename}: {e}"
                )

    return pd.DataFrame(rows)


# ============================================================
# PLOT SAMPLING VERIFICATION
# ============================================================

def plot_sampling(df, output_dir):

    classes = df["Class"].unique()

    plt.figure(figsize=(12, 6))

    for cls in classes:

        subset = df[df["Class"] == cls]

        plt.scatter(
            np.arange(len(subset)),
            subset["Measured Fs (Hz)"],
            label=cls,
            alpha=0.7
        )

    plt.axhline(
        FS_TARGET,
        linestyle="--",
        linewidth=2,
        label=f"Target = {FS_TARGET} Hz"
    )

    tolerance = (
        FS_TARGET *
        FS_TOLERANCE_PERCENT /
        100
    )

    plt.axhline(
        FS_TARGET + tolerance,
        linestyle=":"
    )

    plt.axhline(
        FS_TARGET - tolerance,
        linestyle=":"
    )

    plt.xlabel("File index")
    plt.ylabel("Measured sampling frequency (Hz)")
    plt.title("Sampling Frequency Verification")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "sampling_verification.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT GRAVITY VERIFICATION
# ============================================================

def plot_gravity(df, output_dir):

    classes = df["Class"].unique()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True
    )

    axis_names = ["X", "Y", "Z"]

    for i, axis in enumerate(axis_names):

        col = f"{axis} mean (m/s^2)"

        for cls in classes:

            subset = df[df["Class"] == cls]

            axes[i].scatter(
                np.arange(len(subset)),
                subset[col],
                label=cls,
                alpha=0.7
            )

        axes[i].axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        axes[i].set_ylabel(
            f"{axis} mean (m/s²)"
        )

        axes[i].grid(
            True,
            alpha=0.3
        )

    axes[-1].set_xlabel("File index")

    fig.suptitle(
        "Gravity Removal Verification — Mean Acceleration"
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "gravity_removal_verification.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# LOW FREQUENCY POWER PLOT
# ============================================================

def plot_frequency(df, output_dir):

    classes = df["Class"].unique()

    col = (
        f"Average low-frequency power "
        f"< {LOW_FREQ_LIMIT} Hz (%)"
    )

    plt.figure(figsize=(12, 6))

    for cls in classes:

        subset = df[df["Class"] == cls]

        plt.scatter(
            np.arange(len(subset)),
            subset[col],
            label=cls,
            alpha=0.7
        )

    plt.xlabel("File index")

    plt.ylabel(
        f"Power below {LOW_FREQ_LIMIT} Hz (%)"
    )

    plt.title(
        "Residual Very-Low-Frequency Power"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "frequency_verification.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# TEXT REPORT
# ============================================================

def generate_report(df, output_dir):

    report_path = os.path.join(
        output_dir,
        "verification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ACCELEROMETER PREPROCESSING VERIFICATION REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Target sampling frequency: {FS_TARGET} Hz\n"
        )

        f.write(
            f"Low-frequency verification limit: "
            f"{LOW_FREQ_LIMIT} Hz\n\n"
        )

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        total = len(df)

        sampling_pass = df[
            "Sampling PASS"
        ].sum()

        f.write(
            f"Total files checked: {total}\n"
        )

        f.write(
            f"Sampling verification passed: "
            f"{sampling_pass}/{total}\n\n"
        )

        # ----------------------------------------------------
        # Per class
        # ----------------------------------------------------

        for cls in CLASS_DIRS:

            subset = df[
                df["Class"] == cls
            ]

            if len(subset) == 0:
                continue

            f.write(
                f"\nCLASS: {cls}\n"
            )

            f.write(
                "-" * 50 + "\n"
            )

            f.write(
                f"Files: {len(subset)}\n"
            )

            f.write(
                f"Mean measured Fs: "
                f"{subset['Measured Fs (Hz)'].mean():.4f} Hz\n"
            )

            f.write(
                f"Std measured Fs: "
                f"{subset['Measured Fs (Hz)'].std():.6f} Hz\n"
            )

            f.write(
                f"Maximum dt deviation: "
                f"{subset['Maximum dt deviation (%)'].max():.6f}%\n"
            )

            f.write(
                f"Mean X acceleration: "
                f"{subset['X mean (m/s^2)'].mean():.6f} m/s²\n"
            )

            f.write(
                f"Mean Y acceleration: "
                f"{subset['Y mean (m/s^2)'].mean():.6f} m/s²\n"
            )

            f.write(
                f"Mean Z acceleration: "
                f"{subset['Z mean (m/s^2)'].mean():.6f} m/s²\n"
            )

            low_col = (
                f"Average low-frequency power "
                f"< {LOW_FREQ_LIMIT} Hz (%)"
            )

            f.write(
                f"Mean low-frequency power: "
                f"{subset[low_col].mean():.4f}%\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    root = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(
        root,
        "verification_results"
    )
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("=" * 60)
    print("ACCELEROMETER PREPROCESSING VERIFICATION")
    print("=" * 60)

    df = process_all(root)

    if len(df) == 0:

        print(
            "\nNo valid CSV files found."
        )

        return

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        output_dir,
        "verification_summary.csv"
    )

    df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_sampling(
        df,
        output_dir
    )

    plot_gravity(
        df,
        output_dir
    )

    plot_frequency(
        df,
        output_dir
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    generate_report(
        df,
        output_dir
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    total = len(df)

    passed = int(
        df["Sampling PASS"].sum()
    )

    print(
        f"\nSampling verification: "
        f"{passed}/{total} files passed"
    )

    print(
        f"\nMean measured sampling frequency:"
        f" {df['Measured Fs (Hz)'].mean():.4f} Hz"
    )

    print(
        f"Std of measured sampling frequency:"
        f" {df['Measured Fs (Hz)'].std():.6f} Hz"
    )

    print(
        f"\nMaximum observed dt deviation:"
        f" {df['Maximum dt deviation (%)'].max():.6f}%"
    )

    low_col = (
        f"Average low-frequency power "
        f"< {LOW_FREQ_LIMIT} Hz (%)"
    )

    print(
        f"\nMean residual low-frequency power:"
        f" {df[low_col].mean():.4f}%"
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  verification_summary.csv"
    )

    print(
        "  sampling_verification.png"
    )

    print(
        "  gravity_removal_verification.png"
    )

    print(
        "  frequency_verification.png"
    )

    print(
        "  verification_report.txt"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
