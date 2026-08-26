#!/usr/bin/env python3

"""
Visualize time-domain accelerometer features.

This script is placed INSIDE the time_domain_analysis folder.

Expected structure:

time_domain_analysis/
│
├── visualize_time_domain.py
├── servo_only/
├── piston_only/
├── both_simultaneous/
├── both_alternating/
│
├── summary_servo_only.csv
├── summary_piston_only.csv
├── summary_both_simultaneous.csv
├── summary_both_alternating.csv
└── all_trials_combined.csv

Creates:

time_domain_analysis/
│
└── visualization/
    ├── figures/
    │   ├── RMS_comparison.png
    │   ├── P2P_comparison.png
    │   ├── CrestFactor_comparison.png
    │   ├── Kurtosis_comparison.png
    │   ├── RMS_axiswise.png
    │   ├── P2P_axiswise.png
    │   ├── CrestFactor_axiswise.png
    │   └── Kurtosis_axiswise.png
    │
    ├── class_statistics.csv
    └── summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

CLASS_DIRS = [
    "servo_only",
    "piston_only",
    "both_simultaneous",
    "both_alternating"
]

AXES = [
    "ax",
    "ay",
    "az",
    "aabs"
]

AXIS_LABELS = {
    "ax": "X-axis",
    "ay": "Y-axis",
    "az": "Z-axis",
    "aabs": "Absolute acceleration"
}

CLASS_LABELS = {
    "servo_only": "Servo only",
    "piston_only": "Piston only",
    "both_simultaneous": "Both simultaneous",
    "both_alternating": "Both alternating"
}

METRICS = [
    "RMS",
    "P2P",
    "CrestFactor",
    "Kurtosis"
]

METRIC_LABELS = {
    "RMS": "RMS acceleration (m/s²)",
    "P2P": "Peak-to-peak acceleration (m/s²)",
    "CrestFactor": "Crest factor",
    "Kurtosis": "Excess kurtosis"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data(root):

    # IMPORTANT:
    # root is already time_domain_analysis/

    combined_file = os.path.join(
        root,
        "all_trials_combined.csv"
    )

    if not os.path.exists(combined_file):

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"{combined_file}\n\n"
            "Run the time-domain analysis script first."
        )

    df = pd.read_csv(combined_file)

    required_columns = [
        "class",
        "file",
        "axis",
        "RMS",
        "P2P",
        "CrestFactor",
        "Kurtosis"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# CALCULATE STATISTICS
# ============================================================

def calculate_statistics(df):

    stats = (
        df
        .groupby(["class", "axis"])[METRICS]
        .agg(["mean", "std", "min", "max"])
    )

    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    stats = stats.reset_index()

    return stats


# ============================================================
# PLOT: CLASS COMPARISON
# ============================================================

def plot_metric_by_class(df, metric, output_file):

    grouped = (
        df
        .groupby(["class", "axis"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(CLASS_DIRS))

    width = 0.18

    for i, axis_name in enumerate(AXES):

        subset = grouped[
            grouped["axis"] == axis_name
        ]

        means = []
        stds = []

        for cls in CLASS_DIRS:

            row = subset[
                subset["class"] == cls
            ]

            if len(row):

                means.append(
                    row["mean"].iloc[0]
                )

                stds.append(
                    row["std"].iloc[0]
                )

            else:

                means.append(np.nan)
                stds.append(np.nan)

        offset = (
            i - (len(AXES) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=AXIS_LABELS[axis_name]
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            CLASS_LABELS[c]
            for c in CLASS_DIRS
        ],
        rotation=15
    )

    ax.set_ylabel(
        METRIC_LABELS[metric]
    )

    ax.set_title(
        f"{METRIC_LABELS[metric]}: Motion Class Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PLOT: AXIS COMPARISON
# ============================================================

def plot_metric_by_axis(df, metric, output_file):

    grouped = (
        df
        .groupby(["class", "axis"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(AXES))

    width = 0.18

    for i, cls in enumerate(CLASS_DIRS):

        subset = grouped[
            grouped["class"] == cls
        ]

        means = []
        stds = []

        for axis_name in AXES:

            row = subset[
                subset["axis"] == axis_name
            ]

            if len(row):

                means.append(
                    row["mean"].iloc[0]
                )

                stds.append(
                    row["std"].iloc[0]
                )

            else:

                means.append(np.nan)
                stds.append(np.nan)

        offset = (
            i - (len(CLASS_DIRS) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=CLASS_LABELS[cls]
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            AXIS_LABELS[a]
            for a in AXES
        ]
    )

    ax.set_ylabel(
        METRIC_LABELS[metric]
    )

    ax.set_title(
        f"{METRIC_LABELS[metric]}: Axis-wise Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TEXT SUMMARY
# ============================================================

def create_summary(df, output_file):

    lines = []

    lines.append(
        "TIME-DOMAIN ACCELEROMETER ANALYSIS"
    )

    lines.append("=" * 65)

    lines.append("")

    for metric in METRICS:

        lines.append(
            f"\n{METRIC_LABELS[metric]}"
        )

        lines.append("-" * 65)

        grouped = (
            df
            .groupby("class")[metric]
            .agg(["mean", "std"])
        )

        for cls in CLASS_DIRS:

            if cls not in grouped.index:
                continue

            mean = grouped.loc[
                cls, "mean"
            ]

            std = grouped.loc[
                cls, "std"
            ]

            lines.append(
                f"{CLASS_LABELS[cls]:25s}"
                f"{mean:.4f} ± {std:.4f}"
            )

    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # THE SCRIPT IS INSIDE time_domain_analysis/
    # ========================================================

    root = os.path.dirname(
        os.path.abspath(__file__)
    )

    print("=" * 65)
    print("TIME-DOMAIN ACCELEROMETER VISUALIZATION")
    print("=" * 65)

    print(
        f"\nReading data from:\n{root}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data(root)

    print(
        f"\nLoaded {len(df)} feature records."
    )

    # ========================================================
    # TRIAL COUNT
    # ========================================================

    print("\nTrials detected:")

    for cls in CLASS_DIRS:

        n = df[
            df["class"] == cls
        ]["file"].nunique()

        print(
            f"  {CLASS_LABELS[cls]:25s}: {n}"
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_root = os.path.join(
        root,
        "visualization"
    )

    figures_dir = os.path.join(
        output_root,
        "figures"
    )

    os.makedirs(
        figures_dir,
        exist_ok=True
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print(
        "\nCalculating statistics..."
    )

    stats = calculate_statistics(df)

    stats.to_csv(
        os.path.join(
            output_root,
            "class_statistics.csv"
        ),
        index=False
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print(
        "\nGenerating plots..."
    )

    for metric in METRICS:

        print(
            f"  {metric}"
        )

        plot_metric_by_class(
            df,
            metric,
            os.path.join(
                figures_dir,
                f"{metric}_comparison.png"
            )
        )

        plot_metric_by_axis(
            df,
            metric,
            os.path.join(
                figures_dir,
                f"{metric}_axiswise.png"
            )
        )

    # ========================================================
    # TEXT SUMMARY
    # ========================================================

    create_summary(
        df,
        os.path.join(
            output_root,
            "summary.txt"
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 65)
    print("OVERALL CLASS COMPARISON")
    print("=" * 65)

    for metric in METRICS:

        print(
            f"\n{METRIC_LABELS[metric]}"
        )

        print("-" * 65)

        grouped = (
            df
            .groupby("class")[metric]
            .agg(["mean", "std"])
        )

        for cls in CLASS_DIRS:

            if cls not in grouped.index:
                continue

            mean = grouped.loc[
                cls,
                "mean"
            ]

            std = grouped.loc[
                cls,
                "std"
            ]

            print(
                f"{CLASS_LABELS[cls]:25s}"
                f"{mean:.4f} ± {std:.4f}"
            )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 65)
    print("DONE")
    print("=" * 65)

    print(
        f"\nResults saved in:\n"
        f"{output_root}/"
    )

    print(
        "\nFigures saved in:\n"
        f"{figures_dir}/"
    )


if __name__ == "__main__":
    main()
