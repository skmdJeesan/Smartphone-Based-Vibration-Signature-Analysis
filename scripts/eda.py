#!/usr/bin/env python3
"""
Exploratory Data Analysis for the smartphone vibration-signature project.

Builds a single per-trial feature matrix from the committed time-domain and
frequency-domain per-trial metrics, then produces classification-oriented EDA
in the ICLR publication style (see scripts/iclr_style/).

Outputs
    results/tables/feature_matrix.csv                 60 trials x features
    results/tables/feature_separability_anova.csv     ANOVA F per feature
    results/plots/eda_feature_space.{pdf,png}         class separation, 2 features
    results/plots/eda_feature_distributions.{pdf,png} per-class distributions
    results/plots/eda_separability_ranking.{pdf,png}  most discriminative features
    results/plots/eda_correlation_heatmap.{pdf,png}   feature redundancy
    results/plots/eda_pca_lda.{pdf,png}               multivariate class structure

Run from the repo root:  python3 scripts/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "iclr_style"))
import iclrplot as ip  # noqa: E402

TD = os.path.join(REPO, "results", "time_domain_analysis", "all_trials_combined.csv")
FD = os.path.join(REPO, "results", "frequency_domain_analysis", "all_trials_combined.csv")
PLOTS = os.path.join(REPO, "results", "plots")
TABLES = os.path.join(REPO, "results", "tables")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

ORDER = ["piston_only", "servo_only", "both_simultaneous", "both_alternating"]
SHORT_CLS = {"piston_only": "piston", "servo_only": "servo",
             "both_simultaneous": "both-sim", "both_alternating": "both-alt"}

ip.setup()
np.random.seed(0)
COLORS = dict(zip(ORDER, ip.palette("categorical", 4)))


def short(f):
    """Compact a feature name for axis/tick labels."""
    return (f.replace("PSD_", "").replace("CWT_", "CWT ").replace("_m_s2", "")
            .replace("DominantAmplitude", "DomAmp").replace("DominantFrequency", "DomFreq")
            .replace("GlobalWaveletPower", "GlobWavPow").replace("MeanWaveletPower", "MeanWavPow")
            .replace("SpectralRMS", "SpecRMS").replace("SpectralCentroid", "SpecCentroid")
            .replace("SpectralBandwidth", "SpecBW").replace("TotalSpectralPower", "TotPow")
            .replace("CrestFactor", "Crest").replace("_Hz", ""))


# ------------------------------------------------------------------ data
def build_feature_matrix():
    td = pd.read_csv(TD)
    fd = pd.read_csv(FD).drop(columns=["Fs_Hz", "Nyquist_Hz"], errors="ignore")
    df = td.merge(fd, on=["class", "file", "axis"], how="inner")

    metrics = [c for c in df.columns if c not in ("class", "file", "axis")]
    wide = df.pivot_table(index=["class", "file"], columns="axis", values=metrics)
    wide.columns = [f"{m}_{ax}" for m, ax in wide.columns]
    wide = wide.reset_index()

    wide["xy_TotalPower"] = wide["PSD_TotalSpectralPower_ax"] + wide["PSD_TotalSpectralPower_ay"]
    wide["z_TotalPower"] = wide["PSD_TotalSpectralPower_az"]
    wide["z_over_xy_power"] = wide["z_TotalPower"] / wide["xy_TotalPower"].replace(0, np.nan)
    wide["xy_RMS"] = np.hypot(wide["RMS_ax"], wide["RMS_ay"])

    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide["class"] = pd.Categorical(wide["class"], categories=ORDER, ordered=True)
    return wide.sort_values(["class", "file"]).reset_index(drop=True)


def feature_columns(wide):
    return [c for c in wide.columns if c not in ("class", "file")]


def separability(wide):
    rows = []
    for f in feature_columns(wide):
        groups = [wide.loc[wide["class"] == c, f].dropna().values for c in ORDER]
        if all(len(g) > 1 for g in groups):
            F, p = f_oneway(*groups)
            rows.append({"feature": f, "F": F, "p": p})
    return pd.DataFrame(rows).sort_values("F", ascending=False).reset_index(drop=True)


def _boxes(ax, wide, f, logy=False):
    data = [wide.loc[wide["class"] == c, f].dropna().values for c in ORDER]
    bp = ax.boxplot(data, positions=range(4), widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(color="0.45", linewidth=0.7),
                    capprops=dict(color="0.45", linewidth=0.7),
                    boxprops=dict(linewidth=0.6, edgecolor="0.3"))
    for patch, c in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[c]); patch.set_alpha(0.55)
    for i, (c, v) in enumerate(zip(ORDER, data)):
        ax.scatter(np.random.normal(i, 0.07, len(v)), v, s=4, color=COLORS[c],
                   edgecolor="none", alpha=0.85, zorder=3)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(range(4), [SHORT_CLS[c] for c in ORDER], rotation=30, ha="right")
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ figures
def fig_feature_space(wide):
    fig, ax = ip.figure(width="wide", aspect=0.74)
    for c in ORDER:
        s = wide[wide["class"] == c]
        ax.scatter(s["xy_TotalPower"], s["z_TotalPower"], color=COLORS[c], label=SHORT_CLS[c],
                   s=18, edgecolor="white", linewidth=0.3)
    ax.set(xscale="log", yscale="log",
           xlabel="x-y spectral power (rotation / servo energy)",
           ylabel="z spectral power (linear / piston energy)")
    ax.legend(loc="lower left", handletextpad=0.3, borderpad=0.3)
    ip.finish(fig, os.path.join(PLOTS, "eda_feature_space"))


def fig_distributions(wide):
    feats = ["z_TotalPower", "xy_TotalPower", "z_over_xy_power", "RMS_az", "xy_RMS", "CrestFactor_aabs"]
    logset = {"z_TotalPower", "xy_TotalPower", "z_over_xy_power"}
    fig, axes = ip.figure(width="full", nrows=2, ncols=3, aspect=0.85)
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        _boxes(ax, wide, f, logy=f in logset)
    ip.label_panels(axes, titles=[short(f) for f in feats])
    ip.finish(fig, os.path.join(PLOTS, "eda_feature_distributions"))


def fig_separability(rank):
    top = rank.head(15).iloc[::-1]
    fig, ax = ip.figure(width="wide", height=3.5)
    ax.barh(range(len(top)), top["F"], color=ip.CATEGORICAL[0], edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)), [short(f) for f in top["feature"]])
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlabel("one-way ANOVA F  (higher = better class separation)")
    ax.grid(axis="y", visible=False)
    ip.finish(fig, os.path.join(PLOTS, "eda_separability_ranking"))


def fig_correlation(wide, rank):
    feats = rank.head(15)["feature"].tolist()
    corr = wide[feats].corr().values
    labels = [short(f) for f in feats]
    fig, ax = ip.figure(width="full", aspect=0.98)
    ip.heatmap(ax, corr, xlabels=labels, ylabels=labels, signed=True,
               annotate=True, fmt="{:.2f}", square=True, cbar_label="Pearson r")
    plt.setp(ax.get_xticklabels(), rotation=90)
    ip.finish(fig, os.path.join(PLOTS, "eda_correlation_heatmap"))


def fig_pca_lda(wide):
    feats = feature_columns(wide)
    X = wide[feats].copy()
    X = X.fillna(X.median())
    y = wide["class"].astype(str).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Xs); Xp = pca.transform(Xs)
    Xl = LinearDiscriminantAnalysis(n_components=2).fit(Xs, y).transform(Xs)
    var = pca.explained_variance_ratio_[:2].sum() * 100

    fig, (a1, a2) = ip.figure(width="full", ncols=2, aspect=0.9)
    for c in ORDER:
        m = y == c
        a1.scatter(Xp[m, 0], Xp[m, 1], color=COLORS[c], label=SHORT_CLS[c], s=15,
                   edgecolor="white", linewidth=0.3)
        a2.scatter(Xl[m, 0], Xl[m, 1], color=COLORS[c], label=SHORT_CLS[c], s=15,
                   edgecolor="white", linewidth=0.3)
    a1.set(xlabel="PC1", ylabel="PC2")
    a2.set(xlabel="LD1", ylabel="LD2")
    ip.label_panels([a1, a2], titles=[f"PCA (unsupervised) - {var:.0f}% variance",
                                      "LDA (supervised)"])
    a1.legend(loc="best", handletextpad=0.3, borderpad=0.3)
    ip.finish(fig, os.path.join(PLOTS, "eda_pca_lda"))


def main():
    wide = build_feature_matrix()
    wide.to_csv(os.path.join(TABLES, "feature_matrix.csv"), index=False)
    rank = separability(wide)
    rank.to_csv(os.path.join(TABLES, "feature_separability_anova.csv"), index=False)

    fig_feature_space(wide)
    fig_distributions(wide)
    fig_separability(rank)
    fig_correlation(wide, rank)
    fig_pca_lda(wide)

    print(f"Feature matrix: {wide.shape[0]} trials x {len(feature_columns(wide))} features "
          f"| balance {dict(wide['class'].value_counts().reindex(ORDER))}")
    print("Top 8 features (ANOVA F):")
    print(rank.head(8).to_string(index=False))
    print("Wrote 2 tables and 5 ICLR-style figures (pdf + png).")


if __name__ == "__main__":
    main()
