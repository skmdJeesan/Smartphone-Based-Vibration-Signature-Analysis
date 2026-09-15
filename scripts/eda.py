#!/usr/bin/env python3
"""
Exploratory Data Analysis for the smartphone vibration-signature project.

Builds a single per-trial feature matrix from the committed time-domain and
frequency-domain per-trial metrics, then produces classification-oriented EDA:

    - feature matrix (60 trials x features)                 -> results/tables/
    - per-class feature distributions (boxplots)            -> results/plots/
    - feature separability ranking (one-way ANOVA F)        -> results/plots/
    - feature-feature correlation heatmap                   -> results/plots/
    - PCA and LDA 2-D projections of the 4 classes          -> results/plots/
    - pairwise scatter of the most separable features       -> results/plots/

Run from the repo root:  python3 scripts/eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(REPO, "results", "time_domain_analysis", "all_trials_combined.csv")
FD = os.path.join(REPO, "results", "frequency_domain_analysis", "all_trials_combined.csv")
PLOTS = os.path.join(REPO, "results", "plots")
TABLES = os.path.join(REPO, "results", "tables")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

ORDER = ["piston_only", "servo_only", "both_simultaneous", "both_alternating"]
PALETTE = {"piston_only": "#d1495b", "servo_only": "#2e86ab",
           "both_simultaneous": "#6a994e", "both_alternating": "#e09f3e"}
sns.set_theme(style="whitegrid", context="talk")


# ------------------------------------------------------------------
# 1. Build the per-trial feature matrix (wide: one row per trial)
# ------------------------------------------------------------------
def build_feature_matrix():
    td = pd.read_csv(TD)
    fd = pd.read_csv(FD).drop(columns=["Fs_Hz", "Nyquist_Hz"], errors="ignore")
    df = td.merge(fd, on=["class", "file", "axis"], how="inner")

    metrics = [c for c in df.columns if c not in ("class", "file", "axis")]
    wide = df.pivot_table(index=["class", "file"], columns="axis", values=metrics)
    wide.columns = [f"{m}_{ax}" for m, ax in wide.columns]
    wide = wide.reset_index()

    # engineered, physically interpretable features
    wide["xy_TotalPower"] = wide["PSD_TotalSpectralPower_ax"] + wide["PSD_TotalSpectralPower_ay"]
    wide["z_TotalPower"] = wide["PSD_TotalSpectralPower_az"]
    wide["z_over_xy_power"] = wide["z_TotalPower"] / wide["xy_TotalPower"].replace(0, np.nan)
    wide["xy_RMS"] = np.hypot(wide["RMS_ax"], wide["RMS_ay"])

    # tidy up non-finite values
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide["class"] = pd.Categorical(wide["class"], categories=ORDER, ordered=True)
    wide = wide.sort_values(["class", "file"]).reset_index(drop=True)
    return wide


def feature_columns(wide):
    return [c for c in wide.columns if c not in ("class", "file")]


# ------------------------------------------------------------------
# 2. Separability ranking via one-way ANOVA F across the 4 classes
# ------------------------------------------------------------------
def separability(wide):
    feats = feature_columns(wide)
    rows = []
    for f in feats:
        groups = [wide.loc[wide["class"] == c, f].dropna().values for c in ORDER]
        if all(len(g) > 1 for g in groups):
            F, p = f_oneway(*groups)
            rows.append({"feature": f, "F": F, "p": p})
    rank = pd.DataFrame(rows).sort_values("F", ascending=False).reset_index(drop=True)
    return rank


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------
def plot_boxplots(wide):
    feats = ["z_TotalPower", "xy_TotalPower", "z_over_xy_power",
             "RMS_az", "xy_RMS", "PSD_F95_Hz_aabs",
             "FFT_DominantFrequency_Hz_aabs", "CrestFactor_aabs", "Kurtosis_az"]
    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=wide, x="class", y=f, order=ORDER, hue="class", palette=PALETTE,
                    legend=False, ax=ax, fliersize=0)
        sns.stripplot(data=wide, x="class", y=f, order=ORDER, ax=ax,
                      color="k", size=4, alpha=0.5, jitter=0.2)
        ax.set_title(f, fontsize=15); ax.set_xlabel("")
        ax.set_xticklabels(["piston", "servo", "both-sim", "both-alt"], rotation=0, fontsize=11)
        if f in ("z_TotalPower", "xy_TotalPower", "z_over_xy_power"):
            ax.set_yscale("log")
    fig.suptitle("Feature distributions by class (log scale where noted)", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(PLOTS, "eda_feature_distributions.png"), dpi=130); plt.close(fig)


def plot_separability(rank):
    top = rank.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.barh(top["feature"], top["F"], color="#4c72b0")
    ax.set_xlabel("one-way ANOVA F-statistic (higher = separates the 4 classes better)")
    ax.set_title("Top 20 most discriminative features", fontsize=18, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "eda_separability_ranking.png"), dpi=130); plt.close(fig)


def plot_correlation(wide, rank):
    feats = rank.head(15)["feature"].tolist()
    corr = wide[feats].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation among top-15 features (redundancy check)", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "eda_correlation_heatmap.png"), dpi=130); plt.close(fig)


def plot_pca_lda(wide):
    feats = feature_columns(wide)
    X = wide[feats].copy()
    X = X.fillna(X.median())
    y = wide["class"].astype(str).values
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2).fit(Xs); Xp = pca.transform(Xs)
    lda = LinearDiscriminantAnalysis(n_components=2).fit(Xs, y); Xl = lda.transform(Xs)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 8.5))
    for c in ORDER:
        m = y == c
        a1.scatter(Xp[m, 0], Xp[m, 1], c=PALETTE[c], label=c, s=90, edgecolor="k", linewidth=0.4)
        a2.scatter(Xl[m, 0], Xl[m, 1], c=PALETTE[c], label=c, s=90, edgecolor="k", linewidth=0.4)
    a1.set_title(f"PCA (unsupervised)\nPC1+PC2 explain {pca.explained_variance_ratio_[:2].sum()*100:.0f}% of variance", fontsize=15)
    a1.set_xlabel("PC1"); a1.set_ylabel("PC2"); a1.legend(fontsize=11)
    a2.set_title("LDA (supervised: maximises class separation)", fontsize=15)
    a2.set_xlabel("LD1"); a2.set_ylabel("LD2"); a2.legend(fontsize=11)
    fig.suptitle("Multivariate class structure across all features", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOTS, "eda_pca_lda.png"), dpi=130); plt.close(fig)


def plot_pairs(wide, rank):
    # pick top separable but low-redundancy features
    feats = ["z_TotalPower", "xy_TotalPower", "PSD_F95_Hz_aabs", "CrestFactor_aabs"]
    sub = wide[["class"] + feats].copy()
    sub["class"] = sub["class"].astype(str)
    for f in ("z_TotalPower", "xy_TotalPower"):
        sub[f] = np.log10(sub[f])
        sub = sub.rename(columns={f: f"log10_{f}"})
    g = sns.pairplot(sub, hue="class", hue_order=ORDER, palette=PALETTE,
                     diag_kind="kde", plot_kws=dict(s=60, edgecolor="k", linewidth=0.3))
    g.figure.suptitle("Pairwise scatter of selected features", y=1.02, fontsize=18, fontweight="bold")
    g.savefig(os.path.join(PLOTS, "eda_pairplot.png"), dpi=120); plt.close(g.figure)


def main():
    wide = build_feature_matrix()
    wide.to_csv(os.path.join(TABLES, "feature_matrix.csv"), index=False)

    rank = separability(wide)
    rank.to_csv(os.path.join(TABLES, "feature_separability_anova.csv"), index=False)

    plot_boxplots(wide)
    plot_separability(rank)
    plot_correlation(wide, rank)
    plot_pca_lda(wide)
    plot_pairs(wide, rank)

    print(f"Feature matrix: {wide.shape[0]} trials x {len(feature_columns(wide))} features")
    print("Class balance:\n", wide["class"].value_counts().reindex(ORDER).to_string())
    print("\nTop 12 most discriminative features (ANOVA F):")
    print(rank.head(12).to_string(index=False))
    print("\nWrote feature_matrix.csv, feature_separability_anova.csv and 5 figures.")


if __name__ == "__main__":
    main()
