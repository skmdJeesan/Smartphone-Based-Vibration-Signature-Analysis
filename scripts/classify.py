#!/usr/bin/env python3
"""
Classification of the four motion classes from the engineered feature matrix.

Models (all simple / appropriate for a 60-recording dataset):
    logistic regression, k-NN, LDA, Gaussian naive Bayes, random forest.

Features (chosen from the EDA separability analysis; the two power features are
log-scaled because they span orders of magnitude):
    log10(z_TotalPower)    z-axis (linear / piston) energy   -> isolates servo
    log10(xy_TotalPower)   x-y (rotational / servo) energy   -> isolates piston
    xy_RMS                 x-y vibration level               -> splits both-sim / both-alt

Evaluation
    - stratified held-out validation split (30%), reported per the brief
    - stratified 5-fold cross-validation over all 60 recordings (robust estimate)
    - pooled out-of-fold confusion matrix (every recording predicted once, unseen)

Outputs (results/)
    tables/classification_performance.csv        one row per model
    tables/classification_report_<model>.csv     per-class precision/recall/F1
    tables/confusion_matrix_<model>.csv          pooled out-of-fold confusion
    plots/model_comparison.{pdf,png}             CV accuracy per model
    plots/classification_confusion.{pdf,png}     confusion matrices (best models)
    plots/learning_curves.{pdf,png}              accuracy vs training-set size
    plots/decision_regions.{pdf,png}             2-feature decision boundaries

Run from the repo root:  python3 scripts/classify.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "iclr_style"))
import iclrplot as ip  # noqa: E402

TABLES = os.path.join(REPO, "results", "tables")
PLOTS = os.path.join(REPO, "results", "plots")
FEATURE_MATRIX = os.path.join(TABLES, "feature_matrix.csv")

ORDER = ["piston_only", "servo_only", "both_simultaneous", "both_alternating"]
SHORT = {"piston_only": "piston", "servo_only": "servo",
         "both_simultaneous": "both-sim", "both_alternating": "both-alt"}
RANDOM_STATE = 42
VAL_FRACTION = 0.30

ip.setup()
COLORS = dict(zip(ORDER, ip.palette("categorical", 4)))


def load_xy():
    w = pd.read_csv(FEATURE_MATRIX)
    w = w[w["class"].isin(ORDER)].copy()
    X = pd.DataFrame({
        "log10_z_power": np.log10(w["z_TotalPower"]),
        "log10_xy_power": np.log10(w["xy_TotalPower"]),
        "xy_RMS": w["xy_RMS"],
    })
    return X, w["class"].values, list(X.columns)


def models():
    return {
        "logreg": ("Logistic regression", LogisticRegression(max_iter=2000)),
        "knn": ("k-NN (k=5)", KNeighborsClassifier(n_neighbors=5)),
        "lda": ("LDA", LinearDiscriminantAnalysis()),
        "gnb": ("Gaussian NB", GaussianNB()),
        "rf": ("Random forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
    }


def pipe_of(clf):
    return make_pipeline(StandardScaler(), clf)


# ---------------------------------------------------------------- figures
def plot_model_comparison(perf):
    d = perf.sort_values("cv5_accuracy_mean_%")
    fig, ax = ip.figure(width="wide", aspect=0.6)
    ax.barh(range(len(d)), d["cv5_accuracy_mean_%"], xerr=d["cv5_accuracy_std_%"],
            color=ip.CATEGORICAL[0], edgecolor="white", linewidth=0.4,
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor="0.35"))
    ax.set_yticks(range(len(d)), d["model"])
    ax.set_xlim(80, 100)
    ax.set_xlabel("5-fold cross-validated accuracy (%)")
    ax.grid(axis="y", visible=False)
    ip.finish(fig, os.path.join(PLOTS, "model_comparison"))


def plot_confusions(items):
    fig, axes = ip.figure(width="full", ncols=len(items), aspect=1.05)
    axes = np.atleast_1d(axes).ravel()
    labels = [SHORT[c] for c in ORDER]
    for ax, (name, cm, acc) in zip(axes, items):
        ip.heatmap(ax, cm, xlabels=labels, ylabels=labels, signed=False, variant="single",
                   annotate=True, fmt="{:.0f}", square=True, cbar_label="recordings")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ip.label_panels(axes, titles=[f"{n} - {a*100:.0f}%" for n, _, a in items])
    ip.finish(fig, os.path.join(PLOTS, "classification_confusion"))


def plot_learning_curves(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.25, 1.0, 6)
    fig, axes = ip.figure(width="full", nrows=2, ncols=3, aspect=0.8)
    axes = np.atleast_1d(axes).ravel()
    cat = ip.palette("categorical", 2)
    for ax, (key, (name, clf)) in zip(axes, models().items()):
        n, tr, te = learning_curve(pipe_of(clf), X, y, cv=cv, train_sizes=sizes,
                                   scoring="accuracy", shuffle=True, random_state=RANDOM_STATE)
        for scores, lab, c in [(tr, "train", cat[1]), (te, "cross-val", cat[0])]:
            m, s = scores.mean(1) * 100, scores.std(1) * 100
            ax.plot(n, m, marker="o", color=c, label=lab)
            ax.fill_between(n, m - s, m + s, color=c, alpha=0.15, linewidth=0)
        ax.set_ylim(60, 103); ax.set_xlabel("training recordings"); ax.set_ylabel("accuracy (%)")
        ax.legend(loc="lower right", fontsize=7)
    axes[-1].axis("off")
    ip.label_panels(axes[:5], titles=[name for _, (name, _) in models().items()])
    ip.finish(fig, os.path.join(PLOTS, "learning_curves"))


def plot_decision_regions(X, y):
    # 2-feature view (the two energy features) so the boundary is drawable
    f = ["log10_z_power", "log10_xy_power"]
    X2 = X[f].values
    sc = StandardScaler().fit(X2); Z = sc.transform(X2)
    clf = LinearDiscriminantAnalysis().fit(Z, y)
    x0, x1 = Z[:, 0].min() - .5, Z[:, 0].max() + .5
    y0, y1 = Z[:, 1].min() - .5, Z[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    grid = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    idx = {c: i for i, c in enumerate(ORDER)}
    grid_i = np.vectorize(idx.get)(grid)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLORS[c] for c in ORDER])
    fig, ax = ip.figure(width="wide", aspect=0.8)
    ax.contourf(xx, yy, grid_i, alpha=0.18, levels=[-.5, .5, 1.5, 2.5, 3.5], cmap=cmap)
    for c in ORDER:
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], color=COLORS[c], label=SHORT[c], s=18,
                   edgecolor="white", linewidth=0.3)
    ax.set_xlabel("z energy  (standardised log10 z-power)")
    ax.set_ylabel("x-y energy  (standardised log10 xy-power)")
    ax.legend(loc="lower left", handletextpad=0.3)
    ip.finish(fig, os.path.join(PLOTS, "decision_regions"))


def main():
    X, y, feats = load_xy()
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=VAL_FRACTION, stratify=y,
                                            random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Features: {feats}")
    print(f"Stratified split: train={len(ytr)}, validation={len(yval)} "
          f"({len(yval)//len(ORDER)}/class)\n")

    perf_rows, conf_items = [], []
    for key, (name, clf) in models().items():
        pipe = pipe_of(clf)
        pipe.fit(Xtr, ytr)
        val_acc = accuracy_score(yval, pipe.predict(Xval))
        cvs = cross_val_score(pipe, X, y, cv=cv)
        pooled = cross_val_predict(pipe, X, y, cv=cv)
        cm = confusion_matrix(y, pooled, labels=ORDER)
        pooled_acc = accuracy_score(y, pooled)

        pd.DataFrame(confusion_matrix(y, pooled, labels=ORDER),
                     index=[SHORT[c] for c in ORDER], columns=[SHORT[c] for c in ORDER]).to_csv(
            os.path.join(TABLES, f"confusion_matrix_{key}.csv"))
        pd.DataFrame(classification_report(y, pooled, labels=ORDER,
                     target_names=[SHORT[c] for c in ORDER], output_dict=True, zero_division=0)).T.to_csv(
            os.path.join(TABLES, f"classification_report_{key}.csv"))

        perf_rows.append({"model": name, "features": ", ".join(feats),
                          "val_accuracy_%": round(val_acc * 100, 1),
                          "cv5_accuracy_mean_%": round(cvs.mean() * 100, 1),
                          "cv5_accuracy_std_%": round(cvs.std() * 100, 1),
                          "pooled_oof_accuracy_%": round(pooled_acc * 100, 1)})
        conf_items.append((key, name, cm, pooled_acc))
        print(f"{name:20s} val={val_acc*100:5.1f}%  CV5={cvs.mean()*100:5.1f}%±{cvs.std()*100:.1f}  "
              f"pooled={pooled_acc*100:5.1f}%")

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(os.path.join(TABLES, "classification_performance.csv"), index=False)

    # figures
    plot_model_comparison(perf)
    best = sorted(conf_items, key=lambda t: -t[3])
    show = [(n, cm, a) for (_, n, cm, a) in best if n in ("LDA", "Random forest")]
    plot_confusions(show)
    plot_learning_curves(X, y)
    plot_decision_regions(X, y)

    print("\nTop misclassification pattern (pooled, best model):")
    _, name, cm, _ = best[0]
    print(f"  {name}: " + pd.DataFrame(cm, index=[SHORT[c] for c in ORDER],
          columns=[SHORT[c] for c in ORDER]).to_string().replace("\n", "\n  "))
    print("\nWrote performance table, per-model reports/confusions, and 4 figures.")


if __name__ == "__main__":
    main()
