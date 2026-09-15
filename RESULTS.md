# Results — Smartphone Vibration Signature Classification

Condensed results summary. Full narrative in [`REPORT.md`](REPORT.md).

## Headline

The four motion modes produce clearly distinct vibration signatures. A simple classifier on
**3 interpretable features** reaches **95–97 % cross-validated accuracy**. Three of the four
classes are classified perfectly; the only recurring confusion is on **both-alternating**.

## Dataset

- **60 recordings** = 4 classes × 15 trials (+ 1 stationary benchmark), ~5 s each at ~402 Hz.
- Each recording → **one 64-feature vector** (`results/tables/feature_matrix.csv`, 60 × 64).
- Perfectly balanced: 15 per class.

## Features used (of the 64 available)

| Feature | Role |
|---|---|
| `log10(z_TotalPower)` | z-axis (linear/piston) energy → isolates **servo** (near zero) |
| `log10(xy_TotalPower)` | x–y (rotational/servo) energy → isolates **piston** (near zero) |
| `xy_RMS` | x–y vibration level → splits **both-sim vs both-alt** |

Chosen from the EDA separability analysis; the data is effectively 2-dimensional (x–y energy vs
z energy), so 3 features suffice and more are redundant.

## Model leaderboard (stratified 5-fold cross-validation, 60 recordings)

| Model | Validation (18 held-out) | 5-fold CV | Pooled out-of-fold |
|---|---:|---:|---:|
| **Random forest** | 100 % | **96.7 % ± 4.1** | 96.7 % |
| **Gaussian NB** | 94.4 % | **96.7 % ± 6.7** | 96.7 % |
| LDA | 94.4 % | 95.0 % ± 4.1 | 95.0 % |
| k-NN (k = 5) | 94.4 % | 93.3 % ± 6.2 | 93.3 % |
| Logistic regression | 88.9 % | 93.3 % ± 6.2 | 93.3 % |

> Quote the **cross-validated** figure (≈ 95–97 % ± ~5 %), not the single split — with 60
> recordings one validation set is noisy.

![Model comparison](results/plots/model_comparison.png)

## Best model — per-class performance (Random forest, pooled out-of-fold)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| piston | 0.94 | 1.00 | 0.97 |
| servo | 1.00 | 1.00 | 1.00 |
| both-simultaneous | 0.94 | 1.00 | 0.97 |
| both-alternating | 1.00 | 0.87 | 0.93 |

Confusion (rows = true, cols = predicted):

```
              piston  servo  both-sim  both-alt
piston           15      0        0        0
servo             0     15        0        0
both-sim          0      0       15        0
both-alt          1      0        1       13    <- the only errors
```

![Confusion matrices](results/plots/classification_confusion.png)

## Does more data help?

![Learning curves](results/plots/learning_curves.png)

No — all models **plateau by ~30 training recordings**. The ~97 % ceiling is set by the
**features** (they carry no timing information), not by dataset size. The only class that
suffers is **both-alternating**, because "alternating vs simultaneous" is a *temporal*
distinction that static per-recording features cannot fully capture.

## Full figure list (`results/plots/`, PDF + PNG)

| File | What it shows |
|---|---|
| `signals_by_class` | representative tri-axial signals per class |
| `spectra_by_class` | FFT amplitude spectra per class |
| `eda_feature_space` | class separation in 2 energy features |
| `eda_feature_distributions` | per-class distributions of 6 features |
| `eda_separability_ranking` | features ranked by ANOVA F |
| `eda_correlation_heatmap` | feature redundancy (two energy blocks) |
| `eda_pca_lda` | PCA vs LDA class structure |
| `model_comparison` | CV accuracy per model |
| `classification_confusion` | confusion matrices (best models) |
| `decision_regions` | 2-feature decision boundaries |
| `learning_curves` | accuracy vs training-set size |

## Tables (`results/tables/`)

- `feature_matrix.csv` — 60 × 64 feature matrix
- `feature_separability_anova.csv` — every feature ranked by ANOVA F
- `classification_performance.csv` — the leaderboard above
- `classification_report_<model>.csv`, `confusion_matrix_<model>.csv` — per model

## Reproduce

```bash
python3 scripts/eda.py          # feature matrix + EDA figures
python3 scripts/signal_plots.py # signal & spectrum figures
python3 scripts/classify.py     # models, curves, confusion
```

## Bottom line

Four mechanical motion modes are cleanly separable from a phone's accelerometer. **~95–97 %**
with a simple, interpretable model. The next accuracy gain is not a fancier model — it is a
**temporal feature** (windowing) to resolve *both-alternating*.
