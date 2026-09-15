# Results: Smartphone Vibration Signature Classification

This file summarizes the results. The full narrative lives in [`REPORT.md`](REPORT.md).

## Headline

The four motion modes produce distinct vibration signatures. A simple classifier on three
interpretable features reaches 95 to 97 percent cross-validated accuracy. Three of the four
classes reach perfect classification. The only recurring confusion falls on both-alternating.

## Dataset

The dataset holds 60 labeled recordings: 4 classes times 15 trials, plus 1 stationary
benchmark used only for verification. Each recording lasts about 5 seconds at about 402 Hz.
The pipeline turns each recording into one 64-feature vector, stored in
`results/tables/feature_matrix.csv` as a 60 by 64 table. The classes stay balanced at 15
recordings each.

## Features used (3 of the 64 available)

| Feature | Role |
|---|---|
| `log10(z_TotalPower)` | z-axis (linear, piston) energy, isolates servo at a near-zero value |
| `log10(xy_TotalPower)` | x-y (rotational, servo) energy, isolates piston at a near-zero value |
| `xy_RMS` | x-y vibration level, splits both-simultaneous from both-alternating |

The separability analysis in the report selected these three features. The data carries
roughly two independent dimensions (x-y energy and z energy), so three features suffice and
more add little.

## Model leaderboard (stratified 5-fold cross-validation, 60 recordings)

| Model | Validation (18 held out) | 5-fold CV | Pooled out-of-fold |
|---|---:|---:|---:|
| Random forest | 100.0 % | 96.7 % ± 4.1 | 96.7 % |
| Gaussian naive Bayes | 94.4 % | 96.7 % ± 6.7 | 96.7 % |
| LDA | 94.4 % | 95.0 % ± 4.1 | 95.0 % |
| k-nearest neighbours (k=5) | 94.4 % | 93.3 % ± 6.2 | 93.3 % |
| Logistic regression | 88.9 % | 93.3 % ± 6.2 | 93.3 % |

Quote the cross-validated figure of about 95 to 97 percent, not the single split. A single
validation set of 18 recordings gives a noisy estimate.

![Model comparison](results/plots/model_comparison.png)

## Best model, per-class performance (random forest, pooled out-of-fold)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| piston | 0.94 | 1.00 | 0.97 |
| servo | 1.00 | 1.00 | 1.00 |
| both-simultaneous | 0.94 | 1.00 | 0.97 |
| both-alternating | 1.00 | 0.87 | 0.93 |

Confusion matrix (rows give the true class, columns give the predicted class):

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

No. All five models reach a plateau by about 30 training recordings. The roughly 97 percent
ceiling comes from the features, which carry no timing information, not from the dataset
size. Only both-alternating suffers, because the distinction between "alternating" and
"simultaneous" is a temporal one that static per-recording features cannot fully capture.

## Figures (`results/plots/`, PDF and PNG)

| File | Content |
|---|---|
| `signals_by_class` | representative tri-axial signals per class |
| `spectra_by_class` | FFT amplitude spectra per class |
| `eda_feature_space` | class separation in two energy features |
| `eda_feature_distributions` | per-class distributions of six features |
| `eda_separability_ranking` | features ranked by ANOVA F |
| `eda_correlation_heatmap` | feature redundancy, the two energy blocks |
| `eda_pca_lda` | PCA and LDA projections |
| `model_comparison` | cross-validated accuracy per model |
| `classification_confusion` | confusion matrices for the two best models |
| `decision_regions` | two-feature decision boundaries |
| `learning_curves` | accuracy against training-set size |

## Tables (`results/tables/`)

| File | Content |
|---|---|
| `feature_matrix.csv` | the 60 by 64 feature matrix |
| `feature_separability_anova.csv` | every feature ranked by ANOVA F |
| `classification_performance.csv` | the model leaderboard above |
| `classification_report_<model>.csv` | per-class precision, recall, F1 |
| `confusion_matrix_<model>.csv` | confusion matrix per model |

## Reproduce

```bash
python3 scripts/eda.py          # feature matrix and EDA figures
python3 scripts/signal_plots.py # signal and spectrum figures
python3 scripts/classify.py     # models, curves, confusion
```

## Bottom line

Four mechanical motion modes separate cleanly from a phone accelerometer, at 95 to 97
percent with a simple interpretable model. The next accuracy gain comes from a temporal
feature through windowing, not from a more complex model.
