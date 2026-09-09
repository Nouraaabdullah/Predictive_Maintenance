# Predictive Maintenance — Machine Failure Classification

## Course

**Advanced Machine Learning Methods — SDAIA Academy**

This repository contains the Final Applied Project for the Advanced Machine Learning Methods course at SDAIA Academy.

---

## Problem Statement

Unexpected machine failures can interrupt industrial operations and lead to avoidable maintenance and production costs.

The objective of this project is to predict whether a machine operating instance will result in a failure using only operational information available before the failure outcome is known.

The model is designed to support preventive-maintenance decisions by identifying high-risk operating conditions that may require inspection before normal operation continues.

- **Prediction moment:** Immediately before deciding whether the current operating instance should continue normally or be referred for maintenance inspection.
- **Decision supported:** Flag a machine operating instance for preventive-maintenance inspection or allow normal operation.
- **Positive class:** Machine failure.

---

## Dataset

- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source / link:** https://archive.ics.uci.edu/dataset/601/ai4i
- **Provider:** UCI Machine Learning Repository
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Redistribution permitted:** Yes, with appropriate attribution
- **Size:** 10,000 rows × 14 original columns
- **Target:** `Machine failure`
- **Positive class (=1):** A machine failure occurred
- **Class balance:** Approximately 3.39% positive

### Why this dataset was selected

This dataset is suitable for a tabular binary-classification project and contains a meaningful rare-event prediction problem.

Its class imbalance makes it appropriate for PR-AUC analysis, threshold selection and imbalance handling. It also supports leakage analysis, gradient boosting, model interpretability and probability calibration while remaining small enough for repeated cross-validation and hyperparameter tuning.

---

## Features

After leakage and identifier removal, the final modelling features were:

- `Type`
- `Air_temperature_K`
- `Process_temperature_K`
- `Rotational_speed_rpm`
- `Torque_Nm`
- `Tool_wear_min`

The following columns were removed:

- `UDI` / `UID` — observation identifiers
- `Product ID` — product identifier
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The five failure-mode columns were excluded because they directly define the overall `Machine failure` target and would therefore create target leakage.

---

## Methodology

### Validation

A **Stratified 5-Fold Cross-Validation** strategy was used because the dataset contains independent rows, no meaningful time dimension, and no repeated machine/entity identifier suitable for group-aware validation.

Stratification preserves the approximately 3.4% machine-failure rate across folds.

The complete dataset was also divided into:

- **60% training**
- **20% calibration**
- **20% final hold-out**

The final hold-out was kept untouched until all model, calibration and threshold decisions were fixed.

### Leakage Controls

All five leakage types covered in the course were considered:

- **Post-outcome leakage:** `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` were removed.
- **ID leakage:** `UDI` / `UID` and `Product ID` were removed.
- **Temporal leakage:** Not applicable because the dataset has no time dimension.
- **Group leakage:** Not applicable because rows are treated as independent observations.
- **Preprocessing leakage:** No learned transformation was fitted on the complete dataset outside the training folds.

---

## Baseline Model

The baseline model was **LightGBM** with:

- learning rate = 0.03
- number-of-trees ceiling = 2,000
- `num_leaves = 31`
- `min_child_samples = 40`
- row subsampling = 0.8
- feature subsampling = 0.8
- L2 regularisation = 1.0
- random seed = 42
- early stopping within each validation fold

Early stopping selected approximately 130–254 trees across the five folds.

### Baseline Cross-Validation Performance

- **OOF ROC-AUC:** 0.9714
- **OOF PR-AUC:** 0.7958
- **Failure base rate:** 0.0338
- **Mean fold ROC-AUC:** 0.9694 ± 0.0205
- **Mean fold PR-AUC:** 0.7960 ± 0.0841

The PR-AUC is substantially above the approximately 3.4% base-rate reference, indicating strong ranking performance for the rare machine-failure class.

---

## Imbalance Handling

The dataset is strongly imbalanced, so class weighting was evaluated using `scale_pos_weight`.

### Before vs After Weighting

| Model | ROC-AUC | PR-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Unweighted LightGBM | 0.9714 | 0.7958 | 87.59% | 59.11% | 0.7059 |
| Class-weighted LightGBM | 0.9698 | 0.7878 | 70.32% | 75.86% | 0.7299 |

Class weighting increased recall at the default threshold but reduced precision and did not improve PR-AUC.

Therefore, the **unweighted LightGBM** was carried forward, and the imbalance trade-off was handled at the decision-threshold layer instead.

---

## Evaluation Metrics

The primary evaluation metrics were:

- **PR-AUC** — main ranking metric because machine failures are rare
- **ROC-AUC** — overall ranking quality
- **Precision** — share of machine-failure alerts that are correct
- **Recall** — share of actual failures detected
- **F1 score** — balance between precision and recall
- **Confusion matrix** — actual decision counts
- **Brier score** — probability-quality metric

At the default 0.5 threshold, the baseline OOF confusion matrix was:

- True negatives: 5,780
- False positives: 17
- False negatives: 83
- True positives: 120

This produced:

- Precision: 87.59%
- Recall: 59.11%
- F1: 0.7059

---

## Threshold Strategy

No documented monetary false-positive or false-negative costs were available for this dataset, so business costs were not invented.

The operating rule was therefore:

**Maximum F1**

The selected raw threshold was:

**0.430**

At this operating point:

- **Precision:** 84.71%
- **Recall:** 65.52%
- **F1:** 0.7389

Compared with the default 0.5 threshold, the lower threshold detects more machine failures while maintaining strong alert precision.

---

## Hyperparameter Tuning

A small reproducible Optuna study was run using:

- **15 trials**
- `TPESampler(seed=42)`
- fixed stratified cross-validation
- PR-AUC as the optimisation objective

The best Optuna search value was approximately **0.7969**.

After re-evaluating the best parameters with the fixed five-fold validation scheme:

- Baseline PR-AUC: 0.7958
- Tuned PR-AUC: 0.8024
- Gain: +0.0067

The gain was very small relative to the observed fold variability.

Therefore, the original baseline LightGBM was retained rather than claiming that the tuned configuration was reliably better.

---

## Model Interpretability

### Global Interpretation

Permutation importance and SHAP were used for global interpretation.

The main predictive drivers were:

1. Torque
2. Air temperature
3. Rotational speed
4. Process temperature / tool wear
5. Product type

Torque caused the largest reduction in PR-AUC when permuted, indicating that the model relies heavily on this feature.

The SHAP beeswarm also showed that high torque, higher air temperature and greater tool wear often push predictions toward higher failure risk.

### Local Interpretation

A case close to the selected threshold was explained using a SHAP waterfall.

For case #4753:

- Predicted raw score: approximately 0.432
- Decision threshold: 0.430

The main positive contributors were:

1. Torque = 62.4 Nm
2. Rotational speed = 1330 rpm
3. Air temperature = 302.2 K

Process temperature slightly reduced the predicted risk, while product type and tool wear had little effect for this individual case.

SHAP values describe how the model uses the features and should not be interpreted as proof of causal relationships.

---

## Calibration

Probability calibration was evaluated on the dedicated 20% calibration split.

Before sigmoid calibration:

- Mean predicted probability: 0.0302
- Observed failure rate: 0.0340
- Brier score: 0.0139
- ROC-AUC: 0.9757
- PR-AUC: 0.7836

After sigmoid calibration:

- Mean predicted probability: 0.0340
- Observed failure rate: 0.0340
- Brier score: 0.0138
- ROC-AUC: 0.9757
- PR-AUC: 0.7836

Sigmoid calibration slightly improved probability quality without changing ranking performance.

The calibrated equivalent of the original 0.430 decision threshold became:

**0.4139**

---

## Optional Ensembling and Stacking

An XGBoost model was also evaluated as a bonus experiment.

### XGBoost

- OOF ROC-AUC: 0.9674
- OOF PR-AUC: 0.7142

LightGBM and XGBoost predictions had a high correlation of approximately **0.934**, indicating limited diversity.

### Simple Average

- ROC-AUC: 0.9707
- PR-AUC: 0.7788

### OOF Stack

- ROC-AUC: 0.9701
- PR-AUC: 0.7852

Neither ensemble method outperformed the single LightGBM model, while both introduced additional operational complexity.

---

## Results

| Model | ROC-AUC | PR-AUC | F1 under rule | Threshold | Brier | Complexity |
|---|---:|---:|---:|---:|---:|---|
| Baseline LightGBM | 0.9714 | 0.7958 | 0.7389 | 0.4300 | 0.0134 | Low |
| Class-weighted LightGBM | 0.9698 | 0.7878 | 0.7462 | 0.6350 | 0.0152 | Low |
| Optuna-tuned LightGBM | 0.9695 | 0.8024 | 0.7519 | 0.3450 | 0.0130 | Medium |
| Simple average | 0.9707 | 0.7788 | 0.7373 | 0.2600 | 0.0145 | High |
| OOF stack | 0.9701 | 0.7852 | 0.7415 | 0.1350 | 0.0145 | High |
| **FINAL — calibrated LightGBM hold-out** | **0.9713** | **0.8117** | **0.7642** | **0.4139** | **0.0125** | Final pipeline |

> Note: Intermediate candidate metrics come from development-stage cross-validation or calibration analyses, while the final row is the one-time untouched hold-out evaluation.

---

## Final Model Selection

The final model is:

**Baseline LightGBM + sigmoid calibration**

The final operating threshold is:

**0.4139**

The baseline model was selected because it provides a strong balance between discrimination, calibration quality, interpretability, reproducibility and operational simplicity.

The Optuna-tuned model produced only a small improvement relative to cross-validation variability, while the ensemble alternatives were more complex and did not outperform the single LightGBM model.

### Final Hold-Out Performance

On the untouched 2,000-row final hold-out:

- **ROC-AUC:** 0.9713
- **PR-AUC:** 0.8117
- **Brier score:** 0.0125
- **Precision:** 85.45%
- **Recall:** 69.12%
- **F1:** 0.7642

Confusion matrix:

- True negatives: 1,924
- False positives: 8
- False negatives: 21
- True positives: 47

The model therefore detects 47 of the 68 true machine failures while generating only eight false failure alerts.

The hold-out ROC-AUC is almost identical to the cross-validation ROC-AUC, and hold-out PR-AUC is slightly higher than the CV estimate. This supports the conclusion that the validation strategy produced a realistic estimate of model generalisation.

---

## Key Findings

1. **LightGBM provides strong performance for rare machine-failure prediction**, with final hold-out PR-AUC = 0.8117 compared with a failure base rate of approximately 0.034.

2. **Class weighting was not necessary for the final model.** It increased recall at the default threshold but did not improve PR-AUC.

3. **Decision-threshold selection had greater practical value than class weighting.** Lowering the raw threshold from 0.5 to 0.43 improved recall while maintaining high precision.

4. **Torque was the strongest global predictive feature**, with air temperature and rotational speed also contributing substantial predictive information.

5. **Sigmoid calibration slightly improved probability quality** while preserving model ranking.

6. **More complex models did not justify replacing the baseline.** Optuna produced only a small gain and the ensemble approaches did not outperform the single LightGBM model.

---

## Limitations

1. **Synthetic dataset:** AI4I is synthetic and does not represent the full complexity of a real industrial plant.

2. **No time dimension:** The dataset does not contain timestamps, so time-aware validation and temporal performance drift could not be assessed.

3. **No repeated-machine identifier:** Machine-to-machine generalisation could not be evaluated using group-aware validation.

4. **Rare positive class:** Failures account for only approximately 3.4% of observations, so positive-class metrics are based on relatively few failure cases.

5. **No documented business cost matrix:** A monetary expected-cost threshold could not be justified, so maximum F1 was used instead.

6. **Failure-mode variables excluded:** `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` cannot be used as predictors because they directly determine the target.

7. **Interpretability is not causality:** SHAP and permutation importance describe model behaviour and do not prove that the identified features cause machine failure.

---

## Recommendations

For a real-world implementation, the same workflow should be evaluated on genuine time-ordered industrial data containing repeated machine identities.

The maintenance team should define operational costs or inspection-capacity constraints so that the decision threshold can be selected according to a real business objective rather than maximum F1.

Model ranking performance, calibration, class prevalence and feature behaviour should also be monitored over time to identify drift and determine when retraining is required.

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
