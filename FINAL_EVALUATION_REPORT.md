# Final Model Evaluation Report

## Executive Summary

The controlled comparison does not establish reliable churn prediction. HistGradientBoosting had the highest mean average precision (AP), 0.287 +/- 0.038, compared with the churn-prevalence baseline of 0.239. Its paired AP difference from Logistic L2 C=1 was only +0.001777 +/- 0.029665. ROC AUC remained close to 0.5 across candidates. These are exploratory repeated-cross-validation results, not evidence of production effectiveness.

## Dataset and Audit

- 1,000 customer rows and 20 CSV columns: 18 candidate features, `CustomerID`, and `Churn`.
- Target encoding is verified in code: `Yes = 1` (churn), `No = 0` (retained).
- Class counts are 239 churn and 761 retained; churn prevalence is 0.239.
- The audit found no missing values, invalid target labels, duplicate IDs, or numeric parsing failures in the audited numeric fields.
- Numeric fields are `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`. The remaining 14 input fields are categorical service, account, and demographic fields.
- Example observed differences include tenure median 38 months for retained customers versus 31 for churn customers; median monthly charges were 67.22 and 71.41 respectively. These summaries are descriptive and do not establish causal or predictive effects.
- `TotalCharges` is numeric and complete. Its correlation with `tenure * MonthlyCharges` is 0.005; 876 rows differ from that approximate calculation by more than 20%, and 12 zero-tenure rows have positive charges. This calculation is only a billing diagnostic: discounts, plan changes, fees, timing, and billing conventions could explain differences. No row or feature was removed on this basis.

### Provenance and Definitions Not Established

The repository does not document the dataset source, sampling frame, collection dates, inclusion/exclusion rules, or whether the rows are representative of an operating customer base. The notebook shows a manual CSV upload, not the CSV's provenance or sampling process. The target definition is only the `Churn` Yes/No field; there is no documented prediction horizon, event date, label-freeze date, censoring rule, or guarantee that every predictor was measured before the churn outcome. Feature names alone cannot establish their operational definitions or timing. Consequently, post-outcome leakage and selection bias cannot be ruled out from this repository alone.

## Evaluation Method

- The experiment used 3 repeats of 5-fold stratified cross-validation: 15 validation folds per candidate.
- The same split IDs were reused for every candidate. Each fold fits imputation, scaling, and one-hot encoding only on that fold's training portion; validation data is transformed by that fitted preprocessor.
- ANN oversampling is applied only to the fold's training portion. Validation folds are not oversampled.
- No fixed test split was created and no threshold was tuned against evaluation-fold labels. Classification metrics below use a fixed descriptive threshold of 0.5.
- Customer identifiers and the `Churn` target are excluded from model features. This controls identifier/target leakage in code, but cannot rule out semantic or temporal leakage without a data dictionary and event timing.
- Fold standard deviations are descriptive. Repeated-CV folds overlap, so they are not independent replicates; no significance test was performed.

## Candidate Results

AP is the primary ranking metric. The no-skill AP reference is the positive-class prevalence (0.239), not majority-class accuracy. Precision, recall, and F1 below are means at the fixed 0.5 threshold.

| Candidate | AP (mean +/- SD) | ROC AUC (mean +/- SD) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.287 +/- 0.038 | 0.526 +/- 0.049 | 0.344 | 0.075 | 0.121 |
| Logistic L2 C=10 | 0.285 +/- 0.031 | 0.534 +/- 0.033 | 0.167 | 0.004 | 0.008 |
| Logistic L2 C=1 | 0.285 +/- 0.030 | 0.533 +/- 0.033 | 0.167 | 0.004 | 0.008 |
| Logistic L2 C=0.1 | 0.284 +/- 0.028 | 0.532 +/- 0.032 | 0.067 | 0.001 | 0.003 |
| Logistic L1 C=0.1 | 0.284 +/- 0.027 | 0.543 +/- 0.041 | 0.000 | 0.000 | 0.000 |
| Logistic L2 C=0.01 | 0.284 +/- 0.027 | 0.531 +/- 0.032 | 0.000 | 0.000 | 0.000 |
| Logistic L2 C=1 balanced | 0.281 +/- 0.030 | 0.530 +/- 0.033 | 0.250 | 0.481 | 0.328 |
| ANN (class-weighted) | 0.278 +/- 0.024 | 0.532 +/- 0.026 | 0.253 | 0.432 | 0.318 |
| HistGradientBoosting balanced | 0.275 +/- 0.031 | 0.519 +/- 0.046 | 0.261 | 0.302 | 0.279 |
| ANN (oversampled) | 0.269 +/- 0.031 | 0.530 +/- 0.042 | 0.261 | 0.397 | 0.313 |
| Random forest | 0.266 +/- 0.035 | 0.509 +/- 0.043 | 0.251 | 0.020 | 0.035 |
| Majority baseline | 0.239 +/- 0.002 | 0.500 +/- 0.000 | 0.000 | 0.000 | 0.000 |

The majority classifier's mean accuracy was 0.761 because 76.1% of rows are retained; at threshold 0.5 it predicts no churners. Accuracy therefore does not measure churn detection quality by itself. AP summarizes precision-recall ranking across thresholds; ROC AUC summarizes ranking between positive and negative cases; precision is the fraction of flagged cases that churn; recall is the fraction of churn cases flagged; F1 is the harmonic mean of precision and recall at a specified threshold. These metrics answer different questions and none alone demonstrates deployment value.

## Paired Comparison and Interpretation

Across the same 15 fold IDs, HistGradientBoosting minus Logistic L2 C=1 had a mean paired AP difference of +0.001777 with fold SD 0.029665. The small mean difference relative to its fold spread does not establish a meaningful improvement. The best candidate's ROC AUC (0.526) remains close to random ranking, and its recall at threshold 0.5 was 0.075. Class weighting raises recall for some models at this threshold but does not improve AP here; rebalancing does not create missing predictive signal.

The numeric summaries, categorical churn-rate table, and `TotalCharges` audit are descriptive. Category rates can be noisy, and group differences may reflect sampling or confounding rather than usable prediction. The `TotalCharges` discrepancy is a signal to investigate the data dictionary and billing process, not proof of corruption.

## Limitations and Evidence Needed

The dataset is small, with 239 positive examples, and its provenance, representativeness, outcome horizon, and feature measurement timing are undocumented. The repository has no independent external or future-period dataset. Repeated cross-validation is useful for internal comparison but cannot establish performance under deployment distribution shift. The previous seed-42 development holdout was examined in prior work and should not be presented as a pristine final estimate.

Before claiming the model is useful in practice:

1. Obtain a data dictionary, source/sampling documentation, feature timestamps, and a precise churn label horizon/censoring policy.
2. Verify that every feature is available at the intended prediction time and that the evaluation cohort matches the intended deployment population.
3. Evaluate on a pre-registered future-period or external holdout that was not used for model/threshold decisions; report uncertainty intervals and subgroup performance.
4. Define intervention capacity and the relative costs of false positives and missed churners. Assess precision/recall, lift, calibration, and business impact at the resulting policy threshold.
5. Monitor post-deployment drift and outcomes before making ongoing effectiveness claims.

No result in this report promises improved performance. Current evidence is weak and does not support describing the system as production-ready for real-world churn targeting.

## Reproduction and Outputs

From the project directory and configured Python environment, run:

```powershell
python model_comparison_experiment.py
```

The run writes fold-level metrics, summaries, paired differences, and feature audit CSVs under `experiments/model_comparison/`. It does not modify the source CSV, production app, or model artifacts.
