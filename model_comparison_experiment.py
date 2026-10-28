from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from train_model import (
    audit_dataset,
    build_candidates,
    build_preprocessor,
    fit_candidate,
    predict_probabilities,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "telecom_data.csv"
OUTPUT_DIR = PROJECT_DIR / "experiments" / "model_comparison"
RANDOM_STATE = 2026
N_SPLITS = 5
N_REPEATS = 3
FIXED_THRESHOLD = 0.5


def load_data():
    dataset = pd.read_csv(DATA_PATH)
    audit_dataset(dataset)

    labels = dataset["Churn"].astype("string").str.strip().str.casefold()
    target = labels.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if target.isna().any():
        invalid = labels[target.isna()].value_counts(dropna=False).to_dict()
        raise ValueError(f"Invalid or missing Churn labels; no rows were dropped: {invalid}")

    id_columns = [column for column in dataset.columns if column.casefold() == "customerid"]
    features = dataset.drop(columns=["Churn", *id_columns]).copy()
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        features[column] = pd.to_numeric(features[column], errors="raise")
    return dataset, features, target.astype(int)


def build_experiment_candidates():
    candidates = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic L2 C=0.01": LogisticRegression(
            penalty="l2", C=0.01, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic L2 C=0.1": LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic L2 C=1": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic L2 C=10": LogisticRegression(
            penalty="l2", C=10.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic L1 C=0.1": LogisticRegression(
            penalty="l1", C=0.1, solver="liblinear", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic L2 C=1 balanced": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting balanced": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    existing_candidates = build_candidates()
    for name in (
        "Random forest",
        "ANN (class-weighted)",
        "ANN (oversampled)",
    ):
        candidates[name] = existing_candidates[name]
    return candidates


def audit_feature_target_relationships(features, target):
    prevalence = float(target.mean())
    print(f"Rows: {len(features)}; churn cases: {int(target.sum())}; prevalence: {prevalence:.3f}")
    print(f"Class counts (0=retained, 1=churn): {target.value_counts().sort_index().to_dict()}")
    print(f"Missing values by feature: {features.isna().sum().to_dict()}")

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    numeric_audit = []
    for column in numeric_columns:
        retained = features.loc[target.eq(0), column]
        churn = features.loc[target.eq(1), column]
        numeric_audit.append(
            {
                "feature": column,
                "type": "numeric",
                "retained_count": int(retained.notna().sum()),
                "churn_count": int(churn.notna().sum()),
                "retained_median": float(retained.median()),
                "churn_median": float(churn.median()),
                "retained_mean": float(retained.mean()),
                "churn_mean": float(churn.mean()),
                "overall_missing": int(features[column].isna().sum()),
            }
        )
    numeric_audit_frame = pd.DataFrame(numeric_audit)

    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    categorical_audit = []
    for column in categorical_columns:
        counts = pd.crosstab(features[column].fillna("<MISSING>"), target)
        for category, row in counts.iterrows():
            total = int(row.sum())
            churn_count = int(row.get(1, 0))
            categorical_audit.append(
                {
                    "feature": column,
                    "category": str(category),
                    "count": total,
                    "churn_count": churn_count,
                    "churn_rate": churn_count / total if total else np.nan,
                }
            )
    categorical_audit_frame = pd.DataFrame(categorical_audit)

    expected_charges = features["tenure"] * features["MonthlyCharges"]
    relative_difference = (features["TotalCharges"] - expected_charges).abs() / expected_charges.where(
        expected_charges.ne(0)
    )
    charge_audit = pd.DataFrame(
        [
            {
                "rows": len(features),
                "missing_total_charges": int(features["TotalCharges"].isna().sum()),
                "total_charges_mean": float(features["TotalCharges"].mean()),
                "total_charges_median": float(features["TotalCharges"].median()),
                "correlation_with_tenure_times_monthly": float(
                    features["TotalCharges"].corr(expected_charges)
                ),
                "rows_over_20pct_relative_difference": int(relative_difference.gt(0.2).sum()),
                "zero_tenure_positive_total_charges": int(
                    ((features["tenure"] == 0) & features["TotalCharges"].gt(0)).sum()
                ),
                "interpretation": "Approximate billing diagnostic only; not a confirmed error or filter.",
            }
        ]
    )
    print("\nNumeric feature/target summaries:")
    print(numeric_audit_frame.to_string(index=False))
    print("\nCategorical feature/target rates (small categories may be noisy):")
    print(categorical_audit_frame.to_string(index=False))
    print("\nTotalCharges relationship audit:")
    print(charge_audit.to_string(index=False))
    return numeric_audit_frame, categorical_audit_frame, charge_audit


def run_experiment():
    source_hash_before = sha256(DATA_PATH.read_bytes()).hexdigest()
    dataset, features, target = load_data()
    numeric_audit, categorical_audit, charge_audit = audit_feature_target_relationships(
        features,
        target,
    )

    candidates = build_experiment_candidates()
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    splits = list(cv.split(features, target))
    fold_records = []
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    for split_index, (train_indices, validation_indices) in enumerate(splits):
        repeat_number = split_index // N_SPLITS + 1
        fold_number = split_index % N_SPLITS + 1
        train_features = features.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        train_labels = target.iloc[train_indices].to_numpy()
        validation_labels = target.iloc[validation_indices]

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        train_matrix = preprocessor.fit_transform(train_features)
        validation_matrix = preprocessor.transform(validation_features)
        for candidate_name, candidate in candidates.items():
            model = fit_candidate(
                candidate_name,
                candidate,
                train_matrix,
                train_labels,
                random_state=RANDOM_STATE + split_index,
            )
            probabilities = predict_probabilities(candidate_name, model, validation_matrix)
            predictions = (probabilities >= FIXED_THRESHOLD).astype(int)
            fold_metrics = {
                "repeat": repeat_number,
                "fold": fold_number,
                "split_id": split_index,
                "model": candidate_name,
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                "validation_churn_prevalence": float(validation_labels.mean()),
                "average_precision": average_precision_score(validation_labels, probabilities),
                "roc_auc": roc_auc_score(validation_labels, probabilities),
                "precision_at_0_5": precision_score(validation_labels, predictions, zero_division=0),
                "recall_at_0_5": recall_score(validation_labels, predictions, zero_division=0),
                "f1_at_0_5": f1_score(validation_labels, predictions, zero_division=0),
                "accuracy_at_0_5": accuracy_score(validation_labels, predictions),
                "predicted_churn_count_at_0_5": int(predictions.sum()),
            }
            fold_records.append(fold_metrics)

    fold_results = pd.DataFrame(fold_records)
    metric_columns = [
        "average_precision",
        "roc_auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "accuracy_at_0_5",
        "predicted_churn_count_at_0_5",
    ]
    summary_records = []
    for model_name, group in fold_results.groupby("model", sort=False):
        record = {
            "model": model_name,
            "fold_count": len(group),
            "mean_validation_churn_prevalence": group["validation_churn_prevalence"].mean(),
        }
        for metric in metric_columns:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_records.append(record)
    summary = pd.DataFrame(summary_records).sort_values(
        "average_precision_mean",
        ascending=False,
    )

    baseline = fold_results.loc[
        fold_results["model"].eq("Majority baseline"),
        ["split_id", "average_precision", "roc_auc"],
    ].rename(
        columns={
            "average_precision": "baseline_average_precision",
            "roc_auc": "baseline_roc_auc",
        }
    )
    reference_logistic = fold_results.loc[
        fold_results["model"].eq("Logistic L2 C=1"),
        ["split_id", "average_precision", "roc_auc"],
    ].rename(
        columns={
            "average_precision": "reference_average_precision",
            "roc_auc": "reference_roc_auc",
        }
    )
    paired = fold_results.merge(baseline, on="split_id").merge(reference_logistic, on="split_id")
    paired = paired.loc[~paired["model"].eq("Majority baseline")].copy()
    paired["ap_delta_vs_majority_baseline"] = (
        paired["average_precision"] - paired["baseline_average_precision"]
    )
    paired["ap_delta_vs_logistic_l2_c1"] = (
        paired["average_precision"] - paired["reference_average_precision"]
    )
    paired_summary = paired.groupby("model", sort=False).agg(
        fold_count=("split_id", "count"),
        mean_ap_delta_vs_majority=("ap_delta_vs_majority_baseline", "mean"),
        std_ap_delta_vs_majority=("ap_delta_vs_majority_baseline", "std"),
        mean_ap_delta_vs_logistic_l2_c1=("ap_delta_vs_logistic_l2_c1", "mean"),
        std_ap_delta_vs_logistic_l2_c1=("ap_delta_vs_logistic_l2_c1", "std"),
    ).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_fold_differences.csv", index=False)
    paired_summary.to_csv(OUTPUT_DIR / "paired_difference_summary.csv", index=False)
    numeric_audit.to_csv(OUTPUT_DIR / "numeric_feature_target_audit.csv", index=False)
    categorical_audit.to_csv(OUTPUT_DIR / "categorical_feature_target_audit.csv", index=False)
    charge_audit.to_csv(OUTPUT_DIR / "total_charges_audit.csv", index=False)

    prevalence = float(target.mean())
    accuracy_baseline = float(target.value_counts(normalize=True).max())
    print(f"\nDataset rows: {len(dataset)}; churn prevalence / no-skill AP baseline: {prevalence:.3f}")
    print(f"Majority-class accuracy baseline (distinct from AP baseline): {accuracy_baseline:.3f}")
    print(
        f"Evaluation: {N_REPEATS} repeats x {N_SPLITS} stratified folds; "
        "identical split IDs for all models; fixed 0.5 threshold for classification metrics."
    )
    print("Thresholds were not tuned. Repeated-fold standard deviations are descriptive, not significance tests.")
    print("\nModel summary (mean +/- fold standard deviation):")
    for _, row in summary.iterrows():
        print(
            f"{row['model']}: AP {row['average_precision_mean']:.3f} +/- "
            f"{row['average_precision_std']:.3f}; AUC {row['roc_auc_mean']:.3f} +/- "
            f"{row['roc_auc_std']:.3f}; precision {row['precision_at_0_5_mean']:.3f}; "
            f"recall {row['recall_at_0_5_mean']:.3f}; F1 {row['f1_at_0_5_mean']:.3f}"
        )
    print("\nPaired AP differences versus logistic L2 C=1:")
    print(paired_summary.to_string(index=False))
    print(
        "\nInterpretation: majority-baseline average precision is tied to positive prevalence, "
        "while majority-class accuracy is a separate quantity. No p-values or claims of "
        "statistical significance are produced."
    )

    source_hash_after = sha256(DATA_PATH.read_bytes()).hexdigest()
    if source_hash_before != source_hash_after:
        raise RuntimeError("The source CSV changed during the experiment.")
    print(f"Source CSV unchanged (SHA-256 {source_hash_after}).")
    print(f"Results written separately under: {OUTPUT_DIR.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    run_experiment()
