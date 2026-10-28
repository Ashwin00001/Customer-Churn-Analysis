from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from train_model import (
    build_candidates,
    build_preprocessor,
    fit_candidate,
    predict_probabilities,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "telecom_data.csv"
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3


def load_experiment_data():
    dataset = pd.read_csv(DATA_PATH)
    labels = dataset["Churn"].astype("string").str.strip().str.casefold()
    target = labels.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if target.isna().any():
        invalid = labels[target.isna()].value_counts(dropna=False).to_dict()
        raise ValueError(f"Invalid or missing Churn labels; no rows were dropped: {invalid}")

    id_columns = [column for column in dataset.columns if column.casefold() == "customerid"]
    features = dataset.drop(columns=["Churn", *id_columns])
    return dataset, features, target.astype(int)


def build_feature_sets(features):
    all_features = features.columns.tolist()
    without_total_charges = [column for column in all_features if column != "TotalCharges"]
    charge_columns = {"MonthlyCharges", "TotalCharges"}
    without_charge_fields = [
        column for column in all_features if column not in charge_columns
    ]
    return {
        "All features": all_features,
        "Without TotalCharges": without_total_charges,
        "Without monthly/total charges": without_charge_fields,
    }


def run_experiment():
    source_hash_before = sha256(DATA_PATH.read_bytes()).hexdigest()
    dataset, features, target = load_experiment_data()
    prevalence = float(target.mean())
    feature_sets = build_feature_sets(features)
    candidates = build_candidates()

    cross_validator = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    splits = list(cross_validator.split(features, target))
    results = defaultdict(list)

    for feature_set_name, columns in feature_sets.items():
        selected_features = features[columns]
        numeric_columns = selected_features.select_dtypes(include="number").columns.tolist()
        categorical_columns = [
            column for column in columns if column not in numeric_columns
        ]

        for candidate_name, candidate_template in candidates.items():
            for split_number, (train_indices, validation_indices) in enumerate(splits):
                train_features = selected_features.iloc[train_indices]
                validation_features = selected_features.iloc[validation_indices]
                train_labels = target.iloc[train_indices].to_numpy()
                validation_labels = target.iloc[validation_indices]

                preprocessor = build_preprocessor(numeric_columns, categorical_columns)
                train_matrix = preprocessor.fit_transform(train_features)
                validation_matrix = preprocessor.transform(validation_features)
                model = fit_candidate(
                    candidate_name,
                    candidate_template,
                    train_matrix,
                    train_labels,
                    random_state=RANDOM_STATE + split_number,
                )
                probabilities = predict_probabilities(
                    candidate_name,
                    model,
                    validation_matrix,
                )
                results[(feature_set_name, candidate_name)].append(
                    {
                        "average_precision": average_precision_score(
                            validation_labels,
                            probabilities,
                        ),
                        "roc_auc": roc_auc_score(validation_labels, probabilities),
                    }
                )

    print(f"Read-only experiment rows: {len(dataset)}; churn prevalence: {prevalence:.3f}")
    print(
        f"Evaluation: {N_REPEATS} repeats of {N_SPLITS}-fold stratified CV "
        f"({len(splits)} validation folds per comparison); no fixed test split or threshold tuning."
    )
    print(
        "Fold standard deviations are descriptive because repeated CV folds overlap; "
        "results are exploratory, not an external performance estimate."
    )
    print(f"Average-precision no-skill baseline (churn prevalence): {prevalence:.3f}\n")
    print("Feature-set/model comparison (mean +/- fold standard deviation):")

    for feature_set_name in feature_sets:
        print(f"\n{feature_set_name}:")
        for candidate_name in candidates:
            fold_results = results[(feature_set_name, candidate_name)]
            average_precision = np.array(
                [result["average_precision"] for result in fold_results]
            )
            roc_auc = np.array([result["roc_auc"] for result in fold_results])
            print(
                f"  {candidate_name}: AP {average_precision.mean():.3f} "
                f"+/- {average_precision.std(ddof=1):.3f}; ROC AUC "
                f"{roc_auc.mean():.3f} +/- {roc_auc.std(ddof=1):.3f}"
            )

    print("\nAverage-precision change versus all features (same model and same folds):")
    for candidate_name in candidates:
        full_scores = np.array(
            [
                result["average_precision"]
                for result in results[("All features", candidate_name)]
            ]
        )
        print(f"  {candidate_name}:")
        for feature_set_name in list(feature_sets)[1:]:
            ablated_scores = np.array(
                [
                    result["average_precision"]
                    for result in results[(feature_set_name, candidate_name)]
                ]
            )
            paired_delta = ablated_scores - full_scores
            print(
                f"    {feature_set_name}: mean paired AP delta "
                f"{paired_delta.mean():+.3f} +/- {paired_delta.std(ddof=1):.3f}"
            )

    source_hash_after = sha256(DATA_PATH.read_bytes()).hexdigest()
    if source_hash_before != source_hash_after:
        raise RuntimeError("The source CSV changed during the read-only experiment.")
    print(f"\nSource CSV unchanged (SHA-256 {source_hash_after}).")


if __name__ == "__main__":
    run_experiment()