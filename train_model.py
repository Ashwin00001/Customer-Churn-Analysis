from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "telecom_data.csv"
KERAS_MODEL_PATH = PROJECT_DIR / "model.keras"
SKLEARN_MODEL_PATH = PROJECT_DIR / "model.pkl"
MODEL_PATH = KERAS_MODEL_PATH
PREPROCESSOR_PATH = PROJECT_DIR / "preprocessor.pkl"
METADATA_PATH = PROJECT_DIR / "feature_metadata.pkl"
MIN_PRECISION_LIFT = 1.25
NUMERIC_AUDIT_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")


def build_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )


def build_candidates():
    return {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Logistic regression (balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "ANN (class-weighted)": None,
        "ANN (oversampled)": None,
    }


def build_ann(input_size, random_state):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)
    model = Sequential(
        [
            Input(shape=(input_size,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_candidate(name, classifier, train_matrix, train_labels, random_state=42):
    train_labels = np.asarray(train_labels, dtype=int)
    if name.startswith("ANN"):
        class_weight = None
        if name == "ANN (oversampled)":
            non_churn_indices = np.flatnonzero(train_labels == 0)
            churn_indices = np.flatnonzero(train_labels == 1)
            oversampled_churn_indices = resample(
                churn_indices,
                replace=True,
                n_samples=len(non_churn_indices),
                random_state=random_state,
            )
            balanced_indices = np.concatenate((non_churn_indices, oversampled_churn_indices))
            train_matrix = train_matrix[balanced_indices]
            train_labels = train_labels[balanced_indices]
        elif name == "ANN (class-weighted)":
            class_counts = np.bincount(train_labels, minlength=2)
            class_weight = {
                class_index: len(train_labels) / (2 * class_count)
                for class_index, class_count in enumerate(class_counts)
            }

        classifier = build_ann(train_matrix.shape[1], random_state)
        fit_options = {
            "epochs": 15,
            "batch_size": 32,
            "verbose": 0,
        }
        if class_weight is not None:
            fit_options["class_weight"] = class_weight
        classifier.fit(train_matrix, train_labels, **fit_options)
        return classifier

    classifier = clone(classifier)
    classifier.fit(train_matrix, train_labels)
    return classifier


def predict_probabilities(name, classifier, feature_matrix):
    if name.startswith("ANN"):
        return classifier.predict(feature_matrix, verbose=0).ravel()
    return classifier.predict_proba(feature_matrix)[:, 1]


def threshold_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    true_positives = int(((predictions == 1) & (np.asarray(labels) == 1)).sum())
    predicted_positive_count = int(predictions.sum())
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f2": fbeta_score(labels, predictions, beta=2, zero_division=0),
        "predicted_positive_count": predicted_positive_count,
        "true_positive_count": true_positives,
        "predicted_positive_rate": float(predictions.mean()),
    }


def select_f2_threshold(labels, probabilities, minimum_precision):
    probabilities = np.asarray(probabilities, dtype=float)
    candidate_thresholds = np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                probabilities,
                np.array([np.nextafter(probabilities.max(), np.inf)]),
            )
        )
    )
    evaluations = [
        (float(threshold), threshold_metrics(labels, probabilities, threshold))
        for threshold in candidate_thresholds
    ]
    eligible = [
        item
        for item in evaluations
        if item[1]["precision"] >= minimum_precision
        and item[1]["predicted_positive_count"] > 0
    ]
    if eligible:
        threshold, metrics = max(
            eligible,
            key=lambda item: (
                item[1]["f2"],
                item[1]["precision"],
                item[1]["recall"],
            ),
        )
        return threshold, metrics, "maximum F2 among thresholds meeting the precision floor"

    print(
        f"No OOF threshold met minimum precision {minimum_precision:.3f}; "
        "using the threshold with highest F0.5 to favor precision over recall."
    )
    threshold, metrics = max(
        evaluations,
        key=lambda item: (
            fbeta_score(
                labels,
                (probabilities >= item[0]).astype(int),
                beta=0.5,
                zero_division=0,
            ),
            item[1]["precision"],
            item[1]["recall"],
        ),
    )
    return threshold, metrics, "fallback: maximum F0.5; configured precision floor was unattainable"


def audit_dataset(dataset):
    print(f"CSV rows retained: {len(dataset)} of {len(dataset)}")
    print(f"Missing values by column: {dataset.isna().sum().to_dict()}")

    labels = dataset["Churn"].astype("string").str.strip().str.casefold()
    print(f"Normalized churn labels: {labels.value_counts(dropna=False).to_dict()}")

    for column in NUMERIC_AUDIT_COLUMNS:
        numeric_values = pd.to_numeric(dataset[column], errors="coerce")
        invalid_values = numeric_values.isna() & dataset[column].notna()
        invalid_count = int(invalid_values.sum())
        print(
            f"{column}: source dtype={dataset[column].dtype}, "
            f"invalid numeric values={invalid_count}"
        )
        if invalid_count:
            examples = dataset.loc[invalid_values, column].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"{column} contains non-numeric values {examples}; "
                "records were not dropped. Correct or explain these source values first."
            )
        dataset[column] = numeric_values

    charges = dataset["TotalCharges"]
    expected_charges = dataset["tenure"] * dataset["MonthlyCharges"]
    relative_difference = (charges - expected_charges).abs() / expected_charges.where(
        expected_charges.ne(0)
    )
    charge_distribution = charges.describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).round(2)
    print(f"TotalCharges distribution:\n{charge_distribution.to_string()}")
    print(
        "TotalCharges correlation with tenure * MonthlyCharges "
        f"(approximate diagnostic only): {charges.corr(expected_charges):.3f}"
    )
    print(
        "Rows with >20% relative difference from tenure * MonthlyCharges "
        f"(review only; not treated as errors): {int(relative_difference.gt(0.2).sum())}"
    )
    print(
        "Zero-tenure rows with positive TotalCharges "
        f"(review billing context): {int(((dataset['tenure'] == 0) & charges.gt(0)).sum())}"
    )


def main():
    dataset = pd.read_csv(DATA_PATH)
    audit_dataset(dataset)

    labels = dataset["Churn"].astype(str).str.strip().str.casefold()
    target = labels.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if target.isna().any():
        invalid_labels = labels[target.isna()].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Unexpected or missing Churn labels {invalid_labels}; "
            "records were not dropped. Correct or explain these labels first."
        )
    target = target.astype(int)

    id_columns = [column for column in dataset.columns if column.casefold() == "customerid"]
    features = dataset.drop(columns=["Churn", *id_columns])
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    prevalence = float(target_train.mean())
    minimum_precision = min(prevalence * MIN_PRECISION_LIFT, 1.0)
    print(
        f"Dataset: {len(dataset)} rows; churn prevalence={float(target.mean()):.3f}; "
        f"train={len(features_train)}, held-out development test={len(features_test)}"
    )
    print("Target encoding: Yes=1 (churn), No=0 (retained).")
    print(
        "Accuracy alone is insufficient for this imbalanced target: an always-retained "
        "classifier can have high accuracy while finding no churners."
    )
    print(
        f"Threshold precision floor: {minimum_precision:.3f} "
        f"({MIN_PRECISION_LIFT:.2f}x training churn prevalence)."
    )

    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cross_validator.split(features_train, target_train))
    candidate_results = {}
    candidates = build_candidates()
    for candidate_name, candidate in candidates.items():
        fold_average_precision = []
        fold_roc_auc = []
        probabilities = np.empty(len(target_train), dtype=float)

        for fold_index, (fit_indices, validation_indices) in enumerate(cv_splits):
            fold_features = features_train.iloc[fit_indices]
            fold_labels = target_train.iloc[fit_indices].to_numpy()
            validation_features = features_train.iloc[validation_indices]
            validation_labels = target_train.iloc[validation_indices]

            preprocessor = build_preprocessor(numeric_columns, categorical_columns)
            fit_matrix = preprocessor.fit_transform(fold_features)
            validation_matrix = preprocessor.transform(validation_features)
            classifier = fit_candidate(
                candidate_name,
                candidate,
                fit_matrix,
                fold_labels,
                random_state=42 + fold_index,
            )
            fold_probabilities = predict_probabilities(
                candidate_name,
                classifier,
                validation_matrix,
            )
            probabilities[validation_indices] = fold_probabilities
            fold_average_precision.append(
                average_precision_score(validation_labels, fold_probabilities)
            )
            fold_roc_auc.append(roc_auc_score(validation_labels, fold_probabilities))

        threshold, oof_metrics, selection_reason = select_f2_threshold(
            target_train,
            probabilities,
            minimum_precision,
        )
        candidate_results[candidate_name] = {
            "average_precision_mean": float(np.mean(fold_average_precision)),
            "average_precision_std": float(np.std(fold_average_precision, ddof=1)),
            "roc_auc_mean": float(np.mean(fold_roc_auc)),
            "roc_auc_std": float(np.std(fold_roc_auc, ddof=1)),
            "oof_probabilities": probabilities,
            "threshold": threshold,
            "oof_metrics": oof_metrics,
            "threshold_selection_reason": selection_reason,
        }

    print("\n5-fold stratified CV (mean +/- sample standard deviation):")
    for candidate_name, result in candidate_results.items():
        print(
            f"{candidate_name}: AP {result['average_precision_mean']:.3f} "
            f"+/- {result['average_precision_std']:.3f}; ROC AUC "
            f"{result['roc_auc_mean']:.3f} +/- {result['roc_auc_std']:.3f}"
        )
        print(
            f"  OOF threshold={result['threshold']:.4f}; "
            f"precision={result['oof_metrics']['precision']:.3f}; "
            f"recall={result['oof_metrics']['recall']:.3f}; "
            f"F2={result['oof_metrics']['f2']:.3f}; "
            f"predicted churners={result['oof_metrics']['predicted_positive_count']}; "
            f"true positives={result['oof_metrics']['true_positive_count']}; "
            f"predicted-positive rate={result['oof_metrics']['predicted_positive_rate']:.3f}; "
            f"selection={result['threshold_selection_reason']}"
        )

    deployable_candidates = [
        name for name in candidates if name != "Majority baseline"
    ]
    selected_name = max(
        deployable_candidates,
        key=lambda name: candidate_results[name]["average_precision_mean"],
    )
    selected_result = candidate_results[selected_name]
    selected_threshold = selected_result["threshold"]
    selected_oof_metrics = selected_result["oof_metrics"]
    print(
        f"\nSelected model: {selected_name}, based on highest mean CV average precision "
        f"({selected_result['average_precision_mean']:.3f}) among non-baseline candidates."
    )
    print(
        f"Selected threshold: {selected_threshold:.4f}; "
        f"criterion: {selected_result['threshold_selection_reason']}"
    )

    final_preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    train_matrix = final_preprocessor.fit_transform(features_train)
    test_matrix = final_preprocessor.transform(features_test)
    model = fit_candidate(
        selected_name,
        candidates[selected_name],
        train_matrix,
        target_train.to_numpy(),
        random_state=42,
    )
    test_probabilities = predict_probabilities(selected_name, model, test_matrix)
    test_predictions = (test_probabilities >= selected_threshold).astype(int)
    test_metrics = {
        "accuracy": accuracy_score(target_test, test_predictions),
        "balanced_accuracy": balanced_accuracy_score(target_test, test_predictions),
        "precision": precision_score(target_test, test_predictions, zero_division=0),
        "recall": recall_score(target_test, test_predictions, zero_division=0),
        "f2": fbeta_score(target_test, test_predictions, beta=2, zero_division=0),
        "average_precision": average_precision_score(target_test, test_probabilities),
        "roc_auc": roc_auc_score(target_test, test_probabilities),
    }
    test_confusion = confusion_matrix(target_test, test_predictions, labels=[0, 1])
    test_positive_count = int(test_predictions.sum())
    test_true_positives = int(
        ((test_predictions == 1) & (target_test.to_numpy() == 1)).sum()
    )
    probability_distribution = pd.Series(test_probabilities).describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).round(3)

    is_keras_model = selected_name.startswith("ANN")
    model_path = KERAS_MODEL_PATH if is_keras_model else SKLEARN_MODEL_PATH
    model_format = "keras" if is_keras_model else "pickle"
    if is_keras_model:
        model.save(str(model_path))
    else:
        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)

    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names": features.columns.tolist(),
        "numeric_defaults": features_train[numeric_columns].median().to_dict(),
        "numeric_minimums": features_train[numeric_columns].min().to_dict(),
        "numeric_maximums": features_train[numeric_columns].max().to_dict(),
        "integer_columns": [
            column
            for column in numeric_columns
            if pd.api.types.is_integer_dtype(features_train[column].dtype)
        ],
        "categorical_options": {
            column: sorted(features_train[column].dropna().astype(str).unique().tolist())
            for column in categorical_columns
        },
        "decision_threshold": selected_threshold,
        "model_name": selected_name,
        "model_file": model_path.name,
        "model_format": model_format,
    }
    with PREPROCESSOR_PATH.open("wb") as preprocessor_file:
        pickle.dump(final_preprocessor, preprocessor_file)
    with METADATA_PATH.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    warnings = []
    if selected_result["average_precision_mean"] <= prevalence:
        warnings.append("Selected CV average precision does not exceed churn prevalence.")
    if selected_result["roc_auc_mean"] <= 0.55:
        warnings.append("Selected CV ROC AUC indicates weak discrimination.")
    if selected_result["threshold_selection_reason"].startswith("fallback"):
        warnings.append("No OOF threshold met the configured precision constraint.")
    if selected_oof_metrics["predicted_positive_count"] in (0, len(target_train)):
        warnings.append("Selected OOF threshold predicts zero or all training customers as churn.")
    if test_positive_count == 0 or test_true_positives == 0:
        warnings.append("Development test diagnostics contain no correctly detected churners.")
    if test_positive_count == len(target_test):
        warnings.append("Development test threshold flags every customer as churn.")

    print("\nSelected-model held-out development diagnostics (not an unbiased final estimate):")
    print(
        f"Rows={len(target_test)}; churn prevalence={target_test.mean():.3f}; "
        f"majority accuracy={target_test.value_counts(normalize=True).max():.3f}; "
        f"predicted churners={test_positive_count}; true positives={test_true_positives}"
    )
    print("Test metrics: " + "; ".join(
        f"{metric}={value:.3f}" for metric, value in test_metrics.items()
    ))
    print(
        "Confusion matrix (actual rows [retained, churn], "
        "predicted columns [retained, churn]):"
    )
    print(test_confusion)
    print(f"Test probability distribution:\n{probability_distribution.to_string()}")
    if warnings:
        print("\nMODEL / TARGETING WARNINGS:")
        for warning in dict.fromkeys(warnings):
            print(f"- {warning}")
        print("Do not treat this model as production-ready without stronger validation.")
    else:
        print("\nNo configured warning was triggered; external validation is still required.")
    print(f"Saved selected model ({selected_name}) and matching preprocessing artifacts.")


if __name__ == "__main__":
    main()