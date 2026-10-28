from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd


class ArtifactError(ValueError):
    """Raised when saved model artifacts are missing or incompatible."""


def validate_metadata(metadata, project_dir, preprocessor=None):
    if not isinstance(metadata, dict):
        raise ArtifactError("Feature metadata must be a dictionary. Run `python train_model.py`.")

    numeric_columns = metadata.get("numeric_columns")
    categorical_columns = metadata.get("categorical_columns")
    if not isinstance(numeric_columns, list) or not isinstance(categorical_columns, list):
        raise ArtifactError("Metadata must include numeric_columns and categorical_columns.")
    if any(not isinstance(name, str) or not name for name in numeric_columns + categorical_columns):
        raise ArtifactError("Feature names must be non-empty strings.")
    if len(set(numeric_columns + categorical_columns)) != len(numeric_columns + categorical_columns):
        raise ArtifactError("Numeric and categorical feature names must be unique and non-overlapping.")

    expected_features = numeric_columns + categorical_columns
    feature_names = metadata.get("feature_names", expected_features)
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != len(set(feature_names))
        or set(feature_names) != set(expected_features)
    ):
        raise ArtifactError("Saved feature_names do not match the numeric and categorical features.")

    threshold = metadata.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ArtifactError("Saved decision_threshold must be numeric and between 0 and 1.")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ArtifactError("Saved decision_threshold must be finite and between 0 and 1.")

    model_format = metadata.get("model_format")
    model_file = metadata.get("model_file")
    if model_format not in {"keras", "pickle"} or not isinstance(model_file, str) or not model_file:
        raise ArtifactError("Metadata must specify model_format ('keras' or 'pickle') and model_file.")

    project_root = Path(project_dir).resolve()
    model_path = (project_root / model_file).resolve()
    if model_path.parent != project_root:
        raise ArtifactError("The selected model path must remain inside the project directory.")
    if not model_path.is_file():
        raise ArtifactError(f"Model artifact `{model_file}` is missing. Run `python train_model.py`.")

    for metadata_key in (
        "numeric_defaults",
        "numeric_minimums",
        "numeric_maximums",
        "integer_columns",
        "categorical_options",
    ):
        if metadata_key not in metadata:
            raise ArtifactError(f"Feature metadata is missing `{metadata_key}`.")
    for column in numeric_columns:
        try:
            default = float(metadata["numeric_defaults"][column])
            minimum = float(metadata["numeric_minimums"][column])
            maximum = float(metadata["numeric_maximums"][column])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"Invalid numeric metadata for `{column}`.") from error
        if not all(math.isfinite(value) for value in (default, minimum, maximum)) or minimum > maximum:
            raise ArtifactError(f"Invalid training range or default for `{column}`.")

    for column in categorical_columns:
        options = metadata["categorical_options"].get(column)
        if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
            raise ArtifactError(f"Invalid categorical options for `{column}`.")

    if preprocessor is not None:
        fitted_feature_names = getattr(preprocessor, "feature_names_in_", None)
        if fitted_feature_names is not None and list(fitted_feature_names) != feature_names:
            raise ArtifactError(
                "Saved preprocessor feature order does not match metadata feature_names. "
                "Run `python train_model.py` to regenerate compatible artifacts."
            )
    normalized_metadata = dict(metadata)
    normalized_metadata["feature_names"] = feature_names
    return normalized_metadata


def load_assets(project_dir):
    project_root = Path(project_dir).resolve()
    metadata_path = project_root / "feature_metadata.pkl"
    preprocessor_path = project_root / "preprocessor.pkl"
    if not metadata_path.is_file() or not preprocessor_path.is_file():
        raise ArtifactError(
            "Model metadata or preprocessing artifact is missing. Run `python train_model.py`."
        )

    try:
        with metadata_path.open("rb") as file:
            metadata = pickle.load(file)
        metadata_had_feature_names = isinstance(metadata, dict) and "feature_names" in metadata
        metadata = validate_metadata(metadata, project_root)
        with preprocessor_path.open("rb") as file:
            preprocessor = pickle.load(file)
    except ArtifactError:
        raise
    except Exception as error:
        raise ArtifactError(
            f"Could not load metadata or preprocessor: {error}. Run `python train_model.py`."
        ) from error

    fitted_feature_names = getattr(preprocessor, "feature_names_in_", None)
    if not metadata_had_feature_names and fitted_feature_names is not None:
        metadata = dict(metadata)
        metadata["feature_names"] = list(fitted_feature_names)
    metadata = validate_metadata(metadata, project_root, preprocessor)
    model_path = project_root / metadata["model_file"]
    try:
        if metadata["model_format"] == "keras":
            from tensorflow.keras.models import load_model

            model = load_model(model_path)
        else:
            with model_path.open("rb") as file:
                model = pickle.load(file)
    except Exception as error:
        raise ArtifactError(
            f"Could not load `{metadata['model_file']}`: {error}. "
            "Check dependencies and run `python train_model.py` if artifacts are incompatible."
        ) from error

    if metadata["model_format"] == "pickle":
        classes = getattr(model, "classes_", None)
        if classes is None or not np.any(np.asarray(classes) == 1):
            raise ArtifactError("The selected scikit-learn model does not expose churn class 1.")
        if not callable(getattr(model, "predict_proba", None)):
            raise ArtifactError("The selected scikit-learn model does not implement predict_proba.")
    return model, preprocessor, metadata


def build_customer_frame(values, metadata):
    feature_names = metadata["feature_names"]
    missing = [name for name in feature_names if name not in values]
    extra = [name for name in values if name not in feature_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing features: {missing}")
        if extra:
            details.append(f"unexpected features: {extra}")
        raise ValueError("Input features do not match the trained model (" + "; ".join(details) + ").")

    numeric_columns = set(metadata["numeric_columns"])
    integer_columns = set(metadata["integer_columns"])
    normalized = {}
    for name in feature_names:
        value = values[name]
        if name in numeric_columns:
            if isinstance(value, bool):
                raise ValueError(f"`{name}` must be numeric, not boolean.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"`{name}` must be numeric.") from error
            if not math.isfinite(numeric_value):
                raise ValueError(f"`{name}` must be a finite number.")
            if numeric_value < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            if name.casefold() == "seniorcitizen" and numeric_value not in (0, 1):
                raise ValueError("`SeniorCitizen` must be 0 or 1.")
            if name in integer_columns:
                if not numeric_value.is_integer():
                    raise ValueError(f"`{name}` must be a whole number.")
                numeric_value = int(numeric_value)
            normalized[name] = numeric_value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty category.")
            normalized[name] = value.strip()

    return pd.DataFrame([normalized], columns=feature_names)


def out_of_training_range_warnings(values, metadata):
    warnings = []
    for column in metadata["numeric_columns"]:
        value = float(values[column])
        minimum = float(metadata["numeric_minimums"][column])
        maximum = float(metadata["numeric_maximums"][column])
        if value < minimum or value > maximum:
            warnings.append(
                f"{column}={value:g} is outside the training range {minimum:g} to {maximum:g}."
            )
    return warnings


def predict_churn_probability(model, preprocessor, metadata, customer):
    feature_names = metadata["feature_names"]
    if list(customer.columns) != feature_names:
        raise ValueError("Customer feature names or order do not match the saved preprocessor.")
    transformed = preprocessor.transform(customer)
    if metadata["model_format"] == "keras":
        raw_probability = np.asarray(model.predict(transformed, verbose=0), dtype=float).reshape(-1)
        if raw_probability.size != 1:
            raise ValueError(f"Keras model returned {raw_probability.size} predictions for one customer.")
        probability = float(raw_probability[0])
    else:
        class_labels = np.asarray(model.classes_)
        positive_indices = np.flatnonzero(class_labels == 1)
        if positive_indices.size != 1:
            raise ValueError("Could not identify exactly one positive churn class (1).")
        class_probabilities = np.asarray(model.predict_proba(transformed), dtype=float)
        if class_probabilities.ndim != 2 or class_probabilities.shape[0] != 1:
            raise ValueError("Classifier returned an unexpected probability array shape.")
        probability = float(class_probabilities[0, positive_indices[0]])

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned a non-finite probability or a value outside [0, 1].")
    return probability


def is_flagged_for_intervention(probability, threshold):
    probability = float(probability)
    threshold = float(threshold)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and between 0 and 1.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1.")
    return probability >= threshold
