import hashlib
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from prediction_service import (
    ArtifactError,
    build_customer_frame,
    is_flagged_for_intervention,
    load_assets,
    out_of_training_range_warnings,
    predict_churn_probability,
    validate_metadata,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class IdentityPreprocessor:
    def __init__(self, feature_names):
        self.feature_names_in_ = np.asarray(feature_names)

    def transform(self, frame):
        return frame.to_numpy()


class ReversedClassClassifier:
    classes_ = np.array([1, 0])

    def predict_proba(self, matrix):
        return np.tile(np.array([[0.73, 0.27]]), (len(matrix), 1))


class FakeKerasModel:
    def predict(self, matrix, verbose=0):
        return np.full((len(matrix), 1), 0.64)


class PredictionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.preprocessor, cls.metadata = load_assets(PROJECT_DIR)

    def default_values(self):
        values = dict(self.metadata["numeric_defaults"])
        values.update(
            {
                name: options[0]
                for name, options in self.metadata["categorical_options"].items()
            }
        )
        return values

    def test_production_artifacts_load_and_feature_order_matches(self):
        self.assertEqual(self.metadata["model_file"], "model.pkl")
        self.assertEqual(self.metadata["model_format"], "pickle")
        self.assertEqual(
            list(self.preprocessor.feature_names_in_),
            self.metadata["feature_names"],
        )
        self.assertTrue(np.array_equal(self.model.classes_, np.array([0, 1])))

    def test_keras_artifact_format_loads_without_touching_production_files(self):
        keras_artifact = PROJECT_DIR / "model.keras"
        if not keras_artifact.is_file():
            self.skipTest("No existing Keras artifact is available for compatibility testing.")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            shutil.copy2(keras_artifact, temporary_root / "model.keras")
            shutil.copy2(PROJECT_DIR / "preprocessor.pkl", temporary_root / "preprocessor.pkl")
            keras_metadata = dict(self.metadata)
            keras_metadata["model_file"] = "model.keras"
            keras_metadata["model_format"] = "keras"
            with (temporary_root / "feature_metadata.pkl").open("wb") as file:
                pickle.dump(keras_metadata, file)

            keras_model, keras_preprocessor, loaded_metadata = load_assets(temporary_root)
            values = dict(loaded_metadata["numeric_defaults"])
            values.update(
                {
                    name: options[0]
                    for name, options in loaded_metadata["categorical_options"].items()
                }
            )
            customer = build_customer_frame(values, loaded_metadata)
            probability = predict_churn_probability(
                keras_model,
                keras_preprocessor,
                loaded_metadata,
                customer,
            )
            self.assertTrue(np.isfinite(probability))
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_default_input_produces_finite_positive_class_probability(self):
        customer = build_customer_frame(self.default_values(), self.metadata)
        probability = predict_churn_probability(
            self.model,
            self.preprocessor,
            self.metadata,
            customer,
        )
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_positive_class_probability_uses_label_one_not_column_order(self):
        metadata = {"feature_names": ["value"], "model_format": "pickle"}
        customer = pd.DataFrame([[2.0]], columns=["value"])
        probability = predict_churn_probability(
            ReversedClassClassifier(),
            IdentityPreprocessor(["value"]),
            metadata,
            customer,
        )
        self.assertEqual(probability, 0.73)

    def test_keras_prediction_path_returns_probability(self):
        metadata = {"feature_names": ["value"], "model_format": "keras"}
        customer = pd.DataFrame([[2.0]], columns=["value"])
        probability = predict_churn_probability(
            FakeKerasModel(),
            IdentityPreprocessor(["value"]),
            metadata,
            customer,
        )
        self.assertEqual(probability, 0.64)

    def test_threshold_boundary_is_inclusive(self):
        self.assertTrue(is_flagged_for_intervention(0.5, 0.5))
        self.assertFalse(is_flagged_for_intervention(0.4999, 0.5))

    def test_unseen_category_is_accepted_by_fitted_encoder(self):
        values = self.default_values()
        values["gender"] = "Synthetic unseen category for test"
        customer = build_customer_frame(values, self.metadata)
        probability = predict_churn_probability(
            self.model,
            self.preprocessor,
            self.metadata,
            customer,
        )
        self.assertTrue(np.isfinite(probability))

    def test_out_of_training_range_warns_without_blocking(self):
        values = self.default_values()
        values["tenure"] = 100
        customer = build_customer_frame(values, self.metadata)
        self.assertTrue(out_of_training_range_warnings(values, self.metadata))
        probability = predict_churn_probability(
            self.model,
            self.preprocessor,
            self.metadata,
            customer,
        )
        self.assertTrue(np.isfinite(probability))

    def test_missing_model_artifact_has_training_instruction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ArtifactError, "python train_model.py"):
                load_assets(Path(temporary_directory))

    def test_incompatible_preprocessor_order_is_rejected(self):
        metadata = {
            "numeric_columns": ["first", "second"],
            "categorical_columns": [],
            "feature_names": ["first", "second"],
            "decision_threshold": 0.5,
            "model_format": "pickle",
            "model_file": "model.pkl",
            "numeric_defaults": {"first": 0.0, "second": 0.0},
            "numeric_minimums": {"first": -1.0, "second": -1.0},
            "numeric_maximums": {"first": 1.0, "second": 1.0},
            "integer_columns": [],
            "categorical_options": {},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            (Path(temporary_directory) / "model.pkl").touch()
            with self.assertRaisesRegex(ArtifactError, "feature order"):
                validate_metadata(
                    metadata,
                    temporary_directory,
                    IdentityPreprocessor(["second", "first"]),
                )

    def test_input_feature_names_and_order_are_enforced(self):
        values = self.default_values()
        customer = build_customer_frame(values, self.metadata)
        self.assertEqual(list(customer.columns), self.metadata["feature_names"])
        with self.assertRaisesRegex(ValueError, "missing features"):
            build_customer_frame({}, self.metadata)

    def test_all_navigation_pages_render(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")
        for page in ("Predict Churn", "Model Information", "About Project", "Overview"):
            navigation.set_value(page).run()
            self.assertFalse(app.exception, f"Navigation failed for {page}")

    def test_streamlit_prediction_form_returns_probability_and_threshold(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")
        navigation.set_value("Predict Churn").run()
        submit = next(button for button in app.button if button.label == "Predict Churn Risk")
        submit.click().run()
        self.assertFalse(app.exception)
        metric_labels = {metric.label for metric in app.metric}
        self.assertIn("Predicted churn probability · class 1 (Yes)", metric_labels)
        self.assertIn("Intervention threshold", metric_labels)
        result_messages = [
            message.value for message in list(app.success) + list(app.warning)
        ]
        self.assertTrue(
            any(
                message in {"Flagged for Intervention", "Below Intervention Threshold"}
                for message in result_messages
            )
        )

    def test_prediction_workflow_does_not_modify_source_artifacts_or_reports(self):
        protected_paths = [
            PROJECT_DIR / name
            for name in (
                "telecom_data.csv",
                "app.py",
                "model.pkl",
                "model.keras",
                "preprocessor.pkl",
                "feature_metadata.pkl",
            )
        ]
        protected_paths.extend(sorted((PROJECT_DIR / "experiments" / "model_comparison").glob("*.csv")))
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in protected_paths
            if path.is_file()
        }
        values = self.default_values()
        customer = build_customer_frame(values, self.metadata)
        predict_churn_probability(self.model, self.preprocessor, self.metadata, customer)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in protected_paths
            if path.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
