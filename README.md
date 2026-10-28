# Telecom Customer Churn Prediction

This project trains an artificial neural network on telecom customer data and provides churn predictions through a Streamlit app.

## Project Files

- `telecom_data.csv`: customer dataset, including the `Churn` target column
- `train_model.py`: preprocessing, training, evaluation, and artifact generation
- `app.py`: local Streamlit prediction interface
- `prediction_service.py`: artifact validation, feature preparation, and model inference
- `tests/test_prediction_workflow.py`: prediction and Streamlit workflow tests
- `requirements.txt`: runtime dependencies
- `ANN_PROJECT.ipynb`: original Colab notebook

## Requirements

The verified workspace uses Python 3.10.22 on Windows, with TensorFlow 2.15.1, scikit-learn 1.5.2, pandas 2.3.3, and Streamlit 1.60.0. Python 3.10 is used for compatibility with the pinned TensorFlow release. The Keras ANN candidate uses Dense(64, ReLU), Dropout(0.3), Dense(32, ReLU), and a sigmoid output, trained with Adam and binary cross-entropy for 15 epochs at batch size 32. Cross-validation can select a different candidate if it performs better.

Create and activate a virtual environment, then install the dependencies:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" venv --python 3.10 .venv310
.venv310\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the Python 3.10 runtime is not installed, install it with `uv python install 3.10` before creating the environment. If `uv` is on PATH, `uv` can be used instead of the full executable path above.

## Train the Model

From the project directory, run:

```powershell
python train_model.py
```

Training audits labels, missing values, numeric parsing, and `TotalCharges` while retaining every source row. The charge comparison against tenure times monthly charges is approximate and diagnostic only; discrepancies do not drop rows or features. Model candidates (majority baseline, logistic regression with and without class weighting, random forest, and class-weighted and oversampled Keras ANN variants) use the same five stratified folds. Preprocessing and oversampling are fitted only on each fold's training portion. The deployable model is selected by mean cross-validation average precision, excluding the majority baseline. The baseline average precision is the churn prevalence.

Thresholds are selected from out-of-fold training probabilities by maximizing F2 subject to a configurable minimum precision of `MIN_PRECISION_LIFT` times training churn prevalence. If no threshold meets that precision constraint, the script reports this and uses the maximum F0.5 threshold as an explicit precision-favoring fallback. Review the printed out-of-fold positive count and precision/recall before using predictions operationally.

- `model.keras` or `model.pkl`: trained model selected by cross-validation
- `preprocessor.pkl`: fitted imputation, scaling, and one-hot encoding pipeline
- `feature_metadata.pkl`: input fields and form defaults/options

The printed held-out metrics are development diagnostics, not an unbiased final estimate: this split has been examined during earlier experiments. Use a new external or future-period dataset for final validation. Accuracy is reported but should not be used alone because the majority class makes up 76.1% of this dataset.

## Launch the App

After training completes, run:

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The four-page dashboard provides an overview and churn distribution, a metadata-driven prediction form, model information, and project notes. It shows the class-1 probability against the saved intervention threshold and warns that the weakly validated model is for educational demonstration only.

The dashboard summarizes the source dataset and selected saved model, accepts the metadata-defined customer features, warns about values outside training ranges, and shows class-1 churn probability beside the configured intervention threshold. The threshold is a decision rule, not a guarantee of churn. The app does not retrain the model when it starts; run `python train_model.py` only when intentionally regenerating artifacts.

## Tests and Limitations

Run the prediction workflow tests with:

```powershell
python -m unittest discover -s tests -v
```

The model has weak validation discrimination and is for educational demonstration only. Do not use it for consequential customer decisions. See [FINAL_EVALUATION_REPORT.md](FINAL_EVALUATION_REPORT.md) for detailed evaluation, data-provenance gaps, and requirements for external or future-period validation.

## Feature Ablation Experiment

Run `python feature_ablation_experiment.py` to compare repeated stratified-CV results for all features, without `TotalCharges`, and without both charge fields. This read-only experiment does not tune a threshold, create a fixed test split, modify the CSV, or save model artifacts. Its repeated-CV results are exploratory; use an external or future-period dataset for final validation.

## Model Comparison Experiment

Run `python model_comparison_experiment.py` for a separate, read-only model comparison. It audits feature/target relationships and compares regularized logistic regression, HistGradientBoosting (with and without class weighting), the majority baseline, random forest, and the existing ANN imbalance variants. All candidates use the same repeated stratified-CV splits. Average precision is primary; ROC AUC, precision, recall, F1, accuracy, and predicted churn counts are also reported. Classification metrics use a fixed 0.5 threshold; no threshold tuning is performed.

The experiment writes fold-level metrics, summaries, paired fold differences, and feature-audit tables under `experiments/model_comparison/`. Average precision's no-skill baseline is the positive-class prevalence; this is distinct from majority-class accuracy. Paired differences and fold standard deviations are descriptive only, not significance tests. Repeated CV uses this dataset for both training and validation folds, so it is exploratory, not an external performance estimate. The script does not change `telecom_data.csv`, production model artifacts, or the Streamlit app.

See [FINAL_EVALUATION_REPORT.md](FINAL_EVALUATION_REPORT.md) for dataset limitations, detailed candidate results, interpretation, and evidence needed before making real-world performance claims.

