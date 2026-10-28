from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from prediction_service import (
    ArtifactError,
    build_customer_frame,
    is_flagged_for_intervention,
    load_assets,
    out_of_training_range_warnings,
    predict_churn_probability,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "telecom_data.csv"
EVALUATION_REPORT_PATH = PROJECT_DIR / "FINAL_EVALUATION_REPORT.md"
NAVIGATION = ("Overview", "Predict Churn", "Model Information", "About Project")
OTHER_CATEGORY = "Other / enter value"


st.set_page_config(
    page_title="Telecom Churn AI | Analytics",
    page_icon="TC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --canvas: #F7F9FC;
        --navy: #172554;
        --navy-soft: #253765;
        --teal: #0D9488;
        --teal-soft: #CCFBF1;
        --ink: #172033;
        --muted: #64748B;
        --line: #E2E8F0;
        --white: #FFFFFF;
        --amber: #B45309;
        --amber-soft: #FEF3C7;
    }
    [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247, 249, 252, 0.92); }
    [data-testid="stMain"] > div { background: var(--canvas); }
    .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label { color: #E2E8F0; }
    .brand { display: flex; align-items: center; gap: 0.75rem; margin: 0.5rem 0 1.8rem; }
    .brand-mark { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 12px; background: var(--teal); color: white; font-weight: 800; }
    .brand-title { color: white; font-size: 1.05rem; font-weight: 750; line-height: 1.15; }
    .brand-subtitle { color: #AAB7D0; font-size: 0.76rem; margin-top: 0.2rem; }
    .side-label { color: #94A3B8; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; margin: 1rem 0 0.35rem; }
    .sidebar-note { color: #C0CBDD; font-size: 0.78rem; line-height: 1.5; border-top: 1px solid rgba(226,232,240,.18); padding-top: 1rem; margin-top: 2rem; }
    .eyebrow { color: var(--teal); font-size: 0.72rem; font-weight: 800; text-transform: uppercase; margin-bottom: 0.35rem; }
    .page-subtitle { color: var(--muted); font-size: 1rem; margin-top: -0.5rem; }
    .status-pill { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.75rem; border-radius: 999px; background: #E6F7F4; color: #0F766E; font-size: 0.78rem; font-weight: 700; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--teal); }
    [data-testid="stMetric"] { background: var(--white); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(23,37,84,.045); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--navy); font-size: 1.55rem; }
    .section-heading { color: var(--navy); font-size: 0.92rem; font-weight: 750; padding: 0.4rem 0 0.65rem; border-bottom: 1px solid var(--line); margin: 0.25rem 0 0.75rem; }
    .workflow-step { height: 100%; background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(23,37,84,.035); }
    .step-number { color: var(--teal); font-size: 0.72rem; font-weight: 800; text-transform: uppercase; }
    .step-title { color: var(--navy); font-weight: 750; margin: 0.35rem 0; }
    .step-copy { color: var(--muted); font-size: 0.87rem; line-height: 1.5; }
    .probability-track { position: relative; height: 14px; background: #E2E8F0; border-radius: 999px; margin: 1.15rem 0 0.55rem; overflow: visible; }
    .probability-fill { height: 14px; border-radius: 999px; background: var(--teal); }
    .probability-fill.flagged { background: #D97706; }
    .threshold-tick { position: absolute; top: -5px; width: 3px; height: 24px; background: var(--navy); border-radius: 2px; transform: translateX(-50%); }
    .scale-labels { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.74rem; }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button { background: var(--teal); border: 1px solid var(--teal); color: white; border-radius: 10px; min-height: 2.8rem; font-weight: 700; transition: background .15s ease, box-shadow .15s ease; }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover { background: #0F766E; border-color: #0F766E; box-shadow: 0 4px 12px rgba(13,148,136,.18); }
    [data-testid="stAlert"] { border-radius: 12px; }
    @media (max-width: 800px) { .block-container { padding: 1.1rem 1rem 2rem; } .brand { margin-bottom: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_assets():
    return load_assets(PROJECT_DIR)


@st.cache_data
def load_dataset_summary():
    dataset = pd.read_csv(DATA_PATH)
    labels = dataset["Churn"].astype("string").str.strip().str.casefold()
    valid = labels.isin(["yes", "no"])
    churned = int(labels.eq("yes").sum())
    retained = int(labels.eq("no").sum())
    valid_count = int(valid.sum())
    return {
        "total": len(dataset),
        "features": len([column for column in dataset if column not in {"Churn", "CustomerID"}]),
        "churned": churned,
        "retained": retained,
        "churn_rate": churned / valid_count if valid_count else 0.0,
    }


def section_for_feature(feature):
    name = feature.casefold()
    if name in {"gender", "seniorcitizen", "partner", "dependents"}:
        return "Customer Information"
    if name in {"tenure", "phoneservice", "multiplelines", "contract", "paperlessbilling"}:
        return "Account and Contract Details"
    if name in {"internetservice", "onlinesecurity", "onlinebackup", "techsupport", "streamingtv", "streamingmovies"}:
        return "Internet and Additional Services"
    if name in {"monthlycharges", "totalcharges", "paymentmethod"}:
        return "Billing Information"
    return "Other Details"


def render_feature_input(feature, metadata, values):
    if feature in metadata["categorical_columns"]:
        categories = list(metadata["categorical_options"][feature])
        options = categories + ([OTHER_CATEGORY] if OTHER_CATEGORY not in categories else [])
        choice = st.selectbox(feature, options=options, key=f"category_{feature}")
        if choice == OTHER_CATEGORY:
            values[feature] = st.text_input(
                f"Enter {feature}",
                key=f"unseen_{feature}",
                help="The fitted encoder ignores unseen one-hot categories.",
            )
        else:
            values[feature] = choice
        return

    if feature in metadata["numeric_columns"]:
        default = metadata["numeric_defaults"][feature]
        minimum = metadata["numeric_minimums"][feature]
        maximum = metadata["numeric_maximums"][feature]
        is_integer = feature in metadata["integer_columns"]
        values[feature] = st.number_input(
            feature,
            value=int(default) if is_integer else float(default),
            step=1 if is_integer else 0.01,
            key=f"numeric_{feature}",
            help=f"Training range: {minimum:g} to {maximum:g}. Outside-range values are allowed and will be flagged.",
        )
        return

    raise ArtifactError(f"Feature `{feature}` has no input type in saved metadata.")


def render_overview(summary, metadata):
    st.markdown('<div class="eyebrow">Customer retention intelligence</div>', unsafe_allow_html=True)
    header, status = st.columns([4, 1.2], vertical_alignment="center")
    with header:
        st.title("Telecom Customer Churn Analytics")
        st.markdown('<div class="page-subtitle">Analyze customer profiles and estimate churn risk.</div>', unsafe_allow_html=True)
    with status:
        st.markdown('<div class="status-pill"><span class="status-dot"></span>Saved model loaded</div>', unsafe_allow_html=True)

    st.write("")
    kpis = st.columns(4)
    kpis[0].metric("Total Customers", f"{summary['total']:,}")
    kpis[1].metric("Observed Churn Rate", f"{summary['churn_rate']:.1%}")
    kpis[2].metric("Retained Customers", f"{summary['retained']:,}")
    kpis[3].metric("Selected Model", metadata.get("model_name", "Saved classifier"))

    st.write("")
    chart_column, note_column = st.columns([1.5, 1])
    with chart_column:
        st.markdown('<div class="section-heading">Observed churn distribution</div>', unsafe_allow_html=True)
        distribution = pd.DataFrame(
            {
                "Customer group": ["Churned", "Retained"],
                "Customers": [summary["churned"], summary["retained"]],
            }
        )
        chart = (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X("Customers:Q", title="Customers", scale=alt.Scale(domain=[0, summary["total"] * 1.05])),
                y=alt.Y("Customer group:N", title=None, sort=["Churned", "Retained"]),
                color=alt.Color(
                    "Customer group:N",
                    scale=alt.Scale(domain=["Churned", "Retained"], range=["#D97706", "#0D9488"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Customer group:N", title="Group"),
                    alt.Tooltip("Customers:Q", title="Customers", format=","),
                ],
            )
            .properties(height=150)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"Retained: {summary['retained']:,} ({summary['retained'] / summary['total']:.1%}) · "
            f"Churned: {summary['churned']:,} ({summary['churned'] / summary['total']:.1%})"
        )
    with note_column:
        st.markdown('<div class="section-heading">Prediction workflow</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="workflow-step"><div class="step-number">01 · Profile</div>'
            '<div class="step-title">Enter account details</div><div class="step-copy">Use the saved feature definitions and defaults from the training metadata.</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            '<div class="workflow-step"><div class="step-number">02 · Score</div>'
            '<div class="step-title">Apply fitted preprocessing</div><div class="step-copy">The saved transformer encodes categories and scales numeric inputs consistently with training.</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            '<div class="workflow-step"><div class="step-number">03 · Review</div>'
            '<div class="step-title">Compare with threshold</div><div class="step-copy">The class-1 probability is compared with the configured intervention rule.</div></div>',
            unsafe_allow_html=True,
        )

    st.warning(
        "Educational demonstration only. Validation found weak predictive discrimination; "
        "do not use this model for consequential customer decisions."
    )


def render_prediction_page(model, preprocessor, metadata):
    st.markdown('<div class="eyebrow">Scoring workspace</div>', unsafe_allow_html=True)
    st.title("Predict Churn Risk")
    st.markdown('<div class="page-subtitle">Build a customer profile and review the saved model score against its intervention threshold.</div>', unsafe_allow_html=True)
    st.write("")

    values = {}
    warnings = []
    feature_names = metadata["feature_names"]
    section_features = {name: [] for name in (
        "Customer Information",
        "Account and Contract Details",
        "Internet and Additional Services",
        "Billing Information",
        "Other Details",
    )}
    for feature in feature_names:
        section_features[section_for_feature(feature)].append(feature)

    with st.form("customer_profile", clear_on_submit=False):
        section_names = [name for name, features in section_features.items() if features]
        for section_index, section_name in enumerate(section_names):
            st.markdown(f'<div class="section-heading">{section_name}</div>', unsafe_allow_html=True)
            columns = st.columns(2)
            for index, feature in enumerate(section_features[section_name]):
                with columns[index % 2]:
                    render_feature_input(feature, metadata, values)
            if section_index < len(section_names) - 1:
                st.write("")
        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        try:
            customer = build_customer_frame(values, metadata)
            warnings = out_of_training_range_warnings(values, metadata)
            probability = predict_churn_probability(model, preprocessor, metadata, customer)
            threshold = float(metadata["decision_threshold"])
        except (ArtifactError, ValueError) as error:
            st.error(f"Unable to score this profile: {error}")
            return
        except Exception as error:
            st.error(f"Prediction failed safely: {error}")
            return

        for warning in warnings:
            st.warning(warning)
        flagged = is_flagged_for_intervention(probability, threshold)
        st.markdown('<div class="section-heading">Prediction result</div>', unsafe_allow_html=True)
        probability_column, threshold_column = st.columns(2)
        probability_column.metric("Predicted churn probability · class 1 (Yes)", f"{probability:.1%}")
        threshold_column.metric("Intervention threshold", f"{threshold:.1%}")
        fill_class = "flagged" if flagged else ""
        st.markdown(
            f'<div class="probability-track" role="progressbar" aria-label="Predicted churn probability" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{probability * 100:.2f}">'
            f'<div class="probability-fill {fill_class}" style="width:{probability * 100:.2f}%"></div>'
            f'<div class="threshold-tick" title="Threshold {threshold:.1%}" style="left:{threshold * 100:.2f}%"></div></div>'
            f'<div class="scale-labels"><span>0%</span><span>Threshold {threshold:.1%}</span><span>100%</span></div>',
            unsafe_allow_html=True,
        )
        if flagged:
            st.warning("Flagged for Intervention")
            st.write(f"The estimated churn probability ({probability:.1%}) is at or above the configured threshold ({threshold:.1%}).")
        else:
            st.success("Below Intervention Threshold")
            st.write(f"The estimated churn probability ({probability:.1%}) is below the configured threshold ({threshold:.1%}); this does not guarantee retention.")
        st.caption("The threshold is a configurable model decision, not a guarantee that a customer will or will not churn.")


def render_model_information(metadata):
    st.markdown('<div class="eyebrow">Model card</div>', unsafe_allow_html=True)
    st.title("Model Information")
    st.markdown('<div class="page-subtitle">Details read from the saved model metadata and evaluation report.</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    columns[0].metric("Selected model", metadata.get("model_name", "Unknown"))
    columns[1].metric("Artifact format", metadata["model_format"])
    columns[2].metric("Intervention threshold", f"{float(metadata['decision_threshold']):.2%}")
    st.metric("Input feature count", len(metadata["feature_names"]))

    st.markdown('<div class="section-heading">Inference pipeline</div>', unsafe_allow_html=True)
    st.write("Inputs are assembled in saved feature order, transformed by the fitted preprocessor, then scored by the selected artifact. For scikit-learn models, probability is taken from the model's class label 1 (Yes).")
    st.write("The decision label compares that probability with the saved threshold; it is not a calibrated guarantee or a causal explanation.")
    st.error("Limitations: the final evaluation reports weak discrimination and uses exploratory repeated cross-validation. This model is for education and demonstration only.")
    if EVALUATION_REPORT_PATH.is_file():
        st.markdown("[Open the final evaluation report](FINAL_EVALUATION_REPORT.md)")


def render_about(summary):
    st.markdown('<div class="eyebrow">Project notes</div>', unsafe_allow_html=True)
    st.title("About the Project")
    st.markdown('<div class="page-subtitle">Telecom churn classification demonstration built around the existing training and inference pipeline.</div>', unsafe_allow_html=True)

    sections = [
        ("Problem", "Customer churn is a binary classification task: estimate whether a customer is labeled as having churned."),
        ("Objective", "Provide a transparent interface for reviewing a model probability and a configurable intervention threshold."),
        ("Dataset", f"The supplied CSV contains {summary['total']:,} rows and {summary['features']} candidate input features, with an observed churn rate of {summary['churn_rate']:.1%}. The repository does not establish sampling provenance or an outcome horizon."),
        ("Preprocessing", "The application uses the fitted training preprocessor: numeric imputation/scaling and categorical imputation/one-hot encoding. Unknown categories are handled by the fitted encoder."),
        ("Model comparison", "The project compares regularized logistic regression, HistGradientBoosting, random forest, and class-weighted/oversampled ANN variants using repeated stratified cross-validation. Average precision is compared with the churn-prevalence baseline."),
        ("Evaluation limits", "Validation is exploratory, discrimination is weak, and no independent future-period dataset is available. The project makes no claim of production reliability."),
    ]
    for heading, body in sections:
        st.markdown(f'<div class="section-heading">{heading}</div>', unsafe_allow_html=True)
        st.write(body)


def main():
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">TC</div><div>'
        '<div class="brand-title">Telecom Churn AI</div>'
        '<div class="brand-subtitle">Customer analytics demo</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", NAVIGATION, label_visibility="collapsed", key="navigation")
    st.sidebar.markdown(
        '<div class="sidebar-note">Saved model · Educational use only<br>Weak validation performance; not for consequential decisions.</div>',
        unsafe_allow_html=True,
    )

    try:
        dataset_summary = load_dataset_summary()
    except Exception as error:
        st.error(f"Could not read dataset summary: {error}")
        st.stop()

    try:
        model, preprocessor, metadata = cached_assets()
    except ArtifactError as error:
        st.error(f"The trained model could not be loaded: {error}")
        st.info("From the project directory, run `python train_model.py` and restart the app.")
        st.stop()
    except Exception as error:
        st.error(f"The model artifacts could not be read: {error}")
        st.info("Verify the environment and artifacts, then run `python train_model.py` only if regeneration is intended.")
        st.stop()

    if page == "Overview":
        render_overview(dataset_summary, metadata)
    elif page == "Predict Churn":
        render_prediction_page(model, preprocessor, metadata)
    elif page == "Model Information":
        render_model_information(metadata)
    else:
        render_about(dataset_summary)


main()