# dashboard.py

import streamlit as st
import pandas as pd
from data_loader import load_csv
from feature_engineering import get_baseline_samples, compute_deltas
import analysis as ana
import visualization as viz
import modeling as mdl

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Immune Cell Analysis Dashboard", layout="wide")
st.title("Immune Cell Analysis Dashboard")

# -------------------------------
# Load Data
# -------------------------------
uploaded_file = st.file_uploader("Upload cell-count CSV", type=["csv"])
if uploaded_file:
    df = load_csv(uploaded_file)
    st.success("Data loaded successfully!")

    # -------------------------------
    # Sidebar: Filters and Options
    # -------------------------------
    st.sidebar.header("Filters / Options")

    conditions = df["condition"].unique().tolist()
    selected_condition = st.sidebar.selectbox("Select condition", conditions)

    timepoints = sorted(df["time_from_treatment_start"].unique())
    selected_timepoint = st.sidebar.selectbox("Select timepoint", timepoints)

    sample_types = df["sample_type"].unique().tolist()
    selected_sample_type = st.sidebar.selectbox("Select sample type", sample_types)

    responses = df["response"].dropna().unique().tolist()
    selected_response = st.sidebar.selectbox("Select response", responses + [None])

    st.sidebar.header("Statistical Test")
    test_option = st.sidebar.selectbox(
        "Choose test",
        ["t-test", "Mann-Whitney U", "MANOVA", "Logistic Regression"]
    )

    st.sidebar.header("Boxplot Layout")
    layout_option = st.sidebar.radio("Choose layout", ["Alternating No/Yes", "Separate Subfigures"])

    run_rf = st.sidebar.checkbox("Run Random Forest", value=False)

    # -------------------------------
    # Filter Data
    # -------------------------------
    df_filtered = df[
        (df["condition"] == selected_condition) &
        (df["time_from_treatment_start"] == selected_timepoint) &
        (df["sample_type"] == selected_sample_type)
    ]
    if selected_response:
        df_filtered = df_filtered[df_filtered["response"] == selected_response]

    if len(df_filtered) == 0:
        st.warning("No samples available after filtering. Adjust filters.")
        st.stop()

    st.subheader("Filtered Data Overview")
    st.write(f"Showing {len(df_filtered)} samples after filtering.")
    st.dataframe(df_filtered.head(10))

    # -------------------------------
    # Sample Inspection
    # -------------------------------
    st.subheader("Sample Browser")

    sample_id = st.selectbox("Select a sample to inspect", df_filtered["sample"].tolist())
    sample_row = df_filtered[df_filtered["sample"] == sample_id].iloc[0]

    st.markdown("**Sample Details:**")
    sample_info_cols = [
        "subject", "condition", "sex", "treatment", "response",
        "time_from_treatment_start", "total_count"
    ] + [c for c in df_filtered.columns if c.endswith("_pct")]
    sample_info_cols = [c for c in sample_info_cols if c in df_filtered.columns]
    st.write(sample_row[sample_info_cols])

    st.markdown("**Related Samples (same subject):**")
    related_samples = df[df["subject"] == sample_row["subject"]]
    st.write(related_samples[["sample", "time_from_treatment_start", "response"]])

    # -------------------------------
    # Statistical Analysis
    # -------------------------------
    st.subheader("Statistical Analysis")
    if test_option == "t-test":
        st.write("Running t-tests...")
        ttest_results = ana.univariate_ttests(df_filtered)
        st.dataframe(ttest_results)
    elif test_option == "Mann-Whitney U":
        st.write("Running Mann-Whitney U tests...")
        mw_results = ana.univariate_mannwhitney(df_filtered)
        st.dataframe(mw_results)
    elif test_option == "MANOVA":
        st.write("Running MANOVA...")
        manova_results = ana.manova_test(df_filtered)
        st.text(str(manova_results))
    elif test_option == "Logistic Regression":
        st.write("Running Logistic Regression...")
        logreg_results = ana.logistic_regression_test(df_filtered)
        st.write("Classification Report")
        st.json(logreg_results["classification_report"])
        st.write("Confusion Matrix")
        st.write(logreg_results["confusion_matrix"])
        st.write("Feature Importances")
        st.write(logreg_results["feature_importances"])

    # -------------------------------
    # Boxplots
    # -------------------------------
    st.subheader("Boxplots")
    if layout_option == "Alternating No/Yes":
        fig = viz.boxplot_alternating(df_filtered)
        st.pyplot(fig)
    else:
        fig = viz.boxplot_separate(df_filtered)
        st.pyplot(fig)

    # -------------------------------
    # Random Forest
    # -------------------------------
    if run_rf:
        st.subheader("Random Forest Prediction")
        rf_results = mdl.random_forest_classification(df_filtered, extra_features=["sex"])
        st.write("Classification Report")
        st.json(rf_results["classification_report"])
        st.write("Confusion Matrix")
        st.write(rf_results["confusion_matrix"])
        st.write("Feature Importances")
        st.write(rf_results["feature_importances"])
