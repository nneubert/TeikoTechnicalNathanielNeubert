import streamlit as st
import pandas as pd
import sqlite3
from data_loader import load_csv_to_db, DB_PATH, CSV_PATH
from analysis import full_analysis, CELL_PCT_COLS
import subset_analysis as sub

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Immune Cell Analysis Dashboard", layout="wide")
st.title("Immune Cell Analysis Dashboard")

# -------------------------------
# Load / initialize database
# -------------------------------
load_csv_to_db(CSV_PATH, DB_PATH)
conn = sqlite3.connect(DB_PATH)


# -------------------------------------------------
# Load every sample with authoritative IDs
# -------------------------------------------------
samples = pd.read_sql_query(
    """
    SELECT
        s.sample_id,
        s.sample,
        s.subject_id,
        s.time_from_treatment_start,
        s.total_count,
        subj.subject AS subject_name,
        p.project_name
    FROM samples s
    JOIN subjects subj ON s.subject_id = subj.subject_id
    JOIN projects p ON subj.project_id = p.project_id
    ORDER BY p.project_name, subj.subject, s.time_from_treatment_start;
    """,
    conn,
)

if len(samples) == 0:
    st.error("No samples found in the database.")
    st.stop()


# -------------------------------------------------
# Select sample from dropdown
# -------------------------------------------------
selected_sample = st.selectbox(
    "Select sample",
    samples["sample"],
    index=0
)

# pull *exact* row
sample_row = samples.loc[samples["sample"] == selected_sample].iloc[0]
sample_id = int(sample_row["sample_id"])
subject_id = int(sample_row["subject_id"])


# -------------------------------------------------
# Load metadata + counts for selected sample
# -------------------------------------------------
df_sample = pd.read_sql_query(
    """
    SELECT
        s.sample_id,
        s.sample,
        subj.subject AS subject,
        p.project_name AS project,
        subj.treatment,
        subj.response,
        subj.sample_type,
        subj.condition,
        subj.sex,
        s.time_from_treatment_start,
        s.total_count,
        cc.cell_type,
        cc.count,
        cc.percentage
    FROM samples s
    JOIN subjects subj ON s.subject_id = subj.subject_id
    JOIN projects p ON subj.project_id = p.project_id
    JOIN cell_counts cc ON s.sample_id = cc.sample_id
    WHERE s.sample_id = ?;
    """,
    conn,
    params=(sample_id,)
)

if df_sample.empty:
    st.error("Sample selected, but no metadata or cell counts found. This indicates a key mismatch.")
    st.write("Debug info:", sample_row)
    st.stop()


# -------------------------------------------------
# Pivot counts + percentages
# -------------------------------------------------
df_pivot = df_sample.pivot_table(
    index=[
        "sample", "subject", "project", "treatment", "response",
        "sample_type", "condition", "sex",
        "time_from_treatment_start", "total_count"
    ],
    columns="cell_type",
    values="count",
    aggfunc="first"
).reset_index()

df_pct = df_sample.pivot_table(
    index=["sample"],
    columns="cell_type",
    values="percentage",
    aggfunc="first"
).reset_index()

df_final = df_pivot.merge(df_pct, on="sample", suffixes=("_count", "_pct"))


# -------------------------------------------------
# Display sample metadata + cell counts
# -------------------------------------------------
st.subheader("Selected Sample Metadata + Cell Counts + Percentages")
st.dataframe(df_final)


# -------------------------------------------------
# Load related samples for the same subject
# -------------------------------------------------
df_related = pd.read_sql_query(
    """
    SELECT
        sample,
        time_from_treatment_start,
        total_count
    FROM samples
    WHERE subject_id = ?
    ORDER BY time_from_treatment_start;
    """,
    conn,
    params=(subject_id,)
)

st.subheader("Other samples from the same subject")
st.dataframe(df_related)


# -------------------------------
# PBMC subset (Melanoma, Miraclib) via SQL
# -------------------------------
df_analysis_raw = pd.read_sql_query(
    """
    SELECT
        s.sample,
        subj.subject,
        p.project_name AS project,
        subj.treatment,
        subj.response,
        subj.sample_type,
        subj.condition,
        subj.sex,
        s.time_from_treatment_start,
        s.total_count,
        cc.cell_type,
        cc.count,
        cc.percentage
    FROM samples s
    JOIN subjects subj ON s.subject_id = subj.subject_id
    JOIN projects p ON subj.project_id = p.project_id
    JOIN cell_counts cc ON s.sample_id = cc.sample_id
    WHERE
        subj.sample_type = 'PBMC' AND
        subj.treatment = 'miraclib' AND
        subj.condition = 'melanoma';
    """,
    conn
)

print("Number of rows before pivot:", len(df_analysis_raw))

# -------------------------------
# Pivot to wide format (counts)
# -------------------------------
CELL_COLS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

df_counts = df_analysis_raw.pivot_table(
    index=[
        "sample", "subject", "project", "treatment", "response",
        "sample_type", "condition", "sex",
        "time_from_treatment_start", "total_count"
    ],
    columns="cell_type",
    values="count",
    aggfunc="first"
).reset_index()

df_pct = df_analysis_raw.pivot_table(
    index=["sample"],
    columns="cell_type",
    values="percentage",
    aggfunc="first"
).reset_index()

df_analysis = df_counts.merge(df_pct, on="sample", suffixes=("_count", "_pct"))

# Ensure all expected columns exist
for cell in CELL_COLS:
    if f"{cell}_count" not in df_analysis.columns:
        df_analysis[f"{cell}_count"] = 0
    if f"{cell}_pct" not in df_analysis.columns:
        df_analysis[f"{cell}_pct"] = 0

print("Number of samples in PBMC / Miraclib / Melanoma subset:", df_analysis["sample"].nunique())

st.subheader("PBMC, Melanoma, Miraclib Subset")
st.dataframe(df_analysis)

results = full_analysis(df_analysis)

st.markdown("**Boxplots - Alternating No/Yes Response**")
st.pyplot(results["visualizations"]["boxplot_alternating"])

st.markdown("**Boxplots - Separate Subplots per Response**")
st.pyplot(results["visualizations"]["boxplot_separate"])

# -------------------------------
# Statistical Analysis & Modeling
# -------------------------------
st.subheader("Statistical Analysis Results")

# T-tests
st.markdown("### T-tests")
st.dataframe(results["t_tests"])

# Logistic Regression
st.markdown("### Logistic Regression")
for tp, res in results["logistic_regression"].items():
    st.markdown(f"**Timepoint: {tp}**")
    st.write("Accuracy:", res["accuracy"])
    st.write("Confusion Matrix")
    st.dataframe(res["confusion_matrix"])
    st.write("Feature Importances")
    st.dataframe(res["feature_importances"])
    st.write("Conclusion:", res["conclusion"])

# Random Forest
st.markdown("### Random Forest")
for tp, res in results["random_forest"].items():
    st.markdown(f"**Timepoint: {tp}**")
    st.write("Accuracy:", res["accuracy"])
    st.write("Confusion Matrix")
    st.dataframe(res["confusion_matrix"])
    st.write("Feature Importances")
    st.dataframe(res["feature_importances"])
    st.write("Conclusion:", res["conclusion"])

# -------------------------------
# Baseline Subset Summary (end of dashboard)
# -------------------------------
st.subheader("Subset Summary: Baseline PBMC Samples (Melanoma, Miraclib, At Treatment Start)")

baseline_summary = sub.summarize_baseline_subset_db(conn, condition="melanoma", treatment="miraclib")

st.markdown("**Samples per Project**")
st.json(baseline_summary.get("samples_per_project", {}))

st.markdown("**Subjects by Response**")
st.json(baseline_summary.get("subjects_response", {}))

st.markdown("**Subjects by Sex**")
st.json(baseline_summary.get("subjects_sex", {}))

# -------------------------------
# Close database
# -------------------------------
conn.close()
