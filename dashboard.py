import streamlit as st
import pandas as pd
import sqlite3
from data_loader import load_csv_to_db, DB_PATH, CSV_PATH
from analysis import full_analysis
from feature_engineering import add_total_and_percentages, CELL_PCT_COLS
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

# -------------------------------
# Load full dataset from database
# -------------------------------
df_samples = pd.read_sql_query("""
SELECT s.sample, subj.subject, p.project_name AS project,
       subj.treatment, subj.response, subj.sample_type,
       subj.condition, subj.sex,
       s.time_from_treatment_start, s.total_count,
       cc.cell_type, cc.count, cc.percentage,
       subj.project_id, subj.subject_id
FROM samples s
JOIN subjects subj ON s.subject_id = subj.subject_id
JOIN projects p ON subj.project_id = p.project_id
JOIN cell_counts cc ON cc.sample_id = s.sample_id;
""", conn)

# -------------------------------
# Pivot / preprocess wide-format counts
# -------------------------------
df_wide = df_samples.pivot_table(
    index=["sample", "subject", "project", "treatment", "response", "sample_type",
           "time_from_treatment_start", "total_count", "condition"],
    columns="cell_type",
    values="count",
    aggfunc="sum",
    fill_value=0
).reset_index()
df_wide.columns.name = None

# Ensure all expected cell columns exist
CELL_COLS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
for col in CELL_COLS:
    if col not in df_wide.columns:
        raise ValueError

metadata_cols = ["sample", "subject", "project", "treatment", "response",
                 "sample_type", "time_from_treatment_start", "total_count", "condition"]
df_wide = df_wide[metadata_cols + CELL_COLS]

# Compute percentages
df = add_total_and_percentages(df_wide)

# -------------------------------
# Full dataset overview
# -------------------------------
st.subheader("Full Dataset Overview")
st.write(f"Total samples: {len(df)}")
st.dataframe(df.head(20))

# -------------------------------
# Sample-level inspection
# -------------------------------
st.subheader("Inspect Samples by Subject")
selected_sample = st.selectbox("Select a sample to inspect", df["sample"].unique())

if selected_sample:
    sample_info = df[df["sample"] == selected_sample]
    st.markdown("**Selected Sample Info:**")
    st.dataframe(sample_info)

    subject_id = sample_info["subject"].iloc[0]
    related_samples = df[df["subject"] == subject_id]
    st.markdown(f"**Other samples from the same subject ({subject_id}):**")
    st.dataframe(related_samples)

# -------------------------------
# PBMC subset (Melanoma, Miraclib)
# -------------------------------
st.subheader("PBMC, Melanoma, Miraclib, Subset Visualizations")
df_analysis = df[(df["sample_type"] == "PBMC") & (df["treatment"] == "miraclib") & (df["condition"] == "melanoma")]
print(len(df_analysis))
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
