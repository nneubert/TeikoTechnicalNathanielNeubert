# main.py

import streamlit as st
from data_loader import load_csv
from feature_engineering import get_baseline_samples, compute_deltas
import analysis as ana
import visualization as viz
import modeling as mdl
import dashboard  # This will launch Streamlit when imported

import argparse
import sys
import pandas as pd

def run_analysis(filepath: str):
    """
    Run the analysis pipeline on the CSV file.
    """
    print(f"Loading data from {filepath}...")
    df = load_csv(filepath)
    print(f"Loaded {len(df)} rows.")

    # Feature engineering
    print("Computing deltas...")
    df_delta = compute_deltas(df)

    # Get baseline PBMC samples
    df_baseline = get_baseline_samples(df)
    print(f"Baseline PBMC samples: {len(df_baseline)}")

    # Statistical analysis
    print("Running t-tests...")
    ttest_results = ana.univariate_ttests(df_baseline)
    print(ttest_results)

    print("Running Mann-Whitney U tests...")
    mw_results = ana.univariate_mannwhitney(df_baseline)
    print(mw_results)

    print("Running MANOVA...")
    manova_results = ana.manova_test(df_baseline)
    print(manova_results)

    print("Running Logistic Regression...")
    logreg_results = ana.logistic_regression_test(df_baseline)
    print("Classification Report:")
    print(logreg_results["classification_report"])
    print("Confusion Matrix:")
    print(logreg_results["confusion_matrix"])
    print("Feature Importances:")
    print(logreg_results["feature_importances"])

    # Random Forest
    print("Running Random Forest...")
    rf_results = mdl.random_forest_classification(df_baseline, extra_features=["sex"])
    print("Classification Report:")
    print(rf_results["classification_report"])
    print("Confusion Matrix:")
    print(rf_results["confusion_matrix"])
    print("Feature Importances:")
    print(rf_results["feature_importances"])

    # Visualization
    print("Generating boxplots...")
    fig_alt = viz.boxplot_alternating(df_baseline)
    fig_sep = viz.boxplot_separate(df_baseline)
    fig_alt.show()
    fig_sep.show()


def launch_dashboard():
    """
    Launch the Streamlit dashboard.
    """
    print("Launching dashboard...")
    # Streamlit requires this to be run via `streamlit run main.py`
    # The actual dashboard code lives in dashboard.py
    # This import will start the dashboard
    import dashboard


def main():
    parser = argparse.ArgumentParser(description="Immune Cell Analysis Pipeline")
    parser.add_argument("--csv", type=str, help="Path to cell-count CSV file")
    parser.add_argument("--mode", type=str, default="dashboard", choices=["dashboard", "analysis"],
                        help="Run in 'dashboard' or 'analysis' mode")
    args = parser.parse_args()

    if args.mode == "analysis":
        if not args.csv:
            print("Error: CSV file path must be provided in analysis mode.")
            sys.exit(1)
        run_analysis(args.csv)
    else:
        # Dashboard mode
        launch_dashboard()


if __name__ == "__main__":
    main()
