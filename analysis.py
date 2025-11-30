# analysis.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from feature_engineering import CELL_PCT_COLS, get_baseline_samples, add_total_and_percentages

def ttest_population(df: pd.DataFrame, group_col="response") -> pd.DataFrame:
    results = []
    for cell in CELL_PCT_COLS:
        yes = df[df[group_col]=="yes"][cell]
        no = df[df[group_col]=="no"][cell]
        if len(yes) > 1 and len(no) > 1:
            t, p = ttest_ind(yes, no, equal_var=False)
            diff = yes.mean() - no.mean()
        else:
            t, p, diff = np.nan, np.nan, np.nan
        results.append({"cell_type": cell, "p_value": p, "mean_diff": diff,
                        "significant": p<0.05 if not np.isnan(p) else False,
                        "clinically_significant": abs(diff)>=3 if not np.isnan(diff) else False})
    return pd.DataFrame(results)

def logistic_regression_analysis(df: pd.DataFrame, time_col="time_from_treatment_start") -> dict:
    results = {}
    for t in df[time_col].unique():
        sub = df[df[time_col]==t]
        if len(sub)<2: continue
        X = sub[CELL_PCT_COLS]
        y = (sub["response"]=="yes").astype(int)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[t] = {
            "accuracy": (y_pred==y_test).mean(),
            "confusion_matrix": pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]),
            "feature_importances": pd.Series(clf.coef_[0], index=CELL_PCT_COLS),
            "conclusion": "Does distinguish" if (y_pred==y_test).mean()>=0.8 else "Does NOT distinguish"
        }
    # Overall
    X_all = df[CELL_PCT_COLS]; y_all = (df["response"]=="yes").astype(int)
    X_scaled = StandardScaler().fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results["overall"] = {
        "accuracy": (y_pred==y_test).mean(),
        "confusion_matrix": pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]),
        "feature_importances": pd.Series(clf.coef_[0], index=CELL_PCT_COLS),
        "conclusion": "Does distinguish" if (y_pred==y_test).mean()>=0.8 else "Does NOT distinguish"
    }
    return results

def random_forest_analysis(df: pd.DataFrame, time_col="time_from_treatment_start") -> dict:
    results = {}
    for t in df[time_col].unique():
        sub = df[df[time_col]==t]
        if len(sub)<2: continue
        X = sub[CELL_PCT_COLS]
        y = (sub["response"]=="yes").astype(int)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results[t] = {
            "accuracy": (y_pred==y_test).mean(),
            "confusion_matrix": pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]),
            "feature_importances": pd.Series(rf.feature_importances_, index=CELL_PCT_COLS).sort_values(ascending=False),
            "conclusion": "Does distinguish" if (y_pred==y_test).mean()>=0.8 else "Does NOT distinguish"
        }
    # Overall
    X_all = df[CELL_PCT_COLS]; y_all = (df["response"]=="yes").astype(int)
    X_scaled = StandardScaler().fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["overall"] = {
        "accuracy": (y_pred==y_test).mean(),
        "confusion_matrix": pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]),
        "feature_importances": pd.Series(rf.feature_importances_, index=CELL_PCT_COLS).sort_values(ascending=False),
        "conclusion": "Does distinguish" if (y_pred==y_test).mean()>=0.8 else "Does NOT distinguish"
    }
    return results

def boxplot_alternating(df, timepoints=None):
    if timepoints is None: timepoints=df["time_from_treatment_start"].unique()
    fig, axes = plt.subplots(len(timepoints),1, figsize=(10,4*len(timepoints)), sharey=True)
    width = 0.35; spacing=1.0
    for i,t in enumerate(timepoints):
        ax = axes[i] if len(timepoints)>1 else axes
        positions, data_to_plot = [], []
        for j, cell in enumerate(CELL_PCT_COLS):
            no = df[(df["time_from_treatment_start"]==t)&(df["response"]=="no")][cell]
            yes = df[(df["time_from_treatment_start"]==t)&(df["response"]=="yes")][cell]
            base = j*spacing
            positions.extend([base-width/2, base+width/2])
            data_to_plot.extend([no, yes])
        bplots = ax.boxplot(data_to_plot, positions=positions, widths=width, patch_artist=True, manage_ticks=False)
        for k, box in enumerate(bplots["boxes"]): box.set_facecolor("red" if k%2==0 else "blue")
        ax.set_xticks([j*spacing for j in range(len(CELL_PCT_COLS))])
        ax.set_xticklabels(CELL_PCT_COLS)
        ax.set_ylabel("Percentage")
    plt.tight_layout()
    return fig

def boxplot_separate(df, timepoints=None):
    if timepoints is None: timepoints=df["time_from_treatment_start"].unique()
    fig, axes = plt.subplots(len(timepoints),2, figsize=(14,4*len(timepoints)), sharey=True)
    sns.set(style="whitegrid")
    for i,t in enumerate(timepoints):
        for j, resp in enumerate(["no","yes"]):
            ax = axes[i,j] if len(timepoints)>1 else axes[j]
            sub = df[(df["time_from_treatment_start"]==t)&(df["response"]==resp)]
            df_long = sub.melt(id_vars=["sample"], value_vars=CELL_PCT_COLS, var_name="cell_type", value_name="percentage")
            sns.boxplot(x="cell_type", y="percentage", data=df_long, ax=ax, color="red" if resp=="no" else "blue")
            ax.set_title(f"Time {t} - Response {resp}")
    plt.tight_layout()
    return fig

def full_analysis(df):
    df = add_total_and_percentages(df)
    return {
        "t_tests": ttest_population(df),
        "logistic_regression": logistic_regression_analysis(df),
        "random_forest": random_forest_analysis(df),
        "visualizations": {
            "boxplot_alternating": boxplot_alternating(df),
            "boxplot_separate": boxplot_separate(df)
        }
    }
