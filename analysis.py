# analysis.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CELL_PCT_COLS = ["b_cell_pct", "cd8_t_cell_pct", "cd4_t_cell_pct", "nk_cell_pct", "monocyte_pct"]

# -------------------------------
# Univariate Tests
# -------------------------------

def univariate_ttests(df: pd.DataFrame, response_col="response", alpha=0.05) -> pd.DataFrame:
    """Perform t-tests for each cell type between responders and non-responders."""
    results = []
    for col in CELL_PCT_COLS:
        group_yes = df[df[response_col] == "yes"][col]
        group_no = df[df[response_col] == "no"][col]
        stat, p = ttest_ind(group_yes, group_no, equal_var=False)
        results.append({
            "cell_type": col,
            "t_stat": stat,
            "p_value": p,
            "significant": p < alpha,
            "conclusion": "Evidence of impact" if p < alpha else "No evidence of impact"
        })
    return pd.DataFrame(results)


def univariate_mannwhitney(df: pd.DataFrame, response_col="response", alpha=0.05) -> pd.DataFrame:
    """Perform Mann-Whitney U test for each cell type between responders and non-responders."""
    results = []
    for col in CELL_PCT_COLS:
        group_yes = df[df[response_col] == "yes"][col]
        group_no = df[df[response_col] == "no"][col]
        stat, p = mannwhitneyu(group_yes, group_no, alternative="two-sided")
        results.append({
            "cell_type": col,
            "u_stat": stat,
            "p_value": p,
            "significant": p < alpha,
            "conclusion": "Evidence of impact" if p < alpha else "No evidence of impact"
        })
    return pd.DataFrame(results)

# -------------------------------
# Multivariate Tests
# -------------------------------

def manova_test(df: pd.DataFrame, response_col="response") -> MANOVA:
    """Perform MANOVA on all cell types simultaneously."""
    formula = ' + '.join(CELL_PCT_COLS) + f' ~ {response_col}'
    manova = MANOVA.from_formula(formula, data=df)
    return manova.mv_test()


def logistic_regression_test(df: pd.DataFrame, response_col="response", test_size=0.3, random_state=42):
    """
    Train logistic regression using cell percentages to predict response.
    Returns trained model and evaluation metrics.
    """
    X = df[CELL_PCT_COLS]
    y = (df[response_col] == "yes").astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    feature_importances = pd.Series(clf.coef_[0], index=CELL_PCT_COLS)
    
    return {
        "model": clf,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importances": feature_importances
    }
