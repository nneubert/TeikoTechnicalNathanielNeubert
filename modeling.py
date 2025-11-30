# modeling.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CELL_PCT_COLS = ["b_cell_pct", "cd8_t_cell_pct", "cd4_t_cell_pct", "nk_cell_pct", "monocyte_pct"]

def random_forest_classification(df: pd.DataFrame, response_col="response", extra_features=None,
                                 test_size=0.3, random_state=42, n_estimators=100):
    """
    Train a Random Forest to predict treatment response.
    
    Parameters:
        df: DataFrame containing features and response.
        response_col: Column name for binary response ("yes"/"no").
        extra_features: List of additional features to include (e.g., ["sex"]).
        test_size: Fraction of data to hold out for testing.
        random_state: Random seed.
        n_estimators: Number of trees in the forest.
        
    Returns:
        dict with trained model, classification report, confusion matrix, and feature importances.
    """
    # Prepare features
    feature_cols = CELL_PCT_COLS.copy()
    if extra_features is not None:
        feature_cols += extra_features
    
    X = df[feature_cols].copy()
    
    # Encode categorical features if needed
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.get_dummies(X[col], drop_first=True)
    
    y = (df[response_col] == "yes").astype(int)
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    feature_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    return {
        "model": rf,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importances": feature_importances
    }
