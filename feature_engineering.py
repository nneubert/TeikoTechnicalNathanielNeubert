import pandas as pd

# List of immune cell columns
CELL_COLS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

def add_total_and_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total cell count and relative percentages to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing cell counts.
        
    Returns:
        pd.DataFrame: DataFrame with 'total_count' and '<cell>_pct' columns added.
    """
    df = df.copy()
    df["total_count"] = df[CELL_COLS].sum(axis=1)
    
    for col in CELL_COLS:
        pct_col = f"{col}_pct"
        df[pct_col] = df[col] / df["total_count"] * 100
    
    return df


def get_baseline_samples(
    df: pd.DataFrame, 
    condition: str = "melanoma", 
    treatment: str = "miraclib", 
    sample_type: str = "PBMC", 
    timepoint: int = 0
) -> pd.DataFrame:
    """
    Return baseline samples filtered by condition, treatment, and sample type.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with cell counts and metadata.
        condition (str): Condition to filter by (default 'melanoma').
        treatment (str): Treatment to filter by (default 'miraclib').
        sample_type (str): Sample type to filter by (default 'PBMC').
        timepoint (int): Time from treatment start to consider as baseline (default 0).
        
    Returns:
        pd.DataFrame: Filtered baseline samples.
    """
    baseline_df = df[
        (df["condition"] == condition) &
        (df["treatment"] == treatment) &
        (df["sample_type"] == sample_type) &
        (df["time_from_treatment_start"] == timepoint)
    ].copy()
    return baseline_df


def compute_deltas(df: pd.DataFrame, timepoints=[0, 7, 14]) -> pd.DataFrame:
    """
    Compute delta features (change in cell percentages) for each subject.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns:
                           'subject', 'time_from_treatment_start', <cell>_pct
        timepoints (list[int]): List of timepoints to compute deltas.
        
    Returns:
        pd.DataFrame: One row per subject with delta features.
    """
    df_pivot = df.pivot_table(
        index="subject",
        columns="time_from_treatment_start",
        values=[f"{col}_pct" for col in CELL_COLS]
    )
    
    df_delta = pd.DataFrame(index=df_pivot.index)
    
    # Compute deltas for all pairwise combinations from baseline
    baseline_tp = timepoints[0]
    for tp in timepoints[1:]:
        for col in CELL_COLS:
            delta_col = f"{col}_delta_{baseline_tp}_{tp}"
            df_delta[delta_col] = df_pivot[col][tp] - df_pivot[col][baseline_tp]
    
    # Merge back sex and response (take first non-null per subject)
    subject_info = df.groupby("subject")[["sex", "response"]].first()
    df_delta = df_delta.merge(subject_info, left_index=True, right_index=True)
    
    return df_delta