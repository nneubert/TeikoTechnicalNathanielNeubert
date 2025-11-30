import pandas as pd

# Raw cell counts
CELL_COLS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
# Percentage columns
CELL_PCT_COLS = [f"{c}_pct" for c in CELL_COLS]

def add_total_and_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total counts and percentage columns for each immune cell type.
    Expects raw count columns, not already suffixed with '_pct'.
    """
    # Compute total count
    df["total_count"] = df[CELL_COLS].sum(axis=1)

    # Compute percentage columns
    for col, pct_col in zip(CELL_COLS, CELL_PCT_COLS):
        df[pct_col] = df[col] / df["total_count"] * 100

    return df

def get_baseline_samples(df, condition="melanoma", treatment="miraclib", sample_type="PBMC", timepoint=0):
    """
    Return baseline PBMC samples filtered by condition, treatment, sample type, and timepoint.
    Requires 'condition' column in DataFrame.
    """
    if "condition" not in df.columns:
        raise ValueError("DataFrame must have a 'condition' column for filtering baseline samples.")
    return df[
        (df["condition"] == condition) &
        (df["treatment"] == treatment) &
        (df["sample_type"] == sample_type) &
        (df["time_from_treatment_start"] == timepoint)
    ].copy()

def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a long-format summary table with total count and percentage per population.
    
    Columns:
        sample, total_count, population, count, percentage
    """
    # Melt counts into long format
    summary_df = df.melt(
        id_vars=["sample", "total_count"],
        value_vars=CELL_COLS,
        var_name="population",
        value_name="count"
    )

    # Compute percentage
    summary_df["percentage"] = summary_df["count"] / summary_df["total_count"] * 100

    return summary_df
