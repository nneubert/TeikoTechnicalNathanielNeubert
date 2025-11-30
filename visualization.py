# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

CELL_PCT_COLS = ["b_cell_pct", "cd8_t_cell_pct", "cd4_t_cell_pct", "nk_cell_pct", "monocyte_pct"]

def boxplot_alternating(df, timepoints=[0, 7, 14]):
    """
    Create boxplots for each timepoint, alternating response (no/yes) for each cell type.
    Returns a matplotlib figure for Streamlit.
    """
    fig, axes = plt.subplots(nrows=len(timepoints), ncols=1, figsize=(12, 4*len(timepoints)), sharey=True)
    sns.set(style="whitegrid")

    for i, t in enumerate(timepoints):
        ax = axes[i] if len(timepoints) > 1 else axes
        df_time = df[df["time_from_treatment_start"] == t]
        # Alternate no/yes responses
        order = []
        data = []
        for cell in CELL_PCT_COLS:
            for resp in ["no", "yes"]:
                subset = df_time[df_time["response"] == resp]
                data.append(subset[cell])
                order.append(f"{cell}_{resp}")
        sns.boxplot(data=data, ax=ax)
        ax.set_title(f"Time {t} days")
        ax.set_ylabel("Percentage")
        ax.set_xticklabels(order, rotation=45, ha='right')

    plt.tight_layout()
    return fig


def boxplot_separate(df, timepoints=[0, 7, 14]):
    """
    Create separate boxplots for each response (no/yes) per timepoint.
    Returns a matplotlib figure for Streamlit.
    """
    fig, axes = plt.subplots(nrows=len(timepoints), ncols=2, figsize=(14, 4*len(timepoints)), sharey=True)
    sns.set(style="whitegrid")

    for i, t in enumerate(timepoints):
        for j, resp in enumerate(["no", "yes"]):
            ax = axes[i, j] if len(timepoints) > 1 else axes[j]
            df_sub = df[(df["time_from_treatment_start"] == t) & (df["response"] == resp)]
            df_long = df_sub.melt(id_vars=["sample"], value_vars=CELL_PCT_COLS,
                                  var_name="cell_type", value_name="percentage")
            sns.boxplot(x="cell_type", y="percentage", data=df_long, ax=ax)
            ax.set_title(f"Time {t} days - Response: {resp}")
            ax.set_xlabel("")
            ax.set_ylabel("Percentage" if j == 0 else "")
            if i < len(timepoints) - 1:
                ax.set_xticklabels([])

    plt.tight_layout()
    return fig
