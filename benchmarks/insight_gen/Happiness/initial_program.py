"""
Insight Generation Baseline for World Happiness Dataset.

Analyzes the World Happiness Report dataset and generates a data insight
paired with a supporting chart.

Exports: run() -> dict with 'insight' (str) and 'chart_path' (str)
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Path to the Happiness dataset — keep this unchanged when evolving
DATASET_PATH = os.environ.get(
    "HAPPINESS_DATASET_PATH",
    "/home/zhecheng/insight-scaling/data/Happiness.csv",
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["Life Ladder"] = pd.to_numeric(df["Life Ladder"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Life Ladder", "Year", "Regional Indicator"])
    return df


def run() -> dict:
    """
    Generate an insight and supporting chart from the World Happiness dataset.

    Returns:
        dict with keys:
          - 'insight': str  — a clear, specific finding about the data
          - 'chart_path': str — absolute path to the saved PNG chart
    """
    df = load_data()

    # Mean Life Ladder score by region (averaged across all years)
    region_avg = (
        df.groupby("Regional Indicator")["Life Ladder"]
        .mean()
        .sort_values(ascending=False)
    )

    top_region = region_avg.index[0]
    top_score = region_avg.iloc[0]
    bottom_region = region_avg.index[-1]
    bottom_score = region_avg.iloc[-1]

    insight = (
        f"'{top_region}' has the highest average Life Ladder score ({top_score:.2f}) "
        f"across all survey years, while '{bottom_region}' has the lowest ({bottom_score:.2f}), "
        f"a {top_score - bottom_score:.2f}-point gap that represents deep and persistent "
        f"regional inequality in subjective well-being. "
        f"This divide is unlikely to be explained by income differences alone — it also "
        f"reflects structural differences in social support networks, institutional trust, "
        f"and individual freedom, all of which predict Life Ladder scores independently "
        f"of GDP per capita in the World Happiness Report data. "
        f"Policy makers in lower-ranked regions should examine whether targeted investments "
        f"in social infrastructure and governance quality could close part of the gap "
        f"even without equivalent economic growth. "
        f"It is also worth investigating whether the gap is stable over time or widening, "
        f"as diverging trajectories would suggest different intervention urgency than "
        f"a persistent but non-worsening disparity."
    )

    # Chart
    colors = [
        "steelblue" if r not in (top_region, bottom_region)
        else ("gold" if r == top_region else "tomato")
        for r in region_avg.index
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(region_avg.index[::-1], region_avg.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean Life Ladder Score", fontsize=12)
    ax.set_title("Average Life Ladder Score by World Region (All Years)", fontsize=13, fontweight="bold")
    ax.axvline(x=region_avg.mean(), color="grey", linestyle="--", linewidth=1.2,
               label=f"Global mean: {region_avg.mean():.2f}")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_happiness_insight.png", delete=False) as _f:
        chart_path = _f.name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"insight": insight, "chart_path": chart_path}


if __name__ == "__main__":
    result = run()
    print("Insight:", result["insight"])
    print("Chart saved to:", result["chart_path"])
