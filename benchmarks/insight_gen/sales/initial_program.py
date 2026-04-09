"""
Insight Generation Baseline for Sales Dataset.

Analyzes the sales sample dataset and generates a data insight
paired with a supporting chart.

Exports: run() -> dict with 'insight' (str) and 'chart_path' (str)
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Path to the sales dataset — keep this unchanged when evolving
DATASET_PATH = os.environ.get(
    "SALES_DATASET_PATH",
    "/home/zhecheng/insight-scaling/data/sales_sample.csv",
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["margin"] = pd.to_numeric(df["margin"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["revenue", "date", "region"])
    return df


def run() -> dict:
    """
    Generate an insight and supporting chart from the sales dataset.

    Returns:
        dict with keys:
          - 'insight': str  — a clear, specific finding about the data
          - 'chart_path': str — absolute path to the saved PNG chart
    """
    df = load_data()

    # Total revenue by region
    region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    top_region = region_rev.index[0]
    top_rev = region_rev.iloc[0]
    total_rev = region_rev.sum()
    top_share = top_rev / total_rev * 100

    insight = (
        f"The '{top_region}' region leads in total revenue (${top_rev:,.0f}), "
        f"accounting for {top_share:.1f}% of overall sales — "
        f"{top_rev / region_rev.iloc[-1]:.1f}× more than the lowest-performing region. "
        f"This concentration suggests that sales capacity, customer density, or product-market "
        f"fit differs substantially across geographies in ways that a uniform go-to-market "
        f"strategy may not address effectively. "
        f"Before attributing this gap to market size alone, analysts should examine whether "
        f"revenue per account and win rates also favor the top region, or whether the gap "
        f"is primarily driven by more sales headcount and activity volume. "
        f"If per-account metrics are similar, redistribution of sales resources toward "
        f"underperforming regions could capture disproportionate revenue growth relative to "
        f"the incremental investment required."
    )

    # Chart
    colors = ["tomato" if r == top_region else "steelblue" for r in region_rev.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(region_rev.index, region_rev.values, color=colors, edgecolor="white")
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Total Revenue", fontsize=12)
    ax.set_title("Total Revenue by Region", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_sales_insight.png", delete=False) as _f:
        chart_path = _f.name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"insight": insight, "chart_path": chart_path}


if __name__ == "__main__":
    result = run()
    print("Insight:", result["insight"])
    print("Chart saved to:", result["chart_path"])
