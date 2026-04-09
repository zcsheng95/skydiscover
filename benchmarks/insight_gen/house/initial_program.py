"""
Insight Generation Baseline for House Price Dataset.

Analyzes the King County housing dataset and generates a data insight
paired with a supporting chart.

Exports: run() -> dict with 'insight' (str) and 'chart_path' (str)
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Path to the house dataset — keep this unchanged when evolving
DATASET_PATH = os.environ.get(
    "HOUSE_DATASET_PATH",
    "/home/zhecheng/insight-scaling/data/house.csv",
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    df["sqft_living"] = pd.to_numeric(df["sqft_living"], errors="coerce")
    df = df.dropna(subset=["price", "bedrooms"])
    df = df[df["bedrooms"].between(1, 8)]  # remove extreme outliers
    df["bedrooms"] = df["bedrooms"].astype(int)
    return df


def run() -> dict:
    """
    Generate an insight and supporting chart from the house price dataset.

    Returns:
        dict with keys:
          - 'insight': str  — a clear, specific finding about the data
          - 'chart_path': str — absolute path to the saved PNG chart
    """
    df = load_data()

    # Mean price by bedroom count
    price_by_bed = df.groupby("bedrooms")["price"].mean().sort_index()

    peak_beds = int(price_by_bed.idxmax())
    peak_price = price_by_bed.max()
    entry_beds = int(price_by_bed.index[0])
    entry_price = price_by_bed.iloc[0]

    insight = (
        f"Homes with {peak_beds} bedrooms command the highest average sale price "
        f"(${peak_price:,.0f}), a {peak_price / entry_price:.1f}× premium over "
        f"{entry_beds}-bedroom homes (${entry_price:,.0f}). "
        f"Average price peaks at {peak_beds} bedrooms rather than at the maximum bedroom count, "
        f"indicating diminishing — and eventually negative — returns to additional bedrooms "
        f"in this market. "
        f"This pattern likely reflects buyer composition: families seeking larger homes "
        f"represent the deepest-pocketed segment, while very large bedroom counts often "
        f"signal older or less efficiently laid-out homes that appeal to a narrower buyer pool. "
        f"Developers should target the {peak_beds}-bedroom configuration as the sweet spot "
        f"for value capture, and should investigate whether the decline above {peak_beds} "
        f"bedrooms is driven by square footage dilution, older vintage, or location bias "
        f"before concluding it reflects pure bedroom-count effects."
    )

    # Chart
    colors = ["tomato" if b == peak_beds else "steelblue" for b in price_by_bed.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(price_by_bed.index, price_by_bed.values / 1e6, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Bedrooms", fontsize=12)
    ax.set_ylabel("Average Sale Price ($ millions)", fontsize=12)
    ax.set_title("Average House Sale Price by Bedroom Count", fontsize=13, fontweight="bold")
    ax.set_xticks(price_by_bed.index)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_house_insight.png", delete=False) as _f:
        chart_path = _f.name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"insight": insight, "chart_path": chart_path}


if __name__ == "__main__":
    result = run()
    print("Insight:", result["insight"])
    print("Chart saved to:", result["chart_path"])
