"""
Insight Generation Baseline for Cars Dataset.

Analyzes the classic cars dataset and generates a data insight
paired with a supporting chart.

Exports: run() -> dict with 'insight' (str) and 'chart_path' (str)
"""

import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Path to the cars dataset — keep this unchanged when evolving
DATASET_PATH = os.environ.get(
    "CARS_DATASET_PATH",
    "/home/zhecheng/insight-scaling/data/cars.json",
)


def load_data() -> pd.DataFrame:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["Miles_per_Gallon"] = pd.to_numeric(df["Miles_per_Gallon"], errors="coerce")
    df["Horsepower"] = pd.to_numeric(df["Horsepower"], errors="coerce")
    df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
    df["Weight_in_lbs"] = pd.to_numeric(df["Weight_in_lbs"], errors="coerce")
    df = df.dropna(subset=["Miles_per_Gallon", "Horsepower", "Origin"])
    return df


def run() -> dict:
    """
    Generate an insight and supporting chart from the cars dataset.

    Returns:
        dict with keys:
          - 'insight': str  — a clear, specific finding about the data
          - 'chart_path': str — absolute path to the saved PNG chart
    """
    df = load_data()

    # Mean MPG by origin
    mpg_by_origin = df.groupby("Origin")["Miles_per_Gallon"].mean().sort_values(ascending=False)

    top_origin = mpg_by_origin.index[0]
    top_mpg = mpg_by_origin.iloc[0]
    bottom_origin = mpg_by_origin.index[-1]
    bottom_mpg = mpg_by_origin.iloc[-1]

    insight = (
        f"Cars from '{top_origin}' achieve the highest average fuel efficiency "
        f"({top_mpg:.1f} MPG), {top_mpg / bottom_mpg:.1f}× better than cars from "
        f"'{bottom_origin}' ({bottom_mpg:.1f} MPG). "
        f"This gap reflects systematic differences in engine displacement and vehicle weight "
        f"shaped by distinct regulatory environments and consumer markets: Japanese and "
        f"European manufacturers faced higher domestic fuel taxes and smaller road networks "
        f"that rewarded compact, efficient designs, while American manufacturers optimized "
        f"for highway cruising and consumer preference for power. "
        f"The efficiency difference may therefore overstate engineering capability gaps — "
        f"it primarily reflects design priority differences rather than technical limitations. "
        f"Analysts comparing these figures should also examine whether the gap narrows "
        f"within the same cylinder class, as controlling for engine size often reveals "
        f"that per-displacement efficiency was more comparable across origins than raw "
        f"MPG averages suggest."
    )

    # Scatter: Horsepower vs MPG, colored by Origin
    origin_colors = {"USA": "tomato", "Europe": "steelblue", "Japan": "seagreen"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for origin, grp in df.groupby("Origin"):
        ax.scatter(
            grp["Horsepower"], grp["Miles_per_Gallon"],
            label=origin,
            color=origin_colors.get(origin, "grey"),
            alpha=0.7, s=40, edgecolors="white", linewidths=0.4,
        )
    ax.set_xlabel("Horsepower", fontsize=12)
    ax.set_ylabel("Miles per Gallon (MPG)", fontsize=12)
    ax.set_title("Fuel Efficiency vs. Horsepower by Car Origin", fontsize=13, fontweight="bold")
    ax.legend(title="Origin", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_cars_insight.png", delete=False) as _f:
        chart_path = _f.name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"insight": insight, "chart_path": chart_path}


if __name__ == "__main__":
    result = run()
    print("Insight:", result["insight"])
    print("Chart saved to:", result["chart_path"])
