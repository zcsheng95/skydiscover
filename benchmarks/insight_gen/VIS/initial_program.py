"""
Insight Generation Baseline for VIS Dataset.

Analyzes the IEEE VIS papers dataset and generates a data insight
paired with a supporting chart.

Exports: run() -> dict with 'insight' (str) and 'chart_path' (str)
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Path to the VIS dataset — keep this unchanged when evolving
DATASET_PATH = os.environ.get(
    "VIS_DATASET_PATH",
    "/home/zhecheng/insight-scaling/data/VIS.csv",
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df["CitationCount_CrossRef"] = pd.to_numeric(df["CitationCount_CrossRef"], errors="coerce").fillna(0)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df


def run() -> dict:
    """
    Generate an insight and supporting chart from the VIS dataset.

    Returns:
        dict with keys:
          - 'insight': str  — a clear, specific finding about the data
          - 'chart_path': str — absolute path to the saved PNG chart
    """
    df = load_data()

    # Average CrossRef citations per year
    # NOTE: .index is a property, not a method — never write .index()
    yearly = df.groupby("Year")["CitationCount_CrossRef"].mean().sort_index()
    years = yearly.index.tolist()          # correct: .index (property) then .tolist()
    avg_cites = yearly.values.tolist()

    peak_year = yearly.idxmax()
    peak_avg = yearly.max()
    recent_avg = yearly.iloc[-3:].mean()

    insight = (
        f"Papers published in {peak_year} have the highest average CrossRef citation count "
        f"({peak_avg:.1f}), more than {peak_avg / max(recent_avg, 1):.1f}× the average for the three "
        f"most recent years ({recent_avg:.1f}). This citation lag effect means older IEEE VIS work "
        f"disproportionately defines the field's citation landscape, even as the community grows "
        f"and research directions shift. The pattern likely reflects the typical 5–10 year window "
        f"for a paper to accumulate peak citations, meaning any snapshot comparison "
        f"systematically undervalues recent contributions. Researchers benchmarking impact across "
        f"cohorts should discount raw citation counts for papers published in the last 3–5 years "
        f"and instead compare within same-vintage groups. It also raises the question of whether "
        f"the field's most-cited older work still reflects current methodological priorities, or "
        f"whether citation dominance has become decoupled from contemporary relevance."
    )

    # Chart
    peak_idx = years.index(peak_year)
    colors = ["tomato" if y == peak_year else "steelblue" for y in years]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(years, avg_cites, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=peak_year, color="tomato", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Peak: {peak_year} ({peak_avg:.0f} avg citations)")
    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Avg CrossRef Citations", fontsize=12)
    ax.set_title("Average CrossRef Citation Count of IEEE VIS Papers by Year", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_vis_insight.png", delete=False) as _f:
        chart_path = _f.name
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"insight": insight, "chart_path": chart_path}


if __name__ == "__main__":
    result = run()
    print("Insight:", result["insight"])
    print("Chart saved to:", result["chart_path"])
