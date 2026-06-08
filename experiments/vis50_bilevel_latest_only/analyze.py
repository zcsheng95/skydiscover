#!/usr/bin/env python3
"""Analyze a VIS bilevel evaluator-prompt run.

This script regenerates:
  - evaluator_intervals.png
  - evaluator_interval_summary.md
  - evaluator_prompt_progression_local_changes.md

Usage:
  UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache \
    uv run python scripts/reproduce/vis50_analyze.py results_experiment_vis50/run
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _find_iteration_stats(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("adaevolve_iteration_stats_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No adaevolve_iteration_stats_*.jsonl found in {run_dir}")
    if len(candidates) > 1:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return candidates[0]


def _load_records(stats_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in _read_jsonl(stats_path):
        metrics = obj.get("iteration_result", {}).get("child_program", {}).get("metrics", {}) or {}
        score = metrics.get("combined_score")
        if not isinstance(score, (int, float)):
            continue

        version = metrics.get("evaluator_prompt_version", 0)
        old_score = metrics.get("old_combined_score")
        if not isinstance(old_score, (int, float)):
            old_score = score if version == 0 else None

        records.append(
            {
                "iteration": obj["iteration"],
                "score": float(score),
                "old_score": old_score,
                "latest_score": metrics.get("latest_combined_score"),
                "version": int(version),
                "score_mode": metrics.get("evaluator_prompt_score_mode"),
                "error": metrics.get("error"),
                "insight": metrics.get("insight_text", ""),
            }
        )
    return records


def _version_intervals(history: list[dict[str, Any]], max_iter: int) -> list[tuple[int, int, int]]:
    starts = {0: 1}
    for item in history[1:]:
        starts[int(item["version"])] = int(item["iteration"]) + 1

    intervals: list[tuple[int, int, int]] = []
    versions = sorted(starts)
    for idx, version in enumerate(versions):
        start = starts[version]
        end = starts[versions[idx + 1]] - 1 if idx + 1 < len(versions) else max_iter
        intervals.append((version, start, end))
    return intervals


def _write_prompt_progression(prompt_dir: Path, out_path: Path) -> None:
    versions = sorted(int(path.stem.split("_v")[-1]) for path in prompt_dir.glob("evaluator_prompt_v*.txt"))
    lines = ["# Evaluator Prompt Progression", "", "Only local changes between installed evaluator prompts are shown.", ""]

    for prev, curr in zip(versions, versions[1:]):
        a = (prompt_dir / f"evaluator_prompt_v{prev}.txt").read_text().splitlines()
        b = (prompt_dir / f"evaluator_prompt_v{curr}.txt").read_text().splitlines()
        matcher = difflib.SequenceMatcher(a=a, b=b)
        lines.append(f"## p{prev} -> p{curr}")
        changed = False
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed = True
            lines.extend(["", f"Lines p{prev}:{i1 + 1}-{i2} -> p{curr}:{j1 + 1}-{j2}", "```diff"])
            if tag in {"replace", "delete"}:
                lines.extend(f"-{line}" for line in a[i1:i2])
            if tag in {"replace", "insert"}:
                lines.extend(f"+{line}" for line in b[j1:j2])
            lines.append("```")
        if not changed:
            lines.append("No changes.")
        lines.append("")

    out_path.write_text("\n".join(lines))


def _prompt_diff_summaries(prompt_dir: Path, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for prev, curr in zip(history, history[1:]):
        a = (prompt_dir / f"evaluator_prompt_v{prev['version']}.txt").read_text().splitlines()
        b = (prompt_dir / f"evaluator_prompt_v{curr['version']}.txt").read_text().splitlines()
        diff = list(difflib.unified_diff(a, b, lineterm=""))
        added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
        removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
        summaries.append(
            {
                "from": prev["version"],
                "to": curr["version"],
                "iteration": curr.get("iteration"),
                "added": len(added),
                "removed": len(removed),
                "preview": "; ".join(added[:3])[:500],
            }
        )
    return summaries


def _plot_intervals(
    run_dir: Path,
    records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    intervals: list[tuple[int, int, int]],
) -> Path:
    import matplotlib.pyplot as plt

    out_path = run_dir / "evaluator_intervals.png"
    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b", 3: "#b279a2", 4: "#e45756"}
    max_iter = max(record["iteration"] for record in records)

    fig, ax = plt.subplots(figsize=(12, 5.6))

    for version, start, end in intervals:
        ax.axvspan(start - 0.5, end + 0.5, color=colors.get(version, "#999999"), alpha=0.08)
        ax.text(
            (start + end) / 2,
            0.04,
            f"p{version}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=colors.get(version, "#555555"),
            transform=ax.get_xaxis_transform(),
        )

    for version in sorted({record["version"] for record in records}):
        segment = [record for record in records if record["version"] == version]
        ax.plot(
            [record["iteration"] for record in segment],
            [record["score"] for record in segment],
            marker="o",
            linewidth=1.8,
            label=f"active p{version}",
            color=colors.get(version, "#555555"),
        )

    best_so_far: list[tuple[int, float]] = []
    running = float("-inf")
    for record in records:
        old_score = record.get("old_score")
        if isinstance(old_score, (int, float)):
            running = max(running, float(old_score))
        if running != float("-inf"):
            best_so_far.append((record["iteration"], running))
    if best_so_far:
        ax.plot(
            [iteration for iteration, _ in best_so_far],
            [score for _, score in best_so_far],
            linestyle="--",
            color="#6f6f6f",
            alpha=0.65,
            linewidth=1.6,
            label="best so far under p0",
        )

    for item in history[1:]:
        ax.axvline(item["iteration"], color="#222222", linestyle=":", alpha=0.55)
        ax.text(
            item["iteration"] + 0.15,
            1.01,
            f"p{item['version']} @ {item['iteration']}",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel("Generator iteration")
    ax.set_ylabel("Score w/ active evaluator")
    ax.set_xlim(0.5, max_iter + 0.5)
    ax.set_ylim(-0.03, 0.98)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _write_summary(
    run_dir: Path,
    records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    intervals: list[tuple[int, int, int]],
    outer: list[dict[str, Any]],
    figure_path: Path,
) -> Path:
    installed = [record for record in outer if record.get("outcome") == "installed"]
    installed_by_version = {record.get("installed_version"): record for record in installed}
    failures = [record for record in records if record["score"] == 0 and record.get("error")]
    comparisons = [
        record
        for record in records
        if record["version"] > 0 and isinstance(record.get("old_score"), (int, float))
    ]
    active_gt_old = [record for record in comparisons if record["score"] > record["old_score"]]
    active_lt_old = [record for record in comparisons if record["score"] < record["old_score"]]
    active_eq_old = [record for record in comparisons if record["score"] == record["old_score"]]

    by_version: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_version.setdefault(record["version"], []).append(record)

    lines = [
        "# VIS evaluator interval summary",
        "",
        f"- Iterations: {len(records)}",
        f"- Evaluator versions: {', '.join('p' + str(item['version']) for item in history)}",
        f"- Install iterations: {', '.join('p' + str(item['version']) + '@' + str(item['iteration']) for item in history[1:])}",
        f"- Best active score: {max(record['score'] for record in records):.4f}",
        f"- Failures: {len(failures)} / {len(records)}",
        f"- Latest-vs-p0 comparisons after p0: active > p0: {len(active_gt_old)}, active < p0: {len(active_lt_old)}, equal: {len(active_eq_old)}",
        f"- Figure: {figure_path}",
        "",
        "## Version intervals",
    ]
    for version, start, end in intervals:
        segment = by_version.get(version, [])
        scores = [record["score"] for record in segment]
        errors = [record for record in segment if record.get("error")]
        lines.append(
            f"- p{version}: iterations {start}-{end}, n={len(segment)}, "
            f"mean={mean(scores):.4f}, max={max(scores):.4f}, failures={len(errors)}"
        )

    lines.extend(["", "## Installed prompt changes"])
    for summary in _prompt_diff_summaries(run_dir / "evaluator_prompts", history):
        installed_record = installed_by_version.get(summary["to"], {})
        lines.append(
            f"- p{summary['from']}->p{summary['to']} at iteration {summary['iteration']}: "
            f"+{summary['added']}/-{summary['removed']} lines, "
            f"diversity={installed_record.get('prompt_diversity')}, "
            f"drift={installed_record.get('drift')}, fitness={installed_record.get('fitness')}"
        )
        if summary["preview"]:
            lines.append(f"  Preview: {summary['preview']}")

    lines.extend(
        [
            "",
            "## Notes",
            "- combined_score follows the active evaluator prompt version; old_combined_score remains diagnostic.",
            "- The grey dashed line is cumulative best-so-far under p0.",
            "- The prompt progression report shows only local changes between installed prompts.",
            "",
        ]
    )

    out_path = run_dir / "evaluator_interval_summary.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Run directory containing adaevolve logs and evaluator_prompts/")
    args = parser.parse_args()

    run_dir = args.run_dir
    stats_path = _find_iteration_stats(run_dir)
    history_path = run_dir / "evaluator_prompts" / "evaluator_prompt_history.jsonl"
    outer_path = run_dir / "outer_evaluator_stats.jsonl"

    records = _load_records(stats_path)
    history = _read_jsonl(history_path)
    outer = _read_jsonl(outer_path) if outer_path.exists() else []
    intervals = _version_intervals(history, max(record["iteration"] for record in records))

    figure_path = _plot_intervals(run_dir, records, history, intervals)
    summary_path = _write_summary(run_dir, records, history, intervals, outer, figure_path)
    progression_path = run_dir / "evaluator_prompt_progression_local_changes.md"
    _write_prompt_progression(run_dir / "evaluator_prompts", progression_path)

    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {progression_path}")


if __name__ == "__main__":
    main()
