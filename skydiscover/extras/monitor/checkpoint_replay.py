"""Shared checkpoint replay helpers for monitor viewers and launcher replay."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CheckpointReplayData:
    """Replay-ready monitor state extracted from one checkpoint snapshot."""

    checkpoint_dir: Path
    monitor_programs: List[Dict[str, Any]]
    raw_programs_by_id: Dict[str, Dict[str, Any]]
    program_solutions: Dict[str, str]
    parent_solutions: Dict[str, str]
    best_program_id: Optional[str]
    last_iteration: int
    last_timestamp: Optional[float]


def _ckpt_num(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def find_checkpoint_dir(path: str | Path) -> Optional[Path]:
    """Auto-detect the best checkpoint directory from *path*."""
    p = Path(path)

    # 1. Direct checkpoint dir (metadata.json + programs/)
    if (p / "metadata.json").exists() and (p / "programs").is_dir():
        return p

    # 2. programs/ subdir but no metadata
    if (p / "programs").is_dir() and list((p / "programs").glob("*.json")):
        return p

    # 3. checkpoint_N dirs directly inside path
    ckpts = sorted(p.glob("checkpoint_*"), key=lambda item: _ckpt_num(item.name))
    if ckpts:
        return ckpts[-1]

    # 4. checkpoints/ subdir
    if (p / "checkpoints").is_dir():
        ckpts = sorted(
            (p / "checkpoints").glob("checkpoint_*"),
            key=lambda item: _ckpt_num(item.name),
        )
        if ckpts:
            return ckpts[-1]

    # 5. <subdir>/checkpoints/ (e.g. island/checkpoints/, sequential/checkpoints/)
    if p.is_dir():
        for subdir in sorted(p.iterdir()):
            if not subdir.is_dir():
                continue
            ckpt_dir = subdir / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            ckpts = sorted(
                ckpt_dir.glob("checkpoint_*"),
                key=lambda item: _ckpt_num(item.name),
            )
            if ckpts:
                return ckpts[-1]

    # 6. Flat directory with JSON program files
    jsons = [candidate for candidate in p.glob("*.json") if candidate.name != "metadata.json"]
    if jsons:
        return p

    return None


def load_checkpoint_programs(ckpt_dir: str | Path) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    """Load raw program payloads from one checkpoint directory."""
    p = Path(ckpt_dir)
    programs: Dict[str, Dict[str, Any]] = {}
    best_program_id: Optional[str] = None
    last_iteration = 0

    meta_path = p / "metadata.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        best_program_id = meta.get("best_program_id")
        last_iteration = meta.get("last_iteration", 0)

    programs_dir = p / "programs"
    if programs_dir.is_dir():
        candidates = sorted(programs_dir.glob("*.json"))
    else:
        candidates = sorted(candidate for candidate in p.glob("*.json") if candidate.name != "metadata.json")

    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.warning("Skipping checkpoint program %s: %s", candidate, exc)
            continue
        if not isinstance(data, dict):
            continue
        program_id = data.get("id")
        if not isinstance(program_id, str) or not program_id:
            continue
        programs[program_id] = data

    if not best_program_id and programs:
        best_score = -float("inf")
        for program_id, program in programs.items():
            score = (program.get("metrics") or {}).get("combined_score", 0)
            if isinstance(score, (int, float)) and score > best_score:
                best_score = float(score)
                best_program_id = program_id

    program_list = sorted(programs.values(), key=lambda program: program.get("iteration_found", 0))
    return program_list, best_program_id, last_iteration


def checkpoint_program_to_monitor_format(
    prog: Mapping[str, Any],
    all_programs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Convert one checkpoint program payload into the live-monitor shape."""
    metrics = prog.get("metrics") or {}
    score = metrics.get("combined_score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    parent_id = prog.get("parent_id")
    parent_score = None
    parent_iter = None
    if isinstance(parent_id, str) and parent_id and parent_id in all_programs:
        parent_metrics = all_programs[parent_id].get("metrics") or {}
        parent_score = parent_metrics.get("combined_score")
        parent_iter = all_programs[parent_id].get("iteration_found")

    context_ids = prog.get("other_context_ids") or []
    context_scores = []
    for context_id in context_ids:
        if context_id in all_programs:
            context_metrics = all_programs[context_id].get("metrics") or {}
            context_scores.append(context_metrics.get("combined_score"))
        else:
            context_scores.append(None)

    label_type = "unknown"
    parent_info = prog.get("parent_info")
    if parent_info and isinstance(parent_info, (list, tuple)) and len(parent_info) >= 1:
        label_string = str(parent_info[0]).lower()
        if "diverge" in label_string:
            label_type = "diverge"
        elif "refine" in label_string:
            label_type = "refine"
        elif "crossover" in label_string:
            label_type = "crossover"
    if label_type == "unknown":
        label_type = (prog.get("metadata") or {}).get("label_type", "unknown")

    island = (prog.get("metadata") or {}).get("island")
    image_path = (prog.get("metadata") or {}).get("image_path")
    solution = prog.get("solution", "")

    from skydiscover.extras.monitor.callback import (
        _extract_display_text,
        _extract_error_text,
        _safe_metrics,
    )

    return {
        "id": prog["id"],
        "iteration": prog.get("iteration_found", 0),
        "score": score,
        "metrics": _safe_metrics(metrics),
        "parent_id": parent_id,
        "parent_score": parent_score,
        "parent_iter": parent_iter,
        "context_ids": context_ids,
        "context_scores": context_scores,
        "label_type": label_type,
        "solution_snippet": solution[:500],
        "island": island,
        "generation": prog.get("generation", 0),
        "image_path": image_path,
        "error_text": _extract_error_text(metrics),
        "display_text": _extract_display_text(SimpleNamespace(**prog)),
    }


def load_checkpoint_replay_data(path: str | Path) -> Optional[CheckpointReplayData]:
    """Load replay-ready monitor payloads from the latest checkpoint under *path*."""
    checkpoint_dir = find_checkpoint_dir(path)
    if checkpoint_dir is None:
        return None

    program_list, best_program_id, last_iteration = load_checkpoint_programs(checkpoint_dir)
    if not program_list:
        return None

    raw_programs_by_id = {program["id"]: program for program in program_list if isinstance(program.get("id"), str)}
    monitor_programs: List[Dict[str, Any]] = []
    program_solutions: Dict[str, str] = {}
    parent_solutions: Dict[str, str] = {}
    last_timestamp: Optional[float] = None

    for program in program_list:
        monitor_program = checkpoint_program_to_monitor_format(program, raw_programs_by_id)
        monitor_programs.append(monitor_program)

        program_id = monitor_program["id"]
        program_solutions[program_id] = str(program.get("solution") or "")

        parent_id = monitor_program.get("parent_id")
        if isinstance(parent_id, str) and parent_id and parent_id in raw_programs_by_id:
            parent_solutions[program_id] = str(raw_programs_by_id[parent_id].get("solution") or "")

        timestamp = program.get("timestamp")
        if isinstance(timestamp, (int, float)):
            numeric_timestamp = float(timestamp)
            if last_timestamp is None or numeric_timestamp > last_timestamp:
                last_timestamp = numeric_timestamp

    return CheckpointReplayData(
        checkpoint_dir=checkpoint_dir,
        monitor_programs=monitor_programs,
        raw_programs_by_id=raw_programs_by_id,
        program_solutions=program_solutions,
        parent_solutions=parent_solutions,
        best_program_id=best_program_id,
        last_iteration=last_iteration,
        last_timestamp=last_timestamp,
    )
