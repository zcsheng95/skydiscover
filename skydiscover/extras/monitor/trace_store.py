"""Run-local candidate trace persistence helpers for the VIS monitor."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1
TRACE_DIR_NAME = "monitor_trace"
TRACE_RECORDS_FILENAME = "candidates.jsonl"
TRACE_INDEX_FILENAME = "index.json"
TRACE_SOLUTIONS_DIRNAME = "solutions"


def resolve_trace_dir(run_dir: str | Path) -> Path:
    """Return the run-local trace directory path."""
    return Path(run_dir).expanduser().resolve() / TRACE_DIR_NAME


def trace_record_to_monitor_program(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a persisted trace record back into a monitor-facing program payload."""
    program: Dict[str, Any] = {
        "id": str(record.get("program_id") or record.get("id") or ""),
        "iteration": _coerce_int(record.get("iteration"), 0),
        "score": _coerce_number(record.get("score"), 0.0),
        "metrics": dict(record.get("metrics") or {}),
        "parent_id": _coerce_optional_string(record.get("parent_id")),
        "parent_score": _coerce_optional_number(record.get("parent_score")),
        "parent_iter": _coerce_optional_int(record.get("parent_iter")),
        "context_ids": _coerce_string_list(record.get("context_ids")),
        "context_scores": _coerce_list(record.get("context_scores")),
        "label_type": _coerce_optional_string(record.get("label_type")) or "unknown",
        "solution_snippet": _coerce_optional_string(record.get("solution_snippet")) or "",
        "island": record.get("island"),
        "is_best": bool(record.get("is_best")),
        "generation": _coerce_int(record.get("generation"), 0),
        "image_path": _coerce_optional_string(record.get("image_path")),
        "error_text": _coerce_optional_string(record.get("error_text")),
        "display_text": _coerce_optional_string(record.get("display_text")),
        "run_label": _coerce_optional_string(record.get("run_label")),
        "series_label": _coerce_optional_string(record.get("series_label")),
    }

    for key in (
        "branch_id",
        "branch_source_program_id",
        "branch_local_iteration",
        "branch_run_label",
        "branch_source_run_label",
        "branch_source_series_label",
        "branch_anchor_iteration",
        "branch_display_name",
    ):
        value = record.get(key)
        if value is not None:
            program[key] = value

    return program


class CandidateTraceStore:
    """Persist and replay candidate history for one run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.trace_dir = resolve_trace_dir(self.run_dir)
        self.records_path = self.trace_dir / TRACE_RECORDS_FILENAME
        self.index_path = self.trace_dir / TRACE_INDEX_FILENAME
        self.solutions_dir = self.trace_dir / TRACE_SOLUTIONS_DIRNAME

    def has_trace(self) -> bool:
        """Return ``True`` when this run already has persisted trace data."""
        return self.records_path.exists()

    def append_candidate_record(
        self,
        record: Dict[str, Any],
        *,
        solution_text: str = "",
    ) -> Dict[str, Any]:
        """Append one normalized candidate record and persist its full solution."""
        program_id = str(record.get("program_id") or record.get("id") or "").strip()
        if not program_id:
            raise ValueError("Candidate trace record is missing program_id.")

        stored = self._normalize_record(record)
        stored["program_id"] = program_id
        stored["trace_schema_version"] = int(
            stored.get("trace_schema_version") or TRACE_SCHEMA_VERSION
        )
        stored["timestamp"] = _coerce_number(stored.get("timestamp"), time.time())

        self._ensure_layout()
        solution_path = self.persist_solution(program_id, solution_text)
        stored["solution_path"] = solution_path

        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored, sort_keys=True) + "\n")

        self._write_index(updated_at=stored["timestamp"])
        return stored

    def persist_solution(self, program_id: str, solution_text: str) -> str:
        """Persist the full solution text for a candidate once."""
        relative_path = self.solution_relative_path(program_id)
        if not isinstance(solution_text, str) or not solution_text:
            return relative_path

        self._ensure_layout()
        target = self.trace_dir / relative_path
        if target.exists():
            return relative_path

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(solution_text, encoding="utf-8")
        return relative_path

    def load_candidate_records(self) -> List[Dict[str, Any]]:
        """Load replay-ready candidate records, deduplicated by ``program_id``."""
        if not self.records_path.exists():
            return []

        ordered_ids: List[str] = []
        records_by_id: Dict[str, Dict[str, Any]] = {}
        with self.records_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    logger.warning("Skipping malformed candidate trace line in %s", self.records_path)
                    continue
                if not isinstance(parsed, dict):
                    continue

                record = self._normalize_record(parsed)
                program_id = str(record.get("program_id") or record.get("id") or "").strip()
                if not program_id:
                    continue

                record["program_id"] = program_id
                if program_id not in records_by_id:
                    ordered_ids.append(program_id)
                records_by_id[program_id] = record

        return [records_by_id[program_id] for program_id in ordered_ids]

    def load_solution(
        self,
        program_id: str,
        *,
        record: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Load a persisted full solution by program id."""
        relative_path = None
        if isinstance(record, dict):
            raw_path = record.get("solution_path")
            if isinstance(raw_path, str) and raw_path.strip():
                relative_path = raw_path.strip()

        target = self.trace_dir / (
            relative_path or self.solution_relative_path(program_id)
        )
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception:
            logger.debug("Failed to read trace solution %s", target, exc_info=True)
            return ""

    def solution_relative_path(self, program_id: str) -> str:
        """Return the relative trace path used to store one candidate solution."""
        safe_program_id = _safe_program_id(program_id)
        return f"{TRACE_SOLUTIONS_DIRNAME}/{safe_program_id}.txt"

    def _ensure_layout(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.solutions_dir.mkdir(parents=True, exist_ok=True)

    def _write_index(self, *, updated_at: float) -> None:
        payload = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "updated_at": updated_at,
        }
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(record)
        normalized["program_id"] = str(
            normalized.get("program_id") or normalized.get("id") or ""
        ).strip()
        normalized["iteration"] = _coerce_int(normalized.get("iteration"), 0)
        normalized["score"] = _coerce_number(normalized.get("score"), 0.0)
        normalized["metrics"] = dict(normalized.get("metrics") or {})
        normalized["parent_id"] = _coerce_optional_string(normalized.get("parent_id"))
        normalized["parent_score"] = _coerce_optional_number(normalized.get("parent_score"))
        normalized["parent_iter"] = _coerce_optional_int(normalized.get("parent_iter"))
        normalized["context_ids"] = _coerce_string_list(normalized.get("context_ids"))
        normalized["context_scores"] = _coerce_list(normalized.get("context_scores"))
        normalized["generation"] = _coerce_int(normalized.get("generation"), 0)
        normalized["label_type"] = _coerce_optional_string(normalized.get("label_type")) or "unknown"
        normalized["image_path"] = _coerce_optional_string(normalized.get("image_path"))
        normalized["display_text"] = _coerce_optional_string(normalized.get("display_text"))
        normalized["error_text"] = _coerce_optional_string(normalized.get("error_text"))
        normalized["run_label"] = _coerce_optional_string(normalized.get("run_label"))
        normalized["series_label"] = _coerce_optional_string(normalized.get("series_label"))
        normalized["solution_snippet"] = (
            _coerce_optional_string(normalized.get("solution_snippet")) or ""
        )
        normalized["timestamp"] = _coerce_number(normalized.get("timestamp"), time.time())
        normalized["trace_schema_version"] = int(
            normalized.get("trace_schema_version") or TRACE_SCHEMA_VERSION
        )
        normalized["is_best"] = bool(normalized.get("is_best"))
        return normalized


def _safe_program_id(program_id: str) -> str:
    return (
        str(program_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _coerce_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _coerce_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = _coerce_optional_string(item)
        if text:
            result.append(text)
    return result


def _coerce_optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_number(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if numeric != numeric:  # NaN guard
        return float(fallback)
    return numeric


def _coerce_optional_number(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
