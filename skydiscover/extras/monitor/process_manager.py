"""Minimal child-process manager for the standalone VIS launcher."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from skydiscover.config import Config, _parse_model_spec
from skydiscover.extras.monitor.checkpoint_replay import load_checkpoint_replay_data
from skydiscover.extras.monitor.server import WS_GUID, _ws_encode_text
from skydiscover.extras.monitor.trace_store import (
    CandidateTraceStore,
    trace_record_to_monitor_program,
)

logger = logging.getLogger(__name__)

_MONITOR_URL_RE = re.compile(r"Live monitor:\s*(http://[^\s]+)", re.IGNORECASE)


def _max_timestamp(current: Optional[float], candidate: Any) -> Optional[float]:
    try:
        candidate_value = float(candidate)
    except (TypeError, ValueError):
        return current
    if current is None or candidate_value > current:
        return candidate_value
    return current


@dataclass(frozen=True)
class LauncherRecipe:
    """Fixed run inputs used by the standalone launcher."""

    evaluation_file: str
    initial_program: Optional[str]
    config_path: Optional[str]
    output_root: str
    iteration_budget: Optional[int]
    run_forever: bool
    log_level: Optional[str] = None


@dataclass
class BranchRunHandle:
    """Bookkeeping for a candidate-seeded side branch."""

    branch_id: str
    display_name: str
    run_dir: Path
    config_path: Path
    seed_program_path: Path
    feedback_path: Path
    source_program_id: str
    source_program: Dict[str, Any]
    source_solution: str
    anchor_iteration: int
    parent_run_label: str
    parent_series_label: str
    monitor_port: int
    process: Optional[subprocess.Popen[str]] = None
    socket: Optional[socket.socket] = None
    socket_file: Any = None
    socket_write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    stdout_thread: Optional[threading.Thread] = None
    wait_thread: Optional[threading.Thread] = None
    relay_thread: Optional[threading.Thread] = None
    child_run_label: str = ""
    status: str = "starting"
    last_error: str = ""
    launched_at: float = field(default_factory=time.time)
    seed_program_id: Optional[str] = None


class ProcessManager:
    """Own one child SkyDiscover process and relay its monitor events."""

    def __init__(
        self,
        *,
        config: Config,
        recipe: LauncherRecipe,
        event_callback: Callable[[Dict[str, Any]], None],
        python_executable: Optional[str] = None,
    ):
        self.config = config
        self.recipe = recipe
        self.event_callback = event_callback
        self.python_executable = python_executable or sys.executable

        self.output_root = Path(recipe.output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_root / "launcher_state.json"

        self._lock = threading.RLock()
        self._socket_write_lock = threading.Lock()

        self._process: Optional[subprocess.Popen[str]] = None
        self._child_socket: Optional[socket.socket] = None
        self._child_socket_file = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._wait_thread: Optional[threading.Thread] = None
        self._relay_thread: Optional[threading.Thread] = None
        self._session_token = 0

        self._active_run_dir: Optional[Path] = None
        self._latest_checkpoint: Optional[str] = None
        self._child_pid: Optional[int] = None
        self._child_monitor_port: Optional[int] = None
        self._child_run_label: str = ""
        self._child_series_label: str = self.config.search.type
        self._last_error: str = ""
        self._run_state_cache: Dict[str, Any] = self._default_run_state()
        self._program_owner: Dict[str, str] = {}
        self._solution_cache: Dict[str, Dict[str, str]] = {}
        self._branch_runs: Dict[str, BranchRunHandle] = {}
        self._historical_branch_dirs: Dict[str, Path] = {}

        self._load_state()
        replay_state = self.build_replay_state()
        if replay_state is not None:
            self.event_callback(replay_state)

    def get_run_state(self) -> Dict[str, Any]:
        """Return launcher-aware run state for the outer monitor server."""
        with self._lock:
            self._refresh_from_disk_locked()
            return dict(self._run_state_cache)

    def get_run_label(self) -> str:
        """Return the current human-readable run label."""
        with self._lock:
            if self._child_run_label:
                return self._child_run_label
            if self._active_run_dir is not None:
                return self._active_run_dir.name
            return f"{self.config.search.type}-launcher"

    def get_series_label(self) -> str:
        """Return the current plotted series label."""
        with self._lock:
            return self._child_series_label or self.config.search.type

    def build_replay_state(self) -> Optional[Dict[str, Any]]:
        """Return a replay snapshot reconstructed from persisted trace files."""
        with self._lock:
            return self._build_replay_state_locked()

    def _build_replay_state_locked(self) -> Optional[Dict[str, Any]]:
        if self._active_run_dir is None:
            return None

        replay_programs: list[Dict[str, Any]] = []
        last_event_at: Optional[float] = None

        self._reset_program_cache_locked()

        main_trace_store = CandidateTraceStore(self._active_run_dir)
        if main_trace_store.has_trace():
            for record in main_trace_store.load_candidate_records():
                self._append_replayed_program_locked(
                    owner="main",
                    record=record,
                    trace_store=main_trace_store,
                    replay_programs=replay_programs,
                )
                last_event_at = _max_timestamp(last_event_at, record.get("timestamp"))

        for branch in self._restore_historical_branch_handles_locked():
            branch_trace_store = CandidateTraceStore(branch.run_dir)
            branch_trace_records = (
                branch_trace_store.load_candidate_records() if branch_trace_store.has_trace() else []
            )
            if not branch_trace_records:
                checkpoint_replay = load_checkpoint_replay_data(branch.run_dir)
                if checkpoint_replay is None:
                    logger.warning(
                        "Skipping historical branch %s because neither trace nor checkpoint replay data was found.",
                        branch.branch_id,
                    )
                    continue
                for program in checkpoint_replay.monitor_programs:
                    self._append_replayed_branch_checkpoint_program_locked(
                        branch=branch,
                        program=program,
                        full_solution=checkpoint_replay.program_solutions.get(program["id"], ""),
                        parent_solution=checkpoint_replay.parent_solutions.get(program["id"], ""),
                        replay_programs=replay_programs,
                    )
                last_event_at = _max_timestamp(last_event_at, checkpoint_replay.last_timestamp)
                continue
            for record in branch_trace_records:
                self._append_replayed_branch_program_locked(
                    branch=branch,
                    record=record,
                    trace_store=branch_trace_store,
                    replay_programs=replay_programs,
                )
                last_event_at = _max_timestamp(last_event_at, record.get("timestamp"))

        if not replay_programs:
            return None

        ordered_ids: list[str] = []
        programs_by_id: Dict[str, Dict[str, Any]] = {}
        for program in replay_programs:
            program_id = str(program.get("id") or "").strip()
            if not program_id:
                continue
            if program_id not in programs_by_id:
                ordered_ids.append(program_id)
            programs_by_id[program_id] = program

        programs = [programs_by_id[program_id] for program_id in ordered_ids]
        best_program = None
        best_score = -float("inf")
        best_iteration = 0
        max_iteration = 0
        for program in programs:
            iteration = self._coerce_iteration(program.get("iteration"), 0)
            max_iteration = max(max_iteration, iteration)
            score = program.get("score")
            if isinstance(score, (int, float)) and score >= best_score:
                best_score = float(score)
                best_program = program
                best_iteration = iteration

        program_solutions = {
            program_id: self._solution_cache.get(program_id, {}).get("solution", "")
            for program_id in ordered_ids
        }
        parent_solutions = {
            program_id: self._solution_cache.get(program_id, {}).get("parent_solution", "")
            for program_id in ordered_ids
        }

        return {
            "type": "replace_state",
            "programs": programs,
            "program_solutions": program_solutions,
            "parent_solutions": parent_solutions,
            "best_program_id": best_program.get("id") if isinstance(best_program, dict) else None,
            "stats": {
                "total_programs": len(programs),
                "current_iteration": max_iteration,
                "best_score": 0.0 if best_score == -float("inf") else best_score,
                "iterations_since_improvement": max(0, max_iteration - best_iteration),
                "programs_per_min": 0.0,
                "elapsed_seconds": 0.0,
            },
            "last_event_at": last_event_at,
        }

    def _append_replayed_program_locked(
        self,
        *,
        owner: str,
        record: Dict[str, Any],
        trace_store: CandidateTraceStore,
        replay_programs: list[Dict[str, Any]],
    ) -> None:
        program = trace_record_to_monitor_program(record)
        program_id = str(program.get("id") or "").strip()
        if not program_id:
            return

        parent_solution = ""
        parent_id = record.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            parent_solution = trace_store.load_solution(parent_id)

        message = {
            "type": "new_program",
            "program": program,
            "full_solution": trace_store.load_solution(program_id, record=record),
            "parent_full_solution": parent_solution,
        }
        self._cache_program_message(message, owner=owner)
        replay_programs.append(program)

    def _append_replayed_branch_program_locked(
        self,
        *,
        branch: BranchRunHandle,
        record: Dict[str, Any],
        trace_store: CandidateTraceStore,
        replay_programs: list[Dict[str, Any]],
    ) -> None:
        local_program = trace_record_to_monitor_program(record)
        local_program_id = str(local_program.get("id") or "").strip()
        if not local_program_id:
            return

        parent_solution = ""
        parent_id = record.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            parent_solution = trace_store.load_solution(parent_id)

        message = {
            "type": "new_program",
            "program": local_program,
            "full_solution": trace_store.load_solution(local_program_id, record=record),
            "parent_full_solution": parent_solution,
        }
        translated = self._rewrite_branch_program_event_locked(branch, message)
        if translated is None:
            return
        replay_programs.append(dict(translated.get("program") or {}))

    def _append_replayed_branch_checkpoint_program_locked(
        self,
        *,
        branch: BranchRunHandle,
        program: Dict[str, Any],
        full_solution: str,
        parent_solution: str,
        replay_programs: list[Dict[str, Any]],
    ) -> None:
        message = {
            "type": "new_program",
            "program": dict(program),
            "full_solution": full_solution,
            "parent_full_solution": parent_solution,
        }
        translated = self._rewrite_branch_program_event_locked(branch, message)
        if translated is None:
            return
        replay_programs.append(dict(translated.get("program") or {}))

    def _restore_historical_branch_handles_locked(self) -> list[BranchRunHandle]:
        self._historical_branch_dirs = {}
        if self._active_run_dir is None:
            return []

        branch_root = self._active_run_dir / "branches"
        if not branch_root.is_dir():
            return []

        restored: list[BranchRunHandle] = []
        for meta_path in sorted(branch_root.glob("*/branch_meta.json")):
            branch = self._restore_branch_handle_from_disk_locked(meta_path.parent)
            if branch is None:
                continue
            self._historical_branch_dirs[branch.branch_id] = branch.run_dir
            restored.append(branch)
        return restored

    def _restore_branch_handle_from_disk_locked(
        self,
        branch_dir: Path,
    ) -> Optional[BranchRunHandle]:
        meta_path = branch_dir / "branch_meta.json"
        if not meta_path.exists():
            return None

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read historical branch metadata from %s", meta_path)
            return None

        if not isinstance(metadata, dict):
            return None

        seed_candidate_payload: Dict[str, Any] = {}
        seed_candidate_path = branch_dir / "seed_candidate.json"
        if seed_candidate_path.exists():
            try:
                loaded_seed_payload = json.loads(seed_candidate_path.read_text(encoding="utf-8"))
                if isinstance(loaded_seed_payload, dict):
                    seed_candidate_payload = loaded_seed_payload
            except Exception:
                logger.debug("Failed to read %s", seed_candidate_path, exc_info=True)

        seed_program_path = self._resolve_branch_seed_program_path(
            branch_dir,
            seed_candidate_payload,
        )
        feedback_path = Path(
            seed_candidate_payload.get("feedback_path") or branch_dir / "expert_feedback.md"
        )
        config_path = branch_dir / "branch_config.yaml"
        source_solution = ""
        if seed_program_path.exists():
            try:
                source_solution = seed_program_path.read_text(encoding="utf-8")
            except Exception:
                logger.debug("Failed to read historical branch seed %s", seed_program_path, exc_info=True)

        source_program = metadata.get("source_program")
        if not isinstance(source_program, dict):
            source_program = {}

        source_program_id = str(
            metadata.get("source_program_id") or source_program.get("id") or branch_dir.name
        ).strip()
        branch_id = str(metadata.get("branch_id") or branch_dir.name).strip() or branch_dir.name

        return BranchRunHandle(
            branch_id=branch_id,
            display_name=str(metadata.get("display_name") or f"branch {source_program_id[:8]}"),
            run_dir=branch_dir,
            config_path=config_path,
            seed_program_path=seed_program_path,
            feedback_path=feedback_path,
            source_program_id=source_program_id,
            source_program=dict(source_program),
            source_solution=source_solution,
            anchor_iteration=self._coerce_iteration(metadata.get("anchor_iteration"), 0),
            parent_run_label=str(
                metadata.get("parent_run_label")
                or source_program.get("run_label")
                or (self._active_run_dir.name if self._active_run_dir else "")
            ),
            parent_series_label=str(
                metadata.get("parent_series_label")
                or source_program.get("series_label")
                or self.config.search.type
            ),
            monitor_port=self._coerce_iteration(metadata.get("monitor_port"), 0),
            status="replayed",
        )

    def _resolve_branch_seed_program_path(
        self,
        branch_dir: Path,
        seed_candidate_payload: Dict[str, Any],
    ) -> Path:
        raw_path = seed_candidate_payload.get("seed_program_path")
        if isinstance(raw_path, str) and raw_path.strip():
            return Path(raw_path)

        for candidate in branch_dir.glob("seed_program.*"):
            return candidate
        return branch_dir / "seed_program.py"

    def start_new_run(self) -> None:
        """Start a brand-new child run from the fixed launcher recipe."""
        with self._lock:
            self._ensure_can_start_locked()
            try:
                self._validate_launch_prerequisites_locked()
            except Exception as exc:
                self._set_failed_run_state_locked(
                    error_message=str(exc),
                    output_dir=None,
                    checkpoint_path=None,
                )
                raise
            self._shutdown_branch_runs_locked(clear_records=True)
            self._reset_program_cache_locked()
            self._reset_cached_child_connection_locked()

            run_dir = self._make_run_dir_locked()
            run_dir.mkdir(parents=True, exist_ok=True)

            self._active_run_dir = run_dir
            self._latest_checkpoint = None
            self._child_run_label = run_dir.name
            self._child_series_label = self.config.search.type
            self._last_error = ""
            self._run_state_cache = self._default_run_state(
                status="starting",
                output_dir=str(run_dir),
                checkpoint_path=None,
            )
            self._write_state_locked()

            try:
                self._spawn_child_locked(run_dir=run_dir, checkpoint_path=None)
            except Exception as exc:
                self._set_failed_run_state_locked(
                    error_message=str(exc),
                    output_dir=str(run_dir),
                    checkpoint_path=None,
                )
                raise

        self.event_callback({"type": "reset_state"})
        self.event_callback({"type": "run_state_update"})

    def start_branch_from_program(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Start a candidate-seeded side branch without interrupting the live run."""
        with self._lock:
            self._validate_launch_prerequisites_locked()
            branch = self._prepare_branch_run_locked(msg)
            self._branch_runs[branch.branch_id] = branch
            try:
                self._spawn_branch_locked(branch)
            except Exception:
                self._branch_runs.pop(branch.branch_id, None)
                raise
            self._write_state_locked()

        return {
            "message": (
                f"Branch {branch.display_name} started from {branch.source_program_id[:8]} "
                f"at iteration {branch.anchor_iteration}."
            ),
            "branch": self._branch_summary(branch),
        }

    def resume_latest_run(self) -> None:
        """Resume the latest paused run from its most recent checkpoint."""
        with self._lock:
            self._ensure_can_resume_locked()
            try:
                self._validate_launch_prerequisites_locked()
            except Exception as exc:
                self._last_error = str(exc)
                self._run_state_cache["last_error"] = self._last_error
                self._write_state_locked()
                raise
            run_dir = self._active_run_dir
            if run_dir is None:
                raise RuntimeError("No paused run directory is available to resume.")

            checkpoint_path = self._latest_checkpoint or self._find_latest_checkpoint_locked()
            if not checkpoint_path:
                raise RuntimeError("No checkpoint is available for resume.")

            self._reset_cached_child_connection_locked()
            self._last_error = ""
            self._run_state_cache = self._default_run_state(
                status="starting",
                output_dir=str(run_dir),
                checkpoint_path=checkpoint_path,
            )
            self._write_state_locked()

            try:
                self._spawn_child_locked(run_dir=run_dir, checkpoint_path=checkpoint_path)
            except Exception as exc:
                self._last_error = str(exc)
                self._run_state_cache = self._default_run_state(
                    status="paused",
                    output_dir=str(run_dir),
                    checkpoint_path=checkpoint_path,
                )
                self._run_state_cache["last_error"] = self._last_error
                self._write_state_locked()
                raise

        self.event_callback({"type": "run_state_update"})

    def request_stop(self) -> None:
        """Ask the child run to pause gracefully."""
        if not self._send_to_child({"type": "request_stop_run"}):
            raise RuntimeError("The child monitor is not connected yet.")
        with self._lock:
            self._request_stop_branch_runs_locked()
            self._write_state_locked()

    def proxy_request(self, msg: Dict[str, Any]) -> Optional[bool]:
        """Forward a dashboard request to the child when available.

        Returns:
            ``True`` when the request was forwarded asynchronously.
            ``None`` when no child monitor is connected and local fallback should be used.
        """
        message_type = msg.get("type")
        program_id = str(msg.get("program_id") or "").strip()
        hinted_branch_id = str(msg.get("branch_id") or "").strip()

        if message_type == "request_image":
            return None

        if message_type == "request_program_solution" and program_id:
            with self._lock:
                cached = dict(self._solution_cache.get(program_id, {}))
                owner = self._program_owner.get(program_id) or hinted_branch_id or None
                if not cached.get("solution"):
                    loaded_solution = self._load_solution_from_trace_locked(
                        program_id,
                        owner=owner,
                    )
                    if loaded_solution:
                        cached = dict(self._solution_cache.get(program_id, {}))
            if cached:
                return {
                    "type": "program_solution",
                    "program_id": program_id,
                    "solution": cached.get("solution", ""),
                    "parent_solution": cached.get("parent_solution", ""),
                }
            if owner and owner != "main":
                if self._send_to_branch(owner, msg):
                    return True
                return None
            if self._send_to_child(msg):
                return True
            return None

        if message_type == "request_program_summary" and program_id:
            with self._lock:
                owner = self._program_owner.get(program_id) or hinted_branch_id or None
            if owner and owner != "main":
                if self._send_to_branch(owner, msg):
                    return True
                return None
            if self._send_to_child(msg):
                return True
            return None

        if self._send_to_child(msg):
            return True
        return None

    def shutdown(self) -> None:
        """Stop relay resources and terminate any managed child process."""
        process = None
        with self._lock:
            process = self._process
            self._reset_cached_child_connection_locked()
            self._shutdown_branch_runs_locked(clear_records=False)
            self._write_state_locked()

        if process is None:
            return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                logger.debug("Failed while terminating launcher child", exc_info=True)

    def _default_run_state(
        self,
        *,
        status: str = "idle",
        output_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "search_type": self.config.search.type,
            "max_iterations": self.recipe.iteration_budget,
            "remaining_iterations": self.recipe.iteration_budget,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "launcher_mode": True,
            "child_pid": self._child_pid,
            "last_error": self._last_error or None,
        }

    def _load_state(self) -> None:
        with self._lock:
            if self.state_path.exists():
                try:
                    data = json.loads(self.state_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Failed to read launcher_state.json", exc_info=True)
                    data = {}

                active_run_dir = data.get("active_run_dir")
                self._active_run_dir = Path(active_run_dir) if active_run_dir else None
                self._latest_checkpoint = data.get("latest_checkpoint")
                self._child_monitor_port = data.get("child_monitor_port")
                self._child_run_label = data.get("child_run_label", "") or ""
                self._child_series_label = (
                    data.get("child_series_label", "") or self.config.search.type
                )
                self._last_error = data.get("last_error", "") or ""

                run_state = data.get("run_state")
                if isinstance(run_state, dict):
                    self._run_state_cache = dict(run_state)

            self._run_state_cache.setdefault("launcher_mode", True)
            self._refresh_from_disk_locked()
            self._write_state_locked()

    def _write_state_locked(self) -> None:
        payload = {
            "status": self._run_state_cache.get("status", "idle"),
            "active_run_dir": str(self._active_run_dir) if self._active_run_dir else None,
            "latest_checkpoint": self._latest_checkpoint,
            "child_pid": self._child_pid,
            "child_monitor_port": self._child_monitor_port,
            "updated_at": time.time(),
            "last_error": self._last_error or None,
            "child_run_label": self._child_run_label or None,
            "child_series_label": self._child_series_label or None,
            "branches": [self._branch_summary(branch) for branch in self._branch_runs.values()],
            "recipe": asdict(self.recipe),
            "run_state": self._run_state_cache,
        }
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.state_path)

    def _refresh_from_disk_locked(self) -> None:
        if self._active_run_dir is None:
            if not self._is_process_alive_locked():
                fallback_status = "failed" if self._last_error else "idle"
                self._run_state_cache = self._default_run_state(status=fallback_status)
            return

        latest_checkpoint = self._find_latest_checkpoint_locked()
        if latest_checkpoint:
            self._latest_checkpoint = latest_checkpoint

        run_state_path = self._active_run_dir / "run_state.json"
        if run_state_path.exists():
            try:
                child_state = json.loads(run_state_path.read_text(encoding="utf-8"))
                if isinstance(child_state, dict):
                    self._merge_run_state_locked(child_state)
            except Exception:
                logger.debug("Failed to refresh child run_state.json", exc_info=True)

        if not self._is_process_alive_locked():
            self._finalize_stale_terminal_state_locked()
        else:
            self._run_state_cache["child_pid"] = self._child_pid

    def _merge_run_state_locked(self, child_state: Dict[str, Any]) -> None:
        merged = dict(child_state)
        merged.setdefault("status", "running")
        merged.setdefault("search_type", self.config.search.type)
        merged["launcher_mode"] = True
        merged["output_dir"] = str(self._active_run_dir) if self._active_run_dir else None
        merged["checkpoint_path"] = child_state.get("checkpoint_path") or self._latest_checkpoint
        merged["child_pid"] = self._child_pid
        if self._last_error:
            merged["last_error"] = self._last_error
        self._run_state_cache = merged
        if merged.get("checkpoint_path"):
            self._latest_checkpoint = merged["checkpoint_path"]
        self._write_state_locked()

    def _finalize_stale_terminal_state_locked(self) -> None:
        """Normalize stale launcher state after the child process has exited.

        This handles cases where the child crashed or was interrupted before it
        could persist a terminal run_state, leaving a stale ``running`` record
        behind that would otherwise keep ``Start`` disabled in the UI.
        """
        status = self._run_state_cache.get("status")
        if status not in {None, "", "starting", "running"}:
            self._run_state_cache["child_pid"] = None
            return

        fallback_status = "failed" if self._last_error else "completed"
        self._run_state_cache["status"] = fallback_status
        self._run_state_cache["launcher_mode"] = True
        self._run_state_cache["output_dir"] = (
            str(self._active_run_dir) if self._active_run_dir else None
        )
        self._run_state_cache["checkpoint_path"] = (
            self._run_state_cache.get("checkpoint_path") or self._latest_checkpoint
        )
        self._run_state_cache["child_pid"] = None
        if self._last_error:
            self._run_state_cache["last_error"] = self._last_error
        self._write_state_locked()

    def _ensure_can_start_locked(self) -> None:
        if self._is_process_alive_locked():
            raise RuntimeError("A run is already active.")

        status = self._run_state_cache.get("status")
        if status not in {None, "", "idle", "paused", "completed", "failed", "early_stopping"}:
            raise RuntimeError(f"Cannot start a new run while status is '{status}'.")

    def _ensure_can_resume_locked(self) -> None:
        if self._is_process_alive_locked():
            raise RuntimeError("A run is already active.")

        status = self._run_state_cache.get("status")
        if status != "paused":
            raise RuntimeError(f"Resume is only available from paused state, not '{status}'.")

    def _validate_launch_prerequisites_locked(self) -> None:
        missing_credentials = []
        seen_models = set()

        configured_models = (
            list(self.config.llm.models)
            + list(self.config.llm.evaluator_models)
            + list(self.config.llm.guide_models)
        )

        for model in configured_models:
            ident = id(model)
            if ident in seen_models:
                continue
            seen_models.add(ident)

            if getattr(model, "init_client", None):
                continue

            model_name = (getattr(model, "name", None) or "").strip()
            provider, _, _, env_vars = _parse_model_spec(model_name or "openai")
            if not env_vars:
                continue

            api_key = getattr(model, "api_key", None) or self.config.llm.api_key
            if api_key:
                continue

            env_hint = " / ".join(env_vars)
            missing_credentials.append(f"{model_name or provider} ({env_hint})")

        if missing_credentials:
            details = ", ".join(dict.fromkeys(missing_credentials))
            raise RuntimeError(
                "Missing API key for launcher child model(s): "
                f"{details}. Restart the launcher from a shell with the required key set, "
                "or set llm.api_key / llm.models[].api_key in the config."
            )

    def _set_failed_run_state_locked(
        self,
        *,
        error_message: str,
        output_dir: Optional[str],
        checkpoint_path: Optional[str],
    ) -> None:
        self._last_error = error_message
        self._run_state_cache = self._default_run_state(
            status="failed",
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
        )
        self._write_state_locked()

    def _spawn_child_locked(self, *, run_dir: Path, checkpoint_path: Optional[str]) -> None:
        self._session_token += 1
        session_token = self._session_token
        command = self._build_cli_command(run_dir=run_dir, checkpoint_path=checkpoint_path)
        log_path = run_dir / "launcher_child.log"
        repo_root = Path(__file__).resolve().parents[3]

        logger.info("Starting launcher child: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process
        self._child_pid = process.pid
        self._run_state_cache["child_pid"] = process.pid
        self._write_state_locked()

        self._stdout_thread = threading.Thread(
            target=self._consume_process_output,
            args=(process, log_path, session_token),
            name="skydiscover-launcher-stdout",
            daemon=True,
        )
        self._stdout_thread.start()

        self._wait_thread = threading.Thread(
            target=self._wait_for_process,
            args=(process, log_path, session_token),
            name="skydiscover-launcher-wait",
            daemon=True,
        )
        self._wait_thread.start()

    def _prepare_branch_run_locked(self, msg: Dict[str, Any]) -> BranchRunHandle:
        program = msg.get("program")
        if not isinstance(program, dict):
            raise RuntimeError("A selected candidate is required to create a branch.")

        source_program_id = str(program.get("id") or msg.get("program_id") or "").strip()
        if not source_program_id:
            raise RuntimeError("The selected candidate is missing an id.")

        cached_solution = ""
        cached = self._solution_cache.get(source_program_id)
        if isinstance(cached, dict):
            cached_solution = str(cached.get("solution") or "").rstrip()
        if not cached_solution:
            cached_solution = self._load_solution_from_trace_locked(source_program_id).rstrip()

        source_solution = cached_solution or str(msg.get("solution") or "").rstrip()
        if not source_solution:
            raise RuntimeError("The full candidate solution is still loading. Try again in a moment.")

        anchor_iteration = self._coerce_iteration(
            msg.get("anchor_iteration"),
            self._coerce_iteration(program.get("iteration"), 0),
        )
        parent_run_label = self.get_run_label()
        parent_series_label = self.get_series_label()
        branch_id = self._make_branch_id_locked(source_program_id)
        display_name = f"branch {source_program_id[:8]}"
        run_dir = self._make_branch_run_dir_locked(branch_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        file_suffix = self.config.file_suffix or ".py"
        if not file_suffix.startswith("."):
            file_suffix = f".{file_suffix}"
        seed_program_path = run_dir / f"seed_program{file_suffix}"
        feedback_path = run_dir / "expert_feedback.md"
        config_path = run_dir / "branch_config.yaml"
        seed_candidate_path = run_dir / "seed_candidate.json"
        branch_meta_path = run_dir / "branch_meta.json"
        monitor_port = self._reserve_monitor_port_locked()

        feedback_text = str(msg.get("feedback_text") or "").strip()
        feedback_path.write_text(
            self._format_branch_feedback_file(feedback_text),
            encoding="utf-8",
        )
        seed_program_path.write_text(source_solution + ("\n" if source_solution else ""), encoding="utf-8")

        config_payload = asdict(self.config)
        config_payload["prompt"] = config_payload.pop("context_builder", {})
        config_payload["human_feedback_enabled"] = True
        config_payload["human_feedback_file"] = str(feedback_path)
        config_payload["human_feedback_mode"] = "append"
        config_payload.setdefault("monitor", {})
        config_payload["monitor"]["enabled"] = True
        config_payload["monitor"]["host"] = self.config.monitor.host
        config_payload["monitor"]["port"] = monitor_port
        config_payload["monitor"]["summary_interval"] = 0
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        seed_candidate_path.write_text(
            json.dumps(
                {
                    "source_program": program,
                    "source_program_id": source_program_id,
                    "anchor_iteration": anchor_iteration,
                    "source_run_label": str(msg.get("source_run_label") or parent_run_label),
                    "source_series_label": str(
                        msg.get("source_series_label")
                        or program.get("series_label")
                        or parent_series_label
                    ),
                    "seed_program_path": str(seed_program_path),
                    "feedback_path": str(feedback_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        branch_meta_path.write_text(
            json.dumps(
                {
                    "branch_id": branch_id,
                    "display_name": display_name,
                    "mode": "candidate_seeded",
                    "created_at": time.time(),
                    "parent_run_label": parent_run_label,
                    "parent_series_label": parent_series_label,
                    "anchor_iteration": anchor_iteration,
                    "source_program_id": source_program_id,
                    "source_program": program,
                    "source_run_label": str(msg.get("source_run_label") or parent_run_label),
                    "source_series_label": str(
                        msg.get("source_series_label")
                        or program.get("series_label")
                        or parent_series_label
                    ),
                    "feedback_text": feedback_text,
                    "monitor_port": monitor_port,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return BranchRunHandle(
            branch_id=branch_id,
            display_name=display_name,
            run_dir=run_dir,
            config_path=config_path,
            seed_program_path=seed_program_path,
            feedback_path=feedback_path,
            source_program_id=source_program_id,
            source_program=dict(program),
            source_solution=source_solution,
            anchor_iteration=anchor_iteration,
            parent_run_label=parent_run_label,
            parent_series_label=parent_series_label,
            monitor_port=monitor_port,
        )

    def _spawn_branch_locked(self, branch: BranchRunHandle) -> None:
        command = self._build_cli_command(
            run_dir=branch.run_dir,
            checkpoint_path=None,
            initial_program=str(branch.seed_program_path),
            config_path=str(branch.config_path),
            iteration_budget=None,
            run_forever=True,
        )
        log_path = branch.run_dir / "branch_child.log"
        repo_root = Path(__file__).resolve().parents[3]

        logger.info(
            "Starting candidate-seeded branch %s from %s: %s",
            branch.branch_id,
            branch.source_program_id,
            " ".join(command),
        )
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        branch.process = process
        branch.status = "running"

        branch.stdout_thread = threading.Thread(
            target=self._consume_branch_output,
            args=(branch.branch_id, process, log_path),
            name=f"skydiscover-branch-{branch.branch_id[:8]}-stdout",
            daemon=True,
        )
        branch.stdout_thread.start()

        branch.wait_thread = threading.Thread(
            target=self._wait_for_branch_process,
            args=(branch.branch_id, process, log_path),
            name=f"skydiscover-branch-{branch.branch_id[:8]}-wait",
            daemon=True,
        )
        branch.wait_thread.start()

    def _build_cli_command(
        self,
        *,
        run_dir: Path,
        checkpoint_path: Optional[str],
        initial_program: Optional[str] = None,
        config_path: Optional[str] = None,
        iteration_budget: Optional[int] = None,
        run_forever: Optional[bool] = None,
    ) -> list[str]:
        command = [self.python_executable, "-m", "skydiscover.cli"]
        effective_initial_program = (
            initial_program if initial_program is not None else self.recipe.initial_program
        )
        effective_config_path = config_path if config_path is not None else self.recipe.config_path
        effective_run_forever = self.recipe.run_forever if run_forever is None else run_forever
        effective_iteration_budget = (
            self.recipe.iteration_budget if iteration_budget is None else iteration_budget
        )

        if effective_initial_program:
            command.append(effective_initial_program)
        command.append(self.recipe.evaluation_file)
        if effective_config_path:
            command.extend(["--config", effective_config_path])
        command.extend(["--output", str(run_dir)])
        if self.recipe.log_level:
            command.extend(["--log-level", self.recipe.log_level])
        if effective_run_forever:
            command.append("--run-forever")
        elif effective_iteration_budget is not None:
            command.extend(["--iterations", str(effective_iteration_budget)])
        if checkpoint_path:
            command.extend(["--checkpoint", checkpoint_path])
        return command

    def _consume_process_output(
        self, process: subprocess.Popen[str], log_path: Path, session_token: int
    ) -> None:
        stream = process.stdout
        if stream is None:
            return

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in stream:
                log_file.write(line)
                log_file.flush()

                monitor_url = self._extract_monitor_url(line)
                if monitor_url:
                    self._start_relay_thread(monitor_url, session_token)

    def _wait_for_process(
        self, process: subprocess.Popen[str], log_path: Path, session_token: int
    ) -> None:
        try:
            return_code = process.wait()
        except Exception:
            logger.debug("Waiting on child process failed", exc_info=True)
            return

        stdout_thread = None
        with self._lock:
            if session_token != self._session_token:
                return
            self._process = None
            self._child_pid = None
            stdout_thread = self._stdout_thread

        if stdout_thread is not None and stdout_thread.is_alive():
            stdout_thread.join(timeout=1.0)

        self._reset_cached_child_connection(session_token=session_token)

        with self._lock:
            if return_code != 0:
                failure_reason = self._extract_child_failure_reason(log_path)
                self._last_error = failure_reason or f"Child process exited with code {return_code}."
                logger.error("Launcher child failed: %s", self._last_error)
            self._refresh_from_disk_locked()
            status = self._run_state_cache.get("status")
            if return_code != 0 and status not in {"paused", "completed", "early_stopping", "failed"}:
                self._run_state_cache = self._default_run_state(
                    status="failed",
                    output_dir=str(self._active_run_dir) if self._active_run_dir else None,
                    checkpoint_path=self._latest_checkpoint,
                )
            elif status in {None, "", "starting", "running"}:
                self._run_state_cache = self._default_run_state(
                    status="completed" if return_code == 0 else "failed",
                    output_dir=str(self._active_run_dir) if self._active_run_dir else None,
                    checkpoint_path=self._latest_checkpoint,
                )
            self._run_state_cache["child_pid"] = None
            self._write_state_locked()

        self.event_callback({"type": "run_state_update"})

    def _consume_branch_output(
        self,
        branch_id: str,
        process: subprocess.Popen[str],
        log_path: Path,
    ) -> None:
        stream = process.stdout
        if stream is None:
            return

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in stream:
                log_file.write(line)
                log_file.flush()

                monitor_url = self._extract_monitor_url(line)
                if monitor_url:
                    self._start_branch_relay_thread(branch_id, monitor_url)

    def _wait_for_branch_process(
        self,
        branch_id: str,
        process: subprocess.Popen[str],
        log_path: Path,
    ) -> None:
        try:
            return_code = process.wait()
        except Exception:
            logger.debug("Waiting on branch process failed", exc_info=True)
            return

        with self._lock:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                return
            branch.process = None
            expected_stop = branch.status in {"stopping", "terminated"}
            if return_code != 0 and not expected_stop:
                branch.last_error = self._extract_child_failure_reason(log_path) or (
                    f"Branch process exited with code {return_code}."
                )
                branch.status = "failed"
            elif expected_stop:
                branch.last_error = ""
                branch.status = "terminated"
            elif branch.status == "running":
                branch.status = "completed"
            self._write_state_locked()

        self._reset_branch_connection(branch_id)

    def _start_branch_relay_thread(self, branch_id: str, monitor_url: str) -> None:
        parsed = urlparse(monitor_url)
        if parsed.port is None:
            return

        with self._lock:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                return
            if branch.relay_thread is not None and branch.relay_thread.is_alive():
                return

        relay_thread = threading.Thread(
            target=self._relay_branch_monitor,
            args=(branch_id, monitor_url),
            name=f"skydiscover-branch-{branch_id[:8]}-relay",
            daemon=True,
        )
        with self._lock:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                return
            branch.relay_thread = relay_thread
        relay_thread.start()

    def _relay_branch_monitor(self, branch_id: str, monitor_url: str) -> None:
        parsed = urlparse(monitor_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        path = parsed.path or "/"
        if port is None:
            return

        sock = None
        sock_file = None
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock_file = sock.makefile("rwb")
            self._perform_websocket_handshake(sock_file, host=host, port=port, path=path)

            with self._lock:
                branch = self._branch_runs.get(branch_id)
                if branch is None:
                    return
                branch.socket = sock
                branch.socket_file = sock_file

            while True:
                frame = self._read_ws_frame(sock_file)
                if frame is None:
                    break
                self._handle_branch_message(branch_id, frame)
        except Exception:
            logger.debug("Branch monitor relay stopped for %s", branch_id, exc_info=True)
        finally:
            self._reset_branch_connection(branch_id)

    def _handle_branch_message(self, branch_id: str, frame: str) -> None:
        try:
            message = json.loads(frame)
        except Exception:
            return

        translated_events = []
        message_type = message.get("type")
        with self._lock:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                return

            run_label = message.get("run_label")
            if isinstance(run_label, str) and run_label:
                branch.child_run_label = run_label

            if message_type == "init_state":
                programs = message.get("programs", [])
                if isinstance(programs, list):
                    for program in programs:
                        translated = self._rewrite_branch_program_event_locked(
                            branch,
                            {
                                "type": "new_program",
                                "program": program,
                            },
                        )
                        if translated is not None:
                            translated_events.append(translated)
            elif message_type == "new_program":
                translated = self._rewrite_branch_program_event_locked(branch, message)
                if translated is not None:
                    translated_events.append(translated)
            elif message_type == "program_solution":
                translated = self._rewrite_branch_solution_message_locked(branch, message)
                if translated is not None:
                    translated_events.append(translated)
            elif message_type in {"image_data", "program_summary"}:
                translated_events.append(message)

        for event in translated_events:
            self.event_callback(event)

    def _extract_child_failure_reason(self, log_path: Path) -> Optional[str]:
        if not log_path.exists():
            return None

        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            logger.debug("Failed to read launcher child log %s", log_path, exc_info=True)
            return None

        for raw_line in reversed(lines[-120:]):
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("error:"):
                return line.split(":", 1)[1].strip()
            if "OpenAIError:" in line:
                return line.split("OpenAIError:", 1)[1].strip()
            if any(
                marker in line
                for marker in (
                    "RuntimeError:",
                    "ValueError:",
                    "FileNotFoundError:",
                    "ImportError:",
                    "ModuleNotFoundError:",
                    "PermissionError:",
                    "AssertionError:",
                )
            ):
                return line
        return None

    def _rewrite_branch_program_event_locked(
        self,
        branch: BranchRunHandle,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        program = message.get("program")
        if not isinstance(program, dict):
            return None

        rewritten = dict(program)
        program_id = str(rewritten.get("id") or "").strip()
        if not program_id:
            return None

        local_iteration = self._coerce_iteration(
            rewritten.get("iteration"),
            self._coerce_iteration(rewritten.get("iteration_found"), 0),
        )
        rewritten["run_label"] = branch.parent_run_label
        rewritten["series_label"] = branch.display_name
        rewritten["branch_id"] = branch.branch_id
        rewritten["branch_display_name"] = branch.display_name
        rewritten["branch_source_program_id"] = branch.source_program_id
        rewritten["branch_anchor_iteration"] = branch.anchor_iteration
        rewritten["branch_local_iteration"] = local_iteration
        rewritten["branch_run_label"] = branch.child_run_label or branch.run_dir.name
        rewritten["branch_source_run_label"] = branch.source_program.get("run_label", "")
        rewritten["branch_source_series_label"] = (
            branch.source_program.get("series_label") or branch.parent_series_label
        )
        rewritten["iteration"] = branch.anchor_iteration + max(local_iteration, 0)

        source_score = branch.source_program.get("score")
        if local_iteration == 0:
            branch.seed_program_id = program_id
            if not rewritten.get("parent_id"):
                rewritten["parent_id"] = branch.source_program_id
            if rewritten.get("parent_iter") is None:
                rewritten["parent_iter"] = branch.anchor_iteration
            if isinstance(source_score, (int, float)) and rewritten.get("parent_score") is None:
                rewritten["parent_score"] = source_score

        translated = {
            k: v
            for k, v in message.items()
            if k not in {"program", "run_label", "series_label", "stats"}
        }
        translated["type"] = "new_program"
        translated["program"] = rewritten
        translated["run_label"] = branch.parent_run_label
        translated["series_label"] = branch.display_name

        parent_solution = translated.get("parent_full_solution")
        if local_iteration == 0 and not parent_solution:
            translated["parent_full_solution"] = branch.source_solution

        self._cache_program_message(translated, owner=branch.branch_id)
        return translated

    def _rewrite_branch_solution_message_locked(
        self,
        branch: BranchRunHandle,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        program_id = str(message.get("program_id") or "").strip()
        if not program_id:
            return None

        translated = {
            "type": "program_solution",
            "program_id": program_id,
            "solution": message.get("solution", ""),
            "parent_solution": message.get("parent_solution", ""),
        }
        if program_id == branch.seed_program_id and not translated["parent_solution"]:
            translated["parent_solution"] = branch.source_solution
        self._cache_solution_message(translated)
        return translated

    def _start_relay_thread(self, monitor_url: str, session_token: int) -> None:
        parsed = urlparse(monitor_url)
        if parsed.port is None:
            return

        with self._lock:
            if session_token != self._session_token:
                return
            if (
                self._child_monitor_port == parsed.port
                and self._relay_thread is not None
                and self._relay_thread.is_alive()
            ):
                return
            self._child_monitor_port = parsed.port
            self._write_state_locked()

        relay_thread = threading.Thread(
            target=self._relay_child_monitor,
            args=(monitor_url, session_token),
            name="skydiscover-launcher-relay",
            daemon=True,
        )
        self._relay_thread = relay_thread
        relay_thread.start()

    def _relay_child_monitor(self, monitor_url: str, session_token: int) -> None:
        parsed = urlparse(monitor_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        path = parsed.path or "/"
        if port is None:
            return

        sock = None
        sock_file = None
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock_file = sock.makefile("rwb")
            self._perform_websocket_handshake(sock_file, host=host, port=port, path=path)

            with self._lock:
                if session_token != self._session_token:
                    return
                self._child_socket = sock
                self._child_socket_file = sock_file

            while True:
                frame = self._read_ws_frame(sock_file)
                if frame is None:
                    break
                self._handle_child_message(frame, session_token)
        except Exception:
            logger.debug("Child monitor relay stopped", exc_info=True)
        finally:
            self._reset_cached_child_connection(session_token=session_token)

    def _handle_child_message(self, frame: str, session_token: int) -> None:
        try:
            message = json.loads(frame)
        except Exception:
            return

        with self._lock:
            if session_token != self._session_token:
                return

            run_label = message.get("run_label")
            if isinstance(run_label, str) and run_label:
                self._child_run_label = run_label

            series_label = message.get("series_label")
            if isinstance(series_label, str) and series_label:
                self._child_series_label = series_label

            run_state = message.get("run_state")
            if isinstance(run_state, dict):
                self._merge_run_state_locked(run_state)

        message_type = message.get("type")
        if message_type == "init_state":
            self._cache_init_programs(message.get("programs", []), owner="main")
            self.event_callback(self._translate_init_state(message))
            return
        if message_type == "heartbeat":
            return
        if message_type == "run_control_ack" and message.get("action") in {"stop", "resume"}:
            return
        if message_type == "new_program":
            self._cache_program_message(message, owner="main")
        elif message_type == "program_solution":
            self._cache_solution_message(message)

        self.event_callback(message)

    def _translate_init_state(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "replace_state",
            "programs": list(message.get("programs", [])),
            "best_program_id": message.get("best_program_id"),
            "stats": dict(message.get("stats", {})),
            "config_summary": message.get("config_summary", ""),
            "last_event_at": message.get("last_event_at"),
            "summary_model": message.get("summary_model", ""),
            "summary_text": message.get("summary_text", ""),
            "summary_generating": bool(message.get("summary_generating", False)),
            "human_feedback_enabled": bool(message.get("human_feedback_enabled", False)),
            "feedback_text": message.get("feedback_text", ""),
            "feedback_active": bool(message.get("feedback_active", False)),
            "human_feedback_mode": message.get("human_feedback_mode", "append"),
            "human_feedback_current_prompt": message.get("human_feedback_current_prompt", ""),
            "human_feedback_history": list(message.get("human_feedback_history", [])),
        }

    def _cache_init_programs(self, programs: Any, *, owner: str) -> None:
        if not isinstance(programs, list):
            return
        with self._lock:
            for program in programs:
                if not isinstance(program, dict):
                    continue
                program_id = str(program.get("id") or "").strip()
                if not program_id:
                    continue
                self._program_owner[program_id] = owner

    def _cache_program_message(self, message: Dict[str, Any], *, owner: str) -> None:
        program = message.get("program")
        if not isinstance(program, dict):
            return
        program_id = str(program.get("id") or "").strip()
        if not program_id:
            return

        solution = message.get("full_solution")
        parent_solution = message.get("parent_full_solution")
        with self._lock:
            self._program_owner[program_id] = owner
            if isinstance(solution, str) or isinstance(parent_solution, str):
                cached = self._solution_cache.setdefault(
                    program_id,
                    {"solution": "", "parent_solution": ""},
                )
                if isinstance(solution, str):
                    cached["solution"] = solution
                if isinstance(parent_solution, str):
                    cached["parent_solution"] = parent_solution

    def _cache_solution_message(self, message: Dict[str, Any]) -> None:
        program_id = str(message.get("program_id") or "").strip()
        if not program_id:
            return
        with self._lock:
            cached = self._solution_cache.setdefault(
                program_id,
                {"solution": "", "parent_solution": ""},
            )
            if isinstance(message.get("solution"), str):
                cached["solution"] = message["solution"]
            if isinstance(message.get("parent_solution"), str):
                cached["parent_solution"] = message["parent_solution"]

    def _send_to_child(self, msg: Dict[str, Any]) -> bool:
        payload = json.dumps(msg)
        with self._lock:
            sock = self._child_socket
        if sock is None:
            return False

        try:
            with self._socket_write_lock:
                sock.sendall(_ws_encode_text(payload))
            return True
        except Exception:
            logger.debug("Failed to send message to child monitor", exc_info=True)
            self._reset_cached_child_connection()
            return False

    def _send_to_branch(self, branch_id: str, msg: Dict[str, Any]) -> bool:
        payload = json.dumps(msg)
        with self._lock:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                return False
            sock = branch.socket
            write_lock = branch.socket_write_lock
        if sock is None:
            return False

        try:
            with write_lock:
                sock.sendall(_ws_encode_text(payload))
            return True
        except Exception:
            logger.debug("Failed to send message to branch monitor %s", branch_id, exc_info=True)
            self._reset_branch_connection(branch_id)
            return False

    def _perform_websocket_handshake(
        self, sock_file, *, host: str, port: int, path: str
    ) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock_file.write(request.encode("utf-8"))
        sock_file.flush()

        status_line = sock_file.readline().decode("utf-8", errors="replace").strip()
        if "101" not in status_line:
            raise RuntimeError(f"WebSocket handshake failed: {status_line or 'empty response'}")

        headers = {}
        while True:
            line = sock_file.readline().decode("utf-8", errors="replace")
            if line in {"\r\n", "\n", ""}:
                break
            if ":" not in line:
                continue
            key_name, value = line.split(":", 1)
            headers[key_name.strip().lower()] = value.strip()

        expected_accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("utf-8")).digest()
        ).decode("ascii")
        if headers.get("sec-websocket-accept") != expected_accept:
            raise RuntimeError("WebSocket accept key mismatch.")

    def _read_ws_frame(self, sock_file) -> Optional[str]:
        header = self._read_exactly(sock_file, 2)
        if not header:
            return None

        opcode = header[0] & 0x0F
        masked = (header[1] & 0x80) != 0
        length = header[1] & 0x7F

        if opcode == 0x8:
            return None
        if length == 126:
            ext = self._read_exactly(sock_file, 2)
            if not ext:
                return None
            length = int.from_bytes(ext, "big")
        elif length == 127:
            ext = self._read_exactly(sock_file, 8)
            if not ext:
                return None
            length = int.from_bytes(ext, "big")

        mask = self._read_exactly(sock_file, 4) if masked else None
        payload = self._read_exactly(sock_file, length)
        if payload is None:
            return None

        if masked and mask is not None:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

        if opcode == 0x1:
            return payload.decode("utf-8", errors="replace")
        return ""

    def _read_exactly(self, sock_file, length: int) -> Optional[bytes]:
        data = sock_file.read(length)
        if not data or len(data) != length:
            return None
        return data

    def _reset_cached_child_connection(self, *, session_token: Optional[int] = None) -> None:
        with self._lock:
            self._reset_cached_child_connection_locked(session_token=session_token)

    def _reset_branch_connection(self, branch_id: str) -> None:
        with self._lock:
            self._reset_branch_connection_locked(branch_id)

    def _reset_cached_child_connection_locked(
        self, *, session_token: Optional[int] = None
    ) -> None:
        if session_token is not None and session_token != self._session_token:
            return
        if self._child_socket_file is not None:
            try:
                self._child_socket_file.close()
            except Exception:
                logger.debug("Failed to close child monitor file", exc_info=True)
        if self._child_socket is not None:
            try:
                self._child_socket.close()
            except Exception:
                logger.debug("Failed to close child monitor socket", exc_info=True)
        self._child_socket = None
        self._child_socket_file = None
        self._child_monitor_port = None
        self._write_state_locked()

    def _reset_branch_connection_locked(self, branch_id: str) -> None:
        branch = self._branch_runs.get(branch_id)
        if branch is None:
            return
        if branch.socket_file is not None:
            try:
                branch.socket_file.close()
            except Exception:
                logger.debug("Failed to close branch monitor file for %s", branch_id, exc_info=True)
        if branch.socket is not None:
            try:
                branch.socket.close()
            except Exception:
                logger.debug("Failed to close branch monitor socket for %s", branch_id, exc_info=True)
        branch.socket = None
        branch.socket_file = None

    def _shutdown_branch_runs_locked(self, *, clear_records: bool) -> None:
        branch_ids = list(self._branch_runs.keys())
        for branch_id in branch_ids:
            branch = self._branch_runs.get(branch_id)
            if branch is None:
                continue
            self._reset_branch_connection_locked(branch_id)
            process = branch.process
            branch.status = "terminated"
            branch.last_error = ""
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    logger.debug("Failed while terminating branch %s", branch_id, exc_info=True)
            branch.process = None
            if clear_records:
                self._branch_runs.pop(branch_id, None)

        if clear_records:
            self._program_owner = {
                program_id: owner
                for program_id, owner in self._program_owner.items()
                if owner == "main"
            }
            self._solution_cache = {
                program_id: cached
                for program_id, cached in self._solution_cache.items()
                if self._program_owner.get(program_id) == "main"
            }

    def _request_stop_branch_runs_locked(self) -> None:
        for branch in self._branch_runs.values():
            process = branch.process
            if process is None or process.poll() is not None:
                if branch.status in {"starting", "running", "stopping"}:
                    branch.status = "terminated"
                    branch.last_error = ""
                continue
            if branch.status in {"completed", "failed", "terminated"}:
                continue
            branch.status = "stopping"
            branch.last_error = ""
            try:
                process.terminate()
            except Exception:
                logger.debug(
                    "Failed while requesting stop for branch %s",
                    branch.branch_id,
                    exc_info=True,
                )

    def _reset_program_cache_locked(self) -> None:
        self._program_owner = {}
        self._solution_cache = {}
        self._historical_branch_dirs = {}

    def _trace_store_for_owner_locked(self, owner: Optional[str]) -> Optional[CandidateTraceStore]:
        if owner in {None, "", "main"}:
            if self._active_run_dir is None:
                return None
            return CandidateTraceStore(self._active_run_dir)

        live_branch = self._branch_runs.get(owner)
        if live_branch is not None:
            return CandidateTraceStore(live_branch.run_dir)

        historical_branch_dir = self._historical_branch_dirs.get(owner)
        if historical_branch_dir is not None:
            return CandidateTraceStore(historical_branch_dir)

        return None

    def _load_solution_from_trace_locked(
        self,
        program_id: str,
        *,
        owner: Optional[str] = None,
    ) -> str:
        resolved_owner = owner or self._program_owner.get(program_id) or "main"
        trace_store = self._trace_store_for_owner_locked(resolved_owner)
        if trace_store is None:
            return ""

        solution = trace_store.load_solution(program_id)
        if not solution:
            return ""

        cached = self._solution_cache.setdefault(
            program_id,
            {"solution": "", "parent_solution": ""},
        )
        cached["solution"] = solution
        self._program_owner[program_id] = resolved_owner
        return solution

    def _is_process_alive_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _make_run_dir_locked(self) -> Path:
        source_path = self.recipe.initial_program or self.recipe.evaluation_file
        problem_name = Path(source_path).expanduser().resolve().parent.name or "scratch"
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return self.output_root / self.config.search.type / f"{problem_name}_{timestamp}"

    def _make_branch_id_locked(self, source_program_id: str) -> str:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return f"branch_{timestamp}_{source_program_id[:8]}"

    def _make_branch_run_dir_locked(self, branch_id: str) -> Path:
        if self._active_run_dir is not None:
            return self._active_run_dir / "branches" / branch_id
        return self.output_root / self.config.search.type / "branches" / branch_id

    def _branch_summary(self, branch: BranchRunHandle) -> Dict[str, Any]:
        return {
            "branch_id": branch.branch_id,
            "display_name": branch.display_name,
            "run_dir": str(branch.run_dir),
            "source_program_id": branch.source_program_id,
            "anchor_iteration": branch.anchor_iteration,
            "parent_run_label": branch.parent_run_label,
            "parent_series_label": branch.parent_series_label,
            "monitor_port": branch.monitor_port,
            "status": branch.status,
            "last_error": branch.last_error or None,
        }

    def _reserve_monitor_port_locked(self) -> int:
        host = self.config.monitor.host or "127.0.0.1"
        bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((bind_host, 0))
            probe.listen(1)
            return int(probe.getsockname()[1])

    def _format_branch_feedback_file(self, feedback_text: str) -> str:
        header = (
            "# Expert branch brief\n"
            "# Edit this file to steer this branch.\n"
            "# The text below will be appended to the branch system prompt.\n"
        )
        if feedback_text:
            return header + "\n" + feedback_text.strip() + "\n"
        return header

    def _coerce_iteration(self, value: Any, fallback: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    def _find_latest_checkpoint_locked(self) -> Optional[str]:
        if self._active_run_dir is None:
            return None
        checkpoint_dir = self._active_run_dir / "checkpoints"
        if not checkpoint_dir.is_dir():
            return None

        best_iteration = -1
        best_path: Optional[Path] = None
        for path in checkpoint_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                iteration = int(path.name.rsplit("_", 1)[-1])
            except (IndexError, ValueError):
                continue
            if iteration > best_iteration:
                best_iteration = iteration
                best_path = path

        return str(best_path) if best_path else None

    def _extract_monitor_url(self, line: str) -> Optional[str]:
        match = _MONITOR_URL_RE.search(line)
        if not match:
            return None
        return match.group(1).rstrip("/")
