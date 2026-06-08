"""Per-LLM-call telemetry + unit-level cumulative counters.

Writes one JSON line per LLM call to `{run_dir}/telemetry.jsonl` and tracks
running totals readable by unit-record writers at serialize time.

Usage:
    sink = TelemetrySink(run_dir=run_dir, method="adaevolve")
    set_sink(sink)
    with unit_scope(child_program.id):
        # any LLM call under this scope is tagged with unit_id=child_program.id
        ...
    program.metadata["llm_calls_at_creation"] = sink.n_calls
    program.metadata["llm_tokens_at_creation"] = sink.n_tokens
"""

import contextlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("skydiscover.llm.telemetry")

current_unit_id: ContextVar[Optional[str]] = ContextVar("current_unit_id", default=None)
current_role: ContextVar[str] = ContextVar("current_role", default="generator")
_sink: ContextVar[Optional["TelemetrySink"]] = ContextVar("_sink", default=None)


def set_sink(sink: Optional["TelemetrySink"]) -> None:
    _sink.set(sink)


def get_sink() -> Optional["TelemetrySink"]:
    return _sink.get()


class TelemetrySink:
    """Append-only JSONL writer + in-memory cumulative counters.

    Threadsafe within a process. For multi-process executors, each worker
    should construct its own sink pointing at a per-pid spool file
    (`telemetry.worker_{pid}.jsonl`); the parent concatenates at session end.
    """

    def __init__(self, run_dir: str, method: str, filename: str = "telemetry.jsonl") -> None:
        self.run_dir = run_dir
        self.method = method
        self.path = os.path.join(run_dir, filename)
        os.makedirs(run_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._n_calls = 0
        self._n_tokens = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def record(
        self,
        *,
        ts: Optional[float] = None,
        duration_ms: Optional[int] = None,
        unit_id: Optional[str] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ts is None:
            ts = time.time()
        if unit_id is None:
            unit_id = current_unit_id.get()
        if role is None:
            role = current_role.get()

        row: Dict[str, Any] = {
            "ts": ts,
            "duration_ms": duration_ms,
            "unit_id": unit_id,
            "method": self.method,
            "role": role,
            "model": model,
            "api_base": api_base,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": error,
        }
        if extra:
            row.update(extra)

        try:
            line = json.dumps(row, default=str) + "\n"
        except (TypeError, ValueError) as e:
            logger.warning("telemetry: failed to serialize row: %s", e)
            return

        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as e:
                logger.warning("telemetry: failed to append to %s: %s", self.path, e)
                return
            self._n_calls += 1
            if tokens_in:
                self._n_tokens += int(tokens_in)
            if tokens_out:
                self._n_tokens += int(tokens_out)


@contextlib.contextmanager
def unit_scope(unit_id: str) -> Iterator[None]:
    """Tag LLM calls made within this scope with `unit_id`."""
    token = current_unit_id.set(unit_id)
    try:
        yield
    finally:
        current_unit_id.reset(token)


@contextlib.contextmanager
def role_scope(role: str) -> Iterator[None]:
    """Tag LLM calls made within this scope with `role` (e.g., 'evaluator')."""
    token = current_role.set(role)
    try:
        yield
    finally:
        current_role.reset(token)
