"""
WebSocket + HTTP server for the live monitor dashboard.

Single port, zero extra dependencies beyond Python stdlib + optional websockets.
Uses raw asyncio TCP so it works regardless of websockets version.

HTTP GET /        → serves the VIS-style dashboard
HTTP GET /legacy  → serves the legacy monitor dashboard
WS upgrade        → real-time event stream
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from skydiscover.extras.monitor.trace_store import (
    CandidateTraceStore,
    trace_record_to_monitor_program,
)

logger = logging.getLogger(__name__)

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"
VIS_DASHBOARD_PATH = Path(__file__).parent / "dashboard_vis.html"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
PROXIED_CLIENT_MESSAGE_TYPES = {
    "request_program_solution",
    "set_feedback",
    "clear_feedback",
    "request_feedback_state",
    "set_human_feedback_mode",
    "request_system_prompt",
    "request_human_feedback_history",
    "request_image",
    "request_program_summary",
    "request_summary",
}


def _ws_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _ws_encode_text(text: str) -> bytes:
    """Encode a text frame (server→client, unmasked)."""
    payload = text.encode("utf-8")
    length = len(payload)
    if length < 126:
        header = struct.pack("BB", 0x81, length)
    elif length < 65536:
        header = struct.pack("!BBH", 0x81, 126, length)
    else:
        header = struct.pack("!BBQ", 0x81, 127, length)
    return header + payload


async def _ws_read_frame(reader: asyncio.StreamReader) -> Optional[str]:
    """Read one WebSocket frame; return text payload or None on close/error."""
    try:
        header = await reader.readexactly(2)
    except Exception:
        return None
    opcode = header[0] & 0x0F
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7F

    if opcode == 0x8:  # Close
        return None
    if opcode == 0x9:  # Ping — we could reply with pong but we ignore it here
        return None

    if length == 126:
        ext = await reader.readexactly(2)
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = await reader.readexactly(8)
        length = struct.unpack("!Q", ext)[0]

    if masked:
        mask = await reader.readexactly(4)
        data = bytearray(await reader.readexactly(length))
        for i in range(length):
            data[i] ^= mask[i % 4]
        payload = bytes(data)
    else:
        payload = await reader.readexactly(length)

    if opcode == 0x1:  # Text
        return payload.decode("utf-8", errors="replace")
    return None  # Binary / continuation frames ignored


class MonitorServer:
    """
    Single-port HTTP+WebSocket server for live solution discovery monitoring.

    - GET /        →  VIS dashboard
    - GET /legacy  →  legacy dashboard
    - WS upgrade   →  event broadcast
    Runs in a daemon thread with its own asyncio event loop.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_solution_length: int = 10000,
        output_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.max_solution_length = max_solution_length
        self.output_dir = output_dir

        self._queue: queue.Queue = queue.Queue()
        self._trace_store: Optional[CandidateTraceStore] = (
            CandidateTraceStore(output_dir) if output_dir else None
        )

        # In-memory state for reconnecting clients
        self._programs: List[Dict[str, Any]] = []
        self._program_solutions: Dict[str, str] = {}
        self._parent_solutions: Dict[str, str] = {}
        self._trace_records_by_id: Dict[str, Dict[str, Any]] = {}
        self._best_program_id: Optional[str] = None
        self._best_score: float = -float("inf")
        self._stats: Dict[str, Any] = {}
        self._config_summary: str = ""

        # Per-program summary cache
        self._program_summary_cache: Dict[str, str] = {}

        # Human feedback reader (set via set_feedback_reader)
        self._feedback_reader: Optional[Any] = None
        self._feedback_state_cache: Dict[str, Any] = {
            "human_feedback_enabled": False,
            "feedback_text": "",
            "feedback_active": False,
            "human_feedback_mode": "append",
            "human_feedback_current_prompt": "",
            "human_feedback_history": [],
        }

        # AI summary state
        self._summary_model: str = ""
        self._summary_api_key: str = ""
        self._summary_api_base: str = "https://api.openai.com/v1"
        self._summary_top_k: int = 3
        self._summary_interval: int = 0  # 0 = manual only
        self._summary_text: str = ""
        self._summary_generating: bool = False
        self._summary_last_program_count: int = 0
        self._summary_executor: Optional[ThreadPoolExecutor] = None

        self._last_event_at: Optional[float] = None
        self._dashboards: Dict[str, bytes] = {}
        self._start_handler: Optional[Callable[[], None]] = None
        self._stop_handler: Optional[Callable[[], None]] = None
        self._resume_handler: Optional[Callable[[], None]] = None
        self._branch_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self._request_proxy: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._run_state_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._run_label_provider: Optional[Callable[[], str]] = None
        self._series_label_provider: Optional[Callable[[Optional[Dict[str, Any]]], str]] = None
        self._solution_provider: Optional[Callable[[str], Any]] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set[asyncio.StreamWriter] = set()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # set when TCP port is bound
        self._dashboard_html: Optional[bytes] = None

    def start(self) -> None:
        """Load the dashboard and start the server in a daemon thread."""
        self._load_dashboard()
        self._load_trace_replay_state()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait until TCP port is actually bound (up to 5s)
        self._ready_event.wait(timeout=5)
        logger.info(f"Monitor server started → http://localhost:{self.port}/")

    def stop(self) -> None:
        """Signal the server to stop and wait for the thread to finish."""
        self._stop_event.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            # Schedule cancellation of all tasks, then stop the loop
            try:
                loop.call_soon_threadsafe(self._cancel_all_tasks)
            except RuntimeError:
                pass  # Loop already closed
        if self._thread:
            self._thread.join(timeout=5)

    def _cancel_all_tasks(self) -> None:
        """Cancel every pending task on the server's event loop, then stop it."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    def push_event(self, event: Dict[str, Any]) -> None:
        """Enqueue an event for broadcast to all connected WebSocket clients."""
        self._queue.put_nowait(event)

    def set_config_summary(self, summary: str) -> None:
        """Set a human-readable config summary sent to new dashboard clients."""
        self._config_summary = summary

    def set_feedback_reader(self, reader: Any) -> None:
        """Attach a HumanFeedbackReader for dashboard human feedback controls."""
        self._feedback_reader = reader

    def set_start_handler(self, handler: Optional[Callable[[], None]]) -> None:
        """Attach a start handler for launcher-style monitor modes."""
        self._start_handler = handler

    def set_stop_handler(self, handler: Optional[Callable[[], None]]) -> None:
        """Attach a graceful-stop handler for the current live run."""
        self._stop_handler = handler

    def set_resume_handler(self, handler: Optional[Callable[[], None]]) -> None:
        """Attach a resume handler for paused live runs."""
        self._resume_handler = handler

    def set_branch_handler(
        self, handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]
    ) -> None:
        """Attach a candidate-guided branch creation handler."""
        self._branch_handler = handler

    def set_request_proxy(self, handler: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
        """Attach a proxy used to forward dashboard requests elsewhere."""
        self._request_proxy = handler

    def set_run_state_provider(self, provider: Optional[Callable[[], Dict[str, Any]]]) -> None:
        """Override how run-state metadata is resolved."""
        self._run_state_provider = provider

    def set_run_label_provider(self, provider: Optional[Callable[[], str]]) -> None:
        """Override how the human-readable run label is resolved."""
        self._run_label_provider = provider

    def set_series_label_provider(
        self, provider: Optional[Callable[[Optional[Dict[str, Any]]], str]]
    ) -> None:
        """Override how the primary plotted series label is resolved."""
        self._series_label_provider = provider

    def set_solution_provider(self, provider: Optional[Callable[[str], Any]]) -> None:
        """Override how full per-program solutions are resolved on demand."""
        self._solution_provider = provider

    def configure_summary(
        self,
        model: str = "gpt-5-mini",
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        top_k: int = 3,
        interval: int = 0,
    ) -> None:
        """Configure the AI summary generator.

        Args:
            model: OpenAI model name (default gpt-5-mini).
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            api_base: API base URL.
            top_k: Number of top programs to include in summary prompt.
            interval: Auto-generate every N new programs (0 = manual only).
        """
        self._summary_model = model
        self._summary_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._summary_api_base = api_base
        self._summary_top_k = top_k
        self._summary_interval = interval
        self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")
        logger.info(
            f"AI summary configured: model={model}, top_k={top_k}, "
            f"interval={interval or 'manual'}, api_key={'set' if self._summary_api_key else 'MISSING'}"
        )

    def _get_feedback_state(self) -> Dict[str, Any]:
        """Return current human feedback state."""
        if not self._feedback_reader:
            return dict(self._feedback_state_cache)
        text = self._feedback_reader.read()
        return {
            "human_feedback_enabled": True,
            "feedback_text": text,
            "feedback_active": bool(text),
            "human_feedback_mode": self._feedback_reader.mode,
            "human_feedback_current_prompt": self._feedback_reader.get_current_prompt(),
            "human_feedback_history": self._feedback_reader.get_history(),
        }

    def _build_init_state(self) -> Dict[str, Any]:
        """Build the full init_state payload for new/reconnecting WS clients."""
        run_state = self._get_run_state()
        state = {
            "type": "init_state",
            "programs": self._programs,
            "best_program_id": self._best_program_id,
            "stats": self._stats,
            "config_summary": self._config_summary,
            "run_label": self._get_run_label(),
            "series_label": self._get_series_label(run_state),
            "run_state": run_state,
            "last_event_at": self._last_event_at,
            "control_capabilities": self._get_control_capabilities(run_state),
            "branch_capabilities": self._get_branch_capabilities(),
            "summary_enabled": bool(self._summary_model),
            "summary_model": self._summary_model or "",
            "summary_text": self._summary_text,
            "summary_generating": self._summary_generating,
        }
        state.update(self._get_feedback_state())
        return state

    def _get_run_state(self) -> Dict[str, Any]:
        """Read the latest persisted run-state metadata when available."""
        if self._run_state_provider is not None:
            try:
                data = self._run_state_provider()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.debug("Custom run_state provider failed", exc_info=True)

        if not self.output_dir:
            return {}

        state_path = Path(self.output_dir) / "run_state.json"
        if not state_path.exists():
            return {}

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.debug("Failed to read run_state.json", exc_info=True)
        return {}

    def _get_run_label(self) -> str:
        """Return a stable human-readable run label."""
        if self._run_label_provider is not None:
            try:
                label = self._run_label_provider()
                if isinstance(label, str) and label:
                    return label
            except Exception:
                logger.debug("Custom run_label provider failed", exc_info=True)
        if self.output_dir:
            return Path(self.output_dir).name
        return "live-run"

    def _get_series_label(self, run_state: Optional[Dict[str, Any]] = None) -> str:
        """Return the primary series label for live monitors."""
        if self._series_label_provider is not None:
            try:
                label = self._series_label_provider(run_state)
                if isinstance(label, str) and label:
                    return label
            except Exception:
                logger.debug("Custom series_label provider failed", exc_info=True)
        run_state = run_state or self._get_run_state()
        search_type = run_state.get("search_type")
        if isinstance(search_type, str) and search_type:
            return search_type
        if self._config_summary:
            return self._config_summary.split("|", 1)[0].strip()
        return "skydiscover"

    def _get_program_solution_pair(self, program_id: str) -> tuple[str, str]:
        solution = self._program_solutions.get(program_id, "")
        parent_solution = self._parent_solutions.get(program_id, "")

        if self._solution_provider is not None:
            try:
                provided = self._solution_provider(program_id)
            except Exception:
                logger.debug("Custom solution provider failed for %s", program_id, exc_info=True)
            else:
                if isinstance(provided, (list, tuple)) and len(provided) >= 2:
                    provided_solution, provided_parent_solution = provided[0], provided[1]
                    if isinstance(provided_solution, str):
                        solution = provided_solution
                    if isinstance(provided_parent_solution, str):
                        parent_solution = provided_parent_solution
                    self._program_solutions[program_id] = solution
                    self._parent_solutions[program_id] = parent_solution

        if not solution and self._trace_store is not None:
            record = self._trace_records_by_id.get(program_id)
            solution = self._trace_store.load_solution(program_id, record=record)
            if solution:
                self._program_solutions[program_id] = solution

        if not parent_solution:
            record = self._trace_records_by_id.get(program_id)
            if isinstance(record, dict):
                parent_id = record.get("parent_id")
                if isinstance(parent_id, str) and parent_id:
                    parent_solution = self._program_solutions.get(parent_id, "")
                    if not parent_solution and self._trace_store is not None:
                        parent_record = self._trace_records_by_id.get(parent_id)
                        parent_solution = self._trace_store.load_solution(
                            parent_id,
                            record=parent_record,
                        )
                    if parent_solution:
                        self._parent_solutions[program_id] = parent_solution

        return solution, parent_solution

    def _load_trace_replay_state(self) -> None:
        """Best-effort replay of persisted candidate history for this run."""
        if self._trace_store is None or not self._trace_store.has_trace():
            return

        try:
            records = self._trace_store.load_candidate_records()
        except Exception:
            logger.warning("Failed to load monitor trace replay for %s", self.output_dir, exc_info=True)
            return

        if not records:
            return

        programs: List[Dict[str, Any]] = []
        program_solutions: Dict[str, str] = {}
        parent_solutions: Dict[str, str] = {}
        trace_records_by_id: Dict[str, Dict[str, Any]] = {}
        best_program_id: Optional[str] = None
        best_score = -float("inf")
        best_iteration = 0
        last_event_at: Optional[float] = None

        records_by_id = {
            str(record.get("program_id") or ""): record
            for record in records
            if isinstance(record, dict) and record.get("program_id")
        }

        for record in records:
            program = trace_record_to_monitor_program(record)
            program_id = str(program.get("id") or "").strip()
            if not program_id:
                continue

            programs.append(program)
            trace_records_by_id[program_id] = dict(record)

            solution = self._trace_store.load_solution(program_id, record=record)
            program_solutions[program_id] = solution

            parent_id = record.get("parent_id")
            if isinstance(parent_id, str) and parent_id:
                parent_record = records_by_id.get(parent_id)
                if parent_record is not None:
                    parent_solutions[program_id] = self._trace_store.load_solution(
                        parent_id,
                        record=parent_record,
                    )

            score = program.get("score", 0.0)
            if isinstance(score, (int, float)) and score >= best_score:
                best_score = float(score)
                best_program_id = program_id
                best_iteration = int(program.get("iteration") or 0)

            timestamp = record.get("timestamp")
            if isinstance(timestamp, (int, float)):
                if last_event_at is None or float(timestamp) > last_event_at:
                    last_event_at = float(timestamp)

        if not programs:
            return

        current_iteration = max(
            int(program.get("iteration") or 0) for program in programs
        )
        stats = {
            "total_programs": len(programs),
            "current_iteration": current_iteration,
            "best_score": 0.0 if best_score == -float("inf") else best_score,
            "iterations_since_improvement": max(0, current_iteration - best_iteration),
            "programs_per_min": 0.0,
            "elapsed_seconds": 0.0,
        }

        self._replace_cached_state(
            {
                "programs": programs,
                "program_solutions": program_solutions,
                "parent_solutions": parent_solutions,
                "best_program_id": best_program_id,
                "stats": stats,
                "last_event_at": last_event_at,
            }
        )
        self._trace_records_by_id = trace_records_by_id
        logger.info(
            "Replayed %s historical candidate(s) from %s",
            len(programs),
            self._trace_store.records_path,
        )

    def _get_control_capabilities(self, run_state: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Return which run-control actions are currently wired."""
        run_state = run_state or self._get_run_state()
        status = run_state.get("status")
        can_start = self._start_handler is not None and status in {
            None,
            "",
            "idle",
            "paused",
            "completed",
            "failed",
            "early_stopping",
        }
        can_stop = self._stop_handler is not None and status not in {
            "paused",
            "completed",
            "failed",
            "early_stopping",
            "idle",
            "starting",
        }
        can_resume = self._resume_handler is not None and status == "paused"
        return {
            "start": can_start,
            "resume": can_resume,
            "stop": can_stop,
        }

    def _get_branch_capabilities(self) -> Dict[str, bool]:
        """Return which branch actions are available in the current monitor mode."""
        return {
            "create": self._branch_handler is not None,
        }

    def _load_dashboard(self) -> None:
        def _read_html(path: Path, fallback_title: str) -> bytes:
            try:
                return path.read_text(encoding="utf-8").encode("utf-8")
            except FileNotFoundError:
                logger.warning("Dashboard HTML not found at %s", path)
                return (
                    f"<html><body><h1>{fallback_title} not found</h1></body></html>"
                ).encode("utf-8")

        self._dashboards = {
            "/": _read_html(VIS_DASHBOARD_PATH, "VIS dashboard"),
            "/vis": _read_html(VIS_DASHBOARD_PATH, "VIS dashboard"),
            "/legacy": _read_html(DASHBOARD_PATH, "Legacy dashboard"),
        }
        self._dashboard_html = self._dashboards["/"]

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except (RuntimeError, asyncio.CancelledError):
            pass  # Normal on shutdown
        except Exception:
            logger.exception("Monitor server error")
        finally:
            # Drain any remaining cancelled tasks so they don't warn on GC
            try:
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    for t in pending:
                        t.cancel()
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                logger.debug("Error cancelling tasks during stop", exc_info=True)
            try:
                self._loop.close()
            except Exception:
                logger.debug("Error closing event loop", exc_info=True)

    async def _serve(self) -> None:
        # Try configured port, then auto-increment if already in use
        port = self.port
        for attempt in range(10):
            try:
                server = await asyncio.start_server(self._handle_connection, self.host, port)
                break
            except OSError:
                if attempt == 9:
                    raise
                port += 1
        self.port = port
        async with server:
            self._ready_event.set()  # signal that port is bound
            logger.debug(f"Listening on {self.host}:{self.port}")
            consumer = asyncio.create_task(self._consume_queue())
            hb = asyncio.create_task(self._heartbeat())
            try:
                await asyncio.gather(consumer, hb)
            except (asyncio.CancelledError, RuntimeError):
                pass
            finally:
                try:
                    consumer.cancel()
                    hb.cancel()
                except RuntimeError:
                    pass  # Event loop already closed

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Route an incoming connection to HTTP or WebSocket handler."""
        try:
            # Read HTTP request line + headers
            raw_headers: Dict[str, str] = {}
            request_line = (await reader.readline()).decode("utf-8", errors="replace").strip()
            if not request_line:
                writer.close()
                return

            parts = request_line.split()
            path = parts[1] if len(parts) >= 2 else "/"
            path = path.split("?", 1)[0]

            while True:
                line = (await reader.readline()).decode("utf-8", errors="replace").strip()
                if not line:
                    break
                if ":" in line:
                    k, _, v = line.partition(":")
                    raw_headers[k.strip().lower()] = v.strip()

            is_ws = raw_headers.get("upgrade", "").lower() == "websocket"

            if is_ws:
                await self._handle_ws(reader, writer, raw_headers)
            else:
                await self._handle_http(writer, path)
        except Exception:
            logger.debug("Connection handler error", exc_info=True)
        finally:
            try:
                writer.close()
            except Exception:
                logger.debug("Error closing writer", exc_info=True)

    async def _handle_http(self, writer: asyncio.StreamWriter, path: str = "/") -> None:
        """Serve the dashboard HTML over a plain HTTP GET."""
        html = self._dashboards.get(path) or self._dashboard_html or b""
        resp = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(html)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode() + html
        writer.write(resp)
        await writer.drain()

    async def _handle_ws(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: Dict[str, str],
    ) -> None:
        """Complete the WebSocket handshake and enter the read loop."""
        key = headers.get("sec-websocket-key", "")
        accept = _ws_accept_key(key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode()
        writer.write(handshake)
        await writer.drain()

        self._clients.add(writer)
        logger.debug(f"WS client connected ({len(self._clients)} total)")
        try:
            await self._ws_send(writer, json.dumps(self._build_init_state()))
            # Read loop
            while True:
                text = await _ws_read_frame(reader)
                if text is None:
                    break
                await self._handle_client_msg(writer, text)
        except Exception:
            logger.debug("WebSocket handler error", exc_info=True)
        finally:
            self._clients.discard(writer)
            logger.debug(f"WS client disconnected ({len(self._clients)} total)")

    async def _handle_client_msg(self, writer: asyncio.StreamWriter, raw: str) -> None:
        """Dispatch an incoming WebSocket JSON message from a dashboard client."""
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if (
            self._request_proxy is not None
            and t in PROXIED_CLIENT_MESSAGE_TYPES
        ):
            try:
                proxy_result = self._request_proxy(msg)
                if isinstance(proxy_result, dict):
                    await self._ws_send(writer, json.dumps(proxy_result))
                    return
                if proxy_result:
                    return
            except Exception as exc:
                logger.warning("Failed to proxy client message %s: %s", t, exc)

        if t == "request_full_state":
            await self._ws_send(writer, json.dumps(self._build_init_state()))
        elif t == "request_start_run":
            if self._start_handler is None:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "start",
                            "ok": False,
                            "message": "Start is not available in this monitor mode.",
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
                return

            try:
                self._start_handler()
                await self._broadcast(
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "start",
                            "ok": True,
                            "message": "Start requested. Preparing a new run...",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    )
                )
            except Exception as exc:
                logger.warning("Failed to request start: %s", exc)
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "start",
                            "ok": False,
                            "message": f"Failed to request start: {exc}",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
        elif t == "request_program_solution":
            pid = msg.get("program_id", "")
            solution, parent_solution = self._get_program_solution_pair(pid)
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_solution",
                        "program_id": pid,
                        "solution": solution,
                        "parent_solution": parent_solution,
                    }
                ),
            )
        elif t == "set_feedback":
            text = msg.get("text", "").strip()
            if self._feedback_reader:
                self._feedback_reader.write_from_dashboard(text)
                ack = {
                    "type": "feedback_ack",
                    "feedback_text": text,
                    "feedback_active": bool(text),
                    "human_feedback_mode": self._feedback_reader.mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.info(f"Human feedback set from dashboard ({len(text)} chars)")
            else:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "feedback_ack",
                            "feedback_text": "",
                            "feedback_active": False,
                            "error": "Human feedback not enabled",
                        }
                    ),
                )
        elif t == "clear_feedback":
            if self._feedback_reader:
                self._feedback_reader.write_from_dashboard("")
                ack = {
                    "type": "feedback_ack",
                    "feedback_text": "",
                    "feedback_active": False,
                    "human_feedback_mode": self._feedback_reader.mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.info("Human feedback cleared from dashboard")
        elif t == "request_feedback_state":
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "feedback_ack",
                        **self._get_feedback_state(),
                    }
                ),
            )
        elif t == "set_human_feedback_mode":
            mode = msg.get("mode", "append")
            if self._feedback_reader:
                self._feedback_reader.set_mode(mode)
                ack = {
                    "type": "human_feedback_mode_ack",
                    "human_feedback_mode": mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.info(f"Human feedback mode set to: {mode}")
        elif t == "request_system_prompt":
            prompt_text = ""
            if self._feedback_reader:
                prompt_text = self._feedback_reader.get_current_prompt()
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "system_prompt",
                        "prompt_text": prompt_text,
                    }
                ),
            )
        elif t == "request_human_feedback_history":
            history = []
            if self._feedback_reader:
                history = self._feedback_reader.get_history()
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "human_feedback_history",
                        "history": history,
                    }
                ),
            )
        elif t == "request_image":
            image_path = msg.get("image_path", "")
            program_id = msg.get("program_id", "")
            if image_path and os.path.exists(image_path):
                try:
                    import base64 as _b64

                    with open(image_path, "rb") as _f:
                        img_data = _b64.b64encode(_f.read()).decode()
                    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
                    mime = {
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "webp": "image/webp",
                        "gif": "image/gif",
                    }.get(ext, "image/png")
                    await self._ws_send(
                        writer,
                        json.dumps(
                            {
                                "type": "image_data",
                                "program_id": program_id,
                                "data_url": f"data:{mime};base64,{img_data}",
                            }
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to serve image {image_path}: {e}")
        elif t == "request_program_summary":
            pid = msg.get("program_id", "")
            await self._generate_program_summary(writer, pid)
        elif t == "request_summary":
            await self._trigger_summary()
        elif t == "request_stop_run":
            if self._stop_handler is None:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "stop",
                            "ok": False,
                            "message": "Graceful stop is not available in this monitor mode.",
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
                return

            try:
                self._stop_handler()
                await self._broadcast(
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "stop",
                            "ok": True,
                            "message": "Stop requested. Waiting for the current iteration to finish.",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    )
                )
            except Exception as exc:
                logger.warning("Failed to request graceful stop: %s", exc)
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "stop",
                            "ok": False,
                            "message": f"Failed to request graceful stop: {exc}",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
        elif t == "request_resume_run":
            if self._resume_handler is None:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "resume",
                            "ok": False,
                            "message": "Resume is not available in this monitor mode.",
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
                return

            try:
                self._resume_handler()
                await self._broadcast(
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "resume",
                            "ok": True,
                            "message": "Resume requested. Waiting for the runner to restart.",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    )
                )
            except Exception as exc:
                logger.warning("Failed to request resume: %s", exc)
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "resume",
                            "ok": False,
                            "message": f"Failed to request resume: {exc}",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )
                return
        elif t == "request_branch_from_program":
            if self._branch_handler is None:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "branch_ack",
                            "ok": False,
                            "message": "Candidate-guided branching is not available in this monitor mode.",
                            "branch_capabilities": self._get_branch_capabilities(),
                        }
                    ),
                )
                return

            try:
                result = self._branch_handler(msg) or {}
                payload = {
                    "type": "branch_ack",
                    "ok": True,
                    "message": result.get(
                        "message",
                        "Branch requested. Preparing a candidate-seeded run.",
                    ),
                    "branch": result.get("branch", {}),
                    "branch_capabilities": self._get_branch_capabilities(),
                }
                await self._broadcast(json.dumps(payload))
            except Exception as exc:
                logger.warning("Failed to request branch: %s", exc)
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "branch_ack",
                            "ok": False,
                            "message": f"Failed to create branch: {exc}",
                            "branch_capabilities": self._get_branch_capabilities(),
                        }
                    ),
                )
                return

            try:
                if self._resume_handler is None:
                    raise RuntimeError("Resume is not available in this monitor mode.")
                self._resume_handler()
                await self._broadcast(
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "resume",
                            "ok": True,
                            "message": "Resume requested. Waiting for the runner to restart.",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    )
                )
            except Exception as exc:
                logger.warning("Failed to request resume: %s", exc)
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "run_control_ack",
                            "action": "resume",
                            "ok": False,
                            "message": f"Failed to request resume: {exc}",
                            "run_state": self._get_run_state(),
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    ),
                )

    # ─── Queue consumer & broadcast ──────────────────────────

    async def _consume_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            etype = event.get("type")
            if etype == "reset_state":
                self._reset_cached_state(clear_config_summary=False)
                await self._broadcast(json.dumps(self._build_init_state()))
                continue
            elif etype == "replace_state":
                self._replace_cached_state(event)
                await self._broadcast(json.dumps(self._build_init_state()))
                continue
            elif etype == "new_program":
                active_run_label = self._get_run_label()
                if not self._event_matches_active_run(event, active_run_label=active_run_label):
                    stale_program = event.get("program", {})
                    stale_run_label = self._get_program_run_label(stale_program) or event.get(
                        "run_label"
                    )
                    logger.warning(
                        "Dropping stale monitor event for run '%s' while active run is '%s'.",
                        stale_run_label,
                        active_run_label,
                    )
                    continue
                self._last_event_at = time.time()
                p = event.get("program", {})
                # Annotate with human feedback state for replay on reconnect
                if self._feedback_reader:
                    fb = self._feedback_reader.read()
                    p["human_feedback_active"] = bool(fb)
                else:
                    p["human_feedback_active"] = False
                p.setdefault("series_label", self._get_series_label())
                p.setdefault("run_label", self._get_run_label())
                pid = p.get("id", "")
                replaced = False
                if pid:
                    for index, existing in enumerate(self._programs):
                        if isinstance(existing, dict) and existing.get("id") == pid:
                            self._programs[index] = p
                            replaced = True
                            break
                if not replaced:
                    self._programs.append(p)
                if "full_solution" in event:
                    self._program_solutions[pid] = event["full_solution"]
                if "parent_full_solution" in event:
                    self._parent_solutions[pid] = event["parent_full_solution"]
                # Independent best tracking: compare scores directly
                new_score = p.get("score", 0)
                if not isinstance(new_score, (int, float)):
                    new_score = 0
                if new_score > self._best_score:
                    self._best_score = new_score
                    self._best_program_id = pid
                    event["is_best"] = True
                elif event.get("is_best"):
                    self._best_program_id = pid
                    self._best_score = max(self._best_score, new_score)
                self._stats = event.get("stats", self._stats)
            elif etype == "program_solution":
                pid = event.get("program_id", "")
                if pid:
                    self._program_solutions[pid] = event.get("solution", "")
                    self._parent_solutions[pid] = event.get("parent_solution", "")
            elif etype == "program_summary":
                pid = event.get("program_id", "")
                if pid:
                    self._program_summary_cache[pid] = event.get("summary", "")
            elif etype == "feedback_ack":
                self._feedback_state_cache["human_feedback_enabled"] = event.get(
                    "human_feedback_enabled",
                    self._feedback_state_cache.get("human_feedback_enabled", True),
                )
                self._feedback_state_cache["feedback_text"] = event.get(
                    "feedback_text",
                    self._feedback_state_cache.get("feedback_text", ""),
                )
                self._feedback_state_cache["feedback_active"] = event.get(
                    "feedback_active",
                    self._feedback_state_cache.get("feedback_active", False),
                )
                self._feedback_state_cache["human_feedback_mode"] = event.get(
                    "human_feedback_mode",
                    self._feedback_state_cache.get("human_feedback_mode", "append"),
                )
            elif etype == "human_feedback_mode_ack":
                self._feedback_state_cache["human_feedback_mode"] = event.get(
                    "human_feedback_mode",
                    self._feedback_state_cache.get("human_feedback_mode", "append"),
                )
            elif etype == "system_prompt":
                self._feedback_state_cache["human_feedback_current_prompt"] = event.get(
                    "prompt_text",
                    self._feedback_state_cache.get("human_feedback_current_prompt", ""),
                )
            elif etype == "human_feedback_history":
                self._feedback_state_cache["human_feedback_history"] = event.get("history", [])
            elif etype == "summary_update":
                self._summary_text = event.get("summary_text", self._summary_text)
                self._summary_generating = event.get(
                    "summary_generating", self._summary_generating
                )
                if "summary_enabled" in event and not event.get("summary_enabled"):
                    self._summary_model = ""
            elif etype == "discovery_complete":
                self._last_event_at = time.time()

            # Strip full_solution from broadcast (clients request on demand)
            broadcast = {
                k: v for k, v in event.items() if k not in ("full_solution", "parent_full_solution")
            }
            broadcast["run_state"] = self._get_run_state()
            broadcast["run_label"] = self._get_run_label()
            broadcast["series_label"] = self._get_series_label(broadcast["run_state"])
            broadcast["last_event_at"] = self._last_event_at
            broadcast["control_capabilities"] = self._get_control_capabilities(
                broadcast["run_state"]
            )
            # Include current human feedback status in program events
            if etype == "new_program" and self._feedback_reader:
                fb = self._feedback_reader.read()
                broadcast["feedback_active"] = bool(fb)
                broadcast["feedback_text"] = fb if fb else ""
                broadcast["human_feedback_mode"] = self._feedback_reader.mode
            await self._broadcast(json.dumps(broadcast))

            # Auto-trigger AI summary every N new programs
            if (
                etype == "new_program"
                and self._summary_interval > 0
                and self._summary_model
                and not self._summary_generating
            ):
                count = len(self._programs)
                if count - self._summary_last_program_count >= self._summary_interval:
                    await self._trigger_summary()

    def _reset_cached_state(self, *, clear_config_summary: bool = False) -> None:
        """Clear cached run data while keeping dashboard services alive."""
        self._programs = []
        self._program_solutions = {}
        self._parent_solutions = {}
        self._trace_records_by_id = {}
        self._best_program_id = None
        self._best_score = -float("inf")
        self._stats = {}
        self._program_summary_cache = {}
        self._summary_model = ""
        self._summary_text = ""
        self._summary_generating = False
        self._feedback_state_cache = {
            "human_feedback_enabled": False,
            "feedback_text": "",
            "feedback_active": False,
            "human_feedback_mode": "append",
            "human_feedback_current_prompt": "",
            "human_feedback_history": [],
        }
        self._last_event_at = None
        if clear_config_summary:
            self._config_summary = ""

    def _replace_cached_state(self, state: Dict[str, Any]) -> None:
        """Replace cached monitor state from an external snapshot."""
        self._programs = self._filter_programs_for_active_run(list(state.get("programs", [])))
        self._program_solutions = dict(state.get("program_solutions", {}))
        self._parent_solutions = dict(state.get("parent_solutions", {}))
        self._trace_records_by_id = dict(state.get("trace_records_by_id", {}))
        self._best_program_id = state.get("best_program_id")
        self._stats = dict(state.get("stats", {}))
        self._last_event_at = state.get("last_event_at", self._last_event_at)
        if "config_summary" in state:
            self._config_summary = state.get("config_summary", self._config_summary)
        self._summary_model = state.get("summary_model", self._summary_model or "")
        self._summary_text = state.get("summary_text", self._summary_text)
        self._summary_generating = state.get("summary_generating", self._summary_generating)
        self._feedback_state_cache = {
            "human_feedback_enabled": state.get("human_feedback_enabled", False),
            "feedback_text": state.get("feedback_text", ""),
            "feedback_active": state.get("feedback_active", False),
            "human_feedback_mode": state.get("human_feedback_mode", "append"),
            "human_feedback_current_prompt": state.get("human_feedback_current_prompt", ""),
            "human_feedback_history": state.get("human_feedback_history", []),
        }

        valid_program_ids = {
            program.get("id") for program in self._programs if isinstance(program, dict) and program.get("id")
        }
        if self._best_program_id not in valid_program_ids:
            self._best_program_id = None

        self._best_score = -float("inf")
        best_program_id: Optional[str] = self._best_program_id
        for program in self._programs:
            score = program.get("score", 0)
            if isinstance(score, (int, float)) and score > self._best_score:
                self._best_score = score
                best_program_id = program.get("id")
        if self._best_score == -float("inf"):
            self._best_score = 0.0
        self._best_program_id = best_program_id

    def _get_program_run_label(self, program: Any) -> Optional[str]:
        """Return the run label embedded in a program payload, if any."""
        if not isinstance(program, dict):
            return None
        run_label = program.get("run_label")
        if isinstance(run_label, str) and run_label:
            return run_label
        return None

    def _event_matches_active_run(
        self, event: Dict[str, Any], *, active_run_label: Optional[str] = None
    ) -> bool:
        """Return ``True`` when an event belongs to the currently active run."""
        current_run_label = active_run_label or self._get_run_label()
        if not current_run_label:
            return True

        program = event.get("program")
        event_run_label = self._get_program_run_label(program)
        if event_run_label is None:
            raw_run_label = event.get("run_label")
            if isinstance(raw_run_label, str) and raw_run_label:
                event_run_label = raw_run_label

        if event_run_label is None:
            return True
        return event_run_label == current_run_label

    def _filter_programs_for_active_run(self, programs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only programs that belong to the active run label."""
        current_run_label = self._get_run_label()
        if not current_run_label:
            return [program for program in programs if isinstance(program, dict)]

        filtered: List[Dict[str, Any]] = []
        dropped = 0
        for program in programs:
            if not isinstance(program, dict):
                continue
            program_run_label = self._get_program_run_label(program)
            if program_run_label and program_run_label != current_run_label:
                dropped += 1
                continue
            filtered.append(program)

        if dropped:
            logger.warning(
                "Dropped %s stale program(s) while rebuilding monitor state for run '%s'.",
                dropped,
                current_run_label,
            )
        return filtered

    async def _broadcast(self, message: str) -> None:
        if not self._clients:
            return
        dead = set()
        for writer in list(self._clients):
            try:
                await self._ws_send(writer, message)
            except Exception:
                dead.add(writer)
        self._clients -= dead

    async def _ws_send(self, writer: asyncio.StreamWriter, text: str) -> None:
        writer.write(_ws_encode_text(text))
        await writer.drain()

    async def _heartbeat(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(5)
            if self._clients:
                await self._broadcast(
                    json.dumps(
                        {
                            "type": "heartbeat",
                            "timestamp": time.time(),
                            "run_state": self._get_run_state(),
                            "last_event_at": self._last_event_at,
                            "control_capabilities": self._get_control_capabilities(),
                        }
                    )
                )

    async def _generate_program_summary(self, writer: asyncio.StreamWriter, pid: str) -> None:
        """Generate a crisp LLM summary of what changed in a single program."""
        # Return cached if available
        if pid in self._program_summary_cache:
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_summary",
                        "program_id": pid,
                        "summary": self._program_summary_cache[pid],
                    }
                ),
            )
            return

        # Need API key + model
        if not self._summary_model or not self._summary_api_key:
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_summary",
                        "program_id": pid,
                        "summary": "AI summary not configured.",
                    }
                ),
            )
            return

        # Find program data
        prog = None
        for p in self._programs:
            if p.get("id") == pid:
                prog = p
                break
        if not prog:
            return

        # Build prompt
        code = self._program_solutions.get(pid, prog.get("solution_snippet", ""))
        parent_solution = self._parent_solutions.get(pid, "")
        score = prog.get("score", "?")
        parent_score = prog.get("parent_score")
        label = prog.get("label_type", "unknown")

        delta_str = ""
        if isinstance(score, (int, float)) and isinstance(parent_score, (int, float)):
            d = score - parent_score
            delta_str = f" (delta: {'+' if d >= 0 else ''}{d:.4f})"

        # Truncate code for prompt efficiency
        if len(code) > 2000:
            code = code[:2000] + "\n... (truncated)"
        if len(parent_solution) > 2000:
            parent_solution = parent_solution[:2000] + "\n... (truncated)"

        is_image_mode = prog.get("image_path") is not None

        if is_image_mode:
            system = (
                "You are analyzing one step in an image generation run. "
                "Given the parent generation prompt and the child generation prompt, describe in 1-2 concise bullet points "
                "what specifically changed in the prompt.\n\n"
                "Rules:\n"
                "- Be specific: name style changes, subject modifications, added details\n"
                "- Each bullet under 25 words\n"
                "- Start each bullet with `- `\n"
                "- No headers, no sections — just 1-2 bullets"
            )
        else:
            system = (
                "You are analyzing one step in a solution discovery run. "
                "Given the parent code and the child code, describe in 1-2 concise bullet points "
                "what specifically changed.\n\n"
                "Rules:\n"
                "- Be specific: name algorithms, parameters, structural changes\n"
                "- Each bullet under 25 words\n"
                "- Start each bullet with `- `\n"
                "- No headers, no sections — just 1-2 bullets\n"
                "- Consider the evolution label: exploration = trying new ideas, "
                "exploitation = refining current best, diverge = deliberately different strategy"
            )

        content_label = "prompt" if is_image_mode else "code"
        user_parts = [f"Label: {label}{delta_str}"]
        if parent_score is not None:
            user_parts.append(f"Score: {parent_score} -> {score}")
        else:
            user_parts.append(f"Score: {score} (no parent)")

        if parent_solution:
            user_parts.append(f"\nParent {content_label}:\n```\n{parent_solution}\n```")
        user_parts.append(f"\nNew {content_label}:\n```\n{code}\n```")

        prompt_data = {"system": system, "user": "\n".join(user_parts)}

        # Ensure executor exists
        if not self._summary_executor:
            self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")

        # Run LLM call in executor
        result = ""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._summary_executor,
                self._call_program_summary_api,
                prompt_data,
            )
            self._program_summary_cache[pid] = result
        except Exception as e:
            logger.warning(f"Program summary failed for {pid[:8]}: {e}", exc_info=True)
            result = f"Summary unavailable: {e}"

        await self._ws_send(
            writer,
            json.dumps(
                {
                    "type": "program_summary",
                    "program_id": pid,
                    "summary": result or "Summary unavailable (empty response).",
                }
            ),
        )

    def _call_program_summary_api(self, prompt_data: Dict[str, str]) -> str:
        """Call LLM for per-program summary (blocking, runs in executor)."""
        return self._call_llm_api(prompt_data, max_tokens=2048, timeout=120)

    async def _trigger_summary(self) -> None:
        """Trigger async AI summary generation."""
        if not self._summary_model:
            await self._broadcast(
                json.dumps(
                    {
                        "type": "summary_update",
                        "summary_text": "AI summary not configured (no model set).",
                        "summary_generating": False,
                        "summary_enabled": False,
                    }
                )
            )
            return
        if not self._summary_api_key:
            await self._broadcast(
                json.dumps(
                    {
                        "type": "summary_update",
                        "summary_text": "AI summary not configured. Set OPENAI_API_KEY environment variable or summary_api_key in config.",
                        "summary_generating": False,
                        "summary_enabled": False,
                    }
                )
            )
            return
        if self._summary_generating:
            return  # Already in progress

        # Ensure executor exists
        if not self._summary_executor:
            self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")

        self._summary_generating = True
        self._summary_last_program_count = len(self._programs)

        # Notify clients that generation started
        await self._broadcast(
            json.dumps(
                {
                    "type": "summary_update",
                    "summary_text": self._summary_text,
                    "summary_generating": True,
                    "summary_enabled": True,
                }
            )
        )

        try:
            # Build the prompt data from current programs
            top_programs = self._get_top_k_programs()
            if not top_programs:
                self._summary_text = "No scored programs yet. Run some iterations first."
                logger.info("AI summary skipped: no scored programs")
            else:
                prompt_data = self._build_summary_prompt(top_programs)
                logger.info(
                    f"AI summary: calling {self._summary_model} with {len(top_programs)} "
                    f"top programs, api_base={self._summary_api_base}"
                )

                # Run the blocking API call in a thread
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._summary_executor,
                    self._call_llm_api,
                    prompt_data,
                )
                self._summary_text = result or "AI returned empty response."
                logger.info(f"AI summary generated ({len(self._summary_text)} chars)")
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}", exc_info=True)
            self._summary_text = f"Summary generation failed: {e}"
        finally:
            self._summary_generating = False

        # Broadcast the result
        await self._broadcast(
            json.dumps(
                {
                    "type": "summary_update",
                    "summary_text": self._summary_text,
                    "summary_generating": False,
                    "summary_enabled": True,
                }
            )
        )

    def _get_top_k_programs(self) -> List[Dict[str, Any]]:
        """Get top-k programs by score across all islands."""
        if not self._programs:
            return []
        scored = [p for p in self._programs if isinstance(p.get("score"), (int, float))]
        scored.sort(key=lambda p: p["score"], reverse=True)

        # Deduplicate by score (keep best per unique score to show diversity)
        seen_scores = set()
        unique = []
        for p in scored:
            key = round(p["score"], 6)
            if key not in seen_scores:
                seen_scores.add(key)
                unique.append(p)
            if len(unique) >= self._summary_top_k:
                break
        # Fall back to just top-k if not enough unique
        if len(unique) < self._summary_top_k:
            unique = scored[: self._summary_top_k]
        return unique

    def _compute_solution_discovery_analysis(self) -> str:
        """Compute evolution progress, improvement patterns, and stagnation analysis."""
        programs = self._programs
        if not programs:
            return ""

        scored = [p for p in programs if isinstance(p.get("score"), (int, float))]
        if not scored:
            return ""

        lines = []
        n = len(scored)

        improvements = 0
        regressions = 0
        total_with_parent = 0
        improvement_deltas = []
        for p in scored:
            parent_score = p.get("parent_score")
            if isinstance(parent_score, (int, float)):
                total_with_parent += 1
                delta = p["score"] - parent_score
                if delta > 0:
                    improvements += 1
                    improvement_deltas.append(delta)
                elif delta < 0:
                    regressions += 1

        if total_with_parent > 0:
            hit_rate = improvements / total_with_parent * 100
            avg_gain = (
                sum(improvement_deltas) / len(improvement_deltas) if improvement_deltas else 0
            )
            lines.append("=== Improvement Rate ===")
            lines.append(
                f"  {improvements}/{total_with_parent} programs improved over parent ({hit_rate:.0f}% hit rate)"
            )
            lines.append(f"  Avg improvement when positive: {avg_gain:+.4f}")

        if n >= 10:
            quarter = max(n // 4, 1)
            early_scores = [p["score"] for p in scored[:quarter]]
            mid_scores = [p["score"] for p in scored[quarter : quarter * 2]]
            recent_scores = [p["score"] for p in scored[-quarter:]]
            early_avg = sum(early_scores) / len(early_scores)
            mid_avg = sum(mid_scores) / len(mid_scores) if mid_scores else early_avg
            recent_avg = sum(recent_scores) / len(recent_scores)

            lines.append("\n=== Score Trend ===")
            lines.append(
                f"  Early avg (first {quarter}): {early_avg:.4f}  |  "
                f"Mid avg: {mid_avg:.4f}  |  "
                f"Recent avg (last {quarter}): {recent_avg:.4f}"
            )
            if recent_avg > mid_avg + 0.001:
                lines.append("  Trend: IMPROVING")
            elif recent_avg < mid_avg - 0.005:
                lines.append("  Trend: REGRESSING")
            elif abs(recent_avg - mid_avg) < 0.001 and n > 30:
                lines.append("  Trend: PLATEAUED")
            else:
                lines.append("  Trend: STABLE")

        if n >= 5:
            best_so_far = -float("inf")
            streak = 0
            longest_streak = 0
            for p in scored:
                if p["score"] > best_so_far:
                    best_so_far = p["score"]
                    streak = 0
                else:
                    streak += 1
                    longest_streak = max(longest_streak, streak)
            lines.append("\n=== Stagnation ===")
            lines.append(
                f"  Current non-improving streak: {streak} iterations  |  "
                f"Longest streak: {longest_streak}"
            )

        islands: Dict[Any, list] = {}
        for p in scored:
            isl = p.get("island")
            if isl is not None:
                islands.setdefault(isl, []).append(p["score"])
        if len(islands) > 1:
            lines.append(f"\n=== Island Diversity ({len(islands)} islands) ===")
            for isl in sorted(islands.keys()):
                scores = islands[isl]
                lines.append(
                    f"  Island {isl}: {len(scores)} programs, "
                    f"best={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}"
                )

        return "\n".join(lines)

    def _build_summary_prompt(self, top_programs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build the system + user prompt for the summary LLM call."""
        system = (
            "You are an expert analyst monitoring a solution discovery process. "
            "You will be given run statistics, evolution progress data, and the source code "
            "of the top-performing programs from the current run.\n\n"
            "Respond using EXACTLY this markdown structure:\n\n"
            "## Status\n"
            "One sentence: is the search improving, stagnating, or plateauing? "
            "Cite the score trend numbers.\n\n"
            "## Key Techniques\n"
            "Bullet list of the main algorithmic ideas found in the top programs' code. "
            "Be specific — name the techniques (e.g. 'Kalman filter with adaptive Q', "
            "'hexagonal lattice packing', 'exponential moving average').\n\n"
            "## Diversity\n"
            "Are the top programs converging on one approach or exploring different strategies? "
            "One sentence.\n\n"
            "## Recommendation\n"
            "One specific, actionable suggestion grounded in the code. "
            "For example: **try wavelet denoising** — the top programs all use simple "
            "moving averages which limits frequency response.\n\n"
            "Rules:\n"
            "- Use markdown: **bold** for key terms, `- ` for bullets, `##` for sections\n"
            "- Be concise — max 250 words total\n"
            "- Every claim must reference what you see in the actual code"
        )

        # Build user message with stats + solution discovery analysis + top-k programs
        parts = []
        if self._stats:
            parts.append(
                f"Run: {self._config_summary}\n"
                f"Total programs: {self._stats.get('total_programs', len(self._programs))}\n"
                f"Current iteration: {self._stats.get('current_iteration', '?')}\n"
                f"Best score: {self._stats.get('best_score', '?')}\n"
                f"Programs/min: {self._stats.get('programs_per_min', '?')}\n"
                f"Elapsed: {self._stats.get('elapsed_seconds', '?')}s\n"
                f"Iterations since improvement: {self._stats.get('iterations_since_improvement', '?')}"
            )

        # Add solution discovery analysis
        solution_discovery_analysis = self._compute_solution_discovery_analysis()
        if solution_discovery_analysis:
            parts.append(f"\n{solution_discovery_analysis}")

        for i, p in enumerate(top_programs, 1):
            pid = p.get("id", "?")
            code = self._program_solutions.get(pid, p.get("solution_snippet", ""))
            # Truncate code to keep prompt reasonable
            if len(code) > 2000:
                code = code[:2000] + "\n... (truncated)"
            island_str = f", island={p.get('island')}" if p.get("island") is not None else ""
            parts.append(
                f"\n--- Top Program #{i} ---\n"
                f"ID: {pid}\n"
                f"Score: {p.get('score', '?')}\n"
                f"Iteration: {p.get('iteration', '?')}{island_str}\n"
                f"Metrics: {json.dumps(p.get('metrics', {}))}\n"
                f"Code:\n{code}"
            )

        return {"system": system, "user": "\n".join(parts)}

    def _call_llm_api(
        self, prompt_data: Dict[str, str], max_tokens: int = 8192, timeout: int = 180
    ) -> str:
        """Call OpenAI-compatible API (blocking, runs in executor thread)."""
        url = f"{self._summary_api_base}/chat/completions"
        body = json.dumps(
            {
                "model": self._summary_model,
                "messages": [
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]},
                ],
                "max_completion_tokens": max_tokens,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._summary_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"API error {e.code}: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e
