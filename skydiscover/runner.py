import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from skydiscover.config import Config, build_output_dir, load_config, resolve_iteration_budget
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.registry import create_database, get_program
from skydiscover.search.route import get_discovery_controller
from skydiscover.search.utils.logging_utils import setup_search_logging
from skydiscover.utils.code_utils import extract_solution_language
from skydiscover.utils.metrics import format_metrics, get_score

logger = logging.getLogger(__name__)


class Runner:
    """Top-level entry point for a discovery run.

    Loads config, creates the database and discovery controller, runs the
    search loop, and saves checkpoints + best program.

    Args:
        initial_program_path: path to the starting solution file.
        evaluation_file: path to the user's evaluator script (must define evaluate()).
        config_path: optional YAML config file (ignored if config is provided).
        config: optional pre-built Config object (takes priority over config_path).
        output_dir: where to write logs, checkpoints, and best program.
            Auto-generated from search type + problem name if omitted.
    """

    def __init__(
        self,
        evaluation_file: str,
        initial_program_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config if config is not None else load_config(config_path)
        self.name = self.config.search.type
        self.output_dir = output_dir or build_output_dir(
            self.name, initial_program_path or "scratch"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

        # Load the initial program (can be optional)
        self.initial_program_path = initial_program_path
        self.initial_program_solution = (
            self._load_initial_program() if initial_program_path else None
        )
        if self.initial_program_solution and not self.config.language:
            self.config.language = extract_solution_language(self.initial_program_solution)
        if not self.config.language:
            self.config.language = "python"

        # Set the file extension
        ext = os.path.splitext(initial_program_path)[1] if initial_program_path else ".py"
        ext = ext or ".py"
        self.file_extension = ext if ext.startswith(".") else f".{ext}"
        if self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension

        # Create the database
        self.database = create_database(self.config.search.type, self.config.search.database)
        self.database.language = self.config.language or "python"
        self.evaluation_file = evaluation_file

        # Initialize the discovery controller
        self.discovery_controller: Optional[DiscoveryController] = None
        self._monitor_pause_requested = False
        self._signal_shutdown_requested = False
        self._resume_requested = threading.Event()

        logger.info(f"Runner ready: search={self.name}, program={self.initial_program_path}")

    @property
    def initial_score(self) -> Optional[float]:
        """Score of the seed program, or None if unavailable."""
        if not self.database or not self.database.programs or not self.initial_program_solution:
            return None

        seed_solution = self.initial_program_solution
        seed_prog = None
        for prog in self.database.programs.values():
            if prog.solution == seed_solution:
                seed_prog = prog
                break
        if seed_prog is None:
            for prog in self.database.programs.values():
                if prog.iteration_found == 0:
                    seed_prog = prog
                    break

        if seed_prog and seed_prog.metrics:
            return get_score(seed_prog.metrics)
        return None

    async def run(
        self,
        iterations: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        run_forever: bool = False,
    ) -> Optional[Program]:
        """Entrypoint for the discovery process.

        Args:
            iterations: max iterations (uses config.max_iterations if None).
            checkpoint_path: resume from this checkpoint directory if provided.
            run_forever: run indefinitely until interrupted or early stopping triggers.

        Returns:
            Best Program found, or None if no valid programs were produced.
        """
        max_iterations = resolve_iteration_budget(
            self.config,
            iterations=iterations,
            run_forever=run_forever,
        )

        start_iteration = 0
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Create the discovery controller input
        controller_input = DiscoveryControllerInput(
            config=self.config,
            evaluation_file=self.evaluation_file,
            database=self.database,
            file_suffix=self.config.file_suffix,
            output_dir=self.output_dir,
        )

        # Get the discovery controller
        self.discovery_controller = get_discovery_controller(controller_input)

        # Add initial program to database if not resuming
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and self.initial_program_solution is not None
        )

        if should_add_initial:
            await self._add_initial_program(start_iteration)
        else:
            logger.info(
                f"Resuming from iteration {start_iteration} with {len(self.database.programs)} programs"
            )

        discovery_start = start_iteration + 1 if should_add_initial else start_iteration
        remaining_iterations = self._compute_remaining_iteration_budget(
            max_iterations,
            discovery_start,
        )

        # Start the monitor
        monitor_server = None
        early_stopped = False
        latest_checkpoint: Optional[str] = checkpoint_path
        final_status = "failed"
        try:
            monitor_server = self._start_monitor(max_iterations)
            self._setup_human_feedback(monitor_server)
            self._setup_monitor_summary(monitor_server)
            self._push_existing_to_monitor()
            self._install_signal_handlers()
            self.database.log_status()

            if remaining_iterations == 0:
                logger.info("Iteration budget already exhausted; skipping discovery loop")
                final_status = "completed"
            else:
                while True:
                    cycle_start = discovery_start
                    cycle_budget = remaining_iterations
                    completed_iterations_seen = set()

                    if self.discovery_controller is not None:
                        self.discovery_controller.shutdown_event.clear()
                    self._monitor_pause_requested = False
                    self._resume_requested.clear()

                    self._write_run_state(
                        "running",
                        max_iterations=max_iterations,
                        checkpoint_path=latest_checkpoint,
                        remaining_iterations=cycle_budget,
                    )
                    self._emit_monitor_state(monitor_server)

                    def progress_cb(iteration: int) -> None:
                        if iteration not in completed_iterations_seen:
                            completed_iterations_seen.add(iteration)
                        self._write_run_state(
                            "running",
                            max_iterations=max_iterations,
                            checkpoint_path=latest_checkpoint,
                            remaining_iterations=self._decrement_iteration_budget(
                                cycle_budget,
                                len(completed_iterations_seen),
                            ),
                        )
                        self._emit_monitor_state(monitor_server)

                    if self.discovery_controller is not None:
                        self.discovery_controller.progress_callback = progress_cb

                    def checkpoint_cb(iteration: int) -> None:
                        nonlocal latest_checkpoint
                        self._sync_database()
                        saved_path = self._save_checkpoint(iteration)
                        latest_checkpoint = saved_path
                        executed = len(completed_iterations_seen) + (
                            0 if iteration in completed_iterations_seen else 1
                        )
                        self._write_run_state(
                            "running",
                            max_iterations=max_iterations,
                            checkpoint_path=saved_path,
                            remaining_iterations=self._decrement_iteration_budget(
                                cycle_budget,
                                executed,
                            ),
                        )
                        self._emit_monitor_state(monitor_server)

                    if cycle_budget == 0:
                        logger.info("No remaining iteration budget for this run cycle")
                        final_status = "completed"
                        break

                    try:
                        await self.discovery_controller.run_discovery(
                            cycle_start,
                            cycle_budget,
                            checkpoint_callback=checkpoint_cb,
                        )
                    finally:
                        if self.discovery_controller is not None:
                            self.discovery_controller.progress_callback = None

                    self._sync_database()
                    if self.database.programs:
                        latest_checkpoint = self._save_checkpoint(self.database.last_iteration)

                    early_stopped = self.discovery_controller.early_stopping_triggered
                    shutdown_requested = self.discovery_controller.shutdown_event.is_set()
                    executed_iterations = self._count_executed_iterations(
                        cycle_start,
                        self.database.last_iteration,
                    )
                    remaining_iterations = self._decrement_iteration_budget(
                        cycle_budget,
                        executed_iterations,
                    )
                    budget_exhausted = (
                        remaining_iterations is not None and remaining_iterations == 0
                    )

                    if early_stopped:
                        final_status = "early_stopping"
                        break

                    if budget_exhausted:
                        final_status = "completed"
                        break

                    if shutdown_requested:
                        final_status = "paused"
                        self._write_run_state(
                            "paused",
                            max_iterations=max_iterations,
                            checkpoint_path=latest_checkpoint,
                            remaining_iterations=remaining_iterations,
                        )
                        self._emit_monitor_state(monitor_server)

                        if self._should_wait_for_monitor_resume(monitor_server):
                            logger.info(
                                "Discovery paused from monitor; waiting for a resume request..."
                            )
                            resumed = await self._wait_for_resume_request()
                            if resumed:
                                discovery_start = self.database.last_iteration + 1
                                logger.info(
                                    "Resume requested from monitor; continuing at iteration %s",
                                    discovery_start,
                                )
                                continue

                        break

                    final_status = "completed"
                    break

            if final_status in {"completed", "early_stopping"}:
                best = self._get_best_program()
                if best:
                    try:
                        test_result = await self.discovery_controller.evaluator.evaluate_program(
                            best.solution, best.id, mode="test"
                        )
                        for k, v in test_result.metrics.items():
                            best.metrics[f"test_{k}"] = v
                        logger.info(
                            f"Test evaluation for best program: {format_metrics(test_result.metrics)}"
                        )
                        self._save_best_program(best)
                    except Exception as e:
                        logger.warning(f"Test-mode re-evaluation failed: {e}")

            self._write_run_state(
                final_status,
                max_iterations=max_iterations,
                checkpoint_path=latest_checkpoint,
                remaining_iterations=remaining_iterations,
            )
            self._emit_monitor_state(monitor_server)
        except Exception as exc:
            self._sync_database()
            if self.database.programs:
                latest_checkpoint = self._save_checkpoint(self.database.last_iteration)
            self._write_run_state(
                "failed",
                max_iterations=max_iterations,
                checkpoint_path=latest_checkpoint,
                remaining_iterations=remaining_iterations,
                error=str(exc),
            )
            self._emit_monitor_state(monitor_server)
            raise

        finally:
            # Stop the monitor
            early_stopped = (
                self.discovery_controller is not None
                and self.discovery_controller.early_stopping_triggered
            ) or early_stopped
            if self.discovery_controller is not None:
                self.discovery_controller.close()
            self.discovery_controller = None

            if monitor_server:
                try:
                    reason = "early_stopping" if early_stopped else final_status
                    if final_status != "paused":
                        monitor_server.push_event({"type": "discovery_complete", "reason": reason})
                        time.sleep(0.5)
                    monitor_server.stop()
                except Exception:
                    logger.debug("Failed to stop monitor server", exc_info=True)

        # Get the best program
        best_program = self._get_best_program()
        if best_program:
            status = "early stopping" if early_stopped else final_status.replace("_", " ")
            logger.info(f"Discovery {status}. Best: {format_metrics(best_program.metrics)}")
            self._save_best_program(best_program)
            return best_program

        logger.warning("No valid programs found")
        return None

    # ------------------------------------------------------------------
    # Initial program
    # ------------------------------------------------------------------

    async def _add_initial_program(self, start_iteration: int) -> None:
        logger.info("Adding initial program to database")
        program_id = str(uuid.uuid4())

        initial_image_path = None
        if self.config.language == "image":
            logger.info("Generating initial image from seed text...")
            img_dir = os.path.join(self.output_dir, "generated_images")
            try:
                result = await self.discovery_controller.llms.generate(
                    system_message="Generate an image based on the following description. Also provide brief reasoning about your creative choices.",
                    messages=[{"role": "user", "content": self.initial_program_solution}],
                    image_output=True,
                    output_dir=img_dir,
                    program_id=program_id,
                )
                initial_image_path = result.image_path
                logger.info(f"Initial image: {initial_image_path}")
            except Exception as e:
                logger.warning(f"Failed to generate initial image: {e}")

        eval_input = (
            initial_image_path
            if self.config.language == "image" and initial_image_path
            else self.initial_program_solution
        )
        eval_result = await self.discovery_controller.evaluator.evaluate_program(
            eval_input, program_id
        )
        metrics = eval_result.metrics

        if not initial_image_path and isinstance(metrics.get("image_path"), str):
            initial_image_path = metrics.pop("image_path")

        program = get_program(
            self.config, self.initial_program_solution, program_id, metrics, start_iteration
        )
        program.artifacts = eval_result.artifacts

        if initial_image_path:
            program.metadata = program.metadata or {}
            program.metadata["image_path"] = initial_image_path

        self.database.add(program)
        try:
            self.database.initial_program_id = program.id
            self.database.initial_program_score = get_score(program.metrics or {})
        except Exception as e:
            logger.warning(f"Failed to set initial program score: {e}")

    # ------------------------------------------------------------------
    # Monitor and feedback setup
    # ------------------------------------------------------------------

    def _start_monitor(self, max_iterations: Optional[int]):
        if not self.config.monitor.enabled:
            return None
        try:
            from skydiscover.extras.monitor import MonitorServer, create_monitor_callback

            server = MonitorServer(
                host=self.config.monitor.host,
                port=self.config.monitor.port,
                max_solution_length=self.config.monitor.max_solution_length,
                output_dir=self.output_dir,
            )
            server.set_stop_handler(self._request_pause_from_monitor)
            server.set_resume_handler(self._request_resume_from_monitor)
            server.set_solution_provider(self._resolve_monitor_solution)
            budget_label = "unbounded" if max_iterations is None else str(max_iterations)
            server.set_config_summary(f"{self.name} | max_iter={budget_label}")
            server.start()

            callback = create_monitor_callback(server, self.database, time.time())
            self.discovery_controller.monitor_callback = callback

            url = f"http://localhost:{server.port}/"
            print(f"\n  Live monitor: {url}\n", flush=True)
            logger.info(f"Live monitor: {url}")
            return server
        except Exception as e:
            logger.warning(f"Failed to start monitor: {e}")
            return None

    def _resolve_monitor_solution(self, program_id: str) -> Tuple[str, str]:
        program = self.database.get(program_id)
        if program is None:
            return "", ""

        solution = getattr(program, "solution", "") or ""
        parent_solution = ""
        parent_id = getattr(program, "parent_id", None)
        if parent_id:
            parent_program = self.database.get(parent_id)
            if parent_program is not None:
                parent_solution = getattr(parent_program, "solution", "") or ""

        return solution, parent_solution

    def _setup_human_feedback(self, monitor_server) -> None:
        if not (self.config.human_feedback_enabled or monitor_server):
            return
        try:
            from skydiscover.context_builder import HumanFeedbackReader

            path = self.config.human_feedback_file or os.path.join(
                self.output_dir, "human_feedback.md"
            )
            mode = getattr(self.config, "human_feedback_mode", "append")
            reader = HumanFeedbackReader(path, mode=mode)
            self.discovery_controller.feedback_reader = reader
            if monitor_server:
                monitor_server.set_feedback_reader(reader)
            logger.info(f"Human feedback: {path}")
        except Exception as e:
            logger.warning(f"Failed to set up human feedback: {e}")

    def _setup_monitor_summary(self, monitor_server) -> None:
        if not (monitor_server and self.config.monitor.summary_model):
            return
        try:
            monitor_server.configure_summary(
                model=self.config.monitor.summary_model,
                api_key=self.config.monitor.summary_api_key or "",
                api_base=self.config.monitor.summary_api_base,
                top_k=self.config.monitor.summary_top_k,
                interval=self.config.monitor.summary_interval,
            )
        except Exception as e:
            logger.warning(f"Failed to configure AI summary: {e}")

    def _push_existing_to_monitor(self) -> None:
        if not (self.discovery_controller.monitor_callback and self.database.programs):
            return
        emit_once = getattr(self.discovery_controller, "_emit_program_to_monitor", None)
        for prog in self.database.programs.values():
            try:
                if callable(emit_once):
                    emit_once(prog, getattr(prog, "iteration_found", 0))
                else:
                    self.discovery_controller.monitor_callback(
                        prog, getattr(prog, "iteration_found", 0)
                    )
            except Exception:
                logger.debug("Monitor callback failed for program %s", prog.id, exc_info=True)
        logger.info(f"Pushed {len(self.database.programs)} existing program(s) to monitor")

    def _install_signal_handlers(self) -> None:
        def on_signal(signum, frame):
            logger.info(f"Signal {signum} received, shutting down...")
            self._signal_shutdown_requested = True
            self._resume_requested.set()
            if self.discovery_controller is not None:
                self.discovery_controller.request_shutdown()

            def force_exit(signum, frame):
                sys.exit(128 + signum)

            # After the first termination signal, ensure subsequent SIGINT/SIGTERM
            # cause an immediate exit instead of re-running the soft handler.
            signal.signal(signal.SIGINT, force_exit)
            signal.signal(signal.SIGTERM, force_exit)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    # ------------------------------------------------------------------
    # Checkpointing and saving
    # ------------------------------------------------------------------

    def _sync_database(self) -> None:
        """Ensure we have the controller's latest database"""
        db = getattr(self.discovery_controller, "database", None)
        if db is not None and db is not self.database:
            self.database = db

    def _request_pause_from_monitor(self) -> None:
        """Request a graceful pause from the live monitor."""
        self._monitor_pause_requested = True
        if self.discovery_controller is not None:
            self.discovery_controller.request_shutdown()

    def _request_resume_from_monitor(self) -> None:
        """Resume a paused monitor-controlled run."""
        self._resume_requested.set()

    def _should_wait_for_monitor_resume(self, monitor_server) -> bool:
        """Return True when a monitor-triggered pause should stay interactive."""
        return (
            monitor_server is not None
            and self._monitor_pause_requested
            and not self._signal_shutdown_requested
        )

    async def _wait_for_resume_request(self) -> bool:
        """Wait until the monitor asks to resume or a signal asks us to exit."""
        while not self._resume_requested.is_set():
            await asyncio.sleep(0.1)

        should_resume = not self._signal_shutdown_requested
        self._resume_requested.clear()
        self._monitor_pause_requested = False
        if self.discovery_controller is not None:
            self.discovery_controller.shutdown_event.clear()
        return should_resume

    def _compute_remaining_iteration_budget(
        self,
        max_iterations: Optional[int],
        discovery_start: int,
    ) -> Optional[int]:
        """Return the remaining bounded budget from a given discovery start point."""
        if max_iterations is None:
            return None

        has_initial_program = bool(self.initial_program_solution) or bool(
            getattr(self.database, "initial_program_id", None)
        )
        completed_iterations = max(0, discovery_start - (1 if has_initial_program else 0))
        return max(0, max_iterations - completed_iterations)

    def _count_executed_iterations(self, cycle_start: int, last_iteration: int) -> int:
        """Count completed iterations within the current cycle."""
        if last_iteration < cycle_start:
            return 0
        return last_iteration - cycle_start + 1

    def _decrement_iteration_budget(
        self,
        remaining_iterations: Optional[int],
        executed_iterations: int,
    ) -> Optional[int]:
        """Decrease a bounded iteration budget by executed work."""
        if remaining_iterations is None:
            return None
        return max(0, remaining_iterations - max(0, executed_iterations))

    def _emit_monitor_state(self, monitor_server) -> None:
        """Prompt the monitor UI to refresh run-state immediately."""
        if monitor_server is None:
            return
        try:
            monitor_server.push_event({"type": "run_state_update"})
        except Exception:
            logger.debug("Failed to emit run_state_update", exc_info=True)

    def _setup_logging(self) -> None:
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        setup_search_logging(log_level=self.config.log_level, log_dir=log_dir, name=self.name)

    def _load_initial_program(self) -> str:
        with open(self.initial_program_path, "r") as f:
            return f.read()

    def _save_checkpoint(self, iteration: int) -> str:
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.database.save(checkpoint_path, iteration)

        best = self._get_best_program()
        if best:
            with open(
                os.path.join(checkpoint_path, f"best_program{self.file_extension}"), "w"
            ) as f:
                f.write(best.solution)
            with open(os.path.join(checkpoint_path, "best_program_info.json"), "w") as f:
                from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

                json.dump(
                    {
                        "id": best.id,
                        "generation": best.generation,
                        "iteration": best.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best.metrics,
                        "language": best.language,
                        "timestamp": best.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                    cls=SafeJSONEncoder,
                )
            logger.info(f"Checkpoint {iteration}: best={format_metrics(best.metrics)}")

        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Loaded checkpoint (iteration {self.database.last_iteration})")

    def _get_best_program(self) -> Optional[Program]:
        if self.database.best_program_id:
            prog = self.database.get(self.database.best_program_id)
            if prog:
                return prog
        return self.database.get_best_program()

    def _save_best_program(self, program: Program) -> None:
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        code_path = os.path.join(best_dir, f"best_program{self.file_extension}")
        with open(code_path, "w") as f:
            f.write(program.solution)

        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
                cls=SafeJSONEncoder,
            )

        if self.config.language == "image" and program.metadata:
            img = program.metadata.get("image_path")
            if img and os.path.exists(img):
                import shutil

                shutil.copy2(img, os.path.join(best_dir, "best_image" + os.path.splitext(img)[1]))

        logger.info(f"Best program saved to {best_dir}")

    def _write_run_state(
        self,
        status: str,
        *,
        max_iterations: Optional[int],
        checkpoint_path: Optional[str] = None,
        remaining_iterations: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist lightweight run metadata for future monitoring/resume UIs."""
        best = self._get_best_program()
        payload: Dict[str, Any] = {
            "status": status,
            "search_type": self.name,
            "max_iterations": max_iterations,
            "last_iteration": self.database.last_iteration,
            "num_programs": len(self.database.programs),
            "updated_at": time.time(),
        }
        if remaining_iterations is not None:
            payload["remaining_iterations"] = remaining_iterations
        if checkpoint_path:
            payload["checkpoint_path"] = checkpoint_path
        if best:
            payload["best_program_id"] = best.id
            payload["best_score"] = get_score(best.metrics or {})
        if error:
            payload["error"] = error

        state_path = os.path.join(self.output_dir, "run_state.json")
        with open(state_path, "w") as f:
            json.dump(payload, f, indent=2)
