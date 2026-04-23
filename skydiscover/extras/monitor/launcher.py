"""Standalone VIS launcher for managing long-running SkyDiscover runs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from skydiscover.config import load_config, resolve_iteration_budget
from skydiscover.extras.monitor.process_manager import LauncherRecipe, ProcessManager
from skydiscover.extras.monitor.server import MonitorServer

logger = logging.getLogger(__name__)


class StandaloneLauncher:
    """Long-lived outer monitor that can start and manage one child run."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        config_path: Optional[str],
        initial_program: Optional[str],
        evaluation_file: str,
        output_root: str,
        iterations: Optional[int],
        run_forever: bool,
        log_level: Optional[str],
    ):
        self.config = load_config(config_path)
        self.recipe = LauncherRecipe(
            evaluation_file=evaluation_file,
            initial_program=initial_program,
            config_path=config_path,
            output_root=output_root,
            iteration_budget=resolve_iteration_budget(
                self.config,
                iterations=iterations,
                run_forever=run_forever,
            ),
            run_forever=run_forever,
            log_level=log_level,
        )

        if not self.config.monitor.enabled:
            raise ValueError(
                "Standalone launcher requires monitor.enabled=true in the child SkyDiscover config."
            )

        from skydiscover.extras.external import KNOWN_EXTERNAL

        if self.config.search.type in KNOWN_EXTERNAL:
            raise ValueError(
                f"Standalone launcher currently supports native SkyDiscover runs only, not '{self.config.search.type}'."
            )

        self.monitor_server = MonitorServer(
            host=host,
            port=port,
            max_solution_length=self.config.monitor.max_solution_length,
            output_dir=output_root,
        )
        self.process_manager = ProcessManager(
            config=self.config,
            recipe=self.recipe,
            event_callback=self.monitor_server.push_event,
        )

        budget_label = (
            "unbounded"
            if self.recipe.iteration_budget is None
            else str(self.recipe.iteration_budget)
        )
        self.monitor_server.set_config_summary(
            f"{self.config.search.type} | max_iter={budget_label} | launcher"
        )
        self.monitor_server.set_start_handler(self.process_manager.start_new_run)
        self.monitor_server.set_stop_handler(self.process_manager.request_stop)
        self.monitor_server.set_resume_handler(self.process_manager.resume_latest_run)
        self.monitor_server.set_branch_handler(self.process_manager.start_branch_from_program)
        self.monitor_server.set_request_proxy(self.process_manager.proxy_request)
        self.monitor_server.set_run_state_provider(self.process_manager.get_run_state)
        self.monitor_server.set_run_label_provider(self.process_manager.get_run_label)
        self.monitor_server.set_series_label_provider(
            lambda _run_state=None: self.process_manager.get_series_label()
        )

    def start(self) -> None:
        """Start the outer monitor service."""
        self.monitor_server.start()
        url = f"http://localhost:{self.monitor_server.port}/"
        print(f"\n  Launcher monitor: {url}\n", flush=True)
        logger.info("Launcher monitor: %s", url)

    def stop(self) -> None:
        """Stop the launcher and its managed child process."""
        try:
            self.process_manager.shutdown()
        finally:
            self.monitor_server.stop()


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the standalone launcher."""
    parser = argparse.ArgumentParser(
        description="SkyDiscover standalone VIS launcher",
    )
    parser.add_argument(
        "initial_program",
        nargs="?",
        default=None,
        help="Path to the initial program file (optional).",
    )
    parser.add_argument(
        "evaluation_file",
        help="Path to the evaluator file or benchmark directory.",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to configuration file (YAML).")
    parser.add_argument(
        "--output-root",
        default="outputs/launcher_runs",
        help="Directory under which launcher-managed runs are created.",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=None,
        help="Iteration budget override for child runs.",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        default=False,
        help="Run indefinitely until stopped or early stopping triggers.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Launcher host.")
    parser.add_argument("--port", type=int, default=8765, help="Launcher port.")
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level for the launcher process.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the standalone launcher until interrupted."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level or "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.initial_program and not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found", file=sys.stderr)
        return 1
    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found", file=sys.stderr)
        return 1

    try:
        launcher = StandaloneLauncher(
            host=args.host,
            port=args.port,
            config_path=args.config,
            initial_program=args.initial_program,
            evaluation_file=args.evaluation_file,
            output_root=args.output_root,
            iterations=args.iterations,
            run_forever=args.run_forever,
            log_level=args.log_level,
        )
        launcher.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.debug("Standalone launcher failed", exc_info=True)
        return 1
    finally:
        if "launcher" in locals():
            launcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
