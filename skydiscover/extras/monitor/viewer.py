"""
Replay viewer for completed SkyDiscover runs.

Loads checkpoint data and serves the live monitor dashboard
for interactive exploration of past runs.

Usage:
    python -m skydiscover.extras.monitor.viewer <path> [--port PORT] [--summary-model MODEL]

<path> can be:
    - A checkpoint directory (contains metadata.json + programs/)
    - An output directory (contains island/checkpoints/ or checkpoints/)
    - A directory containing program JSON files
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from skydiscover.extras.monitor.checkpoint_replay import (
    checkpoint_program_to_monitor_format,
    find_checkpoint_dir,
    load_checkpoint_programs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay viewer for completed SkyDiscover runs",
    )
    parser.add_argument("path", help="Output directory or checkpoint path")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--summary-model",
        default="",
        help="LLM model for per-program summaries (default: gpt-5-mini). "
        "Requires OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_dir = find_checkpoint_dir(args.path)
    if not ckpt_dir:
        print(f"Error: no checkpoint data found in '{args.path}'")
        sys.exit(1)

    logger.info(f"Loading from: {ckpt_dir}")
    prog_list, best_id, last_iter = load_checkpoint_programs(ckpt_dir)
    if not prog_list:
        print(f"Error: no programs found in '{ckpt_dir}'")
        sys.exit(1)

    logger.info(f"Loaded {len(prog_list)} programs (best={best_id}, last_iter={last_iter})")

    all_progs = {p["id"]: p for p in prog_list}
    monitor_programs = [checkpoint_program_to_monitor_format(p, all_progs) for p in prog_list]

    # Start server
    from skydiscover.extras.monitor.server import MonitorServer

    server = MonitorServer(host=args.host, port=args.port, output_dir=args.path)

    # Configure per-program & global summary
    summary_model = args.summary_model
    if not summary_model and os.environ.get("OPENAI_API_KEY"):
        summary_model = "gpt-5-mini"
    if summary_model:
        server.configure_summary(model=summary_model, interval=0)

    server.start()

    # Push all programs
    best_score = -float("inf")
    for mp in monitor_programs:
        pid = mp["id"]
        s = mp["score"]
        is_best = (pid == best_id) or (s > best_score)
        if s > best_score:
            best_score = s

        solution = all_progs[pid].get("solution", "")
        parent_solution = ""
        if mp["parent_id"] and mp["parent_id"] in all_progs:
            parent_solution = all_progs[mp["parent_id"]].get("solution", "")

        server.push_event(
            {
                "type": "new_program",
                "program": mp,
                "stats": {
                    "total_programs": len(monitor_programs),
                    "current_iteration": last_iter,
                    "best_score": best_score,
                    "iterations_since_improvement": 0,
                    "programs_per_min": 0,
                    "elapsed_seconds": 0,
                },
                "is_best": is_best,
                "full_solution": solution[: server.max_solution_length],
                "parent_full_solution": parent_solution[: server.max_solution_length],
            }
        )

    # Wait for queue to flush
    time.sleep(1.5)

    print(f"\n  Dashboard ready at http://localhost:{args.port}/")
    print(f"  {len(prog_list)} programs loaded from {ckpt_dir}")
    if summary_model:
        print(f"  Per-program summaries: {summary_model}")
    else:
        print("  Per-program summaries: disabled (set OPENAI_API_KEY or --summary-model)")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    server.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()
