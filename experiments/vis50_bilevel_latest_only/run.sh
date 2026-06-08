#!/usr/bin/env bash
set -euo pipefail

# Reproduce the VIS bilevel evaluator-prompt experiment used in the local analysis.
#
# Usage:
#   experiments/vis50_bilevel_latest_only/run.sh [output_root]
#
# Defaults:
#   output_root = experiments/vis50_bilevel_latest_only/results/vis50_bilevel_latest_only
#
# Requires an OpenAI API key in the environment or in .env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUT_ROOT="${SCRIPT_DIR}/results/vis50_bilevel_latest_only"
OUT_ROOT="${1:-${DEFAULT_OUT_ROOT}}"
case "${OUT_ROOT}" in
  /*) ;;
  *) OUT_ROOT="${REPO_ROOT}/${OUT_ROOT}" ;;
esac
RUN_DIR="${OUT_ROOT}/run"
CONFIG_PATH="${OUT_ROOT}/config.yaml"
MODEL="${MODEL:-gpt-4o-mini}"
ITERATIONS="${ITERATIONS:-50}"
SATURATION_WINDOW="${SATURATION_WINDOW:-5}"

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"

set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export JUDGE_MODEL="${MODEL}"
export VIS_DATASET_PATH="${PWD}/benchmarks/insight_gen/VIS/data/VIS.csv"

unset OPENAI_API_BASE
unset OPENAI_BASE_URL

uv run python - "${CONFIG_PATH}" "${MODEL}" "${ITERATIONS}" "${SATURATION_WINDOW}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model = sys.argv[2]
iterations = int(sys.argv[3])
saturation_window = int(sys.argv[4])

base_path = Path("benchmarks/insight_gen/VIS/config.yaml")
cfg = yaml.safe_load(base_path.read_text())

cfg["max_iterations"] = iterations
cfg["checkpoint_interval"] = saturation_window

llm = cfg.setdefault("llm", {})
llm["api_base"] = "https://api.openai.com/v1"
llm["models"] = [{"name": model, "weight": 1.0}]
llm["evaluator_models"] = [{"name": model, "weight": 1.0}]
llm["guide_models"] = [{"name": model, "weight": 1.0}]

cfg.setdefault("monitor", {})["enabled"] = False
cfg["human_feedback_enabled"] = False

search = cfg.setdefault("search", {})
search["type"] = "adaevolve"
search["num_context_programs"] = 4

db = search.setdefault("database", {})
db.update(
    {
        "evaluator_prompt_evolution_enabled": True,
        "evaluator_prompt_window_size": saturation_window,
        "evaluator_prompt_saturation_threshold": 0.005,
        "evaluator_prompt_min_interval": saturation_window,
        "evaluator_prompt_generator_score_mode": "latest_only",
        "outer_evaluator_mode": "adaevolve",
        "outer_evaluator_max_iterations": 8,
        "outer_evaluator_population_size": 4,
        "outer_evaluator_operator_exploration_intensity": 0.4,
        "outer_evaluator_samples_per_eval": 2,
        "outer_evaluator_alpha_prompt_diversity": 2.0,
        "outer_evaluator_alpha_discrimination": 0.0,
        "outer_evaluator_beta_within_variance": 1.0,
        "outer_evaluator_gamma_drift": 0.25,
        "outer_evaluator_baseline_mode": "latest_only",
        "outer_evaluator_drift_control_mode": "soft",
        "outer_evaluator_probe_top_k": 3,
        "outer_evaluator_probe_mid_k": 2,
        "outer_evaluator_probe_low_k": 1,
        "outer_evaluator_canary_top_k": 1,
        "outer_evaluator_canary_mid_k": 1,
        "outer_evaluator_canary_low_k": 1,
    }
)

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(config_path)
PY

uv run skydiscover-run \
  benchmarks/insight_gen/VIS/initial_program.py \
  benchmarks/insight_gen/VIS/evaluator.py \
  -c "${CONFIG_PATH}" \
  -s adaevolve \
  -m "${MODEL}" \
  --api-base https://api.openai.com/v1 \
  -i "${ITERATIONS}" \
  -o "${RUN_DIR}" \
  -l INFO

uv run python "${SCRIPT_DIR}/analyze.py" "${RUN_DIR}"
