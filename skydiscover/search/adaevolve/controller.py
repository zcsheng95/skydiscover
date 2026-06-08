"""
AdaEvolve Controller - Evolution loop with adaptive search intensity.

A clean implementation that uses the adaptive database for all
exploration/exploitation decisions. No explicit stagnation tracking -
search intensity handles exploration automatically.

Features:
- Adaptive sampling based on accumulated improvement signal
- Mode-aware prompting (exploration vs exploitation)
- Paradigm breakthrough for high-level strategy shifts
- Sibling context for learning from previous attempts
- Comprehensive JSON logging of all AdaEvolve signals
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.telemetry import get_sink
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.adaevolve.evaluator_prompt_evolution import (
    BiLevelOrchestrator,
    EvaluatorPromptEvolutionManager,
    OuterEvaluatorConfig,
    ProbeRecord,
    merge_bilevel_metrics,
    scoped_evaluator_env,
    score_from_metrics,
)
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class AdaEvolveController(DiscoveryController):
    """
    AdaEvolve evolution controller with adaptive search intensity.

    Key Features:
    1. Adaptive sampling: Search intensity per island determines exploration/exploitation
    2. Mode-aware prompting: Different guidance for exploration vs exploitation
    3. Sibling context: Shows previous mutations for learning
    4. Error retry: Retries failed generations with error context
    5. Island rotation: UCB-based selection via database.end_iteration()
    6. Paradigm breakthrough: High-level strategy shifts when globally stuck

    No explicit stagnation tracking - search intensity handles exploration
    automatically based on accumulated improvement signal.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs
        self.evaluator_prompt_manager: Optional[EvaluatorPromptEvolutionManager] = None
        self.bilevel_orchestrator: Optional[BiLevelOrchestrator] = None

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self.context_builder = AdaEvolveContextBuilder(self.config)
        self._init_evaluator_prompt_evolution(db_config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.info(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None

        logger.info(
            f"AdaEvolveController initialized "
            f"(language={self.config.language}, "
            f"paradigm_breakthrough={self.database.use_paradigm_breakthrough}, "
            f"evaluator_prompt_evolution={self.evaluator_prompt_manager is not None})"
        )

    def _init_evaluator_prompt_evolution(self, db_config) -> None:
        """Initialize AdaEvolve evaluator prompt evolution when enabled."""
        if not getattr(db_config, "evaluator_prompt_evolution_enabled", False):
            return

        output_dir = self.output_dir or getattr(self.database.config, "db_path", None) or "."
        original_prompt = self._load_original_evaluator_prompt()
        self.evaluator_prompt_manager = EvaluatorPromptEvolutionManager(
            output_dir=output_dir,
            original_prompt=original_prompt,
            guide_llms=self.guide_llms,
            window_size=getattr(db_config, "evaluator_prompt_window_size", 10),
            saturation_threshold=getattr(db_config, "evaluator_prompt_saturation_threshold", 0.005),
            min_interval=getattr(db_config, "evaluator_prompt_min_interval", 10),
            old_score_tolerance=getattr(db_config, "evaluator_prompt_old_score_tolerance", 0.02),
            penalty_weight=getattr(db_config, "evaluator_prompt_penalty_weight", 2.0),
            env_var=getattr(
                db_config,
                "evaluator_prompt_env_var",
                "SKYDISCOVER_EVALUATOR_SYSTEM_PROMPT_PATH",
            ),
            score_mode=getattr(db_config, "evaluator_prompt_generator_score_mode", "latest_only"),
            max_versions=getattr(db_config, "evaluator_prompt_max_versions", 5),
            max_feedback_chars=getattr(db_config, "evaluator_prompt_max_feedback_chars", 12000),
        )

        # Bi-level outer-evaluator orchestrator (used for both single_shot
        # and adaevolve modes; only skipped when mode=off). single_shot
        # reuses the orchestrator's drift-gate machinery so the two modes
        # differ only in whether a population search is run.
        outer_mode = getattr(db_config, "outer_evaluator_mode", "single_shot")
        if outer_mode in ("adaevolve", "single_shot"):
            self.bilevel_orchestrator = BiLevelOrchestrator(
                manager=self.evaluator_prompt_manager,
                config=OuterEvaluatorConfig(
                    max_iterations=getattr(db_config, "outer_evaluator_max_iterations", 12),
                    population_size=getattr(db_config, "outer_evaluator_population_size", 6),
                    samples_per_eval=getattr(db_config, "outer_evaluator_samples_per_eval", 5),
                    judge_temperature=getattr(db_config, "outer_evaluator_judge_temperature", 0.7),
                    operator_exploration_intensity=getattr(
                        db_config, "outer_evaluator_operator_exploration_intensity", 0.4
                    ),
                    alpha_prompt_diversity=getattr(
                        db_config, "outer_evaluator_alpha_prompt_diversity", 2.0
                    ),
                    alpha_discrimination=getattr(db_config, "outer_evaluator_alpha_discrimination", 0.0),
                    beta_within_variance=getattr(db_config, "outer_evaluator_beta_within_variance", 1.0),
                    gamma_drift=getattr(db_config, "outer_evaluator_gamma_drift", 0.25),
                    baseline_mode=getattr(db_config, "outer_evaluator_baseline_mode", "latest_only"),
                    drift_control_mode=getattr(db_config, "outer_evaluator_drift_control_mode", "soft"),
                    drift_hard_cap=getattr(db_config, "outer_evaluator_drift_hard_cap", 0.15),
                    cumulative_drift_cap=getattr(db_config, "outer_evaluator_cumulative_drift_cap", 0.30),
                    canary_top_k=getattr(db_config, "outer_evaluator_canary_top_k", 1),
                    canary_mid_k=getattr(db_config, "outer_evaluator_canary_mid_k", 1),
                    canary_low_k=getattr(db_config, "outer_evaluator_canary_low_k", 1),
                    probe_top_k=getattr(db_config, "outer_evaluator_probe_top_k", 3),
                    probe_mid_k=getattr(db_config, "outer_evaluator_probe_mid_k", 2),
                    probe_low_k=getattr(db_config, "outer_evaluator_probe_low_k", 1),
                ),
                probe_evaluator=self._make_probe_evaluator(),
                guide_llms=self.guide_llms,
            )

    def _load_original_evaluator_prompt(self) -> str:
        """Read the evaluator's baseline SYSTEM_PROMPT when available."""
        module = getattr(self.evaluator, "_eval_module", None)
        prompt = getattr(module, "SYSTEM_PROMPT", None)
        if isinstance(prompt, str):
            return prompt
        return ""

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )

        logger.info(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
            }

            # Add child program info if available
            if child_program:
                _cp_meta = child_program.get("metadata") or {}
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                    "metadata": {
                        "created_at": _cp_meta.get("created_at"),
                        "llm_calls_at_creation": _cp_meta.get("llm_calls_at_creation"),
                        "llm_tokens_at_creation": _cp_meta.get("llm_tokens_at_creation"),
                    },
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation."""
        total = start_iteration + max_iterations
        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
            finally:
                # CRITICAL: Tell database iteration is complete
                # This handles island rotation (UCB) and migration
                self.database.end_iteration(iteration)

        logger.info("AdaEvolve completed")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.info(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")

        return self.database.get_best_program()

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program."""
        # Find a seed program
        seed_program = None
        for i in range(self.database.num_islands):
            size = self.database.get_island_size(i)
            if size > 0 and seed_program is None:
                population = self.database.get_island_population(i)
                if population:
                    seed_program = population[0]
                    break

        if seed_program is None:
            logger.warning("No seed program found")
            return

        # Seed empty islands
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed_program.solution,
                    language=seed_program.language,
                    metrics=seed_program.metrics.copy() if seed_program.metrics else {},
                    iteration_found=seed_program.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.info(f"Seeded island {i}")

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        """Execute one evolution iteration."""
        iteration_start_time = time.time()

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            await self._generate_paradigms_if_needed()

        result = await self._run_normal_step(iteration)

        iteration_time = time.time() - iteration_start_time

        if result.error:
            logger.warning(f"Iteration {iteration}: {result.error}")
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
        else:
            self._process_result(result, iteration, checkpoint_callback)
            await self._maybe_evolve_evaluator_prompt(result, iteration)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )

    async def _generate_paradigms_if_needed(self) -> None:
        """Generate new paradigms if stagnating and none active."""
        if self.paradigm_generator is None:
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.info(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        """Run a normal iteration with optional retry."""
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            result = await self._generate_child(iteration, error_context=last_error)
            if not result.error:
                return result
            last_error = result.error
            logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
        )

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # Add to database (database handles which island)
        self.database.add(child, iteration=iteration, parent_id=result.parent_id)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.info(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

        # Checkpoint callback
        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)

    def _make_probe_evaluator(self):
        """Build an async probe-evaluator callable used by the bi-level orchestrator.

        Signature: async (prompt_text: Optional[str], probes, K: int) -> Dict[probe_id, List[float]].
        - prompt_text=None or empty → use baseline p_0 (clears the env var override)
        - prompt_text non-empty → write to a temp file and point the evaluator's env var at it
        - K samples are obtained by repeated evaluator.evaluate_program calls; the judge
          temperature env var is set so the underlying judge LLM call returns varied scores
        """
        manager = self.evaluator_prompt_manager
        outer_cfg_temp = getattr(
            self.config.search.database, "outer_evaluator_judge_temperature", 0.7
        )

        async def _evaluate(prompt_text: Optional[str], probes, K: int):
            import tempfile

            results: Dict[str, List[float]] = {p.program_id: [] for p in probes}
            tmp_path: Optional[str] = None
            if prompt_text:
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="outer_prompt_")
                with os.fdopen(fd, "w") as f:
                    f.write(prompt_text)

            env_updates = {
                manager.env_var: tmp_path,  # None → clears override → uses baseline SYSTEM_PROMPT
                "SKYDISCOVER_EVALUATOR_TEMPERATURE": str(outer_cfg_temp),
            }
            try:
                with scoped_evaluator_env(self.evaluator, env_updates):
                    for probe in probes:
                        for k in range(K):
                            try:
                                eval_result = await self.evaluator.evaluate_program(
                                    probe.eval_input, f"outer:{probe.program_id}:k{k}"
                                )
                                score = score_from_metrics(eval_result.metrics or {})
                                if score is not None:
                                    results[probe.program_id].append(score)
                            except Exception as e:
                                logger.warning(
                                    f"Probe eval failed for {probe.program_id} k={k}: {e}"
                                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            return results

        return _evaluate

    async def _maybe_evolve_evaluator_prompt(
        self,
        result: SerializableResult,
        iteration: int,
    ) -> None:
        """Update the active evaluator prompt when recent scores saturate."""
        manager = self.evaluator_prompt_manager
        if manager is None or not result.child_program_dict:
            return

        child = Program(**result.child_program_dict)
        if not manager.record_candidate(child, iteration):
            return

        if not manager.can_create_version():
            manager.monitor.mark_triggered(iteration)
            return

        outer_mode = getattr(
            self.config.search.database, "outer_evaluator_mode", "single_shot"
        )
        if outer_mode == "off":
            manager.monitor.mark_triggered(iteration)
            return

        manager.ensure_old_score_floor(self.database.programs.values())

        if outer_mode in ("adaevolve", "single_shot") and self.bilevel_orchestrator is not None:
            feedback = manager.aggregate_feedback(self.database.programs.values(), iteration)
            try:
                version = await self._run_bilevel_outer(
                    iteration, mode=outer_mode, feedback=feedback
                )
            except Exception as e:
                logger.warning(
                    f"Outer evaluator step ({outer_mode}) failed: {e}", exc_info=True
                )
                manager.monitor.mark_triggered(iteration)
                return
            if version:
                logger.info(
                    "Evaluator prompt evolved (%s) to v%d at iteration %d: %s",
                    outer_mode,
                    version.version,
                    iteration,
                    version.path,
                )
            else:
                # No install (drift-gated rejection / no viable candidate).
                manager.monitor.mark_triggered(iteration)
            return

        # Fallback: legacy single-shot rewrite without gates (no orchestrator).
        feedback = manager.aggregate_feedback(self.database.programs.values(), iteration)
        try:
            version = await manager.optimize_prompt(feedback, iteration)
        except Exception as e:
            logger.warning(f"Evaluator prompt evolution failed: {e}", exc_info=True)
            manager.monitor.mark_triggered(iteration)
            return
        if version:
            logger.info(
                "Evaluator prompt evolved to v%d at iteration %d: %s",
                version.version,
                iteration,
                version.path,
            )

    async def _run_bilevel_outer(
        self, iteration: int, *, mode: str = "adaevolve", feedback: str = ""
    ):
        """Freeze probes + canary, then run the orchestrator in the given mode.

        mode="adaevolve" runs the full population search; mode="single_shot"
        runs one gated guide-LLM rewrite. Both share probe-freeze, baseline
        caching, canary persistence, and drift gates.

        Returns the installed PromptVersion or None if no install happened.
        """
        orch = self.bilevel_orchestrator
        if orch is None:
            return None
        cfg = orch.config
        all_programs = list(self.database.programs.values())

        # Freeze probe set (stratified top/mid/low).
        strata = orch.stratified_select(
            all_programs,
            top_k=cfg.probe_top_k,
            mid_k=cfg.probe_mid_k,
            low_k=cfg.probe_low_k,
        )
        probes: List[ProbeRecord] = []
        for tier in ("top", "mid", "low"):
            for p in strata[tier]:
                probes.append(
                    ProbeRecord(program_id=p.id, eval_input=p.solution or "", tier=tier)
                )
        if not probes:
            logger.info("Bi-level outer skipped: no eligible probe programs")
            return None

        # Compute reference baseline J(c; p_ref) per probe (one sample).
        await orch.refresh_reference_baselines(probes)

        # Persistent canary set (frozen on first saturation, reused thereafter).
        canaries = await orch.ensure_canary(all_programs)
        await orch.refresh_reference_baselines(canaries)

        if mode == "single_shot":
            return await orch.run_single_shot(
                iteration=iteration,
                feedback=feedback,
                probes=probes,
                canaries=canaries,
            )
        return await orch.run(
            iteration=iteration,
            feedback=feedback,
            probes=probes,
            canaries=canaries,
        )

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
    ) -> SerializableResult:
        """Generate and evaluate a single child program."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            # Sample parent and context programs (database returns standard framework dicts)
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
            )

            # Unpack parent dict (standard framework pattern)
            if not parent_dict:
                logger.error("sample() returned empty parent dict")
                return SerializableResult(
                    error="Empty parent dict from sample()", iteration=iteration
                )
            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]

            # Read sampling mode stashed by database.sample()
            sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"

            # Capture sampling mode and intensity for logging
            self._last_sampling_mode = sampling_mode
            current_island = self.database.current_island
            if self.database.use_adaptive_search:
                self._last_sampling_intensity = self.database.adapter.get_search_intensity(
                    current_island
                )
            else:
                self._last_sampling_intensity = self.database.fixed_intensity

            # When paradigm is active, use best program as parent
            # This ensures paradigm (designed from best) is applied to best, not random parent
            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            if paradigm:
                best_program = self.database.get_best_program()
                if best_program:
                    parent_dict = {parent_label: best_program}
                    parent = best_program
                    # Keep context_programs_dict from sampling for diversity

            # Gather siblings for sibling context
            siblings = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                "evaluator_feedback_programs": self._collect_evaluator_feedback_programs(
                    parent, context_programs_dict
                ),
                # AdaEvolve-specific keys (consumed by AdaEvolveContextBuilder)
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
            }
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (AdaEvolveContextBuilder handles paradigm/sibling/error formatting)
            prompt = self.context_builder.build_prompt(parent_dict, context)

            # Mark paradigm as used after prompt is built
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Generate and evaluate
            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    @staticmethod
    def _collect_evaluator_feedback_programs(
        parent: Program,
        context_programs: Dict[str, List[Program]],
        *,
        max_context_programs: int = 3,
    ) -> List[tuple[str, Program]]:
        """Select programs whose evaluator evidence should guide the next move."""
        selected: List[tuple[str, Program]] = [("parent", parent)]
        seen = {parent.id}
        for label, programs in (context_programs or {}).items():
            for program in programs:
                if program.id in seen:
                    continue
                metrics = program.metrics or {}
                has_feedback = any(
                    metrics.get(key)
                    for key in (
                        "latest_judge_evidence",
                        "latest_judge_conclusion",
                        "judge_evidence",
                        "judge_conclusion",
                    )
                )
                if not has_feedback:
                    continue
                selected.append((label, program))
                seen.add(program.id)
                if len(selected) >= max_context_programs + 1:
                    return selected
        return selected

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from skydiscover.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                )
                response = result.text or ""
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image", iteration=iteration
                    )
            else:
                result = await self._call_llm(prompt["system"], prompt["user"])
                response = result.text
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            return SerializableResult(error=f"LLM error: {e}", iteration=iteration)

        if not response and self.config.language != "image":
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self._evaluate_candidate(eval_input, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            return SerializableResult(error=f"Evaluation error: {e}", iteration=iteration)

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {
            "changes": changes,
            "parent_metrics": parent.metrics,
            "created_at": time.time(),
        }
        _sink = get_sink()
        if _sink is not None:
            child_metadata["llm_calls_at_creation"] = _sink.n_calls
            child_metadata["llm_tokens_at_creation"] = _sink.n_tokens
        if image_path:
            child_metadata["image_path"] = image_path
        if self.evaluator_prompt_manager is not None:
            child_metadata["evaluator_prompt_version"] = metrics.get("evaluator_prompt_version", 0)
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
        )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            iteration=iteration,
        )

    async def _evaluate_candidate(self, eval_input: str, child_id: str):
        """Evaluate a candidate once or with old/latest evaluator prompts."""
        manager = self.evaluator_prompt_manager
        if manager is None:
            return await self.evaluator.evaluate_program(eval_input, child_id)

        if not manager.has_evolved_prompt:
            with scoped_evaluator_env(self.evaluator, {manager.env_var: None}):
                eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            score = score_from_metrics(eval_result.metrics or {}) or 0.0
            eval_result.metrics = dict(eval_result.metrics or {})
            eval_result.metrics.setdefault("old_combined_score", score)
            eval_result.metrics.setdefault("latest_combined_score", score)
            eval_result.metrics.setdefault("evaluator_prompt_version", 0)
            if manager.old_score_floor is not None:
                eval_result.metrics.setdefault("old_score_floor", manager.old_score_floor)
            return eval_result

        prompt_path = manager.active_prompt_path
        with scoped_evaluator_env(self.evaluator, {manager.env_var: prompt_path}):
            latest_result = await self.evaluator.evaluate_program(
                eval_input, f"{child_id}:latest_v{manager.active_version}"
            )
        with scoped_evaluator_env(self.evaluator, {manager.env_var: None}):
            old_result = await self.evaluator.evaluate_program(eval_input, f"{child_id}:old_v0")

        old_score_floor = manager.old_score_floor
        if old_score_floor is None:
            old_score_floor = manager.ensure_old_score_floor(self.database.programs.values())

        latest_result.metrics = merge_bilevel_metrics(
            old_metrics=old_result.metrics or {},
            latest_metrics=latest_result.metrics or {},
            prompt_version=manager.active_version,
            old_score_floor=old_score_floor,
            penalty_weight=manager.penalty_weight,
            score_mode=getattr(manager, "score_mode", "latest_only"),
        )
        latest_result.artifacts = dict(latest_result.artifacts or {})
        old_artifacts = old_result.artifacts or {}
        if old_artifacts:
            latest_result.artifacts["old_evaluator_artifacts"] = old_artifacts
        return latest_result
