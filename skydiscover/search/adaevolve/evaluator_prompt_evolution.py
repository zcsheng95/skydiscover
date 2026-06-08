"""Evaluator prompt evolution helpers for AdaEvolve."""

from __future__ import annotations

import difflib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    version: int
    path: str


class ScoreSaturationMonitor:
    """Track best-so-far scores and trigger when recent improvement stalls."""

    def __init__(self, window_size: int, saturation_threshold: float, min_interval: int):
        self.window_size = max(1, int(window_size))
        self.saturation_threshold = float(saturation_threshold)
        self.min_interval = max(0, int(min_interval))
        self.history: List[tuple[int, float]] = []
        self.best_so_far = float("-inf")
        self.last_trigger_iteration: Optional[int] = None

    def record(self, iteration: int, score: Optional[float]) -> bool:
        if score is None:
            return False
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return False

        self.best_so_far = max(self.best_so_far, score_f)
        self.history.append((int(iteration), self.best_so_far))
        return self.should_trigger(int(iteration))

    def should_trigger(self, iteration: int) -> bool:
        if len(self.history) < self.window_size:
            return False
        if (
            self.last_trigger_iteration is not None
            and iteration - self.last_trigger_iteration < self.min_interval
        ):
            return False

        window = self.history[-self.window_size :]
        improvement = window[-1][1] - window[0][1]
        return improvement < self.saturation_threshold

    def mark_triggered(self, iteration: int) -> None:
        self.last_trigger_iteration = int(iteration)


def constrained_latest_score(
    old_score: float,
    latest_score: float,
    old_score_floor: float,
    penalty_weight: float,
) -> float:
    """Optimize latest score while penalizing drops below the old evaluator floor."""
    old_f = float(old_score or 0.0)
    latest_f = float(latest_score or 0.0)
    floor_f = float(old_score_floor or 0.0)
    if old_f >= floor_f:
        return latest_f
    return max(0.0, latest_f - float(penalty_weight) * (floor_f - old_f))


def score_from_metrics(metrics: Dict[str, Any], key: str = "combined_score") -> Optional[float]:
    value = (metrics or {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def merge_bilevel_metrics(
    old_metrics: Dict[str, Any],
    latest_metrics: Dict[str, Any],
    *,
    prompt_version: int,
    old_score_floor: float,
    penalty_weight: float,
    score_mode: str = "latest_only",
) -> Dict[str, Any]:
    """Merge two evaluator runs, preserving latest artifacts as primary metrics."""
    old_score = score_from_metrics(old_metrics) or 0.0
    latest_score = score_from_metrics(latest_metrics) or 0.0
    normalized_mode = (score_mode or "latest_only").strip().lower()
    if normalized_mode == "latest_only":
        combined = latest_score
    elif normalized_mode == "constrained_latest":
        combined = constrained_latest_score(
            old_score=old_score,
            latest_score=latest_score,
            old_score_floor=old_score_floor,
            penalty_weight=penalty_weight,
        )
    else:
        logger.warning(
            "Unknown evaluator prompt score mode %r; falling back to latest_only",
            score_mode,
        )
        normalized_mode = "latest_only"
        combined = latest_score

    merged = dict(latest_metrics or {})
    merged["old_combined_score"] = old_score
    merged["latest_combined_score"] = latest_score
    merged["old_score_floor"] = float(old_score_floor or 0.0)
    merged["evaluator_prompt_version"] = int(prompt_version)
    merged["evaluator_prompt_score_mode"] = normalized_mode
    merged["combined_score"] = combined

    for field in ("judge_evidence", "judge_conclusion", "error"):
        old_value = (old_metrics or {}).get(field)
        latest_value = (latest_metrics or {}).get(field)
        if old_value:
            merged[f"old_{field}"] = old_value
        if latest_value:
            merged[f"latest_{field}"] = latest_value
    return merged


class EvaluatorPromptEvolutionManager:
    """Persist and evolve evaluator prompt versions for AdaEvolve."""

    def __init__(
        self,
        *,
        output_dir: str,
        original_prompt: str,
        guide_llms: LLMPool,
        window_size: int,
        saturation_threshold: float,
        min_interval: int,
        old_score_tolerance: float,
        penalty_weight: float,
        env_var: str,
        score_mode: str,
        max_versions: int,
        max_feedback_chars: int,
    ):
        self.output_dir = Path(output_dir)
        self.prompt_dir = self.output_dir / "evaluator_prompts"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.prompt_dir / "evaluator_prompt_history.jsonl"
        self.original_prompt = original_prompt or ""
        self.guide_llms = guide_llms
        self.monitor = ScoreSaturationMonitor(
            window_size=window_size,
            saturation_threshold=saturation_threshold,
            min_interval=min_interval,
        )
        self.old_score_tolerance = float(old_score_tolerance)
        self.penalty_weight = float(penalty_weight)
        self.env_var = env_var
        self.score_mode = (score_mode or "latest_only").strip().lower()
        self.max_versions = max(1, int(max_versions))
        self.max_feedback_chars = max(1000, int(max_feedback_chars))
        self.active_version = 0
        self.active_prompt_path: Optional[str] = None
        self.old_score_floor: Optional[float] = None
        self._write_prompt_version(0, self.original_prompt, reason="baseline")

    @property
    def has_evolved_prompt(self) -> bool:
        return self.active_version > 0 and self.active_prompt_path is not None

    def record_candidate(self, program: Program, iteration: int) -> bool:
        score = score_from_metrics(program.metrics or {})
        return self.monitor.record(iteration, score)

    def can_create_version(self) -> bool:
        return self.active_version + 1 < self.max_versions

    def ensure_old_score_floor(self, programs: Iterable[Program]) -> float:
        if self.old_score_floor is not None:
            return self.old_score_floor
        best_old = 0.0
        for program in programs:
            metrics = program.metrics or {}
            score = score_from_metrics(metrics, "old_combined_score")
            if score is None:
                score = score_from_metrics(metrics, "combined_score")
            if score is not None:
                best_old = max(best_old, score)
        self.old_score_floor = max(0.0, best_old - self.old_score_tolerance)
        return self.old_score_floor

    def aggregate_feedback(self, programs: Iterable[Program], current_iteration: int) -> str:
        recent = [
            p
            for p in programs
            if isinstance(p.iteration_found, int)
            and p.iteration_found <= current_iteration
            and p.metrics
        ]
        recent.sort(key=lambda p: p.iteration_found, reverse=True)
        recent = recent[: self.monitor.window_size]

        chunks: List[str] = []
        for program in recent:
            metrics = program.metrics or {}
            parts = [
                f"program_id: {program.id}",
                f"iteration: {program.iteration_found}",
                f"combined_score: {metrics.get('combined_score')}",
                f"old_combined_score: {metrics.get('old_combined_score')}",
                f"latest_combined_score: {metrics.get('latest_combined_score')}",
                f"evaluator_prompt_score_mode: {metrics.get('evaluator_prompt_score_mode')}",
            ]
            for key in (
                "judge_evidence",
                "judge_conclusion",
                "old_judge_evidence",
                "old_judge_conclusion",
                "latest_judge_evidence",
                "latest_judge_conclusion",
                "error",
                "old_error",
                "latest_error",
            ):
                value = metrics.get(key)
                if value:
                    parts.append(f"{key}: {value}")
            chunks.append("\n".join(parts))

        text = "\n\n---\n\n".join(chunks)
        if len(text) > self.max_feedback_chars:
            return text[: self.max_feedback_chars] + "\n\n[truncated]"
        return text

    async def propose_revision(self, feedback: str) -> Optional[str]:
        """Run one guide-LLM rewrite of the active prompt and return revised text.

        Does NOT install the new version; callers decide whether to install
        (e.g. after passing drift gates).
        """
        if not self.can_create_version():
            logger.info("Evaluator prompt max_versions reached; skipping prompt evolution")
            return None

        current_prompt = self._read_active_prompt()
        prior_section = self._format_prior_attempts(current_prompt)
        system_message = (
            "You revise evaluator system prompts. Preserve the evaluator's exact JSON "
            "output schema, score dimensions, and integer score ranges. Return only the "
            "complete revised system prompt, with no markdown fences or commentary. "
            "When prior rewrite attempts are provided, do not produce a near-paraphrase "
            "of any of them — the saturation problem persisted across those attempts, "
            "so a different rubric direction is required."
        )
        user_message = f"""\
Current evaluator system prompt (active version):

{current_prompt}

{prior_section}Recent candidate scores and judge feedback:

{feedback or "(no textual feedback available)"}

Revise the evaluator system prompt to explore a different rubric direction while
keeping scores anchored to the current chart-grounded evaluator task.
Constraints:
- Preserve the same task: judge one candidate insight against one chart image.
- Preserve all score dimensions and the required JSON fields.
- Keep scores as integer 0-100 values.
- Keep the evaluator objective grounded only in visible chart evidence.
- Do not introduce new output keys or change the JSON contract.

When prior rewrite attempts are listed above, take an explicitly different
direction. Concrete options to consider (pick one not yet attempted):
- Tighten rubric thresholds with concrete numeric anchors (e.g. score bands
  like 0-29 / 30-59 / 60-84 / 85-100 with descriptive criteria for each).
- Add explicit anti-patterns or counter-examples per dimension that should
  trigger penalties.
- Introduce a chart-evidence checklist the judge must walk through before
  scoring (e.g. mandatory visual-element identification step).
- Reweight which dimension dominates ties when scores are close.
"""
        response = await self.guide_llms.generate(
            system_message=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        revised = self._clean_prompt(response.text)
        if not revised:
            logger.warning("Evaluator prompt optimizer returned an empty prompt")
            return None
        return revised

    def install_revision(
        self, revised: str, *, iteration: int, reason: str = "saturation"
    ) -> PromptVersion:
        """Persist the revised prompt as the next version and make it active."""
        next_version = self.active_version + 1
        path = self._write_prompt_version(
            next_version,
            revised,
            reason=reason,
            iteration=iteration,
        )
        self.active_version = next_version
        self.active_prompt_path = str(path)
        self.monitor.mark_triggered(iteration)
        return PromptVersion(version=next_version, path=str(path))

    async def optimize_prompt(self, feedback: str, iteration: int) -> Optional[PromptVersion]:
        """Legacy single-shot rewrite without drift gates. Kept for backward compat."""
        revised = await self.propose_revision(feedback)
        if revised is None:
            return None
        return self.install_revision(revised, iteration=iteration, reason="saturation")

    def _read_prior_evolved_prompts(self) -> List[Tuple[int, str]]:
        """Return [(version, text)] for v1..v_{active-1} sorted ascending."""
        out: List[Tuple[int, str]] = []
        for ver in range(1, self.active_version):
            path = self.prompt_dir / f"evaluator_prompt_v{ver}.txt"
            if not path.exists():
                continue
            try:
                out.append((ver, path.read_text()))
            except OSError:
                continue
        return out

    def _format_prior_attempts(self, current_prompt: str) -> str:
        """Build a compact "previously tried" section for the guide LLM.

        Includes the v0 -> v1 -> ... -> v_{active-1} chain of unified diffs so
        the optimiser can see what directions have already been explored.
        Returns empty string when no prior evolved versions exist.
        """
        priors = self._read_prior_evolved_prompts()
        if not priors:
            return ""

        chain: List[Tuple[int, str]] = [(0, self.original_prompt)] + priors
        chain.append((self.active_version, current_prompt))
        chunks: List[str] = []
        for (a_ver, a_text), (b_ver, b_text) in zip(chain[:-1], chain[1:]):
            diff_lines = list(
                difflib.unified_diff(
                    (a_text or "").splitlines(),
                    (b_text or "").splitlines(),
                    fromfile=f"v{a_ver}",
                    tofile=f"v{b_ver}",
                    lineterm="",
                    n=1,
                )
            )
            diff_text = "\n".join(diff_lines) if diff_lines else "(no textual change)"
            chunks.append(f"# Transition v{a_ver} -> v{b_ver}\n{diff_text}")

        body = "\n\n".join(chunks)
        # Cap to keep payload bounded; keep most recent transitions.
        cap = max(2000, self.max_feedback_chars // 2)
        if len(body) > cap:
            body = "[older transitions truncated]\n" + body[-cap:]

        return (
            "Prior rewrite attempts (saturation has persisted across these, so a "
            "fundamentally different rubric direction is required):\n\n"
            f"{body}\n\n"
        )

    def _read_active_prompt(self) -> str:
        if self.active_prompt_path:
            try:
                return Path(self.active_prompt_path).read_text()
            except OSError:
                logger.warning("Could not read active evaluator prompt; using original")
        return self.original_prompt

    def _write_prompt_version(
        self,
        version: int,
        prompt: str,
        *,
        reason: str,
        iteration: Optional[int] = None,
    ) -> Path:
        path = self.prompt_dir / f"evaluator_prompt_v{version}.txt"
        path.write_text(prompt or "")
        with self.history_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "version": version,
                        "path": str(path),
                        "reason": reason,
                        "iteration": iteration,
                    }
                )
                + "\n"
            )
        return path

    @staticmethod
    def _clean_prompt(text: str) -> str:
        cleaned = (text or "").strip()
        fence = re.search(r"```(?:text|markdown)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        return cleaned


@dataclass
class ProbeRecord:
    """A frozen probe used to evaluate evaluator prompts during outer search."""

    program_id: str
    eval_input: str  # the program solution string passed to evaluator.evaluate_program
    tier: str  # "top" | "mid" | "low"
    baseline_mean: Optional[float] = None  # J(c; p_ref), filled at probe-set freeze time


@dataclass
class OuterIndividual:
    """A candidate evaluator prompt within the outer AdaEvolve population."""

    version_id: int  # 0 reserved for p_0 (locked)
    text: str
    parent_version: Optional[int]
    operator: str  # "seed" | "branch" | "refine" | "merge"
    fitness: float = float("-inf")
    prompt_diversity: float = 0.0
    discrimination: float = 0.0
    within_variance: float = 0.0
    drift: float = 0.0
    per_probe_means: Dict[str, float] = field(default_factory=dict)
    per_probe_stds: Dict[str, float] = field(default_factory=dict)
    disqualified: bool = False
    disqualified_reason: str = ""


class _OuterSaturationMonitor(ScoreSaturationMonitor):
    """Same logic as inner saturation monitor; renamed for log clarity."""


@dataclass
class OuterEvaluatorConfig:
    max_iterations: int = 12
    population_size: int = 6
    samples_per_eval: int = 5
    judge_temperature: float = 0.7
    operator_exploration_intensity: float = 0.4
    alpha_prompt_diversity: float = 2.0
    alpha_discrimination: float = 0.0
    beta_within_variance: float = 1.0
    gamma_drift: float = 0.25
    baseline_mode: str = "latest_only"  # "latest_only" anchors drift to p_t; "p0" anchors to original
    drift_control_mode: str = "soft"  # "soft" logs/penalizes drift; "strict" rejects over caps
    drift_hard_cap: float = 0.15
    cumulative_drift_cap: float = 0.30
    canary_top_k: int = 1
    canary_mid_k: int = 1
    canary_low_k: int = 1
    probe_top_k: int = 3
    probe_mid_k: int = 2
    probe_low_k: int = 1


class BiLevelOrchestrator:
    """Outer-loop AdaEvolve over evaluator prompts.

    Coordinates: probe-set freeze, reference baseline J(c; p_ref) caching, persistent canary,
    population-based prompt search via guide-LLM operators, diversity-driven
    fitness scoring with soft + hard drift control, cumulative-drift install gate,
    and telemetry.

    The probe evaluator is injected so unit tests can mock LLM calls.
    """

    def __init__(
        self,
        *,
        manager: "EvaluatorPromptEvolutionManager",
        config: OuterEvaluatorConfig,
        probe_evaluator,  # async callable: (prompt_text: str | None, probes: List[ProbeRecord], K: int) -> Dict[str, List[float]]
        guide_llms: Optional[LLMPool] = None,
    ):
        self.manager = manager
        self.config = config
        self._probe_evaluator = probe_evaluator
        self.guide_llms = guide_llms or manager.guide_llms

        self.canary_dir = manager.prompt_dir / "canary"
        self.canary_dir.mkdir(parents=True, exist_ok=True)
        self.canary_manifest = self.canary_dir / "manifest.json"

        self.outer_stats_path = manager.output_dir / f"outer_evaluator_stats.jsonl"

        self.outer_saturation = self._new_outer_saturation_monitor()

    def _new_outer_saturation_monitor(self) -> _OuterSaturationMonitor:
        return _OuterSaturationMonitor(
            window_size=max(3, self.config.max_iterations // 3),
            saturation_threshold=1e-4,
            min_interval=1,
        )

    def _strict_drift_control(self) -> bool:
        return (self.config.drift_control_mode or "soft").strip().lower() == "strict"

    def _baseline_mode(self) -> str:
        mode = (self.config.baseline_mode or "latest_only").strip().lower()
        if mode in {"latest", "active", "pt", "p_t", "latest_only"}:
            return "latest_only"
        if mode in {"p0", "old", "original", "baseline"}:
            return "p0"
        return "latest_only"

    def _reference_prompt_text(self) -> Optional[str]:
        if self._baseline_mode() == "p0":
            return None
        if self.manager.has_evolved_prompt:
            return self.manager._read_active_prompt()
        return None

    def _reference_prompt_version(self) -> int:
        if self._baseline_mode() == "p0":
            return 0
        return int(self.manager.active_version)

    async def refresh_reference_baselines(self, probes: List[ProbeRecord]) -> None:
        if not probes:
            return
        baseline_samples = await self._probe_evaluator(self._reference_prompt_text(), probes, 1)
        for probe in probes:
            samples = baseline_samples.get(probe.program_id, [])
            probe.baseline_mean = (sum(samples) / len(samples)) if samples else None

    # ------------------------------------------------------------------
    # Probe-set freeze (stratified top/mid/low)
    # ------------------------------------------------------------------
    @staticmethod
    def stratified_select(
        programs: List[Program],
        *,
        top_k: int,
        mid_k: int,
        low_k: int,
        score_key: str = "combined_score",
    ) -> Dict[str, List[Program]]:
        scored = [
            p
            for p in programs
            if (p.metrics or {}).get(score_key) is not None
        ]
        scored.sort(key=lambda p: float(p.metrics[score_key]), reverse=True)
        n = len(scored)
        if n == 0:
            return {"top": [], "mid": [], "low": []}
        top = scored[: min(top_k, n)]
        remaining = scored[len(top) :]
        # Mid = around the median
        if remaining and mid_k > 0:
            median_idx = len(remaining) // 2
            half = max(1, mid_k // 2)
            lo = max(0, median_idx - half)
            hi = min(len(remaining), lo + mid_k)
            mid = remaining[lo:hi]
            remaining = remaining[:lo] + remaining[hi:]
        else:
            mid = []
        # Low = bottom of the remainder
        low = remaining[-low_k:] if (remaining and low_k > 0) else []
        return {"top": top, "mid": mid, "low": low}

    # ------------------------------------------------------------------
    # Canary persistence (frozen at the very first saturation)
    # ------------------------------------------------------------------
    def _load_canary(self) -> List[ProbeRecord]:
        if not self.canary_manifest.exists():
            return []
        try:
            data = json.loads(self.canary_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return [
            ProbeRecord(
                program_id=item["program_id"],
                eval_input=(self.canary_dir / f"{item['program_id']}.solution").read_text(),
                tier=item["tier"],
                baseline_mean=item.get("baseline_mean"),
            )
            for item in data
        ]

    def _save_canary(self, canaries: List[ProbeRecord]) -> None:
        for probe in canaries:
            (self.canary_dir / f"{probe.program_id}.solution").write_text(probe.eval_input)
        manifest = [
            {
                "program_id": p.program_id,
                "tier": p.tier,
                "baseline_mean": p.baseline_mean,
            }
            for p in canaries
        ]
        self.canary_manifest.write_text(json.dumps(manifest, indent=2))

    async def ensure_canary(self, programs: List[Program]) -> List[ProbeRecord]:
        existing = self._load_canary()
        if existing:
            return existing
        cfg = self.config
        strata = self.stratified_select(
            programs,
            top_k=cfg.canary_top_k,
            mid_k=cfg.canary_mid_k,
            low_k=cfg.canary_low_k,
        )
        canaries: List[ProbeRecord] = []
        for tier in ("top", "mid", "low"):
            for p in strata[tier]:
                canaries.append(
                    ProbeRecord(program_id=p.id, eval_input=p.solution or "", tier=tier)
                )
        if not canaries:
            return []
        # Cache reference baseline J(c; p_ref) per canary.
        await self.refresh_reference_baselines(canaries)
        self._save_canary(canaries)
        return canaries

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        m = sum(values) / len(values)
        if len(values) < 2:
            return m, 0.0
        var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        return m, var ** 0.5

    def _compute_fitness(
        self,
        per_probe_samples: Dict[str, List[float]],
        baseline: Dict[str, float],
        *,
        candidate_prompt: str = "",
        reference_prompts: Optional[Iterable[str]] = None,
    ) -> Tuple[float, float, float, float, float, Dict[str, float], Dict[str, float]]:
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for pid, samples in per_probe_samples.items():
            m, s = self._mean_std(samples)
            means[pid] = m
            stds[pid] = s
        within_var = (
            sum(s * s for s in stds.values()) / len(stds) if stds else 0.0
        )
        # Discrimination = std of per-probe means.
        if means:
            mean_of_means = sum(means.values()) / len(means)
            if len(means) >= 2:
                disc_var = sum((m - mean_of_means) ** 2 for m in means.values()) / (len(means) - 1)
                discrim = disc_var ** 0.5
            else:
                discrim = 0.0
        else:
            discrim = 0.0
        # Drift = mean abs deviation from baseline per probe (skip probes lacking baseline).
        drifts = [abs(means[pid] - baseline[pid]) for pid in means if pid in baseline]
        drift = sum(drifts) / len(drifts) if drifts else 0.0
        prompt_diversity = self._prompt_diversity(candidate_prompt, reference_prompts or [])
        cfg = self.config
        fitness = (
            cfg.alpha_prompt_diversity * prompt_diversity
            + cfg.alpha_discrimination * discrim
            - cfg.beta_within_variance * within_var
            - cfg.gamma_drift * drift
        )
        return fitness, prompt_diversity, discrim, within_var, drift, means, stds

    @staticmethod
    def _prompt_token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

    @classmethod
    def _prompt_distance(cls, left: str, right: str) -> float:
        """Text distance in [0, 1], mixing lexical and sequence novelty."""
        left = left or ""
        right = right or ""
        if not left and not right:
            return 0.0

        left_tokens = cls._prompt_token_set(left)
        right_tokens = cls._prompt_token_set(right)
        if left_tokens or right_tokens:
            union = left_tokens | right_tokens
            lexical_distance = 1.0 - (len(left_tokens & right_tokens) / len(union))
        else:
            lexical_distance = 0.0

        sequence_distance = 1.0 - difflib.SequenceMatcher(None, left, right).ratio()
        return max(0.0, min(1.0, 0.5 * lexical_distance + 0.5 * sequence_distance))

    @classmethod
    def _prompt_diversity(
        cls, candidate_prompt: str, reference_prompts: Iterable[str]
    ) -> float:
        """Nearest-neighbor prompt novelty.

        Using the minimum distance prevents the outer loop from rewarding a prompt
        that is diverse on average but still a near-duplicate of an existing prompt.
        """
        refs = [text for text in reference_prompts if text is not None]
        if not refs:
            return 0.0
        return min(cls._prompt_distance(candidate_prompt, ref) for ref in refs)

    def _historical_prompt_references(self) -> List[str]:
        refs = [self.manager.original_prompt]
        refs.extend(text for _, text in self.manager._read_prior_evolved_prompts())
        return refs

    # ------------------------------------------------------------------
    # Population & operators
    # ------------------------------------------------------------------
    def _seed_population(self) -> List[OuterIndividual]:
        # v0 (locked) + current p_t (or another copy of p_0 if no evolved prompt yet)
        pop: List[OuterIndividual] = [
            OuterIndividual(
                version_id=0,
                text=self.manager.original_prompt,
                parent_version=None,
                operator="seed",
            )
        ]
        active_text = self.manager._read_active_prompt()
        if self.manager.has_evolved_prompt and active_text and active_text != self.manager.original_prompt:
            pop.append(
                OuterIndividual(
                    version_id=self.manager.active_version,
                    text=active_text,
                    parent_version=0,
                    operator="seed",
                )
            )
        return pop

    def _select_parent(self, pop: List[OuterIndividual]) -> OuterIndividual:
        # Exclude locked p_0 from being mutated, prefer best-fitness amongst rest.
        candidates = [ind for ind in pop if ind.version_id != 0 and not ind.disqualified]
        if not candidates:
            # Force first child off p_0 if nothing else exists.
            return pop[0]
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        # Simple epsilon-greedy: best with prob 0.7, random otherwise.
        if random.random() < 0.7 or len(candidates) == 1:
            return candidates[0]
        return random.choice(candidates)

    def _select_partner(
        self, pop: List[OuterIndividual], exclude: OuterIndividual
    ) -> Optional[OuterIndividual]:
        others = [
            ind for ind in pop
            if ind is not exclude and not ind.disqualified and ind.version_id != 0
        ]
        if not others:
            return None
        others.sort(key=lambda x: x.fitness, reverse=True)
        return others[0]

    def _operator_exploration_intensity(self) -> float:
        return min(1.0, max(0.0, float(self.config.operator_exploration_intensity)))

    def _select_operator(
        self, pop: List[OuterIndividual], parent: OuterIndividual
    ) -> Tuple[str, Optional[OuterIndividual]]:
        """Weighted random operator sampling, mirroring AdaEvolve mode sampling.

        branch is the exploration operator, refine is exploitation, and merge is
        the balanced/cross-pollination operator.
        """
        intensity = self._operator_exploration_intensity()
        rand = random.random()
        if rand < intensity:
            return "branch", None
        if rand < intensity + (1.0 - intensity) * 0.7:
            return "refine", None
        partner = self._select_partner(pop, parent)
        if partner is None:
            return "refine", None
        return "merge", partner

    async def _apply_operator(
        self,
        op: str,
        parent: OuterIndividual,
        partner: Optional[OuterIndividual],
        feedback: str,
    ) -> Optional[str]:
        if op == "branch":
            system = (
                "You revise evaluator system prompts. Preserve the JSON output schema, "
                "the four score dimensions, and the integer 0-100 score range. Take a "
                "STRUCTURALLY DIFFERENT direction from the parent prompt: different "
                "rubric organization, different anchor strategy, different reasoning steps. "
                "Keep the scoring objective close to the current reference evaluator and do not "
                "add new output fields. "
                "Return only the complete revised system prompt, no markdown fences."
            )
            user = (
                f"Parent evaluator prompt:\n\n{parent.text}\n\nFeedback:\n{feedback}\n\n"
                "Produce a prompt that increases evaluator-prompt diversity while staying "
                "anchored to the same chart-grounded judgment task."
            )
        elif op == "refine":
            system = (
                "You revise evaluator system prompts. Preserve the JSON output schema, "
                "score dimensions, and integer 0-100 range. Make a SMALL TARGETED edit "
                "to the parent prompt aimed at increasing prompt novelty without increasing "
                "score drift or within-candidate variance. Return only the revised prompt."
            )
            user = (
                f"Parent prompt:\n\n{parent.text}\n\nObserved fitness components for parent: "
                f"prompt_diversity={parent.prompt_diversity:.4f}, "
                f"discrimination={parent.discrimination:.4f}, "
                f"within_variance={parent.within_variance:.4f}, "
                f"drift={parent.drift:.4f}.\n\nFeedback:\n{feedback}\n\n"
                "Apply a targeted refinement."
            )
        elif op == "merge" and partner is not None:
            system = (
                "You merge two evaluator system prompts into one. Preserve the JSON output "
                "schema, score dimensions, and integer 0-100 range. Combine diverse rubric "
                "structures from both parents while keeping score behavior close to the reference evaluator. "
                "Return only the merged prompt."
            )
            user = (
                f"Parent A:\n\n{parent.text}\n\nParent B:\n\n{partner.text}\n\n"
                f"Feedback:\n{feedback}\n\nProduce a merged prompt."
            )
        else:
            return None
        try:
            response = await self.guide_llms.generate(
                system_message=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            logger.warning(f"Outer operator {op} LLM call failed: {e}")
            return None
        cleaned = self.manager._clean_prompt(getattr(response, "text", "") or "")
        return cleaned or None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def _log_iter(self, record: Dict[str, Any]) -> None:
        try:
            with self.outer_stats_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.warning(f"Could not write outer evaluator stats: {e}")

    # ------------------------------------------------------------------
    # Cumulative drift gate (canary-set)
    # ------------------------------------------------------------------
    async def passes_cumulative_drift_gate(
        self, candidate_prompt_text: str, canaries: List[ProbeRecord]
    ) -> Tuple[bool, float]:
        if not canaries:
            return True, 0.0
        per_probe_samples = await self._probe_evaluator(
            candidate_prompt_text, canaries, self.config.samples_per_eval
        )
        means = {pid: (sum(s) / len(s)) for pid, s in per_probe_samples.items() if s}
        baseline = {p.program_id: p.baseline_mean for p in canaries if p.baseline_mean is not None}
        drifts = [abs(means[pid] - baseline[pid]) for pid in means if pid in baseline]
        if not drifts:
            return True, 0.0
        cum_drift = sum(drifts) / len(drifts)
        return cum_drift <= self.config.cumulative_drift_cap, cum_drift

    # ------------------------------------------------------------------
    # Single-shot mode (one LLM rewrite, same gates as adaevolve mode)
    # ------------------------------------------------------------------
    async def run_single_shot(
        self,
        *,
        iteration: int,
        feedback: str,
        probes: List[ProbeRecord],
        canaries: List[ProbeRecord],
    ) -> Optional[PromptVersion]:
        """One guide-LLM rewrite, with soft or strict drift control.

        This mirrors `run` but does not search a population: exactly one
        candidate prompt is proposed. In soft mode drift is logged/penalized;
        in strict mode drift caps can reject the candidate.
        """
        cfg = self.config
        baseline_prompt_version = self._reference_prompt_version()
        baseline = {p.program_id: p.baseline_mean for p in probes if p.baseline_mean is not None}

        revised = await self.manager.propose_revision(feedback)
        if not revised:
            self._log_iter(
                {
                    "saturation_iteration": iteration,
                    "mode": "single_shot",
                    "drift_control_mode": cfg.drift_control_mode,
                    "baseline_mode": cfg.baseline_mode,
                    "baseline_prompt_version": baseline_prompt_version,
                    "outcome": "no_revision_proposed",
                }
            )
            return None

        per_probe_samples = await self._probe_evaluator(revised, probes, cfg.samples_per_eval)
        reference_prompts = self._historical_prompt_references() + [self.manager._read_active_prompt()]
        fitness, div, disc, wv, dr, means, stds = self._compute_fitness(
            per_probe_samples,
            baseline,
            candidate_prompt=revised,
            reference_prompts=reference_prompts,
        )

        record_base = {
            "saturation_iteration": iteration,
            "mode": "single_shot",
            "drift_control_mode": cfg.drift_control_mode,
            "baseline_mode": cfg.baseline_mode,
            "baseline_prompt_version": baseline_prompt_version,
            "fitness": fitness,
            "prompt_diversity": div,
            "discrimination": disc,
            "within_variance": wv,
            "drift": dr,
            "drift_over_hard_cap": dr > cfg.drift_hard_cap,
            "per_probe_means": means,
            "per_probe_stds": stds,
        }

        if self._strict_drift_control() and dr > cfg.drift_hard_cap:
            self._log_iter(
                {
                    **record_base,
                    "outcome": "rejected_drift_hard_cap",
                    "drift_hard_cap": cfg.drift_hard_cap,
                }
            )
            logger.info(
                f"Single-shot rewrite rejected: drift {dr:.4f} > cap {cfg.drift_hard_cap}"
            )
            return None

        ok, cum_drift = await self.passes_cumulative_drift_gate(revised, canaries)
        record_base["cumulative_drift"] = cum_drift
        record_base["cumulative_drift_over_cap"] = not ok
        if self._strict_drift_control() and not ok:
            self._log_iter(
                {
                    **record_base,
                    "outcome": "rejected_cumulative_drift",
                    "cumulative_drift": cum_drift,
                    "cumulative_drift_cap": cfg.cumulative_drift_cap,
                }
            )
            logger.info(
                f"Single-shot rewrite rejected: cumulative drift {cum_drift:.4f} > cap {cfg.cumulative_drift_cap}"
            )
            return None

        version = self.manager.install_revision(
            revised, iteration=iteration, reason="single_shot_gated"
        )
        self._log_iter(
            {
                **record_base,
                "outcome": "installed",
                "installed_version": version.version,
            }
        )
        return version

    # ------------------------------------------------------------------
    # Main outer loop
    # ------------------------------------------------------------------
    async def run(
        self,
        *,
        iteration: int,
        feedback: str = "",
        probes: List[ProbeRecord],
        canaries: List[ProbeRecord],
    ) -> Optional[PromptVersion]:
        cfg = self.config
        baseline_prompt_version = self._reference_prompt_version()
        baseline = {p.program_id: p.baseline_mean for p in probes if p.baseline_mean is not None}
        self.outer_saturation = self._new_outer_saturation_monitor()

        pop = self._seed_population()
        # Score the seed (excluding the locked p_0 — its fitness is implicit reference).
        for ind in pop:
            if ind.version_id == 0:
                continue
            samples = await self._probe_evaluator(ind.text, probes, cfg.samples_per_eval)
            reference_prompts = (
                self._historical_prompt_references()
                + [other.text for other in pop if other is not ind]
            )
            fitness, div, disc, wv, dr, means, stds = self._compute_fitness(
                samples,
                baseline,
                candidate_prompt=ind.text,
                reference_prompts=reference_prompts,
            )
            ind.fitness = fitness
            ind.prompt_diversity = div
            ind.discrimination = disc
            ind.within_variance = wv
            ind.drift = dr
            ind.per_probe_means, ind.per_probe_stds = means, stds
            if self._strict_drift_control() and dr > cfg.drift_hard_cap:
                ind.disqualified = True
                ind.disqualified_reason = "drift_hard_cap"

        best: Optional[OuterIndividual] = None
        best_fitness = float("-inf")

        next_version_counter = max((i.version_id for i in pop), default=0) + 1

        for outer_it in range(cfg.max_iterations):
            parent = self._select_parent(pop)
            op, partner = self._select_operator(pop, parent)
            child_text = await self._apply_operator(op, parent, partner, feedback=feedback)
            if not child_text:
                continue
            child = OuterIndividual(
                version_id=next_version_counter,
                text=child_text,
                parent_version=parent.version_id,
                operator=op,
            )
            next_version_counter += 1

            samples = await self._probe_evaluator(child.text, probes, cfg.samples_per_eval)
            reference_prompts = self._historical_prompt_references() + [ind.text for ind in pop]
            fitness, div, disc, wv, dr, means, stds = self._compute_fitness(
                samples,
                baseline,
                candidate_prompt=child.text,
                reference_prompts=reference_prompts,
            )
            child.fitness = fitness
            child.prompt_diversity = div
            child.discrimination = disc
            child.within_variance = wv
            child.drift = dr
            child.per_probe_means, child.per_probe_stds = means, stds
            drift_over_hard_cap = dr > cfg.drift_hard_cap
            if self._strict_drift_control() and drift_over_hard_cap:
                child.disqualified = True
                child.disqualified_reason = "drift_hard_cap"
            else:
                if fitness > best_fitness:
                    best, best_fitness = child, fitness

            pop.append(child)
            # Trim population to size, never remove the locked p_0.
            if len(pop) > cfg.population_size + 1:  # +1 for the locked seed
                survivors = [pop[0]] + sorted(
                    [p for p in pop[1:]],
                    key=lambda x: (not x.disqualified, x.fitness),
                    reverse=True,
                )[: cfg.population_size]
                pop = survivors

            self._log_iter(
                {
                    "outer_iteration": outer_it,
                    "saturation_iteration": iteration,
                    "drift_control_mode": cfg.drift_control_mode,
                    "baseline_mode": cfg.baseline_mode,
                    "baseline_prompt_version": baseline_prompt_version,
                    "operator_exploration_intensity": self._operator_exploration_intensity(),
                    "operator": op,
                    "parent_version": parent.version_id,
                    "child_version": child.version_id,
                    "fitness": child.fitness,
                    "prompt_diversity": child.prompt_diversity,
                    "discrimination": child.discrimination,
                    "within_variance": child.within_variance,
                    "drift": child.drift,
                    "drift_over_hard_cap": drift_over_hard_cap,
                    "disqualified": child.disqualified,
                    "disqualified_reason": child.disqualified_reason,
                    "per_probe_means": child.per_probe_means,
                    "per_probe_stds": child.per_probe_stds,
                }
            )

            if self.outer_saturation.record(outer_it, best_fitness if best_fitness != float("-inf") else None):
                logger.info(f"Outer search saturated at outer_iter={outer_it}")
                break

        if best is None:
            self._log_iter(
                {
                    "saturation_iteration": iteration,
                    "drift_control_mode": cfg.drift_control_mode,
                    "baseline_mode": cfg.baseline_mode,
                    "baseline_prompt_version": baseline_prompt_version,
                    "outcome": "no_viable_candidate_drift_bound",
                }
            )
            return None

        # Layer-C cumulative drift gate against canary set.
        ok, cum_drift = await self.passes_cumulative_drift_gate(best.text, canaries)
        if self._strict_drift_control() and not ok:
            self._log_iter(
                {
                    "saturation_iteration": iteration,
                    "drift_control_mode": cfg.drift_control_mode,
                    "baseline_mode": cfg.baseline_mode,
                    "baseline_prompt_version": baseline_prompt_version,
                    "outcome": "outer_run_aborted_cumulative_drift",
                    "cumulative_drift": cum_drift,
                    "cumulative_drift_cap": cfg.cumulative_drift_cap,
                    "cumulative_drift_over_cap": True,
                }
            )
            logger.info(
                f"Outer run aborted: cumulative drift {cum_drift:.4f} > cap {cfg.cumulative_drift_cap}"
            )
            return None

        # Install winning prompt as next version on the manager.
        next_version = self.manager.active_version + 1
        path = self.manager._write_prompt_version(
            next_version,
            best.text,
            reason="bilevel_outer",
            iteration=iteration,
        )
        self.manager.active_version = next_version
        self.manager.active_prompt_path = str(path)
        self.manager.monitor.mark_triggered(iteration)
        self._log_iter(
            {
                "saturation_iteration": iteration,
                "drift_control_mode": cfg.drift_control_mode,
                "baseline_mode": cfg.baseline_mode,
                "baseline_prompt_version": baseline_prompt_version,
                "outcome": "installed",
                "installed_version": next_version,
                "fitness": best.fitness,
                "prompt_diversity": best.prompt_diversity,
                "discrimination": best.discrimination,
                "within_variance": best.within_variance,
                "drift": best.drift,
                "cumulative_drift": cum_drift,
                "cumulative_drift_cap": cfg.cumulative_drift_cap,
                "cumulative_drift_over_cap": not ok,
            }
        )
        return PromptVersion(version=next_version, path=str(path))


def scoped_evaluator_env(evaluator: Any, updates: Dict[str, Optional[str]]):
    """Temporarily update an evaluator's env_vars mapping."""

    class _ScopedEvaluatorEnv:
        def __enter__(self):
            self.old_env = dict(getattr(evaluator, "env_vars", {}) or {})
            new_env = dict(self.old_env)
            for key, value in updates.items():
                if value is None:
                    new_env.pop(key, None)
                else:
                    new_env[key] = value
            evaluator.env_vars = new_env
            return evaluator

        def __exit__(self, exc_type, exc, tb):
            evaluator.env_vars = self.old_env
            return False

    return _ScopedEvaluatorEnv()
