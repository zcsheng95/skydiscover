from types import SimpleNamespace

import pytest

from skydiscover.config import Config
from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.adaevolve.evaluator_prompt_evolution import (
    BiLevelOrchestrator,
    EvaluatorPromptEvolutionManager,
    OuterEvaluatorConfig,
    OuterIndividual,
    ProbeRecord,
    ScoreSaturationMonitor,
    constrained_latest_score,
    merge_bilevel_metrics,
)
from skydiscover.search.base_database import Program


def test_saturation_monitor_does_not_trigger_before_window():
    monitor = ScoreSaturationMonitor(
        window_size=3,
        saturation_threshold=0.01,
        min_interval=5,
    )

    assert monitor.record(1, 0.5) is False
    assert monitor.record(2, 0.5) is False


def test_saturation_monitor_triggers_when_best_so_far_stalls():
    monitor = ScoreSaturationMonitor(
        window_size=3,
        saturation_threshold=0.01,
        min_interval=5,
    )

    assert monitor.record(1, 0.5) is False
    assert monitor.record(2, 0.502) is False
    assert monitor.record(3, 0.504) is True


def test_saturation_monitor_respects_min_interval_after_mark_triggered():
    monitor = ScoreSaturationMonitor(
        window_size=3,
        saturation_threshold=0.01,
        min_interval=5,
    )

    assert monitor.record(1, 0.5) is False
    assert monitor.record(2, 0.5) is False
    assert monitor.record(3, 0.5) is True
    monitor.mark_triggered(3)
    assert monitor.record(4, 0.5) is False
    assert monitor.record(8, 0.5) is True


def test_bilevel_score_keeps_latest_when_old_score_above_floor():
    assert (
        constrained_latest_score(
            old_score=0.8,
            latest_score=0.9,
            old_score_floor=0.75,
            penalty_weight=2.0,
        )
        == 0.9
    )


def test_bilevel_score_penalizes_when_old_score_below_floor():
    assert constrained_latest_score(
        old_score=0.70,
        latest_score=0.9,
        old_score_floor=0.75,
        penalty_weight=2.0,
    ) == pytest.approx(0.8)


def test_bilevel_score_never_drops_below_zero():
    assert (
        constrained_latest_score(
            old_score=0.0,
            latest_score=0.1,
            old_score_floor=0.75,
            penalty_weight=2.0,
        )
        == 0.0
    )


def test_adaevolve_evaluator_prompt_config_loads_from_dict():
    config = Config.from_dict(
        {
            "search": {
                "type": "adaevolve",
                "database": {
                    "evaluator_prompt_evolution_enabled": True,
                    "evaluator_prompt_window_size": 7,
                    "evaluator_prompt_saturation_threshold": 0.001,
                    "evaluator_prompt_min_interval": 11,
                    "evaluator_prompt_old_score_tolerance": 0.03,
                    "evaluator_prompt_penalty_weight": 3.0,
                    "evaluator_prompt_env_var": "CUSTOM_PROMPT_PATH",
                    "evaluator_prompt_generator_score_mode": "latest_only",
                    "evaluator_prompt_max_versions": 4,
                    "evaluator_prompt_max_feedback_chars": 9999,
                    "outer_evaluator_alpha_prompt_diversity": 1.5,
                    "outer_evaluator_alpha_discrimination": 0.25,
                    "outer_evaluator_baseline_mode": "latest_only",
                    "outer_evaluator_drift_control_mode": "strict",
                },
            }
        }
    )

    db = config.search.database
    assert db.evaluator_prompt_evolution_enabled is True
    assert db.evaluator_prompt_window_size == 7
    assert db.evaluator_prompt_saturation_threshold == 0.001
    assert db.evaluator_prompt_min_interval == 11
    assert db.evaluator_prompt_old_score_tolerance == 0.03
    assert db.evaluator_prompt_penalty_weight == 3.0
    assert db.evaluator_prompt_env_var == "CUSTOM_PROMPT_PATH"
    assert db.evaluator_prompt_generator_score_mode == "latest_only"
    assert db.evaluator_prompt_max_versions == 4
    assert db.evaluator_prompt_max_feedback_chars == 9999
    assert db.outer_evaluator_alpha_prompt_diversity == 1.5
    assert db.outer_evaluator_alpha_discrimination == 0.25
    assert db.outer_evaluator_baseline_mode == "latest_only"
    assert db.outer_evaluator_drift_control_mode == "strict"


def test_vis_evaluator_prompt_override_reads_env_file(monkeypatch, tmp_path):
    from benchmarks.insight_gen.VIS import evaluator

    monkeypatch.delenv(evaluator.EVALUATOR_SYSTEM_PROMPT_ENV_VAR, raising=False)
    assert evaluator._get_system_prompt() == evaluator.SYSTEM_PROMPT

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("override prompt")
    monkeypatch.setenv(evaluator.EVALUATOR_SYSTEM_PROMPT_ENV_VAR, str(prompt_path))
    assert evaluator._get_system_prompt() == "override prompt"


class _FakeEvaluator:
    def __init__(self, env_var: str, prompt_path: str):
        self.env_var = env_var
        self.prompt_path = prompt_path
        self.env_vars = {"UNCHANGED": "1"}
        self.calls = []

    async def evaluate_program(self, eval_input, child_id):
        prompt_path = self.env_vars.get(self.env_var)
        self.calls.append((child_id, dict(self.env_vars)))
        if prompt_path == self.prompt_path:
            return EvaluationResult(
                metrics={
                    "combined_score": 0.9,
                    "judge_evidence": "latest evidence",
                    "judge_conclusion": "latest conclusion",
                    "image_path": "/tmp/latest.png",
                },
                artifacts={"primary": "latest"},
            )
        return EvaluationResult(
            metrics={
                "combined_score": 0.7,
                "judge_evidence": "old evidence",
                "judge_conclusion": "old conclusion",
            },
            artifacts={"primary": "old"},
        )


@pytest.mark.asyncio
async def test_dual_evaluation_latest_only_uses_active_prompt_score(tmp_path):
    env_var = "SKYDISCOVER_EVALUATOR_SYSTEM_PROMPT_PATH"
    prompt_path = str(tmp_path / "evaluator_prompt_v1.txt")
    evaluator = _FakeEvaluator(env_var, prompt_path)
    controller = object.__new__(AdaEvolveController)
    controller.evaluator = evaluator
    controller.database = SimpleNamespace(programs={})
    controller.evaluator_prompt_manager = SimpleNamespace(
        env_var=env_var,
        has_evolved_prompt=True,
        active_prompt_path=prompt_path,
        active_version=1,
        old_score_floor=0.75,
        penalty_weight=2.0,
        score_mode="latest_only",
        ensure_old_score_floor=lambda programs: 0.75,
    )

    result = await controller._evaluate_candidate("solution", "child")

    assert result.metrics["latest_combined_score"] == 0.9
    assert result.metrics["old_combined_score"] == 0.7
    assert result.metrics["old_score_floor"] == 0.75
    assert result.metrics["evaluator_prompt_version"] == 1
    assert result.metrics["evaluator_prompt_score_mode"] == "latest_only"
    assert result.metrics["combined_score"] == pytest.approx(0.9)
    assert result.metrics["old_judge_evidence"] == "old evidence"
    assert result.metrics["latest_judge_evidence"] == "latest evidence"
    assert result.metrics["image_path"] == "/tmp/latest.png"
    assert result.artifacts["primary"] == "latest"
    assert result.artifacts["old_evaluator_artifacts"] == {"primary": "old"}
    assert evaluator.env_vars == {"UNCHANGED": "1"}


def test_merge_bilevel_metrics_can_still_use_constrained_latest():
    merged = merge_bilevel_metrics(
        old_metrics={"combined_score": 0.7},
        latest_metrics={"combined_score": 0.9},
        prompt_version=1,
        old_score_floor=0.75,
        penalty_weight=2.0,
        score_mode="constrained_latest",
    )

    assert merged["combined_score"] == pytest.approx(0.8)
    assert merged["latest_combined_score"] == pytest.approx(0.9)
    assert merged["old_combined_score"] == pytest.approx(0.7)
    assert merged["evaluator_prompt_score_mode"] == "constrained_latest"


def test_format_prior_attempts_returns_empty_when_no_evolved_versions(tmp_path):
    manager = EvaluatorPromptEvolutionManager(
        output_dir=str(tmp_path),
        original_prompt="BASELINE PROMPT",
        guide_llms=None,
        window_size=2,
        saturation_threshold=0.01,
        min_interval=1,
        old_score_tolerance=0.02,
        penalty_weight=2.0,
        env_var="X",
        score_mode="latest_only",
        max_versions=5,
        max_feedback_chars=4000,
    )
    assert manager._format_prior_attempts("BASELINE PROMPT") == ""


def test_format_prior_attempts_includes_chain_diffs(tmp_path):
    manager = EvaluatorPromptEvolutionManager(
        output_dir=str(tmp_path),
        original_prompt="line one\nline two\nbaseline",
        guide_llms=None,
        window_size=2,
        saturation_threshold=0.01,
        min_interval=1,
        old_score_tolerance=0.02,
        penalty_weight=2.0,
        env_var="X",
        score_mode="latest_only",
        max_versions=5,
        max_feedback_chars=4000,
    )
    (manager.prompt_dir / "evaluator_prompt_v1.txt").write_text(
        "line one\nline two MODIFIED\nbaseline"
    )
    (manager.prompt_dir / "evaluator_prompt_v2.txt").write_text(
        "line one\nline two MODIFIED\nbaseline\nadded line"
    )
    manager.active_version = 2
    manager.active_prompt_path = str(manager.prompt_dir / "evaluator_prompt_v2.txt")

    section = manager._format_prior_attempts(
        "line one\nline two MODIFIED\nbaseline\nadded line"
    )
    assert "Prior rewrite attempts" in section
    assert "Transition v0 -> v1" in section
    assert "Transition v1 -> v2" in section
    assert "MODIFIED" in section
    assert "added line" in section


# ---------------------------------------------------------------------------
# BiLevelOrchestrator tests
# ---------------------------------------------------------------------------


def _mk_program(pid: str, score: float, solution: str = "code") -> Program:
    return Program(
        id=pid,
        solution=solution,
        metrics={"combined_score": score},
        iteration_found=int(score * 100),
    )


def _mk_manager(tmp_path) -> EvaluatorPromptEvolutionManager:
    class _DummyPool:
        async def generate(self, system_message, messages):
            return SimpleNamespace(text=messages[-1]["content"][:200])

    return EvaluatorPromptEvolutionManager(
        output_dir=str(tmp_path),
        original_prompt="P0 baseline rubric",
        guide_llms=_DummyPool(),
        window_size=3,
        saturation_threshold=0.005,
        min_interval=1,
        old_score_tolerance=0.02,
        penalty_weight=2.0,
        env_var="SKYDISCOVER_EVALUATOR_SYSTEM_PROMPT_PATH",
        score_mode="latest_only",
        max_versions=10,
        max_feedback_chars=4000,
    )


def test_stratified_select_picks_top_mid_low():
    progs = [_mk_program(f"p{i}", 0.1 * i) for i in range(1, 11)]
    strata = BiLevelOrchestrator.stratified_select(progs, top_k=3, mid_k=2, low_k=1)
    assert len(strata["top"]) == 3
    assert len(strata["mid"]) == 2
    assert len(strata["low"]) == 1
    top_ids = {p.id for p in strata["top"]}
    low_ids = {p.id for p in strata["low"]}
    # Top should be the highest scorers; low the lowest (after top removed).
    assert "p10" in top_ids
    assert "p1" in low_ids
    # No probe appears in two tiers.
    all_ids = top_ids | {p.id for p in strata["mid"]} | low_ids
    assert len(all_ids) == 6


def test_prompt_diversity_penalizes_near_duplicates():
    baseline = (
        "Score chart insights using correctness, specificity, depth, and so-what "
        "quality. Return JSON."
    )
    near_duplicate = (
        "Score chart insights using correctness, specificity, depth, and so-what "
        "quality. Return only JSON."
    )
    different = (
        "Use a four-stage rubric with evidence inventory, claim mapping, penalty "
        "triggers, and score-band anchors before returning JSON."
    )

    assert BiLevelOrchestrator._prompt_diversity(
        different, [baseline]
    ) > BiLevelOrchestrator._prompt_diversity(near_duplicate, [baseline])


def test_generator_guidance_includes_latest_evaluator_feedback():
    builder = AdaEvolveContextBuilder(Config.from_dict({}))
    parent = Program(
        id="parent-1",
        solution="def run(): return {}",
        metrics={
            "combined_score": 0.81,
            "latest_combined_score": 0.88,
            "old_combined_score": 0.76,
            "evaluator_prompt_version": 2,
            "evaluator_prompt_score_mode": "latest_only",
            "latest_judge_evidence": "p_t says the insight needs a clearer chart-grounded effect size.",
            "latest_judge_conclusion": "Good direction, but evidence is underspecified.",
            "old_judge_evidence": "p0 accepted the broad trend but wanted tighter traceability.",
            "old_judge_conclusion": "Partially supported.",
        },
    )

    section = builder._build_search_guidance(
        {"parent": parent},
        {"evaluator_feedback_programs": [("parent", parent)]},
    )

    assert "LATEST EVALUATOR EVIDENCE FOR NEXT MOVE" in section
    assert "evaluator_prompt_score_mode" not in section
    assert "evaluator_prompt_version" not in section
    assert "parent-1" not in section
    assert "### parent" in section
    assert "latest_combined_score (p_t): 0.8800" in section
    assert "old_combined_score (p0): 0.7600" in section
    assert "p_t says the insight needs a clearer chart-grounded effect size" in section
    assert "p0 accepted the broad trend" in section


def test_generator_prompt_score_breakdown_filters_internal_artifacts():
    builder = AdaEvolveContextBuilder(Config.from_dict({}))
    parent = Program(
        id="parent-1",
        solution="def run(): return {}",
        metrics={
            "combined_score": 0.81,
            "latest_combined_score": 0.88,
            "old_combined_score": 0.76,
            "old_score_floor": 0.7,
            "evaluator_prompt_version": 2,
            "evaluator_prompt_score_mode": "latest_only",
            "image_path": "/tmp/chart.png",
            "image_path_stable": "/tmp/chart-stable.png",
            "test_combined_score": 0.5,
            "latest_judge_evidence": "rendered only in the dedicated evidence block",
            "old_judge_evidence": "diagnostic evidence belongs in the evidence block",
            "insight_text": "A useful verbal insight remains visible.",
            "Correctness & Factuality": 1.0,
        },
    )

    user = "\n".join(
        [
            builder._format_metrics(parent.metrics),
            builder._format_current_program({"parent": parent}, "python"),
        ]
    )

    assert "insight_text: A useful verbal insight remains visible." in user
    assert "Correctness & Factuality: 1.0000" in user
    assert "latest_combined_score: 0.8800" in user
    assert "old_combined_score: 0.7600" in user
    assert "old_score_floor" not in user
    assert "evaluator_prompt_version" not in user
    assert "evaluator_prompt_score_mode" not in user
    assert "image_path" not in user
    assert "- test_combined_score" not in user
    assert "latest_judge_evidence" not in user
    assert "old_judge_evidence" not in user


def test_compute_fitness_rewards_prompt_diversity_and_controls_drift(tmp_path):
    mgr = _mk_manager(tmp_path)

    async def fake_eval(prompt, probes, K):
        return {p.program_id: [0.5] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(samples_per_eval=3),
        probe_evaluator=fake_eval,
    )
    samples = {"a": [0.8, 0.82, 0.78], "b": [0.4, 0.42, 0.38], "c": [0.6, 0.61, 0.59]}
    baseline = {"a": 0.80, "b": 0.40, "c": 0.60}
    candidate_prompt = "Use score bands and explicit penalty triggers for each dimension."
    reference_prompt = "P0 baseline rubric"
    fitness, div, disc, wv, dr, means, stds = orch._compute_fitness(
        samples,
        baseline,
        candidate_prompt=candidate_prompt,
        reference_prompts=[reference_prompt],
    )
    # Means should match samples means.
    assert abs(means["a"] - 0.8) < 1e-6
    # Drift near zero since baseline matches.
    assert dr < 1e-3
    # Prompt diversity is the rewarded term by default.
    assert div > 0
    # Discrimination > 0 (means are 0.8/0.4/0.6 spread).
    assert disc > 0.1
    # Within-variance > 0 (samples have small spread).
    assert wv > 0
    # Fitness defaults to diversity-first: 2*diversity - within_var - 0.25*drift.
    assert fitness == pytest.approx(2.0 * div - wv - 0.25 * dr, abs=1e-6)


def test_outer_operator_selection_uses_weighted_random_sampling(tmp_path, monkeypatch):
    mgr = _mk_manager(tmp_path)

    async def stable_eval(prompt, probes, K):
        return {p.program_id: [0.5] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(operator_exploration_intensity=0.4),
        probe_evaluator=stable_eval,
    )
    p0 = OuterIndividual(version_id=0, text="p0", parent_version=None, operator="seed")
    parent = OuterIndividual(version_id=1, text="parent", parent_version=0, operator="branch")
    partner = OuterIndividual(
        version_id=2,
        text="partner",
        parent_version=1,
        operator="refine",
        fitness=1.0,
    )
    pop = [p0, parent, partner]

    monkeypatch.setattr("skydiscover.search.adaevolve.evaluator_prompt_evolution.random.random", lambda: 0.39)
    op, selected_partner = orch._select_operator(pop, parent)
    assert op == "branch"
    assert selected_partner is None

    monkeypatch.setattr("skydiscover.search.adaevolve.evaluator_prompt_evolution.random.random", lambda: 0.80)
    op, selected_partner = orch._select_operator(pop, parent)
    assert op == "refine"
    assert selected_partner is None

    monkeypatch.setattr("skydiscover.search.adaevolve.evaluator_prompt_evolution.random.random", lambda: 0.99)
    op, selected_partner = orch._select_operator(pop, parent)
    assert op == "merge"
    assert selected_partner is partner

    op, selected_partner = orch._select_operator([p0, parent], parent)
    assert op == "refine"
    assert selected_partner is None


@pytest.mark.asyncio
async def test_p0_locked_never_mutated(tmp_path):
    mgr = _mk_manager(tmp_path)
    operator_calls = []

    class _Pool:
        async def generate(self, system_message, messages):
            operator_calls.append(messages[-1]["content"])
            return SimpleNamespace(text="DIFFERENT_STRUCTURE_REWRITE_v1")

    async def stable_eval(prompt, probes, K):
        # Constant scores → drift = 0, but no discrimination either.
        return {p.program_id: [0.5] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(max_iterations=3, samples_per_eval=2),
        probe_evaluator=stable_eval,
        guide_llms=_Pool(),
    )
    probes = [
        ProbeRecord(program_id="t1", eval_input="codeA", tier="top", baseline_mean=0.5),
        ProbeRecord(program_id="m1", eval_input="codeB", tier="mid", baseline_mean=0.5),
    ]
    latest_feedback = (
        "LATEST P_T EVIDENCE: latest_combined_score=0.88; "
        "latest_judge_evidence says the current evaluator rewarded concrete effect sizes."
    )
    await orch.run(iteration=10, feedback=latest_feedback, probes=probes, canaries=[])
    # Every operator call must have included one of the existing prompts as parent.
    # Critically, the guide-LLM was never asked to mutate "P0 baseline rubric" *as a parent only*
    # because the locked seed (version_id=0) is excluded by _select_parent.
    # Verify no operator user-message was about mutating an empty/baseline-derived parent only.
    assert all(call for call in operator_calls)
    assert all(latest_feedback in call for call in operator_calls)
    # The original baseline prompt file v0 must still exist with original content.
    v0 = (mgr.prompt_dir / "evaluator_prompt_v0.txt").read_text()
    assert v0 == "P0 baseline rubric"


@pytest.mark.asyncio
async def test_drift_hard_cap_disqualifies(tmp_path):
    mgr = _mk_manager(tmp_path)

    class _Pool:
        async def generate(self, system_message, messages):
            return SimpleNamespace(text="HIGH_DRIFT_REWRITE")

    async def drifting_eval(prompt, probes, K):
        # Active prompt drifts probes to 0.0; baseline computed at 0.8 makes drift huge.
        if prompt is None:
            return {p.program_id: [0.8] for p in probes}
        return {p.program_id: [0.0] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(
            max_iterations=2,
            samples_per_eval=2,
            drift_control_mode="strict",
            drift_hard_cap=0.05,
        ),
        probe_evaluator=drifting_eval,
        guide_llms=_Pool(),
    )
    probes = [
        ProbeRecord(program_id="t1", eval_input="x", tier="top", baseline_mean=0.8),
    ]
    result = await orch.run(iteration=5, probes=probes, canaries=[])
    # Every candidate disqualified by drift_hard_cap → no install.
    assert result is None
    # The active version must not have advanced.
    assert mgr.active_version == 0


@pytest.mark.asyncio
async def test_soft_drift_control_tracks_but_does_not_disqualify(tmp_path):
    mgr = _mk_manager(tmp_path)

    class _Pool:
        async def generate(self, system_message, messages):
            return SimpleNamespace(text="HIGH_DRIFT_BUT_DIVERSE_REWRITE")

    async def drifting_eval(prompt, probes, K):
        if prompt is None:
            return {p.program_id: [0.8] for p in probes}
        return {p.program_id: [0.0] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(
            max_iterations=2,
            samples_per_eval=2,
            drift_hard_cap=0.05,
        ),
        probe_evaluator=drifting_eval,
        guide_llms=_Pool(),
    )
    probes = [
        ProbeRecord(program_id="t1", eval_input="x", tier="top", baseline_mean=0.8),
    ]
    result = await orch.run(iteration=5, probes=probes, canaries=[])
    assert result is not None
    assert mgr.active_version == 1


@pytest.mark.asyncio
async def test_canary_created_once_and_reused(tmp_path):
    mgr = _mk_manager(tmp_path)

    async def baseline_eval(prompt, probes, K):
        return {p.program_id: [0.5] for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(canary_top_k=1, canary_mid_k=1, canary_low_k=1),
        probe_evaluator=baseline_eval,
    )
    progs = [_mk_program(f"p{i}", 0.1 * i) for i in range(1, 6)]
    canaries_first = await orch.ensure_canary(progs)
    assert len(canaries_first) == 3
    # All baselines populated.
    assert all(c.baseline_mean is not None for c in canaries_first)

    # Mutate program list; canary should still be the original IDs (read from disk).
    other_progs = [_mk_program("zzz", 0.99)]
    canaries_second = await orch.ensure_canary(other_progs)
    assert {c.program_id for c in canaries_second} == {c.program_id for c in canaries_first}
    # Manifest file exists.
    assert (orch.canary_dir / "manifest.json").exists()


@pytest.mark.asyncio
async def test_latest_only_reference_baseline_uses_active_prompt(tmp_path):
    mgr = _mk_manager(tmp_path)
    mgr.install_revision("P1 active rubric", iteration=1)
    seen_prompts = []

    async def eval_fn(prompt, probes, K):
        seen_prompts.append(prompt)
        score = 0.9 if prompt == "P1 active rubric" else 0.4
        return {p.program_id: [score] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(baseline_mode="latest_only"),
        probe_evaluator=eval_fn,
    )
    probes = [ProbeRecord(program_id="probe_1", eval_input="x", tier="top")]

    await orch.refresh_reference_baselines(probes)

    assert seen_prompts == ["P1 active rubric"]
    assert probes[0].baseline_mean == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_p0_reference_baseline_uses_original_prompt_sentinel(tmp_path):
    mgr = _mk_manager(tmp_path)
    mgr.install_revision("P1 active rubric", iteration=1)
    seen_prompts = []

    async def eval_fn(prompt, probes, K):
        seen_prompts.append(prompt)
        score = 0.9 if prompt == "P1 active rubric" else 0.4
        return {p.program_id: [score] * K for p in probes}

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(baseline_mode="p0"),
        probe_evaluator=eval_fn,
    )
    probes = [ProbeRecord(program_id="probe_1", eval_input="x", tier="top")]

    await orch.refresh_reference_baselines(probes)

    assert seen_prompts == [None]
    assert probes[0].baseline_mean == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_cumulative_drift_cap_aborts_install(tmp_path):
    mgr = _mk_manager(tmp_path)

    class _Pool:
        async def generate(self, system_message, messages):
            return SimpleNamespace(text="WINNING_REWRITE_TEXT")

    async def eval_fn(prompt, probes, K):
        # On probes (3 of them), candidate prompt gives mean 0.5 vs baseline 0.5 → drift 0,
        # but on canaries (1 here) the prompt gives 0.9 vs baseline 0.5 → drift 0.4 > cumulative cap.
        canary_ids = {"canary_1"}
        if prompt is None:
            return {p.program_id: [0.5] for p in probes}
        return {
            p.program_id: ([0.9] * K if p.program_id in canary_ids else [0.5] * K)
            for p in probes
        }

    orch = BiLevelOrchestrator(
        manager=mgr,
        config=OuterEvaluatorConfig(
            max_iterations=2,
            samples_per_eval=2,
            drift_control_mode="strict",
            drift_hard_cap=1.0,  # disable per-probe disqualification
            cumulative_drift_cap=0.10,
        ),
        probe_evaluator=eval_fn,
        guide_llms=_Pool(),
    )
    probes = [
        ProbeRecord(program_id=f"probe_{i}", eval_input="x", tier="top", baseline_mean=0.5)
        for i in range(3)
    ]
    canaries = [
        ProbeRecord(program_id="canary_1", eval_input="x", tier="top", baseline_mean=0.5),
    ]
    result = await orch.run(iteration=7, probes=probes, canaries=canaries)
    assert result is None
    assert mgr.active_version == 0
