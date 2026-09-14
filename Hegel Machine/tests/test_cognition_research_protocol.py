"""Reachable protocol errors: intervention access, denominators and paired inference."""
from dataclasses import asdict, replace

import pytest

from hegel_machine.cognition import experiments
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ActionRecord, EpisodeResult, Probe, ResearchConfig
from hegel_machine.cognition.research_protocol import confidence_summary, fresh_evidence_summary, paired_seed_effect


def test_continuous_interventions_obey_published_bounds_and_replay_realizations():
    environment = make_suite()[3]
    probe = Probe("invented", ({"x": 0.137, "z": -0.813}, {"x": -0.927, "z": 0.613}))
    first = environment.observe(probe)
    environment.observe(Probe("other", ({"x": 0.111, "z": 0.999},)))
    assert environment.observe(Probe("same_trajectory_new_name", probe.sequence)) == first
    assert make_suite()[3].observe(probe) == first
    assert first[0].history == ()
    assert first[1].history[0]["__target__"] == first[0].target
    with pytest.raises(ValueError, match="bounds"):
        environment.observe(Probe("outside", ({"x": 2.0, "z": 0.0},)))
    with pytest.raises(ValueError, match="length"):
        environment.observe(Probe("too_long", ({"x": 0.0, "z": 0.0},) * 13))


def test_confirmation_reserves_compositions_parameter_intervals_and_longer_history():
    development = make_suite(partition="development")
    confirmation = make_suite(partition="confirmation")
    assert len(development) == 17 and len(confirmation) == 12
    assert max(environment._mechanism.history_lag for environment in development) == 2
    assert max(environment._mechanism.history_lag for environment in confirmation) == 3
    assert all(not (environment._mechanism.lag_weight and environment._mechanism.nonlinear_weight)
               for environment in development)
    assert any(environment._mechanism.lag_weight and environment._mechanism.nonlinear_weight
               for environment in confirmation)
    assert 2.1 <= confirmation[0]._mechanism.slope <= 2.7
    for environment in development + confirmation:
        assert "partition" not in asdict(environment.spec)
        assert "mechanism" not in asdict(environment.spec)
        assert environment.spec.description == development[0].spec.description
    assert make_suite(11, "confirmation")[0]._mechanism.slope != confirmation[0]._mechanism.slope


def test_renamed_tasks_and_same_scope_regime_counterexamples_are_actual_data_changes():
    suite = make_suite()
    original, renamed = suite[0], suite[14]
    original_rows = original.observe(original.spec.initial_probes[0])
    renamed_rows = renamed.observe(renamed.spec.initial_probes[0])
    assert [row.target for row in original_rows] == [row.target for row in renamed_rows]
    assert set(renamed_rows[0].features) == {"u", "v"}
    before, after = suite[1], suite[-1]
    assert before.spec.task_id == after.spec.task_id
    probe = before.spec.available_probes[3]
    assert [row.target for row in before.observe(probe)] != [row.target for row in after.observe(probe)]
    assert suite[15].evaluation_metadata["phase"] == "unrelated"


def test_confidence_intervals_resample_seeds_and_pair_only_same_completed_episodes():
    assert confidence_summary([2.0])["interval_95"] is None
    assert confidence_summary([1.0, 3.0])["n"] == 2
    assert confidence_summary([1.0, 1.0, 1.0])["interval_95"] == [1.0, 1.0]

    def episode(index, value):
        return {"episode_index": index, "task_id": "same_scope",
                "evaluation": {"loss_kind": "squared_error", "loss": value}}

    reference = {"episodes": [episode(0, 1.0), episode(1, 4.0)]}
    alternative = {"episodes": [episode(0, 3.0), episode(1, None)]}
    effect = paired_seed_effect(reference, alternative)["squared_error"]
    assert effect == {"mean_difference": 2.0, "paired_tasks": 1, "attempted_tasks": 2}


class _ProtocolAgent:
    def __init__(self, config, library=None):
        self.config = config
        self.library = library if library is not None else KnowledgeLibrary()

    def run(self, environment):
        if environment.spec.task_id == "t02":
            raise RuntimeError("deliberate task failure")
        rows = environment.observe(environment.spec.initial_probes[0])
        value = 0.0 if self.config.use_memory else 1.0
        return EpisodeResult(
            task_id=environment.spec.task_id, program=Program("const", value=value), fallback=value,
            observations=rows, actions=[], questions=[], observation_cost=len(rows), compute_cost=3,
            validation_loss=1.0, initial_loss=1.0, selected_operation="refit", selected_memory_id=None,
            concepts_added=[], memory_events=[], stop_reason="round_budget",
        )


def test_failed_episodes_keep_the_denominator_and_have_no_final_evaluation(monkeypatch):
    suite = make_suite()[:2]
    called = []
    for environment in suite:
        original = environment.evaluation_rows

        def evaluated(original=original, task=environment.spec.task_id):
            called.append(task)
            return original()

        monkeypatch.setattr(environment, "evaluation_rows", evaluated)
    monkeypatch.setattr(experiments, "CognitiveAgent", _ProtocolAgent)
    monkeypatch.setattr(experiments, "make_suite", lambda seed, partition: suite)
    result = experiments.run_sequence(ResearchConfig())
    assert result["task_count"] == 2 and result["completed_tasks"] == 1
    assert result["rates"]["prediction_coverage"] == 0.5
    assert result["rates"]["failure_rate"] == 0.5
    assert result["costs_complete"] is False
    assert result["episodes"][1]["compute_cost"] is None
    assert result["episodes"][1]["error"] == "deliberate task failure"
    assert called == ["t01"]


def test_multi_seed_research_summary_preserves_pairing_and_omits_full_episode_traces(monkeypatch):
    monkeypatch.setattr(experiments, "CognitiveAgent", _ProtocolAgent)
    monkeypatch.setattr(experiments, "make_suite", lambda seed, partition: make_suite(seed, partition)[:2])
    result = experiments.run_research(
        ResearchConfig(), seeds=[7, 11], partition="development",
        systems=("cognitive_agent", "without_knowledge"),
    )
    assert result["seeds"] == [7, 11]
    effect = result["paired_loss_alternative_minus_cognitive"]["without_knowledge"]["squared_error"]
    assert effect["n"] == 2
    assert effect["paired_tasks"] == 2 and effect["attempted_tasks"] == 4
    assert result["systems"]["cognitive_agent"]["rates"]["failure_rate"]["mean"] == 0.5
    assert "episodes" not in result
    assert all("episodes" not in row for row in result["seed_results"])


def test_comparison_ablation_flags_and_llm_execution_are_explicit(monkeypatch):
    seen = []
    original_run = experiments.run_sequence

    def recorded(config, **kwargs):
        seen.append(config)
        return original_run(config, **kwargs)

    monkeypatch.setattr(experiments, "CognitiveAgent", _ProtocolAgent)
    monkeypatch.setattr(experiments, "make_suite", lambda seed, partition: make_suite(seed, partition)[:1])
    monkeypatch.setattr(experiments, "run_sequence", recorded)
    systems = ("single_step_planning", "without_abstractions", "without_structural_mapping", "cognitive_agent")
    result = experiments.compare(replace(ResearchConfig(), proposer="codex"), systems=systems)
    assert all(config.proposer == "search" for config in seen)
    assert result["runs"]["single_step_planning"]["config"]["joint_planning"] is False
    assert result["runs"]["without_abstractions"]["config"]["use_abstractions"] is False
    assert result["runs"]["without_structural_mapping"]["config"]["structural_transfer"] is False
    seen.clear()
    experiments.compare(ResearchConfig(), include_llm=True, systems=("llm_code_agent",))
    assert len(seen) == 1 and seen[0].proposer == "codex"


def test_fresh_error_rates_use_observed_losses_and_their_actual_decision_denominators():
    def action(kind, **details):
        return {"action": kind, "details": details}

    episode = {"expression": "x", "selected_memory_id": "known", "actions": [
        action("revise"), action("prediction_check", revision_loss=2.0, parent_loss=1.0),
        action("revise"), action("rollback", revised_loss=0.2, parent_loss=1.0),
        action("revise"), action("rollback", revised_loss=3.0, parent_loss=1.0),
        action("revise"),
        action("retrieve", retrieved=["known", "not_tested"], tested=[
            {"memory_id": "known", "operation": "reuse", "loss": 1.0},
            {"memory_id": "known", "operation": "abstract_reuse", "loss": 0.1}]),
        action("reuse_check", loss=0.4, expression="x", memory_id="known", accepted=False),
    ], "memory_events": [{"action": "restricted"}, {"action": "revoked"}]}
    measured = fresh_evidence_summary([episode], minimum_improvement=0.01)
    assert measured["false_extension_rate"] == 1.0
    assert measured["false_rejection_rate"] == 0.5
    assert measured["observed_revision_failure_rate"] == pytest.approx(2 / 3)
    assert measured["revision_check_coverage"] == 0.75
    assert measured["transfer_candidates_tested"] == 2
    assert measured["direct_candidates_tested"] == measured["abstract_candidates_tested"] == 1
    assert measured["failed_reuse_checks_retained_at_episode_end"] == 1
    assert measured["scope_restrictions"] == measured["revocations"] == 1
    assert fresh_evidence_summary([], 0.01)["false_rejection_rate"] is None


def test_refitted_knowledge_is_separate_from_direct_reuse_and_unknown_tokens_stay_unknown(monkeypatch):
    class Agent(_ProtocolAgent):
        def run(self, environment):
            result = super().run(environment)
            result.selected_operation = "abstract_reuse"
            result.selected_memory_id = "source_concept"
            result.llm_usage = {"calls": 1, "input_tokens": None, "output_tokens": None,
                                "total_tokens": None, "costs_complete": False}
            return result

    monkeypatch.setattr(experiments, "CognitiveAgent", Agent)
    monkeypatch.setattr(experiments, "make_suite", lambda seed, partition: make_suite(seed, partition)[:1])
    result = experiments.run_sequence(replace(ResearchConfig(), proposer="codex"))
    assert result["direct_reuse_tasks"] == 0 and result["abstract_reuse_tasks"] == 1
    assert result["metrics_by_phase"]["acquire"]["direct_reuses"] == 0
    assert result["metrics_by_phase"]["acquire"]["abstract_reuses"] == 1
    assert result["llm_usage"]["calls"] == 1
    assert result["llm_usage"]["total_tokens"] is None
    assert result["costs_complete"] is False


def test_final_counterfactual_errors_use_new_outcomes_without_changing_learning(monkeypatch):
    good = Program("add", children=(
        Program("mul", children=(Program("const", value=1.4), Program("feature", name="x"))),
        Program("const", value=0.3),
    ))
    bad = Program("const", value=0.0)
    state = {"running": False, "zero_evaluation": False, "evaluation_calls": 0}

    class Agent(_ProtocolAgent):
        def run(self, environment):
            state["running"] = True
            result = super().run(environment)
            result.program = bad
            result.actions = [
                ActionRecord(0, "revise", "Recorded learning choice", details={
                    "before_program": good.to_dict(), "after_program": bad.to_dict()}),
                ActionRecord(0, "rollback", "Recorded learning choice", details={
                    "rejected_program": good.to_dict(), "restored_program": bad.to_dict()}),
            ]
            state["running"] = False
            return result

    environment = make_suite()[0]
    original_evaluation = environment.evaluation_rows

    def evaluation_rows():
        assert state["running"] is False
        state["evaluation_calls"] += 1
        rows = original_evaluation()
        return tuple(replace(row, target=0.0) for row in rows) if state["zero_evaluation"] else rows

    monkeypatch.setattr(environment, "evaluation_rows", evaluation_rows)
    monkeypatch.setattr(experiments, "CognitiveAgent", Agent)
    monkeypatch.setattr(experiments, "make_suite", lambda seed, partition: [environment])
    first = experiments.run_sequence(ResearchConfig())
    measured = first["final_counterfactual"]
    assert measured["wrongly_accepted_events"] == measured["scored_revision_events"] == 1
    assert measured["wrongly_rejected_events"] == measured["scored_rollback_events"] == 1
    assert measured["wrong_acceptance_rate"] == measured["wrong_rejection_rate"] == 1.0
    event_measurement = first["episodes"][0]["evaluation"]["final_counterfactual"]
    assert event_measurement["unique_programs_compared"] == 2
    assert event_measurement["additional_programs_scored"] == 1
    assert event_measurement["additional_evaluation_units"] == 192

    state["zero_evaluation"] = True
    second = experiments.run_sequence(ResearchConfig())
    assert second["final_counterfactual"]["wrong_acceptance_rate"] == 0.0
    assert second["final_counterfactual"]["wrong_rejection_rate"] == 0.0
    assert first["episodes"][0]["actions"] == second["episodes"][0]["actions"]
    assert first["knowledge"] == second["knowledge"]
    assert state["evaluation_calls"] == 2


def test_missing_revision_asts_are_not_counted_as_correct_final_decisions():
    rows = make_suite()[0].evaluation_rows()
    measured = experiments.evaluate_revision_decisions(
        [{"action": "revise", "details": {}}], rows,
        fallback=0.0, risk_quantile=0.5, minimum_improvement=0.002,
    )
    assert measured["revision_events"] == measured["missing_program_records"] == 1
    assert measured["scored_revision_events"] == 0
    assert measured["wrong_acceptance_rate"] is None
