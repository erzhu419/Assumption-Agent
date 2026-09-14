"""Experiment adapters must preserve the actual partition and intervention."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cognition_v3_research.py"
    spec = importlib.util.spec_from_file_location("cognition_v3_research_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_v2_receives_forward_environment_without_changing_its_config_schema(runner, monkeypatch):
    original_suite = runner.reference_experiments.make_suite
    seen = {}

    def no_agent_run(config):
        seen.update(asdict(config))
        environments = runner.reference_experiments.make_suite(config.seed, partition=config.partition)
        assert len(environments) == 12
        assert all(env.evaluation_metadata["partition"] == "forward" for env in environments)
        return {"config": asdict(config), "partition": config.partition}

    monkeypatch.setattr(runner.reference_experiments, "run_sequence", no_agent_run)
    result = runner.run_system("reference_v2", runner.ResearchConfig(seed=101, partition="forward"))
    assert seen["partition"] == "confirmation"
    assert "selection_policy" not in seen
    assert result["partition"] == "forward"
    assert result["reference_config_partition"] == "confirmation"
    assert runner.reference_experiments.make_suite is original_suite


def test_live_ablations_change_only_the_named_config_and_keep_resource_caps(runner, monkeypatch):
    seen = []
    base = runner.ResearchConfig(seed=7, observation_budget=73, compute_budget=17000)

    def no_agent_run(config, **kwargs):
        seen.append((config, kwargs))
        return {"config": asdict(config)}

    monkeypatch.setattr(runner.live_experiments, "run_sequence", no_agent_run)
    for name in runner.SYSTEM_CHANGES:
        runner.run_system(name, base, remove_concept_id="concept_1" if name == "erase_specific_concept" else None)
    for name, (config, kwargs) in zip(runner.SYSTEM_CHANGES, seen):
        expected = asdict(base) | runner.SYSTEM_CHANGES[name]
        assert asdict(config) == expected
        assert config.proposer == "search"
        assert kwargs["remove_concept_ids"] == ({"concept_1"} if name == "erase_specific_concept" else None)


def test_compaction_keeps_unknown_costs_deletion_and_independent_decision_counts(runner):
    run = {
        "system": "erase_specific_concept", "costs_complete": False,
        "llm_usage": {"total_tokens": None}, "knowledge": {"concepts": []},
        "episodes": [{
            "episode_index": 11, "task_id": "f02", "compute_cost": None,
            "program": {"op": "const"}, "actions": [{"action": "revise", "details": {}}],
            "intervention": {"requested_ids": ["concept_1"], "removed_ids": ["concept_1", "template_1"]},
            "evaluation": {"loss": None, "final_counterfactual": {
                "wrongly_rejected_events": 1, "scored_rollback_events": 2, "events": [{"action": "rollback"}]}}
        }],
    }
    compact = runner.compact_run(run)
    episode = compact["episodes"][0]
    assert compact["costs_complete"] is False and compact["llm_usage"]["total_tokens"] is None
    assert episode["compute_cost"] is None
    assert episode["episode_index"] == 11
    assert episode["intervention"] == run["episodes"][0]["intervention"]
    assert episode["evaluation"]["final_counterfactual"] == {
        "wrongly_rejected_events": 1, "scored_rollback_events": 2}
    assert episode["action_counts"] == {"revise": 1}
    assert "program" not in episode and "actions" not in episode


def test_report_renaming_preserves_pairing_when_task_id_reappears(runner, monkeypatch):
    from hegel_machine.cognition.research_protocol import paired_seed_effect

    def sequence(system, losses):
        return {"system": system, "algorithm_version": "test", "config": {"seed": 7},
                "episodes": [{"episode_index": index, "task_id": "repeated_task",
                              "evaluation": {"loss_kind": "squared_error", "loss": value}}
                             for index, value in enumerate(losses)]}

    def summary(comparisons, seeds, partition):
        runs = comparisons[0]["runs"]
        assert set(runs) == {"cognitive_agent", "reference_v2"}
        effect = paired_seed_effect(runs["cognitive_agent"], runs["reference_v2"])
        return {"systems": {name: {} for name in runs}, "seed_results": [{"systems": {name: {} for name in runs}}],
                "paired_loss_alternative_minus_cognitive": {"reference_v2": effect}, "inference": {}}

    monkeypatch.setattr(runner, "summarize_research", summary)
    payload = {"seeds": [7], "systems": ["reference_v2", "cognitive_v3"], "partition": "development",
               "runs": [sequence("reference_v2", [1.5, 3.0]), sequence("cognitive_v3", [1.0, 4.0])],
               "design": {}, "limitations": []}
    result = runner.partition_summary(payload)
    effect = result["paired_loss_alternative_minus_v3"]["reference_v2"]["squared_error"]
    assert effect == {"mean_difference": -0.25, "paired_tasks": 2, "attempted_tasks": 2}
    assert set(result["systems"]) == {"reference_v2", "cognitive_v3"}
    assert set(result["seed_results"][0]["systems"]) == {"reference_v2", "cognitive_v3"}
    assert result["inference"]["paired_sign"] == "positive favors cognitive_v3"
