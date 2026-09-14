"""Unknown long-memory and noisy tasks use the ordinary research entry point."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=384, compute_budget=240000,
    minimum_improvement=1e-7, active_probes=False, max_rounds=4)


def run(task, seed, library=None, measurement_budget=1536):
    lab = UnknownTaskLab(task, seed)
    result = UnifiedResearchAgent(replace(CONFIG, seed=seed), library=library,
        measurement_budget=measurement_budget).run(lab)
    return lab, result


def accumulates(program):
    expanded = program.expanded()
    return (expanded.op == "accumulate" or any(accumulates(child) for child in expanded.children))


@pytest.mark.parametrize("task", [8, 9])
def test_discovered_state_predicts_longer_prefixes_without_old_outputs(task):
    lab, result = run(task, 7901 + task)
    assert result["status"] == "completed"
    program = Program.from_dict(result["program"])
    assert accumulates(program)
    rows = lab.evaluation_rows(result["selected_channels"], seed=9011, length=96, trajectories=2)
    unlabelled = [Observation(row.features, row.target,
        tuple({name: value for name, value in past.items() if name != "__target__"}
              for past in row.history)) for row in rows]
    assert loss(program, unlabelled, result["fallback"], .5) < 1e-7


def test_state_transfer_reopens_when_the_learned_decay_changes():
    library = KnowledgeLibrary()
    _, source = run(9, 7921, library)
    _, rebound = run(10, 7922, library)
    lab, changed = run(11, 7923, library)
    assert set(source["concepts_added"]) & set(rebound["used_sources"])
    assert any(event["status"] == "counterexample" for event in changed["transfer"])
    assert changed["selected_kind"] == "continuous_latent_state"
    rows = lab.evaluation_rows(changed["selected_channels"], seed=9191, length=96, trajectories=2)
    assert loss(Program.from_dict(changed["program"]), rows, changed["fallback"], .5) < 1e-7


def test_noisy_method_reuse_buys_evidence_that_can_refute_the_old_measurement_choice():
    library = KnowledgeLibrary()
    _, source = run(12, 7907, library)
    assert source["status"] == "completed"
    _, changed = run(13, 7908, library)
    trials = [event for event in changed["actions"]
              if event["action"] == "test_learned_acquisition_method"]
    assert trials and trials[0]["learned_trial_rows"] >= 32
    assert len(trials[0]["validation_channels"]) == 2
    assert not trials[0]["noise_assessment"]["adequate"]
    assert changed["risk_quantile"] == .9 and changed["loss_kind"] == "pinball"
    assert changed["selected_channels"]
    assert changed["costs"]["observation_cost"] <= CONFIG.observation_budget


def test_unmeasured_independent_signal_is_not_certified_as_noise():
    _, result = run(13, 8101, measurement_budget=0)
    assert result["status"] == "unresolved"
    assert result["concepts_added"] == []
    assert result["costs"]["measurement_cost"] == 0
