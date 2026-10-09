"""Matched offline controls keep knowledge and final scoring separate."""

import importlib.util
from pathlib import Path

import pytest

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


path = Path(__file__).resolve().parents[1] / "scripts/run_current_offline_ablation.py"
spec = importlib.util.spec_from_file_location("current_offline_ablation", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_offline_controls_match_budget_isolate_library_and_finish_before_scoring():
    events = []

    class RecordedLab(UnknownTaskLab):
        def acquire(self, *args, **kwargs):
            events.append("learning")
            return super().acquire(*args, **kwargs)

        def evaluation_rows(self, *args, **kwargs):
            events.append("evaluation")
            return super().evaluation_rows(*args, **kwargs)

    config = ResearchConfig(observation_budget=96, compute_budget=12000,
        minimum_improvement=1e-7, max_rounds=1, seed=73)
    plan = study.protocol(73, config=config, order=(0, 0), measurement_budget=384)
    plan["evaluation"].update(length=8, trajectories=1)
    result = study.run_benchmark(plan, lab_factory=lambda item: RecordedLab(**item))
    reports = result["results"]
    assert len(reports) == 4 and result["external_model_calls"] == 0
    assert len({tuple(row["resource_limits"].items()) for row in reports.values()}) == 1
    for row in reports.values():
        assert row["tasks"][0]["available_concepts"] == []
        assert len(row["tasks"]) == 2
        assert all(task["evaluation_rows"] == 8 for task in row["tasks"])
        assert all(row["costs"][key] <= limit for key, limit in row["resource_limits"].items())
    assert reports["full"]["tasks"][1]["available_concepts"]
    assert reports["no_reuse"]["learned_concepts"] == 0
    assert all(not row["used_sources"] for row in reports["no_reuse"]["tasks"])
    assert reports["fixed_library"]["learned_concepts"] == 0
    assert reports["passive_experiments"]["revisit_count"] == 0
    assert events and events.index("evaluation") == events.count("learning")


def test_default_cohort_places_reusable_source_before_target():
    assert [item["task_index"] for item in study.protocol(73)["tasks"]] == [1, 2, 3]


@pytest.mark.parametrize("seed", [73, 12901])
def test_full_adaptive_arm_keeps_compute_for_frozen_confirmation(seed, monkeypatch):
    """A tight adaptive task cap must still reach its final confirmation gate."""
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    limits = []
    original_run = UnifiedResearchAgent.run

    def record_limit(self, *args, **kwargs):
        if kwargs.get("previous") is None:
            limits.append(self.config.compute_budget)
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, "run", record_limit)
    config = ResearchConfig(observation_budget=96, compute_budget=12000,
        minimum_improvement=1e-7, max_rounds=1, seed=seed)
    plan = study.protocol(seed, config=config, order=(1, 2, 3), measurement_budget=384)
    session = study._train_arm(plan, "full")
    # The researcher's releasable reserve is part of the full task limit;
    # the session must not subtract another permanently unavailable reserve.
    assert limits == [config.compute_budget] * len(plan["tasks"])
    target = session.state["results"][1]["research"]
    actions = target["actions"]
    # A solved observation revision can confirm earlier, before the general
    # final gate. Either route must buy an actual independent comparison.
    assert any(action.get("action") in {"validate_frozen_program", "confirm_observation_research"}
               for action in actions)
    assert target["confirmation_loss"] is not None
    assert not any(action.get("action") == "defer_frozen_confirmation" for action in actions)
    assert target["costs"]["compute_cost"] <= config.compute_budget


@pytest.mark.parametrize("compute_budget", [12000, 24000])
def test_measured_recall_is_confirmed_before_expanding_condition_research(compute_budget):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.meta_prior_research import _sources

    config = ResearchConfig(observation_budget=96, compute_budget=compute_budget,
        minimum_improvement=1e-7, max_rounds=1, seed=12901)
    plan = study.protocol(12901, config=config, order=(1, 2, 3), measurement_budget=384)
    plan["evaluation"].update(length=16, trajectories=1)
    session = study._train_arm(plan, "full")
    report = session.evaluate(**plan["evaluation"])
    source = report["results"][1]["research"]
    target = report["results"][2]["research"]
    assert set(_sources(Program.from_dict(target["program"]))) & set(source["concepts_added"])
    assert target["status"] == "completed"
    assert target["confirmation_loss"] <= config.minimum_improvement
    assert any(item["accepted"] for item in target["transfer"])
    assert all(row["mse"] <= plan["solved_threshold"] for row in report["evaluation"])
    assert all(row["research"]["costs"]["compute_cost"] <= compute_budget
               for row in report["results"])
