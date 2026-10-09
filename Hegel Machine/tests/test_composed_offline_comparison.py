"""Exercise the four-arm driver with actual learning and terminal scoring."""

from collections import Counter
from dataclasses import asdict
import importlib.util
from math import isfinite
from pathlib import Path

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import task_sequence
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


path = Path(__file__).resolve().parents[1] / "scripts/run_composed_offline_comparison.py"
spec = importlib.util.spec_from_file_location("composed_offline_comparison", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_four_arms_keep_task_local_learning_and_share_terminal_protocol():
    events = []
    evaluations = []

    class RecordedLab(UnknownTaskLab):
        def acquire(self, *args, **kwargs):
            events.append("learning")
            return super().acquire(*args, **kwargs)

        def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
            events.append("evaluation")
            evaluations.append({"task_seed": self.seed, "evaluation_seed": seed,
                "length": length, "trajectories": trajectories,
                "measurement_rate": sum(channel.cost_per_row for channel in self.channels
                    if channel.name in channels)})
            return super().evaluation_rows(channels, seed=seed, length=length,
                trajectories=trajectories)

    plan = study.protocol(seed=12901)
    plan["tasks"] = task_sequence(12901, order=(0, 0))
    plan["research_config"] = asdict(ResearchConfig(observation_budget=96,
        compute_budget=12000, minimum_improvement=1e-7, max_rounds=1, seed=12901))
    plan["measurement_budget"] = 384
    plan["evaluation"] = {"length": 8, "trajectories": 1, "seed_offset": 1000003}
    result = study.run_benchmark(plan, lab_factory=lambda task: RecordedLab(**task))
    arms = result["results"]
    assert set(arms) == {"full", "fixed_library", "passive_experiments", "task_local_memory"}

    # Forgetting between tasks must preserve the normal researcher's ability to
    # learn within each task; switching use_memory off is a different control.
    local = arms["task_local_memory"]
    assert all(not task["available_concepts"] for task in local["tasks"])
    assert all(task["concepts_added"] for task in local["tasks"])
    assert all(count > 0 for count in local["per_task_learned_concepts"])
    assert local["learned_concepts"] == sum(local["per_task_learned_concepts"])
    assert local["learned_concepts_scope"] == "sum_over_independent_task_libraries"
    assert not arms["full"]["tasks"][0]["available_concepts"]
    assert arms["full"]["tasks"][1]["available_concepts"]

    # A terminal score from one arm must never precede another arm's purchases.
    first_evaluation = events.index("evaluation")
    assert first_evaluation > 0
    assert set(events[:first_evaluation]) == {"learning"}
    assert events[first_evaluation:] == ["evaluation"] * 8
    expected_seeds = Counter({(task["seed"], task["seed"] + 1000003): 4
        for task in plan["tasks"]})
    assert Counter((row["task_seed"], row["evaluation_seed"]) for row in evaluations) == expected_seeds
    assert all(row["length"] == 8 and row["trajectories"] == 1 for row in evaluations)

    for arm_index, arm in enumerate(arms.values()):
        assert arm["resource_limits"] == {
            "observation_cost": 192, "measurement_cost": 768, "compute_cost": 24000}
        assert len(arm["tasks"]) == 2
        assert all(isfinite(task["mse"]) for task in arm["tasks"])
        for key, cap in arm["resource_limits"].items():
            assert 0 <= arm["costs"][key] <= cap
            charged = sum(task["costs"][key] for task in arm["tasks"])
            if key == "compute_cost":
                charged += arm["screening_compute_cost"]
            assert arm["costs"][key] == charged
        terminal = evaluations[2 * arm_index:2 * arm_index + 2]
        expected_costs = {"observation_cost": 16, "compute_cost": 16,
            "measurement_cost": sum(8 * row["measurement_rate"] for row in terminal)}
        assert arm["evaluation_costs"] == expected_costs
        assert all(task["evaluation_rows"] == 8 for task in arm["tasks"])
        for key in expected_costs:
            assert arm["evaluation_costs"][key] == sum(
                task["evaluation_costs"][key] for task in arm["tasks"])
    assert result["external_model_calls"] == 0
