"""Two paid sources survive each other's target-local transfer failures."""

import importlib.util
from pathlib import Path

from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


path = Path(__file__).resolve().parents[1] / "scripts/run_two_source_research_cycle.py"
spec = importlib.util.spec_from_file_location("two_source_research_cycle", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_two_naturally_learned_sources_keep_independent_value_after_local_failure():
    events = []

    class RecordedLab(UnknownTaskLab):
        def acquire(self, *args, **kwargs):
            events.append("learning")
            return super().acquire(*args, **kwargs)

        def evaluation_rows(self, *args, **kwargs):
            events.append("evaluation")
            return super().evaluation_rows(*args, **kwargs)

    plan = study.protocol()
    result = study.run_cycle(plan, lab_factory=lambda task: RecordedLab(**task))
    arms = result["results"]
    full = arms["full"]
    source_a = set(full["tasks"][0]["concepts_added"])
    source_b = set(full["tasks"][1]["concepts_added"])
    assert source_a and source_b and source_a.isdisjoint(source_b)
    assert not full["tasks"][0]["available_concepts"]
    assert not source_a.intersection(full["tasks"][1]["actual_call_sources"])
    for name, deleted, preserved in (
        ("delete_a", source_a, source_b),
        ("delete_b", source_b, source_a),
        ("delete_both", source_a | source_b, set()),
    ):
        assert arms[name]["tasks"][:2] == full["tasks"][:2]
        available = set(arms[name]["tasks"][2]["available_concepts"])
        assert not deleted.intersection(available)
        assert preserved <= available
        assert not deleted.intersection(arms[name]["tasks"][2]["actual_call_sources"])
    assert result["cycle_passed"], result["cycle_checks"]
    assert len(result["source_deletion_effects"]) == 4
    assert all(effect["sequence_pareto_saving"] for effect in result["source_deletion_effects"].values())
    assert source_a.intersection(full["tasks"][5]["actual_call_sources"])
    assert source_b.intersection(full["tasks"][6]["actual_call_sources"])
    assert all(task["evaluation_rows"] == 256 for arm in arms.values() for task in arm["tasks"])
    assert events.index("evaluation") == events.count("learning")
    assert result["external_model_calls"] == 0
