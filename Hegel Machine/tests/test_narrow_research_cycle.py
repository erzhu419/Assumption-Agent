"""The narrow acceptance case must exercise actual learning and later use."""

import importlib.util
from pathlib import Path

from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab

path = Path(__file__).resolve().parents[1] / "scripts/run_narrow_research_cycle.py"
spec = importlib.util.spec_from_file_location("narrow_research_cycle", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_empty_library_cycle_and_exact_source_deletion_finish_before_terminal_scoring():
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
    full, removed = (result["results"][name] for name in ("full", "delete_learned_source"))
    sources = set(result["source_ids"])
    assert sources and full["tasks"][0] == removed["tasks"][0]
    assert not sources.intersection(removed["tasks"][1]["available_concepts"])
    assert not sources.intersection(removed["tasks"][1]["actual_call_sources"])
    assert result["cycle_passed"], result["cycle_checks"]
    assert result["source_deletion_effect"]["first_successor_pareto_saving"]
    assert all(task["evaluation_rows"] == 256 for arm in result["results"].values()
               for task in arm["tasks"])
    assert events.index("evaluation") == events.count("learning")
    assert result["external_model_calls"] == 0
