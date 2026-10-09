"""Cost lineage and failed-publication orchestration without model calls."""

from copy import deepcopy

from scripts import run_temporal_conditional_method as runner
from scripts.run_temporal_method_learning import COSTS


def _costs(value):
    return {key: value for key in COSTS}


def test_prior_learning_counted_once_and_failed_publication_stops_transfer(tmp_path, monkeypatch):
    original = {"files": {"shared/old_plan.py": "# Original acquisition procedure"},
                "manifest": {"research_programs": [{"id": "old", "path": "shared/old_plan.py"}]},
                "lessons": ["Original lesson"]}
    experience = {"agent_arm": "hegel", "task_position": 1, "status": "completed",
                  "paid_observations": [], "events": [], "outgoing_knowledge": original,
                  "costs": _costs(999)}
    seed = {"status": "completed", "method_path": "shared/consolidated_method.py",
            "method_source": "def select_actions(spec, observations, memory): return []\n",
            "method_entry": {"id": "seed", "path": "shared/consolidated_method.py"},
            "lesson": "Seed lesson", "costs": _costs(777)}
    seed_summary = {"common_learning_costs": {"combined": _costs(42)}}
    before = deepcopy((experience, seed, seed_summary))
    provider_paths = []

    class NoModelCalls:
        def next_action(self, request):
            raise AssertionError("this orchestration fixture must never call a model")

    def provider(path):
        provider_paths.append(path)
        return NoModelCalls()

    class FailedPractice:
        def __init__(self, supplied_provider, workspace, *, knowledge, config, confirmation_seed):
            workspace.mkdir(parents=True, exist_ok=True)
            assert isinstance(supplied_provider, NoModelCalls)
            assert knowledge["files"]["shared/old_plan.py"] == original["files"]["shared/old_plan.py"]
            assert knowledge["files"][seed["method_path"]] == seed["method_source"]
            assert config.observation_budget == 768 and config.confirmation_rows == 48

        def run(self, laboratory, *, on_checkpoint):
            assert laboratory.public_spec()["max_probe_length"] == 24
            result = {"status": "confirmation_budget_exhausted", "costs": _costs(7), "events": []}
            on_checkpoint(result)
            return result

    monkeypatch.setattr(runner, "ConditionalMethodPracticeAgent", FailedPractice)
    summary = runner.run_conditional_study(experience, seed, seed_summary, tmp_path,
                                            provider_factory=provider)
    assert summary["status"] == "method_not_published"
    assert summary["conditions"] == {}
    assert provider_paths == [tmp_path / "practice"]
    common = summary["common_learning_costs"]
    assert common["seed_study_learning"] == _costs(42)
    assert common["conditional_practice"] == _costs(7)
    assert common["combined"] == _costs(49)
    assert (experience, seed, seed_summary) == before
    assert (tmp_path / "source_at_run/scripts/run_temporal_conditional_method.py").is_file()
    assert not (tmp_path / "method_retained_memory.json").exists()
    assert runner.protocol()["transfer_limits"]["observation_budget"] == 240
