"""Ordinary Python receives and can resume the transitions it already purchased."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


def decision(action, **arguments):
    return {"action": action, "arguments": arguments, "usage": {"total_tokens": 1}}


def query(request, channels):
    bounds = request["capabilities"]["feature_bounds"]
    return decision("query", channels=channels, sequence=[
        {name: low + fraction * (high - low) for name, (low, high) in bounds.items()}
        for fraction in (.2, .8)])


class RecordingObjectLab(UnifiedObjectLab):
    def __init__(self):
        super().__init__(objects=4, readout="total", seed=71)
        self.purchased_batches = []

    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        self.purchased_batches.append(deepcopy(batch))
        return batch


READ_TRANSITIONS = "import json\nprint(json.dumps(json.load(open('transitions.json'))))\n"


def test_python_reads_exact_paid_transitions_for_each_view_without_command_view_measurements(tmp_path):
    class Provider:
        def next_action(self, request):
            step = len(request["events"])
            if step < 3:
                return query(request, [[], ["coarse"], ["fine"]][step])
            assert "transitions_file" not in request["events"][0]["feedback"]
            assert request["events"][-1]["feedback"]["transitions_file"] == "transitions.json"
            return decision("run", code=READ_TRANSITIONS)

    laboratory = RecordingObjectLab()
    agent = CodeAgent(Provider(), tmp_path, CodeAgentConfig(
        observation_budget=6, measurement_budget=30, max_model_calls=4))
    report = agent.run(laboratory)
    execution = report["events"][-1]["feedback"]
    assert execution["returncode"] == 0
    actual = json.loads(execution["stdout"])
    assert actual == [
        {"channels": ["coarse"], "observation_start": 2,
         "transitions": laboratory.purchased_batches[1].transitions},
        {"channels": ["fine"], "observation_start": 4,
         "transitions": laboratory.purchased_batches[2].transitions},
    ]
    assert report["costs"]["observation_cost"] == laboratory.costs["observation_cost"] == 6
    assert report["costs"]["measurement_cost"] == laboratory.costs["measurement_cost"] == 30
    for batch in actual:
        names = set(laboratory.spec_for(batch["channels"]).feature_names)
        for offset, transition in enumerate(batch["transitions"]):
            assert set(transition) == {"before", "after"}
            assert set(transition["before"]) == set(transition["after"]) == names
            assert transition["before"] == agent.observations[batch["observation_start"] + offset].features


def test_restored_python_can_use_old_paid_transitions_and_append_a_new_view(tmp_path):
    class Provider:
        def next_action(self, request):
            step = len(request["events"])
            if step in (0, 2):
                return query(request, ["fine"] if step == 0 else ["coarse"])
            return decision("run", code=READ_TRANSITIONS)

    laboratory = RecordingObjectLab()
    config = CodeAgentConfig(observation_budget=4, measurement_budget=30, max_model_calls=4)
    original = CodeAgent(Provider(), tmp_path / "original", config)
    saved = []

    def interrupt(state):
        saved.append(json.loads(json.dumps(state)))
        raise RuntimeError("pause after the paid query")

    with pytest.raises(RuntimeError, match="pause after the paid query"):
        original.run(laboratory, on_checkpoint=interrupt)
    paid = deepcopy(laboratory.purchased_batches[0].transitions)
    restored = CodeAgent(Provider(), tmp_path / "restored", config, state=saved[0])
    assert laboratory.costs["observation_cost"] == 2
    report = restored.run(laboratory)
    first_execution, final_execution = [event["feedback"] for event in report["events"]
                                         if event["action"] == "run"]
    assert first_execution["returncode"] == final_execution["returncode"] == 0
    before = json.loads(first_execution["stdout"])
    after = json.loads(final_execution["stdout"])
    assert before == [{"channels": ["fine"], "observation_start": 0, "transitions": paid}]
    assert after[0] == before[0]
    assert after[1] == {"channels": ["coarse"], "observation_start": 2,
                        "transitions": laboratory.purchased_batches[1].transitions}
    assert report["costs"]["observation_cost"] == laboratory.costs["observation_cost"] == 4
    assert report["costs"]["measurement_cost"] == laboratory.costs["measurement_cost"] == 30
    assert restored.to_dict()["transition_batches"] == after
