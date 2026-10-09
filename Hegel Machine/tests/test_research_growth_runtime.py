"""Failures that would make the public joint runtime silently misleading."""

import json
from dataclasses import replace

from hegel_machine.cognition.growth_lab import GrowthLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_growth import GrowthResearcher


class OtherCoordinates:
    """A different observed coordinate system, without declared role labels."""

    def __init__(self, source):
        self.source = source
        self.spec = TaskSpec("different_instrument", "Predict the measured response",
            ("field", "pressure"), (), (),
            feature_bounds={"field": (200., 400.), "pressure": (20., 60.)}, max_probe_length=64)
        self.observation_cost = 0

    def observe(self, probe):
        # The response is an independently supplied physical law. The research
        # interface exposes neither this mapping nor the source expression.
        self.observation_cost += probe.cost
        result = []
        names = sorted(self.source.required_features())
        for features in probe.sequence:
            original = {names[0]: .01 * features["field"] - 3.,
                        names[1]: .05 * features["pressure"] - 2.}
            result.append(Observation(features, self.source.predict(Observation(original, 0.))))
        return result


def test_runtime_calls_learned_abstraction_across_anonymous_units_and_offsets():
    config = ResearchConfig(seed=8601, observation_budget=384, compute_budget=200000)
    first = GrowthResearcher(config)
    learned = first.run(GrowthLab(seed=8601))
    state = json.loads(json.dumps(first.to_dict()))
    second = GrowthResearcher(config, state=state)
    target = OtherCoordinates(Program.from_dict(learned["program"]))
    result = second.run(target)
    program = Program.from_dict(result["program"])
    assert result["status"] == "confirmed"
    assert program.op == "call"
    assert program.definition.source_concept_id == learned["abstraction_ids"][0]
    assert program.required_features() == {"field", "pressure"}
    assert result["validation_loss"] < 1e-10
    assert result["costs"]["observation_cost"] == target.observation_cost


def test_unfunded_measurements_and_confirmation_do_not_create_supported_knowledge():
    config = ResearchConfig(seed=8601, observation_budget=24, compute_budget=200000)
    researcher = GrowthResearcher(config)
    result = researcher.run(GrowthLab(seed=8601))
    assert result["status"] == "budget_exhausted"
    assert result["costs"]["observation_cost"] == 24
    assert not researcher.library.concepts
    assert not researcher.methods.methods
    # A noisy, hidden-input apparatus remains unresolved if no measurement can
    # be purchased. It must not learn unsupported deterministic knowledge.
    researcher = GrowthResearcher(replace(config, observation_budget=72), measurement_budget=0)
    result = researcher.run(GrowthLab(seed=8601, stage=3))
    assert result["status"] == "budget_exhausted"
    assert result["costs"]["measurement_cost"] == 0
    assert not researcher.library.concepts


def test_cli_saves_public_growth_state(tmp_path, capsys):
    from hegel_machine.cli import main
    path = tmp_path / "growth.json"
    assert main(["cognition", "growth", "--checkpoint", str(path), "--tasks", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    state = json.loads(path.read_text())
    assert report["status"] == "confirmed"
    assert len(state["tasks"]) == 1 and state["abstractions"]
