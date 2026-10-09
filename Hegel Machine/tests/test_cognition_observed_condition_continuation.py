"""A fresh exact point fit keeps the measured condition question open."""

from dataclasses import asdict, replace
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class PaidConditionLab:
    channels = (MeasurementChannel("meter", 1, "Read the condition instrument."),)

    def __init__(self):
        self.spec = TaskSpec("condition", "Predict the measured response.", ("drive", "bias"), (), (),
            feature_bounds={"drive": (-1., 1.), "bias": (-1., 1.)}, max_probe_length=32)
        self.random = Random(837301)
        self.observation_cost = self.measurement_cost = 0

    def spec_for(self, channels=()):
        return replace(self.spec, feature_names=(*self.spec.feature_names, *channels))

    def acquire(self, probe, channels=()):
        assert all(set(point) == set(self.spec.feature_bounds) for point in probe.sequence)
        rows, history = [], []
        for point in probe.sequence:
            reading = .2 * point["bias"] + self.random.uniform(-1., 1.)
            target = point["drive"] * (1. if reading >= 0. else -1.) + .3 * reading
            features = {**point, **({"meter": reading} if "meter" in channels else {})}
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        self.observation_cost += probe.cost
        self.measurement_cost += probe.cost * len(channels)
        return MeasurementBatch(rows, probe.cost, probe.cost * len(channels), tuple(channels))


def known_response():
    drive, meter = Program("feature", name="drive"), Program("feature", name="meter")
    response = Program("if", children=(Program("ge", children=(meter, Program("const", value=0.))),
        drive, Program("mul", children=(Program("const", value=-1.), drive))))
    return Program("add", children=(response, Program("mul", children=(Program("const", value=.3), meter))))


@pytest.mark.parametrize("previous_status", ["unresolved", "completed"])
def test_continuation_keeps_exact_fresh_fit_distinct_from_cut_resolution(monkeypatch, previous_status):
    # Isolate the continuation acceptance transition; the meter purchases,
    # original targets and condition assessment remain real.
    import hegel_machine.cognition.condition_response_actions as condition_actions
    monkeypatch.setattr(condition_actions, "research_condition_response_actions",
        lambda agent, spec, selected, training, validation, finalists, *args, **kwargs:
            (selected, training, finalists))
    lab, batches = PaidConditionLab(), []
    for index in range(2):
        probe = Probe(f"paid:{index}", tuple({"drive": -.9 + .075 * step,
            "bias": .7 if step % 2 else -.7} for step in range(24)))
        batch = lab.acquire(probe, ("meter",))
        batches.append({"probe": asdict(probe), "channels": ["meter"],
                       "observations": [asdict(row) for row in batch.observations]})
    program = known_response()
    previous = {"task_id": lab.spec.task_id, "status": previous_status, "program": program.to_dict(),
        "fallback": 0., "selected_kind": "conditional_response", "selected_channels": ["meter"],
        "research_program": {"selection_steps": [{"operation": "discover_conditional_coordinates",
            "program": program.to_dict(), "question": "Locate the measured condition cut", "loss": 0.,
            "gate_coordinate": Program("feature", name="meter").to_dict(), "threshold_interval": [-.2, .2]}]},
        "actions": [], "transfer": [], "batches": batches, "research_graph": {}, "noise_assessment": None,
        "costs": {"observation_cost": 48, "measurement_cost": 48, "compute_cost": 0, "queries": 2}}
    config = ResearchConfig(observation_budget=96, compute_budget=24000, minimum_improvement=1e-7,
        max_rounds=1, use_memory=False, active_probes=False, seed=837301)
    report = UnifiedResearchAgent(config, measurement_budget=96).run(lab, previous=previous)
    checked = [event for event in report["actions"] if event["action"] == "validate_continued_program"]
    assert len(checked) == 1 and checked[0]["loss"] <= 1e-7
    assert checked[0]["current_fit_confirmed"] and not checked[0]["accepted"]
    assert report["status"] == "unresolved"
    state = report["condition_uncertainty"]
    assert state["status"] == "unresolved" and state["fit_rows"] == 96
    assert state["original_target_loss"] <= 1e-7
    assert state["expected_prediction_ambiguity"] > 1e-7
    assert report["continuation_state"]["condition_uncertainty"] == state
    assert any(item["condition_uncertainty"] == state
               for item in report["continuation_state"]["alternatives"])
    assert report["costs"]["observation_cost"] == lab.observation_cost == 96
    assert report["costs"]["measurement_cost"] == lab.measurement_cost == 96
    restored = UnifiedResearchAgent(config, measurement_budget=96).run(lab,
        previous=json.loads(json.dumps(report)))
    assert restored["status"] == "unresolved" and restored["condition_uncertainty"] == state
    assert restored["visit_costs"]["observation_cost"] == 0
