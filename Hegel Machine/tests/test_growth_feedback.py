"""Fresh failure and view selection must remain attached to their actual causes."""

import json
from random import Random

from hegel_machine.cognition import research_growth
from hegel_machine.cognition.growth_methods import instantiate
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.models import Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_growth import GrowthResearcher


def constant(value):
    return Program("const", value=value)


def add(*terms):
    result = terms[0]
    for term in terms[1:]:
        result = Program("add", children=(result, term))
    return result


def measured_fit(program, training, validation, budget):
    # This controlled proposer changes the candidate, never the observations
    # or their evaluated loss. It is a regression fixture, not a benchmark arm.
    assert budget.spend(len(training) + len(validation))
    mse = lambda rows: sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
    return {"program": program, "loss": mse(validation), "training_loss": mse(training),
            "events": [], "terms": [], "alternatives": [], "used_priors": [],
            "scope_boundaries": []}


class SmoothLab:
    channels = ()

    def __init__(self):
        self.spec = TaskSpec("smooth", "Predict the response", ("x",), (), (),
                             feature_bounds={"x": (-1., 1.)}, max_probe_length=64)
        self.observation_cost = 0

    def observe(self, probe):
        self.observation_cost += probe.cost
        return [Observation(dict(point), 1. + point["x"] ** 2) for point in probe.sequence]


def test_fresh_counterexample_marks_replayed_parent_despite_zero_internal_error(monkeypatch):
    x = Program("feature", name="x")
    correct = add(constant(1.), Program("mul", children=(x, x)))
    monkeypatch.setattr(research_growth, "discover", lambda training, validation, budget, **_:
                        measured_fit(correct, training, validation, budget))
    config = ResearchConfig(seed=419, observation_budget=256, compute_budget=20000, max_rounds=2)
    prefix = GrowthResearcher(config)
    assert prefix.run(SmoothLab())["status"] == "confirmed"
    saved = json.loads(json.dumps(prefix.to_dict()))
    parent_id = saved["methods"]["methods"][0]["method_id"]

    def memorizer(training, validation, budget, **_):
        # Narrow intervals interpolate every actual fitting/selection sample,
        # but predict zero at newly purchased random controls.
        program = constant(0.)
        for row in [*training, *validation]:
            value = row.features["x"]
            at_sample = Program("and", children=(
                Program("ge", children=(x, constant(value - 1e-12))),
                Program("lt", children=(x, constant(value + 1e-12)))))
            program = Program("if", children=(at_sample, constant(row.target), program))
        return measured_fit(program, training, validation, budget)

    monkeypatch.setattr(research_growth, "discover", memorizer)
    researcher = GrowthResearcher(config, state=saved)
    lab = SmoothLab()
    result = researcher.run(lab)
    replay = next(i for i, event in enumerate(result["events"]) if event["action"] == "replay_method")
    replay_fit = next(event for event in result["events"][replay:] if event["action"] == "fit")
    assert replay_fit["details"]["loss"] == 0.
    confirmation = next(event for event in result["events"][replay:]
                        if event["action"] == "confirm" and replay_fit["event_id"] in event["consumes"])
    assert confirmation["details"]["mse"] > 1.
    assert confirmation["details"]["model_confirmed"] is False
    parent = next(method for method in researcher.methods.methods if method["method_id"] == parent_id)
    assert [example["success"] for example in parent["examples"]] == [True, False]
    assert result["status"] != "confirmed"
    assert result["costs"]["observation_cost"] == lab.observation_cost


def test_exhausted_search_preserves_its_paid_nonzero_model_and_stops_acquisition(monkeypatch):
    def expensive_mean(training, validation, budget, **_):
        program = constant(sum(row.target for row in training) / len(training))
        result = measured_fit(program, training, validation, budget)
        # The candidate generator used the rest of its allocation; subsequent
        # acquisition cannot fund another fit or confirmation calculation.
        budget.spend(budget.remaining)
        return result

    monkeypatch.setattr(research_growth, "discover", expensive_mean)
    researcher = GrowthResearcher(ResearchConfig(seed=419, observation_budget=256,
                                                compute_budget=100, max_rounds=3))
    lab = SmoothLab()
    result = researcher.run(lab)
    assert result["status"] == "budget_exhausted"
    assert result["costs"]["compute_cost"] == 100
    assert result["costs"]["observation_cost"] == lab.observation_cost == 24
    assert Program.from_dict(result["program"]).value > 1.
    assert not researcher.library.concepts


class ChangingInputLab:
    channels = (MeasurementChannel("signal", 1, "instrument"),
                MeasurementChannel("decoy", 1, "instrument"))

    def __init__(self):
        self.spec = TaskSpec("measured", "Predict the response", ("x",), (), (),
                             feature_bounds={"x": (-1., 1.)}, max_probe_length=64)
        self.random = Random(239)
        self.observation_cost = self.measurement_cost = 0

    def acquire(self, probe, channels=()):
        rows = []
        for point in probe.sequence:
            signal, decoy = self.random.uniform(-1., 1.), self.random.uniform(-1., 1.)
            readings = {"signal": signal, "decoy": decoy}
            rows.append(Observation({**point, **{name: readings[name] for name in channels}},
                                    1. + point["x"] + signal))
        self.observation_cost += probe.cost
        self.measurement_cost += probe.cost * len(channels)
        return MeasurementBatch(rows, probe.cost, probe.cost * len(channels), tuple(channels))


def test_later_refit_keeps_paid_instrument_selection_in_its_executable_ancestry(monkeypatch):
    def incomplete_then_refined(training, validation, budget, **_):
        program = add(constant(1.), Program("feature", name="x"))
        if "signal" in training[0].features:
            # A useful view can be adopted before its first response is exact;
            # the next acquisition lets this controlled proposer refine it.
            coefficient = .9 if len(training) + len(validation) <= 48 else 1.
            program = add(program, Program("mul", children=(constant(coefficient),
                                                            Program("feature", name="signal"))))
        return measured_fit(program, training, validation, budget)

    monkeypatch.setattr(research_growth, "discover", incomplete_then_refined)
    config = ResearchConfig(seed=419, observation_budget=384, compute_budget=20000, max_rounds=3)
    researcher, lab = GrowthResearcher(config, measurement_budget=1024), ChangingInputLab()
    result = researcher.run(lab)
    assert result["status"] == "confirmed" and result["selected_channels"] == ["signal"]
    choice = next(event for event in result["events"] if event["action"] == "change_representation")
    winner = next(event for event in result["events"] if event["event_id"] in choice["consumes"])
    assert 1e-6 < winner["details"]["loss"] < .02
    refit = next(event for event in result["events"]
                 if event["action"] == "fit" and choice["event_id"] in event["consumes"])
    assert refit["details"]["loss"] < 1e-12
    stored = researcher.methods.methods[-1]
    actions = instantiate(stored, lab.spec, lab.channels)
    comparison = next(action for action in actions if action.get("subquestions"))
    assert [question["channels"] for question in comparison["subquestions"]] == [("signal",), ("decoy",)]
    final_fit = [action for action in actions if action["action"] == "fit"][-1]
    assert comparison["node_id"] in final_fit["depends_on"]
    assert actions[-1]["channels_from"] == comparison["node_id"]
    assert result["costs"]["observation_cost"] == lab.observation_cost
    assert result["costs"]["measurement_cost"] == lab.measurement_cost
