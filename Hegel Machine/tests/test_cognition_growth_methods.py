"""Research traces must become executable, cheaper experiments on a new task."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.growth_methods import ResearchMethodBook, instantiate
from hegel_machine.cognition.growth_procedures import procedure_work
from hegel_machine.cognition.measurement_lab import MeasurementChannel
from hegel_machine.cognition.models import Observation, Probe, TaskSpec


class RenamedInstrumentLab:
    def __init__(self, *, renamed=False, coefficient=2.):
        self.position = "new_position" if renamed else "position"
        self.drive = "new_drive" if renamed else "drive"
        self.channels = ("wide_new", "fine_new") if renamed else ("wide", "fine")
        bounds = ({self.drive: (-5., 5.), self.position: (100., 200.)} if renamed else
                  {self.position: (-1., 1.), self.drive: (-2., 2.)})
        self.spec = TaskSpec("target" if renamed else "source", "Measure the response",
            tuple(bounds), (), (), feature_bounds=bounds,
            context={"control_roles": {self.position: "position", self.drive: "drive"}})
        self.coefficient = coefficient
        self.observation_cost = self.measurement_cost = 0

    def acquire(self, probe, channels):
        self.observation_cost += probe.cost
        self.measurement_cost += probe.cost * sum(3 if name == self.channels[1] else 1 for name in channels)
        low, high = self.spec.feature_bounds[self.position]
        result = []
        for row in probe.sequence:
            x = 2. * (row[self.position] - low) / (high - low) - 1.
            reading = x if channels == (self.channels[1],) else abs(x)
            features = ({"reading": reading} if len(channels) == 1 else
                        {name: x if name == self.channels[1] else abs(x) for name in channels})
            result.append(Observation(features, self.coefficient * x))
        return result

    def diagnostics(self, alias_error=1.):
        return {"feature_bounds": self.spec.feature_bounds,
                "control_roles": self.spec.context["control_roles"],
                "channel_names": self.channels, "alias_error": alias_error,
                "residual_error": 0., "prior_error": 0., "repeat_variance": 0.,
                "feature_count": 2}


def _slope(rows):
    return (sum(row.features["reading"] * row.target for row in rows) /
            sum(row.features["reading"] ** 2 for row in rows))


def _loss(rows, slope):
    return sum((row.target - slope * row.features["reading"]) ** 2 for row in rows) / len(rows)


def _sequence(lab, positions):
    low, high = lab.spec.feature_bounds[lab.position]
    return [{lab.position: low + (value + 1.) / 2. * (high - low), lab.drive: 0.}
            for value in positions]


def _discover(lab):
    """An actual unsuccessful coarse fit precedes the successful fine fit."""
    sequence = _sequence(lab, (-1., 0., 1.))
    validation = _sequence(lab, (-.75, .25, .75))
    coarse = lab.acquire(Probe("coarse", tuple(sequence)), (lab.channels[0],))
    fine = lab.acquire(Probe("fine", tuple(sequence)), (lab.channels[1],))
    coefficient = _slope(fine)
    confirmation = lab.acquire(Probe("confirm", tuple(validation)), (lab.channels[1],))
    assert _loss(coarse, _slope(coarse)) > 1.
    assert _loss(confirmation, coefficient) < 1e-12
    return [
        {"event_id": "coarse", "action": "measure", "produces": ["coarse_rows"],
         "recipe": {"sequence": sequence, "channel": lab.channels[0]}},
        {"event_id": "failed_fit", "action": "fit", "consumes": ["coarse_rows"],
         "produces": ["failed_model"], "success": False},
        {"event_id": "invalid_child", "action": "compose", "consumes": ["failed_model"],
         "produces": ["invalid_composition"]},
        {"event_id": "fine", "action": "measure", "produces": ["fine_rows"],
         "recipe": {"sequence": sequence, "channel": lab.channels[1]}},
        {"event_id": "fit", "action": "fit", "consumes": ["fine_rows"],
         "produces": ["model"], "recipe": {"estimator": "linear_through_origin"}},
        {"event_id": "confirm", "action": "confirm", "consumes": ["model"],
         "produces": ["confirmed"], "recipe": {"sequence": validation, "channel": lab.channels[1]}}]


def test_learned_evidence_dag_rebinds_and_reduces_paid_research_on_second_task():
    source = RenamedInstrumentLab()
    book = ResearchMethodBook()
    method = book.learn(_discover(source), source.diagnostics(), success=True, task_id="source")
    assert [node["action"] for node in method["nodes"]] == ["measure", "fit", "confirm"]
    assert [node["depends_on"] for node in method["nodes"]] == [[], ["step_0"], ["step_1"]]
    payload = json.loads(json.dumps(book.to_dict()))
    recovered = ResearchMethodBook.from_dict(payload)
    target = RenamedInstrumentLab(renamed=True, coefficient=3.)
    actions = instantiate(recovered.propose(target.diagnostics()), target.spec, target.channels)
    assert actions[0]["channels"] == ("fine_new",)
    repeated = instantiate(method, target.spec, target.channels, attempt=1)
    assert repeated[0]["probe"].probe_id != actions[0]["probe"].probe_id
    assert repeated[0]["probe"].sequence == actions[0]["probe"].sequence
    assert [row["new_position"] for row in actions[0]["probe"].sequence] == [100., 150., 200.]
    assert all(row["new_drive"] == 0. for row in actions[0]["probe"].sequence)
    assert set(actions[0]["probe"].sequence[0]) == {"new_position", "new_drive"}
    model = None
    for action in actions:
        if action["action"] == "measure":
            rows = target.acquire(action["probe"], action["channels"])
        elif action["action"] == "fit":
            model = _slope(rows)
        elif action["action"] == "confirm":
            measured_loss = _loss(target.acquire(action["probe"], action["channels"]), model)
    cold = RenamedInstrumentLab(renamed=True, coefficient=3.)
    _discover(cold)
    assert measured_loss < 1e-12 and model == pytest.approx(3.)
    assert (target.observation_cost, target.measurement_cost) == (6, 18)
    assert (cold.observation_cost, cold.measurement_cost) == (9, 21)
    # Learning cost is visible, not claimed as an immediately amortized win.
    assert source.observation_cost + target.observation_cost == 15
    assert source.measurement_cost + target.measurement_cost == 39


def test_failed_transfer_edits_executable_child_and_learns_diagnostic_scope():
    lab = RenamedInstrumentLab()
    original = _discover(lab)
    book = ResearchMethodBook()
    parent = book.learn(original, lab.diagnostics(alias_error=1.), success=True, task_id="first")
    modified = deepcopy(original)
    modified.insert(-1, {"event_id": "repeat", "action": "replicate", "consumes": ["model"],
        "produces": ["repeated_model"], "recipe": {"sequence": _sequence(lab, (-1., -1., 1., 1.)),
        "channel": lab.channels[1]}})
    modified[-1]["consumes"] = ["repeated_model"]
    changed = lab.diagnostics(alias_error=0.)
    changed["repeat_variance"] = .4
    book.record_outcome(parent["method_id"], changed, success=False, task_id="changed")
    child = book.revise(parent["method_id"], modified, changed, success=True, task_id="changed")
    assert child["parent_method_id"] == parent["method_id"] and child["version"] == 2
    assert [item["action"] for item in child["nodes"]] == ["measure", "fit", "replicate", "confirm"]
    assert child["nodes"][-1]["depends_on"] == ["step_2"]
    assert book.propose(changed)["method_id"] == child["method_id"]
    assert book.propose(lab.diagnostics())["method_id"] == parent["method_id"]
    assert next(item for item in book.methods if item["method_id"] == parent["method_id"])[
        "applicability"]["conditions"] == [{"feature": "alias_error", "op": ">", "value": .5}]
    executable = instantiate(child, lab.spec, lab.channels)
    assert executable[2]["probe"].cost == 4
    with pytest.raises(ValueError, match="change an executable"):
        book.revise(child["method_id"], modified, changed, success=True, task_id="no_edit")


def test_failed_unconfirmed_or_unbindable_workflows_are_not_suggested():
    lab = RenamedInstrumentLab()
    trace = _discover(lab)
    book = ResearchMethodBook()
    book.learn(trace, lab.diagnostics(), success=False, task_id="failed")
    assert book.propose(lab.diagnostics()) is None
    book = ResearchMethodBook()
    book.learn(trace, lab.diagnostics(), success=True, task_id="successful")
    assert book.propose({**lab.diagnostics(), "channel_names": ("only_coarse",)}) is None
    other = {**lab.diagnostics(), "feature_bounds": {"other": (-1., 1.)}, "control_roles": {"other": "other"}}
    assert book.propose(other) is None
    # Contradictory outcomes with indistinguishable diagnostics do not justify
    # a fabricated applicability condition that silently discards the failure.
    book.learn(trace, lab.diagnostics(), success=False, task_id="same_diagnostics_failed")
    assert book.propose(lab.diagnostics()) is None


def test_metadata_preserves_evidence_ancestry_and_only_measured_outcomes_change_scope():
    lab = RenamedInstrumentLab()
    trace = _discover(lab)
    # A bookkeeping stage aliases observations for the fit without becoming an
    # executable method stage or retaining its unrelated, bulky library state.
    trace.insert(-2, {"event_id": "retain", "action": "retain_knowledge",
        "consumes": ["fine_rows"], "produces": ["named_rows"],
        "recipe": {"library": {"irrelevant": [42] * 100}}})
    trace[-2]["consumes"] = ["named_rows"]
    book = ResearchMethodBook()
    parent = book.learn(trace, lab.diagnostics(), success=True, task_id="first")
    assert [item["action"] for item in parent["nodes"]] == ["measure", "fit", "confirm"]
    assert parent["nodes"][1]["depends_on"] == ["step_0"]
    assert "irrelevant" not in json.dumps(book.to_dict())
    modified = deepcopy(trace)
    modified[-2]["recipe"]["estimator"] = "new_estimator"
    book.revise(parent["method_id"], modified, lab.diagnostics(), success=True, task_id="edited")
    unchanged = next(item for item in book.methods if item["method_id"] == parent["method_id"])
    assert len(unchanged["examples"]) == 1
    # A failed replay is an observation even when no successful edit follows.
    measured = book.record_outcome(parent["method_id"], lab.diagnostics(),
        success=False, task_id="actual_failed_replay")
    assert measured["applicability"]["status"] == "conflicted"
    assert len(measured["examples"]) == 2


def test_failed_confirmation_is_data_and_instrument_choice_is_remeasured_after_rank_change():
    lab = RenamedInstrumentLab()
    train = _sequence(lab, (-1., 0., 1.))
    validation = _sequence(lab, (-.75, .25, .75))
    trace = [
        {"event_id": "pilot", "action": "measure", "produces": ["pilot_rows"],
         "recipe": {"sequence": train, "channels": lab.channels,
                    "channel_binding": "compare_available"}},
        {"event_id": "winner", "action": "fit", "consumes": ["pilot_rows"],
         "produces": ["fitted"], "recipe": {"instrument_comparison": {
             "candidate_channels": list(lab.channels), "selected_channel": lab.channels[1],
             "losses": {lab.channels[0]: 8. / 3., lab.channels[1]: 0.}}}},
        {"event_id": "challenge", "action": "confirm", "consumes": ["fitted"],
         "produces": ["challenge_rows"], "details": {"model_confirmed": False},
         "recipe": {"sequence": validation, "channels": [lab.channels[1]]}},
        {"event_id": "refit", "action": "fit", "consumes": ["pilot_rows", "challenge_rows"],
         "produces": ["refitted"]},
        {"event_id": "end", "action": "confirm", "consumes": ["refitted"],
         "details": {"model_confirmed": True},
         "recipe": {"sequence": validation, "channels": [lab.channels[1]]}},
    ]
    book = ResearchMethodBook()
    method = book.learn(trace, lab.diagnostics(), success=True, task_id="compared")
    assert [node["action"] for node in method["nodes"]] == [
        "measure", "fit", "vary_controls", "fit", "confirm"]
    target = RenamedInstrumentLab(renamed=True, coefficient=3.)
    # The published order changes independently of instrument meaning.
    published = (MeasurementChannel("fine_new", 3, "instrument"),
                 MeasurementChannel("wide_new", 1, "instrument"))
    actions = instantiate(method, target.spec, published)
    assert actions[0]["channels"] == ("fine_new", "wide_new")
    assert [question["channels"] for question in actions[1]["subquestions"]] == [
        ("fine_new",), ("wide_new",)]
    pilot = target.acquire(actions[0]["probe"], actions[0]["channels"])
    losses = {}
    for question in actions[1]["subquestions"]:
        rows = [Observation({"reading": row.features[question["channels"][0]]}, row.target)
                for row in pilot]
        losses[question["channels"]] = _loss(rows, _slope(rows))
    assert target.observation_cost == 3 and target.measurement_cost == 12
    selected = min(losses, key=losses.get)
    assert selected == ("fine_new",) and selected != (published[1].name,)
    assert actions[-1]["channels_from"] == "step_1"
    assert "channels" not in actions[-1]
    assert _loss(target.acquire(actions[-1]["probe"], selected), 3.) < 1e-12
    assert actions[3]["consumes"] == ["step_0:evidence", "step_2:evidence"]
    assert procedure_work(actions, published) == {
        "observation_cost": 9, "measurement_cost_upper_bound": 30, "fit_questions": 3}


def test_repeated_setup_questions_come_from_observed_trace_not_a_fixed_repeat_count():
    lab = RenamedInstrumentLab()
    sequence = _sequence(lab, (-1., -1., -1., 1., 1.))
    book = ResearchMethodBook()
    method = book.learn([
        {"action": "replicate", "recipe": {"sequence": sequence, "channels": []},
         "produces": ["repeats"]},
        {"action": "fit", "consumes": ["repeats"], "produces": ["fit"]},
        {"action": "confirm", "consumes": ["fit"]},
    ], lab.diagnostics(), success=True, task_id="repeat_source")
    action = instantiate(method, lab.spec, lab.channels)[0]
    assert action["probe"].cost == 5
    assert action["subquestions"][0]["groups"] == [3, 2]
    assert action["subquestions"][0]["consumes"] == action["produces"]
