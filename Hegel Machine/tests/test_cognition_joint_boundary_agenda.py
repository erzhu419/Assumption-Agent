"""A final revision can use startup and mature evidence from one paid query."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition import probing
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec


def _paid_case():
    payload = json.loads((Path(__file__).parent / "fixtures" /
                          "cognition_v018_final_s01_agenda.json").read_text())
    rows = [Observation(row["features"], row["target"], tuple(row["history"]))
            for row in payload["observations"]]
    used = [Probe(item["probe_id"], tuple(item["sequence"])) for item in payload["used"]]
    spec = TaskSpec("public_apparatus", "Predict the controlled response", tuple(payload["feature_bounds"]),
                    (), (), feature_bounds=payload["feature_bounds"],
                    max_probe_length=payload["max_probe_length"])
    return payload, spec, rows, Program.from_dict(payload["program"]), used


def _plan(spec, rows, program, used, *, remaining=20, rounds=1):
    budget = SearchBudget(1500)
    probe, details = probing.transition_probe(spec, [Candidate("retained", program, "state")], rows,
        budget=budget, remaining=remaining, round_number=3, used=used,
        followup_observation_cost=16, remaining_rounds=rounds)
    return probe, details, budget


def test_last_revision_collects_startup_and_both_mature_boundaries_from_paid_trace():
    payload, spec, rows, program, used = _paid_case()
    short, short_details, short_budget = _plan(spec, rows, program, used, rounds=2)
    joint, details, budget = _plan(spec, rows, program, used)
    assert payload["old_plan"]["selected_probe_id"] == "transition_startup_3"
    assert short.cost == 1 and short_details["design"] == "startup_interval_refinement"
    assert details["design"] == "joint_boundary_revision"
    assert joint.cost == details["observation_quote"] == 13
    # This is a different allocation of already priced options, not uncharged
    # extra prediction work when the final round becomes scarce.
    assert budget.spent == short_budget.spent == 95
    assert {item["previous_state"] for item in details["boundary_support"]} == {None, False, True}
    assert all(item["scheduled"] for item in details["pending_boundaries"])

    # Only the public observation method is used; no evaluation row is read.
    environment = make_suite(payload["seed"], "stateful")[payload["source_task_index"] - 53]
    acquired = environment.observe(joint)
    outcome = probing.transition_outcome(details, acquired)
    assert outcome["observed_rows"] == joint.cost
    assert outcome["reached_boundaries"] == outcome["attempted_boundaries"] == 3
    assert not outcome["unreached_boundary_indices"]
    actual = [item for item in outcome["targets"] if item["condition_reached"]]
    assert {item["previous_state"] for item in actual} == {None, False, True}
    # The old query finds only the startup error. The joint trajectory also
    # observes a contrary mature response within the still broad lower gap.
    assert any(item["phase"] == "startup" and
               program.predict(acquired[item["observation_index"]]) != item["observed_response"]
               for item in actual)
    mature = next(item for item in actual if item["previous_state"] is True)
    assert mature["refines_interval"]
    assert program.predict(acquired[mature["observation_index"]]) != mature["observed_response"]
    a, b = mature["observed_interval"]
    assert mature["observed_coordinate"] == pytest.approx((a + b) / 2)


def test_unaffordable_mature_questions_remain_pending_after_short_query():
    _, spec, rows, program, used = _paid_case()
    probe, details, _ = _plan(spec, rows, program, used, remaining=4)
    assert probe.cost == 1
    assert all(item["phase"] == "startup" for item in details["boundary_targets"])
    pending = [item for item in details["pending_boundaries"] if not item["scheduled"]]
    assert {item["previous_state"] for item in pending} == {False, True}
    assert max(item["relative_width"] for item in pending) > .2


def test_single_control_agenda_retains_distinct_history_conditions():
    _, spec, rows, program, used = _paid_case()
    spec = TaskSpec("single_control", spec.description, ("x",), (), (),
                    feature_bounds={"x": spec.feature_bounds["x"]}, max_probe_length=16)
    single_rows = [Observation({"x": row.features["x"]}, row.target,
                              tuple({"x": past["x"], "__target__": past["__target__"]}
                                    for past in row.history)) for row in rows]
    single_used = [Probe(probe.probe_id, tuple({"x": point["x"]} for point in probe.sequence)) for probe in used]
    probe, details, _ = _plan(spec, single_rows, program, single_used)
    assert details["policy"] == "conditional_transition_interval"
    assert probe.cost == 9  # One reset response and two complete paired blocks.
    assert {item["previous_state"] for item in details["boundary_support"]} == {None, False, True}


def test_preparation_success_is_observed_at_the_required_lag_not_assumed():
    design = {"state_midpoint": .5, "boundary_targets": [
        {"boundary_index": 0, "observation_index": 2, "phase": "mature",
         "history_depth": 2, "previous_state": True, "coordinate": {"x": 1.},
         "observed_interval": [-.8, .8]},
        {"boundary_index": 1, "observation_index": 3, "phase": "mature",
         "history_depth": 2, "previous_state": False, "coordinate": {"x": 1.},
         "observed_interval": [-.8, .8]}]}
    # Upper extreme commands produced opposite responses across preparation
    # rows. The relevant state is two rows ago, not the most recent response.
    history = [{"x": 1., "__target__": 0.}, {"x": 1., "__target__": 1.}]
    rows = [Observation({"x": 1.}, 0.), Observation({"x": 1.}, 1., tuple(history[:1])),
            Observation({"x": 0.}, 0., tuple(history)),
            Observation({"x": 0.}, 0., tuple([*history, {"x": 0., "__target__": 0.}]))]
    outcome = probing.transition_outcome(design, rows)
    assert [item["prepared_state"] for item in outcome["targets"]] == [False, True]
    assert not any(item["condition_reached"] or item["refines_interval"] for item in outcome["targets"])
    assert outcome["unreached_boundary_indices"] == [0, 1]
    # A startup proposal also cannot claim success if an apparatus actually
    # returns it with an existing trajectory history.
    startup = {"state_midpoint": .5, "boundary_targets": [
        {**design["boundary_targets"][0], "observation_index": 2, "phase": "startup",
         "history_depth": 0, "previous_state": None}]}
    assert not probing.transition_outcome(startup, rows)["targets"][0]["condition_reached"]


def test_joint_agenda_charges_projection_and_each_prospective_prediction(monkeypatch):
    _, spec, rows, program, used = _paid_case()
    calls = []
    evaluate = probing.predict

    def measured(program, row, fallback):
        calls.append(row)
        return evaluate(program, row, fallback)

    monkeypatch.setattr(probing, "predict", measured)
    probe, details, budget = _plan(spec, rows, program, used)
    assert probe.cost <= 20
    # The retained program has one current-control coordinate. Its acquired
    # row projection is paid once; all other charged work is actual prediction.
    assert budget.spent == len(rows) + len(calls)
    before = len(calls)
    probing.transition_outcome(details, rows[:probe.cost])
    assert len(calls) == before
