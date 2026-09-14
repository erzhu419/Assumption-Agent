"""Observable research questions compete through actual prediction returns."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Observation, TaskSpec
from hegel_machine.cognition.research_opportunities import ResearchOpportunities, opportunity_condition
from test_cognition_experiment_frontier_runtime import _saved_question


def threshold(value=.5, name="x"):
    return Program("ge", children=(Program("feature", name=name), Program("const", value=value)))


def offer(*, name="x", task_id="source", action="transition", kind="experiment"):
    spec = TaskSpec(task_id, "Predict this measured response", (name,), (), (),
                    feature_bounds={name: (-1., 1.)}, max_probe_length=8)
    rows = [Observation({name: value}, float(value >= .35)) for value in (-.8, -.2, .2, .8)]
    design = {"boundary_support": [{"history_depth": 0, "previous_state": None,
               "free_controls": [name], "relative_width": .3}], "unresolved_boundary_width": .3}
    profile = method_profile(spec, rows, threshold(name=name), residual_loss=0., baseline_loss=.25,
                             boundary_evidence=design)
    return {"kind": kind, "action": action, "profile": profile, "design": design,
            "observation_cost": 8, "compute_cost": 400}, spec, rows


def measured(before, after, rows, *, baseline=.25, execution_complete=True):
    old, new = loss(before, rows, 0., .5), loss(after, rows, 0., .5)
    return {"execution_complete": execution_complete, "reference_rows": len(rows),
            "gain": old-new, "normalized_gain": (old-new) / baseline,
            "before_loss": old, "after_loss": new}


def rank(book, offers, **kwargs):
    return book.rank(offers, **{"observation_remaining": 100, "compute_remaining": 10000, **kwargs})


def test_known_method_and_new_zero_error_experiment_compete_without_width_becoming_gain():
    experiment, _, _ = offer()
    method = {**experiment, "kind": "method", "action": "representation", "steps": ["representation"],
              "observation_cost": 10, "compute_cost": 1000, "expected_gain": .4}
    book = ResearchOpportunities()
    quotes = rank(book, [experiment, method])
    assert [row["kind"] for row in quotes] == ["method", "experiment"]
    exploration = quotes[1]
    assert exploration["exploration"] and exploration["value"] == 0.
    assert exploration["expected_gain"] is None and exploration["investment_gain"] == 0.
    assert rank(book, [experiment])  # Zero old error still leaves this concrete first question open.
    resolved = deepcopy(experiment)
    resolved["design"] = {}
    resolved["profile"].pop("boundary")
    assert rank(book, [resolved]) == []


def test_actual_new_counterexample_teaches_the_experiment_to_compete_above_an_old_method():
    experiment, _, _ = offer()
    method = {**experiment, "kind": "method", "action": "representation", "expected_gain": .2,
              "observation_cost": 10, "compute_cost": 800}
    book = ResearchOpportunities()
    fresh = [Observation({"x": value}, float(value >= .35)) for value in (.4, -.5, .9, .2)]
    feedback = measured(threshold(.5), threshold(.35), fresh)
    book.observe(experiment, feedback, observation_cost=8, compute_cost=600)
    quote = rank(book, [method, experiment])[0]
    assert quote["kind"] == "experiment" and not quote["exploration"]
    assert quote["basis"] == "measured_continuation_return"
    assert quote["expected_gain"] == feedback["normalized_gain"] == 1.
    assert quote["expected_compute_cost"] == 600


@pytest.mark.parametrize("replacement,expected_gain", [(threshold(.45), 0.), (threshold(.75), -1.)])
def test_narrower_intervals_cannot_hide_neutral_or_harmful_prediction_returns(replacement, expected_gain):
    experiment, _, _ = offer()
    fresh = [Observation({"x": value}, float(value >= .35)) for value in (-.5, .2, .6, .9)]
    feedback = {**measured(threshold(.5), replacement, fresh), "interval_reduction": .25,
                "boundary_width_before": .3, "boundary_width_after": .05}
    book = ResearchOpportunities()
    record = book.observe(experiment, feedback, observation_cost=12, compute_cost=800)
    assert record["gain"] == expected_gain
    assert record["interval_reduction"] == .25
    narrowed = deepcopy(experiment)
    narrowed["design"]["boundary_support"][0]["relative_width"] = .01
    narrowed["design"]["unresolved_boundary_width"] = .01
    narrowed["profile"]["boundary"]["unresolved_width"] = .01
    narrowed["profile"]["boundary"]["branches"][0]["relative_width"] = .01
    narrowed["profile"]["observation_count"] += 40
    narrowed["profile"]["recent_actions"] = [{"action": "transition", "gain": expected_gain}]
    assert opportunity_condition(experiment) == opportunity_condition(narrowed)
    assert rank(book, [narrowed]) == []


def test_actual_overspending_changes_both_the_winner_and_whether_a_visit_can_execute():
    experiment, _, _ = offer()
    method = {**experiment, "kind": "method", "action": "representation", "expected_gain": .3,
              "observation_cost": 8, "compute_cost": 800}
    fresh = [Observation({"x": .4}, 1.)]
    book = ResearchOpportunities()
    feedback = measured(threshold(.5), threshold(.35), fresh)
    # Same measured gain for both observed continuations, but the experiment
    # takes far more fitting work than its original executable quote promised.
    feedback["normalized_gain"] = .5
    book.observe(experiment, feedback, observation_cost=12, compute_cost=6000)
    quotes = rank(book, [experiment, method])
    assert quotes[0]["kind"] == "method"
    assert quotes[1]["expected_compute_cost"] == 6000
    assert rank(book, [experiment], visit_compute_budget=5999) == []
    assert rank(book, [experiment], visit_observation_budget=11) == []
    assert rank(book, [experiment], visit_compute_budget=6000, visit_observation_budget=12)


@pytest.mark.parametrize("initial_response", [0., 1e-5])
def test_zero_or_tiny_initial_loss_cannot_buy_all_future_research_with_a_large_ratio(initial_response):
    experiment, spec, _ = offer()
    old, revised = Program("const", value=0.), Program("const", value=.25)
    initial = [Observation({"x": point}, initial_response) for point in (-.8, -.2, .2, .8)]
    baseline = loss(old, initial, 0., .5)
    assert baseline <= 1.000001e-10
    experiment.update(observation_cost=10, compute_cost=4000,
        profile=method_profile(spec, initial, old, residual_loss=baseline, baseline_loss=baseline))
    fresh = [Observation({"x": point}, .25) for point in (-.6, -.1, .3, .7)]
    feedback = measured(old, revised, fresh, baseline=max(baseline, 1e-8))
    assert feedback["gain"] == .0625 and feedback["normalized_gain"] == 6250000.
    method = {**experiment, "kind": "method", "action": "representation",
              "observation_cost": 10, "compute_cost": 400, "expected_gain": .5}
    book = ResearchOpportunities()
    record = book.observe(experiment, feedback, observation_cost=10, compute_cost=4000)
    assert record["gain"] == 1.
    assert record["raw_gain"] == .0625
    assert record["reported_normalized_gain"] == 6250000.
    assert record["reference_rows"] == len(fresh)
    restored = ResearchOpportunities(json.loads(json.dumps(book.to_dict())))
    quotes = rank(restored, [experiment, method])
    assert [quote["kind"] for quote in quotes] == ["method", "experiment"]
    assert quotes[1]["expected_gain"] == 1. and quotes[1]["value"] == 2.5


def test_large_measured_deterioration_keeps_its_actual_magnitude_but_bounds_allocation_return():
    experiment, _, _ = offer()
    rows = [Observation({"x": .4}, .25)]
    feedback = measured(Program("const", value=.25), Program("const", value=0.), rows, baseline=1e-8)
    book = ResearchOpportunities()
    record = book.observe(experiment, feedback, observation_cost=8, compute_cost=400)
    assert record["gain"] == -1.
    assert record["raw_gain"] == -.0625 and record["reported_normalized_gain"] == -6250000.
    assert rank(book, [experiment]) == []


def test_complete_nonpositive_residual_continuations_do_not_return_as_fresh_ordinary_offers():
    ordinary, _, _ = offer(kind="residual", action="research")
    ordinary["design"] = {}
    ordinary["profile"]["residual"]["state"] = "unresolved"
    ordinary["profile"].pop("boundary")
    book = ResearchOpportunities()
    assert rank(book, [ordinary])[0]["exploration"]
    book.observe(ordinary, {"execution_complete": True, "reference_rows": 8,
        "normalized_gain": 0., "gain": 0.}, observation_cost=16, compute_cost=1000)
    continued = {**ordinary, "task_id": "renamed", "seed": 901, "query": "another attempt"}
    continued["profile"] = deepcopy(ordinary["profile"])
    continued["profile"]["observation_count"] += 16
    assert rank(book, [continued]) == []


def test_new_control_or_binary_preparation_support_reopens_exploration_after_failure():
    experiment, _, _ = offer()
    book = ResearchOpportunities()
    book.observe(experiment, {"execution_complete": True, "reference_rows": 8,
        "normalized_gain": 0., "gain": 0.}, observation_cost=8, compute_cost=400)
    changed_controls = deepcopy(experiment)
    changed_controls["profile"]["data"]["coverage"][0]["bins"] = [0, 2, 3]
    assert rank(book, [changed_controls])[0]["exploration"]
    changed_preparation = deepcopy(experiment)
    changed_preparation["design"]["boundary_support"][0]["previous_state"] = True
    assert rank(book, [changed_preparation])[0]["exploration"]
    conditional = deepcopy(experiment)
    conditional["profile"]["data"]["conditional_coverage"] = [
        {"condition": "previous_response_1", "coverage": [{"role": 0, "bins": [0, 3]}]}]
    assert rank(book, [conditional])[0]["exploration"]


def test_unanalyzed_external_evidence_reopens_once_without_inventing_prediction_gain():
    ordinary, _, _ = offer(kind="residual", action="research")
    ordinary["design"] = {}
    ordinary["profile"].pop("boundary")
    book = ResearchOpportunities()
    neutral = {"execution_complete": True, "reference_rows": 8,
               "normalized_gain": 0., "gain": 0.}
    book.observe(ordinary, neutral, observation_cost=8, compute_cost=400)
    assert rank(book, [ordinary]) == []
    delivery = {**ordinary, "incoming_evidence": True}
    quote = rank(book, [delivery])[0]
    assert quote["exploration"] and quote["expected_gain"] is None
    assert quote["value"] == 0. and quote["investment_gain"] == 0.
    assert opportunity_condition(delivery)["incoming_evidence"] is True
    assert not delivery["profile"]["residual"]["new_counterexample"]
    book.observe(delivery, neutral, observation_cost=8, compute_cost=400)
    repeated = deepcopy(delivery)
    repeated["profile"]["observation_count"] += 32
    restored = ResearchOpportunities(json.loads(json.dumps(book.to_dict())))
    assert rank(restored, [repeated]) == []


def test_old_saved_zero_return_keeps_its_meaning_when_new_alternative_fields_were_absent():
    experiment, _, _ = offer()
    rows = [Observation({"x": value}, float(value >= .35)) for value in (-.5, .2, .6, .9)]
    book = ResearchOpportunities()
    book.observe(experiment, measured(threshold(.5), threshold(.45), rows),
                 observation_cost=8, compute_cost=400)
    legacy = json.loads(json.dumps(book.to_dict()))
    legacy["outcomes"][0]["condition"].pop("decision_alternatives")
    legacy["outcomes"][0]["condition"].pop("knowledge_alternatives")
    restored = ResearchOpportunities(legacy)
    assert restored.estimate(experiment)["trials"] == 1
    assert restored.estimate(experiment)["expected_gain"] == 0.
    assert rank(restored, [experiment]) == []
    assert restored.to_dict() == legacy
    newly_fitted = deepcopy(experiment)
    representation = method_profile(TaskSpec("new", "Observed response", ("x",), (), ()),
                                    [], Program("feature", name="x"))["representation"]
    structure = {key: representation[key] for key in ("kind", "families", "history_depth", "structure")}
    newly_fitted["knowledge_alternatives"] = [structure]
    assert rank(restored, [newly_fitted])[0]["exploration"]
    decision = deepcopy(experiment)
    decision["design"]["alternatives"] = [{"representation": structure}]
    assert rank(restored, [decision])[0]["exploration"]


@pytest.mark.parametrize("complete,reference_rows", [(False, 4), (True, 0)])
def test_incomplete_or_uncompared_work_has_cost_but_does_not_manufacture_failure(complete, reference_rows):
    experiment, _, _ = offer()
    book = ResearchOpportunities()
    record = book.observe(experiment, {"execution_complete": complete, "reference_rows": reference_rows,
        "normalized_gain": -1., "interval_reduction": .3}, observation_cost=4, compute_cost=600)
    assert record["gain"] is None and record["compute_cost"] == 600
    quote = rank(book, [experiment])[0]
    assert quote["exploration"] and quote["trials"] == 0 and quote["incomplete_trials"] == 1


def test_normalization_and_role_conditions_survive_output_scaling_names_and_json_resume():
    def scaled(scale, name):
        item, spec, _ = offer(name=name, task_id=name, action="refit", kind="revision")
        old = Program("mul", children=(Program("const", value=.5*scale), Program("feature", name=name)))
        new = Program("mul", children=(Program("const", value=scale), Program("feature", name=name)))
        rows = [Observation({name: value}, scale*value) for value in (-.8, -.2, .2, .8)]
        baseline = loss(Program("const", value=0.), rows, 0., .5)
        item.update(design={}, profile=method_profile(spec, rows, old,
            residual_loss=loss(old, rows, 0., .5), baseline_loss=baseline))
        return item, measured(old, new, rows, baseline=baseline)

    source, first = scaled(1., "x")
    target, second = scaled(100., "renamed")
    assert first["normalized_gain"] == pytest.approx(second["normalized_gain"])
    assert opportunity_condition(source) == opportunity_condition(target)
    source.update(task_id="private_task_name", seed=123, method_id="run_local_99")
    book = ResearchOpportunities()
    book.observe(source, first, observation_cost=8, compute_cost=500)
    saved = json.dumps(book.to_dict())
    assert "private_task_name" not in saved and "run_local_99" not in saved and '"seed"' not in saved
    restored = ResearchOpportunities(json.loads(saved))
    assert rank(restored, [target]) == rank(book, [target])
    assert rank(restored, [target])[0]["trials"] == 1


def test_measured_knowledge_return_remains_separate_from_current_prediction_and_execution_cost():
    item, _, _ = offer(kind="method", action="representation")
    item.update(expected_gain=0., expected_knowledge_gain=.4, knowledge_compute_cost=1200)
    quote = rank(ResearchOpportunities(), [item], visit_compute_budget=400)[0]
    assert quote["expected_gain"] == 0. and quote["expected_knowledge_gain"] == .4
    assert quote["expected_compute_cost"] == 400 and quote["investment_compute_cost"] == 1600
    assert not quote["exploration"]


def test_real_zero_error_saved_experiment_receives_its_executed_continuation_feedback():
    session, environment, workspace = _saved_question()
    frontier = workspace.experiment_frontier
    item = {"kind": "experiment", "action": "transition", "design": frontier["design"],
        "profile": method_profile(environment.spec, workspace.observations, workspace.incumbent,
            residual_loss=0., baseline_loss=workspace.initial_loss, boundary_evidence=frontier["design"]),
        "observation_cost": frontier["observation_cost"] + frontier["comparison_observation_cost"],
        "compute_cost": session.config.compute_budget}
    book = ResearchOpportunities()
    assert rank(book, [item], compute_remaining=session.config.compute_budget)[0]["exploration"]
    result = CognitiveAgent(session.config).run(environment, research_state=workspace.to_dict())
    assert any(action.action == "execute_research_question" for action in result.actions)
    feedback = next(action.details for action in result.actions if action.action == "continuation_feedback")
    narrowed = result.research_state["experiment_frontier"]
    assert narrowed["uncertainty"] < frontier["uncertainty"]
    record = book.observe(item, {**feedback, "execution_complete": True,
        "interval_reduction": frontier["uncertainty"] - narrowed["uncertainty"]},
        observation_cost=result.observation_cost, compute_cost=result.compute_cost)
    assert record["reference_rows"] > 0
    assert record["gain"] == feedback["normalized_gain"]
    assert record["observation_cost"] == sum(probe.cost for probe in environment.queries)
    assert record["compute_cost"] == result.compute_cost
    # This real experiment narrows the problem, but the common representative
    # rows did not distinguish the old and new predictor. Keep that zero.
    assert record["interval_reduction"] > 0. and record["gain"] == 0.
    assert book.estimate(item)["expected_gain"] == 0.
    assert rank(book, [item], compute_remaining=session.config.compute_budget) == []
