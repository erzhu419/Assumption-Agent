"""Future paid labels resolve a frozen goal candidate at planned round ends.

The real saved trajectory below tests evidence consumption, not a counterfactual
optimization result. No test in this file buys a simulator observation.
"""

from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, variance
from math import sqrt

import pytest

from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch
from hegel_machine.cognition.validation_continuation import evidence_summary, plan_validation_round


@pytest.fixture(scope="module")
def saved_goal_failure():
    path = (Path(__file__).resolve().parents[1] / "artifacts" /
        "sc_kg_active_noise_v046_20260910/objective_continuation/replay/runs" /
        "value_precision_3_mechanism_change_noise_0.03_objective_continuation.json")
    return json.loads(path.read_text())


def _frozen_goal(saved):
    names = ("x1", "x2", "x3")
    spec = TaskSpec("saved_goal_validation", "Minimize the measured objective", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    state = deepcopy(saved["objective_research"])
    research = deepcopy(state["scope_attempts"].pop())
    research.update(phase="validating", validation=research["validation"][:3])
    research.pop("status", None)
    state.update(scope_research=research, pending=None, shared_pending=None,
                 shared_predictions=[], shared_outcomes=[], accepted_revision=None)
    rows = []
    for point, output in zip(saved["initial_points"], saved["initial_outputs"]):
        rows.append({"observation": {"features": dict(zip(names, (x / 100. for x in point))),
                                    "target": output[1]},
                     "context": {"global_index": len(rows), "stage": len(rows) + 1,
                                 "objective": output[0]}})
    rows.extend(deepcopy(saved["recovery"]["observations"][:25]))
    assert len(rows) == 57
    return ObjectiveResearch(spec, state=state), rows


def _consume_saved_next(controller, rows, saved):
    index = len(rows)
    record, action = saved["simulations"][index - 32], saved["actions"][index - 32]
    point = dict(zip(controller.spec.feature_names, (x / 100. for x in record["point"])))
    before = len(controller.state["scope_research"]["validation"])

    def forecast(features):
        assert len(rows) == index
        assert len(controller.state["scope_research"]["validation"]) == before
        # These are the saved public forecasts from before this real label.
        return {"features": deepcopy(features), "objective_mean": action["mu_before"][0],
                "objective_noise": .001, "feasible": True}

    request = controller.prepare_shared_control(rows, forecast, point,
        scope_start=0, stage=index, remaining_calls=64 - index, purpose="optimization")
    assert request is not None and request["expected_global_index"] == index
    context = {"global_index": index, "stage": index + 1, "purpose": "optimization",
               "objective": record["output"][0]}
    rows.append({"observation": {"features": deepcopy(point), "target": record["output"][1]},
                 "context": context})
    controller.observe(point, record["output"], context=context)
    return request


def _empirical_lower(validation):
    gains = [row["previous_error"] - row["candidate_error"] for row in validation]
    return mean(gains) - 2 * sqrt(variance(gains) / len(gains))


def test_real_positive_but_uncertain_goal_survives_until_a_complete_future_pair(saved_goal_failure):
    controller, rows = _frozen_goal(saved_goal_failure)
    candidate = deepcopy(controller.state["scope_research"]["candidate"])
    installed = deepcopy(controller.state["model"])
    _consume_saved_next(controller, rows, saved_goal_failure)
    research = controller.state["scope_research"]
    assert research is not None and research["phase"] == "validating"
    assert research["status"] == "inconclusive_future_predictions"
    assert len(research["validation"]) == 4 and research["validation_target"] == 6
    assert _empirical_lower(research["validation"]) < 0
    assert controller.accepted_revision is None
    assert research["candidate"] == candidate and controller.state["model"] == installed

    _consume_saved_next(controller, rows, saved_goal_failure)
    assert len(research["validation"]) == 5
    assert _empirical_lower(research["validation"]) > 0
    assert controller.accepted_revision is None  # The round was planned to end at six.
    assert research["candidate"] == candidate and controller.state["model"] == installed

    _consume_saved_next(controller, rows, saved_goal_failure)
    revision = controller.accepted_revision
    assert revision is not None and revision["validation_indices"] == list(range(54, 60))
    assert revision["model"] == candidate and controller.state["model"] == installed
    assert revision["validation_mse"] < revision["previous_mse"]
    assert set(revision["training_indices"]).isdisjoint(revision["validation_indices"])
    assert len(controller.state["shared_predictions"]) == len(controller.state["shared_outcomes"]) == 3
    assert len(controller.state["requests"]) == len(saved_goal_failure["objective_research"]["requests"])


def test_json_resume_preserves_the_unfinished_pair_without_retrospective_refitting(saved_goal_failure):
    controller, rows = _frozen_goal(saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    state = json.loads(json.dumps(controller.to_dict()))
    restored = ObjectiveResearch(controller.spec, state=state)
    assert restored.to_dict() == state
    assert restored.accepted_revision is None
    _consume_saved_next(restored, rows, saved_goal_failure)
    assert restored.accepted_revision["model"] == state["scope_research"]["candidate"]
    assert restored.state["fits"] == state["fits"]
    assert restored.state["model"] == state["model"]


def test_a_repeated_control_cannot_finish_a_continuation_pair(saved_goal_failure):
    controller, rows = _frozen_goal(saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    before = deepcopy(controller.state["scope_research"])
    point = deepcopy(before["validation"][-1]["features"])
    request = controller.prepare_shared_control(rows,
        lambda features: pytest.fail("an already tested control cannot add another independent point"),
        point, scope_start=0, stage=len(rows), remaining_calls=6, purpose="optimization")
    assert request is None and controller.state["scope_research"] == before
    assert controller.accepted_revision is None


def test_future_evidence_with_negative_mean_still_rejects_the_candidate(saved_goal_failure):
    controller, rows = _frozen_goal(saved_goal_failure)
    research = controller.state["scope_research"]
    for item in research["validation"]:
        item.update(previous_error=0., candidate_error=1.)
    index = len(rows)
    point = {"x1": .97, "x2": .13, "x3": .57}
    controller._record_request(point, "objective_confirmation", "validate_objective_scope", index,
                              previous_mean=0., candidate_mean=1.)
    installed = deepcopy(controller.state["model"])
    controller.observe(point, [0., -1.], context={"global_index": index,
        "stage": index + 1, "purpose": "objective_confirmation"})
    assert controller.state["scope_research"] is None
    assert controller.accepted_revision is None and controller.state["model"] == installed
    assert controller.state["scope_attempts"][-1]["status"] == "future_predictions_do_not_improve"


def _competing_controls():
    return [{"features": {"x1": .01, "x2": .02, "x3": .03},
             "objective_mean": 0., "candidate_mean": 2., "objective_noise": .001, "feasible": True},
            {"features": {"x1": .99, "x2": .98, "x3": .97},
             "objective_mean": 2., "candidate_mean": 0., "objective_noise": .001, "feasible": True}]


def test_a_supported_midpoint_still_prices_the_last_control_of_its_unfinished_round(saved_goal_failure):
    controller, rows = _frozen_goal(saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    validation = controller.state["scope_research"]["validation"]
    assert evidence_summary(validation)["decision"] == "supported"
    assert controller.accepted_revision is None
    before = deepcopy(validation)
    budget = SearchBudget(1000)
    plan = plan_validation_round(validation, _competing_controls(),
        target_count=6, remaining_calls=1, budget=budget)
    assert plan is not None and plan["new_calls"] == 1
    assert plan["expected_gain"] > 0 and plan["target_count"] == 6
    assert budget.spent == plan["evaluated_scenarios"] > 0
    assert validation == before


@pytest.mark.parametrize("case", ["one_call", "no_compute", "same_ranking", "already_tested",
                                  "infeasible_alternative", "duplicate_control"])
def test_unaffordable_or_decision_irrelevant_pairs_do_not_buy_measurements(saved_goal_failure, case):
    validation = deepcopy(saved_goal_failure["objective_research"]["scope_attempts"][-1]["validation"])
    assert evidence_summary(validation)["decision"] == "inconclusive"
    controls = _competing_controls()
    remaining, budget = 2, SearchBudget(1000)
    if case == "one_call":
        remaining = 1
    elif case == "no_compute":
        budget = SearchBudget(0)
    elif case == "same_ranking":
        controls[0]["candidate_mean"], controls[1]["candidate_mean"] = 10., 12.
    elif case == "already_tested":
        controls[0]["features"] = deepcopy(validation[0]["features"])
    elif case == "infeasible_alternative":
        controls[1]["feasible"] = False
    elif case == "duplicate_control":
        controls[1]["features"] = deepcopy(controls[0]["features"])
    before = deepcopy((validation, controls))
    assert plan_validation_round(validation, controls, target_count=6,
                                 remaining_calls=remaining, budget=budget) is None
    assert (validation, controls) == before
    assert budget.spent == 0


@pytest.mark.parametrize("remaining", [1, 2, 7])
def test_shared_continuation_keeps_the_frozen_candidate_without_extra_planning(saved_goal_failure, remaining):
    controller, rows = _frozen_goal(saved_goal_failure)
    _consume_saved_next(controller, rows, saved_goal_failure)
    frozen = deepcopy(controller.state["scope_research"]["candidate"])
    fits = deepcopy(controller.state["fits"])
    cost = controller.state["compute_cost"]
    assert controller.next_control(rows,
        lambda features: pytest.fail("shared continuation must not scan extra probes"),
        scope_start=0, stage=len(rows), remaining_calls=remaining) is None
    assert controller.state["scope_research"]["candidate"] == frozen
    assert controller.state["scope_research"]["validation_target"] == 6
    assert controller.state["fits"] == fits and controller.state["compute_cost"] == cost
