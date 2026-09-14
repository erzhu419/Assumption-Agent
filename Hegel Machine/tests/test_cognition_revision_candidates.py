"""Select explanations with paid prediction evidence and informative controls."""

from dataclasses import replace
from random import Random
from statistics import mean

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.revision_candidates import (
    distinguishing_controls, propose_competitors, select_competitor,
)


def _spec(names=("x",)):
    return TaskSpec("candidate_comparison", "Predict observations", names, (), (),
                    feature_bounds={name: (0., 1.) for name in names})


def _candidate(identifier, program, cv_loss=0.):
    return {"candidate_id": identifier, "program": program.to_dict(),
            "training_loss": 0., "cv_loss": cv_loss}


def test_training_predictions_can_discover_a_nonlinear_second_coordinate_among_nuisance_inputs():
    spec = _spec(("x1", "x2", "x3"))
    rng = Random(481)
    rows = []
    for _ in range(16):
        point = {name: rng.random() for name in spec.feature_names}
        rows.append(Observation(point, .7 + 1.3 * point["x2"] ** 2 + rng.gauss(0., .003)))
    budget = SearchBudget(20000)
    candidates = propose_competitors(spec, rows, budget)
    best = candidates[0]
    learned = Program.from_dict(best["program"])
    assert learned.required_features() == {"x2"}
    assert best["cv_loss"] < .0001
    assert len(best["cv_errors"]) == len(rows)
    assert len(rows) < budget.spent <= budget.limit
    for point in ({"x1": .99, "x2": .25, "x3": .01},
                  {"x1": .01, "x2": .75, "x3": .99}):
        assert learned.predict(Observation(point, 0.)) == pytest.approx(.7 + 1.3 * point["x2"] ** 2, abs=.015)


def test_distinguishing_measurements_can_choose_a_quadratic_over_a_shorter_linear_explanation():
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    candidates = [_candidate("linear", x), _candidate("quadratic", square)]
    budget = SearchBudget(1000)
    controls = distinguishing_controls(_spec(), candidates, budget, seed=19, count=4)
    assert len(controls) == 4 and .4 < controls[0]["x"] < .6
    assert budget.spent >= len(candidates) * len(controls)
    # Both hypotheses agree at the previously seen endpoints. Every selected
    # control is measured once, after both predictions have been recorded.
    rows = []
    for point in controls:
        predictions = {item["candidate_id"]: Program.from_dict(item["program"]).predict(Observation(point, 0.))
                       for item in candidates}
        rows.append({"observation": {"target": point["x"] ** 2}, "candidate_predictions": predictions})
    selected, measured = select_competitor(candidates, rows, noise_variance=0., minimum_error=1e-7, maximum_loss=1.)
    assert selected["candidate_id"] == "quadratic"
    assert selected["validation_loss"] == 0.
    assert len(measured) == 2
    assert all(len(item["validation_errors"]) == 4 for item in measured)
    assert mean(item["linear"] - item["quadratic"] for item in (row["candidate_predictions"] for row in rows)) > .1


def test_an_unmeasured_candidate_cannot_borrow_another_candidates_validation():
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    candidates = [_candidate("linear", x, .1), _candidate("quadratic", square)]
    rows = [{"observation": {"target": value ** 2},
             "candidate_predictions": {"linear": value, **({"quadratic": value ** 2} if index else {})}}
            for index, value in enumerate((.2, .4, .6, .8))]
    selected, measured = select_competitor(candidates, rows, noise_variance=0., minimum_error=1e-7, maximum_loss=1e-6)
    assert selected is None
    assert [item["candidate_id"] for item in measured] == ["linear"]


def _decision_spec():
    return replace(_spec(("x1", "x2", "x3")),
                   context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})


def _decision_rows(objective, constraint=lambda point: -1.):
    rng = Random(615)
    points = [{name: rng.uniform(.05, .95) for name in ("x1", "x2", "x3")} for _ in range(16)]
    return [{"observation": {"features": point, "target": constraint(point)},
             "context": {"objective": objective(point)}} for point in points]


def _feasible_competitors():
    x1 = Program("feature", name="x1")
    first = Program("sub", children=(x1, Program("const", value=2.)))
    second = Program("sub", children=(Program("mul", children=(x1, x1)), Program("const", value=2.)))
    return [_candidate("linear_constraint", first), _candidate("quadratic_constraint", second)]


def test_reversing_paid_objective_evidence_moves_the_goal_probe_along_the_relevant_coordinate():
    controls = []
    for direction in (1., -1.):
        budget = SearchBudget(20000)
        chosen = distinguishing_controls(_decision_spec(), _feasible_competitors(), budget,
            seed=39, count=4, decision_rows=_decision_rows(lambda point: direction * point["x2"]))
        assert len(chosen) == 4
        controls.append(chosen)
    assert controls[0][0]["x2"] < .01
    assert controls[1][0]["x2"] > .99


def test_paid_quadratic_objective_evidence_can_move_the_goal_probe_to_an_interior_minimum():
    objective = lambda point: (point["x2"] - .63) ** 2
    budget = SearchBudget(20000)
    rows = _decision_rows(objective)
    controls = distinguishing_controls(_decision_spec(), _feasible_competitors(), budget,
        seed=39, count=4, decision_rows=rows)
    assert len(controls) == 4
    assert controls[0]["x2"] == pytest.approx(.63, abs=.01)
    assert objective(controls[0]) < min(row["context"]["objective"] for row in rows)


def test_a_constraint_excluding_the_old_goal_moves_the_probe_into_a_predicted_feasible_region():
    x2 = Program("feature", name="x2")
    constraints = [_candidate(f"lower_bound_{bound}",
        Program("sub", children=(Program("const", value=bound), x2))) for bound in (.4, .45)]
    objective = lambda point: point["x2"]
    rows = _decision_rows(objective, lambda point: .4 - point["x2"])
    controls = distinguishing_controls(_decision_spec(), constraints, SearchBudget(20000),
        seed=39, count=4, decision_rows=rows)
    assert len(controls) == 4
    assert .4 <= controls[0]["x2"] < .47
    assert min(Program.from_dict(item["program"]).predict(Observation(controls[0], 0.))
               for item in constraints) <= 0.


def test_goal_planning_spends_compute_and_preserves_real_discriminating_measurements():
    spec, candidates = _decision_spec(), _feasible_competitors()
    plain = SearchBudget(20000)
    distinguishing_controls(spec, candidates, plain, seed=39, count=4)
    augmented = SearchBudget(20000)
    details = {}
    controls = distinguishing_controls(spec, candidates, augmented, seed=39, count=4,
        decision_rows=_decision_rows(lambda point: -point["x2"]), details=details)
    assert len(controls) == 4 and plain.spent < augmented.spent <= augmented.limit
    assert details["compute_cost"] == augmented.spent
    assert details["objective_model"]["training_rows"] == 16
    separating = [item for item in details["selected"] if item["purpose"] == "hypothesis_disagreement"]
    goal = [item for item in details["selected"] if item["purpose"].startswith("objective_")]
    assert len(separating) >= 2 and 1 <= len(goal) <= 2
    assert all(item["disagreement"] > 0 for item in separating)
    # Goal probes do not replace the entire comparison: the two remaining
    # controls still produce different predictions from the frozen hypotheses.
    assert len({tuple(sorted(point.items())) for point in controls}) == 4
    programs = [Program.from_dict(item["program"]) for item in candidates]
    assert sum(abs(programs[0].predict(Observation(point, 0.))
                   - programs[1].predict(Observation(point, 0.))) > .01
               for point in controls[2:]) == 2
    limited = SearchBudget(1)
    incomplete = distinguishing_controls(spec, candidates, limited, seed=39, count=4,
        decision_rows=_decision_rows(lambda point: -point["x2"]))
    assert len(incomplete) < 4 and limited.spent <= limited.limit
