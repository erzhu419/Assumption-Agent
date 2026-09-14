"""Observed residuals select executable edge conditions inside retained calls."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_revisions import propose_relation_revisions
from test_cognition_relation_type_revisions import (
    RADIUS, evidence as unconditional_evidence, mse, retained_composition, task)


def greater(source, target):
    return source > target


def lesser(source, target):
    return source < target


def greater_and_positive(source, target):
    return source > target and source > 0.


def greater_or_positive(source, target):
    return source > target or source > 0.


def conditional_flow(fields, direction, condition):
    changes = [0., 0., 0.]
    for source in range(3):
        for target in range(3):
            distance = fields[f"x{target}"] - fields[f"x{source}"]
            in_range = 0. < (distance if direction == "right" else -distance) < RADIUS
            source_quantity, target_quantity = fields[f"q{source}"], fields[f"q{target}"]
            if in_range and condition(source_quantity, target_quantity):
                changes[source] -= source_quantity
                changes[target] += source_quantity
    assert abs(sum(changes)) < 1e-12
    return sum(fields[f"f{index}"] * changes[index] for index in range(3))


def trajectory(points, direction, condition, prefix=""):
    previous_flow, history, inner_values, rows = None, [], [], []
    for point in points:
        flow = conditional_flow(point, direction, condition)
        event = 0. if previous_flow is None else flow - previous_flow
        inner_values.append(1.3 * sum(point[f"q{index}"] for index in range(3)) - .7 * event)
        features = {prefix + name: value for name, value in point.items()}
        rows.append(Observation(features, sum(inner_values[-3:]), tuple(history)))
        history.append(features)
        previous_flow = flow
    return rows


def conditional_evidence(direction, condition, seed, lengths=(11, 13, 17), prefix=""):
    random, rows = Random(seed), []
    for length in lengths:
        points = []
        for step in range(length):
            focus = step % 3
            point = {**{f"f{index}": float(index == focus) for index in range(3)},
                     "x0": 0., "x1": -random.uniform(.1, 1.2), "x2": random.uniform(.1, 1.2),
                     **{f"q{index}": random.uniform(-2., 2.) for index in range(3)}}
            if step % 7 == 4:
                point["q1"] = point["q0"]
            points.append(point)
        rows.extend(trajectory(points, direction, condition, prefix))
    return rows


def learn(parent, rows):
    budget = SearchBudget(12000)
    # No condition/direction labels enter the TaskSpec or proposal arguments.
    entries = propose_relation_revisions(task(parent), rows, budget, [parent])
    assert entries, "paid residuals should yield a relation-condition explanation"
    chosen = min(entries, key=lambda item: item["training_loss"])
    assert chosen["training_loss"] < 1e-7
    assert budget.spent <= budget.limit
    return chosen


@pytest.mark.parametrize("direction,condition", [("right", greater), ("left", lesser)])
def test_quantity_comparison_is_discovered_inside_the_complete_temporal_response(direction, condition):
    parent = retained_composition()
    original = deepcopy(parent.to_dict())
    rows = conditional_evidence(direction, condition, 883)
    assert mse(parent, rows) > .01
    chosen = learn(parent, rows)
    program = chosen["candidate"].program
    assert mse(program, conditional_evidence(direction, condition, 887, (1, 2, 7, 29))) < 1e-7
    steps = chosen["research_program"]["selection_steps"]
    assert any(step["operation"] == "revise_relation_condition" for step in steps)

    # Geometry stays fixed. Quantity order alone changes whether transport is
    # present; a radius/position-only replacement cannot supply this gate.
    a = {"f0": 1., "f1": 0., "f2": 0., "x0": 0., "x1": -.4, "x2": 1.2,
         "q0": .2, "q1": 1.4, "q2": -.5}
    b = {**a, "q0": a["q1"], "q1": a["q0"]}
    controlled = trajectory([a, b, a, a, b, b, a], direction, condition)
    opposite = trajectory([a, b, a, a, b, b, a], direction, lesser if condition is greater else greater)
    assert mse(program, controlled) < 1e-7
    assert max(abs(left.target - right.target) for left, right in zip(controlled, opposite)) > .1
    assert parent.to_dict() == original


@pytest.mark.parametrize("direction,condition", [
    ("right", greater_and_positive), ("left", greater_or_positive)])
def test_same_generator_combines_two_comparison_atoms(direction, condition):
    parent = retained_composition()
    rows = conditional_evidence(direction, condition, 907, (17, 19, 23))
    chosen = learn(parent, rows)
    restored = Program.from_dict(json.loads(json.dumps(chosen["candidate"].program.to_dict())))
    renamed = restored.remap_features({name: "new_" + name for name in parent.required_features()})
    assert mse(renamed, conditional_evidence(direction, condition, 911, (1, 2, 4, 31), "new_")) < 1e-7
    # Both atomic conditions disagree with the composed target on fresh rows.
    # This prevents a single named quantity-order template satisfying the test.
    combined = conditional_evidence(direction, condition, 919, (23,))
    order_only = conditional_evidence(direction, greater, 919, (23,))
    positive_only = conditional_evidence(direction, lambda source, target: source > 0., 919, (23,))
    assert sum((a.target - b.target) ** 2 for a, b in zip(combined, order_only)) > .1
    assert sum((a.target - b.target) ** 2 for a, b in zip(combined, positive_only)) > .1


def test_conditional_replacement_preserves_prior_success_without_layered_expression_growth():
    parent = retained_composition()
    original = deepcopy(parent.to_dict())
    first = learn(parent, conditional_evidence("right", greater, 929))["candidate"].program
    first_saved = json.loads(json.dumps(first.to_dict()))
    second = learn(first, conditional_evidence("right", lesser, 937))["candidate"].program
    second_saved = json.loads(json.dumps(second.to_dict()))
    third = learn(second, conditional_evidence("right", greater, 941))["candidate"].program
    assert mse(first, conditional_evidence("right", greater, 947, (1, 3, 23))) < 1e-7
    assert mse(second, conditional_evidence("right", lesser, 953, (1, 3, 23))) < 1e-7
    assert mse(third, conditional_evidence("right", greater, 967, (1, 3, 23))) < 1e-7
    assert parent.to_dict() == original and first.to_dict() == first_saved and second.to_dict() == second_saved
    # Replacing one predicate twice should stay comparable to its first new
    # explanation, rather than repeatedly duplicating every ancestor program.
    assert max(second.size, third.size) <= 3 * first.size


def test_successful_law_and_exhausted_budget_do_not_force_condition_expansion():
    parent = retained_composition()
    budget = SearchBudget(5000)
    assert propose_relation_revisions(task(parent), unconditional_evidence("symmetric", 971), budget, [parent]) == []
    assert budget.spent < budget.limit
    rows = conditional_evidence("right", greater_and_positive, 977)
    for limit in (0, 1, 40):
        limited = SearchBudget(limit)
        assert propose_relation_revisions(task(parent), rows, limited, [parent]) == []
        assert limited.spent <= limit
