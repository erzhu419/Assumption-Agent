"""A changed transport law must survive its learned event/window context."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe)
from hegel_machine.cognition.equivalent_object_programs import canonical_object_program
from hegel_machine.cognition.relation_revisions import propose_relation_revisions
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch


RADIUS = .8


def retained_composition(radius=RADIUS):
    arg = lambda name: Program("arg", name=name)
    feature = lambda name: Program("feature", name=name)

    def call(name, arguments, body, children):
        definition = Definition(name, tuple(arguments), body, name)
        return Program("call", name=name, definition=definition, children=tuple(children))

    names = tuple(f"{role}{index}" for index in range(3) for role in ("f", "x", "q"))
    relation = call("relation", names, Program("neighbor_sum", children=(
        Program("const", value=radius), *(arg(name) for name in names))), [feature(name) for name in names])
    total = call("total", ("a", "b", "c"), Program("object_sum", children=tuple(arg(name) for name in "abc")),
                 [feature(f"q{index}") for index in range(3)])
    event = call("event", ("flow",), Program("event_delta", lag=1, children=(arg("flow"),)), [relation])
    mixed = call("mixed", ("total", "event"), Program("add", children=(
        Program("mul", children=(Program("const", value=1.3), arg("total"))),
        Program("mul", children=(Program("const", value=-.7), arg("event"))))), [total, event])
    return call("window", ("value",), Program("window_sum", lag=3, children=(arg("value"),)), [mixed])


def physical_flow(fields, direction, radius=RADIUS):
    quantities = [fields[f"q{index}"] for index in range(3)]
    positions = [fields[f"x{index}"] for index in range(3)]
    changes = [0., 0., 0.]
    for source in range(3):
        for target in range(3):
            if target == source:
                continue
            distance = positions[target] - positions[source]
            active = (abs(distance) < radius if direction == "symmetric" else
                      0. < distance < radius if direction == "right" else 0. < -distance < radius)
            if active:
                changes[source] -= quantities[source]
                changes[target] += quantities[source]
    assert abs(sum(changes)) < 1e-12
    return sum(fields[f"f{index}"] * changes[index] for index in range(3))


def evidence(direction, seed, lengths=(7, 11, 17), prefix=""):
    random, rows = Random(seed), []
    for length in lengths:
        points = [{"f0": 1., "f1": 0., "f2": 0., "x0": 0.,
                   "x1": -random.uniform(.1, 1.2), "x2": random.uniform(.1, 1.2),
                   **{f"q{index}": random.uniform(-2., 2.) for index in range(3)}} for _ in range(length)]
        # The same quantities and absolute distances make every symmetric
        # radius model predict equally on these two complete trajectories.
        for reflection in (1., -1.):
            history, previous_flow, values = [], None, []
            for original in points:
                fields = {name: value * reflection if name.startswith("x") else value
                          for name, value in original.items()}
                flow = physical_flow(fields, direction)
                delta = 0. if previous_flow is None else flow - previous_flow
                values.append(1.3 * sum(fields[f"q{index}"] for index in range(3)) - .7 * delta)
                renamed = {prefix + name: value for name, value in fields.items()}
                rows.append(Observation(renamed, sum(values[-3:]), tuple(history)))
                history.append(renamed)
                previous_flow = flow
    return rows


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def task(program):
    return TaskSpec("relation_type", "", tuple(sorted(program.required_features())), (), ())


@pytest.mark.parametrize("direction", ["right", "left"])
def test_direction_revision_changes_full_predictions_where_every_symmetric_radius_is_ambiguous(direction):
    parent = retained_composition()
    original = deepcopy(parent.to_dict())
    rows = evidence(direction, 823)
    mirror = evidence(direction, 827, lengths=(7,))
    assert all(parent.predict(a) == parent.predict(b) for a, b in zip(mirror[:7], mirror[7:]))
    assert max(abs(a.target - b.target) for a, b in zip(mirror[:7], mirror[7:])) > .1
    assert mse(parent, rows) > .01

    budget = SearchBudget(50000)
    proposals = propose_relation_revisions(task(parent), rows, budget, [parent])
    assert proposals
    selected = min(proposals, key=lambda item: item["training_loss"])
    # Selection uses purchased training rows; independent new trajectories
    # include reset-only and short histories as well as a mature window.
    assert mse(selected["candidate"].program, evidence(direction, 829, (1, 2, 7, 19))) < 1e-7
    assert parent.to_dict() == original
    assert "window" in selected["candidate"].parents
    assert budget.spent <= budget.limit


def test_revised_relation_remains_executable_after_json_restore_and_role_rename():
    parent = retained_composition()
    rows = evidence("right", 839)
    proposals = propose_relation_revisions(task(parent), rows, SearchBudget(50000), [parent])
    selected = min(proposals, key=lambda item: item["training_loss"])
    restored = Program.from_dict(json.loads(json.dumps(selected["candidate"].program.to_dict())))
    renamed = restored.remap_features({name: "changed_" + name for name in parent.required_features()})
    assert mse(renamed, evidence("right", 853, (1, 3, 13), "changed_")) < 1e-7
    assert mse(parent, evidence("symmetric", 857, (1, 3, 13))) < 1e-24


def test_symmetric_evidence_does_not_propose_a_new_transport_type():
    parent = retained_composition()
    rows = evidence("symmetric", 859)
    assert mse(parent, rows) < 1e-24
    budget = SearchBudget(10000)
    assert propose_relation_revisions(task(parent), rows, budget, [parent]) == []
    assert budget.spent < budget.limit


def test_insufficient_budget_preserves_the_existing_callable_law():
    parent = retained_composition()
    original = deepcopy(parent.to_dict())
    rows = evidence("left", 863)
    for limit in (0, 1, 50):
        budget = SearchBudget(limit)
        assert propose_relation_revisions(task(parent), rows, budget, [parent]) == []
        assert budget.spent <= limit
    assert parent.to_dict() == original


def physical_response(row, direction, radius=RADIUS):
    frames = [*row.history, row.features]
    result = 0.
    for index in range(max(0, len(frames) - 3), len(frames)):
        flow = physical_flow(frames[index], direction, radius)
        event = 0. if index == 0 else flow - physical_flow(frames[index - 1], direction, radius)
        result += 1.3 * sum(frames[index][f"q{group}"] for group in range(3)) - .7 * event
    return result


def test_opposite_transport_laws_plan_and_resolve_a_complete_response_experiment():
    parent = retained_composition()
    entries = [min(propose_relation_revisions(task(parent), evidence(direction, 877),
                   SearchBudget(50000), [parent]), key=lambda item: item["training_loss"])
               for direction in ("right", "left")]
    assert canonical_object_program(entries[0]["candidate"].program) != canonical_object_program(
        entries[1]["candidate"].program)
    # Different research calls may independently number their winning entry 0.
    # The experimental hypotheses use distinct task-level identities.
    entries = [{**entry, "candidate": replace(entry["candidate"], candidate_id=f"direction:{index}")}
               for index, entry in enumerate(entries)]
    bounds = {name: ((1., 1.) if name == "f0" else (0., 0.) if name.startswith("f") else (-1.2, 1.2))
              for name in parent.required_features()}
    spec = replace(task(parent), feature_bounds=bounds)
    budget = SearchBudget(80000)
    plan = plan_competing_object_probe(spec, entries, budget, serial=89, length=9)
    assert plan["probe"] is not None
    assert plan["predicted_separation"] > 1e-7
    history, measured = [], []
    for position, fields in enumerate(plan["probe"].sequence):
        row = Observation(fields, 0., tuple(history))
        measured.append(replace(row, target=physical_response(row, "right")))
        for hypothesis in plan["hypotheses"]:
            assert hypothesis["predictions"][position] == Program.from_dict(hypothesis["program"]).predict(row)
        history.append(fields)
    assert measured[0].target == pytest.approx(1.3 * sum(measured[0].features[f"q{i}"] for i in range(3)))
    assessment = assess_competing_object_probe(plan, [(row, row) for row in measured], budget)
    assert assessment["status"] == "distinguished"
    assert mse(entries[0]["candidate"].program, measured) < 1e-7
    assert mse(entries[1]["candidate"].program, measured) > 1e-7
    assert budget.spent <= budget.limit


def test_direction_interval_revision_preserves_a_symmetric_same_radius_component():
    parent = retained_composition()
    paid = evidence("right", 881, (7, 11))
    revised = propose_relation_revisions(task(parent), paid, SearchBudget(50000), [parent])[0]
    directional = revised["candidate"].program
    whole = Program("add", children=(directional,
        Program("mul", children=(Program("const", value=.4), parent))))
    # The consuming answer contains both relation types at the same radius.
    # Only the uncertain directional component is named by this revision.
    training = [replace(row, target=row.target + .4 * physical_response(row, "symmetric")) for row in paid]
    payload = json.loads(json.dumps(revised["research_program"]))
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(whole, [], budget, research_program=payload)
    assert research.intervals and all(item["relation"] == "right" for item in research.intervals)
    alternatives = research.alternatives(training, budget)
    assert len(alternatives) >= 2
    low, high = research.intervals[0]["radius_interval"]
    history = [{"f0": 1., "f1": 0., "f2": 0., "x0": 0., "x1": -1.2, "x2": 1.2,
                "q0": .3, "q1": 1.1, "q2": -.7}]
    rows = []
    for distance in (low + .4 * (high - low), low + .6 * (high - low), 1.2):
        fields = {**history[-1], "x1": -distance, "x2": distance}
        rows.append(Observation(fields, 0., tuple(history)))
        history.append(fields)
    for alternative in alternatives:
        radius = next(iter(alternative["parameter_values"].values()))
        for row in rows:
            expected = physical_response(row, "right", radius) + .4 * physical_response(row, "symmetric", RADIUS)
            assert alternative["candidate"].program.predict(row) == pytest.approx(expected, abs=1e-6)
    assert budget.spent <= budget.limit
