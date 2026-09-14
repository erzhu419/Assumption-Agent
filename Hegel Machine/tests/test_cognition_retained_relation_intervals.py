"""Stored parameter uncertainty must remain executable inside consuming calls."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe)
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_nested_relation_revisions import composition, mse


def prior(interval=(.5, .7)):
    return {"prior_id": "event_prior", "source_concept_id": "event", "status": "supported", "structure": {
        "radius_interval": list(interval), "fits": [{"radius": .6, "family": "event_change",
                                                      "radius_interval": list(interval)}]}}


def observations(program, distances, *, seed=431, prefix=""):
    random, history, result = Random(seed), [], []
    for distance in distances:
        features = {prefix + name: value for name, value in {
            "f0": 1., "f1": 0., "x0": 0., "x1": distance,
            "q0": random.uniform(-2., 2.), "q1": random.uniform(-2., 2.)}.items()}
        row = Observation(features, 0., tuple(history))
        result.append(replace(row, target=program.predict(row)))
        history.append(features)
    return result


def task(program):
    names = tuple(sorted(program.required_features()))
    bounds = {name: ((1., 1.) if name.endswith("f0") else
                    (0., 0.) if name.endswith("f1") else (-2., 2.)) for name in names}
    return TaskSpec("inherited_relation", "", names, (), (), feature_bounds=bounds)


def test_zero_residual_still_exposes_full_renamed_window_and_event_alternatives():
    parent = composition(.6, "renamed_")
    original = deepcopy(parent.to_dict())
    paid = observations(parent, (.1, 1.2) * 6, prefix="renamed_")
    budget = SearchBudget(30000)
    research = RelationIntervalResearch(parent, [prior()], budget)
    alternatives = research.alternatives(paid, budget)
    assert len(alternatives) >= 2
    assert mse(parent, paid) == 0.
    assert all(mse(item["candidate"].program, paid) < 1e-7 for item in alternatives)

    # Exactly one unobserved boundary crossing occurs, followed by enough rows
    # to expose the event and the entire window, including its startup behavior.
    rows = observations(parent, (.1, .1, .6, 1.2, 1.2, 1.2, .1), prefix="renamed_")
    expected = [[composition(radius, "renamed_").predict(row) for row in rows] for radius in (.5, .7)]
    predicted = [[item["candidate"].program.predict(row) for row in rows] for item in alternatives]
    assert all(min(max(abs(a - b) for a, b in zip(values, reference))
                   for reference in expected) < 1e-6 for values in predicted)
    assert max(max(values) - min(values) for values in zip(*predicted)) > .1
    assert parent.to_dict() == original


def test_inherited_interval_drives_a_real_discriminating_experiment_and_uses_its_feedback():
    parent, actual = composition(.6), composition(.7)
    paid = observations(parent, (.1, 1.2) * 6)
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(parent, [prior()], budget)
    alternatives = research.alternatives(paid, budget)
    plan = plan_competing_object_probe(task(parent), alternatives, budget, serial=37,
        questions=research.questions(alternatives))
    assert plan["probe"] is not None
    assert plan["predicted_separation"] > 1e-7
    history, acquired = [], []
    for position, point in enumerate(plan["probe"].sequence):
        row = Observation(point, 0., tuple(history))
        acquired.append(replace(row, target=actual.predict(row)))
        for hypothesis in plan["hypotheses"]:
            assert hypothesis["predictions"][position] == Program.from_dict(hypothesis["program"]).predict(row)
        history.append(point)
    assessed = assess_competing_object_probe(plan, [(row, row) for row in acquired], budget)
    assert assessed["status"] == "distinguished"
    revised = research.alternatives([*paid, *acquired], budget)
    assert revised
    assert all(mse(item["candidate"].program, acquired) < 1e-7 for item in revised)
    assert budget.spent <= 50000


@pytest.mark.parametrize("program,priors", [
    (composition(), []),
    (composition(), [prior((.6, .6))]),
    (Program("feature", name="q0"), [prior()]),
])
def test_precise_or_absent_relation_knowledge_adds_no_competing_research(program, priors):
    rows = observations(program, (.1, 1.2) * 4)
    budget = SearchBudget(1000)
    research = RelationIntervalResearch(program, priors, budget)
    assert research.alternatives(rows, budget) == []
    assert research.questions([]) == []


def test_exhausted_budget_does_not_change_the_stored_definition():
    parent = composition()
    original = deepcopy(parent.to_dict())
    budget = SearchBudget(1)
    research = RelationIntervalResearch(parent, [prior()], budget)
    assert research.alternatives(observations(parent, (.1, 1.2) * 4), budget) == []
    assert parent.to_dict() == original
    assert budget.spent <= 1


def test_saved_derived_interval_survives_a_renamed_parameterized_call():
    parent, actual = composition(.6), composition(.7)
    paid = observations(actual, (.1, .58, 1.2) * 5)
    research = RelationIntervalResearch(parent, [prior()], SearchBudget(3000))
    selected = research.alternatives(paid, SearchBudget(30000))[-1]
    source_program = selected["candidate"].program
    names = tuple(sorted(source_program.required_features()))

    def parameterize(node):
        if node.op == "feature":
            return Program("arg", name=node.name)
        return replace(node, children=tuple(parameterize(child) for child in node.children))

    definition = Definition("derived:definition", names, parameterize(source_program), "derived")
    consumer = Program("call", name=definition.definition_id, definition=definition,
        children=tuple(Program("feature", name="fresh_" + name) for name in names))
    record = json.loads(json.dumps({"source_concept_id": "derived", "status": "supported",
                                   "research_program": selected["research_program"]}))
    expected = source_program.remap_features({name: "fresh_" + name for name in names})
    assert mse(consumer, observations(expected, (.1, .62, .66, 1.2, .1), prefix="fresh_")) < 1e-24

    inherited = RelationIntervalResearch(consumer, [record], SearchBudget(10000))
    assert inherited.intervals
    assert all(item["radius_interval"] == [.58, .7] for item in inherited.intervals)
    next_paid = observations(composition(.7, "fresh_"), (.1, .58, 1.2) * 5, prefix="fresh_")
    alternatives = inherited.alternatives(next_paid, SearchBudget(30000))
    assert len(alternatives) >= 2
    assert all(mse(item["candidate"].program, next_paid) < 1e-7 for item in alternatives)
    unobserved = observations(expected, (.1, .64, 1.2, .1), prefix="fresh_")
    values = [[item["candidate"].program.predict(row) for row in unobserved] for item in alternatives]
    assert max(max(row) - min(row) for row in zip(*values)) > .1
