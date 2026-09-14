"""Conditional radius questions must activate a geometric edge in the full response."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_revisions import _expose
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_nested_relation_revisions import call, mse


GATE = Program("lt", children=(Program("arg", name="edge_target_quantity"),
                               Program("arg", name="edge_source_quantity")))


def conditional_composition(radius, direction):
    names = ("f0", "x0", "q0", "f1", "x1", "q1")
    arg = lambda name: Program("arg", name=name)
    feature = lambda name: Program("feature", name=name)
    relation = call("conditional_relation", names,
        Program("relation_sum", name=direction, children=(Program("const", value=radius), GATE,
                                                        *(arg(name) for name in names))),
        [feature(name) for name in names])
    event = call("event", ("value",), Program("event_delta", lag=1, children=(arg("value"),)), [relation])
    total = Program("add", children=(feature("q0"), feature("q1")))
    response = Program("add", children=(total, event))
    return call("window", ("value",), Program("window_sum", lag=3, children=(arg("value"),)), [response])


def observe(program, sequence):
    history, rows = [], []
    for fields in sequence:
        row = Observation(fields, 0., tuple(history))
        rows.append(replace(row, target=program.predict(row)))
        history.append(fields)
    return rows


@pytest.mark.parametrize("direction,sign", [("right", 1.), ("left", -1.)])
def test_conditional_radius_questions_use_positions_and_active_quantities_to_buy_a_full_probe(direction, sign):
    parent, actual = conditional_composition(.624, direction), conditional_composition(.630, direction)
    original = deepcopy(parent.to_dict())
    payload = {"selection_steps": [{"operation": "revise_relation_condition", "relation": direction,
        "predicate": GATE.to_dict(), "radius": .624, "radius_interval": [.616, .632]}]}
    base = {"f0": 1., "f1": 0., "x0": 0., "x1": sign * .624, "q0": .8, "q1": .1}
    paid = observe(parent, [{**base, "x1": sign * distance} for distance in (.1, 1.2) * 4])
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(parent, [], budget, research_program=payload)
    alternatives = research.alternatives(paid, budget)
    assert len(alternatives) == 3 and all(mse(entry["candidate"].program, paid) == 0. for entry in alternatives)
    questions = research.questions(alternatives)
    active = lambda fields: any(question["predicate"].predict(Observation(fields, 0.)) for question in questions)
    assert active(base)
    assert active({**base, "f0": 0., "f1": 1.})
    assert not active({**base, "x1": sign * .9})
    assert not active({**base, "q0": .1, "q1": .8})

    bounds = {"f0": (1., 1.), "f1": (0., 0.), "x0": (0., 0.),
              "x1": (-2., 2.), "q0": (-2., 2.), "q1": (-2., 2.)}
    spec = TaskSpec("conditional_radius", "", tuple(bounds), (), (), feature_bounds=bounds)
    plan = plan_competing_object_probe(spec, alternatives, budget, serial=53, length=9, questions=questions)
    assert plan["probe"] is not None
    assert plan["predicted_separation"] > 1e-7
    assert plan["generation_cost"] <= 6000
    acquired = observe(actual, plan["probe"].sequence)
    for hypothesis in plan["hypotheses"]:
        frozen = Program.from_dict(hypothesis["program"])
        assert hypothesis["predictions"] == [frozen.predict(row) for row in acquired]
    assessment = assess_competing_object_probe(plan, [(row, row) for row in acquired], budget)
    assert assessment["status"] == "distinguished"
    revised = research.alternatives([*paid, *acquired], budget)
    assert revised and all(mse(entry["candidate"].program, acquired) < 1e-7 for entry in revised)
    assert sum(high - low for low, high in research.parameters[0]["radius_intervals"]) < .016
    assert parent.to_dict() == original
    assert budget.spent <= 50000


def test_compact_interval_replacement_keeps_the_input_call_and_a_consumable_template():
    names = ("f0", "x0", "q0", "f1", "x1", "q1", "unused")
    feature = lambda name: Program("feature", name=name)
    arguments = tuple(Program("add", children=(Program("const", value=.03), feature(name)))
                      if name.startswith("x") else feature(name) for name in names)

    def parent_at(radius):
        def parameterize(node):
            if node.op == "feature":
                return Program("arg", name=node.name)
            return replace(node, children=tuple(parameterize(child) for child in node.children))
        body = _expose(conditional_composition(radius, "right"), SearchBudget(2000))
        return call("retained_condition", names, parameterize(body), arguments)

    parent = parent_at(.624)
    original = deepcopy(parent.to_dict())
    payload = {"selection_steps": [{"operation": "revise_relation_condition", "relation": "right",
        "predicate": GATE.to_dict(), "radius": .624, "radius_interval": [.616, .632],
        "compact_replacement": True}]}
    base = {"f0": 1., "f1": 0., "x0": 0., "x1": .1, "q0": .8, "q1": .1, "unused": 3.}
    paid = observe(parent, [{**base, "x1": distance} for distance in (.1, 1.2) * 4])
    budget = SearchBudget(30000)
    research = RelationIntervalResearch(parent, [], budget, research_program=payload)
    alternatives = research.alternatives(paid, budget)
    assert len(alternatives) == 3
    unobserved = observe(parent, [{**base, "x1": distance} for distance in (.1, .622, .628, 1.2, .1)])
    for entry in alternatives:
        program = entry["candidate"].program
        assert program.op == "call" and program.children == parent.children
        assert program.definition.arguments == parent.definition.arguments
        radius = next(iter(entry["parameter_values"].values()))
        assert [program.predict(row) for row in unobserved] == [parent_at(radius).predict(row) for row in unobserved]

    selected = alternatives[-1]
    restored = Program.from_dict(json.loads(json.dumps(selected["candidate"].program.to_dict())))
    saved = json.loads(json.dumps(selected["research_program"]))
    step = saved["selection_steps"][-1]
    assert Program.from_dict(step["parameter_template"]).op != "call"
    inherited = RelationIntervalResearch(restored, [], budget, research_program=saved)
    assert inherited.intervals[0]["radius_interval"] == [.616, .632]
    assert len(inherited.alternatives(paid, budget)) == 3
    assert parent.to_dict() == original
