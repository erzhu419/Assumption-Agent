"""Parameter revisions preserve full nonlinear consumers without growing deltas."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_nested_relation_revisions import call, composition, mse


def nonlinear_consumer(radius=.8, *, prefix="", coordinate_scale=1.):
    def calibrate(node):
        if node.op == "feature" and node.name.startswith(prefix + "x"):
            return Program("add", children=(Program("const", value=.03),
                Program("mul", children=(Program("const", value=coordinate_scale), node))))
        return replace(node, children=tuple(calibrate(child) for child in node.children))

    signal = calibrate(composition(radius, prefix))
    total = call("outside_total", ("left", "right"), Program("add", children=(
        Program("arg", name="left"), Program("arg", name="right"))),
        [Program("feature", name=prefix + "q0"), Program("feature", name=prefix + "q1")])
    arg = lambda name: Program("arg", name=name)
    response = Program("add", children=(
        Program("window_sum", lag=2, children=(Program("mul", children=(arg("signal"), arg("signal"))),)),
        Program("mul", children=(Program("const", value=.2), arg("baseline")))))
    return call("nonlinear_consumer", ("signal", "baseline"), response, [signal, total])


def physical_rows(distances, *, radius, prefix="", coordinate_scale=1.):
    """Physical exchange and its two consumer windows, without Program prediction."""
    rows, history, flows, mixtures, responses = [], [], [], [], []
    for index, distance in enumerate(distances):
        q0, q1 = .2 + .07 * (index % 3), 1.1 - .11 * (index % 4)
        fields = {prefix + name: value for name, value in {
            "f0": 1., "f1": 0., "x0": 0., "x1": distance, "q0": q0, "q1": q1}.items()}
        flow = q1 - q0 if abs(coordinate_scale * distance) < radius else 0.
        change = 0. if not flows else flow - flows[-1]
        flows.append(flow)
        mixtures.append(1.3 * (q0 + q1) - .7 * change)
        responses.append(sum(mixtures[-3:]))
        target = sum(value ** 2 for value in responses[-2:]) + .2 * (q0 + q1)
        rows.append(Observation(fields, target, tuple(history)))
        history.append({**fields, "__target__": target})
    return rows


def restore(value):
    return json.loads(json.dumps(value))


def wrap_saved(program, prefix, source="saved_nonlinear"):
    names = tuple(sorted(program.required_features()))

    def parameterize(node):
        if node.op == "feature":
            return Program("arg", name=node.name)
        return replace(node, children=tuple(parameterize(child) for child in node.children))

    definition = Definition(source, names, parameterize(program), source)
    return Program("call", name=source, definition=definition,
        children=tuple(Program("feature", name=prefix + name) for name in names))


@pytest.mark.parametrize("prefix,scale", [("", 1.), ("fresh_", 2.)])
def test_compact_repeated_parameter_research_preserves_nonlinear_nested_history(prefix, scale):
    parent = nonlinear_consumer(prefix=prefix, coordinate_scale=scale)
    original = deepcopy(parent.to_dict())
    distances = tuple(distance / scale for distance in (1.2, .1, .1, 1.2) * 3)
    paid = physical_rows(distances, radius=.8, prefix=prefix, coordinate_scale=scale)
    assert mse(parent, paid) < 1e-24
    procedure = None
    current = parent
    budget = SearchBudget(160000)
    for _ in range(3):
        research = RelationIntervalResearch(current, [], budget,
            research_program=procedure,
            structure=None if procedure else {"fits": [{"radius": .8}], "radius_interval": [.2, 1.]})
        entries = research.alternatives(paid, budget)
        assert len(entries) == 3
        for entry in entries:
            candidate = entry["candidate"].program
            radius = next(iter(entry["parameter_values"].values()))
            for sequence in ((.1,), (1.2, .6), (1.2, .4, .4, .9, .9, .3, 1.2)):
                held = physical_rows([distance / scale for distance in sequence], radius=radius,
                                     prefix=prefix, coordinate_scale=scale)
                assert mse(candidate, held) < 1e-22
            # A saved call must not make the full response grow by another old
            # response plus a correction on every subsequent parameter study.
            assert candidate.size <= parent.size
        selected = entries[-1]
        current = Program.from_dict(restore(selected["candidate"].program.to_dict()))
        procedure = restore(selected["research_program"])
    assert parent.to_dict() == original
    assert budget.spent < budget.limit


def test_compact_saved_nonlinear_counterexample_is_used_then_reopened_locally():
    parent = nonlinear_consumer()
    source_rows = physical_rows((1.2, .7, .7, .7, 1.2, 1.2, 1.2), radius=.8)
    budget = SearchBudget(180000)
    research = RelationIntervalResearch(parent, [], budget,
        structure={"fits": [{"radius": .8}], "radius_interval": [.2, 1.]})
    source_entries = research.alternatives(source_rows, budget, experiments=[source_rows])
    selected = source_entries[-1]
    source = Program.from_dict(restore(selected["candidate"].program.to_dict()))
    record = restore({"source_concept_id": "saved_nonlinear", "status": "supported",
                      "research_program": selected["research_program"]})
    original = deepcopy((parent.to_dict(), source.to_dict(), record))
    next_call = wrap_saved(source, "next_")
    inherited = RelationIntervalResearch(next_call, [record], budget)
    uninformative = physical_rows((1.2, .1, .1, 1.2) * 2, radius=.8, prefix="next_")
    entries = inherited.alternatives(uninformative, budget)
    held = physical_rows((1.2, .7, .7, .7, 1.2, 1.2), radius=.8, prefix="next_")
    assert entries and all(mse(item["candidate"].program, held) < 1e-22 for item in entries)
    assert all(mse(item["candidate"].program, uninformative) < 1e-22 for item in entries)

    changed = physical_rows((1.2, .6, .6, .6, 1.2, .45, .45, .45, 1.2, 1.2),
                            radius=.5, prefix="next_")
    assert mse(next_call, changed) > 1e-3
    revised = inherited.alternatives(changed, budget, experiments=[changed])
    assert revised and inherited.evidence_updates
    assert all(mse(item["candidate"].program, changed) < 1e-22 for item in revised)
    assert all(.45 < next(iter(item["parameter_values"].values())) < .6 for item in revised)
    assert (parent.to_dict(), source.to_dict(), record) == original
    assert mse(source, source_rows) < 1e-22
    # A return to the old mechanism still consumes the untouched saved law.
    returning = RelationIntervalResearch(wrap_saved(source, "return_"), [record], budget)
    returned_rows = physical_rows((1.2, .7, .7, 1.2, 1.2), radius=.8, prefix="return_")
    returned = returning.alternatives(returned_rows, budget)
    assert returned and all(mse(item["candidate"].program, returned_rows) < 1e-22 for item in returned)
    assert budget.spent < budget.limit
