"""Executable edge predicates keep lexical scope through calls and history."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.equivalent_object_programs import canonical_object_program
from hegel_machine.cognition.language import Definition, EvaluationContext, Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.relation_language import EDGE_ARGUMENTS, relation_objects


def node(op, *children, **kwargs):
    return Program(op, children=children, **kwargs)


def arg(name):
    return Program("arg", name=name)


def relation(predicate, direction="right"):
    return node("relation_sum", Program("const", value=.9), predicate,
        *(Program("feature", name=f"{role}{index}") for index in range(3) for role in ("f", "x", "q")),
        name=direction)


def rows():
    history, result = [], []
    for values in ((-.6, .4, -.1), (.6, -.4, .1), (.2, .2, .2)):
        features = {f"{role}{index}": value for index in range(3)
            for role, value in (("f", float(index == 1)), ("x", .6 * (index - 1)), ("q", values[index]))}
        result.append(Observation(features, 0., tuple(history)))
        history.append(features)
    return result


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("greater", [False, True])
def test_generic_comparison_executes_the_same_signed_edge_flux_as_its_expansion(direction, greater):
    source, target = arg("edge_source_quantity"), arg("edge_target_quantity")
    predicate = node("lt", target, source) if greater else node("not", node("ge", source, target))
    program = relation(predicate, direction)
    expanded = program.expanded()
    for row in rows():
        q = [row.features[f"q{i}"] for i in range(3)]
        order = ((0, 1), (1, 2)) if direction == "right" else ((1, 0), (2, 1))
        expected = sum((float(j == 1) - float(i == 1)) * q[i] for i, j in order
                       if (q[i] > q[j] if greater else q[i] < q[j]))
        assert program.predict(row) == pytest.approx(expected)
        assert expanded.predict(row) == pytest.approx(expected)
    assert program.op == "relation_sum"
    assert expanded.op == "object_sum"
    assert program.required_features() == expanded.required_features()
    assert not any(name in program.required_features() for name in EDGE_ARGUMENTS)


def test_edge_arguments_shadow_outer_call_arguments_but_external_thresholds_remain_bound():
    predicate = node("and", node("lt", arg("edge_target_quantity"), arg("edge_source_quantity")),
        node("lt", arg("threshold"), arg("edge_source_quantity")))
    body = node("add", relation(predicate), arg("edge_source_quantity"))
    definition = Definition("outer", ("edge_source_quantity", "threshold"), body)
    call = node("call", Program("const", value=11.), Program("const", value=-2.),
                name="outer", definition=definition)
    direct = relation(node("lt", arg("edge_target_quantity"), arg("edge_source_quantity")))
    for row in rows():
        assert call.predict(row) == pytest.approx(11. + direct.predict(row))
        assert call.expanded().predict(row) == pytest.approx(call.predict(row))


def test_relation_passed_as_call_argument_is_keyed_with_its_bound_predicate():
    predicate = node("lt", arg("edge_source_quantity"), arg("edge_target_quantity"))
    program = relation(predicate)
    definition = Definition("twice", ("value",), node("add", arg("value"), arg("value")))
    call = node("call", program, name="twice", definition=definition)
    row = rows()[0]
    context = EvaluationContext(row)
    assert call.predict(row, context) == 2 * program.predict(row)
    assert call.predict(row, context) == call.expanded().predict(row)
    assert context.cache_hits > 0


def test_predicate_calls_history_and_field_renaming_survive_json():
    comparison = Definition("edge_test", ("a", "b"), node("lt", arg("a"), arg("b")))
    predicate = node("call", arg("edge_target_quantity"), arg("edge_source_quantity"),
                     name="edge_test", definition=comparison)
    program = node("window_sum", node("event_delta", relation(predicate), lag=1), lag=3)
    restored = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    mapping = {name: "renamed_" + name for name in program.required_features()}
    renamed = restored.remap_features(mapping)
    for row in rows():
        rebound = Observation({mapping[k]: v for k, v in row.features.items()}, 0.,
            tuple({mapping[k]: v for k, v in past.items()} for past in row.history))
        assert renamed.predict(rebound) == pytest.approx(program.predict(row))
        assert renamed.expanded().predict(rebound) == pytest.approx(program.predict(row))
    assert program.history_depth == 3
    assert program.predict(rows()[0]) == 0.


def test_predicate_history_reads_the_same_edge_expression_at_its_past_context():
    predicate = node("lt", Program("const", value=0.),
                     node("event_delta", arg("edge_source_quantity"), lag=1))
    program = relation(predicate)
    for row in rows():
        assert program.predict(row) == pytest.approx(program.expanded().predict(row))


def test_nested_relation_predicate_uses_its_own_edges_and_outer_object_arguments():
    source, target = arg("edge_source_quantity"), arg("edge_target_quantity")
    inner = node("relation_sum", Program("const", value=1.), node("lt", source, target),
        Program("const", value=0.), Program("const", value=0.), target,
        Program("const", value=1.), Program("const", value=.5), source, name="right")
    program = relation(node("lt", Program("const", value=0.), inner))
    expected = relation(node("and", node("lt", target, source),
                             node("lt", Program("const", value=0.), target)))
    extra = replace(rows()[0], features={**rows()[0].features, "q0": .6, "q1": .4, "q2": .1})
    for row in [*rows(), extra]:
        assert program.predict(row) == pytest.approx(expected.predict(row))
        assert program.expanded().predict(row) == pytest.approx(expected.predict(row))


def test_canonical_edges_retain_condition_and_are_independent_of_object_enumeration():
    predicate = node("lt", arg("edge_source_quantity"), arg("edge_target_quantity"))
    program = relation(predicate)
    reordered = replace(program, children=(*program.children[:2],
        *(part for item in reversed(relation_objects(program)) for part in item)))
    canonical = canonical_object_program(program)
    assert canonical == canonical_object_program(reordered)
    opposite = replace(program, children=(program.children[0], node("not", predicate), *program.children[2:]))
    assert canonical != canonical_object_program(opposite)
    for row in rows():
        assert canonical.predict(row) == pytest.approx(program.predict(row))
    always = relation(Program("const", value=1.))
    old = node("neighbor_sum", always.children[0], *always.children[2:], name="right")
    assert canonical_object_program(always) == canonical_object_program(old)
