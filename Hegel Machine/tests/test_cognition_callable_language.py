"""Learned calls preserve temporal semantics and execute shared work only once."""

import json

import pytest

from hegel_machine.cognition.language import Definition, EvaluationContext, Program
from hegel_machine.cognition.models import Observation


def arg(name):
    return Program("arg", name=name)


def binary(op, left, right):
    return Program(op, children=(left, right))


def call(definition, *arguments):
    return Program("call", name=definition.definition_id, children=arguments, definition=definition)


def nodes(program):
    yield program
    for child in program.children:
        yield from nodes(child)


def test_call_arguments_are_lazy_under_a_real_startup_guard():
    definition = Definition(
        "history_sum", ("ready", "previous", "current"),
        Program("if", children=(arg("ready"), binary("add", arg("previous"), arg("current")), arg("current"))),
        "learned_history")
    program = call(definition, Program("available", lag=1),
                   Program("lag", name="__target__", lag=1), Program("feature", name="x"))
    startup = Observation({"x": 2.}, 2.)
    mature = Observation({"x": 2.}, 5., ({"x": 1., "__target__": 3.},))

    assert program.predict(startup) == program.expanded().predict(startup) == 2.
    assert program.predict(mature) == program.expanded().predict(mature) == 5.
    assert program.minimum_history_depth == 0
    assert program.history_depth == 1
    assert program.observed_variables() == {("x", 0), ("__target__", 1)}
    assert program.required_features() == {"x"}
    assert program.size == program.expanded().size == 6


def test_unused_argument_does_not_create_a_false_history_or_feature_requirement():
    constant = Definition("stable", ("unused",), Program("const", value=4.))
    program = call(constant, Program("lag", name="unmeasured", lag=4))
    row = Observation({}, 4.)
    assert program.predict(row) == 4.
    assert program.observed_variables() == program.required_features() == set()
    assert program.history_depth == program.minimum_history_depth == 0
    assert program.size == 1


def test_nested_definition_json_roundtrip_preserves_calls_and_role_remapping():
    square = Definition("square", ("u",), binary("mul", arg("u"), arg("u")), "source_a")
    square_plus = Definition("square_plus", ("u",), binary("add", call(square, arg("u")), arg("u")), "source_b")
    program = binary("add", call(square_plus, Program("feature", name="x")),
                     call(square, Program("feature", name="x")))
    payload = json.loads(json.dumps(program.to_dict()))
    assert set(payload["definitions"]) == {"square", "square_plus"}
    assert "definitions" not in payload["children"][0]
    assert payload["definitions"]["square_plus"]["body"]["children"][0]["op"] == "call"
    assert "definition" not in payload["definitions"]["square_plus"]["body"]["children"][0]
    restored = Program.from_dict(payload)
    assert restored == program
    # Both references recover one shared definition, rather than copied bodies.
    assert restored.children[0].definition.body.children[0].definition is restored.children[1].definition
    renamed = restored.remap_features({"x": "position"})
    assert renamed.children[0].definition is restored.children[0].definition
    assert renamed.required_features() == {"position"}
    assert renamed.observed_variables() == {("position", 0)}
    row = Observation({"position": 3.}, 21.)
    assert renamed.predict(row) == renamed.expanded().predict(row) == 21.
    assert all(node.op not in {"arg", "call"} for node in nodes(renamed.expanded()))
    assert renamed.size == renamed.expanded().size
    assert Definition.from_dict(json.loads(json.dumps(square_plus.to_dict()))) == square_plus


def test_shared_calls_reduce_actual_execution_inside_and_between_programs():
    definition = Definition("quadratic_interaction", ("x", "z"),
                            binary("add", binary("mul", arg("x"), arg("x")),
                                   binary("mul", arg("x"), arg("z"))))
    x, z = Program("feature", name="x"), Program("feature", name="z")
    g = call(definition, x, z)
    programs = [binary("add", binary("mul", g, g), g), binary("add", g, x)]
    row = Observation({"x": 2., "z": 3.}, 0.)
    shared, expanded = EvaluationContext(row), EvaluationContext(row)
    called_values = [program.predict(row, context=shared) for program in programs]
    expanded_values = [program.expanded().predict(row, context=expanded) for program in programs]

    assert called_values == expanded_values == [110., 12.]
    assert shared.call_evaluations == 1 and shared.cache_hits == 3
    assert shared.primitive_steps < expanded.primitive_steps
    # Count call dispatches too; the benefit is not a discounted AST size.
    assert shared.primitive_steps + shared.call_evaluations + shared.cache_hits < expanded.primitive_steps
    assert all(program.size == program.expanded().size for program in programs)


def test_changed_arguments_and_new_observations_execute_their_own_calls():
    definition = Definition("quadratic_interaction", ("a", "b"),
                            binary("add", binary("mul", arg("a"), arg("a")),
                                   binary("mul", arg("a"), arg("b"))))
    x, z = Program("feature", name="x"), Program("feature", name="z")
    first, swapped = call(definition, x, z), call(definition, z, x)
    row = Observation({"x": 2., "z": 3.}, 0.)
    context = EvaluationContext(row)
    assert context.evaluate(first) == 10.
    assert context.evaluate(swapped) == 15.
    assert context.call_evaluations == 2 and context.cache_hits == 0
    assert context.evaluate(first) == 10. and context.cache_hits == 1
    later = Observation({"x": 5., "z": 3.}, 0.)
    later_context = EvaluationContext(later)
    assert first.predict(later, context=later_context) == 40.
    assert later_context.call_evaluations == 1 and later_context.cache_hits == 0
    with pytest.raises(ValueError, match="one observation"):
        first.predict(later, context=context)


def test_nested_argument_scopes_share_only_the_same_bound_computation():
    square = Definition("square", ("u",), binary("mul", arg("u"), arg("u")))
    shift = Definition("shift", ("u",), binary("add", arg("u"), Program("const", value=1.)))
    pair = Definition("pair", ("u",), binary("add", call(square, call(shift, arg("u"))), call(square, arg("u"))))
    x = Program("feature", name="x")
    program = call(pair, call(shift, x))
    row = Observation({"x": 2.}, 25.)
    context = EvaluationContext(row)
    assert program.predict(row, context=context) == program.expanded().predict(row) == 25.
    evaluations = context.call_evaluations
    # This is square(3), already evaluated inside pair, not square(2).
    assert call(square, call(shift, x)).predict(row, context=context) == 9.
    assert context.call_evaluations == evaluations
    assert call(square, x).predict(row, context=context) == 4.
    assert context.call_evaluations == evaluations + 1


def test_existing_plain_ast_keeps_its_serialization_and_identity_expansion():
    payload = {"op": "add", "value": None, "name": None, "lag": 0, "children": [
        {"op": "feature", "value": None, "name": "x", "lag": 0, "children": []},
        {"op": "const", "value": 1., "name": None, "lag": 0, "children": []}]}
    program = Program.from_dict(payload)
    assert program.to_dict() == payload
    assert program.expanded() is program
    assert program.size == 3
    assert program.predict(Observation({"x": 2.}, 3.)) == 3.
