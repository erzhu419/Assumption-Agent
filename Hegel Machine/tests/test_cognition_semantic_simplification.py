"""Equivalent learned representations should not create false memory modes."""

import json

import pytest

from hegel_machine.cognition.language import Definition, EvaluationContext, MissingHistory, Program
from hegel_machine.cognition.models import Observation


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def test_zero_decay_representation_from_f03_becomes_a_current_coordinate():
    x = Program("feature", name="x")
    memory = Program("accumulate", value=0., children=(x, constant(-0.)))
    # The method's f03 expression retained the same input in recurrent and raw
    # coordinates. Keep its independently fitted coefficients and sum order.
    original = binary("add", binary("add", binary("mul", constant(-.89036251),
                      binary("mul", memory, memory)), binary("mul", constant(.5000001), memory)),
                      binary("mul", constant(.4999999), x))
    simplified = original.simplified()
    assert "accumulate" not in simplified.describe()
    assert simplified.history_depth == 0 < original.history_depth
    assert simplified.observed_variables() == original.observed_variables() == {("x", 0)}
    assert simplified.size < original.size
    for length in (0, 1, 4, 32):
        row = Observation({"x": -.37}, 0., tuple({"x": index / 10.} for index in range(length)))
        before, after = EvaluationContext(row), EvaluationContext(row)
        assert original.predict(row, before) == simplified.predict(row, after)
        assert after.primitive_steps < before.primitive_steps
    assert simplified.simplified() is simplified


@pytest.mark.parametrize("decay", [.000001, .3, 1., -1.])
def test_nonzero_decay_and_meaningful_initial_state_are_retained(decay):
    original = Program("accumulate", value=decay,
                       children=(Program("feature", name="x"), constant(2.)))
    assert original.simplified() is original
    row = Observation({"x": 0.}, 0., ({"x": 1.},))
    assert original.predict(row) != 0.


@pytest.mark.parametrize("missing_component", ["drive", "initial"])
def test_zero_decay_keeps_missing_history_failures(missing_component):
    lag = Program("lag", name="x", lag=1)
    original = Program("accumulate", value=0., children=(
        lag if missing_component == "drive" else Program("feature", name="x"),
        lag if missing_component == "initial" else constant(0.)))
    assert original.simplified() is original
    with pytest.raises(MissingHistory):
        original.simplified().predict(Observation({"x": 2.}, 0.))
    if missing_component == "drive":
        # Even a currently available lag fails when replay reaches the first
        # prefix; dropping replay must not turn it into an executable model.
        with pytest.raises(MissingHistory):
            original.simplified().predict(Observation({"x": 2.}, 0., ({"x": 1.},)))


def test_algebraic_identities_keep_lazy_branches_and_required_reads():
    x, lag = Program("feature", name="x"), Program("lag", name="x", lag=2)
    guarded = Program("if", children=(Program("available", lag=2),
                      binary("add", lag, constant(0.)), binary("mul", constant(1.), x)))
    result = guarded.simplified()
    assert result == Program("if", children=(Program("available", lag=2), lag, x))
    assert result.minimum_history_depth == 0
    assert result.predict(Observation({"x": 2.}, 0.)) == 2.
    assert result.predict(Observation({"x": 2.}, 0., ({"x": 3.}, {"x": 4.}))) == 3.
    assert binary("mul", constant(0.), lag).simplified().children[1] is lag
    assert binary("sub", lag, lag).simplified().op == "sub"
    assert Program("if", children=(constant(0.), lag, x)).simplified() is x


def test_calls_keep_shared_definitions_and_lazy_unused_arguments():
    x = Program("feature", name="x")
    identity = Definition("input_coordinate", ("input", "unused"), Program("arg", name="input"), "source_f03")
    call = Program("call", name=identity.definition_id, definition=identity,
                   children=(binary("add", x, constant(0.)), Program("lag", name="x", lag=4)))
    wrapped = Program("accumulate", value=0., children=(call, constant(0.)))
    result = wrapped.simplified()
    assert result.op == "call" and result.definition is identity
    assert result.children[0] is x
    assert result.expanded().simplified() is x
    row = Observation({"x": 4.}, 0., ({"x": 2.},))
    assert wrapped.predict(row) == result.predict(row) == 4.
    restored = Program.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.definition.source_concept_id == "source_f03"
    assert restored.predict(row) == 4.


def test_existing_definition_is_not_silently_redefined_by_simplification():
    identity = Definition("old_memory", ("x",), Program("accumulate", value=0.,
                          children=(Program("arg", name="x"), constant(0.))))
    call = Program("call", name=identity.definition_id, definition=identity,
                   children=(Program("feature", name="x"),))
    assert call.simplified() is call
    assert call.definition.body.op == "accumulate"
    assert call.expanded().simplified() == Program("feature", name="x")


def test_constant_folding_does_not_round_parameters_or_overflow():
    expression = binary("mul", binary("add", constant(.5), constant(.5)),
                        Program("feature", name="x"))
    assert expression.simplified() == Program("feature", name="x")
    overflow = binary("mul", constant(1e300), constant(1e300))
    assert overflow.simplified() is overflow
    near_one = binary("mul", constant(.999999999), Program("feature", name="x"))
    assert near_one.simplified() is near_one


def test_temporal_reductions_keep_trajectory_length_dependence():
    x = Program("feature", name="x")
    assert Program("window_sum", lag=1, children=(x,)).simplified() is x
    for op in ("window_sum", "event_delta"):
        expression = Program(op, lag=3, children=(constant(2.),))
        assert expression.simplified() is expression
