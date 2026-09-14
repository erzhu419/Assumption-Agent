"""Zero-decay current-input fits carry no state into later research."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Definition, EvaluationContext, MissingHistory, Program
from hegel_machine.cognition.models import Observation


def _binary(op, left, right):
    return Program(op, children=(left, right))


def _nodes(program):
    yield program
    for child in program.children:
        yield from _nodes(child)


_CASES = json.loads((Path(__file__).parent / "fixtures" /
                    "cognition_v017_current_drive_programs.json").read_text())["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[f"{item['source'].split('seed')[1].split('_')[0]}_{item['task_id']}"
                                             for item in _CASES])
def test_saved_current_drive_programs_preserve_values_and_remove_only_zero_decay(case):
    original = Program.from_dict(case["program"])
    simplified = original.simplified()
    accumulators = [node for node in _nodes(original) if node.op == "accumulate"]
    zero_decay = [node for node in accumulators if node.value == 0.]
    remaining = [node for node in _nodes(simplified) if node.op == "accumulate"]
    assert remaining == [node for node in accumulators if node.value != 0.]
    if zero_decay:
        assert simplified.history_depth == 0 < original.history_depth
        assert simplified.size < original.size
    else:
        assert simplified is original
    low, high = case["feature_bounds"]["x"]
    for length in (0, 1, 7, 32):
        for x in (low, 0., high):
            for z in (low, 0., high):
                history = tuple({"x": (low, high, 0.)[index % 3],
                                 "z": (high, 0., low)[index % 3], "__target__": 0.}
                                for index in range(length))
                row = Observation({"x": x, "z": z}, 0., history)
                before, after = EvaluationContext(row), EvaluationContext(row)
                assert original.predict(row, before) == simplified.predict(row, after)
                if zero_decay:
                    assert after.primitive_steps < before.primitive_steps
    assert Program.from_dict(json.loads(json.dumps(simplified.to_dict()))) == simplified


def test_compound_drive_keeps_bound_shared_calls_and_replay_reduction():
    x, z = Program("feature", name="x"), Program("feature", name="z")
    definition = Definition("learned_affine_drive", ("u", "v", "unused"),
        _binary("add", _binary("mul", Program("const", value=.37), Program("arg", name="u")),
                       _binary("mul", Program("const", value=-1.1), Program("arg", name="v"))),
        "learned_source")
    drive = Program("call", name=definition.definition_id, definition=definition,
                    children=(x, z, Program("lag", name="unmeasured", lag=4)))
    state = Program("accumulate", value=0., children=(drive, Program("const", value=4.)))
    original = _binary("add", state, state)
    simplified = original.simplified()
    assert simplified.children == (drive, drive)
    assert all(node.definition is definition for node in simplified.children)
    row = Observation({"x": 2., "z": -1.}, 0., ({"x": -1., "z": 1.}, {"x": 0., "z": -2.}))
    before, after = EvaluationContext(row), EvaluationContext(row)
    assert original.predict(row, before) == simplified.predict(row, after)
    assert after.call_evaluations == 1 and after.cache_hits == 1
    assert after.primitive_steps < before.primitive_steps
    restored = Program.from_dict(json.loads(json.dumps(simplified.to_dict())))
    assert restored.children[0].definition is restored.children[1].definition
    assert restored.children[0].definition.source_concept_id == "learned_source"


@pytest.mark.parametrize("op", ["lag", "window_sum", "event_delta", "accumulate"])
def test_composite_temporal_drive_stays_a_temporal_program(op):
    x = Program("feature", name="x")
    if op == "lag":
        temporal = Program("lag", name="x", lag=1)
    elif op == "accumulate":
        temporal = Program(op, value=.2, children=(x, Program("const", value=0.)))
    else:
        temporal = Program(op, lag=2, children=(x,))
    original = Program("accumulate", value=0., children=(
        _binary("add", x, temporal), Program("const", value=0.)))
    assert original.simplified() is original
    assert original.history_depth > 0
    row = Observation({"x": 3.}, 0., ({"x": 1.}, {"x": 2.}))
    if op == "lag":
        with pytest.raises(MissingHistory):
            original.simplified().predict(row)
    else:
        assert original.simplified().predict(row) == original.predict(row)


def test_current_conditional_arithmetic_keeps_its_input_order_and_parameters():
    x = Program("feature", name="x")
    square = _binary("mul", x, x)
    drive = Program("if", children=(_binary("lt", x, Program("const", value=.127)),
                    _binary("add", square, Program("const", value=.999999999)),
                    _binary("sub", Program("const", value=.37), square)))
    original = Program("accumulate", value=0., children=(drive, Program("const", value=0.)))
    assert original.simplified() is drive
    for x_value in (-2.25, -.01, .127, 2.25):
        row = Observation({"x": x_value}, 0., ({"x": -1.}, {"x": 1.}))
        assert original.predict(row) == drive.predict(row)
