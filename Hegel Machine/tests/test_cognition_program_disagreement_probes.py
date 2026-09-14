"""Program-derived boundaries must expose real forecast differences."""

from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import _forecast, _probe_pool
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


def _feature(name):
    return Program("feature", name=name)


def _constant(value):
    return Program("const", value=value)


def _binary(op, left, right):
    return Program(op, children=(left, right))


def _spec(bounds):
    return TaskSpec("public", "public commands", tuple(bounds), (), (), feature_bounds=bounds)


def _neighbors(radius, *, measured=False):
    prefix = "m_" if measured else ""
    field = lambda name: _feature(prefix + name)
    return Program("neighbor_sum", children=(_constant(radius),
        field("selector"), field("left"), field("q_left"),
        _constant(0.), field("right"), field("q_right")))


def _separation(programs, sequence):
    predictions = [_forecast(program, sequence) for program in programs]
    common = [row for row in zip(*predictions) if all(value is not None for value in row)]
    return max((max(row) - min(row)) ** 2 for row in common) if common else 0.


def test_distance_interval_missed_by_the_generic_pool_is_found_without_a_grid():
    spec = _spec({"left": (-1., 1.), "right": (-1., 1.), "selector": (0., 1.),
                  "q_left": (-2., 2.), "q_right": (-2., 2.)})
    programs = [_neighbors(.73100001), _neighbors(.73100002)]
    assert all(_separation(programs, sequence) == 0. for sequence in _probe_pool(spec, 31, 4))
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, budget, serial=31)
    assert proposals and proposals[0]["predicted_separation"] > 1e-4
    best = proposals[0]
    assert best["reason"] == "program_predicate_interval"
    assert best["boundary"]["thresholds"] == [.73100001, .73100002]
    distances = [abs(point["left"] - point["right"]) for point in best["probe"].sequence]
    assert any(.73100001 < distance < .73100002 for distance in distances)
    assert best["predictions"] == [_forecast(program, best["probe"].sequence) for program in programs]
    assert 0 < budget.spent <= budget.limit
    json.dumps({**best, "probe": asdict(best["probe"])})


def test_calibration_inversion_handles_nonlinear_reversed_renamed_coordinates():
    spec = _spec({"knob_a": (-1., 1.), "knob_b": (-1., 1.), "knob_c": (0., 1.),
                  "knob_d": (-2., 2.), "knob_e": (-2., 2.)})
    a = _feature("knob_a")
    bindings = {
        "m_left": _binary("add", _constant(.5), _binary("mul", _constant(-2.),
            _binary("mul", a, _binary("mul", a, a)))),
        "m_right": _binary("add", _constant(.2), _binary("mul", _constant(1.5), _feature("knob_b"))),
        "m_selector": _feature("knob_c"), "m_q_left": _feature("knob_d"),
        "m_q_right": _feature("knob_e"),
    }
    programs = [_neighbors(.62000001, measured=True), _neighbors(.62000002, measured=True)]
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, budget, binding=bindings, serial=92)
    assert proposals and proposals[0]["predicted_separation"] > 1e-4
    best = proposals[0]
    assert best["boundary"]["control"] in {"knob_a", "knob_b"}
    for point in best["probe"].sequence:
        assert set(point) == set(spec.feature_bounds)
        assert all(low <= point[name] <= high for name, (low, high) in spec.feature_bounds.items())
    executable = [bind_measurement_arguments(program, bindings) for program in programs]
    assert best["predictions"] == [_forecast(program, best["probe"].sequence) for program in executable]
    assert budget.spent <= 6000


@pytest.mark.parametrize("temporal", ["lag", "window_sum", "event_delta", "call"])
def test_narrow_predicates_in_temporal_and_callable_programs_use_actual_history(temporal):
    spec = _spec({"drive": (-1., 1.)})
    programs = []
    for threshold in (.318100001, .318100002):
        field = Program("lag", name="drive", lag=2) if temporal == "lag" else _feature("drive")
        condition = _binary("ge", field, _constant(threshold))
        program = Program("if", children=(condition, _constant(3.), _constant(-2.)))
        if temporal in {"window_sum", "event_delta"}:
            program = Program(temporal, lag=3, children=(program,))
        elif temporal == "call":
            definition = Definition(f"gate_{threshold}", ("input",),
                Program("if", children=(_binary("lt", Program("arg", name="input"), _constant(threshold)),
                    _constant(-2.), _constant(3.))))
            program = Program("call", name=definition.definition_id, definition=definition,
                              children=(_feature("drive"),))
        programs.append(program)
    assert all(_separation(programs, sequence) == 0. for sequence in _probe_pool(spec, 13, 5))
    proposals = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), serial=13)
    assert proposals and proposals[0]["predicted_separation"] > 0.
    best = proposals[0]
    assert best["predictions"] == [_forecast(program, best["probe"].sequence) for program in programs]
    if temporal == "event_delta":
        assert best["probe"].sequence[0] != best["probe"].sequence[1]


def test_missing_calibration_infeasible_interval_and_exhausted_search_request_nothing():
    spec = _spec({"command": (-1., 1.)})
    measured = [_binary("lt", _feature("sensor"), _constant(value)) for value in (.1, .10001)]
    assert propose_program_disagreement_probes(spec, measured, SearchBudget(6000)) == []
    inaccessible = [_binary("lt", _feature("command"), _constant(value)) for value in (10., 10.01)]
    assert propose_program_disagreement_probes(spec, inaccessible, SearchBudget(6000)) == []
    feasible = [_binary("lt", _feature("command"), _constant(value)) for value in (.1, .10001)]
    empty = SearchBudget(0)
    assert propose_program_disagreement_probes(spec, feasible, empty) == [] and empty.spent == 0
    small = SearchBudget(5)
    assert propose_program_disagreement_probes(spec, feasible, small) == [] and small.spent <= 5
