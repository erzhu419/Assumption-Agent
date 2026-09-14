"""Several narrow program conditions must hold in the same real command."""

from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_condition_sets import (
    compound_predicate_regions, solve_compound_region,
)
from hegel_machine.cognition.program_disagreement_probes import (
    _forecast, propose_program_disagreement_probes,
)
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


def feature(name):
    return Program("feature", name=name)


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def public_spec(bounds):
    return TaskSpec("compound", "public commands", tuple(bounds), (), (), feature_bounds=bounds)


def window(name, low, high):
    return binary("and", binary("ge", feature(name), const(low)),
                  binary("lt", feature(name), const(high)))


def programs(count=2, form="product", names=("x", "y", "z")):
    origins = (.2312, -.4117, .7931)
    result = []
    for alternative in (2e-8, 3e-8):
        gates = [window(name, low, low + (alternative if index == count - 1 else 4e-8))
                 for index, (name, low) in enumerate(zip(names[:count], origins))]
        expression = const(7.)
        if form == "nested":
            for gate in reversed(gates):
                expression = Program("if", children=(gate, expression, const(0.)))
        elif form == "conjunction":
            condition = gates[0]
            for gate in gates[1:]:
                condition = binary("and", condition, gate)
            expression = Program("if", children=(condition, expression, const(0.)))
        else:
            for gate in gates:
                expression = binary("mul", expression, gate)
        result.append(expression)
    return result


@pytest.mark.parametrize("count,form", [(2, "product"), (3, "product"),
                                        (2, "nested"), (3, "conjunction")])
def test_independent_narrow_gates_are_solved_simultaneously(count, form):
    spec = public_spec({name: (-1., 1.) for name in ("x", "y", "z")[:count]})
    laws = programs(count, form)
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, laws, budget, serial=23)
    assert proposals and proposals[0]["predicted_separation"] == 49.
    selected = proposals[0]
    assert selected["boundary"]["solver_strategy"] == "compound_conditions"
    constraints = selected["boundary"]["constraints"]
    assert len(constraints) == count
    final = selected["probe"].sequence[-1]
    for condition in constraints:
        value = Program.from_dict(condition["expression"]).predict(Observation(final, 0.))
        assert condition["lower"] < value < condition["upper"]
    assert selected["predictions"] == [_forecast(program, selected["probe"].sequence,
        lambda units: True) for program in laws]
    assert 0 < budget.spent <= 6000
    json.dumps({**selected, "probe": asdict(selected["probe"])})


def test_shared_reversed_controls_keep_every_previously_solved_gate_true():
    spec = public_spec({"new_a": (-1., 1.), "new_b": (-1., 1.)})
    a, b = feature("new_a"), feature("new_b")
    a3 = binary("mul", a, binary("mul", a, a))
    bindings = {"sensor_x": binary("add", a3, b),
                "sensor_y": binary("sub", a3, b)}
    laws = programs(names=("sensor_x", "sensor_y", "sensor_z"))
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, laws, budget, binding=bindings, serial=19)
    assert proposals and proposals[0]["predicted_separation"] == 49.
    selected = proposals[0]
    assert selected["boundary"]["solver_strategy"] == "compound_conditions"
    constraints = selected["boundary"]["constraints"]
    assert {item["measured_expression"]["name"] for item in constraints} == {"sensor_x", "sensor_y"}
    for point in selected["probe"].sequence:
        assert all(low <= point[name] <= high for name, (low, high) in spec.feature_bounds.items())
    final = Observation(selected["probe"].sequence[-1], 0.)
    assert all(item["lower"] < Program.from_dict(item["expression"]).predict(final) < item["upper"]
               for item in constraints)


def test_path_alternatives_respect_negation_and_do_not_merge_incompatible_branches():
    x = window("x", .23, .23000004)
    laws = []
    for high in (-.40999998, -.40999997):
        y = window("y", -.41, high)
        # Only the false branch reaches the jointly narrow nonzero response.
        laws.append(Program("if", children=(Program("not", children=(binary("and", x, y),)),
                                            const(0.), const(4.))))
    spec = public_spec({"x": (-1., 1.), "y": (-1., 1.)})
    proposals = propose_program_disagreement_probes(spec, laws, SearchBudget(6000), serial=8)
    assert proposals and proposals[0]["predicted_separation"] == 16.
    # Impossible shared-control inequalities must not be declared a solution.
    binding = {"x": feature("knob"), "y": feature("knob")}
    groups = compound_predicate_regions(laws, binding)
    assert groups == []


def test_jointly_infeasible_cells_exhaust_no_more_than_the_original_budget():
    spec = public_spec({"a": (0., 1.), "b": (0., 1.)})
    laws = programs()
    bindings = {"x": binary("add", feature("a"), feature("b")),
                "y": binary("sub", feature("a"), feature("b"))}
    # x=.2312 and y=-.4117 imply a<0, although each condition is separately feasible.
    for limit in (100, 6000):
        budget = SearchBudget(limit)
        assert propose_program_disagreement_probes(spec, laws, budget, binding=bindings) == []
        assert budget.spent <= limit


def test_compound_operation_rebinds_current_cells_and_forecasts_event_history():
    laws = [Program("event_delta", lag=2, children=(law,)) for law in programs(2, "nested")]
    spec = public_spec({"left": (-2., 2.), "right": (-3., 3.)})
    bindings = {"x": binary("mul", const(-.5), feature("left")),
                "y": binary("mul", const(1. / 3.), feature("right"))}
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, laws, budget, binding=bindings, serial=23)
    assert proposals and proposals[0]["sequence_mode"] == "cycle"
    assert any(value < 0. for prediction in proposals[0]["predictions"] for value in prediction)
    selected = proposals[0]
    executable = [bind_measurement_arguments(law, bindings) for law in laws]
    assert selected["predictions"] == [_forecast(law, selected["probe"].sequence,
        lambda units: True) for law in executable]
    region = next(region for region in compound_predicate_regions(laws, bindings)
        if region["constraints"][-1]["lower"] == -.4117 + 2e-8)
    solved, details = solve_compound_region(region, {"left": 1., "right": 1.},
                                            spec.feature_bounds, SearchBudget(6000).spend)
    assert solved is not None and len(details["constraints"]) == 2


def test_inclusive_gates_reachable_only_at_public_endpoints_can_discriminate():
    condition = binary("and", binary("ge", feature("x"), const(1.)),
                       binary("ge", feature("y"), const(1.)))
    spec = public_spec({"x": (0., 1.), "y": (0., 1.)})
    proposals = propose_program_disagreement_probes(spec, [condition, const(0.)], SearchBudget(6000))
    assert proposals and proposals[0]["predicted_separation"] == 1.
    assert proposals[0]["probe"].sequence[-1] == {"x": 1., "y": 1.}
    assert all(item["lower_inclusive"] for item in proposals[0]["boundary"]["constraints"])


def test_constant_left_and_negation_preserve_a_reachable_closed_singleton():
    at_one = binary("and", binary("ge", feature("x"), const(1.)),
                    binary("ge", const(1.), feature("x")))
    y_at_least_one = Program("not", children=(binary("lt", feature("y"), const(1.)),))
    condition = binary("and", at_one, y_at_least_one)
    spec = public_spec({"x": (0., 1.), "y": (0., 1.)})
    proposals = propose_program_disagreement_probes(spec, [condition, const(0.)], SearchBudget(6000))
    assert proposals and proposals[0]["predicted_separation"] == 1.
    assert proposals[0]["probe"].sequence[-1] == {"x": 1., "y": 1.}
    singleton = proposals[0]["boundary"]["constraints"][0]
    assert singleton["lower"] == singleton["upper"] == 1.
    assert singleton["lower_inclusive"] and singleton["upper_inclusive"]
    # Negating the closed upper condition makes the same public endpoint invalid.
    impossible = binary("and", Program("not", children=(binary("ge", const(1.), feature("x")),)),
                        y_at_least_one)
    assert propose_program_disagreement_probes(spec, [impossible, const(0.)], SearchBudget(6000)) == []
