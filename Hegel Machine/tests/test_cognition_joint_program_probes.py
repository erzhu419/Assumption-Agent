"""Jointly reachable boundaries that cannot be reached on a single axis."""

from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import (
    _forecast, predicate_boundary_groups, propose_program_disagreement_probes,
    solve_joint_boundary,
)
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


def field(name):
    return Program("feature", name=name)


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def gates(expression, thresholds):
    return [Program("if", children=(binary("lt", expression, constant(threshold)),
                                   constant(-2.), constant(3.))) for threshold in thresholds]


def spec(bounds):
    return TaskSpec("joint", "public actuator bounds", tuple(bounds), (), (), feature_bounds=bounds)


@pytest.mark.parametrize("operation,thresholds", [
    ("add", (1.99900001, 1.99900002)),
    ("mul", (.99900001, .99900002)),
])
def test_joint_near_limit_interval_missed_by_every_old_axis_is_reached(operation, thresholds):
    commands = spec({"a": (0., 1.), "b": (0., 1.)})
    expression = binary(operation, field("a"), field("b"))
    programs = gates(expression, thresholds)
    target = sum(thresholds) / 2
    # Reproduce the actual published one-axis operation and its three contexts.
    rng = Random(524287 + 1009 * 17)
    for _ in range(3):
        point = {name: rng.uniform(*interval) for name, interval in commands.feature_bounds.items()}
        for control in commands.feature_names:
            old = SearchBudget(6000)
            solution, _ = solve_joint_boundary(expression, target, point, commands.feature_bounds,
                old.spend, strategy="coordinate", control=control)
            assert solution is None

    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(commands, programs, budget, serial=17)
    assert proposals and proposals[0]["predicted_separation"] == 25.
    selected = proposals[0]
    assert selected["boundary"]["solver_strategy"] == "joint_segment"
    assert set(selected["boundary"]["controls"]) == {"a", "b"}
    assert selected["boundary"]["measured_expression"] == expression.to_dict()
    assert any(thresholds[0] < expression.predict(Observation(point, 0.)) < thresholds[1]
               for point in selected["probe"].sequence)
    assert 0 < budget.spent <= 6000


def test_reversed_coupled_nonlinear_calibration_requires_mixed_joint_direction():
    commands = spec({"new_a": (0., 1.), "new_b": (0., 1.)})
    a, b = field("new_a"), field("new_b")
    a3 = binary("mul", a, binary("mul", a, a))
    # Sensor columns both depend on both knobs; their difference is 2*(a^3-b).
    bindings = {"measured_left": binary("add", a3, binary("mul", constant(-2.), b)),
                "measured_right": binary("mul", constant(-1.), a3)}
    measured = binary("sub", field("measured_left"), field("measured_right"))
    programs = gates(measured, (1.99900001, 1.99900002))
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(commands, programs, budget,
                                                   binding=bindings, serial=42)
    assert proposals and proposals[0]["boundary"]["solver_strategy"] == "joint_segment"
    selected = proposals[0]
    assert selected["boundary"]["measured_expression"] == measured.to_dict()
    assert selected["boundary"]["solved_values"]["new_a"] > .99
    assert selected["boundary"]["solved_values"]["new_b"] < .01
    executable = [bind_measurement_arguments(program, bindings) for program in programs]
    assert selected["predictions"] == [
        _forecast(program, selected["probe"].sequence, lambda units: True) for program in executable]
    assert budget.spent <= 6000


def test_joint_boundary_operation_can_be_reexecuted_on_rebound_predicate():
    commands = spec({"renamed_x": (0., 2.), "renamed_y": (-1., 1.)})
    bindings = {"s_x": binary("mul", constant(.5), field("renamed_x")),
                "s_y": binary("mul", constant(.5), binary("add", field("renamed_y"), constant(1.)))}
    measured = binary("add", field("s_x"), field("s_y"))
    groups = predicate_boundary_groups(gates(measured, (1.99700001, 1.99700002)), bindings)
    assert len(groups) == 1 and groups[0]["measured_expression"] == measured
    budget = SearchBudget(6000)
    solved, metadata = solve_joint_boundary(groups[0]["expression"], sum(groups[0]["thresholds"]) / 2,
        {"renamed_x": .4, "renamed_y": -.7}, commands.feature_bounds, budget.spend)
    assert solved is not None and metadata["solver_strategy"] == "joint_segment"
    value = groups[0]["expression"].predict(Observation(solved, 0.))
    assert 1.99700001 < value < 1.99700002


def test_joint_root_must_change_complete_program_forecasts_and_stay_bounded():
    commands = spec({"a": (0., 1.), "b": (0., 1.)})
    expression = binary("add", field("a"), field("b"))
    unreachable = gates(expression, (2.01, 2.02))
    assert propose_program_disagreement_probes(commands, unreachable, SearchBudget(6000)) == []
    # Syntactically distinct predicates masked by the same response cannot buy data.
    masked = [binary("mul", constant(0.), program)
              for program in gates(expression, (1.99900001, 1.99900002))]
    assert propose_program_disagreement_probes(commands, masked, SearchBudget(6000)) == []
    for limit in (0, 50, 170):
        budget = SearchBudget(limit)
        assert propose_program_disagreement_probes(commands,
            gates(expression, (1.99900001, 1.99900002)), budget) == []
        assert budget.spent <= limit


def test_joint_cycle_keeps_entry_and_exit_history_when_maintaining_has_no_event():
    commands = spec({"a": (0., 1.), "b": (0., 1.)})
    programs = [Program("event_delta", lag=2, children=(program,)) for program in
                gates(binary("add", field("a"), field("b")), (1.99900001, 1.99900002))]
    proposals = propose_program_disagreement_probes(commands, programs, SearchBudget(6000), serial=17)
    assert proposals and proposals[0]["sequence_mode"] == "cycle"
    sequence = proposals[0]["probe"].sequence
    assert sequence[0] != sequence[1]
    assert sequence[-1] == sequence[0]
    assert any(value < 0. for prediction in proposals[0]["predictions"] for value in prediction)
    assert proposals[0]["predictions"] == [
        _forecast(program, sequence, lambda units: True) for program in programs]
