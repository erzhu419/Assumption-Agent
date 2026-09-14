"""Repeated object terms must not make a finite transfer question inexhaustible."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.transfer_condition_research import plan_transfer_condition_probe


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def object_call(objects):
    fields = tuple(f"{role}{index}" for index in range(objects) for role in ("q", "x", "u")) + ("focus",)
    argument = lambda name: Program("arg", name=name)
    selectors = [Program("if", children=(binary("and",
        binary("ge", argument("focus"), constant(index - .5)),
        binary("lt", argument("focus"), constant(index + .5))), constant(1.), constant(0.)))
        for index in range(objects)]
    flow = Program("neighbor_sum", children=(constant(.8),
        *(part for index in range(objects)
          for part in (selectors[index], argument(f"x{index}"), argument(f"q{index}")))))
    total = Program("object_sum", children=tuple(binary("add", argument(f"q{index}"),
        binary("mul", constant(.2), argument(f"u{index}"))) for index in range(objects)))
    body = Program("window_sum", lag=3, children=(binary("add", binary("mul", constant(1.3), total),
        binary("mul", constant(-.7), Program("event_delta", lag=1, children=(flow,)))),))
    definition = Definition("complete_object_response", fields, body, "previous_object_concept")
    return Program("call", name=definition.definition_id, definition=definition,
                   children=tuple(Program("feature", name=name) for name in fields))


def observed_cycles(program, objects):
    rows = []
    phase_length = program.history_depth + 2
    for focus in range(objects):
        history, previous_flow, recent = [], None, []
        for active in ([False] * phase_length + [True] * phase_length + [False] * phase_length):
            point = {"focus": float(focus if active else (focus + 1) % objects)}
            for index in range(objects):
                point.update({f"x{index}": index * (.1 if active else 1.1),
                              f"q{index}": (index + 1.) ** 2,
                              f"u{index}": .1 * (index + 1.)})
            selected = int(point["focus"])
            flow = sum(point[f"q{other}"] - point[f"q{selected}"] for other in range(objects)
                       if other != selected and abs(point[f"x{other}"] - point[f"x{selected}"]) < .8)
            event = 0. if previous_flow is None else flow - previous_flow
            recent.append(1.3 * sum(point[f"q{index}"] + .2 * point[f"u{index}"]
                                    for index in range(objects)) - .7 * event)
            rows.append(Observation(point, sum(recent[-3:]), tuple(history)))
            history.append(point)
            previous_flow = flow
    return rows


@pytest.mark.parametrize("objects", [3, 4])
def test_complete_object_cycles_cover_every_relation_without_a_permanent_path_cutoff(objects):
    program = object_call(objects)
    original = program.to_dict()
    rows = observed_cycles(program, objects)
    assert max(abs(program.predict(row) - row.target) for row in rows) < 1e-10
    # These full 3/4-object computations duplicate each active condition in
    # incoming/outgoing terms and retain an unconditional total contribution.
    spec = TaskSpec("object_transfer", "public commands", tuple(program.required_features()), (), (),
                    feature_bounds={name: (-5., 20.) for name in program.required_features()})
    budget = SearchBudget(480000)
    result = plan_transfer_condition_probe(spec, program, [program], budget, paid_rows=rows)
    assert result["forecast_status"] == "covered_by_paid_evidence"
    assert result["probe"] is None
    assert result["transfer_scope"]["all_conditions_covered"]
    assert budget.spent <= budget.limit
    assert program.to_dict() == original


def test_twenty_fifth_independent_condition_remains_unknown_after_the_first_page_is_observed():
    fields = tuple(f"x{index}" for index in range(25))
    branches = tuple(Program("if", children=(
        binary("ge", Program("feature", name=name), constant(.2)), constant(1.), constant(0.)))
        for name in fields)
    program = Program("object_sum", children=branches)
    point = {name: .5 if index < 24 else -.5 for index, name in enumerate(fields)}
    paid = [Observation(point, 24.)]
    spec = TaskSpec("many_branches", "independent public controls", fields, (), (),
                    feature_bounds={name: (-1., 1.) for name in fields})
    result = plan_transfer_condition_probe(spec, constant(24.), [program], SearchBudget(64000), paid_rows=paid)
    assert not result["transfer_scope"]["all_conditions_covered"]
    assert result["forecast_status"] != "covered_by_paid_evidence"
    scope = result["transfer_scope"]["calls"][0]
    assert not scope["scope_complete"]
    assert all(item["covered"] for item in scope["conditions"])
    assert result["hypotheses"][1]["program"] == program.to_dict()
    last_active = replace(paid[0], features={**point, fields[-1]: .5})
    assert program.predict(last_active) == 25.
