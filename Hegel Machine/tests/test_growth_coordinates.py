from random import Random

from hegel_machine.cognition.growth_coordinates import normalized_role_bindings
from hegel_machine.cognition.growth_models import abstract_program
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def _source_program():
    x, y = Program("feature", name="x"), Program("feature", name="y")
    body = Program("add", children=(Program("mul", children=(x, x)),
        Program("mul", children=(Program("const", value=.5), Program("mul", children=(x, y))))))
    definition = abstract_program(body, "source_response")
    return Program("call", name=definition.definition_id, definition=definition, children=(x, y))


def _target_rows(seed, count):
    random = Random(seed)
    rows = []
    for _ in range(count):
        x, y = random.uniform(-1., 1.), random.uniform(-1., 1.)
        rows.append(Observation({"pressure": 40. + 20. * y, "field": 300. + 100. * x}, x * x + .5 * x * y))
    return rows


def test_range_and_role_binding_preserves_call_and_predicts_fresh_coordinates():
    source = _source_program()
    rows = _target_rows(73, 32)
    budget = SearchBudget(1000)
    bindings = normalized_role_bindings(source, {"x": (-1., 1.), "y": (-1., 1.)},
        {"pressure": (20., 60.), "field": (200., 400.)}, rows, budget)
    assert len(bindings) == 2
    best = bindings[0]
    assert best["mapping"] == {"x": "field", "y": "pressure"}
    assert best["affine_coordinates"] == {
        "x": {"target": "field", "scale": .01, "offset": -3.},
        "y": {"target": "pressure", "scale": .05, "offset": -2.}}
    assert best["program"].op == "call"
    assert best["program"].definition == source.definition
    assert best["loss"] < 1e-20
    assert bindings[1]["loss"] > .01
    assert budget.spent == len(rows) * len(bindings)
    assert max(abs(best["program"].predict(row) - row.target) for row in _target_rows(79, 48)) < 1e-10


def test_normalized_binding_stops_at_paid_budget_and_requires_public_source_ranges():
    source, rows = _source_program(), _target_rows(17, 16)
    source_bounds = {"x": (-1., 1.), "y": (-1., 1.)}
    target_bounds = {"pressure": (20., 60.), "field": (200., 400.)}
    budget = SearchBudget(31)
    bindings = normalized_role_bindings(source, source_bounds, target_bounds, rows, budget)
    assert len(bindings) == 1 and budget.spent == 16
    absent = SearchBudget(1000)
    assert normalized_role_bindings(source, {"x": (-1., 1.)}, target_bounds, rows, absent) == []
    assert absent.spent == 0
