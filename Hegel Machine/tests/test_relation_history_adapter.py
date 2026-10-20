"""The existing relation engine must retain author-supplied temporal inputs."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.world_relation_research import research_relation


def row(current, previous, target, index):
    return {"features": {"x": current}, "history": [{"x": previous}],
            "target": target, "evidence": [index]}


def probe(current, previous):
    return {"features": {"x": current}, "history": [{"x": previous}],
            "actions": [{"kind": "trajectory", "sequence":
                         [{"x": previous}, {"x": current}]}]}


def temporal_binding():
    current = Program("feature", name="x")
    previous = Program("if", children=(Program("available", lag=1),
        Program("lag", name="x", lag=1), current))
    return {"rows": [row(1., 1., 1., 0), row(2., 2., 2., 1)],
            "hypotheses": [current.to_dict(), previous.to_dict()],
            "probes": [probe(2., 5.)]}


def test_history_discriminates_frozen_candidates_and_freshly_refutes_one():
    request = temporal_binding()
    first = research_relation(request, [{"index": 0}, {"index": 1}])
    assert [item["fit_mse"] for item in first["candidates"]] == [0., 0.]
    assert first["selected_probe"]["purpose"] == "discrimination"
    assert first["selected_probe"]["history"] == [{"x": 5.}]
    assert first["selected_probe"]["predictions"] == [2., 5.]

    request["rows"].append(row(2., 5., 5., 2))
    request["probes"] = [probe(2., 7.)]
    selected = research_relation(request, [{"index": i} for i in range(3)], first)
    assert [item["status"] for item in selected["candidates"]] == ["refuted", "plausible"]
    assert selected["candidates"][0]["validation_mse"] == 9.
    # The current controls already occurred. A new history is a new test.
    assert selected["selected_probe"]["purpose"] == "confirmation"
    assert selected["selected_probe"]["predictions"] == [7.]

    request["rows"].append(row(2., 7., 9., 3))
    rejected = research_relation(request, [{"index": i} for i in range(4)], selected)
    assert rejected["status"] == "refuted"
    old = next(item for item in rejected["rejected_candidates"]
               if item["program"] == selected["selected_program"])
    assert old["confirmation_mse"] == 4.
    assert old["confirmation_evidence"] == [3]


def test_declared_temporal_basis_fits_equal_current_inputs_with_different_histories():
    lag = Program("lag", name="x", lag=1)
    request = {"rows": [row(2., 1., 5., 0), row(2., 3., 11., 1)],
               "basis": [lag.to_dict()], "probes": [probe(2., 4.)]}
    result = research_relation(request, [{"index": 0}, {"index": 1}])
    assert result["candidates"][0]["fit_mse"] < 1e-12
    assert result["selected_probe"]["predictions"] == pytest.approx([14.])


def test_all_probe_scoring_uses_the_supplied_remaining_account():
    budget = SearchBudget(17, spent=7)
    result = research_relation(temporal_binding(), [{"index": 0}, {"index": 1}],
                               compute_budget=budget)
    # Two candidates * two rows; two candidates * two-step probe; two readouts.
    assert result["compute_cost"] == 10
    assert budget.spent == 17
    insufficient = SearchBudget(16, spent=7)
    with pytest.raises(ValueError, match="bounded research allowance"):
        research_relation(temporal_binding(), [{"index": 0}, {"index": 1}],
                          compute_budget=insufficient)
    assert insufficient.spent == 15


@pytest.mark.parametrize("where", ["rows", "probes"])
def test_nonfinite_authored_history_is_rejected(where):
    request = temporal_binding()
    request[where][0]["history"][0]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        research_relation(request, [{"index": 0}, {"index": 1}])


def test_explicit_coordinate_export_preserves_temporal_program_predictions():
    x = Program("feature", name="x")
    z = Program("feature", name="z")
    c = Program("feature", name="c")
    lag = Program("if", children=(Program("available", lag=1),
        Program("lag", name="z", lag=1), Program("const", value=0.)))
    window = Program("window_sum", lag=3, children=(Program("mul", children=(x, z)),))
    law = Program("if", children=(Program("ge", children=(c, Program("const", value=.2))),
        Program("add", children=(window, lag)), Program("sub", children=(x, lag))))
    with pytest.raises(ValueError, match="only control u"):
        export_temporal_program(law)
    namespace = {}
    exec(compile(export_temporal_program(law, feature_names=("x", "z", "c")),
                 "exported_relation.py", "exec"), namespace)
    history = []
    for point in ({"x": 1., "z": 2., "c": 0.},
                  {"x": -2., "z": .5, "c": .4},
                  {"x": .5, "z": -3., "c": .4},
                  {"x": 2., "z": 1., "c": -.3}):
        example = {"features": point, "history": deepcopy(history)}
        expected = law.predict(Observation(point, 0., tuple(history)))
        assert namespace["predict"](example) == pytest.approx(expected)
        history.append(point)
