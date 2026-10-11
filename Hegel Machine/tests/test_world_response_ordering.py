"""Task panels and query order must cover alternatives, not just small settings."""

from copy import deepcopy

from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_consequences import _ordered_experiments, _panel


def experiment(number, setting, positives, *, reference=False):
    value = {"id": f"{number}_{setting}", "number": number, "prediction": setting,
             "predictions": {f"member_{i}": i in positives for i in range(4)}}
    if reference:
        value["restore_frequency"] = 20.
    return value


def alternatives():
    # Many nearby suggestions for the same member must not hide another regime.
    return [experiment(5, value, {0}) for value in range(10, 20)] + [
        experiment(5, 40, {1}), experiment(5, 200, {2}), experiment(5, 900, {3})]


def test_new_panel_covers_distinct_predictions_and_keeps_every_physical_setting():
    state = {"target_numbers": [5]}
    pool = alternatives()
    ordered, cost = _ordered_experiments(state, pool)
    panel = _panel(state, ordered, {}, set())
    assert [p["frequency"] for p in panel] == [10, 40, 200, 900]
    assert {e["id"] for e in ordered} == {e["id"] for e in pool}
    assert len(ordered) == len(pool)
    assert cost > 0


def test_copied_forecast_columns_do_not_weight_the_panel_or_order():
    state = {"target_numbers": [5]}
    pool = alternatives()
    copies = deepcopy(pool)
    for e in copies:
        e["predictions"].update({f"copy_{i}": e["predictions"]["member_0"] for i in range(100)})
    original, original_cost = _ordered_experiments(state, pool)
    duplicated, duplicated_cost = _ordered_experiments(state, copies)
    assert [e["id"] for e in original] == [e["id"] for e in duplicated]
    assert original_cost == duplicated_cost
    assert _panel(state, original, {}, set()) == _panel(state, duplicated, {}, set())


def test_actual_reference_boundaries_precede_member_proposals_even_if_all_predict_negative():
    state = {"target_numbers": [5], "control_grids": {"1": {"step": 1.}},
             "response_settings": [{"crystal": 1, "frequency": 20., "activated": True},
                                   {"crystal": 1, "frequency": 10., "activated": False}]}
    refs = [experiment(1, 2., {0, 1}, reference=True),
            experiment(1, 15., {0}, reference=True),
            experiment(1, 19., set(), reference=True),
            experiment(1, 21., set(), reference=True)]
    ordered, _ = _ordered_experiments(state, alternatives() + refs)
    assert [(e["number"], e["prediction"]) for e in ordered[:4]] == [
        (5, 10), (1, 19.), (5, 40), (1, 21.)]
    assert [e["prediction"] for e in ordered if e["number"] == 1] == [19., 21., 15., 2.]


def test_retained_panel_is_not_replaced_by_a_different_order_until_measured():
    state = {"target_numbers": [5]}
    ordered, _ = _ordered_experiments(state, alternatives())
    prior = {"panel": [{"crystal": 5, "frequency": float(i)} for i in range(10, 14)]}
    assert _panel(state, ordered, prior, set()) == prior["panel"]
    assert _panel(state, ordered, prior, {(5, 10.)}) != prior["panel"]


def test_panel_alternates_unfinished_targets_and_excludes_paid_points():
    state = {"target_numbers": [4, 5, 6], "completed": [6]}
    pool = [experiment(n, f, {i}) for n in (4, 5, 6) for i, f in enumerate((10, 40, 200, 900))]
    ordered, _ = _ordered_experiments(state, pool)
    panel = _panel(state, ordered, {}, {(4, 10)})
    assert [(p["crystal"], p["frequency"]) for p in panel] == [(5, 10), (4, 40), (5, 40), (4, 200)]


def test_panel_initialization_leaves_budget_for_real_experiment_comparisons(monkeypatch):
    comparisons = []

    def expensive_signatures(features, responses, family, queries=(), **kwargs):
        if not queries or (len(queries) == 2 and not any(q["activated"] for q in queries)):
            return {"status": "SAT_WITNESS", "compute_cost": 1}
        if len(queries) == 3:
            comparisons.append((queries[0]["crystal"], queries[0]["activated"]))
        return {"status": "UNKNOWN", "compute_cost": kwargs["compute_limit"]}

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 1000)
    monkeypatch.setattr(consequences, "query_response_family", expensive_signatures)
    rows = [{"crystal": n, "measurements": {"x": {"value": float(n)}}} for n in range(1, 5)]
    state = {"research_table": {"references": rows[:2], "targets": rows[2:]},
             "target_numbers": [3, 4], "consequence_panel": [
                 {"crystal": 3, "frequency": 10.}, {"crystal": 4, "frequency": 20.}]}
    estimators = [{"id": "linear", "basis": [Program("feature", name="x").to_dict()]}]
    result = consequences.assess_response_consequences(state, estimators, {"compute_cost": 0,
        "experiments": [experiment(1, 11., {0, 1}), experiment(2, 12., {0, 1})]})
    assert comparisons == [(1, False), (1, True), (2, False), (2, True)]
    research = result["family_research"]
    assert research["compute_cost"] <= 1000
    assert research["possible_signatures"] == [[False, False], [False, True], [True, False], [True, True]]
    assert all(e["guaranteed_nonlocal_eliminations"] == 0 for e in research["experiment_values"])
