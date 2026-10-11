"""Frozen knowledge should distinguish target consequences, not just members."""

from copy import deepcopy

from hegel_machine.cognition import world_response_consequences as consequences


IDS = ("a", "b", "c", "d")


def experiment(identifier, number, predictions, setting=100.):
    return {"id": identifier, "number": number, "prediction": setting,
            "predictions": dict(zip(IDS, map(bool, predictions))),
            "procedure": {"path": "/research.py", "max_steps": 512}}


def comparison(experiments):
    incumbent = experiments[0] if experiments else None
    hypotheses = [{"id": identifier, "source_id": "source_" + identifier,
                   "program": {"op": "const", "value": index},
                   "lower": 0., "upper": 1.}
                  for index, identifier in enumerate(IDS)]
    return {"experiments": deepcopy(experiments), "hypotheses": hypotheses,
            "candidates": [{"id": "candidate_a", "fit_sources": [{"source_id": "source_a"}]}],
            "selection": {"selected_experiment": deepcopy(incumbent),
                          "hypotheses": deepcopy(hypotheses),
                          "diagnosis": "discriminating_experiment_available"},
            "selected_number": incumbent["number"] if incumbent else None,
            "selected_setting": incumbent["prediction"] if incumbent else None,
            "compute_cost": 7}


def select(result, targets=(4, 5), completed=()):
    return consequences.select_frozen_target_experiment(
        result, {"target_numbers": list(targets), "completed": list(completed)})


def values(result):
    return {item["experiment_id"]: item for item in result["member_consequences"]["experiments"]}


def cross_target_case():
    # The reference separates a/b and c/d, without changing either target's
    # possible answer. Either target separates the two joint target forecasts.
    return comparison([experiment("reference", 1, (0, 1, 0, 1)),
                       experiment("target4", 4, (0, 0, 1, 1), 104.),
                       experiment("target5", 5, (0, 0, 1, 1), 105.)])


def test_common_positive_target_is_an_actual_trial_not_an_automatic_confirmation(monkeypatch):
    def no_family_solver(*args, **kwargs):
        raise AssertionError("finite member choice must not reopen the full family")

    monkeypatch.setattr(consequences, "query_response_family", no_family_solver)
    result = comparison([experiment("reference", 1, (0, 1, 0, 1)),
                         experiment("target4", 4, (1, 1, 1, 1), 104.)])
    state = {"target_numbers": [4], "completed": []}
    before = deepcopy(state)
    chosen = consequences.select_frozen_target_experiment(result, state)
    assert chosen is result
    assert state == before
    assert chosen["selection"]["selected_experiment"]["id"] == "target4"
    assert chosen["selection"]["diagnosis"] == "frozen_target_confirmation"
    assert chosen["selected_number"] == 4
    assert chosen["selected_setting"] == 104.
    assert chosen["selection"]["consequence_scope"] == "frozen_numeric_response_members"


def test_only_learning_the_tested_target_point_has_no_nonlocal_value():
    result = select(comparison([experiment("reference", 1, (0, 1, 0, 1)),
                                experiment("target4", 4, (0, 0, 1, 1))]), targets=(4,))
    score = values(result)["target4"]
    assert score["guaranteed_reduction"] == 0
    assert score["possible_reduction"] == 0
    assert all(branch["reduction"] == 0 for branch in score["branches"])
    assert result["selection"]["selected_experiment"]["id"] == "reference"


def test_cross_target_consequence_beats_member_only_separation_and_preserves_knowledge(monkeypatch):
    def no_family_solver(*args, **kwargs):
        raise AssertionError("finite consequences are available from the prediction table")

    monkeypatch.setattr(consequences, "query_response_family", no_family_solver)
    result = cross_target_case()
    saved = deepcopy(result)
    result = select(result)
    scores = values(result)
    assert scores["reference"]["guaranteed_reduction"] == 0
    assert scores["target4"]["guaranteed_reduction"] == 1
    assert [branch["reduction"] for branch in scores["target4"]["branches"]] == [1, 1]
    assert result["selection"]["selected_experiment"]["id"] == "target4"
    assert result["selection"]["diagnosis"] == "frozen_target_discrimination"
    assert result["selected_number"] == 4 and result["selected_setting"] == 104.
    assert result["hypotheses"] == saved["hypotheses"]
    assert result["candidates"] == saved["candidates"]
    assert result["experiments"] == saved["experiments"]
    assert result["selection"]["hypotheses"] == saved["selection"]["hypotheses"]
    audit = result["member_consequences"]
    assert audit["scope"] == "frozen_numeric_response_members"
    assert audit["incumbent_experiment_id"] == "reference"
    assert audit["selected_experiment_id"] == "target4"
    assert audit["compute_cost"] > 0
    assert result["compute_cost"] == saved["compute_cost"] + audit["compute_cost"]


def test_impossible_outcome_is_not_a_guaranteed_or_possible_improvement():
    result = select(comparison([experiment("reference", 1, (0, 1, 0, 1)),
                                experiment("constant", 2, (0, 0, 0, 0)),
                                experiment("target4", 4, (0, 0, 1, 1))]), targets=(4,))
    score = values(result)["constant"]
    assert score["guaranteed_reduction"] == score["possible_reduction"] == 0
    assert any(branch["surviving_members"] == 0 for branch in score["branches"])
    assert result["selection"]["selected_experiment"]["id"] == "reference"


def test_single_sided_opportunity_remains_available_when_both_outcomes_are_possible():
    result = select(comparison([experiment("constant", 1, (0, 0, 0, 0)),
                                experiment("useful_reference", 2, (0, 0, 0, 1)),
                                experiment("target4", 4, (0, 0, 1, 1))]), targets=(4,))
    score = values(result)["useful_reference"]
    assert score["guaranteed_reduction"] == 0
    assert score["possible_reduction"] == 1
    assert all(branch["surviving_members"] > 0 for branch in score["branches"])
    assert result["selection"]["selected_experiment"]["id"] == "useful_reference"


def test_equivalent_members_do_not_multiply_target_consequences():
    ordinary = cross_target_case()
    crowded = deepcopy(ordinary)
    for index in range(7):
        identifier = "a_copy_" + str(index)
        model = {**deepcopy(crowded["hypotheses"][0]), "id": identifier,
                 "source_id": "another_source_" + str(index)}
        crowded["hypotheses"].append(model)
        crowded["selection"]["hypotheses"].append(deepcopy(model))
        for item in crowded["experiments"]:
            item["predictions"][identifier] = item["predictions"]["a"]
        selected = crowded["selection"]["selected_experiment"]
        selected["predictions"][identifier] = selected["predictions"]["a"]
    before = deepcopy(crowded["hypotheses"])
    plain, duplicated = select(ordinary), select(crowded)
    assert plain["selection"]["selected_experiment"]["id"] == duplicated["selection"]["selected_experiment"]["id"]
    for identifier, score in values(plain).items():
        duplicate_score = values(duplicated)[identifier]
        assert (score["guaranteed_reduction"], score["possible_reduction"]) == (
            duplicate_score["guaranteed_reduction"], duplicate_score["possible_reduction"])
        assert [branch["reduction"] for branch in score["branches"]] == [
            branch["reduction"] for branch in duplicate_score["branches"]]
    assert duplicated["hypotheses"] == before


def test_positive_completed_target_is_not_selected_again():
    result = select(comparison([experiment("reference", 1, (0, 1, 0, 1)),
                                experiment("completed_target", 4, (1, 1, 1, 1)),
                                experiment("target5", 5, (0, 0, 1, 1))]), completed=(4,))
    assert result["selection"]["selected_experiment"]["id"] == "reference"


def test_equal_guaranteed_gain_keeps_the_incumbent_even_if_later_in_pool():
    result = cross_target_case()
    incumbent = deepcopy(result["experiments"][2])
    result["selection"]["selected_experiment"] = incumbent
    result["selected_number"], result["selected_setting"] = incumbent["number"], incumbent["prediction"]
    result = select(result)
    assert values(result)["target4"]["guaranteed_reduction"] == values(result)["target5"]["guaranteed_reduction"]
    assert result["selection"]["selected_experiment"]["id"] == "target5"


def test_empty_experiment_pool_does_not_reopen_research():
    result = comparison([])
    before = deepcopy(result)
    assert select(result) == before
