from copy import deepcopy

import pytest

from hegel_machine.cognition.world_experiment_choice import select_experiment


def design():
    return {"hypotheses": [{"id": name, "claim": f"Explanation {name}"} for name in ("a", "b", "c")],
            "experiments": [
                {"id": "navigate", "outcome": "The control menu opens", "actions": [{"action": "TALK", "arg1": 7}],
                 "predictions": {"a": True, "b": True, "c": True}},
                {"id": "contrast", "outcome": "The device responds under the altered condition",
                 "actions": [{"action": "USE", "arg1": 8, "arg2": 7}],
                 "predictions": {"a": True, "b": True, "c": False}}]}


def test_pairwise_agreement_still_selects_discriminating_physical_experiment():
    source = design()
    original = deepcopy(source)
    result = select_experiment(source)
    assert result["diagnosis"] == "discriminating_experiment_available"
    assert result["selected_experiment"] == source["experiments"][1]
    assert result["predicted_separation"] == pytest.approx(2 / 9)
    assert result["compute_cost"] == 6
    assert source == original
    result["selected_experiment"]["actions"][0]["arg2"] = 99
    result["hypotheses"][0]["claim"] = "Changed"
    assert source == original


def test_agreeing_forecasts_do_not_purchase_an_uninformative_experiment():
    source = design()
    source["experiments"] = source["experiments"][:1]
    result = select_experiment(source)
    assert result["diagnosis"] == "observational_ambiguity"
    assert result["selected_experiment"] is None


def test_equal_separation_keeps_authored_order():
    source = design()
    source["experiments"][0]["predictions"]["a"] = False
    assert select_experiment(source)["selected_experiment"]["id"] == "navigate"


@pytest.mark.parametrize("forecasts", [{"a": True, "b": False},
    {"a": True, "b": False, "c": None}, {"a": True, "b": False, "c": 0},
    {"a": True, "b": False, "c": True, "unknown": False}])
def test_missing_unknown_or_nonboolean_forecasts_are_not_fabricated(forecasts):
    source = design()
    source["experiments"][0]["predictions"] = forecasts
    with pytest.raises(ValueError, match="one Boolean for every hypothesis"):
        select_experiment(source)
