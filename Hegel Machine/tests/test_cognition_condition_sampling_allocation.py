"""The price of resolving a measured boundary includes enough real trials."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_research_returns import plan_condition_research_allocation
from hegel_machine.cognition.models import Probe


@pytest.fixture
def rare_boundary():
    payload = json.loads((Path(__file__).parent / "fixtures/condition_allocation_root03.json").read_text())
    saved = payload["plan"]["probe"]
    payload["plan"]["probe"] = Probe(saved["probe_id"], tuple(saved["sequence"]))
    return payload


def allocation(case, **limits):
    return plan_condition_research_allocation(case["state"], case["plan"],
        observation_remaining=limits.pop("observation_remaining", 288),
        measurement_remaining=limits.pop("measurement_remaining", 288),
        compute_remaining=limits.pop("compute_remaining", 320000),
        measurement_rate=limits.pop("measurement_rate", 1),
        setup_observation_cost=48, setup_compute_cost=120000,
        batch_compute_cost=2000, **limits)


def test_real_rare_boundary_prices_more_than_one_fixed_confirmation_batch(rare_boundary):
    plan = allocation(rare_boundary)
    probability = rare_boundary["plan"]["interval_exposure"][0]["estimated_probability"]
    assert probability * 24 < .15  # Actual root03 paid estimate, before any new trial.
    assert plan is not None and plan["selected_rows"] > 24
    assert plan["batch_length"] == 24
    assert plan["selected_rows"] % plan["batch_length"] == 0
    assert plan["estimated_information_probability"] > 1 - (1 - probability) ** 24
    assert plan["resource_costs"]["observation_cost"] >= plan["selected_rows"] + 48
    assert plan["resource_costs"]["measurement_cost"] >= plan["selected_rows"] + 48
    assert 120000 < plan["resource_costs"]["compute_cost"] <= 320000


@pytest.mark.parametrize("limits", [
    {"observation_remaining": 96},
    {"measurement_remaining": 96},
    {"measurement_rate": 2, "measurement_remaining": 192},
    {"compute_remaining": 124000},
])
def test_sampling_allocation_preserves_setup_and_obeys_every_resource_limit(rare_boundary, limits):
    plan = allocation(rare_boundary, **limits)
    assert plan is not None
    assert 24 <= plan["selected_rows"] <= 48
    costs = plan["resource_costs"]
    assert costs["observation_cost"] <= limits.get("observation_remaining", 288)
    assert costs["measurement_cost"] <= limits.get("measurement_remaining", 288)
    assert costs["compute_cost"] <= limits.get("compute_remaining", 320000)
    assert costs["observation_cost"] >= 48 + plan["selected_rows"]
    assert costs["measurement_cost"] >= (48 + plan["selected_rows"]) * limits.get("measurement_rate", 1)


@pytest.mark.parametrize("limits", [
    {"observation_remaining": 71}, {"measurement_remaining": 71},
    {"compute_remaining": 121999},
])
def test_no_trial_is_promised_by_spending_the_reserved_fresh_confirmation(rare_boundary, limits):
    assert allocation(rare_boundary, **limits) is None


def test_saturated_information_does_not_buy_all_available_rows(rare_boundary):
    easy = deepcopy(rare_boundary)
    easy["plan"]["interval_exposure"][0]["estimated_probability"] = .5
    plan = allocation(easy)
    assert plan["selected_rows"] == 24
    assert plan["estimated_information_probability"] > .999
    # Merely changing the units or location of the same measured bracket is
    # not a reason to keep sampling once its information chance is saturated.
    shifted = deepcopy(easy)
    for entry in shifted["state"]["conditions"]:
        entry["threshold_intervals"] = [[500. + 1000. * value for value in interval]
                                         for interval in entry["threshold_intervals"]]
    for entry in shifted["plan"]["interval_exposure"]:
        entry["threshold_interval"] = [500. + 1000. * value for value in entry["threshold_interval"]]
    assert allocation(shifted)["selected_rows"] == plan["selected_rows"]


def test_large_response_contrast_cannot_replace_a_zero_chance_of_observing_it(rare_boundary):
    rare_boundary["plan"]["interval_exposure"][0]["estimated_probability"] = 0.
    assert rare_boundary["plan"]["interval_exposure"][0]["response_disagreement"] > 100.
    assert allocation(rare_boundary) is None
