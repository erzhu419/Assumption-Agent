"""Measured prediction gains, ties, noise and counterexamples decide adoption."""

import json

import pytest

from hegel_machine.cognition.prior_adoption import PriorAdoption


def test_numerically_better_but_already_accurate_model_does_not_replace_default():
    policy = PriorAdoption()
    for _ in range(8):
        assert policy.observe(1e-5, 1e-9, 0.) == "default"
    assert all(row["reason"] == "both_models_meet_accuracy" for row in policy.to_dict()["decisions"])


def test_new_predictive_advantage_is_adopted_and_later_counterexamples_revoke_it():
    policy = PriorAdoption()
    for index in range(4):
        selected = policy.observe(.8, .01, 0., context={"stage": index})
        assert selected == ("default" if index < 3 else "learned")
    for index in range(4):
        selected = policy.observe(.01, .8, 0., context={"stage": index + 4})
    assert selected == "default"
    assert [row["selected"] for row in policy.to_dict()["decisions"]] == ["learned", "default"]
    assert len(policy.to_dict()["observations"]) == 8


def test_measurement_noise_prevents_adoption_from_an_unsupported_tiny_gain():
    policy = PriorAdoption()
    for _ in range(4):
        policy.observe(.11, .1, 0., noise_variance=.1)
    decision = policy.to_dict()["decisions"][-1]
    assert decision["mean_gain"] > 0.
    assert decision["uncertainty_radius"] > decision["mean_gain"]
    assert policy.selected == "default" and len(policy.to_dict()["pending"]) == 4


def test_resuming_partial_evidence_preserves_the_actual_adoption_decision():
    policy = PriorAdoption()
    for _ in range(2):
        policy.observe(.8, .01, 0.)
    restored = PriorAdoption(state=json.loads(json.dumps(policy.to_dict())))
    for _ in range(2):
        assert restored.observe(.8, .01, 0.) == policy.observe(.8, .01, 0.)
    assert restored.to_dict() == policy.to_dict()
    assert restored.selected == "learned"


def test_repair_restores_a_previously_useful_prior_and_supersedes_only_polluted_decisions():
    policy = PriorAdoption(batch_size=2)
    for _ in range(2):
        policy.observe(1., 0., 0.)
    assert policy.selected == "learned"
    snapshot = policy.comparison_snapshot()
    original_decision = policy.to_dict()["decisions"][0]
    # A large measurement error followed by a predictor fitted to that error
    # produces two apparent losses for the otherwise useful learned model.
    policy.observe(10., 0., 10., context={"global_index": 44})
    policy.observe(0., 10., 0., context={"global_index": 45})
    assert policy.selected == "default"
    before = policy.to_dict()
    restored = policy.repair_observation(2, 3, comparison_state=snapshot,
        default_mean=1., learned_mean=0., noise_variance=0.)
    state = policy.to_dict()
    assert restored == policy.selected == "learned"
    assert state["decisions"][0] == original_decision
    assert state["decisions"][1]["selected"] == "default"
    assert state["decisions"][1]["superseded_by_observation_repair"] == 2
    assert state["pending"] == [3] and len(state["observations"]) == 4
    assert state["observations"][2]["observation_status"] == "disputed"
    assert state["observations"][3]["gain"] == -100.
    assert state["observations"][3]["repaired_prediction"]["gain"] == 1.
    for old_row, new_row in zip(before["observations"], state["observations"]):
        for key in ("target", "context", "default_mean", "learned_mean", "default_error", "learned_error"):
            assert old_row[key] == new_row[key]
    # Repair does not exempt the prior from future evidence in either direction.
    assert policy.observe(1., 0., 0.) == "learned"
    for _ in range(2):
        policy.observe(0., 1., 0.)
    assert policy.selected == "default"
    for _ in range(2):
        policy.observe(1., 0., 0.)
    assert policy.selected == "learned"
    assert all(2 not in decision["observation_indices"]
               for decision in policy.to_dict()["decisions"][2:])


def test_repaired_repeat_completes_the_existing_comparison_without_reusing_the_disputed_measurement():
    policy = PriorAdoption()
    for _ in range(3):
        policy.observe(1., 0., 0.)
    snapshot = policy.comparison_snapshot()
    policy.observe(0., 1., 10.)
    policy.observe(0., 10., 0.)
    assert policy.selected == "default"
    before = policy.to_dict()
    assert policy.repair_observation(3, 4, comparison_state=snapshot,
        default_mean=1., learned_mean=0., noise_variance=0.) == "learned"
    state = policy.to_dict()
    assert len(state["observations"]) == len(before["observations"]) == 5
    assert state["decisions"][-1]["observation_indices"] == [0, 1, 2, 4]
    assert state["decisions"][-1]["mean_gain"] == 1.
    assert not state["pending"]
    assert all(decision["superseded_by_observation_repair"] == 3
               for decision in state["decisions"][:-1])


@pytest.mark.parametrize("repair_noise,expected_selection", [(0., "learned"), (100., "default")])
def test_json_resumption_keeps_repaired_predictions_and_their_uncertainty_in_pending_comparison(
        repair_noise, expected_selection):
    policy = PriorAdoption()
    for _ in range(2):
        policy.observe(1., 0., 0.)
    snapshot = policy.comparison_snapshot()
    policy.observe(0., 1., 10.)
    policy.observe(0., 10., 0.)
    policy.repair_observation(2, 3, comparison_state=snapshot,
        default_mean=1., learned_mean=0., noise_variance=repair_noise)
    assert policy.to_dict()["pending"] == [0, 1, 3]
    restored = PriorAdoption(state=json.loads(json.dumps(policy.to_dict())))
    for current in (policy, restored):
        assert current.observe(1., 0., 0.) == expected_selection
        state = current.to_dict()
        assert len(state["observations"]) == 5
        assert state["decisions"][-1]["observation_indices"] == [0, 1, 3, 4]
        assert state["decisions"][-1]["mean_gain"] == 1.
        assert state["observations"][3]["gain"] == -100.
        assert state["observations"][3]["repaired_prediction"]["gain"] == 1.
    assert restored.to_dict() == policy.to_dict()
