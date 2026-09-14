"""Paid repeated labels inform noise without supplying hidden learner state."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.boundary_noise import (
    noise_tail_factor, repeated_noise_scales, scaled_predictive_innovation,
    update_noise_weights,
)
from hegel_machine.cognition.models import SearchBudget


def observation(index, point, target, *, disputed=False):
    return {"observation": {"features": {"x1": point}, "target": target},
            "context": {"global_index": index},
            **({"observation_status": "disputed"} if disputed else {})}


def estimate(rows, *, scope=0, predictor=lambda features: 2.):
    return repeated_noise_scales(rows, predictor, SearchBudget(200),
                                 scope_start=scope, feature_names=("x1",))


def test_within_control_centering_removes_distinct_unknown_means():
    rows = [observation(0, .2, 10.), observation(1, .2, 12.), observation(2, .2, 14.),
            observation(3, .8, -4.), observation(4, .8, -2.)]
    original = deepcopy(rows)
    evidence = estimate(rows)
    shifted = deepcopy(rows)
    for row in shifted:
        row["observation"]["target"] += 100. if row["observation"]["features"]["x1"] == .2 else -30.
    assert estimate(shifted) == evidence
    assert evidence["degrees_of_freedom"] == 3
    assert evidence["standardized_sum_squares"] == pytest.approx(5.)
    assert evidence["scales"] == sorted(evidence["scales"])
    assert sum(evidence["weights"]) == pytest.approx(1.)
    assert rows == original


def test_singletons_obsolete_scope_and_disputed_labels_supply_no_noise_degrees():
    ignored = [observation(1, .2, 90.), observation(2, .2, -90.),
               observation(10, .4, 1.), observation(11, .4, 100., disputed=True)]
    queried = []

    def predict(features):
        queried.append(features["x1"])
        return 2.

    assert estimate(ignored, scope=10, predictor=predict) is None
    assert queried == []
    rows = [*ignored, observation(12, .8, 3.), observation(13, .8, 5.)]
    evidence = estimate(rows, scope=10, predictor=predict)
    assert evidence["degrees_of_freedom"] == 1
    assert evidence["constraint_scope_start"] == 10
    assert queried == [.8]
    assert evidence["groups"][0]["indices"] == [12, 13]
    assert evidence["standardized_sum_squares"] == pytest.approx(1.)


def test_larger_measured_repeat_scatter_increases_scale_without_new_degrees():
    small = [observation(0, .2, 9.), observation(1, .2, 11.)]
    large = [observation(0, .2, 8.), observation(1, .2, 12.)]
    before, after = estimate(small), estimate(large)
    assert before["degrees_of_freedom"] == after["degrees_of_freedom"] == 1
    assert after["scales"] == pytest.approx([4 * scale for scale in before["scales"]])


def test_latent_scale_changes_only_constraint_label_noise_without_mutating_prediction():
    innovation = [1.25, -2.]
    prediction = {"constraint_noise": .1, "constraint_variance": .3,
                  "objective_noise": .8, "objective_variance": .2}
    inputs = deepcopy((innovation, prediction))
    draw = scaled_predictive_innovation(innovation, prediction, 4.)
    assert len(draw) == 2 and draw[0] == innovation[0]
    assert draw[1] ** 2 * (.3 + .1) == pytest.approx(innovation[1] ** 2 * (.3 + 4 * .1))
    assert draw[1] < 0.
    assert (innovation, prediction) == inputs
    assert "scale" not in prediction
    assert scaled_predictive_innovation(innovation, prediction, 1.) == innovation


def test_future_repeat_contrast_reweights_noise_and_counts_only_one_new_degree():
    evidence = estimate([observation(0, .2, 9.), observation(1, .2, 11.)])
    original = deepcopy(evidence)
    before = noise_tail_factor(evidence, alpha=.05)
    quiet = update_noise_weights(evidence, previous_values=[9., 11.], target=10., predicted_noise=2.)
    loud = update_noise_weights(evidence, previous_values=[9., 11.], target=20., predicted_noise=2.)
    assert noise_tail_factor(quiet, alpha=.05)["variance_factor"] < before["variance_factor"]
    assert noise_tail_factor(loud, alpha=.05)["variance_factor"] > before["variance_factor"]
    assert quiet["updated_degrees"] == loud["updated_degrees"] == 1
    assert quiet["degrees_of_freedom"] == loud["degrees_of_freedom"] == 1
    assert loud["contrast"] ** 2 == pytest.approx(2 / 3 * 100)
    following = update_noise_weights(loud, previous_values=[9., 11., 20.],
                                     target=12., predicted_noise=2.)
    assert following["updated_degrees"] == 2
    assert evidence == original


def test_noise_weight_update_ignores_old_scatter_and_first_labels():
    evidence = estimate([observation(0, .2, 9.), observation(1, .2, 11.)])
    assert update_noise_weights(evidence, previous_values=[], target=500., predicted_noise=2.) == evidence
    narrow = update_noise_weights(evidence, previous_values=[9., 11.], target=12., predicted_noise=2.)
    wide = update_noise_weights(evidence, previous_values=[-90., 110.], target=12., predicted_noise=2.)
    assert narrow["weights"] == wide["weights"]
    assert narrow["updated_degrees"] == 1


def test_tail_quantile_mixes_scales_before_applying_the_public_probability():
    from math import sqrt
    from scipy.stats import norm

    evidence = {"scales": [.25, 1., 4., 16.], "weights": [.1, .2, .3, .4]}
    original = deepcopy(evidence)
    result = noise_tail_factor(evidence, alpha=.05)
    tail = sum(weight * norm.sf(result["quantile"] / sqrt(scale))
               for weight, scale in zip(evidence["weights"], evidence["scales"]))
    assert tail == pytest.approx(.05)
    assert result["tail_probability"] == pytest.approx(.05)
    assert result["component_evaluations"] == 4 * result["cdf_evaluations"]
    assert evidence == original
    assert noise_tail_factor({"scales": [4.], "weights": [1.]}, alpha=.05)["variance_factor"] == pytest.approx(4.)


def test_zero_noise_repeats_have_zero_tail_without_inventing_observation_noise():
    evidence = estimate([observation(0, .2, 10.), observation(1, .2, 10.)])
    result = noise_tail_factor(evidence, alpha=.05)
    assert result["quantile"] == result["variance_factor"] == result["tail_probability"] == 0.
    updated = update_noise_weights(evidence, previous_values=[10., 10.], target=10., predicted_noise=0.)
    assert updated["weights"] == evidence["weights"]
    assert noise_tail_factor(updated, alpha=.05)["variance_factor"] == 0.
