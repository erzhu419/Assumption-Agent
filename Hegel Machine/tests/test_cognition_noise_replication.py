"""Paid repeats separate a missing temporal relation from random responses."""

from dataclasses import replace
from random import Random

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.noise_replication import assess_noise_replication


def test_fresh_paid_repeat_exposes_missing_window_relation_without_private_labels():
    lab = AnonymousObjectLab(objects=3, apparatus_seed=9013,
                             readout="window_total_event", name_seed=42114, wiring_seed=2680)
    probe = lab.random_probe(60113, length=12, name="original")
    original = lab.acquire(probe, ("fine",)).observations
    before = dict(lab.costs)
    repeated = lab.acquire(Probe("independent_repeat", probe.sequence), ("fine",)).observations
    budget = SearchBudget(1000)
    result = assess_noise_replication(Program("const", value=0.), original, repeated, budget)
    assert lab.observation_cost - before["observation_cost"] == 12
    assert lab.measurement_cost - before["measurement_cost"] == 108
    assert lab.acquisitions[-1]["probe_id"] != lab.acquisitions[-2]["probe_id"]
    assert result["repeatable_error"] and result["matched_rows"] == 12
    assert result["repeat_mse"] == 0. and result["original_model_loss"] > .1
    assert result["compute_cost"] == budget.spent > 0


def trajectories(seed):
    rng, history, rows = Random(seed), [], []
    for index in range(16):
        features = {"input": index / 16}
        target = features["input"] + rng.uniform(-.2, .2)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def test_random_response_history_does_not_hide_matching_external_inputs():
    result = assess_noise_replication(Program("feature", name="input"),
        trajectories(71), trajectories(171), SearchBudget(1000))
    assert result["matched_rows"] == 16
    assert result["repeat_mse"] > 1e-4
    assert not result["repeatable_error"]
    assert result["reason"] == "response_varies_on_repeat"


def test_different_input_prefix_is_not_a_repeat_of_a_temporal_experiment():
    original = trajectories(71)
    changed = [replace(row, history=tuple({**past, "input": past["input"] + 1.}
                                         for past in row.history)) for row in original]
    result = assess_noise_replication(Program("const", value=0.), original,
                                     changed, SearchBudget(1000))
    assert result["matched_rows"] == 1
    assert not result["repeatable_error"]
    assert result["reason"] == "insufficient_matched_rows"


def test_nested_target_history_predictor_requires_the_same_response_inputs():
    definition = Definition("previous_response", (), Program("lag", name="__target__", lag=1))
    program = Program("call", name=definition.definition_id, definition=definition)
    result = assess_noise_replication(program, trajectories(71), trajectories(171), SearchBudget(1000))
    assert result["target_history_required"]
    assert result["matched_rows"] == 1 and not result["repeatable_error"]


def test_fixed_quantile_and_predictor_fallback_remain_the_scored_objective():
    rows = [Observation({"input": 2.}, 2.), Observation({"input": 2.}, 2.)]
    missing_history = Program("lag", name="input", lag=1)
    exact = assess_noise_replication(missing_history, rows, rows, SearchBudget(100),
                                    fallback=2., risk_quantile=.9)
    assert exact["original_model_loss"] == 0. and not exact["repeatable_error"]
    wrong = assess_noise_replication(missing_history, rows, rows, SearchBudget(100),
                                    fallback=0., risk_quantile=.9)
    assert wrong["original_model_loss"] == 1.8 and wrong["repeatable_error"]
