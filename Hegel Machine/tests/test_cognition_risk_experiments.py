"""New paid evidence can distinguish a weak rival without certifying ignorance."""

from copy import deepcopy
import json
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev

import pytest

from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.risk_experiments import distinguish_risk_explanations
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import laboratory_for
from test_cognition_observation_research import ActualPurchases


FIXTURE = Path(__file__).with_name("fixtures") / "risk_experiment_95401901.json"


def test_actual_weak_noisy_acceptance_is_reopened_by_new_frozen_comparison():
    saved = json.loads(FIXTURE.read_text())
    lab = laboratory_for(saved["task"])
    budget, actions = SearchBudget(240000), []
    paid = ActualPurchases(lab, budget, actions)
    for item in saved["probes"]:
        paid.acquire(item["channels"], "original_selection_evidence",
            probe=Probe(item["probe_id"], tuple(item["sequence"])))
    channels = tuple(saved["channels"])
    batches = [[row for _, row in batch] for batch in paid.paired_for(channels)]
    training, validation = [row for batch in batches[:-1] for row in batch], batches[-1]
    program = Program.from_dict(saved["program"])
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    support = agent._support(program, training, validation, lab.spec_for(channels), budget)
    assert support["adequate"] and support["reason"] == "noise_compatible"
    assert any(0. < item["gain"] < item["margin"]
               for item in support["residual_comparisons"])
    # The fixture holds only the old executable program and public controls.
    # Every response and competing residual is obtained again from real calls.
    original = deepcopy(support)
    start = paid.remaining()
    result = distinguish_risk_explanations(support, spec=lab.spec_for(channels),
        acquire=paid.acquire, channels=channels, remaining=paid.remaining, budget=budget,
        minimum_improvement=1e-7, measurement_rate=4, reserve_rows=48)
    assert support == original
    assert result["status"] == "supported" and not result["support"]["adequate"]
    assert result["support"]["reason"] == "predictable_residual"
    plans = [event for event in result["events"] if event["action"] == "plan_risk_experiment"]
    first = plans[0]
    assert first["required_additional_rows"] == max(16, ceil(
        len(validation) * (first["gain_margin"] / first["predicted_gain"]) ** 2))
    new_batches = [batch for batch in paid.batches
                   if batch["purpose"] == "distinguish_risk_explanations"]
    fresh = [row for batch in new_batches for _, row in batch["pairs"]]
    assert fresh and all(row is not old for row in fresh for old in [*training, *validation])
    events = [event for event in result["events"] if event["action"] == "compare_risk_explanations"]
    for event in events:
        rows = fresh[:event["new_row_end"]]
        assert event["accumulated_new_rows"] == len(rows)
        baseline = Program.from_dict(event["baseline_program"])
        for comparison in event["comparisons"]:
            correction = Program.from_dict(comparison["program"])
            gains = [point_loss(predict(baseline, row, event["baseline_fallback"]), row.target, .9)
                - point_loss(predict(baseline, row, event["baseline_fallback"])
                             + predict(correction, row, 0.), row.target, .9) for row in rows]
            assert comparison["gain"] == pytest.approx(mean(gains))
            assert comparison["margin"] == pytest.approx(max(1e-7, 3 * stdev(gains) / sqrt(len(rows))))
    winner = result["support"]["structural_counterexample"]
    assert winner["gain"] > winner["margin"]
    assert result["support"]["validation_rows"] == len(fresh)
    for resource in ("observation", "measurement", "compute"):
        assert result[resource + "_cost"] == start[resource + "_remaining"] - paid.remaining()[resource + "_remaining"]
    assert paid.remaining()["observation_remaining"] >= 48
    # Distinguishing a rival refutes this incumbent. It is not a confirmation
    # of the revised explanation, and creates no action reward by itself.
    assert not any(event["action"] == "risk_research_feedback" for event in actions)


def frozen_comparison_case(*, q=.5, new_targets=(0.,), observations=96, max_length=32):
    """A bounded paid acquisition callback around two frozen scalar hypotheses.

    Gains, margins and the training noise allowance are computed from the
    supplied responses; they are not injected expected action rewards.
    """
    old_targets = [0.] * 4 + [1.] * 12 if q == .5 else [-1.] * 12 + [2.] * 4
    training = old_targets * 2
    before = [point_loss(0., target, q) for target in old_targets]
    gains = [value - point_loss(1., target, q) for value, target in zip(before, old_targets)]
    training_losses = [point_loss(0., target, q) for target in training]
    support = {"adequate": True, "reason": "noise_compatible", "fixed_quantile": q,
        "training_rows": len(training), "validation_rows": len(old_targets),
        "training_loss": mean(training_losses), "validation_loss": mean(before),
        "noise_loss_limit": mean(training_losses) + 3 * stdev(training_losses)
            * sqrt(1 / len(training) + 1 / len(old_targets)),
        "structural_counterexample": None,
        "counterexample_baseline_program": Program("const", value=0.).to_dict(),
        "counterexample_fallback": 0.,
        "residual_comparisons": [{"kind": "frozen_offset", "gain": mean(gains),
            "margin": 3 * stdev(gains) / sqrt(len(gains)),
            "program": Program("const", value=1.).to_dict()}]}
    spec = TaskSpec("public", "Observe a noisy response", ("u", "meter"), (), (),
        risk_quantile=q, feature_bounds={"u": (-1., 1.)}, max_probe_length=max_length)
    budget, rows, acquired_candidates = SearchBudget(10000), [], []

    def remaining():
        return {"observation_remaining": observations - len(rows),
                "measurement_remaining": 3 * (observations - len(rows)),
                "compute_remaining": budget.remaining}

    def acquire(channels, purpose, candidates, *, rows_count):
        assert channels == ("meter",) and purpose == "distinguish_risk_explanations"
        assert rows_count <= remaining()["observation_remaining"]
        acquired_candidates.append([item["candidate"].program.to_dict() for item in candidates])
        batch, history = [], []
        for index in range(rows_count):
            features = {"u": (-1.) ** index, "meter": 0.}
            target = new_targets[(len(rows) + index) % len(new_targets)]
            batch.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        rows.extend(batch)
        return batch

    result = distinguish_risk_explanations(support, spec=spec, acquire=acquire,
        channels=("meter",), remaining=remaining, budget=budget, minimum_improvement=1e-7,
        measurement_rate=3, reserve_rows=8)
    assert result["observation_cost"] == len(rows)
    assert result["measurement_cost"] == 3 * len(rows)
    assert result["compute_cost"] == budget.spent == 2 * len(rows)
    assert all(item == acquired_candidates[0] for item in acquired_candidates)
    return support, result, rows


@pytest.mark.parametrize("q,target,status,adequate", [
    (.5, 0., "no_positive_measured_gain", True),
    (.5, .5, "no_positive_measured_gain", True),
    (.5, -2., "baseline_loss_shift", False),
    (.9, .1, "baseline_miscalibrated", False),
])
def test_new_negative_or_null_rival_does_not_bypass_the_incumbent_noise_checks(q, target, status, adequate):
    original, result, rows = frozen_comparison_case(q=q, new_targets=(target,))
    assert rows and result["status"] == status
    assert result["support"]["adequate"] is adequate
    assert result["support"]["structural_counterexample"] is None
    assert result["support"]["residual_comparisons"][0]["gain"] <= 1e-7
    assert result["support"]["validation_loss"] == pytest.approx(point_loss(0., target, q))
    assert original["adequate"] and original["validation_rows"] == 16


def test_unaffordable_initial_experiment_is_deferred_without_inventing_new_evidence():
    original, result, rows = frozen_comparison_case(observations=8)
    assert result["status"] == "deferred" and not rows
    assert result["support"]["adequate"] == original["adequate"]
    assert result["support"]["residual_comparisons"] == original["residual_comparisons"]
    assert not any(item["action"] == "compare_risk_explanations" for item in result["events"])


def test_started_experiment_with_remaining_positive_uncertainty_cannot_claim_noise_compatibility():
    original, result, rows = frozen_comparison_case(
        new_targets=(0.,) * 7 + (1.,) * 9, observations=48, max_length=16)
    assert rows and result["status"] == "unresolved"
    assert not result["support"]["adequate"]
    assert result["support"]["reason"] == "risk_experiment_unfunded"
    comparison, = result["support"]["residual_comparisons"]
    assert 0. < comparison["gain"] < comparison["margin"]
    assert result["support"]["validation_rows"] == len(rows)
    assert original["residual_comparisons"][0]["gain"] != comparison["gain"]
