"""Price independent evidence about one frozen response revision."""

from copy import deepcopy

from .condition_marginal_value import sampling_applicability
from .condition_research_returns import _information_probability
from .evaluation import predict
from .language import Program
from .research_resources import research_resource_share


def record_pair_confirmation(previous, plan, rows, budget, *, measurement_rate=0,
                             tolerance=1e-7):
    """Score a purchased batch without changing either explanation or its fallback.

    A reading in the planned strip is not itself a response distinction. Score
    the actual two predictions, and retain both errors for the caller's usual
    confirmation/refutation decision. This records costs already paid for the
    rows and pays only the two new prediction evaluations here.
    """
    if not budget.spend(2 * len(rows)):
        return None
    pair = plan["confirmation_pair"]
    previous = previous or {}
    if previous.get("confirmation_pair") != pair:
        previous = {}
    baseline = Program.from_dict(pair["baseline_program"])
    selected = Program.from_dict(pair["program"])
    old = [predict(baseline, row, pair["baseline_fallback"]) for row in rows]
    new = [predict(selected, row, pair["fallback"]) for row in rows]
    baseline_loss = sum((value - row.target) ** 2 for value, row in zip(old, rows)) / len(rows)
    selected_loss = sum((value - row.target) ** 2 for value, row in zip(new, rows)) / len(rows)
    disagreement = sum((left - right) ** 2 for left, right in zip(old, new)) / len(rows)
    distinguished = disagreement > tolerance
    baseline_counterexample = baseline_loss > tolerance
    selected_counterexample = selected_loss > tolerance
    counterexample = baseline_counterexample or selected_counterexample
    # A distinction only tells us which frozen response is preferable when one
    # of them explains the new rows.  When both responses miss the same fresh
    # trajectory, the result is evidence against the response family itself;
    # the next action must reopen response research instead of buying another
    # sample from this pair.
    common_response_mismatch = baseline_counterexample and selected_counterexample
    probability = _information_probability(None, plan, len(rows), tolerance)
    costs = {"observation_cost": len(rows), "measurement_cost": len(rows) * measurement_rate,
             "compute_cost": 2 * len(rows)}
    outcome = {"baseline_loss": baseline_loss, "selected_loss": selected_loss,
        "prediction_disagreement": disagreement, "distinguished": distinguished,
        "response_counterexample": counterexample,
        "common_response_mismatch": common_response_mismatch,
        "predicted_information_probability": probability, **costs}
    return {"confirmation_pair": deepcopy(pair), "batches": previous.get("batches", 0) + 1,
        "distinguished": previous.get("distinguished", False) or distinguished,
        "response_counterexample": previous.get("response_counterexample", False) or counterexample,
        "reopen_required": previous.get("reopen_required", False) or common_response_mismatch,
        **{name: previous.get(name, 0) + value for name, value in costs.items()},
        "sampling_trials": [*previous.get("sampling_trials", []), {
            "sampling_context": deepcopy(plan.get("sampling_context")),
            "distinguished": distinguished, "predicted_information_probability": probability}],
        "last_outcome": outcome}


def consider_pair_confirmation(plan, previous=None, *, paid_loss,
        observation_remaining, compute_remaining, measurement_remaining=None,
        measurement_rate=0, compute_quote=0, observation_reserve=0,
        measurement_reserve=0, compute_reserve=0, tolerance=1e-7):
    """Price the next frozen-pair batch against the current shared resources.

    Value is a chance of obtaining independent discriminating evidence, not an
    accuracy improvement. Prior misses calibrate only this pair and applicable
    reading backgrounds. The caller has already paid for any new planning;
    ``compute_quote`` contains only the prospective batch's work.
    """
    previous = previous or {}
    if previous.get("confirmation_pair") != plan["confirmation_pair"]:
        previous = {}
    length = plan["probe"].cost
    probability = _information_probability(None, plan, length, tolerance)
    observed, predicted = 0., 0.
    for trial in previous.get("sampling_trials", []):
        weight = sampling_applicability(trial.get("sampling_context"), plan.get("sampling_context"))
        observed += weight * int(trial["distinguished"])
        predicted += weight * trial["predicted_information_probability"]
    reliability = min(1., (1. + observed) / (1. + predicted))
    risk = plan.get("expected_disagreement", 0.)
    risk_fraction = risk / (risk + max(0., paid_loss)) if risk > tolerance else 0.
    value = probability * reliability * risk_fraction
    costs = {"observation_cost": length, "measurement_cost": length * measurement_rate,
             "compute_cost": compute_quote}
    remaining = {"observation_cost": observation_remaining,
        "measurement_cost": measurement_remaining, "compute_cost": compute_remaining}
    share = research_resource_share(costs, remaining)
    affordable = (length <= max(0, observation_remaining - observation_reserve)
        and compute_quote <= max(0, compute_remaining - compute_reserve)
        and (measurement_remaining is None
             or costs["measurement_cost"] <= max(0, measurement_remaining - measurement_reserve)))
    if previous.get("reopen_required"):
        buy, reason = False, "reopen_response_model_after_common_mismatch"
    elif previous.get("distinguished") or previous.get("response_counterexample"):
        buy, reason = False, "return_measured_pair_evidence"
    elif not affordable:
        buy, reason = False, "leave_pair_settlement_affordable"
    elif value > share:
        buy, reason = True, "independent_pair_information_repays_resources"
    else:
        buy, reason = False, "retain_pair_for_more_valuable_evidence"
    return {"buy": buy, "reason": reason,
        "probability_of_new_information": probability,
        "sampling_reliability": reliability,
        "calibrated_information_probability": probability * reliability,
        "response_risk_fraction": risk_fraction,
        "expected_information_value": value, "resource_share": share,
        "quoted_costs": costs, "paid_response_loss": paid_loss,
        "expected_disagreement": risk, "current_pair_trials": previous.get("batches", 0)}
