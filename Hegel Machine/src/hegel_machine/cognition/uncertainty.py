"""Learn residual uncertainty and choose decisions under fixed task costs."""

from __future__ import annotations

from copy import deepcopy
from math import sqrt
from statistics import mean, pstdev, stdev

from .evaluation import predict
from .models import Observation, SearchBudget
from .risk_search import empirical_quantile, trajectory_blocks, weighted_quantile


def _quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (index - low) * (values[high] - values[low])


def action_and_interval(program, row, fallback, model, costs):
    center = predict(program, row, fallback)
    if not model or (model.get("kind") == "nominal" and not model.get("action_rule")):
        return center, center, center
    residuals = model["residual_interval"]
    side = None
    if model.get("kind") == "conditional_interval":
        value = row.features[model["feature"]]
        if model.get("transform") == "abs":
            value = abs(value)
        side = int(value >= model["threshold"])
        residuals = model["intervals"][side]
    low, high = center + residuals[0], center + residuals[1]
    if model.get("action_rule") == "empirical_cost_quantile":
        offset = model["action_offsets"][side] if side is not None else model["action_offset"]
        return center + offset + model.get("calibration_offset", 0.0), low, high
    under, over = costs
    # Minimizing the worst endpoint loss balances underage and overage costs.
    action = (under * high + over * low) / (under + over)
    return action, low, high


def decision_loss(action, target, costs):
    under, over = costs
    return under * max(target - action, 0.0) + over * max(action - target, 0.0)


def risk_calibration_quote(*, training_rows, validation_rows, training_trajectories,
                           feature_count, history_depth, max_probe_length,
                           tail_fit=True):
    """Price residual fitting and a fresh complete-decision comparison.

    The fit quote covers every supported descriptor and its actual trajectory
    folds. Constant features can make execution cheaper. A second pair is a
    separate possible follow-up, bought only if the first evidence conflicts.
    Reporting and the predictor's own search are priced by their callers.
    """
    length = min(max_probe_length, max(8, 4 * history_depth + 4))
    cross_validated = training_trajectories >= 2
    folds = min(3, training_trajectories)
    descriptor_cost = folds * training_rows if cross_validated else training_rows + validation_rows
    final_fit = training_rows * (2 if cross_validated and tail_fit else 1)
    final_fit += (3 if cross_validated else 1) * validation_rows
    fitting = (training_rows + (2 + 2 * feature_count) * descriptor_cost + final_fit
               if training_rows and validation_rows else 0)
    return {"trajectory_length": length, "fit_compute_cost": fitting,
            "comparison_observation_cost": 2 * length,
            "comparison_compute_cost": 4 * length,
            "followup_observation_cost": 2 * length,
            "followup_compute_cost": 4 * length}


def select_decision_plan(previous_program, previous_model: dict,
                         candidate_program, candidate_model: dict,
                         followup: list[Observation], *, fallback: float,
                         costs: tuple[float, float], budget: SearchBudget,
                         minimum_improvement: float = .002):
    """Compare complete deployed decisions, including a changed predictor.

    Each residual model stays attached to the predictor it was fitted around.
    ``followup`` contains already acquired trajectories unused for fitting or
    selecting either proposed program/model combination.
    Each trajectory is one paired trial: its two rules see identical outcomes.
    Two standard errors of the trajectory gains form a conservative selection
    margin, not a formal confidence guarantee. A single trajectory cannot
    estimate that variation and therefore cannot replace the previous rule.

    Comparing both rules costs two candidate-row evaluations per observation.
    The supplied models are never mutated, including when retaining old state.
    """
    blocks = trajectory_blocks(followup)
    details = {
        "selected": "previous", "adopted": False,
        "rows": len(followup), "trajectories": len(blocks),
        "compute_cost": 0, "observation_cost": 0,
        "fixed_costs": list(costs), "minimum_improvement": minimum_improvement,
        "comparison_scope": "paired_independent_followup_trajectories",
        "program_changed": previous_program != candidate_program,
        "before_loss": None, "after_loss": None, "gain": None,
        "standard_error": None, "support_margin": None,
        "trajectory_comparisons": [],
    }
    if len(blocks) < 2:
        details["reason"] = "insufficient_trajectory_support"
        return previous_program, deepcopy(previous_model), details
    if not budget.spend(2 * len(followup)):
        details["reason"] = "insufficient_budget"
        return previous_program, deepcopy(previous_model), details
    details["compute_cost"] = 2 * len(followup)
    for block in blocks:
        before = mean(decision_loss(action_and_interval(previous_program, row, fallback, previous_model, costs)[0],
                                    row.target, costs) for row in block)
        after = mean(decision_loss(action_and_interval(candidate_program, row, fallback, candidate_model, costs)[0],
                                   row.target, costs) for row in block)
        details["trajectory_comparisons"].append({"rows": len(block), "before_loss": before,
                                                  "after_loss": after, "gain": before - after})
    comparisons = details["trajectory_comparisons"]
    gain = mean(trial["gain"] for trial in comparisons)
    standard_error = stdev(trial["gain"] for trial in comparisons) / sqrt(len(blocks))
    margin = max(minimum_improvement, 2 * standard_error)
    details.update(before_loss=mean(trial["before_loss"] for trial in comparisons),
                   after_loss=mean(trial["after_loss"] for trial in comparisons),
                   gain=gain, standard_error=standard_error, support_margin=margin)
    if gain > margin:
        details.update(selected="candidate", adopted=True, reason="supported_paired_gain")
        return candidate_program, deepcopy(candidate_model), details
    details["reason"] = "insufficient_paired_gain"
    return previous_program, deepcopy(previous_model), details


def select_decision_model(program, previous_model: dict, candidate_model: dict,
                          followup: list[Observation], *, fallback: float,
                          costs: tuple[float, float], budget: SearchBudget,
                          minimum_improvement: float = .002) -> tuple[dict, dict]:
    """Compare calibrations attached to the same fixed predictor."""
    _, model, details = select_decision_plan(program, previous_model, program, candidate_model,
        followup, fallback=fallback, costs=costs, budget=budget,
        minimum_improvement=minimum_improvement)
    return model, details


def _interval_decision_model(program, training: list[Observation], validation: list[Observation], *,
                             fallback: float, costs: tuple[float, float], budget: SearchBudget) -> dict:
    """The 0.5.0 endpoint decision procedure, retained for a measured ablation."""
    if not training or not validation or not budget.spend(len(training)):
        return {"kind": "nominal", "status": "insufficient_budget"}
    residuals = [(row, row.target - predict(program, row, fallback)) for row in training]

    def interval(values):
        return [_quantile(values, 0.05), _quantile(values, 0.95)]

    values = [residual for _, residual in residuals]
    full_interval = interval(values)
    models = [{"kind": "nominal"},
              {"kind": "global_interval", "residual_interval": full_interval}]
    for name in training[0].features:
        threshold = _quantile([row.features[name] for row in training], 0.5)
        groups = [[value for row, value in residuals if (row.features[name] >= threshold) == side]
                  for side in (False, True)]
        if min(map(len, groups)) >= 3:
            models.append({"kind": "conditional_interval", "feature": name, "threshold": threshold,
                           "residual_interval": full_interval, "intervals": [interval(group) for group in groups]})
    measured = []
    for model in models:
        if not budget.spend(len(validation)):
            break
        value = mean(decision_loss(action_and_interval(program, row, fallback, model, costs)[0], row.target, costs)
                     for row in validation)
        measured.append((value, model))
    if not measured:
        return {"kind": "nominal", "status": "insufficient_budget"}
    score, chosen = min(measured, key=lambda item: item[0])
    return {**chosen, "fixed_costs": list(costs), "validation_decision_loss": score,
            "alternatives": [{"kind": model["kind"], "loss": value} for value, model in measured],
            "status": "measured_selection"}


def _fit_quantile_residual(descriptor, residuals, q):
    values = [value for _, value in residuals]
    interval = [_quantile(values, .05), _quantile(values, .95)]
    model = {**descriptor, "residual_interval": interval,
             "action_rule": "empirical_cost_quantile",
             "action_offset": 0.0 if descriptor["kind"] == "nominal" else empirical_quantile(values, q)}
    if descriptor["kind"] != "conditional_interval":
        return model
    name = descriptor["feature"]
    transform = abs if descriptor.get("transform") == "abs" else float
    threshold = _quantile([transform(row.features[name]) for row, _ in residuals], .5)
    groups = [[value for row, value in residuals if (transform(row.features[name]) >= threshold) == side]
              for side in (False, True)]
    if min(map(len, groups)) < 3:
        return None
    model.update(threshold=threshold, action_offsets=[empirical_quantile(group, q) for group in groups],
                 intervals=[[_quantile(group, .05), _quantile(group, .95)] for group in groups])
    return model


def fit_decision_model(program, training: list[Observation], validation: list[Observation], *,
                       fallback: float, costs: tuple[float, float], budget: SearchBudget,
                       adaptive: bool = True, tail_fit: bool = True) -> dict:
    """Select fixed-cost quantile actions by entire held-out trajectories.

    Structure selection cross-validates residual models conditional on the
    supplied fixed predictor. A separate final block calibrates the selected
    action, pooling its residual distribution with a centered training prior.
    ``tail_fit=False`` keeps the 0.6 shrinkage procedure; ``adaptive=False``
    keeps the earlier endpoint procedure used by the no-risk-learning ablation. Intervals
    summarize residual spread; actual outcome losses decide which model wins.
    """
    if not adaptive:
        return _interval_decision_model(program, training, validation, fallback=fallback,
                                        costs=costs, budget=budget)
    if not training or not validation or not budget.spend(len(training)):
        return {"kind": "nominal", "status": "insufficient_budget"}
    under, over = costs
    q = under / (under + over)
    residuals = {id(row): row.target - predict(program, row, fallback) for row in training}
    blocks = trajectory_blocks(training)
    cross_validated = len(blocks) >= 2
    fold_count = min(3, len(blocks))
    if cross_validated:
        folds = [([row for index, block in enumerate(blocks) if index % fold_count != fold for row in block],
                  [block for index, block in enumerate(blocks) if index % fold_count == fold])
                 for fold in range(fold_count)]
    else:
        folds = [(training, trajectory_blocks(validation))]
    descriptors = [{"kind": "nominal"}, {"kind": "global_interval"}]
    for name in sorted(set.intersection(*(set(row.features) for row in training))):
        for transform in ("identity", "abs"):
            values = [abs(row.features[name]) if transform == "abs" else row.features[name] for row in training]
            if min(values) < max(values):
                descriptors.append({"kind": "conditional_interval", "feature": name, "transform": transform})
    # Keep enough work to refit one model and to score, then calibrate, the
    # reserved observations. All candidate folds are charged even if rejected.
    reserve = len(training) * (2 if cross_validated and tail_fit else 1) + (3 if cross_validated else 1) * len(validation)
    measured = []
    for descriptor in descriptors:
        needed = sum(len(fitting) + sum(map(len, checking)) for fitting, checking in folds)
        if budget.remaining < reserve + needed:
            continue
        trajectory_scores = []
        complete = True
        for fitting, checking in folds:
            budget.spend(len(fitting))
            fitted = _fit_quantile_residual(descriptor, [(row, residuals[id(row)]) for row in fitting], q)
            if fitted is None:
                complete = False
                break
            for block in checking:
                budget.spend(len(block))
                trajectory_scores.append(mean(decision_loss(action_and_interval(program, row, fallback, fitted, costs)[0],
                                                           row.target, costs) for row in block))
        if complete and trajectory_scores:
            measured.append({"descriptor": descriptor, "loss": mean(trajectory_scores),
                             "trajectory_losses": trajectory_scores,
                             "standard_error": pstdev(trajectory_scores) / sqrt(len(trajectory_scores))})
    if not measured:
        return {"kind": "nominal", "status": "insufficient_budget"}
    best = min(measured, key=lambda item: item["loss"])
    # Prefer a simpler explanation when the measured improvement is smaller
    # than trajectory-to-trajectory variation. This is a selection heuristic.
    complexity = {"nominal": 0, "global_interval": 1, "conditional_interval": 2}
    eligible = [item for item in measured if item["loss"] <= best["loss"] + best["standard_error"]]
    chosen = min(eligible, key=lambda item: (complexity[item["descriptor"]["kind"]], item["loss"]))
    budget.spend(len(training))
    model = _fit_quantile_residual(chosen["descriptor"], [(row, residuals[id(row)]) for row in training], q)
    budget.spend(len(validation))
    actions = [action_and_interval(program, row, fallback, model, costs)[0] for row in validation]
    before_calibration = mean(decision_loss(action, row.target, costs) for action, row in zip(actions, validation))
    calibration_offset = 0.0
    calibration_weight = 0.0
    prior_rows = 0.0
    if cross_validated:
        budget.spend(len(validation))
        prior_rows = 1.0 / min(q, 1.0 - q)
        if tail_fit:
            prior_rows = min(len(training), prior_rows)
        calibration_weight = len(validation) / (len(validation) + prior_rows)
        calibration_residuals = [row.target - action for row, action in zip(validation, actions)]
        if tail_fit:
            budget.spend(len(training))
            training_residuals = [row.target - action_and_interval(program, row, fallback, model, costs)[0]
                                  for row in training]
            anchor = empirical_quantile(training_residuals, q)
            # Pool distributions before taking their quantile. Shrinking an
            # already-computed small-sample quantile still lets one extreme
            # residual determine an arbitrarily large correction. Centering
            # the prior preserves the selected action absent new evidence.
            pooled = [(value - anchor, prior_rows / len(training)) for value in training_residuals]
            pooled.extend((value, 1.0) for value in calibration_residuals)
            calibration_offset = weighted_quantile(pooled, q)
        else:
            calibration_offset = calibration_weight * empirical_quantile(calibration_residuals, q)
    after_calibration = before_calibration
    if calibration_offset:
        budget.spend(len(validation))
        after_calibration = mean(decision_loss(action + calibration_offset, row.target, costs)
                                 for action, row in zip(actions, validation))
    return {**model, "fixed_costs": list(costs), "risk_quantile": q,
            "validation_decision_loss": before_calibration,
            "calibration_decision_loss": after_calibration,
            "calibration_offset": calibration_offset, "calibration_weight": calibration_weight,
            "calibration_prior_rows": prior_rows,
            "calibration_method": "pooled_centered_residual_distribution" if tail_fit else "shrunk_calibration_quantile",
            "calibration_rows": len(validation) if cross_validated else 0,
            "selection_source": "whole_trajectory_cross_validation" if cross_validated else "heldout_trajectory",
            "comparison_scope": "residual_models_conditional_on_fixed_predictor",
            "interval_source": "training_residual_quantiles",
            "alternatives": [{**item["descriptor"], "loss": item["loss"],
                              "trajectory_losses": item["trajectory_losses"],
                              "standard_error": item["standard_error"]} for item in measured],
            "status": "measured_selection"}
