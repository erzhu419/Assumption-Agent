"""Price another real experiment on a retained, unresolved reading condition.

The saved summary belongs to the existing candidate/research draft. Information
gain is measured interval narrowing or a new response contradiction; a program
revision still needs its own independent prediction comparison.
"""

from copy import deepcopy
from math import expm1, log1p

from .action_policy import ActionOption, ResearchActionPolicy
from .condition_marginal_value import condition_information_record, sampling_applicability
from .research_resources import research_resource_share


def _question(state):
    if state is None:
        return None
    return {"status": state["status"], "program": deepcopy(state.get("program")),
        "original_target_loss": state.get("original_target_loss"),
        "expected_prediction_ambiguity": state.get("expected_prediction_ambiguity"),
        "conditions": [{"condition_field": item["condition_field"],
            "threshold_intervals": deepcopy(item.get("threshold_intervals", [])),
            "needs_threshold_revision": item.get("needs_threshold_revision", False)}
            for item in state.get("conditions", [])]}


def _evidence_change(before, after, tolerance):
    """Compare actual response support, with reading-unit invariant contraction."""
    if before is None or after is None:
        return {"interval_narrowed": False, "new_counterexample": False,
                "relative_interval_reduction": 0.}
    old = {item["condition_field"]: item for item in before.get("conditions", [])}
    reductions = []
    for item in after.get("conditions", []):
        previous = old.get(item["condition_field"])
        if previous is None:
            continue
        intervals, retained = previous.get("threshold_intervals", []), item.get("threshold_intervals", [])
        width = sum(right - left for left, right in intervals)
        # A different program can place its new gate elsewhere. That is a
        # program revision, not narrowing evidence for the old response.
        if width > 0. and retained and all(any(left <= lo <= hi <= right
                for left, right in intervals) for lo, hi in retained):
            reductions.append(max(0., 1. - sum(hi - lo for lo, hi in retained) / width))
    loss_before, loss_after = before.get("original_target_loss"), after.get("original_target_loss")
    counterexample = (after["status"] == "contradicted" and before["status"] != "contradicted"
        or loss_before is not None and loss_after is not None
        and loss_before <= tolerance < loss_after
        or any(item.get("needs_threshold_revision") and not old.get(
            item["condition_field"], {}).get("needs_threshold_revision", False)
            for item in after.get("conditions", [])))
    reduction = max(reductions, default=0.)
    return {"interval_narrowed": reduction > 1e-12,
            "relative_interval_reduction": reduction, "new_counterexample": bool(counterexample)}


def _same_question(before, after):
    # Fitting roundoff in the recorded loss does not change the question.
    return (before is not None and after is not None
        and before["program"] == after["program"]
        and before["conditions"] == after["conditions"])


def _current_calibration(question, previous, experiments, plan=None):
    """Use trials of the remaining question, not successes on wider cuts."""
    context = (plan or {}).get("sampling_context")
    if not experiments:
        trials = previous.get("sampling_trials")
        if trials is None:
            return (previous.get("current_question_returns", {})
                    if context is None and _same_question(previous.get("last_question"), question) else {})
        current = dict(batches=0, information_batches=0., predicted_information_batches=0.)
        for trial in trials:
            if not _same_question(trial["question"], question):
                continue
            weight = sampling_applicability(trial.get("sampling_context"), context)
            current["batches"] += int(weight > 0.)
            current["information_batches"] += weight * int(trial["informative"])
            current["predicted_information_batches"] += weight * trial["predicted_information_probability"]
        return current
    brackets = {item["condition_field"]: sorted(map(tuple, item["threshold_intervals"]))
                for item in (question or {}).get("conditions", [])}
    current = {"batches": 0, "information_batches": 0,
               "predicted_information_batches": 0.}
    for experiment in experiments:
        outcome = experiment.get("research_return")
        if outcome is None:
            continue
        # A response revision starts another question even when its reading
        # cut stayed in the same interval. Later actual trials can calibrate it.
        if outcome.get("changed_program"):
            current = dict(batches=0, information_batches=0, predicted_information_batches=0.)
            continue
        interval = experiment.get("threshold_interval")
        if (interval is None or brackets.get(experiment.get("condition_field")) != [tuple(interval)]
                or outcome.get("interval_narrowed")):
            continue
        weight = sampling_applicability(outcome.get("sampling_context",
            experiment.get("sampling_context")), context)
        current["batches"] += int(weight > 0.)
        current["information_batches"] += weight * int(outcome["informative"])
        current["predicted_information_batches"] += weight * outcome["predicted_information_probability"]
    return current


def _current_exposures(state, plan):
    """Keep exposure to the currently retained response's remaining brackets.

    A partial structure can nominate a wider, already tested interval. Its
    chance of producing a reading does not price this response's smaller gap.
    """
    exposures = plan.get("interval_exposure", [])
    conditions = (state or {}).get("conditions", [])
    if not conditions:
        return exposures
    brackets = {(item["condition_field"], tuple(interval))
        for item in conditions for interval in item.get("threshold_intervals", [])}
    return [item for item in exposures
        if (item["condition_field"], tuple(item["threshold_interval"])) in brackets
        and item["condition_field"] == plan.get("condition_field", item["condition_field"])]


def _current_response_risk(state, plan):
    """Return total and targeted squared-response ambiguity, not measured error."""
    if state is None:
        return None, None, []
    total = state.get("expected_prediction_ambiguity")
    exposures = _current_exposures(state, plan)
    brackets = {(item["condition_field"], tuple(item["threshold_interval"]))
                for item in exposures}
    if plan.get("interval_exposure") and not brackets:
        return total, 0., []
    fields = {field for field, _ in brackets}
    if not fields and plan.get("condition_field") is not None:
        fields.add(plan["condition_field"])
    conditions = state.get("conditions", [])
    targeted = [item for item in conditions
                if not fields or item["condition_field"] in fields]
    if not targeted:
        return total, None, []
    risks, opportunities = [], []
    for item in targeted:
        trials = [trial for trial in item.get("prediction_trials", [])
            if not brackets or (item["condition_field"],
                tuple(trial["threshold_interval"])) in brackets]
        values = [trial.get("expected_prediction_ambiguity") for trial in trials]
        if not values:
            values = [item.get("expected_prediction_ambiguity",
                               total if len(conditions) == 1 else None)]
        if any(value is None for value in values):
            return total, None, []
        risks.extend(values)
        for index, value in enumerate(values):
            relevant = [entry for entry in exposures
                if entry["condition_field"] == item["condition_field"]
                and (not trials or entry["threshold_interval"] == trials[index]["threshold_interval"])]
            opportunities.append((value, {**plan, "interval_exposure": relevant}))
    return total, max(risks, default=None), opportunities


def _remaining_cost_estimate(record, compute_spent):
    """Remove already paid planning from a historical complete phase quote."""
    costs = ResearchActionPolicy._estimated_costs(record)
    costs["compute_cost"] = max(0., costs["compute_cost"] - compute_spent)
    return {"trials": 1, **{f"mean_{resource}_cost": costs[f"{resource}_cost"]
        for resource in ("observation", "measurement", "compute")}}


def _information_size(question, previous, plan, marginal_record):
    """A hit need not resolve the whole boundary's response uncertainty.

    Keep actual contraction size across nearby questions of the same response.
    The hit calibration remains separate: a small contraction is a real hit,
    while its expected benefit is only the measured fraction of uncertainty.
    Local and transferred records may contain the same observations, so take
    the more cautious estimate rather than counting that evidence twice.
    """
    gain, events = 0., 0.
    for trial in previous.get("sampling_trials", []):
        if ("marginal_information_gain" not in trial or question is None
                or (trial.get("question") or {}).get("program") != question.get("program")):
            continue
        weight = sampling_applicability(trial.get("sampling_context"), plan.get("sampling_context"))
        gain += weight * trial["marginal_information_gain"]
        events += weight * int(trial["information_event"])
    local = (1. + gain) / (1. + events)
    transferred = ((1. + marginal_record.get("measured_information_gain_sum", 0.))
        / (1. + marginal_record.get("measured_information_events", 0.)))
    return min(1., local, transferred)


def _information_probability(state, plan, length, tolerance):
    # Competing models can nominate the same interval more than once. Their
    # forecasts do not create independent chances to observe that reading.
    exposures = {}
    for item in _current_exposures(state, plan):
        if item.get("response_disagreement", 0.) <= tolerance:
            continue
        field = item["condition_field"]
        interval = tuple(item["threshold_interval"])
        exposures.setdefault(field, {})[interval] = max(item.get("estimated_probability", 0.),
                                                       exposures.get(field, {}).get(interval, 0.))
    probabilities = []
    for intervals in exposures.values():
        # For overlapping alternative brackets use the largest event estimate;
        # add only disjoint events, rather than counting rival fits as evidence.
        groups = []
        for (left, right), probability in sorted(intervals.items()):
            if groups and left < groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], right)
                groups[-1][2] = max(groups[-1][2], probability)
            else:
                groups.append([left, right, probability])
        probabilities.append(min(1., sum(item[2] for item in groups)))
    if probabilities:
        # Dependence between instruments is unknown. The largest single-field
        # chance is enough to justify the experiment without inventing joint
        # independence. Its raw residual draws are the planner's sampling model.
        probability = min(1., max(0., max(probabilities)))
        return 1. if probability >= 1. else -expm1(length * log1p(-probability))
    # A startup/history competitor may differ even without a reading cut.
    # When brackets exist, the trajectory difference can itself be generated
    # by those rare gap events, so it must not override their exposure estimate.
    if not plan.get("interval_exposure") and plan.get("expected_disagreement", 0.) > tolerance:
        return 1.
    return 0.


def plan_condition_research_allocation(condition_state, plan, *, observation_remaining,
        compute_remaining, measurement_remaining=None, measurement_rate=0,
        setup_observation_cost=0, setup_compute_cost=0, minimum_compute_cost=0,
        batch_compute_cost=0, previous=None, previous_experiments=(), tolerance=1e-7):
    """Compare payable sample counts before committing a research allocation.

    Fixed discovery/refitting costs and a required fresh comparison belong to
    ``setup``. The priced rows are subsequent independent information batches;
    their exposure forecast measures a chance of learning, never confirmation.
    The caller can reprice after each real batch with its remaining resources.
    """
    length = len(plan["probe"].sequence)
    maximum_batches = max(0, (observation_remaining - setup_observation_cost) // length)
    if measurement_rate and measurement_remaining is not None:
        maximum_batches = min(maximum_batches,
            max(0, (measurement_remaining // measurement_rate - setup_observation_cost) // length))
    if not maximum_batches:
        return None
    calibration = _current_calibration(_question(condition_state), previous or {}, previous_experiments, plan)
    reliability = min(1., (1. + calibration.get("information_batches", 0))
                       / (1. + calibration.get("predicted_information_batches", 0.)))
    alternatives = []
    for batches in range(1, int(maximum_batches) + 1):
        count = batches * length
        cost = {"observation_cost": setup_observation_cost + count,
            "measurement_cost": (setup_observation_cost + count) * measurement_rate,
            "compute_cost": max(minimum_compute_cost,
                                setup_compute_cost + batches * batch_compute_cost)}
        if cost["compute_cost"] > compute_remaining:
            break
        chance = _information_probability(condition_state, plan, count, tolerance)
        if chance <= 0.:
            return None
        share = (cost["observation_cost"] / observation_remaining
            + cost["compute_cost"] / max(1, compute_remaining)
            + (cost["measurement_cost"] / max(1, measurement_remaining)
               if measurement_remaining is not None else 0.))
        alternatives.append({"selected_rows": count, "batches": batches,
            "estimated_information_probability": chance,
            "calibrated_information_probability": chance * reliability,
            "resource_costs": cost, "resource_share": share,
            "information_per_cost": chance * reliability / share})
    if not alternatives:
        return None
    chosen = max(alternatives, key=lambda item: item["information_per_cost"])
    return {**chosen, "batch_length": length, "sampling_reliability": reliability,
        "setup_observation_cost": setup_observation_cost,
        "setup_compute_cost": setup_compute_cost, "minimum_compute_cost": minimum_compute_cost,
        "batch_compute_cost": batch_compute_cost, "alternatives": alternatives,
        "basis": "paid_exposure_and_complete_research_cost",
        "requires_fresh_confirmation": True}


def consider_condition_research(condition_state, plan, previous=None, *,
        observation_remaining, compute_remaining, measurement_remaining=None,
        measurement_rate=0, compute_quote=0, observation_reserve=0,
        measurement_reserve=0, compute_reserve=0, tolerance=1e-7,
        new_counterexample=False, previous_experiments=(), marginal_record=None,
        compute_spent_before_decision=0, expected_cycle_costs=None):
    """Decide whether one quoted batch repays its scarce resource commitment.

    A useful first experiment is allowed. Failed batches reduce confidence in
    the sampling forecast in proportion to their predicted information chance;
    a rare miss carries little negative evidence. New real response evidence
    can reopen a stopped question without forgetting its earlier paid costs.
    """
    previous = previous or {}
    length = len(plan["probe"].sequence)
    available_obs = max(0, observation_remaining - observation_reserve)
    available_compute = max(0, compute_remaining - compute_reserve)
    available_meter = (None if measurement_remaining is None else
                       max(0, measurement_remaining - measurement_reserve))
    option = ActionOption("distinguish_competing_observed_conditions", length,
                          compute_quote, measurement_cost=length * measurement_rate)
    affordable = (length <= available_obs and compute_quote <= available_compute
        and (available_meter is None or option.measurement_cost <= available_meter))
    chance = _information_probability(condition_state, plan, length, tolerance)
    current = _question(condition_state)
    calibration = _current_calibration(current, previous, previous_experiments, plan)
    reliability = min(1., (1. + calibration.get("information_batches", 0))
                       / (1. + calibration.get("predicted_information_batches", 0.)))
    adjusted_chance = chance * reliability
    marginal_record = condition_information_record(marginal_record or {}, plan)
    information_size = _information_size(current, previous, plan, marginal_record)
    information_yield = min(1., (1. + marginal_record.get("information_event_sum", 0.))
        / (1. + marginal_record.get("predicted_information_event_sum", 0.)))
    # The local forecast and transferred yield can contain the same purchased
    # misses. Use the more cautious estimate without multiplying them twice.
    calibrated_chance = chance * min(reliability, information_yield)
    total_risk, targeted_risk, opportunities = _current_response_risk(condition_state, plan)
    risk_known = total_risk is not None and targeted_risk is not None
    risk_normalizer = (max((condition_state.get("original_target_loss") or 0.)
                           + total_risk, tolerance) if risk_known else None)
    reduction = (max((risk * _information_probability(condition_state, targeted_plan,
        length, tolerance) * min(reliability, information_yield) * information_size
        for risk, targeted_plan in opportunities), default=0.) if risk_known else None)
    # The fraction is the currently reducible uncertain response error. With
    # a fitted response it can approach one; it is not an observed accuracy gain.
    information_value = (reduction / risk_normalizer if risk_known and risk_normalizer > 0.
                         else 0. if risk_known else calibrated_chance * information_size)
    prediction_gain = max(0., marginal_record.get("expected_gain", 0.))
    marginal_value = max(information_value, prediction_gain)
    record = {"trials": previous.get("batches", 0),
        **{f"mean_{resource}_cost": previous.get(f"{resource}_cost", 0.)
           / max(1, previous.get("batches", 0))
           for resource in ("observation", "measurement", "compute")}}
    record = _remaining_cost_estimate(record, compute_spent_before_decision)
    marginal_cost = _remaining_cost_estimate(marginal_record, compute_spent_before_decision)
    share = ResearchActionPolicy._resource_share(option, available_obs, available_compute,
                                                  record, available_meter)
    share = max(share, ResearchActionPolicy._resource_share(option, available_obs,
        available_compute, marginal_cost, available_meter))
    if expected_cycle_costs is not None:
        # Reserve caps decide affordability above. Expected expenditure prices
        # the complete cycle against the pool before reserving its confirmation;
        # the same reserved cost is not charged a second time as opportunity cost.
        share = research_resource_share(expected_cycle_costs,
            {"observation_cost": observation_remaining, "measurement_cost": measurement_remaining,
             "compute_cost": compute_remaining})
    change = _evidence_change(previous.get("last_question"), current, tolerance)
    # A tighter bracket changes what can be learned next, but does not make
    # its next experiment worth any price. Only a new contradiction or law
    # bypasses the old marginal-return estimate to reopen response research.
    reopened = (new_counterexample or change["new_counterexample"]
        or previous.get("last_question") is not None and current is not None
        and previous["last_question"]["program"] != current["program"])
    if not affordable:
        reason, buy = "leave_fresh_confirmation_affordable", False
    elif condition_state is None or condition_state["status"] == "supported" and not new_counterexample:
        reason, buy = "condition_prediction_already_supported", False
    elif risk_known and targeted_risk <= tolerance and not new_counterexample:
        reason, buy = "no_remaining_targeted_response_risk", False
    elif chance <= 0.:
        reason, buy = "no_predicted_original_response_information", False
    elif reopened:
        reason, buy = "reopen_from_new_original_response_evidence", True
    elif marginal_value > share:
        reason, buy = ("first_experiment_has_information_value" if not previous.get("batches")
                       else "information_chance_repays_remaining_resources"), True
    else:
        reason, buy = "retain_question_for_more_valuable_future_evidence", False
    return {"buy": buy, "reason": reason,
        "probability_of_new_information": chance,
        "calibrated_information_probability": adjusted_chance,
        "expected_information_value": information_value,
        "current_response_risk_total": total_risk,
        "targeted_response_risk": targeted_risk,
        "predicted_response_risk_reduction": reduction,
        "response_risk_normalizer": risk_normalizer,
        "response_risk_known": risk_known,
        "compute_spent_before_decision": compute_spent_before_decision,
        "estimated_remaining_compute_cost": max(compute_quote,
            record["mean_compute_cost"], marginal_cost["mean_compute_cost"])
            if expected_cycle_costs is None else expected_cycle_costs["compute_cost"],
        "expected_cycle_costs": expected_cycle_costs,
        "expected_information_fraction": information_size,
        "information_yield": information_yield,
        "empirical_marginal_prediction_gain": prediction_gain,
        "expected_marginal_value": marginal_value,
        "marginal_prediction_trials": marginal_record.get("trials", 0),
        "sampling_reliability": reliability, "resource_share": share,
        "current_question_trials": calibration.get("batches", 0),
        "current_question_information_batches": calibration.get("information_batches", 0),
        "current_question_predicted_information_batches": calibration.get("predicted_information_batches", 0.),
        "reopened": bool(reopened), "quoted_observation_cost": length,
        "quoted_measurement_cost": option.measurement_cost,
        "quoted_compute_cost": compute_quote}


def record_condition_research(previous, before, after, plan, *, observation_cost,
        measurement_cost, compute_cost, changed_program=False, tolerance=1e-7,
        observed_after=None):
    """Retain one experiment's paid outcome inside the current partial method."""
    previous = previous or {}
    start, end = _question(before), _question(after)
    change = _evidence_change(start, end, tolerance)
    changed = changed_program or start is not None and end is not None and start["program"] != end["program"]
    informative = bool(changed or change["interval_narrowed"] or change["new_counterexample"])
    observed = _question(observed_after) if observed_after is not None else end
    information_change = _evidence_change(start, observed, tolerance)
    same_response = start is not None and observed is not None and start["program"] == observed["program"]
    reduction = information_change["relative_interval_reduction"] if same_response else 0.
    old_ambiguity = (start or {}).get("expected_prediction_ambiguity")
    new_ambiguity = (observed or {}).get("expected_prediction_ambiguity")
    ambiguity_reduction = (max(0., min(1., 1. - new_ambiguity / old_ambiguity))
        if reduction > 0. and old_ambiguity is not None and old_ambiguity > 0.
        and new_ambiguity is not None else 0.)
    information_gain = max(reduction, ambiguity_reduction,
                           float(information_change["new_counterexample"]))
    information_event = bool(information_change["interval_narrowed"]
                             or information_change["new_counterexample"])
    probability = _information_probability(before, plan, observation_cost, tolerance)
    calibration = _current_calibration(start, previous, (), plan)
    current_returns = ({"batches": calibration.get("batches", 0) + 1,
        "information_batches": calibration.get("information_batches", 0) + int(informative),
        "predicted_information_batches": calibration.get("predicted_information_batches", 0.) + probability}
        if _same_question(start, end) else
        {"batches": 0, "information_batches": 0, "predicted_information_batches": 0.})
    return {"batches": previous.get("batches", 0) + 1,
        "information_batches": previous.get("information_batches", 0) + int(informative),
        "predicted_information_batches": previous.get("predicted_information_batches", 0.) + probability,
        "observation_cost": previous.get("observation_cost", 0) + observation_cost,
        "measurement_cost": previous.get("measurement_cost", 0) + measurement_cost,
        "compute_cost": previous.get("compute_cost", 0) + compute_cost,
        "current_question_returns": current_returns,
        "sampling_trials": [*previous.get("sampling_trials", []),
            {"question": start, "sampling_context": deepcopy(plan.get("sampling_context")),
             "informative": informative, "predicted_information_probability": probability,
             "marginal_information_gain": information_gain, "information_event": information_event}],
        "last_question": end,
        "last_outcome": {**change, "changed_program": bool(changed),
            "sampling_context": deepcopy(plan.get("sampling_context")),
            "marginal_information_gain": information_gain,
            "information_event": information_event,
            "observed_response_counterexample": information_change["new_counterexample"],
            "relative_prediction_ambiguity_reduction": ambiguity_reduction,
            "informative": informative, "predicted_information_probability": probability,
            "observation_cost": observation_cost, "measurement_cost": measurement_cost,
            "compute_cost": compute_cost}}
