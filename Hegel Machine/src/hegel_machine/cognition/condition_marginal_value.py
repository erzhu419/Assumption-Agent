"""Returns of additional condition research after a response has already fitted.

Information outcomes and independently measured prediction gains stay separate.
The controller supplies complete phase costs; recording information never bills
those resources again or inherits the original method's improvement.
"""

from copy import deepcopy

from .action_policy import ResearchActionPolicy


CONDITION_FOLLOWUP_ACTION = "condition_followup_research"


def sampling_applicability(previous, current):
    """Transfer hit evidence across overlapping measured sampling questions."""
    if previous is None or current is None:
        return float(previous is current)
    if previous["basis"] != current["basis"]:
        return 0.
    old, new = previous["normalized_background"], current["normalized_background"]
    if len(old) != len(new):
        return 0.
    distance = sum((left - right) ** 2 for left, right in zip(old, new))
    radius = (previous.get("applicability_radius", previous["neighbourhood_radius"]) ** 2
              + current.get("applicability_radius", current["neighbourhood_radius"]) ** 2)
    background = 1. / (1. + distance / radius) if radius else float(distance == 0.)
    old_forecast, new_forecast = previous.get("reading_forecast"), current.get("reading_forecast")
    if old_forecast is None or new_forecast is None:
        return background if old_forecast is new_forecast else 0.
    if len(old_forecast["question_cdf"]) != len(new_forecast["question_cdf"]):
        return 0.
    # A miss calibrates both the instrument model and the interval being
    # sought. Use the paid sample resolution; changing a name or a unit does
    # not reset evidence, while a different measured distribution can.
    model_distance = sum((left - right) ** 2 for left, right in zip(
        old_forecast["control_sensitivity"], new_forecast["control_sensitivity"]))
    model_resolution = 1. / old_forecast["paid_rows"] + 1. / new_forecast["paid_rows"]
    reading_distance = sum((left - right) ** 2 for left, right in zip(
        old_forecast["residual_quantiles"], new_forecast["residual_quantiles"]))
    reading_distance += sum((left - right) ** 2
        for old_interval, new_interval in zip(old_forecast["question_cdf"], new_forecast["question_cdf"])
        for left, right in zip(old_interval, new_interval))
    reading_resolution = 1. / old_forecast["local_rows"] + 1. / new_forecast["local_rows"]
    return background / (1. + model_distance / model_resolution
                         + reading_distance / reading_resolution)


def condition_information_record(record, plan):
    """Keep response gains transferable; scope reading-hit forecasts separately."""
    current = plan.get("sampling_context")
    if not record.get("sampling_information"):
        # Unscoped historical trials did not test this conditional instrument
        # model. They remain available for their original unscoped forecast.
        return record if current is None else {**record,
            "information_event_sum": 0., "predicted_information_event_sum": 0.,
            "measured_information_gain_sum": 0., "measured_information_events": 0.}
    observed, predicted, gain, events = 0., 0., 0., 0.
    for entry in record["sampling_information"]:
        weight = sampling_applicability(entry["sampling_context"], current)
        observed += weight * entry["information_event_sum"]
        predicted += weight * entry["predicted_information_event_sum"]
        gain += weight * entry.get("measured_information_gain_sum", 0.)
        events += weight * entry.get("measured_information_events", 0.)
    return {**record, "information_event_sum": observed,
            "predicted_information_event_sum": predicted,
            "measured_information_gain_sum": gain, "measured_information_events": events}


def instrument_cycle_cost_estimate(record, plan, reserved_costs):
    """Predict complete execution costs while retaining the full hard reserve.

    An unseen cycle starts at its reserve. Applicable complete paid revisits
    progressively replace that prior; incomplete trials cannot make a complete
    experiment and fresh confirmation appear artificially cheap.
    """
    weight_sum = 0.
    costs = dict(reserved_costs)
    for cycle in record.get("instrument_revisits", []):
        weight = sampling_applicability(cycle["sampling_context"], plan.get("sampling_context"))
        weight_sum += weight
        for resource in costs:
            costs[resource] += weight * cycle[resource]
    return {resource: (cost / (1. + weight_sum) if resource == "compute_cost"
            else max(reserved_costs[resource], cost / (1. + weight_sum)))
            for resource, cost in costs.items()}


def condition_followup_policy(spec, rows, selected, experience):
    """Share returns only between publicly equivalent executable structures."""
    roles = {name: f"command_{index}" for index, name in enumerate(
        name for name in spec.feature_names if name in spec.feature_bounds)}
    roles.update({name: f"reading_{index}" for index, name in enumerate(
        name for name in spec.feature_names if name not in spec.feature_bounds)})

    def structure(node, *, threshold=False):
        if node.op == "const":
            return {"op": "threshold" if threshold else "zero" if node.value == 0. else "constant"}
        result = {"op": node.op}
        if node.name is not None:
            result["role"] = roles[node.name]
        if node.lag:
            result["lag"] = node.lag
        if node.children:
            result["children"] = [structure(child,
                threshold=node.op in ("ge", "lt") and index == 1)
                for index, child in enumerate(node.children)]
        return result

    policy = ResearchActionPolicy(rows, spec, experience)
    policy.set_condition(CONDITION_FOLLOWUP_ACTION)
    policy.set_action_knowledge(CONDITION_FOLLOWUP_ACTION, {
        "response": structure(selected["candidate"].program.expanded()),
        "commands": len(spec.feature_bounds),
        "readings": len(spec.feature_names) - len(spec.feature_bounds)})
    return policy


def record_condition_followup(policy, *, before_loss=None, after_loss=None,
        observation_cost, measurement_cost, compute_cost):
    """Settle one adjacent research phase on a fresh common-evidence pair."""
    costs = dict(observation_cost=observation_cost, measurement_cost=measurement_cost,
                 compute_cost=compute_cost)
    if before_loss is None or after_loss is None:
        return policy.record_unscored_attempt(CONDITION_FOLLOWUP_ACTION,
            status="unresolved", **costs)
    policy.update(CONDITION_FOLLOWUP_ACTION, before_loss=before_loss,
                  after_loss=after_loss, **costs)
    return deepcopy(policy.action_record(CONDITION_FOLLOWUP_ACTION))


def record_instrument_revisit_cost(policy, *, observation_cost, measurement_cost,
        compute_cost, sampling_context=None, screening_compute_cost=0):
    """Price one finished revisit after it has produced a fresh paired result.

    The caller includes every later research phase and the original paid
    screen. Remove that sunk screen from future execution cost; phase returns
    have already been updated and are not another trial or resource charge.
    """
    cycle = {"sampling_context": deepcopy(sampling_context),
        "observation_cost": observation_cost, "measurement_cost": measurement_cost,
        "compute_cost": max(0, compute_cost - screening_compute_cost),
        "screening_compute_cost": screening_compute_cost}
    policy.action_record(CONDITION_FOLLOWUP_ACTION).setdefault("instrument_revisits", []).append(cycle)
    return deepcopy(cycle)


def record_condition_information(policy, outcome):
    """Remember measured response information without claiming predictive gain."""
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    record["information_batches"] = record.get("information_batches", 0) + 1
    record["information_gain_sum"] = (record.get("information_gain_sum", 0.)
        + outcome.get("marginal_information_gain", max(
            outcome.get("relative_interval_reduction", 0.) if not outcome.get("changed_program") else 0.,
            float(outcome.get("new_counterexample", False)))))
    record["predicted_information_sum"] = (record.get("predicted_information_sum", 0.)
        + outcome["predicted_information_probability"])
    # Calibration compares event frequency with event probability. A small
    # genuine contraction is still a successful observation, not a failed hit.
    event = outcome.get("information_event", bool(outcome.get("interval_narrowed")
        or outcome.get("observed_response_counterexample") or outcome.get("new_counterexample")))
    record["information_event_sum"] = record.get("information_event_sum", 0) + int(event)
    record["predicted_information_event_sum"] = (
        record.get("predicted_information_event_sum", 0.)
        + outcome["predicted_information_probability"])
    context = outcome.get("sampling_context")
    sampling = record.setdefault("sampling_information", [])
    entry = next((item for item in sampling if item["sampling_context"] == context), None)
    if entry is None:
        entry = {"sampling_context": deepcopy(context), "information_event_sum": 0,
                 "predicted_information_event_sum": 0., "batches": 0}
        sampling.append(entry)
    entry["batches"] += 1
    entry["information_event_sum"] += int(event)
    entry["predicted_information_event_sum"] += outcome["predicted_information_probability"]
    if "marginal_information_gain" in outcome:
        entry["measured_information_gain_sum"] = (entry.get("measured_information_gain_sum", 0.)
            + outcome["marginal_information_gain"])
        entry["measured_information_events"] = entry.get("measured_information_events", 0) + int(event)
    record["last_information_outcome"] = deepcopy(outcome)
    return deepcopy(record)
