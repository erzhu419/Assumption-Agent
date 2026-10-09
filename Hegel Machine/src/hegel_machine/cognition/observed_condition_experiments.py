"""Change legal control backgrounds while buying a real condition reading."""

from dataclasses import replace
from bisect import bisect_left, bisect_right
from itertools import product
from math import ceil, sqrt
from statistics import median

from .evaluation import initial_prediction, predict
from .language import MissingHistory, Program
from .models import Observation, Probe
from .research_programs import _Fitter
from .observed_condition_uncertainty import _at_reading, _gates


def _threshold(program, field, value):
    if (program.op in ("ge", "lt") and program.children[0].op == "feature"
            and program.children[0].name == field and program.children[1].op == "const"):
        return replace(program, children=(program.children[0], Program("const", value=value)))
    return replace(program, children=tuple(_threshold(child, field, value)
                                          for child in program.children))


def _condition_side(program, field, side):
    """Compare branch laws without changing their measured reading value."""
    if (program.op in ("ge", "lt") and program.children[0].op == "feature"
            and program.children[0].name == field and program.children[1].op == "const"):
        return Program("const", value=float(side if program.op == "ge" else not side))
    return replace(program, children=tuple(_condition_side(child, field, side)
                                          for child in program.children))


def _condition_intervals(entry, fields):
    """Current recorded brackets, rather than older brackets replaced by them."""
    state = entry.get("condition_uncertainty") or {}
    current = (state.get("conditions", []) if state.get("program")
               == entry["candidate"].program.to_dict() else [])
    found = [{"condition_field": item["condition_field"],
              "threshold_interval": list(interval),
              "source_operation": "current_paid_condition_support",
              "candidate_id": entry["candidate"].candidate_id}
             for item in current if item["condition_field"] in fields
             for interval in item.get("threshold_intervals", []) if interval[0] < interval[1]]
    current_fields = {item["condition_field"] for item in current}
    steps = (entry.get("research_program") or {}).get("selection_steps", [])
    nested = [inner for step in steps for inner in step.get("inner_selection_steps", [])]
    for step in [*reversed(steps), *reversed(nested)]:
        coordinate, interval = step.get("gate_coordinate"), step.get("threshold_interval")
        if not coordinate or not interval:
            continue
        gate = Program.from_dict(coordinate)
        if (gate.op != "feature" or gate.name not in fields or gate.name in current_fields
                or any(item["condition_field"] == gate.name for item in found)):
            continue
        if interval[0] < interval[1]:
            found.append({"condition_field": gate.name,
                "threshold_interval": list(interval),
                "source_operation": step.get("operation"),
                "candidate_id": entry["candidate"].candidate_id})
    return found


def _cdf(values, value):
    """Interpolate only the support of the actually purchased residuals."""
    if value <= values[0]:
        return 0.
    if value >= values[-1]:
        return 1.
    index = bisect_right(values, value)
    lower, upper = values[index - 1:index + 1]
    return (index - 1 + (value - lower) / (upper - lower)) / (len(values) - 1)


def _confirmation_cdf(values, value):
    """P(reading < value) for the confirmation cells, including tied atoms."""
    index = bisect_left(values, value)
    if index == 0:
        return 0.
    if index == len(values):
        return 1.
    lower, upper = values[index - 1:index + 1]
    return (index - 1 + (value - lower) / (upper - lower)) / (len(values) - 1)


def _interval_exposure(values, left, right, budget):
    """Smooth a gap's mass over paid neighbours, including readings inside it."""
    if not budget.spend(2 * len(values).bit_length()):
        return None
    lower_index, upper_index = bisect_right(values, left), bisect_left(values, right)
    below_count, above_count = min(8, lower_index), min(8, len(values) - upper_index)
    if not budget.spend(below_count + above_count + 6):
        return None
    below = values[lower_index - below_count:lower_index]
    above = values[upper_index:upper_index + above_count]
    if not below or not above or above[-1] <= below[0]:
        # No local support spans this prospective interval. Keep the purchased
        # residual CDF rather than extrapolating a new instrument distribution.
        return {"estimated_probability": _cdf(values, right) - _cdf(values, left),
                "probability_basis": "paid_residual_cdf"}
    local_rows = len(below) + max(0, upper_index - lower_index) + len(above)
    density = max(1, local_rows - 1) / ((len(values) + 1) * (above[-1] - below[0]))
    return {"estimated_probability": min(1., max(0., (right - left) * density)),
            "probability_basis": "paid_local_residual_density",
            "residual_support_interval": [below[0], above[-1]],
            "local_residual_rows": local_rows}


def _local_residuals(rows, residuals, controls, bounds, point, budget):
    """Forecast an instrument from its nearest paid command neighbourhood.

    The neighbourhood grows with the available data. All boundary ties stay,
    so a repeated fixed-background experiment remains one measured sample.
    """
    if not budget.spend(len(rows) * (2 * len(controls) + 1)):
        return None
    distances = [sum(((row.features[name] - point[name]) /
        (bounds[name][1] - bounds[name][0])) ** 2 for name in controls) for row in rows]
    radius_squared = sorted(distances)[ceil(sqrt(len(rows))) - 1]
    indices = [index for index, distance in enumerate(distances) if distance <= radius_squared]
    closest = rows[min(range(len(rows)), key=distances.__getitem__)]
    spacings = [sum(((row.features[name] - closest.features[name]) /
        (bounds[name][1] - bounds[name][0])) ** 2 for name in controls) for row in rows]
    # Repeated controls can make the residual radius zero. The closest
    # distinct paid control still defines the model's spatial resolution;
    # an infinitesimal command change does not erase a measured miss.
    resolution = min((distance for distance in spacings if distance > 0.), default=0.)
    context = {"basis": "paid_local_control_residuals", "normalized_background": [
        (point[name] - bounds[name][0]) / (bounds[name][1] - bounds[name][0]) for name in controls],
        "neighbourhood_radius": sqrt(radius_squared),
        "applicability_radius": sqrt(max(radius_squared, resolution))}
    return indices, [residuals[index] for index in indices], context


def _instrument_signature(values, shifts, budget):
    """Describe a fitted instrument in units of its actually paid variation."""
    if not budget.spend(2 * len(values) + len(shifts)):
        return None
    centre = sum(values) / len(values)
    scale = sqrt(sum((value - centre) ** 2 for value in values) / len(values))
    if scale == 0.:
        return None
    return {"reading_scale": scale, "paid_rows": len(values),
            "control_sensitivity": [shift / scale for shift in shifts]}


def _sampling_question_context(context, instrument, ordered, predicted_mean, intervals, budget,
                               *, cdf=_cdf):
    """Save the paid residual shape and the current interval's CDF location."""
    if not budget.spend(15 + 2 * len(intervals) * len(ordered).bit_length()):
        return None
    quantiles = []
    for numerator in range(5):
        position = numerator * (len(ordered) - 1) / 4.
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        value = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        quantiles.append(value / instrument["reading_scale"])
    return {**context, "reading_forecast": {
        "control_sensitivity": instrument["control_sensitivity"],
        "paid_rows": instrument["paid_rows"], "local_rows": len(ordered),
        "residual_quantiles": quantiles,
        "question_cdf": [[cdf(ordered, left - predicted_mean),
                          cdf(ordered, right - predicted_mean)]
                         for left, right in sorted(intervals)]}}


def refresh_condition_sampling(spec, rows, plan, budget):
    """Update a fixed experiment's reading forecast from its paid new readings."""
    start = budget.spent
    controls = tuple(spec.feature_bounds)
    field = plan["condition_field"]
    calibration = [Observation({name: row.features[name] for name in controls},
                               row.features[field]) for row in rows]
    fit = _Fitter(calibration, budget, .5).fit([Program("feature", name=name) for name in controls])
    if fit is None or not budget.spend(2 * len(rows)):
        return None
    mean = fit["program"]
    residuals = [row.target - mean.predict(row) for row in calibration]
    point = plan["fixed_control_background"]
    local = _local_residuals(rows, residuals, controls, spec.feature_bounds, point, budget)
    if local is None:
        return None
    _, residuals, context = local
    predicted_mean = mean.predict(Observation(point, 0.))
    ordered = sorted(residuals)
    if not budget.spend(2 * len(controls)):
        return None
    shifts = [mean.predict(Observation({**point, name: upper}, 0.))
              - mean.predict(Observation({**point, name: lower}, 0.))
              for name, (lower, upper) in spec.feature_bounds.items()]
    instrument = _instrument_signature([row.features[field] for row in rows], shifts, budget)
    if instrument is None:
        return None
    exposures = []
    for exposure in plan.get("interval_exposure", []):
        left, right = exposure["threshold_interval"]
        probability = _interval_exposure(ordered, left - predicted_mean,
                                         right - predicted_mean, budget)
        if probability is None:
            return None
        exposures.append({**exposure, **probability})
    context = _sampling_question_context(context, instrument, ordered, predicted_mean,
        plan.get("sampling_question_intervals", [item["threshold_interval"] for item in exposures]), budget)
    if context is None:
        return None
    return {**plan, "interval_exposure": exposures, "sampling_context": context,
        "local_reading_rows": len(residuals), "paid_reading_rows": len(rows),
        "reading_mean_program": mean.to_dict(), "reading_mean_loss": fit["loss"],
        "empirical_noise_support": [ordered[0], ordered[-1]],
        "sampling_update_compute_cost": budget.spent - start}


def _disagreement(hypotheses, row, budget, *, partial=False):
    if not budget.spend(2 * len(hypotheses)):
        return None
    predictions, partial_predictions = [], []
    for hypothesis in hypotheses:
        try:
            value = (predict(hypothesis["program"], row, hypothesis["fallback"])
                     if "fallback" in hypothesis else hypothesis["program"].predict(row))
            predictions.append(value)
            if hypothesis.get("partial_structure"):
                partial_predictions.append(value)
        except (MissingHistory, KeyError):
            # An unguarded lag cannot predict a fresh probe's startup.
            continue
    difference = (max(predictions) - min(predictions)) ** 2 if predictions else 0.
    if partial:
        return difference, ((max(partial_predictions) - min(partial_predictions)) ** 2
                            if partial_predictions else 0.)
    return difference


def _boundary_response_risk(program, row, interval, probability, budget, *, comparison=None):
    """Predict the loss from a gap event anywhere in its actual window."""
    field = interval["condition_field"]
    left, right = interval["threshold_interval"]
    programs = [item["program"] for item in comparison] if comparison is not None else [program]
    width = max((width for candidate in programs for name, _, width in _gates(candidate)
                 if name == field), default=1)
    if comparison is None:
        low, high = (_threshold(program, field, value) for value in (left, right))
        baseline = 0.
    else:
        if not budget.spend(2):
            return None
        baseline = (predict(comparison[0]["program"], row, comparison[0]["fallback"])
                    - predict(comparison[1]["program"], row, comparison[1]["fallback"]))
    differences = []
    for position in (-1, *range(max(0, len(row.history) - width + 1), len(row.history))):
        if not budget.spend(2):
            return None
        altered = _at_reading(row, field, (left + right) / 2., position)
        try:
            differences.append((predict(comparison[0]["program"], altered, comparison[0]["fallback"])
                - predict(comparison[1]["program"], altered, comparison[1]["fallback"]) - baseline)
                if comparison is not None else low.predict(altered) - high.predict(altered))
        except (MissingHistory, KeyError):
            continue
    # The paid residual model proposes independent future reading draws.
    # Keep signed cross terms when several gap events affect one response.
    # The frozen pair's existing disagreement is already priced by its
    # trajectory; these are only the increments induced by additional gaps.
    return ((1. - probability) * sum(value * value for value in differences)
            + probability * sum(differences) ** 2 + 2. * baseline * sum(differences))


def expected_confirmation_contrast(hypotheses, features, field, intervals, residuals,
                                   reading_mean, budget):
    """Integrate rare reading events instead of replaying a past hit as future fact.

    Adjacent paid residuals define equal-mass uniform cells; tied values retain
    point mass. A no-gap reference trajectory and per-position response moments
    share this distribution. The moment expansion is exact for additive event
    responses, including signed windows, and approximates nonlinear interactions.
    """
    ordered = sorted(value + reading_mean for value in residuals)
    cells = ([(ordered[0], ordered[0], 1.)] if len(ordered) == 1 else
             [(left, right, 1. / (len(ordered) - 1)) for left, right in zip(ordered, ordered[1:])])
    cuts = sorted({value for item in intervals for value in item["threshold_interval"]})
    if not budget.spend(len(cells) * (len(cuts) + len(intervals) + 1)):
        return None
    outside, event_cells = [], [[] for _ in intervals]
    for left, right, weight in cells:
        edges = [left, *(cut for cut in cuts if left < cut < right), right]
        parts = [(left, right, weight)] if left == right else [
            (low, high, weight * (high - low) / (right - left))
            for low, high in zip(edges, edges[1:])]
        for low, high, mass in parts:
            point = (low + high) / 2.
            index = next((index for index, item in enumerate(intervals)
                if item["threshold_interval"][0] <= point < item["threshold_interval"][1]), None)
            (outside if index is None else event_cells[index]).append((low, high, mass))
    probabilities = [sum(item[2] for item in cells) for cells in event_cells]
    representatives = [sum((low + high) / 2. * mass for low, high, mass in cells) / probability
        if probability else 0. for cells, probability in zip(event_cells, probabilities)]
    outside_mass = sum(item[2] for item in outside)
    history, forecasts, baseline_differences = [], [], []
    for index, template in enumerate(features):
        if not budget.spend(len(outside) + 2):
            return None
        # Midpoint quadrature in the conditional no-gap CDF has no privileged
        # past observation or temporal ordering. A full-mass event can use its
        # upper cut as a zero-probability reference for the moment expansion.
        reading = max(cuts) if not outside else outside[-1][1]
        quantile = (index + .5) / len(features) * outside_mass
        for left, right, mass in outside:
            if quantile <= mass:
                reading = left + (right - left) * quantile / mass
                break
            quantile -= mass
        row = Observation({**template, field: reading}, 0., tuple(history))
        new = predict(hypotheses[0]["program"], row, hypotheses[0]["fallback"])
        old = predict(hypotheses[1]["program"], row, hypotheses[1]["fallback"])
        forecasts.append(row)
        baseline_differences.append(new - old)
        history.append({**row.features, "__target__": new})
    width = max((width for item in hypotheses for name, _, width in _gates(item["program"])
                 if name == field), default=1)
    expected, interval_effects = [], [0.] * len(intervals)
    for row, baseline in zip(forecasts, baseline_differences):
        mean_delta, variance_delta = 0., 0.
        for position in (-1, *range(max(0, len(row.history) - width + 1), len(row.history))):
            first, second = 0., 0.
            for index, (probability, reading) in enumerate(zip(probabilities, representatives)):
                if not probability:
                    continue
                if not budget.spend(2):
                    return None
                altered = _at_reading(row, field, reading, position)
                difference = (predict(hypotheses[0]["program"], altered, hypotheses[0]["fallback"])
                    - predict(hypotheses[1]["program"], altered, hypotheses[1]["fallback"]))
                delta = difference - baseline
                # Intervals at the same position are mutually exclusive.
                first += probability * delta
                second += probability * delta * delta
                interval_effects[index] += delta * delta / len(forecasts)
            mean_delta += first
            variance_delta += max(0., second - first * first)
        expected.append((baseline + mean_delta) ** 2 + variance_delta)
    trajectory = sum(value * value for value in baseline_differences) / len(forecasts)
    total = sum(expected) / len(expected)
    return {"expected_disagreement": total, "trajectory_disagreement": trajectory,
        "boundary_disagreement": total - trajectory, "startup_disagreement": expected[0],
        "interval_exposure": [{**item, "estimated_probability": probability,
            "probability_basis": "paid_residual_cdf", "response_disagreement": effect}
            for item, probability, effect in zip(intervals, probabilities, interval_effects)]}


def _supports_forecast(entry, measured_loss, tolerance):
    """A fitted structure does not validate its present response parameters."""
    program = entry["candidate"].program.to_dict()
    state = entry.get("condition_uncertainty") or {}
    if state.get("program") == program and state.get("original_target_loss") is not None:
        measured_loss = max(measured_loss, state["original_target_loss"])
    noise = entry.get("response_noise_evidence") or {}
    replication = noise.get("replication", {}) if noise.get("program") == program else {}
    if replication.get("repeatable_error"):
        return False
    if measured_loss <= tolerance:
        return True
    # Noise-inconclusive parameters remain research proposals. A measured
    # repeat can also keep noisy forecasts in play, but cannot explain a new
    # error beyond its own observed allowance. No apparatus noise label enters.
    limit = replication.get("repeatable_loss_limit")
    return bool(replication.get("noise_supported") and limit is not None
                and measured_loss <= limit)


def _local_condition_support(entries, fields, tolerance):
    """Keep measured local geometry when a complete response needs refitting."""
    nominated, intervals = [], []
    for entry in entries:
        evidence = list(entry.get("structure_nominations", ()))
        if entry.get("observed_condition_evidence"):
            evidence.append(entry["observed_condition_evidence"])
        for item in evidence:
            field = item.get("control")
            if field not in fields:
                continue
            for support in item.get("partial_support") or (item,):
                local_loss = support.get("local_loss", support.get("loss", float("inf")))
                supported = support.get("response_supported") and local_loss <= tolerance
                if (supported or support.get("gain", 0.) > tolerance) and field not in nominated:
                    nominated.append(field)
                interval = support.get("threshold_interval")
                if supported and support.get("condition_identified") and interval and interval[0] < interval[1]:
                    intervals.append({"condition_field": field, "threshold_interval": list(interval),
                        "source_operation": "paid_local_condition_support",
                        "candidate_id": entry["candidate"].candidate_id})
    return nominated, intervals


def plan_observed_condition_probe(spec, rows, selected, budget, *, serial=0, length=24,
                                  finalists=(), previous_experiments=(), partial_competitors=(),
                                  validation=(), tolerance=1e-7, confirmation_baseline=None):
    """Choose a legal, fixed background to separate competing explanations.

    Purchased reading residuals supply a prospective trajectory and an
    empirical estimate of each unresolved bracket's exposure. Neither these
    forecasts nor a threshold midpoint set the instrument's realised reading.
    The returned probe holds raw controls fixed for the duration experiment.
    """
    rows = [*rows, *(row for row in validation if row not in rows)]
    if len(rows) < 8 or selected is None:
        return None
    fields = [name for name in spec.feature_names if name not in spec.feature_bounds
              and all(name in row.features for row in rows)
              and len({row.features[name] for row in rows}) > 1]
    if not fields or not spec.feature_bounds:
        return None
    previous = [item for item in previous_experiments
                if item.get("fixed_control_background")]
    purchased_backgrounds = [item["fixed_control_background"] for item in previous]
    entries = ([selected, confirmation_baseline] if confirmation_baseline is not None
               else list((selected, *finalists, *partial_competitors)))
    start = budget.spent
    if not budget.spend(len(rows) * len(entries)):
        return None
    fallback = initial_prediction(rows, .5)
    predictions = [[predict(entry["candidate"].program, row,
        entry.get("fallback", fallback) if confirmation_baseline is not None else fallback)
        for entry in entries] for row in rows]
    errors = [[(value - row.target) ** 2 for value in predicted]
              for row, predicted in zip(rows, predictions)]
    losses = [sum(values[index] for values in errors) / len(rows) for index in range(len(entries))]
    supported = [index for index, entry in enumerate(entries)
                 if _supports_forecast(entry, losses[index], tolerance)]
    if confirmation_baseline is not None:
        # Confirmation asks whether this concrete repair differs from the old
        # response. Its already-refuted baseline is still part of that question.
        supported = [0, 1]
    # Keep every structure in the caller's refit pool. Only the concrete
    # parameters supported on this shared evidence can price an extrapolation.
    forecasting = [entries[index] for index in supported]
    reference_index = (0 if 0 in supported else min(supported or range(len(entries)), key=losses.__getitem__))
    reference = entries[reference_index]
    nominated_fields, local_intervals = _local_condition_support(entries, fields, tolerance)
    intervals = [interval for entry in forecasting for interval in _condition_intervals(entry, fields)]
    intervals.extend(local_intervals)
    if confirmation_baseline is not None:
        contrasting_intervals = []
        for name in fields:
            cuts = sorted({cut for entry in entries for field, cut, _
                           in _gates(entry["candidate"].program.expanded()) if field == name})
            contrasting_intervals.extend({"condition_field": name,
                "threshold_interval": [left, right], "source_operation": "frozen_response_cuts"}
                for left, right in zip(cuts, cuts[1:]))
        intervals = contrasting_intervals or intervals
    field = intervals[0]["condition_field"] if intervals else (nominated_fields or fields)[0]
    distinct_intervals = []
    for item in intervals:
        if (item["condition_field"] == field and not any(
                item["threshold_interval"] == old["threshold_interval"] for old in distinct_intervals)):
            distinct_intervals.append(item)
    intervals = distinct_intervals
    values = sorted(row.features[field] for row in rows)
    left, right = (intervals[0]["threshold_interval"] if intervals else
                   (values[len(values) // 4], values[3 * len(values) // 4]))
    if left >= right:
        return None
    controls = tuple(spec.feature_bounds)
    calibration_rows = [Observation({name: row.features[name] for name in controls},
                                    row.features[field]) for row in rows]
    fitter = _Fitter(calibration_rows, budget, .5)
    fit = fitter.fit([Program("feature", name=name) for name in controls])
    if fit is None:
        return None
    mean = fit["program"]
    if not budget.spend(2 * len(rows)):
        return None
    residuals = [row.target - mean.predict(row) for row in calibration_rows]
    ordered_residuals = sorted(residuals)
    hypotheses = []
    for entry in forecasting:
        program = entry["candidate"].program.expanded()
        variants = [(program, "response", None)]
        for interval in (() if confirmation_baseline is not None else _condition_intervals(entry, (field,))):
            for endpoint in interval["threshold_interval"]:
                endpoint_program = _threshold(program, field, endpoint)
                if endpoint_program != program:
                    variants.append((endpoint_program, "threshold_endpoint", endpoint))
        for program, kind, endpoint in variants:
            is_partial = bool(entry.get("partial_support")) and kind == "response"
            duplicate = next((item for item in hypotheses if program == item["program"]
                and (confirmation_baseline is None or item["fallback"] == entry.get("fallback", fallback))), None)
            if duplicate is None:
                hypotheses.append({"program": program, "candidate_id": entry["candidate"].candidate_id,
                    "kind": kind, "threshold_endpoint": endpoint, "partial_structure": is_partial,
                    **({"fallback": entry.get("fallback", fallback)} if confirmation_baseline is not None else {})})
            elif is_partial:
                duplicate["partial_structure"] = True
    if confirmation_baseline is not None and len(hypotheses) < 2:
        return None
    selected_program = reference["candidate"].program.expanded()
    sides = tuple(_condition_side(selected_program, field, side) for side in (False, True))
    branch_pair = sides if (confirmation_baseline is None and forecasting and previous
                            and sides[0] != sides[1]) else ()
    centre = {name: (bounds[0] + bounds[1]) / 2.
              for name, bounds in spec.feature_bounds.items()}
    shifts = {}
    for name, (lower, upper) in spec.feature_bounds.items():
        if not budget.spend(2):
            return None
        shifts[name] = (mean.predict(Observation({**centre, name: upper}, 0.))
                        - mean.predict(Observation({**centre, name: lower}, 0.)))
    driver = max(controls, key=lambda name: abs(shifts[name]))
    lower, upper = spec.feature_bounds[driver]
    slope = shifts[driver] / (upper - lower)
    points = (dict(zip(controls, endpoints)) for endpoints in
              product(*(spec.feature_bounds[name] for name in controls)))
    choices = []
    # An unresolved affine response need not already contain a gate. Its
    # largest paid residual still nominates a background for separating laws.
    residual_indices = supported or [reference_index]
    counterexample = rows[max(range(len(rows)),
        key=lambda index: max(errors[index][candidate] for candidate in residual_indices))]
    if confirmation_baseline is not None:
        counterexample = rows[max(range(len(rows)),
            key=lambda index: (predictions[index][0] - predictions[index][1]) ** 2)]
    selected_error = max(values[reference_index] for values in errors)
    counterexample_point = {name: counterexample.features[name] for name in controls}
    anchors = list(dict.fromkeys((sum(item["threshold_interval"]) / 2. for item in intervals))) or [
        (left + right) / 2.]
    if confirmation_baseline is not None:
        anchors = list(dict.fromkeys([*anchors, counterexample.features[field]]))
    count = min(length, spec.max_probe_length)
    current_state = reference.get("condition_uncertainty") or {}
    threshold_question = (bool(forecasting) and current_state.get("status") == "unresolved"
        and current_state.get("program") == reference["candidate"].program.to_dict()
        and current_state.get("conditions")
        and all(not item.get("needs_threshold_revision")
                and item.get("status") != "contradicted"
                for item in current_state["conditions"]))
    current_brackets = {(item["condition_field"], tuple(interval))
        for item in current_state.get("conditions", [])
        for interval in item.get("threshold_intervals", [])}
    partial_question = sum(bool(item["partial_structure"]) for item in hypotheses) > 1
    forecast_length = 0 if not forecasting else count if (confirmation_baseline is not None or partial_question or threshold_question) else (min(count, max(8, *(item["program"].history_depth + 2
                                         for item in hypotheses)))
                       if len(hypotheses) > 1 or intervals or branch_pair else 0)
    instrument = _instrument_signature(values, shifts.values(), budget) if forecast_length else None
    if forecast_length and instrument is None:
        return None
    candidate_work = (forecast_length * (2 * len(hypotheses) + 1 + len(branch_pair))
                      + (2 * len(hypotheses) + 2 * len(rows).bit_length() + 22)
                      * len(intervals) + (2 * forecast_length * ((max(
                          item["program"].history_depth for item in hypotheses)
                          if confirmation_baseline is not None else selected_program.history_depth)
                          + (2 if confirmation_baseline is not None else 1))
                      * len(intervals) if threshold_question or confirmation_baseline is not None else 0) + 3) if forecast_length else 4
    if forecast_length:
        candidate_work += len(rows) * (2 * len(controls) + 1)
    if confirmation_baseline is not None:
        # The shared distribution's integration cost depends on the paid
        # neighbourhood, which is known only after the local lookup below.
        candidate_work = len(rows) * (2 * len(controls) + 1) + 3
    context_work = 16 + 2 * len(intervals) * len(rows).bit_length() if forecast_length else 0
    evaluated = []
    for raw_point in (counterexample_point, centre, *points):
        # A measured failure of the old background is evidence about the
        # experiment. Its next version may also change the reading distribution
        # instead of repeatedly forcing every command point to the same mean.
        for anchor in ((None, *anchors) if confirmation_baseline is not None
                       else (*anchors, None) if previous else anchors):
            # This is one research step within a larger joint fit. Complete
            # comparable forecasts while leaving that fit its quoted work.
            if (candidate_work + context_work > budget.remaining
                    or budget.spent - start + candidate_work + context_work > 15000):
                break
            if not budget.spend(3 if forecast_length else 4):
                break
            point = dict(raw_point)
            if anchor is not None and abs(shifts[driver]) > 1e-10:
                predicted = mean.predict(Observation(point, 0.)) + median(residuals)
                point[driver] = min(upper, max(lower, point[driver] + (anchor - predicted) / slope))
            if point in purchased_backgrounds:
                continue
            if not forecast_length:
                # No competing law is available yet. Keep the paid-residual
                # background experiment and its fee; invented future disagreement
                # cannot help this initial duration discovery.
                choices.append((0., point, 0., 0., [], 0., 0., 0., 0., None, len(rows),
                                ordered_residuals))
                continue
            if point in evaluated:
                continue
            evaluated.append(point)
            predicted_mean = mean.predict(Observation(point, 0.))
            neighbourhood = _local_residuals(rows, residuals, controls,
                spec.feature_bounds, point, budget)
            if neighbourhood is None:
                break
            local_indices, local_residuals, sampling_context = neighbourhood
            local_ordered = sorted(local_residuals)
            if confirmation_baseline is not None:
                cut_count = len({value for item in intervals for value in item["threshold_interval"]})
                cell_count = max(1, len(local_residuals) - 1)
                width = max((width for item in hypotheses for name, _, width in _gates(item["program"])
                             if name == field), default=1)
                integration_work = (cell_count * (cut_count + len(intervals) + 1)
                    + forecast_length * (cell_count + cut_count + 2)
                    + 2 * forecast_length * width * len(intervals))
                if (integration_work + context_work > budget.remaining
                        or budget.spent - start + integration_work + context_work > 15000):
                    break
                templates = [{**point, **{name: rows[local_indices[index % len(local_indices)]].features[name]
                                         for name in fields}}
                             for index in range(forecast_length)]
                contrast = expected_confirmation_contrast(hypotheses, templates, field,
                    intervals, local_residuals, predicted_mean, budget)
                if contrast is None:
                    break
                choices.append((contrast["expected_disagreement"], point,
                    contrast["trajectory_disagreement"], contrast["boundary_disagreement"],
                    contrast["interval_exposure"], contrast["startup_disagreement"],
                    0., 0., 0., sampling_context, len(local_residuals), local_ordered))
                continue
            history, differences, exposure, branch_differences, partial_differences, forecast_rows = [], [], [], [], [], []
            exhausted = False
            for index in range(forecast_length):
                readings = {name: rows[local_indices[index % len(local_indices)]].features[name] for name in fields}
                readings[field] = predicted_mean + local_residuals[index % len(local_residuals)]
                row = Observation({**point, **readings}, 0., tuple(history))
                forecast_rows.append(row)
                result = _disagreement(hypotheses, row, budget, partial=True)
                if result is None:
                    exhausted = True
                    break
                difference, partial_difference = result
                differences.append(difference)
                partial_differences.append(partial_difference)
                if branch_pair:
                    if not budget.spend(len(branch_pair)):
                        exhausted = True
                        break
                    try:
                        branch_differences.append((sides[1].predict(row) - sides[0].predict(row)) ** 2)
                    except (MissingHistory, KeyError):
                        branch_differences.append(0.)
                # The fixed trajectory is shared evidence. For response-lag
                # hypotheses the selected law supplies the prospective target.
                if not budget.spend(1):
                    exhausted = True
                    break
                try:
                    target = (predict(selected_program, row, selected.get("fallback", fallback))
                        if confirmation_baseline is not None else selected_program.predict(row))
                except MissingHistory:
                    target = 0.
                history.append({**row.features, "__target__": target})
            if exhausted:
                break
            for interval in intervals:
                lower_cut, upper_cut = interval["threshold_interval"]
                probability = _interval_exposure(local_ordered,
                    lower_cut - predicted_mean, upper_cut - predicted_mean, budget)
                if probability is None:
                    exhausted = True
                    break
                if confirmation_baseline is not None or threshold_question:
                    sensitivities = [_boundary_response_risk(selected_program, row, interval,
                        probability["estimated_probability"], budget,
                        comparison=hypotheses if confirmation_baseline is not None else None)
                        for row in forecast_rows]
                    if any(value is None for value in sensitivities):
                        exhausted = True
                        break
                    difference = sum(sensitivities) / len(sensitivities)
                else:
                    boundary_features = dict(history[-1])
                    boundary_features.pop("__target__")
                    boundary_features[field] = (lower_cut + upper_cut) / 2.
                    boundary = Observation(boundary_features, 0., tuple(history[:-1]))
                    difference = _disagreement(hypotheses, boundary, budget)
                    if difference is None:
                        exhausted = True
                        break
                exposure.append({**interval, **probability,
                                 "response_disagreement": difference})
            if exhausted:
                break
            trajectory_disagreement = sum(differences) / forecast_length
            # Price the current response's remaining question. A wider local
            # proposal stays available, but its exposure is not evidence that
            # this background will distinguish the current narrower bracket.
            boundary_disagreement = sum(item["estimated_probability"] * item["response_disagreement"]
                for item in exposure if confirmation_baseline is not None or not threshold_question or
                (item["condition_field"], tuple(item["threshold_interval"])) in current_brackets)
            cutoff = sum(intervals[0]["threshold_interval"]) / 2. if intervals else (left + right) / 2.
            lower_probability = _cdf(local_ordered, cutoff - predicted_mean)
            branch_balance = 4. * lower_probability * (1. - lower_probability)
            branch_contrast = sum(branch_differences) / forecast_length if branch_pair else 0.
            score = trajectory_disagreement + boundary_disagreement
            choices.append((max(0., score) if confirmation_baseline is not None else score, point,
                            trajectory_disagreement, boundary_disagreement, exposure, differences[0],
                            branch_contrast, branch_balance, sum(partial_differences) / forecast_length,
                            sampling_context, len(local_residuals),
                            local_ordered))
        if budget.remaining < 2 * len(hypotheses) * forecast_length:
            break
    if not choices:
        return None
    # A response that already explains the paid targets has a boundary
    # question. A globally poorer family cannot replace that question merely
    # by predicting a larger difference on an unmeasured background.
    if confirmation_baseline is not None:
        choice = max(choices, key=lambda item: item[0])
    elif threshold_question:
        choice = max(choices, key=lambda item: (item[3], item[0]))
    elif partial_question:
        choice = max(choices, key=lambda item: (item[8], item[0]))
    else:
        choice = max(choices, key=(lambda item: (item[6] * item[7], item[0]))
                     if branch_pair and not threshold_question else lambda item: item[0])
    (disagreement, point, temporal, boundary, exposure, startup, branch_contrast,
        branch_balance, partial_difference, sampling_context, local_row_count, chosen_residuals) = choice
    if disagreement <= 1e-12 and (confirmation_baseline is not None or selected_error <= 1e-12):
        return None
    sampling_intervals = [item["threshold_interval"] for item in exposure if confirmation_baseline is not None or not threshold_question
        or (item["condition_field"], tuple(item["threshold_interval"])) in current_brackets]
    if sampling_context is not None:
        if not budget.spend(1):
            return None
        sampling_context = _sampling_question_context(sampling_context, instrument,
            chosen_residuals, mean.predict(Observation(point, 0.)),
            sampling_intervals, budget,
            cdf=_confirmation_cdf if confirmation_baseline is not None else _cdf)
        if sampling_context is None:
            return None
    # Hold every command constant. The purchased reading can still vary;
    # startup and its response then distinguish temporal gate placement and
    # duration without a changing quadratic control response confounding them.
    sequence = tuple(dict(point) for _ in range(count))
    return {"probe": Probe(f"observed_condition:{serial}:{field}", sequence),
        "condition_field": field, "threshold_interval": [left, right],
        "background_disagreement": disagreement, "distribution_control": driver,
        "experiment_kind": "fixed_background", "expected_disagreement": disagreement,
        **({"confirmation_pair": {
            "baseline_program": confirmation_baseline["candidate"].program.to_dict(),
            "baseline_fallback": confirmation_baseline.get("fallback", fallback),
            "program": selected["candidate"].program.to_dict(),
            "fallback": selected.get("fallback", fallback), "basis": "frozen_response_predictions"}}
            if confirmation_baseline is not None else {}),
        "duration_disagreement": temporal, "boundary_disagreement": boundary,
        "startup_disagreement": startup, "threshold_intervals": intervals,
        "predicted_branch_contrast": branch_contrast,
        "estimated_branch_balance": branch_balance,
        "experiment_revision_basis": ("frozen_response_revision" if confirmation_baseline is not None
                                      else "unmeasured_temporal_boundary" if threshold_question
                                      else "cross_background_partial_support" if partial_question
                                      else "measured_background_misfit" if previous or not forecasting else None),
        "partial_structure_disagreement": partial_difference,
        "previous_experiment_count": len(previous),
        "forecast_length": forecast_length,
        "interval_exposure": exposure, "planner_candidate_count": len(choices),
        "competing_hypotheses": [{key: value for key, value in item.items() if key != "program"}
                                 for item in hypotheses],
        "sampling_basis": ("paid_residual_cdf" if confirmation_baseline is not None
                           else "paid_local_residual_density" if intervals else "paid_reading_quantiles"),
        "sampling_context": sampling_context, "local_reading_rows": local_row_count,
        "sampling_question_intervals": sampling_intervals,
        "purpose": ("confirm_frozen_response_revision" if confirmation_baseline is not None
                    else "distinguish_competing_condition_and_duration"), "fixed_control_background": point,
        "reading_mean_program": mean.to_dict(), "reading_mean_loss": fit["loss"],
        "paid_reading_rows": len(rows), "empirical_noise_support": [chosen_residuals[0], chosen_residuals[-1]],
        "compute_cost": budget.spent - start}
