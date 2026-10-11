"""Choose experiments by task risk reduction or raw trajectory disagreement."""

from __future__ import annotations

from math import exp
from statistics import mean, pvariance
from random import Random

from .evaluation import point_loss, predict
from .models import Candidate, Observation, Probe, SearchBudget, TaskSpec


def construct_probes(spec: TaskSpec, candidates: list[Candidate], *, round_number: int,
                     seed: int = 7, max_history: int = 4) -> list[Probe]:
    """Build interventions from program boundaries and alternate histories.

    This uses only the public controllable ranges and candidate programs. It
    does not consult a task generator or identify the hidden mechanism.
    """
    if not spec.feature_bounds:
        return []
    rng = Random(seed + round_number * 113)
    middle = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
    length = min(spec.max_probe_length, max(4, max_history + 2))
    results = []
    sequences = set()

    def add(sequence):
        key = tuple(tuple(sorted(point.items())) for point in sequence)
        if key in sequences:
            return
        sequences.add(key)
        results.append(Probe(f"constructed_{round_number}_{len(results)}", tuple(sequence)))

    boundaries = {name: [] for name in spec.feature_bounds}

    def visit(program):
        if program.op in {"ge", "lt"}:
            left, right = program.children
            if left.op == "feature" and right.op == "const" and left.name in boundaries:
                boundaries[left.name].append(right.value)
        for child in program.children:
            visit(child)

    for candidate in candidates:
        visit(candidate.program)
    for name, (low, high) in spec.feature_bounds.items():
        ascending = [low + (high - low) * i / max(1, length - 1) for i in range(length)]
        add([{**middle, name: value} for value in ascending])
        add([{**middle, name: value} for value in reversed(ascending)])
        # Equal final input with different histories can split explanations
        # that agree on all currently observed memoryless points.
        add([{**middle, name: low if i < length - 1 else middle[name]} for i in range(length)])
        add([{**middle, name: high if i < length - 1 else middle[name]} for i in range(length)])
        for boundary in sorted(set(boundaries[name]))[:2]:
            epsilon = (high - low) / 25
            left, right = max(low, boundary - epsilon), min(high, boundary + epsilon)
            if low <= left < right <= high:
                add([{**middle, name: left if i % 2 == 0 else right} for i in range(length)])
    for _ in range(2):
        add([{name: rng.uniform(*spec.feature_bounds[name]) for name in spec.feature_names}
             for _ in range(length)])
    return results


def observed_classes(candidates: list[Candidate], observations: list[Observation],
                     fallback: float, budget: SearchBudget) -> list[list[str]]:
    """Report empirical tolerance groups without deleting their programs."""
    groups = {}
    for candidate in candidates:
        if not budget.spend(len(observations)):
            break
        vector = tuple(round(predict(candidate.program, row, fallback), 6) for row in observations)
        groups.setdefault(vector, []).append(candidate.candidate_id)
    return list(groups.values())


def choose_probe(probes: list[Probe], candidates: list[Candidate], fallback: float,
                 budget: SearchBudget) -> tuple[Probe | None, float]:
    best_probe, best_value = None, -1.0
    for probe in probes:
        cost = probe.cost * len(candidates)
        if not budget.spend(cost):
            break
        predictions = []
        for candidate in candidates:
            history, outcomes = [], []
            for features in probe.sequence:
                value = predict(candidate.program, Observation(features, 0.0, tuple(history)), fallback)
                outcomes.append(value)
                history.append({**features, "__target__": value})
            predictions.append(outcomes)
        # Average disagreement can reward a useful split even when some pair
        # of Boolean candidates agrees. Maximin pair distance cannot do this.
        value = sum(pvariance(column) for column in zip(*predictions)) / probe.cost
        if value > best_value:
            best_probe, best_value = probe, value
    return best_probe, max(0.0, best_value)


def _sequence_key(probe: Probe):
    return tuple(tuple(sorted(point.items())) for point in probe.sequence)


def _sample_rows(rows, maximum):
    if len(rows) <= maximum:
        return list(rows)
    return [rows[round(index * (len(rows) - 1) / (maximum - 1))] for index in range(maximum)]


def _control_ranges(spec, observations):
    ranges = dict(spec.feature_bounds)
    for name in spec.feature_names:
        if name not in ranges:
            values = [row.features[name] for row in observations]
            ranges[name] = (min(values), max(values)) if values else (-1.0, 1.0)
    return ranges


def _control_descriptor(features, history, names, ranges):
    def normalized(point, name):
        low, high = ranges[name]
        return (point[name] - low) / (high - low) if high != low else 0.0

    current = [normalized(features, name) for name in names]
    descriptor = list(current)
    # Contrasts expose controls previously varied together; lagged controls and
    # their increments distinguish a changed input from a constant trajectory.
    descriptor.extend(current[left] - current[right]
                      for left in range(len(names)) for right in range(left))
    for lag in (1, 2):
        previous = history[-lag] if len(history) >= lag else features
        before = [normalized(previous, name) for name in names]
        descriptor.extend(before)
        descriptor.extend(now - old for now, old in zip(current, before))
    return tuple(descriptor)


def _coverage(probe, observed_descriptors, names, ranges):
    visited = list(observed_descriptors)
    history, contributions = [], []
    distinct = []
    for point in probe.sequence:
        descriptor = _control_descriptor(point, history, names, ranges)
        distance = min((mean(abs(left - right) for left, right in zip(descriptor, previous))
                        for previous in visited), default=1.0)
        contributions.append(distance)
        visited.append(descriptor)
        if descriptor not in distinct:
            distinct.append(descriptor)
        history.append(point)
    return {"coverage_per_observation": mean(contributions),
            "distinct_control_histories": len(distinct), "observation_cost": probe.cost}


def _reference_probes(spec, ranges, history_depth):
    # A fixed interior stratified design represents the declared controllable
    # domain. Query extrema and candidate-proposed spikes cannot set its mass.
    length = min(spec.max_probe_length, max(4, history_depth + 1))
    count = 2 * length
    rng = Random(271828)
    columns = {}
    for name in spec.feature_names:
        order = list(range(count))
        rng.shuffle(order)
        low, high = ranges[name]
        columns[name] = [low + (high - low) * (index + 0.5) / count for index in order]
    points = [{name: columns[name][index] for name in spec.feature_names} for index in range(count)]
    return [Probe(f"reference_{index}", tuple(points[index * length:(index + 1) * length]))
            for index in range(2)]


def _trajectory_predictions(candidate, probe, fallback):
    history, values = [], []
    for features in probe.sequence:
        value = predict(candidate.program, Observation(features, 0.0, tuple(history)), fallback)
        values.append(value)
        history.append({**features, "__target__": value})
    return values


def _task_loss(action, target, spec):
    if spec.decision_costs is not None:
        under, over = spec.decision_costs
        return under * max(target - action, 0.0) + over * max(action - target, 0.0)
    return point_loss(action, target, spec.risk_quantile)


def _risk(weights, losses):
    return min(sum(weight * value for weight, value in zip(weights, action_losses))
               for action_losses in losses)


def _expected_query_risk(predictions, weights, reference_losses, noise_scale):
    # Candidate trajectories provide possible outcomes. Normalize the response
    # kernel per hypothesized model, yielding an actual finite observation
    # channel; its posterior risks are weighted by outcome probabilities.
    outcomes = []
    for trajectory in predictions:
        if trajectory not in outcomes:
            outcomes.append(trajectory)
    likelihood = [[exp(-min(700.0, mean((a - b) ** 2 for a, b in zip(outcome, prediction))
                                    / (2 * noise_scale)))
                   for prediction in predictions] for outcome in outcomes]
    column_totals = [sum(row[index] for row in likelihood) for index in range(len(weights))]
    expected = 0.0
    for row in likelihood:
        joint = [weight * probability / total
                 for weight, probability, total in zip(weights, row, column_totals)]
        probability = sum(joint)
        if probability:
            expected += probability * _risk([value / probability for value in joint], reference_losses)
    return expected


def choose_valuable_probe(probes: list[Probe], candidates: list[Candidate], fallback: float,
                          budget: SearchBudget, *, observations: list[Observation],
                          spec: TaskSpec, queried_probes=()) -> tuple[Probe | None, dict]:
    """Select expected reference-task risk reduction per acquired observation.

    This is a bounded finite-model approximation, using at most four weighted
    candidates and public reference controls. Coverage of controls and histories
    resolves ties and supports experiments before a useful competitor exists.
    At most one fifth of remaining compute, capped at 1200 units, is spent here.
    Units include prediction evaluations and cached response/loss comparisons.
    """
    before = budget.spent
    allowance = min(1200, budget.remaining // 5)
    work = SearchBudget(allowance)
    details = {"policy": "expected_task_risk_reduction", "compute_allowance": allowance,
               "candidate_weights": [], "reference_risk": None,
               "expected_risk_reduction": 0.0, "value_per_observation": 0.0,
               "reference_design": "fixed_public_interior_stratified_trajectories",
               "evaluated_probes": [], "compute_cost": 0}
    queried = {_sequence_key(probe) for probe in queried_probes}
    eligible, seen = [], set(queried)
    for probe in probes:
        key = _sequence_key(probe)
        if key not in seen and probe.cost:
            eligible.append(probe)
            seen.add(key)
    details["duplicate_or_queried_trajectories"] = len(probes) - len(eligible)
    if not eligible or not allowance:
        details["status"] = "no_new_trajectory" if not eligible else "insufficient_selection_budget"
        return None, details

    training = _sample_rows(observations, 16)
    ranges = _control_ranges(spec, training)
    observed_descriptors = [_control_descriptor(row.features, row.history, spec.feature_names, ranges)
                            for row in training]
    coverage = {probe.probe_id: _coverage(probe, observed_descriptors, spec.feature_names, ranges)
                for probe in eligible}
    # Examine diverse, non-repeated experiments first if the remaining budget
    # cannot value every trajectory. This order is independent of model outputs.
    eligible.sort(key=lambda probe: (-coverage[probe.probe_id]["coverage_per_observation"],
                                    -coverage[probe.probe_id]["distinct_control_histories"], probe.cost))
    selected = eligible[0]
    unique_candidates = []
    for candidate in candidates:
        if all(candidate.program != previous.program for previous in unique_candidates):
            unique_candidates.append(candidate)
    fitted = []
    for candidate in unique_candidates[:8]:
        if not training or not work.spend(len(training)):
            break
        values = [predict(candidate.program, row, fallback) for row in training]
        measured = mean(_task_loss(value, row.target, spec) for value, row in zip(values, training))
        residual_mse = mean((value - row.target) ** 2 for value, row in zip(values, training))
        fitted.append((measured, residual_mse, candidate))
    fitted.sort(key=lambda item: (item[0], item[2].program.size))
    count = min(4, len(fitted))
    references = _reference_probes(spec, ranges, max((item[2].program.history_depth for item in fitted[:count]), default=0))
    reference_rows = sum(probe.cost for probe in references)
    # Reserve enough local work for at least two valued query alternatives.
    query_rows = sum(probe.cost for probe in eligible[:2])
    while count > 1 and (count + count * count) * (reference_rows + query_rows) > work.remaining:
        count -= 1
    if count > 1:
        fitted = fitted[:count]
        base_loss = mean(_task_loss(fallback, row.target, spec) for row in training)
        temperature = max(fitted[0][0], base_loss * 0.05, 1e-8)
        raw_weights = [exp(-min(700.0, (item[0] - fitted[0][0]) / temperature)) for item in fitted]
        weights = [value / sum(raw_weights) for value in raw_weights]
        noise_scale = max(min(item[1] for item in fitted),
                          pvariance(row.target for row in training) * 1e-6, 1e-8)
        work.spend((count + count * count) * reference_rows)
        reference_predictions = [[value for probe in references
                                  for value in _trajectory_predictions(item[2], probe, fallback)]
                                 for item in fitted]
        reference_losses = [[mean(_task_loss(action, target, spec) for action, target in zip(left, right))
                             for right in reference_predictions] for left in reference_predictions]
        prior_risk = _risk(weights, reference_losses)
        details.update(reference_risk=prior_risk, reference_rows=reference_rows,
                       noise_scale=noise_scale,
                       candidate_weights=[{"candidate_id": item[2].candidate_id, "weight": weight,
                                           "training_loss": item[0]} for item, weight in zip(fitted, weights)])
        best = None
        for probe in eligible:
            if not work.spend((count + count * count) * probe.cost):
                continue
            predictions = [_trajectory_predictions(item[2], probe, fallback) for item in fitted]
            expected = _expected_query_risk(predictions, weights, reference_losses, noise_scale)
            reduction = max(0.0, prior_risk - expected)
            value = reduction / probe.cost
            row = {"probe_id": probe.probe_id, "expected_risk_reduction": reduction,
                   "expected_posterior_risk": expected, "value_per_observation": value,
                   **coverage[probe.probe_id]}
            details["evaluated_probes"].append(row)
            rank = (value, row["coverage_per_observation"], row["distinct_control_histories"], -probe.cost)
            if best is None or rank > best:
                selected, best = probe, rank
                details.update(expected_risk_reduction=reduction, value_per_observation=value)
        details["status"] = "measured_value" if best is not None else "coverage_only"
    else:
        details["status"] = "coverage_only"
        details["candidate_weights"] = [{"candidate_id": item[2].candidate_id, "training_loss": item[0],
                                          "weight": 1.0} for item in fitted[:1]]
    budget.spend(work.spent)
    details.update(selected_probe_id=selected.probe_id, compute_cost=budget.spent - before,
                   **coverage[selected.probe_id])
    return selected, details
