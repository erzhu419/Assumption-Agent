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
        visit(candidate.program.expanded())
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
        add([{name: rng.uniform(*spec.feature_bounds[name]) for name in spec.feature_bounds}
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


def transition_probe(spec, candidates, observations, *, budget, remaining,
                     round_number, used=(), joint_controls=True, evidence_losses=None,
                     followup_observation_cost=0, remaining_rounds=None):
    """Resolve a learned threshold under different observed output states.

    Refine observed mature-state intervals with paired preparations before
    spending a research round on startup alone. Small budgets and predominantly
    startup uncertainty admit shorter experiments. Compare their complete
    observation cost when the controller quotes a subsequent comparison.
    With only one revision round left, collect the affordable distinct boundary
    questions together, so a startup counterexample cannot monopolize that
    last revision. ``remaining_rounds`` counts revisions after this experiment.
    ``evidence_losses`` can reuse paid
    scores over these observations; otherwise fitting comparisons are charged.
    Only candidate expressions, public control ranges and acquired observations
    define these experiments.
    """
    if remaining < 1 or not spec.feature_bounds:
        return None, {}
    levels = sorted({row.target for row in observations})
    if not levels or len(levels) > 2:
        return None, {}
    distinct = {}
    for candidate in candidates:
        distinct.setdefault(candidate.program, candidate)
    candidates = list(distinct.values())
    supported = len(candidates) < 2
    if not supported:
        measured = dict(evidence_losses or {})
        missing = [candidate for candidate in candidates if candidate.program not in measured]
        price = len(missing) * len(observations)
        allowance = budget.remaining // 2 if remaining >= 4 else budget.remaining - len(candidates)
        if price <= allowance and budget.spend(price):
            for candidate in missing:
                measured[candidate.program] = mean(point_loss(predict(candidate.program, row, levels[0]),
                    row.target, spec.risk_quantile) for row in observations)
            best_loss = min(measured[candidate.program] for candidate in candidates)
            candidates = [candidate for candidate in candidates
                          if measured[candidate.program] <= best_loss + 1e-12]
            supported = True
    coverage, coverage_details = None, {}
    if remaining >= 4 and len(levels) == 2:
        coverage, coverage_details = _joint_transition_probe(spec, candidates, observations, levels,
            budget=budget, remaining=remaining, round_number=round_number, used=used,
            joint_controls=joint_controls and len(spec.feature_bounds) > 1)
        if coverage is not None:
            startup_width = coverage_details["startup_relative_width"]
            mature_width = coverage_details["mature_relative_width"]
            boundaries = {(
                tuple(sorted(item["coordinate"].items())), item["history_depth"],
                item["previous_state"], tuple(item["observed_interval"])): item["relative_width"]
                for item in coverage_details.get("boundary_support", ())}
            coverage_return = sum(boundaries.values()) / (coverage.cost + followup_observation_cost)
            startup_return = startup_width / (1 + followup_observation_cost)
            coverage_details.update(comparison_observation_quote=followup_observation_cost,
                boundary_reduction_per_observation=coverage_return,
                startup_reduction_per_observation=startup_return,
                remaining_revision_rounds=remaining_rounds)
            if remaining_rounds == 1 and any(
                    item["previous_state"] is not None
                    for item in coverage_details.get("boundary_support", ())):
                coverage_details["design"] = "joint_boundary_revision"
                return coverage, coverage_details
            if mature_width >= startup_width or coverage_details.get("startup_control") is None:
                return coverage, coverage_details
            if followup_observation_cost and coverage_return >= startup_return:
                return coverage, coverage_details
            # The widest unresolved interval is the initial regime. Its first
            # reading and predictions have already been priced by this design.
            startup = Probe(f"transition_startup_{round_number}", (coverage.sequence[0],))
            key = _sequence_key(startup)
            if all(_sequence_key(previous)[:1] != key for previous in used):
                return startup, {**coverage_details, "design": "startup_interval_refinement",
                    "relative_width": startup_width, "history_depth": 0,
                    "boundary_support": [item for item in coverage_details.get("boundary_support", ())
                                         if item["previous_state"] is None],
                    "unresolved_boundary_width": startup_width,
                    "observation_quote": 1,
                    "boundary_targets": [item for item in coverage_details.get("boundary_targets", ())
                                         if item["phase"] == "startup"],
                    "pending_boundaries": [{**item, "scheduled": item["phase"] == "startup"}
                                           for item in coverage_details.get("pending_boundaries", ())],
                    "candidate_disagreement": coverage_details.get("startup_candidate_disagreement", 0.),
                    "selected_probe_id": startup.probe_id}
    short, details = None, {}
    if supported:
        short, details = _short_transition_probe(spec, candidates, levels[0], budget=budget,
            remaining=remaining, round_number=round_number, used=used, joint_controls=joint_controls)
    if short is not None:
        return short, details
    return coverage, coverage_details


def _boundary_description(choice, bounds):
    width, coordinate, a, b, depth, state = choice
    return {"coordinate": dict(coordinate), "phase": "startup" if state is None else "mature",
            "history_depth": depth if state is not None else 0, "previous_state": state,
            "observed_interval": [a, b], "relative_width": width,
            "free_controls": [name for name in bounds if name not in dict(coordinate)]}


def transition_outcome(design, observations):
    """Record which planned conditions the paid trajectory actually reached.

    Extreme controls are preparation attempts, not proof of the resulting
    response state. Only the returned history establishes that state. This
    records acquired values and does not make additional model predictions.
    """
    targets = []
    for target in design.get("boundary_targets", ()):
        index, depth = target["observation_index"], target["history_depth"]
        if index >= len(observations):
            continue
        row = observations[index]
        startup = target["phase"] == "startup"
        prepared = (row.history[-depth]["__target__"] >= design["state_midpoint"]
                    if depth and len(row.history) >= depth else None)
        reached = (not row.history if startup else
                   prepared is not None and prepared == target["previous_state"])
        value = sum(weight * row.features[name] for name, weight in target["coordinate"].items())
        a, b = target["observed_interval"]
        targets.append({**target, "prepared_state": prepared, "condition_reached": reached,
                        "observed_coordinate": value, "observed_response": row.target,
                        "refines_interval": reached and a < value < b})
    scheduled = {target["boundary_index"] for target in targets}
    reached = {target["boundary_index"] for target in targets if target["condition_reached"]}
    return {"targets": targets, "observed_rows": len(observations),
            "attempted_boundaries": len(scheduled), "reached_boundaries": len(reached),
            "unreached_boundary_indices": sorted(scheduled - reached)}


def _affine_controls(program):
    """Read a current-control affine expression from the shared language."""
    if program.op == "call":
        return _affine_controls(program.expanded())
    if program.op == "feature":
        return {program.name: 1.0}, 0.0
    if program.op == "const":
        return {}, program.value
    if program.op not in {"add", "sub", "mul"}:
        return None
    left, right = (_affine_controls(child) for child in program.children)
    if left is None or right is None:
        return None
    if program.op == "mul":
        if left[0] and right[0]:
            return None
        weights, offset = left if left[0] else right
        scale = right[1] if left[0] else left[1]
        return {name: value * scale for name, value in weights.items()}, offset * scale
    sign = -1.0 if program.op == "sub" else 1.0
    weights = dict(left[0])
    for name, value in right[0].items():
        weights[name] = weights.get(name, 0.0) + sign * value
    return {name: value for name, value in weights.items() if value}, left[1] + sign * right[1]


def _coordinate_splits(program, bounds):
    results = []

    def visit(node):
        if node.op in {"ge", "lt"}:
            left, right = (_affine_controls(child) for child in node.children)
            if left is not None and right is not None:
                weights = dict(left[0])
                for name, value in right[0].items():
                    weights[name] = weights.get(name, 0.0) - value
                weights = {name: value for name, value in weights.items() if value}
                if weights and all(name in bounds for name in weights):
                    scale = max(abs(value) for value in weights.values())
                    if weights[sorted(weights)[0]] < 0:
                        scale = -scale
                    coordinate = tuple(sorted((name, value / scale) for name, value in weights.items()))
                    results.append((coordinate, (right[1] - left[1]) / scale))
        for child in node.children:
            visit(child)

    visit(program.expanded())
    return results


def _coordinate_point(weights, target, bounds, anchor):
    """Move a desired control vector onto a feasible affine boundary slice."""
    point = dict(anchor)
    residual = target - sum(value * point[name] for name, value in weights.items())
    order = sorted(weights, key=lambda name: -abs(weights[name]) * (bounds[name][1] - bounds[name][0]))
    for name in order:
        low, high = bounds[name]
        value = min(high, max(low, point[name] + residual / weights[name]))
        residual -= weights[name] * (value - point[name])
        point[name] = value
    return point


def _short_transition_probe(spec, candidates, fallback, *, budget, remaining,
                            round_number, used, joint_controls):
    """Price startup and prepared-state disagreements as separate experiments.

    Candidate boundaries suffice even before observations bracket both response
    levels. Every candidate simulates its own complete history, including the
    preparation; a startup probe never inherits a preceding block's state.
    """
    distinct = []
    for candidate in candidates:
        if all(candidate.program != previous.program for previous in distinct):
            distinct.append(candidate)
    if len(distinct) < 2:
        return None, {}
    bounds = spec.feature_bounds
    coordinates = {}
    for candidate in distinct:
        for coordinate, threshold in _coordinate_splits(candidate.program, bounds):
            if joint_controls or len(coordinate) == 1:
                coordinates.setdefault(coordinate, set()).add(threshold)
    middle = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    depth = max(1, *(candidate.program.history_depth for candidate in distinct))
    length = min(remaining, spec.max_probe_length)
    allowance = budget.remaining
    started = budget.spent
    queried = [_sequence_key(probe) for probe in used]
    seen, best = set(), None
    for coordinate, thresholds in coordinates.items():
        weights = dict(coordinate)
        cuts = sorted(thresholds)
        targets = [*(.5 * (left + right) for left, right in zip(cuts, cuts[1:])), *cuts]
        for target in dict.fromkeys(targets):
            anchors = [middle]
            if joint_controls and len(bounds) > 1:
                anchors.extend({name: bounds[name][(index + side) % 2]
                                for index, name in enumerate(sorted(bounds))} for side in (0, 1))
            for anchor in anchors:
                point = _coordinate_point(weights, target, bounds, anchor)
                sequences = [[point]]
                if depth + 1 <= length:
                    for high in (False, True):
                        preparation = dict(point)
                        for name, weight in coordinate:
                            preparation[name] = bounds[name][int(high == (weight > 0))]
                        sequences.append([*([preparation] * depth), point])
                for sequence in sequences:
                    if len(sequence) > length:
                        continue
                    probe = Probe(f"transition_short_{round_number}_{len(seen)}", tuple(sequence))
                    key = _sequence_key(probe)
                    # A previously acquired prefix already includes exactly
                    # this startup and control history in the binary apparatus.
                    if key in seen or any(previous[:len(key)] == key for previous in queried):
                        continue
                    seen.add(key)
                    price = probe.cost * len(distinct)
                    if budget.spent - started + price > allowance:
                        continue
                    if not budget.spend(price):
                        continue
                    outcomes = [_trajectory_predictions(candidate, probe, fallback) for candidate in distinct]
                    disagreement = sum(pvariance(column) for column in zip(*outcomes))
                    if disagreement <= 1e-8:
                        continue
                    rank = (disagreement / probe.cost, -probe.cost)
                    if best is None or rank > best[0]:
                        best = (rank, probe, coordinate, disagreement)
    if best is None:
        return None, {}
    rank, probe, coordinate, disagreement = best
    return probe, {"policy": "joint_conditional_transition_interval" if joint_controls and len(bounds) > 1
                             else "conditional_transition_interval",
                   "design": "minimal_program_disagreement", "coordinates": [dict(coordinate)],
                   "candidate_disagreement": disagreement, "disagreement_per_observation": rank[0],
                   "startup_control": probe.sequence[0] if probe.cost == 1 else None,
                   "history_depth": 0 if probe.cost == 1 else depth,
                   "selected_probe_id": probe.probe_id}


def _joint_transition_probe(spec, candidates, observations, levels, *, budget,
                            remaining, round_number, used, joint_controls=True):
    """Test learned state coordinates while independently varying their inputs.

    Observed state-conditioned gaps locate the experiment. Free controls vary
    along that boundary, including controls omitted by an axis-only candidate;
    opposite preparations distinguish state-dependent from memoryless rules.
    """
    bounds = spec.feature_bounds
    middle = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    state_midpoint = sum(levels) / 2
    coordinates = {}
    distinct_candidates = []
    for candidate in candidates:
        if any(previous.program == candidate.program for previous in distinct_candidates):
            continue
        distinct_candidates.append(candidate)
        lags = [lag for name, lag in candidate.program.observed_variables() if name == "__target__"]
        depth = min(lags) if lags else 1
        for coordinate, threshold in _coordinate_splits(candidate.program, bounds):
            if joint_controls or len(coordinate) == 1:
                coordinates.setdefault(coordinate, set()).add((threshold, depth))
    if not coordinates:
        return None, {}
    # Leave room to compare prospective outcomes, instead of spending the
    # entire allowance on re-projecting the same observed rows.
    projection_limit = budget.remaining // 2
    choices = []
    for coordinate, thresholds in coordinates.items():
        if len(observations) > projection_limit or not budget.spend(len(observations)):
            break
        projection_limit -= len(observations)
        weights = dict(coordinate)
        lower = sum(weight * bounds[name][weight < 0] for name, weight in coordinate)
        upper = sum(weight * bounds[name][weight > 0] for name, weight in coordinate)
        span = upper - lower
        if span <= 0:
            continue
        projected = [(sum(weight * row.features[name] for name, weight in coordinate), row)
                     for row in observations]
        for threshold, depth in sorted(thresholds):
            for state in (None, False, True):
                rows = sorted(((value, row) for value, row in projected
                    if (not row.history if state is None else len(row.history) >= depth and
                        (row.history[-depth]["__target__"] >= state_midpoint) == state)),
                    key=lambda item: item[0])
                for (a, left), (b, right) in zip(rows, rows[1:]):
                    if left.target != right.target and a <= threshold <= b and b > a:
                        choice = ((b - a) / span, coordinate, a, b, depth, state)
                        if choice not in choices:
                            choices.append(choice)
    if not choices:
        return None, {}
    choices.sort(key=lambda item: -item[0])
    queried = {_sequence_key(probe) for probe in used}
    options, startup_options, seen_blocks = [], [], set()
    prediction_scale = max((levels[1] - levels[0]) ** 2, 1e-12)
    # Price one slice of each observed question before pricing another slice
    # of the same question. A smaller interval can separate different models.
    explored = list(choices)
    widest_startup = next((choice for choice in choices if choice[-1] is None), None)
    if widest_startup is not None:
        explored = [widest_startup, *(choice for choice in choices if choice != widest_startup)]
    for variant in range(3 if joint_controls else 1):
        for choice in explored:
            width, coordinate, a, b, depth, state = choice
            weights = dict(coordinate)
            if state is not None and 2 * (depth + 1) > min(remaining, spec.max_probe_length):
                continue
            anchor = {name: bounds[name][(index + variant + round_number) % 2]
                      for index, name in enumerate(sorted(bounds))}
            if variant == 2 or not joint_controls:
                anchor = middle
            point = _coordinate_point(weights, (a + b) / 2, bounds, anchor)
            novelty = min((mean(abs(point[name] - row.features[name]) / max(high - low, 1e-12)
                                  for name, (low, high) in bounds.items())
                           for row in observations), default=1.0)
            if state is None:
                if not budget.spend(len(distinct_candidates)):
                    break
                initial = [predict(candidate.program, Observation(point, 0.0), levels[0])
                           for candidate in distinct_candidates]
                startup_options.append(((pvariance(initial) / prediction_scale, width, novelty), point, choice))
                # A reset boundary is tested at the first row only. Repeating
                # its coordinate after preparations observes mature behavior,
                # and must not displace a still unresolved mature boundary.
                continue
            preparations = []
            for high in ((True, False) if round_number % 2 == 0 else (False, True)):
                preparation = dict(point)
                for name, weight in coordinate:
                    preparation[name] = bounds[name][int(high == (weight > 0))]
                preparations.append(preparation)
            block = [features for preparation in preparations
                     for features in [*([preparation] * depth), point]]
            key = tuple(tuple(sorted(features.items())) for features in block)
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            work = len(block) * len(distinct_candidates)
            if not budget.spend(work):
                break
            terminal_predictions = []
            for candidate in distinct_candidates:
                predictions = []
                for preparation in preparations:
                    probe = Probe("prospective_transition", tuple([preparation] * depth + [point]))
                    predictions.append(_trajectory_predictions(candidate, probe, levels[0])[-1])
                terminal_predictions.append(predictions)
            disagreement = sum(pvariance(values) for values in zip(*terminal_predictions)) / prediction_scale
            options.append(((disagreement, novelty, width), block, point, choice))
    if not options and not startup_options:
        return None, {}
    options.sort(key=lambda item: item[0], reverse=True)
    length = min(remaining, spec.max_probe_length)
    sequence, selected, targets = [], [], []
    startup = max(startup_options, key=lambda item: item[0]) if startup_options else None
    startup_width = max((choice[0] for choice in choices if choice[-1] is None), default=0.)
    mature_width = max((choice[0] for choice in choices if choice[-1] is not None), default=0.)
    include_startup = startup is not None and (not options or startup_width > mature_width or
                                              1 + min(len(item[1]) for item in options) <= length)
    if include_startup:
        # A startup rule is observable only at row zero. Preparing a state
        # before this point would silently test a different branch instead.
        sequence.append(startup[1])
        selected.append(startup)
        targets.append({**_boundary_description(startup[2], bounds),
                        "boundary_index": choices.index(startup[2]), "observation_index": 0})
    selected_blocks, interval_counts, included_blocks = 0, {}, set()
    for max_per_interval in (1, 2):
        for score, block, point, choice in options:
            # First refine distinct unresolved boundaries. A second slice of
            # the same boundary can then test an omitted control, provided it
            # does not consume the only opportunity to test another state.
            if max_per_interval == 2 and selected_blocks >= 3:
                break
            interval = choice[1:]
            key = tuple(tuple(sorted(features.items())) for features in block)
            if (key not in included_blocks and len(sequence) + len(block) <= length
                    and interval_counts.get(interval, 0) < max_per_interval):
                depth = choice[4]
                for offset in (depth, 2 * depth + 1):
                    targets.append({**_boundary_description(choice, bounds),
                                    "boundary_index": choices.index(choice),
                                    "observation_index": len(sequence) + offset})
                sequence.extend(block)
                selected.append((score, point, choice))
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
                included_blocks.add(key)
                selected_blocks += 1
    probe = Probe(f"{'joint_transition' if joint_controls else 'transition'}_{round_number}", tuple(sequence))
    if not probe.cost or _sequence_key(probe) in queried:
        return None, {}
    scheduled = {target["boundary_index"] for target in targets}
    return probe, {"policy": "joint_conditional_transition_interval" if joint_controls else "conditional_transition_interval",
                   "coordinates": [dict(item[2][1]) for item in selected],
                   "observed_intervals": [[item[2][2], item[2][3]] for item in selected],
                   "relative_width": max(item[2][0] for item in selected),
                   "boundary_support": [_boundary_description(item[2], bounds) for item in selected],
                   "pending_boundaries": [{**_boundary_description(choice, bounds),
                       "boundary_index": index, "scheduled": index in scheduled}
                       for index, choice in enumerate(choices)],
                   "boundary_targets": targets, "state_midpoint": state_midpoint,
                   "observation_quote": probe.cost,
                   "unresolved_boundary_width": max(item[2][0] for item in selected),
                   "startup_relative_width": startup_width,
                   "mature_relative_width": mature_width,
                   "candidate_disagreement": max(item[0][0] for item in selected),
                   "startup_candidate_disagreement": startup[0][0] if startup is not None else 0.,
                   "startup_control": startup[1] if include_startup else None,
                   "history_depth": max(item[2][4] for item in selected),
                   "selected_probe_id": probe.probe_id}


def choose_probe(probes: list[Probe], candidates: list[Candidate], fallback: float,
                 budget: SearchBudget, *, observations=()) -> tuple[Probe | None, float]:
    best_probe, best_value = None, -1.0
    for probe in probes:
        if not probe.cost or not candidates:
            continue
        scenarios = _measurement_scenarios(_sample_rows(observations, 16), probe.sequence[0])
        cost = probe.cost * len(candidates) * len(scenarios)
        if not budget.spend(cost):
            break
        # Average disagreement can reward a useful split even when some pair
        # of Boolean candidates agrees. Maximin pair distance cannot do this.
        values = []
        for scenario in scenarios:
            predictions = [_trajectory_predictions(candidate, probe, fallback, scenario)
                           for candidate in candidates]
            values.append(sum(pvariance(column) for column in zip(*predictions)) / probe.cost)
        value = mean(values)
        if value > best_value:
            best_probe, best_value = probe, value
    return best_probe, max(0.0, best_value)


def _sequence_key(probe: Probe):
    return tuple(tuple(sorted(point.items())) for point in probe.sequence)


def _sample_rows(rows, maximum):
    if len(rows) <= maximum:
        return list(rows)
    return [rows[round(index * (len(rows) - 1) / (maximum - 1))] for index in range(maximum)]


def _measurement_scenarios(training, control_names):
    if not training or not set(training[0].features) - set(control_names):
        return [()]
    # Empirical possible sensor paths, each evaluated under the proposed controls.
    return [training[offset:] + training[:offset]
            for offset in sorted({0, len(training) // 3, 2 * len(training) // 3})]


def _control_ranges(spec, observations):
    ranges = dict(spec.feature_bounds)
    if ranges:
        return ranges
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
    for name in ranges:
        order = list(range(count))
        rng.shuffle(order)
        low, high = ranges[name]
        columns[name] = [low + (high - low) * (index + 0.5) / count for index in order]
    points = [{name: columns[name][index] for name in ranges} for index in range(count)]
    return [Probe(f"reference_{index}", tuple(points[index * length:(index + 1) * length]))
            for index in range(2)]


def _trajectory_predictions(candidate, probe, fallback, exogenous=()):
    history, values = [], []
    for index, point in enumerate(probe.sequence):
        features = {**(exogenous[index % len(exogenous)].features if exogenous else {}), **point}
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
    variances = (noise_scale if isinstance(noise_scale, list) else
                 [[noise_scale] * len(prediction) for prediction in predictions])
    likelihood = [[exp(-min(700.0, mean((a - b) ** 2 / (2 * variance)
                         for a, b, variance in zip(outcome, prediction, distribution))))
                   for prediction, distribution in zip(predictions, variances)] for outcome in outcomes]
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
                          spec: TaskSpec, queried_probes=(), max_compute=None, simulator=None) -> tuple[Probe | None, dict]:
    """Select expected reference-task risk reduction per acquired observation.

    This is a bounded finite-model approximation, using at most four weighted
    candidates and public reference controls. Coverage of controls and histories
    resolves ties and supports experiments before a useful competitor exists.
    At most one fifth of remaining compute, capped at 1200 units, is spent here.
    Units include prediction evaluations and cached response/loss comparisons.
    """
    before = budget.spent
    allowance = min(1200, budget.remaining // 5) if max_compute is None else min(max_compute, budget.remaining)
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
    controls = tuple(ranges)
    if all(all(name in point for name in controls) for row in training
           for point in (row.features, *row.history[-2:])):
        observed_descriptors = [_control_descriptor(row.features, row.history, controls, ranges)
                                for row in training]
    else:
        # Aggregate instruments may omit individual controls from their readings.
        # Their actual paid control sequences still determine experiment coverage;
        # predictor inputs continue to contain only the acquired measurements.
        observed_descriptors = [_control_descriptor(point, probe.sequence[:index], controls, ranges)
                                for probe in queried_probes for index, point in enumerate(probe.sequence)]
    coverage = {probe.probe_id: _coverage(probe, observed_descriptors, controls, ranges)
                for probe in eligible}
    sensor_names = set(spec.feature_names) - set(controls)
    scenarios = _measurement_scenarios(training, controls)
    if sensor_names and training:
        # Uncontrolled measurements are uncertain outcomes of an experiment.
        # Replaying several acquired sensor paths keeps them out of its controls.
        details["sensor_scenarios"] = {"measurements": sorted(sensor_names),
                                        "paths": len(scenarios), "source": "acquired_readings"}
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
    simulation_work = 2 if simulator is not None and simulator.current else 0
    while count > 1 and len(scenarios) * (count + count * count + simulation_work * count) * (reference_rows + query_rows) > work.remaining:
        count -= 1
    if count > 1:
        fitted = fitted[:count]
        base_loss = mean(_task_loss(fallback, row.target, spec) for row in training)
        temperature = max(fitted[0][0], base_loss * 0.05, 1e-8)
        raw_weights = [exp(-min(700.0, (item[0] - fitted[0][0]) / temperature)) for item in fitted]
        weights = [value / sum(raw_weights) for value in raw_weights]
        noise_scale = max(min(item[1] for item in fitted),
                          pvariance(row.target for row in training) * 1e-6, 1e-8)
        work.spend(len(scenarios) * (count + count * count) * reference_rows)
        reference_paths = [[(probe, _trajectory_predictions(item[2], probe, fallback, scenario))
                            for scenario in scenarios for probe in references] for item in fitted]
        reference_predictions = [[value for _, values in paths for value in values] for paths in reference_paths]
        reference_targets = reference_predictions
        if simulation_work:
            reference_targets = [[value for probe, values in paths
                for value in simulator.adjusted(probe, values,
                    baseline_variance=noise_scale, budget=work)["mean"]] for paths in reference_paths]
            # The raw actions remain the task decisions. Only their simulated
            # observation outcomes are revised by the learned response channel.
            details["simulator"] = {key: simulator.current[key]
                                    for key in ("model_id", "correction", "variance")}
        reference_losses = [[mean(_task_loss(action, target, spec) for action, target in zip(left, right))
                             for right in reference_targets] for left in reference_predictions]
        prior_risk = _risk(weights, reference_losses)
        details.update(reference_risk=prior_risk, reference_rows=reference_rows,
                       noise_scale=noise_scale,
                       candidate_weights=[{"candidate_id": item[2].candidate_id, "weight": weight,
                                           "training_loss": item[0]} for item, weight in zip(fitted, weights)])
        best = None
        for probe in eligible:
            if work.remaining < len(scenarios) * (count + count * count + simulation_work * count) * probe.cost:
                continue
            work.spend(len(scenarios) * (count + count * count) * probe.cost)
            risks = []
            for scenario in scenarios:
                predictions = [_trajectory_predictions(item[2], probe, fallback, scenario) for item in fitted]
                conditional_noise = noise_scale
                if simulation_work:
                    distributions = [simulator.adjusted(probe, values, baseline_variance=noise_scale, budget=work)
                                     for values in predictions]
                    predictions = [item["mean"] for item in distributions]
                    conditional_noise = [item["variance"] for item in distributions]
                risks.append(_expected_query_risk(predictions, weights, reference_losses, conditional_noise))
            expected = mean(risks)
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
