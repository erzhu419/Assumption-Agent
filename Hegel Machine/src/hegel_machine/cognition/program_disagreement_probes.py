"""Synthesize interventions from the predicates of executable explanations.

The search changes public commands, using only already learned measurement
expressions. Predicate roots are proposals: the complete programs must predict
different responses on the resulting trajectory before it is returned.
"""

from dataclasses import replace
from math import isfinite
from random import Random

from .language import MissingHistory, Program
from .models import Observation, Probe
from .scale_bindings import bind_measurement_arguments, control_history_depth


def _steady(program):
    """Evaluate a predicate at a maintained command, including lagged fields."""
    if program.op == "lag":
        return Program("feature", name=program.name)
    return replace(program, children=tuple(_steady(child) for child in program.children))


def _boundaries(program):
    if program.op == "neighbor_sum":
        radius = program.children[0]
        objects = [program.children[index:index + 3]
                   for index in range(1, len(program.children), 3)]
        if radius.op == "const":
            for index, (_, position, _) in enumerate(objects):
                for _, other_position, _ in objects[index + 1:]:
                    pair = tuple(sorted((position, other_position), key=repr))
                    yield Program("abs", children=(Program("sub", children=pair),)), float(radius.value)
                    if program.name in {"right", "left"}:
                        yield Program("sub", children=pair), 0.
    elif program.op in {"lt", "ge"}:
        left, right = program.children
        if right.op == "const":
            yield left, float(right.value)
        elif left.op == "const":
            yield right, float(left.value)
        else:
            yield Program("sub", children=(left, right)), 0.
    for child in program.children:
        yield from _boundaries(child)


def _forecast(program, sequence, spend):
    history, values = [], []
    for point in sequence:
        # Match the planner's unit of work: a full prediction is one search
        # evaluation. Root solving below charges its actual expression nodes.
        if not spend(1):
            return None
        try:
            value = program.predict(Observation(point, 0., tuple(history)))
        except MissingHistory:
            value = None
        except KeyError as error:
            if error.args[0] != "__target__":
                raise
            value = None
        if value is not None and not isfinite(value):
            value = None
        values.append(value)
        history.append({**point, **({"__target__": value} if value is not None else {})})
    return values


def _experiment_sequence(context, solved, length, mode, *, history_depth=0, max_length=12):
    if mode == "maintain":
        return tuple(dict(solved) for _ in range(length))
    if mode == "pulse":
        return (dict(context), *(dict(solved) for _ in range(length - 1)))
    if mode == "cycle" and history_depth and 2 * history_depth + 3 <= max_length:
        # A full active block exposes steady response, and the equally long
        # return block exposes delayed/negative effects after deactivation.
        return (dict(context), *(dict(solved) for _ in range(history_depth + 1)),
                *(dict(context) for _ in range(history_depth + 1)))
    return None


def predicate_boundary_groups(programs, binding=None, spend=None):
    """Keep measured predicates alongside their current executable bindings."""
    groups = {}
    for program in programs:
        expanded = program.expanded()
        executable = bind_measurement_arguments(expanded, binding or {})
        if spend is not None and not spend(executable.size):
            break
        for measured, threshold in _boundaries(expanded):
            measured = _steady(measured)
            expression = bind_measurement_arguments(measured, binding or {})
            if not expression.observed_variables():
                continue
            item = groups.setdefault(expression, {
                "expression": expression, "measured_expression": measured, "thresholds": set()})
            item["thresholds"].add(threshold)
    return [{**item, "thresholds": sorted(item["thresholds"])} for item in
            sorted(groups.values(), key=lambda item: (-len(item["thresholds"]), item["expression"].size))]


def solve_joint_boundary(expression, target, point, bounds, spend, *,
                         strategy="joint_segment", control=None):
    """Solve a predicate within public bounds using charged expression work.

    Coordinate search preserves the inexpensive one-control operation. Joint
    search brackets along feasible segments, then adjusts extrema one control
    at a time when reversed or coupled calibration needs mixed directions.
    There is no Cartesian grid and no claim that failure proves infeasibility.
    """
    axes = sorted({name for name, _ in expression.observed_variables()})
    if not axes or any(name not in bounds for name in axes):
        return None, {}

    def evaluate(candidate):
        if not spend(expression.size):
            return None
        try:
            value = expression.predict(Observation(candidate, 0.))
        except MissingHistory:
            return None
        return value if isfinite(value) else None

    def result(candidate):
        changed = [name for name in axes if candidate[name] != point[name]]
        return candidate, {"controls": changed,
                           "solved_values": {name: candidate[name] for name in changed},
                           "solver_strategy": strategy}

    def segment(left, left_error, right, right_error):
        if left_error == 0.:
            return left
        if right_error == 0.:
            return right
        if left_error * right_error > 0.:
            return None
        for _ in range(48):
            middle = {name: (left[name] + right[name]) / 2 for name in point}
            value = evaluate(middle)
            if value is None:
                return None
            error = value - target
            if error == 0. or middle == left or middle == right:
                return middle
            if left_error * error <= 0.:
                right, right_error = middle, error
            else:
                left, left_error = middle, error
        return {name: (left[name] + right[name]) / 2 for name in point}

    if strategy == "coordinate":
        if control not in axes:
            return None, {}
        samples = []
        low, high = bounds[control]
        for fraction in (0., .25, .5, .75, 1.):
            candidate = {**point, control: low + (high - low) * fraction}
            value = evaluate(candidate)
            if value is None:
                return None, {}
            samples.append((candidate, value - target))
        for (left, left_error), (right, right_error) in zip(samples, samples[1:]):
            if left_error * right_error <= 0.:
                candidate = segment(left, left_error, right, right_error)
                return result(candidate) if candidate is not None else (None, {})
        return None, {}

    if strategy != "joint_segment" or len(axes) < 2:
        return None, {}
    samples = []

    def add(candidate):
        value = evaluate(candidate)
        if value is None:
            return None, None
        error = value - target
        for previous, previous_error in samples:
            if error * previous_error <= 0.:
                solved = segment(previous, previous_error, candidate, error)
                return value, solved
        samples.append((candidate, error))
        return value, None

    # Opposite corners solve sums/products near a joint limit immediately.
    # Unrelated selector/amplitude controls retain their observed context.
    for fraction in (0., 1., .5):
        candidate = {**point, **{name: bounds[name][0] + fraction *
                                 (bounds[name][1] - bounds[name][0]) for name in axes}}
        value, solved = add(candidate)
        if solved is not None:
            return result(solved)
        if value is None:
            return None, {}

    # Mixed endpoint directions accommodate reversed and nonlinear learned
    # measurements. Two bounded sweeps improve both extremal witnesses.
    for direction in (-1, 1):
        current, error = min(samples, key=lambda item: direction * item[1])
        current_value = error + target
        for _ in range(2):
            changed = False
            for name in axes:
                low, high = bounds[name]
                for coordinate in (low, (low + high) / 2, high):
                    if coordinate == current[name]:
                        continue
                    candidate = {**current, name: coordinate}
                    value, solved = add(candidate)
                    if solved is not None:
                        return result(solved)
                    if value is None:
                        return None, {}
                    if direction * value < direction * current_value:
                        current, current_value, changed = candidate, value, True
            if not changed:
                break
    return None, {}


def propose_program_disagreement_probes(command_spec, programs, budget, *, binding=None,
                                       serial=0, length=4):
    """Generate program-derived experiments within the planner's 6000 units.

    The existing coordinate operation runs first. If it cannot distinguish the
    programs, jointly move predicate controls within public bounds. Both paths
    forecast complete maintained and pulsed trajectories before proposing any
    purchase of observations.
    """
    start = budget.spent
    allowance = min(6000, budget.remaining)

    def spend(units):
        return budget.spent - start + units <= allowance and budget.spend(units)

    programs = list(programs)
    bounds = command_spec.feature_bounds
    if len(programs) < 2 or not bounds or not allowance:
        return []
    controls = set(bounds)
    required = {name for program in programs for name, _ in program.observed_variables()
                if name != "__target__"} - controls
    bindings = {name: program for name, program in (binding or {}).items()
                if name in required
                and {field for field, _ in program.observed_variables()} <= controls}
    if required - set(bindings):
        return []
    executable = [bind_measurement_arguments(program.expanded(), bindings) for program in programs]
    if any(control_history_depth(program) for program in bindings.values()):
        from .temporal_control_probes import propose_temporal_control_probes
        proposals = propose_temporal_control_probes(command_spec, executable, spend,
            serial=serial, length=length, programs=programs, binding=bindings)
        for proposal in proposals:
            proposal["compute_cost"] = budget.spent - start
        return proposals
    history_depth = max(program.history_depth for program in executable)
    length = min(command_spec.max_probe_length, max(2, length,
                 history_depth + 2))
    if length < 1:
        return []
    groups = [item for item in predicate_boundary_groups(programs, bindings, spend)
              if all(name in controls for name, _ in item["expression"].observed_variables())]
    from .program_condition_sets import compound_predicate_regions, solve_compound_region
    regions = compound_predicate_regions(programs, bindings, spend) if len(groups) > 1 else []
    # An isolated root is not useful when another narrow branch masks it.
    # Leave computation to solve the extracted conditions together.
    simple_limit = budget.spent + min(1000, max(0, allowance - (budget.spent - start)) // 4) if regions else start + allowance

    def simple_spend(units):
        return budget.spent + units <= simple_limit and spend(units)

    rng = Random(524287 + 1009 * serial)
    contexts = [{name: rng.uniform(*interval) for name, interval in bounds.items()}
                for _ in range(3)]
    results, sequences = [], set()

    def proposals(group, target, context, solved, metadata):
        if solved is None:
            return True
        for mode in (("maintain", "pulse", "cycle") if history_depth else ("maintain", "pulse")):
            sequence = _experiment_sequence(context, solved, length, mode, history_depth=history_depth,
                                            max_length=command_spec.max_probe_length)
            if sequence is None:
                continue
            key = tuple(tuple(sorted(point.items())) for point in sequence)
            if key in sequences:
                continue
            sequences.add(key)
            predictions = [_forecast(program, sequence, spend) for program in executable]
            if any(prediction is None for prediction in predictions):
                return False
            common = [row for row in zip(*predictions) if all(value is not None for value in row)]
            separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
            if separation > 0.:
                results.append({"probe": Probe(f"program_predicate:{serial}:{len(results)}", sequence),
                    "reason": "program_predicate_interval", "predicted_separation": separation,
                    "predictions": predictions, "sequence_mode": mode,
                    "boundary": {**({"expression": group["expression"].to_dict(),
                        "measured_expression": group["measured_expression"].to_dict(),
                        "thresholds": group["thresholds"], "target": target} if group else {}),
                        "context": dict(context), **metadata}, "compute_cost": budget.spent - start})
        return True

    def simple_search():
        searches = []
        for group in groups:
            thresholds = group["thresholds"]
            if len(thresholds) > 1:
                targets = [(left + right) / 2 for left, right in zip(thresholds, thresholds[1:])]
            else:
                threshold = thresholds[0]
                delta = max(1., abs(threshold)) * 1e-5
                targets = [threshold - delta, threshold, threshold + delta]
            for target in targets[:6]:
                searches.append((group, target))
                axes = sorted({name for name, _ in group["expression"].observed_variables()})
                for context in contexts:
                    for control in axes:
                        solved, metadata = solve_joint_boundary(group["expression"], target, context,
                            bounds, simple_spend, strategy="coordinate", control=control)
                        if solved is not None:
                            metadata.update(control=control, solved_value=solved[control])
                        if not proposals(group, target, context, solved, metadata) or len(results) >= 8:
                            return
                    if budget.spent >= simple_limit:
                        return
        if results:
            return
        for group, target in searches:
            for context in contexts:
                solved, metadata = solve_joint_boundary(group["expression"], target, context,
                    bounds, simple_spend)
                if not proposals(group, target, context, solved, metadata) or len(results) >= 8:
                    return
                if budget.spent >= simple_limit:
                    return

    simple_search()
    if results:
        return sorted(results, key=lambda item: -item["predicted_separation"])
    for region in regions:
        for context in contexts:
            solved, metadata = solve_compound_region(region, context, bounds, spend)
            if not proposals(None, None, context, solved, metadata) or len(results) >= 8:
                return sorted(results, key=lambda item: -item["predicted_separation"])
            if budget.spent - start >= allowance:
                return sorted(results, key=lambda item: -item["predicted_separation"])
    return sorted(results, key=lambda item: -item["predicted_separation"])
