"""Solve simultaneous executable branch conditions in public command space."""

from math import isfinite

from .language import MissingHistory, Program
from .models import Observation
from .scale_bindings import bind_measurement_arguments
from .synthesis import _solve


def _normalize_path(path):
    bounds = {}
    for expression, threshold, lower, inclusive in path:
        key = expression, lower
        previous = bounds.get(key)
        if previous is None or (threshold > previous[0] if lower else threshold < previous[0]):
            bounds[key] = threshold, inclusive
        elif threshold == previous[0]:
            bounds[key] = threshold, inclusive and previous[1]
    for expression, _ in bounds:
        low, high = bounds.get((expression, True)), bounds.get((expression, False))
        if low and high and (low[0] > high[0] or
                            low[0] == high[0] and not (low[1] and high[1])):
            return None
    return tuple(sorted(((expression, threshold, lower, inclusive)
                         for (expression, lower), (threshold, inclusive) in bounds.items()), key=repr))


def _collect_paths(paths, complete=True):
    """Limit distinct conditional regions, retaining the unconditional term."""
    result, seen, conditional_count = [], set(), 0
    for path in paths:
        normalized = _normalize_path(path)
        if normalized is None or normalized in seen:
            continue
        if normalized:
            if conditional_count == 24:
                return result, False
            conditional_count += 1
        seen.add(normalized)
        result.append(normalized)
    return result, complete


def _cross(left, right):
    a, a_complete = left
    b, b_complete = right
    return _collect_paths((x + y for x in a for y in b), a_complete and b_complete)


def _union(parts):
    return _collect_paths((path for paths, _ in parts for path in paths),
                          all(complete for _, complete in parts))


def _condition_paths(node, truth):
    if node.op == "const":
        return ([()] if bool(node.value) == truth else []), True
    if node.op == "not":
        return _condition_paths(node.children[0], not truth)
    if node.op in {"and", "or", "mul"}:
        left, right = node.children
        together = truth if node.op in {"and", "mul"} else not truth
        a, b = _condition_paths(left, truth), _condition_paths(right, truth)
        return _cross(a, b) if together else _union((a, b))
    if node.op in {"lt", "ge"}:
        left, right = node.children
        lower = (node.op == "ge") == truth
        inclusive = lower
        if right.op == "const":
            expression, threshold = left, float(right.value)
        elif left.op == "const":
            expression, threshold, lower = right, float(left.value), not lower
        else:
            expression, threshold = Program("sub", children=(left, right)), 0.
        return [((expression, threshold, lower, inclusive),)], True
    # Other numeric selectors have no inferred inequality; later complete
    # program forecasts decide whether the proposed path actually activates.
    return [()], True


def _active_paths(node):
    """Return unique paths and whether their actual traversal was complete."""
    if node.op == "const":
        return ([()] if node.value else []), True
    if node.op == "if":
        condition, yes, no = node.children
        return _union((_cross(_condition_paths(condition, True), _active_paths(yes)),
                       _cross(_condition_paths(condition, False), _active_paths(no))))
    if node.op in {"lt", "ge", "and", "or", "not"}:
        return _condition_paths(node, True)
    if node.op in {"mul", "min"}:
        return _cross(_active_paths(node.children[0]), _active_paths(node.children[1]))
    if node.op in {"feature", "lag", "arg", "available"}:
        return [()], True
    if not node.children:
        return [()], True
    return _union(tuple(_active_paths(child) for child in node.children))


def compound_predicate_regions(programs, binding=None, spend=None):
    """Derive bounded symbolic cells from actual nonzero executable paths.

    Adjacent current thresholds delimit candidate cells. Branch conjunctions
    retain shared variable identity and disjunctions retain separate paths.
    The bounded path search is incomplete; no returned cell claims reachability.
    """
    from .program_disagreement_probes import _steady, predicate_boundary_groups

    programs = list(programs)
    groups = predicate_boundary_groups(programs, binding=binding)
    thresholds = {item["expression"]: item["thresholds"] for item in groups}
    regions, seen = [], set()
    for program in programs:
        expanded = program.expanded()
        if spend is not None and not spend(expanded.size):
            break
        for path in _active_paths(expanded)[0]:
            constraints = {}
            for measured, threshold, is_lower, inclusive in path:
                measured = _steady(measured)
                expression = bind_measurement_arguments(measured, binding or {})
                if not expression.observed_variables():
                    continue
                item = constraints.setdefault(expression, {
                    "expression": expression, "measured_expression": measured,
                    "thresholds": thresholds.get(expression, [threshold]),
                    "lower": None, "upper": None,
                    "lower_inclusive": False, "upper_inclusive": False})
                side = "lower" if is_lower else "upper"
                previous = item[side]
                if previous is None or (threshold > previous if is_lower else threshold < previous):
                    item[side], item[f"{side}_inclusive"] = threshold, inclusive
                elif threshold == previous:
                    item[f"{side}_inclusive"] = item[f"{side}_inclusive"] and inclusive
            if len(constraints) < 2 or any(item["lower"] is not None and item["upper"] is not None
                    and (item["lower"] > item["upper"] or (item["lower"] == item["upper"] and
                         not (item["lower_inclusive"] and item["upper_inclusive"])))
                    for item in constraints.values()):
                continue
            candidates = [[]]
            for item in constraints.values():
                low, high = item["lower"], item["upper"]
                cuts = [value for value in item["thresholds"]
                        if (low is None or value > low) and (high is None or value < high)]
                endpoints = [low, *cuts, high]
                cells = []
                for left, right in zip(endpoints, endpoints[1:]):
                    if left is None and right is None:
                        continue
                    target = ((left + right) / 2 if left is not None and right is not None else
                              right - max(1., abs(right)) * 1e-5 if left is None else
                              left + max(1., abs(left)) * 1e-5)
                    cells.append({**item, "lower": left, "upper": right, "target": target,
                        "lower_inclusive": item["lower_inclusive"] if left == low else False,
                        "upper_inclusive": item["upper_inclusive"] if right == high else False})
                cells.sort(key=lambda cell: (cell["lower"] is None or cell["upper"] is None,
                    cell["upper"] - cell["lower"] if cell["lower"] is not None and
                    cell["upper"] is not None else 0.))
                candidates = [(candidate + [cell]) for candidate in candidates for cell in cells][:12]
            for conditions in candidates:
                key = tuple((item["expression"], item["lower"], item["upper"],
                             item["lower_inclusive"], item["upper_inclusive"]) for item in conditions)
                if key in seen:
                    continue
                if spend is not None and not spend(len(conditions)):
                    return regions
                seen.add(key)
                regions.append({"constraints": conditions, "solver_strategy": "compound_conditions"})
                if len(regions) >= 12:
                    return regions
    return regions


def solve_compound_region(region, point, bounds, spend):
    """Fit all current predicate targets together and check every inequality.

    Finite differences use learned measurement expressions, never lab calls.
    A bounded least-squares update handles controls shared by multiple gates;
    each step remains in public bounds, and success checks all conditions.
    """
    constraints = region["constraints"]
    expressions = [item["expression"] for item in constraints]
    axes = sorted({name for expression in expressions for name, _ in expression.observed_variables()})
    axes = [name for name in axes if name in bounds and bounds[name][1] > bounds[name][0]]
    required = {name for expression in expressions for name, _ in expression.observed_variables()}
    if len(constraints) < 2 or not axes or required - set(bounds):
        return None, {}
    cost = sum(expression.size for expression in expressions)
    targets = [item["target"] for item in constraints]

    def evaluate(candidate):
        if not spend(cost):
            return None
        try:
            values = [expression.predict(Observation(candidate, 0.)) for expression in expressions]
        except MissingHistory:
            return None
        return values if all(isfinite(value) for value in values) else None

    def satisfied(values):
        return all((item["lower"] is None or value > item["lower"] or
                    (item["lower_inclusive"] and value == item["lower"])) and
                   (item["upper"] is None or value < item["upper"] or
                    (item["upper_inclusive"] and value == item["upper"]))
                   for item, value in zip(constraints, values))

    def result(candidate, values):
        changed = [name for name in axes if candidate[name] != point[name]]
        return candidate, {"solver_strategy": "compound_conditions", "controls": changed,
            "solved_values": {name: candidate[name] for name in changed},
            "condition_values": values,
            "constraints": [{**item, "expression": item["expression"].to_dict(),
                "measured_expression": item["measured_expression"].to_dict()} for item in constraints]}

    initial = dict(point)
    for seed in (initial, {**initial, **{name: sum(bounds[name]) / 2 for name in axes}}):
        current = seed
        values = evaluate(current)
        if values is None:
            return None, {}
        for _ in range(12):
            if satisfied(values):
                return result(current, values)
            jacobian = [[] for _ in constraints]
            for name in axes:
                low, high = bounds[name]
                fraction = (current[name] - low) / (high - low)
                step = 1e-5 if fraction < 1. - 1e-5 else -1e-5
                changed = evaluate({**current, name: current[name] + step * (high - low)})
                if changed is None:
                    return None, {}
                for row, value, other in zip(jacobian, values, changed):
                    row.append((other - value) / step)
            count = len(axes)
            if not spend(count ** 3 + len(constraints) * count ** 2):
                return None, {}
            gram = [[sum(row[a] * row[b] for row in jacobian) for b in range(count)]
                    for a in range(count)]
            regularization = max(1., *(gram[index][index] for index in range(count))) * 1e-12
            for index in range(count):
                gram[index][index] += regularization
            rhs = [sum(row[index] * (target - value)
                       for row, target, value in zip(jacobian, targets, values)) for index in range(count)]
            direction = _solve(gram, rhs)
            loss = sum((value - target) ** 2 for value, target in zip(values, targets))
            improved = False
            for scale in (1., .5, .25, .125, .0625, .03125):
                candidate = {**current, **{name: max(bounds[name][0], min(bounds[name][1],
                    current[name] + scale * delta * (bounds[name][1] - bounds[name][0])))
                    for name, delta in zip(axes, direction)}}
                updated = evaluate(candidate)
                if updated is None:
                    return None, {}
                if satisfied(updated):
                    return result(candidate, updated)
                new_loss = sum((value - target) ** 2 for value, target in zip(updated, targets))
                if new_loss < loss:
                    current, values, improved = candidate, updated, True
                    break
            if not improved:
                break
    return None, {}
