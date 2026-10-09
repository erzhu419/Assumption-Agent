"""Revise shared window lengths using the purchased response trajectories."""

from dataclasses import replace

from .language import Program
from .relation_uncertainty import _bind
from .research_programs import _Fitter


def _layers(program):
    """Group equal-width siblings, keeping distinct nested time layers apart."""
    found = {}

    def visit(node, path=(), reach=0):
        if node.op == "window_sum":
            key = path, node.lag
            group = found.setdefault(key, {"path": path, "width": node.lag,
                "occurrences": 0, "history_overhead": 0})
            group["occurrences"] += 1
            group["history_overhead"] = max(group["history_overhead"],
                                             reach + node.children[0].history_depth)
        for index, child in enumerate(node.children):
            temporal = node.op in {"window_sum", "event_delta", "accumulate"}
            nested = (*path, (node.op, node.lag, node.value, index)) if temporal else path
            extra = (node.lag - 1 if node.op == "window_sum" else
                     node.lag if node.op == "event_delta" else 0)
            visit(child, nested, reach + extra)

    visit(program)
    return list(found.values())


def _revise_layer(program, layer, width, path=()):
    children = []
    for index, child in enumerate(program.children):
        temporal = program.op in {"window_sum", "event_delta", "accumulate"}
        nested = (*path, (program.op, program.lag, program.value, index)) if temporal else path
        children.append(_revise_layer(child, layer, width, nested))
    changed = program.op == "window_sum" and path == layer["path"] and program.lag == layer["width"]
    return replace(program, children=tuple(children), lag=width if changed else program.lag)


def _describe_path(path):
    return [{"operator": op, "lag": lag, "value": value, "child": child}
            for op, lag, value, child in path]


def fit_temporal_scales(program, training, budget, *, values, quantile, reference_loss):
    """Nominate one window revision, retaining its other layers and symbols.

    Doubled offsets cover both sides of the old scale without assuming a
    unimodal loss. Integer refinement fills the interval around the best paid
    trial. A width beyond the observed history is never proposed. The caller
    retains the complete old response and must independently test any proposal.
    """
    from .mechanism_revision import _additive_terms

    if not training or reference_loss <= 1e-12:
        return None
    history = max(len(row.history) for row in training)
    fitter, trials, best = _Fitter(training, budget, quantile), [], None
    for layer in _layers(program):
        upper = history + 1 - layer["history_overhead"]
        if upper < 2:
            continue
        origin, measured = layer["width"], {}

        def evaluate(width):
            if width == origin or width in measured or not 1 <= width <= upper:
                return
            revised = _revise_layer(program, layer, width)
            symbolic_terms = tuple(dict.fromkeys(term for term in _additive_terms(revised)
                                                 if term.op != "const"))
            numerical = tuple(_bind(term, values) for term in symbolic_terms)
            symbolic = dict(zip(numerical, symbolic_terms))
            if len(symbolic) != len(set(symbolic_terms)):
                return
            before = budget.spent
            fit = fitter.fit(numerical)
            if fit is None:
                return
            record = {"temporal_path": _describe_path(layer["path"]),
                "width_before": origin, "width": width, "loss": fit["loss"],
                "compute_cost": budget.spent - before}
            trials.append(record)
            measured[width] = {"fit": fit, "symbolic": symbolic,
                "temporal_revisions": [{"operator": "window_sum",
                    "temporal_path": _describe_path(layer["path"]),
                    "width_before": origin, "width_after": width,
                    "shared_occurrences": layer["occurrences"], "history_bound": upper}]}

        # Begin nearby but do not stop at a local minimum: doubling exposes
        # wider and shorter scales before refinement spends the remaining work.
        step = 1
        while step < max(origin, upper - origin + 1):
            for width in (origin + step, origin - step):
                evaluate(width)
            if measured and min(item["fit"]["loss"] for item in measured.values()) <= 1e-12:
                break
            if budget.remaining < 3 * len(training):
                break
            step *= 2
        if measured:
            winner = min(measured, key=lambda width: measured[width]["fit"]["loss"])
            tested = sorted({origin, *measured})
            left = max((width for width in tested if width < winner), default=1)
            right = min((width for width in tested if width > winner), default=upper)
            refinement = sorted((width for width in range(left, right + 1)
                                 if width != origin and width not in measured),
                                key=lambda width: (abs(width - winner), width))
            if measured[winner]["fit"]["loss"] > 1e-12:
                for width in refinement:
                    evaluate(width)
                    if min(item["fit"]["loss"] for item in measured.values()) <= 1e-12:
                        break
                    if budget.remaining < 3 * len(training):
                        break
            candidate = min(measured.values(), key=lambda item: item["fit"]["loss"])
            if candidate["fit"]["loss"] < reference_loss and (
                    best is None or candidate["fit"]["loss"] < best["fit"]["loss"]):
                best = candidate
        if best is not None and best["fit"]["loss"] <= 1e-12:
            break
    return {**best, "trials": trials} if best is not None else None
