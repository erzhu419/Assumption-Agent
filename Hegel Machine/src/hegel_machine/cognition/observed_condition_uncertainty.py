"""Keep the prediction consequences of an unmeasured condition interval.

Threshold endpoints are executable hypotheses about the original response.
An empirical density estimates how often a future reading could separate them;
this is a research estimate, not a certificate about an unknown instrument law.
"""

from dataclasses import replace

from .language import MissingHistory, Program
from .models import Observation


def _replace_threshold(program, field, old, value):
    if (program.op in ("ge", "lt") and program.children[0].op == "feature"
            and program.children[0].name == field
            and program.children[1].op == "const"
            and program.children[1].value == old):
        return replace(program, children=(program.children[0], Program("const", value=value)))
    return replace(program, children=tuple(_replace_threshold(child, field, old, value)
                                           for child in program.children))


def _gates(program, windows=()):
    if program.op == "window_sum":
        windows = (*windows, program.lag)
    if (program.op in ("ge", "lt") and program.children[0].op == "feature"
            and program.children[1].op == "const"):
        yield program.children[0].name, program.children[1].value, max(windows, default=1)
    for child in program.children:
        yield from _gates(child, windows)


def _window_widths(program):
    if program.op == "window_sum":
        yield program.lag
    for child in program.children:
        yield from _window_widths(child)


def _selection_steps(plan):
    for step in reversed((plan or {}).get("selection_steps", [])):
        yield step
        yield from _selection_steps({"selection_steps": step.get("inner_selection_steps", [])})


def _density(rows, field, left, right):
    values = sorted(row.features[field] for row in rows if field in row.features)
    below = [value for value in values if value <= left][-8:]
    above = [value for value in values if value >= right][:8]
    if not below or not above or above[-1] <= below[0]:
        return None
    span = above[-1] - below[0]
    # Local order-statistic spacing estimates the mass of the unobserved gap.
    # It stays unchanged when the meter unit or origin changes.
    density = max(1, len(below) + len(above) - 1) / ((len(values) + 1) * span)
    return {"probability": min(1., max(0., (right - left) * density)),
            "density": density, "reading_rows": len(values),
            "support_interval": [below[0], above[-1]]}


def _predict(program, row, budget):
    if not budget.spend(1):
        return None
    try:
        return program.predict(row)
    except (KeyError, MissingHistory):
        return None


def _loss(program, rows, budget):
    errors = []
    for row in rows:
        predicted = _predict(program, row, budget)
        if predicted is None:
            return None
        errors.append((predicted - row.target) ** 2)
    return sum(errors) / len(errors) if errors else None


def _at_reading(row, field, value, position):
    if position < 0:
        return replace(row, features={**row.features, field: value})
    history = list(row.history)
    history[position] = {**history[position], field: value}
    return replace(row, history=tuple(history))


def _ambiguity(program, rows, gate, budget):
    field, threshold, width = gate["condition_field"], gate["threshold"], gate["gate_width"]
    expected = []
    for left, right in gate["threshold_intervals"]:
        density = _density(rows, field, left, right)
        if density is None:
            return None
        margin = (right - left) * 1e-6
        low = _replace_threshold(program, field, threshold, left + margin)
        high = _replace_threshold(program, field, threshold, right - margin)
        contributions = []
        # Reading density uses every paid row. Response sensitivity uses an
        # evenly spaced sample of the actual control/history backgrounds.
        backgrounds = [rows[index * (len(rows) - 1) // min(23, len(rows) - 1)]
                       for index in range(min(24, len(rows)))] if len(rows) > 1 else rows
        for row in backgrounds:
            if field not in row.features:
                continue
            positions = [-1, *range(max(0, len(row.history) - width + 1), len(row.history))]
            differences = []
            for position in positions:
                altered = _at_reading(row, field, (left + right) / 2., position)
                one, other = _predict(low, altered, budget), _predict(high, altered, budget)
                if one is None or other is None:
                    return None
                differences.append(one - other)
            probability = density["probability"]
            # For independent future gap events, each original-target response
            # contribution has a probability p; simultaneous events retain the
            # signed cross terms rather than a global worst-case envelope.
            contributions.append(probability * (1. - probability)
                * sum(value * value for value in differences)
                + probability ** 2 * sum(differences) ** 2)
        expected.append({"threshold_interval": [left, right], **density,
            "expected_prediction_ambiguity": sum(contributions) / max(1, len(contributions)),
            "endpoint_programs": [low.to_dict(), high.to_dict()]})
    return expected


def _narrow(program, rows, gate, budget, tolerance):
    field, threshold = gate["condition_field"], gate["threshold"]
    values = sorted({features[field] for row in rows
                     for features in (row.features, *row.history) if field in features})
    intervals, trials = [], []
    for left, right in gate["threshold_intervals"]:
        boundaries = [left, *(value for value in values if left < value < right), right]
        for lower, upper in zip(boundaries, boundaries[1:]):
            hypothesis = _replace_threshold(program, field, threshold, (lower + upper) / 2.)
            measured = _loss(hypothesis, rows, budget)
            if measured is None:
                return None
            trials.append({"threshold_interval": [lower, upper], "original_target_loss": measured})
            if measured <= tolerance:
                if intervals and intervals[-1][1] == lower:
                    intervals[-1][1] = upper
                else:
                    intervals.append([lower, upper])
    return intervals, trials


def assess_observed_condition_uncertainty(spec, rows, selected, budget, *,
                                         previous=None, tolerance=1e-7):
    """Return a serialisable condition question, or None for a smooth response.

    selected is the normal assessed candidate entry. Endpoint variants retain
    its complete original response, including windowed intercepts and branches.
    New target evidence may narrow a cut interval or contradict the fixed
    response. No purchased reading is treated as a controllable input.
    """
    if selected is None or not rows:
        return None
    program = selected["candidate"].program
    gates = list(dict.fromkeys(_gates(program)))
    gates = [item for item in gates if item[0] not in spec.feature_bounds]
    if not gates:
        return None
    start = budget.spent
    original_loss = _loss(program, rows, budget)
    response_width = max(_window_widths(program), default=1)
    supported = []
    for field, threshold, width in gates:
        step = next((step for step in _selection_steps(selected.get("research_program"))
            if step.get("threshold_interval") and step.get("gate_coordinate", {}).get("op") == "feature"
            and step["gate_coordinate"].get("name") == field
            and (step["threshold_interval"][0] <= threshold <= step["threshold_interval"][1]
                 or step.get("threshold") == threshold)), None)
        if step is None:
            continue
        interval = list(step["threshold_interval"])
        placement = "inside_window" if width > 1 else step.get("placement", "outside_window")
        gate = {"condition_field": field, "control": field, "threshold": threshold,
                "width": response_width, "gate_width": width, "placement": placement,
                "width_candidates": [response_width], "placement_candidates": [placement],
                "threshold_intervals": [interval], "threshold_interval": interval}
        old = next((item for item in (previous or {}).get("conditions", [])
                    if item["condition_field"] == field and item["threshold"] == threshold
                    and (previous or {}).get("program") == program.to_dict()), None)
        if old and old.get("threshold_intervals"):
            gate["threshold_intervals"] = [list(item) for item in old["threshold_intervals"]]
        narrowed = _narrow(program, rows, gate, budget, tolerance)
        if narrowed is None:
            gate["status"] = "unresolved"
            gate["expected_prediction_ambiguity"] = None
        elif not narrowed[0]:
            gate.update(status="contradicted", threshold_intervals=[],
                        original_target_trials=narrowed[1], expected_prediction_ambiguity=None)
        else:
            gate["threshold_intervals"], gate["original_target_trials"] = narrowed
            conditional_loss = min(item["original_target_loss"] for item in narrowed[1])
            gate["conditional_loss"] = conditional_loss
            gate["gain"] = max(0., original_loss - conditional_loss) if original_loss is not None else 0.
            gate["needs_threshold_revision"] = original_loss is not None and original_loss > tolerance
            gate["threshold_interval"] = list(gate["threshold_intervals"][0])
            ambiguity = _ambiguity(program, rows, gate, budget)
            gate["prediction_trials"] = ambiguity or []
            gate["endpoint_programs"] = [entry for trial in (ambiguity or [])
                                          for entry in trial["endpoint_programs"]]
            risk = max((item["expected_prediction_ambiguity"] for item in ambiguity), default=0.) if ambiguity is not None else None
            gate["expected_prediction_ambiguity"] = risk
            gate["status"] = ("supported" if risk is not None and risk <= tolerance
                              and not gate["needs_threshold_revision"] else "unresolved")
        supported.append(gate)
    if not supported:
        return None
    risks = [item["expected_prediction_ambiguity"] for item in supported
             if item["expected_prediction_ambiguity"] is not None]
    status = ("contradicted" if any(item["status"] == "contradicted" for item in supported)
              else "supported" if all(item["status"] == "supported" for item in supported)
              else "unresolved")
    return {"kind": "observed_condition_uncertainty", "status": status,
            "program": program.to_dict(), "conditions": supported,
            "fit_rows": len(rows), "expected_prediction_ambiguity": max(risks, default=None),
            "original_target_loss": original_loss,
            "basis": "paid_local_density_and_original_response_endpoint_hypotheses",
            "compute_cost": budget.spent - start}


def update_observed_condition_uncertainty(state, rows, budget, *, tolerance=1e-7):
    """Revise saved evidence against additional real original-target rows."""
    program = Program.from_dict(state["program"])
    from .models import Candidate, TaskSpec
    steps = [{"gate_coordinate": {"op": "feature", "name": item["condition_field"]},
              "threshold_interval": item["threshold_interval"], "threshold": item["threshold"]}
             for item in state["conditions"] if item.get("threshold_intervals")]
    if not steps:
        return {**state, "status": "contradicted", "fit_rows": len(rows), "compute_cost": 0}
    selected = {"candidate": Candidate("condition_evidence", program, "research_program"),
                "research_program": {"selection_steps": steps}}
    spec = TaskSpec("condition_evidence", "", tuple(sorted(program.required_features())), (), ())
    return assess_observed_condition_uncertainty(spec, rows, selected, budget,
                                                previous=state, tolerance=tolerance)
