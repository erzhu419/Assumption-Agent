"""Paid contexts and crossed controls for two unknown temporal conditions."""

from .conditional_experiments import _sequence
from .conditional_coordinates import _masked
from .continuous_state import _sum_terms
from .language import Program
from .models import Observation, Probe
from .research_programs import _Fitter, _rank


def varying_condition_delta_terms(control, background, threshold, width, placement):
    """Piecewise quadratic responses minus the actual constant reference input."""
    feature = Program("feature", name=control)
    basis = (Program("const", value=1.), feature, Program("mul", children=(feature, feature)))
    condition = Program("ge", children=(feature, Program("const", value=threshold)))
    reference_active = background >= threshold
    terms = []
    for power, term in enumerate(basis):
        for side in (True, False):
            changed = _masked(term, condition, side, width, placement)
            if side == reference_active:
                reference = Program("const", value=background ** power)
                if width > 1:
                    reference = Program("window_sum", lag=width, children=(reference,))
                changed = Program("sub", children=(changed, reference))
            terms.append(changed)
    return terms


def propose_composed_condition_experiments(spec, paid_rows, programs, budget, *,
                                           serial=0, observation_budget=120):
    """Expose each control at both public endpoint contexts before naming gates.

    An AND can hide both controls at the low context; an OR can hide them at the
    high context. Buying the two matched contexts exposes both possibilities
    while retaining a common history and the other controls in each experiment.
    """
    controls = tuple(name for name in spec.feature_names if name in spec.feature_bounds)
    length = min(12, spec.max_probe_length)
    pack_cost = (len(controls) + 1) * length
    if len(controls) < 2 or length < 8 or observation_budget < 2 * pack_cost:
        return []
    packs = []
    for context in range(2):
        start = budget.spent
        background = {name: spec.feature_bounds[name][context] for name in controls}
        if not budget.spend(length * (len(controls) + 1)):
            return []
        reference = Probe(f"composed:{serial}:{context}:reference",
                          tuple(dict(background) for _ in range(length)))
        interventions = []
        for control in controls:
            sequence, levels, prefix = _sequence(background, control,
                spec.feature_bounds[control], length, "sweep")
            interventions.append({"probe": Probe(f"composed:{serial}:{context}:{control}", sequence),
                "control": control, "mode": "sweep", "levels": levels,
                "prefix_length": prefix, "background": dict(background)})
        packs.append({"reference": reference, "interventions": interventions,
            "background": background, "context": "low" if context == 0 else "high",
            "observation_cost": pack_cost, "measurement_cost": 0,
            "compute_cost": budget.spent - start})
    return packs


def _supported_gates(spec, evidence):
    supported = {}
    for item in evidence:
        control = item.get("control")
        if control in spec.feature_bounds and item.get("gain", 0.) > 1e-10:
            supported.setdefault(control, []).append(item)
    return [(name, supported[name]) for name in spec.feature_names if name in supported]


def propose_composed_cross_experiment(spec, evidence, budget, *, serial=0,
                                      observation_budget=48):
    """Cross two measured gates while keeping response controls and prefix fixed."""
    gates = _supported_gates(spec, evidence)
    length = min(12, spec.max_probe_length)
    if len(gates) != 2 or length < 8 or observation_budget < 4 * length:
        return None
    start = budget.spent
    if not budget.spend(4 * length):
        return None
    controls = [name for name, _ in gates]
    background = {name: low + (high - low) * (.37 + .03 * index)
        for index, name in enumerate(spec.feature_names)
        for low, high in [spec.feature_bounds[name]]}
    for name in controls:
        background[name] = spec.feature_bounds[name][0]
    reference = Probe(f"composed_cross:{serial}:00",
                      tuple(dict(background) for _ in range(length)))
    interventions = []
    for cell in ((1, 0), (0, 1), (1, 1)):
        points = [dict(background) for _ in range(length)]
        for point in points[1:]:
            for name, selected in zip(controls, cell):
                point[name] = spec.feature_bounds[name][selected]
        interventions.append({"probe": Probe(f"composed_cross:{serial}:{cell[0]}{cell[1]}", tuple(points)),
            "controls": controls, "cell": list(cell), "mode": "cross",
            "prefix_length": 1, "background": dict(background)})
    intervals, widths = [], []
    for _, items in gates:
        left = max(item["threshold_interval"][0] for item in items)
        right = min(item["threshold_interval"][1] for item in items)
        intervals.append([left, right])
        widths.extend(width for item in items for width in
                      item.get("width_candidates", [item["width"]]))
    return {"reference": reference, "interventions": interventions,
        "controls": controls, "threshold_intervals": intervals,
        "width_candidates": sorted(set(widths)), "background": background,
        "gate_evidence": [item for _, items in gates for item in items],
        "observation_cost": 4 * length, "measurement_cost": 0,
        "compute_cost": budget.spent - start}


def assess_composed_cross_experiment(spec, pack, rows_by_probe, budget, *,
                                      minimum_improvement=1e-10):
    """Fit logic and temporal placement to the purchased crossed responses.

    Two branch constants compete first. Their paid failure leaves the Boolean
    composition unresolved: the controls may also change either branch's
    continuous response. A subsequent paid response curve resolves that case.
    """
    from .composed_conditions import compose_condition_response
    start = budget.spent
    probes = [pack["reference"], *(item["probe"] for item in pack["interventions"])]
    rows = [row for probe in probes for row in rows_by_probe[probe.probe_id]]
    result = {"status": "budget_exhausted", "controls": pack["controls"],
        "threshold_intervals": pack["threshold_intervals"],
        "background": dict(pack["background"]), "gate_evidence": pack["gate_evidence"],
        "compute_cost": 0, "matched_rows": len(rows)}
    if not budget.spend(len(rows)):
        return result
    conditions = tuple(Program("ge", children=(Program("feature", name=name),
        Program("const", value=(interval[0] + interval[1]) / 2.)))
        for name, interval in zip(pack["controls"], pack["threshold_intervals"]))
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    constant = Program("const", value=1.)
    widths = pack["width_candidates"] or list(range(1, min(12, len(probes[0].sequence))))
    placements = ("both_inside", "both_outside", "first_inside", "second_inside")
    trials = []
    for width in widths:
        for logic in ("and", "or", "xor"):
            for placement in placements:
                terms = tuple(compose_condition_response(constant, conditions, logic,
                    width, placement, side=side) for side in (True, False))
                fit = fitter.fit(terms)
                if fit is not None:
                    trials.append((fit, logic, width, placement))
    if not trials:
        result["compute_cost"] = budget.spent - start
        return result
    best, logic, width, placement = min(trials, key=lambda item: _rank(item[0]))
    equivalents = [trial for trial in trials if trial[0]["loss"] <= best["loss"] + minimum_improvement]
    result.update(status="conditional", loss=best["loss"], logic=logic,
        cross_program=best["program"].to_dict(),
        logic_candidates=sorted({trial[1] for trial in equivalents}),
        width_candidates=sorted({trial[2] for trial in equivalents}),
        placement_candidates=sorted({trial[3] for trial in equivalents}),
        corner_losses=[{"logic": trial[1], "width": trial[2], "placement": trial[3],
                        "loss": trial[0]["loss"]} for trial in trials],
        compute_cost=budget.spent - start)
    varying = best["loss"] > minimum_improvement
    if varying:
        # Four endpoint cells cannot identify arbitrary continuous responses
        # and their truth table together. Retain these competing compositions
        # until a denser physical experiment supplies the missing distinction.
        result.update(response_model="varying_response", prototype_status="counterexample",
            logic_candidates=["and", "or", "xor"], width_candidates=list(widths),
            placement_candidates=list(placements))
    else:
        result.update(response_model="constant_branches", prototype_status="supported_on_cross")
    cells = {(0, 0): rows_by_probe[pack["reference"].probe_id]}
    cells.update({tuple(item["cell"]): rows_by_probe[item["probe"].probe_id]
                  for item in pack["interventions"]})
    active_contexts = []
    for index, control in enumerate(pack["controls"]):
        axis = [item for item in pack["gate_evidence"] if item["control"] == control
                and item.get("gain", 0.) > minimum_improvement]
        if varying and axis:
            measured_axis = max(axis, key=lambda item: item["gain"])
            active_contexts.append({"control": control,
                "background": dict(measured_axis["background"]),
                "threshold_interval": list(pack["threshold_intervals"][index]),
                "width_candidates": measured_axis.get("width_candidates", [measured_axis["width"]]),
                "placement_candidates": measured_axis.get("placement_candidates", [measured_axis["placement"]]),
                "response_model": "varying_response", "prototype_status": "counterexample"})
            continue
        background = dict(pack["background"])
        other = pack["controls"][1 - index]
        other_value = 1 if logic == "and" else 0
        background[other] = spec.feature_bounds[other][other_value]
        low_cell, high_cell = [other_value, other_value], [other_value, other_value]
        low_cell[index], high_cell[index] = 0, 1
        if not budget.spend(4):
            break
        # Each crossed cell has held its endpoint for longer than the measured
        # width. Its plateau/W is the one-step response at this same background.
        active_contexts.append({"control": control, "background": background,
            "threshold_interval": list(pack["threshold_intervals"][index]),
            "width_candidates": list(result["width_candidates"]),
            "placement_candidates": (axis[0].get("placement_candidates", [axis[0]["placement"]])
                                     if axis else ["inside_window", "outside_window"]),
            "low_response": cells[tuple(low_cell)][-1].target / width,
            "high_response": cells[tuple(high_cell)][-1].target / width})
    result["active_contexts"] = active_contexts
    result["compute_cost"] = budget.spent - start
    return result


def propose_composed_response_probes(spec, context, *, serial=0, length=12):
    """Measure a response curve through independently reset, retained backgrounds.

    Startup responses separate branch coefficients from temporal buildup. A
    single ramp can confound those explanations even when its training fit is
    exact, so every quoted row here is an actual independent physical reset.
    """
    control = context["control"]
    if control not in spec.feature_bounds:
        return []
    length = min(length, spec.max_probe_length)
    if length < 8:
        return []
    low, high = spec.feature_bounds[control]
    points = [{**context["background"], control: low + (high - low) * index / (length - 1)}
              for index in range(length)]
    return [Probe(f"composed_response:{serial}:{control}:{index}", (point,))
            for index, point in enumerate(points)]


def assess_composed_response_probe(spec, context, rows, budget, *, minimum_improvement=1e-10):
    """Fit two continuous branch functions on a purchased conditional curve.

    All other coordinates retain their measured background. A quadratic in
    this one varied coordinate is therefore a testable response hypothesis;
    the complete subsequent model still has to explain the full trajectories.
    """
    rows = list(rows)
    start, count = budget.spent, len(rows)
    result = {**context, "status": "unresolved_response_curve", "curve_rows": count,
              "compute_cost": 0}
    if count < 8 or not budget.spend(count):
        return result
    control = context["control"]
    feature = Program("feature", name=control)
    basis = (Program("const", value=1.), feature, Program("mul", children=(feature, feature)))
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    upper = max(1, max(len(row.history) + 1 for row in rows))
    widths = [1] if all(not row.history for row in rows) else list(dict.fromkeys(
        int(width) for width in context.get("width_candidates", [])
        if width is not None and 1 <= int(width) <= upper)) or list(range(1, upper + 1))
    placements = [placement for placement in context.get("placement_candidates", [])
                  if placement in ("inside_window", "outside_window")]
    placements = list(dict.fromkeys(placements)) or ["inside_window", "outside_window"]
    if widths == [1]:
        placements = ["inside_window"]
    levels = sorted({row.features[control] for row in rows})
    left_bound, right_bound = context["threshold_interval"]
    brackets = [(max(left, left_bound), min(right, right_bound))
                for left, right in zip(levels, levels[1:])
                if min(right, right_bound) > max(left, left_bound)]
    trials, smooth = [], []
    for width in widths:
        plain = [Program("window_sum", lag=width, children=(term,)) if width > 1 else term
                 for term in basis]
        fit = fitter.fit(plain)
        if fit is not None:
            smooth.append(fit)
        for interval in brackets:
            condition = Program("ge", children=(feature, Program("const", value=sum(interval) / 2.)))
            for placement in placements:
                terms = [_masked(term, condition, side, width, placement)
                         for term in basis for side in (True, False)]
                fit = fitter.fit(terms)
                if fit is not None:
                    trials.append((fit, interval, width, placement, terms))
    if not trials:
        result["compute_cost"] = budget.spent - start
        return result
    best, interval, width, placement, terms = min(trials, key=lambda item: _rank(item[0]))
    smooth_loss = min((fit["loss"] for fit in smooth), default=float("inf"))
    coefficients = dict(zip(best["fit_terms"], best["coefficients"][1:]))
    branch_responses = {}
    for side, name in ((False, "low"), (True, "high")):
        branches = [_masked(term, Program("ge", children=(feature,
                    Program("const", value=sum(interval) / 2.))), side, width, placement)
                    for term in basis]
        branch = _sum_terms(basis, [coefficients.get(term, 0.) for term in branches],
                            best["coefficients"][0])
        branch_responses[name] = branch.to_dict()
    result.update(curve_loss=best["loss"], smooth_loss=smooth_loss,
        threshold_interval=list(interval), curve_width=width, curve_placement=placement,
        branch_responses=branch_responses, response_model="varying_response",
        response_program=best["program"].to_dict(),
        curve_trials=[{"threshold_interval": list(item[1]), "width": item[2],
                       "placement": item[3], "loss": item[0]["loss"]} for item in trials],
        compute_cost=budget.spent - start)
    if best["loss"] <= minimum_improvement and best["loss"] + minimum_improvement < smooth_loss:
        result["status"] = "supported_response_curve"
    return result


def propose_composed_boundary_probe(spec, context, *, serial=0):
    """A reset measurement bisects one gate with its other gate activated."""
    control = context["control"]
    left, right = context["threshold_interval"]
    point = dict(context["background"])
    point[control] = (left + right) / 2.
    return Probe(f"composed_boundary:{serial}:{control}", (point,))


def assess_composed_boundary_probe(evidence, context, row, budget, *, response_tolerance=1e-7):
    """Compare measured constants or paid branch functions at the current input."""
    responses = context.get("branch_responses")
    if responses:
        branches = [Program.from_dict(responses[side]) for side in ("low", "high")]
        if not budget.spend(4 + sum(program.size for program in branches)):
            return None
        low_response, high_response = [program.predict(Observation(dict(row.features), 0.))
                                       for program in branches]
    elif "low_response" in context and "high_response" in context:
        if not budget.spend(4):
            return None
        low_response, high_response = context["low_response"], context["high_response"]
    else:
        return {**evidence, "boundary_counterexamples": [*evidence.get("boundary_counterexamples", []),
            {"control": context["control"], "features": dict(row.features), "target": row.target,
             "reason": "unresolved_continuous_branch_response"}]}
    if abs(high_response - low_response) <= response_tolerance:
        return {**evidence, "boundary_counterexamples": [*evidence.get("boundary_counterexamples", []),
            {"control": context["control"], "features": dict(row.features), "target": row.target,
             "reason": "branch_responses_equivalent_at_input"}]}
    if responses and context.get("status") != "supported_response_curve":
        return evidence
    control, value = context["control"], row.features[context["control"]]
    high_error = abs(row.target - high_response)
    low_error = abs(row.target - low_response)
    tolerance = response_tolerance * (1. + max(abs(low_response), abs(high_response)))
    if min(high_error, low_error) > tolerance:
        return {**evidence, "boundary_counterexamples": [*evidence.get("boundary_counterexamples", []),
            {"control": control, "features": dict(row.features), "target": row.target,
             "low_response": low_response, "high_response": high_response}]}
    high = high_error < low_error
    left, right = context["threshold_interval"]
    interval = [left, value] if high else [value, right]
    updated = {**evidence, "threshold_intervals": [list(item) for item in evidence["threshold_intervals"]],
        "active_contexts": [dict(item) for item in evidence["active_contexts"]]}
    index = evidence["controls"].index(control)
    updated["threshold_intervals"][index] = interval
    for item in updated["active_contexts"]:
        if item["control"] == control:
            item["threshold_interval"] = interval
    return updated
