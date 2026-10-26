"""Matched physical controls for cold-start condition and duration discovery."""

from .language import Program
from .models import Observation, Probe
from .program_disagreement_probes import _forecast, _legal_point, _legal_sequence, _public_control_levels
from .research_programs import _Fitter, _rank


def _sequence(background, control, interval, length, mode):
    low, high = interval
    points = [dict(background) for _ in range(length)]
    if mode == "sweep":
        levels = [low + (high - low) * fraction for fraction in (0., .25, .5, .75, 1.)]
        prefix = 2
        block = max(1, (length - prefix) // len(levels))
        for index in range(prefix, length):
            points[index][control] = levels[min(len(levels) - 1, (index - prefix) // block)]
    else:
        levels, prefix = [background[control], low, high], 1
        if mode == "pulse":
            # The same switch before and after buildup distinguishes an outer
            # condition's unknown amplitude from its unknown window duration.
            for start in (prefix, length - 3):
                points[start][control] = low
                points[start + 1][control] = high
        else:
            for index in range(prefix, length):
                points[index][control] = low if index < length // 2 else high
    return tuple(points), levels, prefix


def _separation(programs, reference, intervention, budget):
    predictions = []
    for program in programs:
        original = _forecast(program, reference, budget.spend)
        changed = _forecast(program, intervention, budget.spend)
        if original is None or changed is None:
            break
        predictions.append([None if left is None or right is None else right - left
                            for left, right in zip(original, changed)])
    common = [row for row in zip(*predictions) if all(value is not None for value in row)]
    return sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.


def propose_conditional_experiments(spec, paid_rows, programs, budget, *, serial=0,
                                    observation_budget=96, control_levels=None):
    """Cover every public control in two retained, nonzero backgrounds.

    One reference is shared by the interventions in each background. The first
    context scans levels; the second tests transient switches, choosing pulse
    or step by the current explanations' forecast disagreement. Acquisition is
    performed by the caller, which pays the entire quoted pack before fitting.
    """
    bounds = spec.feature_bounds
    legal_levels = _public_control_levels(bounds, control_levels)
    controls = tuple(name for name in spec.feature_names if name in bounds)
    length = min(12, spec.max_probe_length)
    pack_cost = (len(controls) + 1) * length
    if not controls or length < 8 or observation_budget < pack_cost:
        return []
    programs = [program for program in programs
                if {name for name, _ in program.observed_variables()} <= set(controls)][:4]
    # Off-center and unequal coordinates retain both linear and quadratic
    # branch differences when a zero or sign-symmetric background hides them.
    backgrounds = [{name: low + (high - low) * fraction
        for index, name in enumerate(controls)
        for low, high in [bounds[name]]
        for fraction in [( .31 + .05 * (index % 3)) if context == 0
                         else (.72 - .04 * (index % 3))]}
        for context in range(min(2, observation_budget // pack_cost))]
    if legal_levels is not None:
        backgrounds = [_legal_point(background, legal_levels) for background in backgrounds]
    packs = []
    for context, background in enumerate(backgrounds):
        before = budget.spent
        reference_sequence = tuple(dict(background) for _ in range(length))
        interventions = []
        for control in controls:
            modes = ("sweep",) if context == 0 else ("pulse", "step")
            trials = []
            for mode in modes:
                if not budget.spend(length):
                    break
                sequence, levels, prefix = _sequence(background, control, bounds[control], length, mode)
                sequence = _legal_sequence(sequence, legal_levels, budget.spend)
                if sequence is None:
                    break
                if legal_levels is not None:
                    levels = list(dict.fromkeys(point[control] for point in sequence[prefix:]))
                score = _separation(programs, reference_sequence, sequence, budget)
                trials.append((score, mode, sequence, levels, prefix))
            if not trials:
                break
            score, mode, sequence, levels, prefix = max(trials, key=lambda item: item[0])
            interventions.append({"probe": Probe(f"conditional:{serial}:{context}:{control}:{mode}", sequence),
                "control": control, "mode": mode, "levels": levels, "prefix_length": prefix,
                "background": dict(background), "predicted_separation": score})
        if len(interventions) != len(controls):
            break
        packs.append({"reference": Probe(f"conditional:{serial}:{context}:reference", reference_sequence),
            "interventions": interventions, "background": background,
            "observation_cost": pack_cost, "measurement_cost": 0,
            "compute_cost": budget.spent - before,
            "predicted_separation": sum(plan["predicted_separation"] for plan in interventions)})
    return packs


def assess_conditional_experiment(spec, plan, reference_rows, measured_rows, budget,
                                  *, minimum_improvement=1e-10):
    """Compare smooth and switched explanations of an actually measured delta.

    Frozen backgrounds reduce the response to one controlled coordinate. A
    smooth linear/quadratic temporal response competes with a switched response
    before a control can become a proposed gate. Equivalent widths and nesting
    orders remain explicit when this experiment has not distinguished them.
    """
    reference_rows, measured_rows = list(reference_rows), list(measured_rows)
    count, start = len(measured_rows), budget.spent
    control, background = plan["control"], plan["background"]
    result = {"status": "budget_exhausted", "control": control,
        "background": dict(background), "mode": plan["mode"],
        "gain": 0., "conditional_gain": 0., "compute_cost": 0,
        "matched_rows": count}
    if not budget.spend(2 * count):
        return result
    rows = [Observation(dict(changed.features), changed.target - original.target,
                        tuple(dict(point) for point in changed.history))
            for original, changed in zip(reference_rows, measured_rows)]
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    reference_fitter = _Fitter(reference_rows, budget, spec.risk_quantile)
    feature = Program("feature", name=control)
    constant = Program("const", value=float(background[control]))
    difference = Program("sub", children=(feature, constant))
    square_difference = Program("sub", children=(
        Program("mul", children=(feature, feature)),
        Program("const", value=background[control] ** 2)))
    upper = max(len(row.history) for row in rows)
    reference_widths = []
    if budget.spend(2 * count):
        mean = sum(row.target for row in reference_rows) / count
        reference_widths.append((sum((row.target - mean) ** 2 for row in reference_rows) / count, 1))
    for width in range(2, max(1, upper) + 1):
        fit = reference_fitter.fit((Program("window_sum", lag=width,
            children=(Program("const", value=1.),)),))
        if fit is not None:
            reference_widths.append((fit["loss"], width))
    reference_supported = []
    if reference_widths:
        reference_loss = min(item[0] for item in reference_widths)
        result["reference_loss"] = reference_loss
        if reference_loss <= minimum_improvement:
            reference_supported = [width for measured_loss, width in reference_widths
                                   if measured_loss <= reference_loss + minimum_improvement]
            result["reference_width_candidates"] = reference_supported
    smooth = []
    for width in range(1, max(1, upper) + 1):
        terms = [Program("window_sum", lag=width, children=(term,)) if width > 1 else term
                 for term in (difference, square_difference)]
        fit = fitter.fit(terms)
        if fit is not None:
            smooth.append((fit, width))
    if not smooth:
        result["compute_cost"] = budget.spent - start
        return result
    smooth_fit, smooth_width = min(smooth, key=lambda item: _rank(item[0]))
    result.update(smooth_loss=smooth_fit["loss"], smooth_width=smooth_width)
    if smooth_fit["loss"] <= minimum_improvement:
        result.update(status="smooth", compute_cost=budget.spent - start)
        return result

    levels = sorted(set(row.features[control] for row in rows))
    trials = []
    # The paid reference startup already excluded other durations. Search the
    # same supported candidates directly, preserving funds for a local scan
    # of the unresolved gate interval and its joint response refit.
    trial_widths = reference_supported or range(1, max(1, upper) + 1)
    for width in trial_widths:
        window_count = Program("window_sum", lag=width, children=(Program("const", value=1.),))
        for left, right in zip(levels, levels[1:]):
            threshold = (left + right) / 2.
            condition = Program("ge", children=(feature, Program("const", value=threshold)))
            changed_condition = Program("sub", children=(condition,
                Program("const", value=float(background[control] >= threshold))))
            for placement in ("inside_window", "outside_window"):
                term = (Program("window_sum", lag=width, children=(changed_condition,))
                        if placement == "inside_window" else
                        Program("mul", children=(changed_condition, window_count)))
                fit = fitter.fit((term,))
                if fit is not None:
                    trials.append((fit, width, placement, [left, right]))
    if not trials:
        result["compute_cost"] = budget.spent - start
        return result
    usable = [trial for trial in trials if not reference_supported or trial[1] in reference_supported]
    if not usable:
        usable = trials
    best, width, placement, interval = min(usable, key=lambda item: _rank(item[0]))
    response_model = "constant_branches"
    if best["loss"] > minimum_improvement:
        # A coordinate can vary smoothly within either branch while also
        # deciding which branch is active. A constant jump alone does not
        # exclude such a measured condition.
        from .composed_condition_experiments import varying_condition_delta_terms
        varying = []
        for proposed_width in trial_widths:
            for left, right in zip(levels, levels[1:]):
                threshold = (left + right) / 2.
                for proposed_placement in ("inside_window", "outside_window"):
                    terms = varying_condition_delta_terms(control, background[control],
                        threshold, proposed_width, proposed_placement)
                    fit = fitter.fit(terms)
                    if fit is not None:
                        varying.append((fit, proposed_width, proposed_placement, [left, right]))
        if varying:
            revised = min(varying, key=lambda item: _rank(item[0]))
            if revised[0]["loss"] + minimum_improvement < best["loss"]:
                best, width, placement, interval = revised
                response_model = "varying_response"
                usable = [*usable, *varying]
    gain = smooth_fit["loss"] - best["loss"]
    equivalent = [trial for trial in usable
                  if trial[0]["loss"] <= best["loss"] + minimum_improvement]
    if response_model == "varying_response":
        # Five purchased levels can support several piecewise quadratics.
        # Keep every observationally equivalent cut for the denser experiment
        # instead of turning one polynomial interpolation into a physical cut.
        interval = [min(trial[3][0] for trial in equivalent),
                    max(trial[3][1] for trial in equivalent)]
    result.update(status="conditional" if gain > minimum_improvement else "smooth",
        conditional_loss=best["loss"], gain=gain, conditional_gain=gain,
        threshold_interval=interval, width=width, placement=placement,
        width_candidates=sorted({trial[1] for trial in equivalent}),
        placement_candidates=sorted({trial[2] for trial in equivalent}),
        threshold_intervals=sorted({tuple(trial[3]) for trial in equivalent}),
        response_model=response_model,
        compute_cost=budget.spent - start)
    return result


def propose_conditional_boundary_probe(spec, evidence, *, serial=0, length=24, control_levels=None):
    """Purchase a local gate scan inside the currently measured interval."""
    control, background = evidence["control"], evidence["background"]
    left, right = evidence["threshold_interval"]
    low, high = spec.feature_bounds[control]
    legal_levels = _public_control_levels(spec.feature_bounds, control_levels)
    left, right = max(left, low), min(right, high)
    length = min(length, spec.max_probe_length)
    if length < 3 or right <= left:
        return None
    if legal_levels is not None:
        scan = [value for value in legal_levels[control] if left <= value <= right]
        if not scan:
            return None
        background = _legal_point(background, legal_levels)
        points = [dict(background) for _ in range(min(length, len(scan) + 1))]
        for index, point in enumerate(points[1:]):
            point[control] = scan[index * max(0, len(scan) - 1) // max(1, len(points) - 2)]
        return Probe(f"conditional_boundary:{serial}:{control}", tuple(points))
    points = [dict(background) for _ in range(length)]
    for index in range(1, length):
        points[index][control] = left + (right - left) * (index - 1) / (length - 2)
    return Probe(f"conditional_boundary:{serial}:{control}", tuple(points))
