"""Discover two measured conditions and their positions around a window.

Matched interventions nominate controls and boundary intervals. The complete
purchased trajectories then compare Boolean compositions, temporal placement,
and both response laws together. A retained procedure supplies hypotheses for
the next task; its old parameters do not constrain the new fit.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace

from .conditional_coordinates import _coordinates
from .language import Program
from .models import Observation, TaskSpec
from .research_programs import _entry, _Fitter, _rank


LOGICS = ("and", "or", "xor")
PLACEMENTS = ("both_inside", "both_outside", "first_inside", "second_inside")


def _truth(logic, first, second):
    if logic == "and":
        return first and second
    if logic == "or":
        return first or second
    return first != second


def _condition(conditions, logic):
    first, second = conditions
    if logic in ("and", "or"):
        return Program(logic, children=(first, second))
    return Program("if", children=(first, Program("not", children=(second,)), second))


def compose_condition_response(term, conditions, logic, width, placement, *, side=True):
    """Evaluate one response coordinate on one side of a two-gate condition.

For a mixed placement, the current outer gate selects which truth table the
historical inner gate uses. This keeps the two physical response laws shared
across all four gate combinations instead of inventing four independent laws.
"""
    zero = Program("const", value=0.)

    def window(expression):
        return Program("window_sum", lag=width, children=(expression,)) if width > 1 else expression

    def value(first, second):
        return term if _truth(logic, first, second) == side else zero

    if placement == "both_outside":
        branches = (window(term), zero) if side else (zero, window(term))
        return Program("if", children=(_condition(conditions, logic), *branches))
    if placement == "both_inside" or width == 1:
        branches = (term, zero) if side else (zero, term)
        return window(Program("if", children=(_condition(conditions, logic), *branches)))
    if placement == "first_inside":
        inside, outside = conditions
        branches = [window(Program("if", children=(inside, value(True, outer), value(False, outer))))
                    for outer in (True, False)]
    else:
        outside, inside = conditions
        branches = [window(Program("if", children=(inside, value(outer, True), value(outer, False))))
                    for outer in (True, False)]
    return Program("if", children=(outside, *branches))


def composed_search_allowance(feature_count, row_count, history_length=0):
    """Quote comparisons of twelve compositions and local parameter revisions."""
    basis = feature_count + feature_count * (feature_count + 1) // 2 + 1
    return row_count * (40 * (2 * basis + 2) + 2 * max(1, history_length))


def _supported_gates(spec, evidence):
    """Normalize actual joint evidence or the paid single-control observations."""
    joint = evidence if isinstance(evidence, dict) else next(
        (item for item in evidence if item.get("controls")), None)
    if joint and joint.get("status") == "conditional":
        controls = joint.get("controls", [])
        intervals = joint.get("threshold_intervals", [])
        if len(controls) >= 2 and len(intervals) >= 2:
            gates = [{"coordinate": Program("feature", name=name), "interval": list(interval)}
                     for name, interval in zip(controls[:2], intervals[:2])
                     if name in spec.feature_names and len(interval) == 2 and interval[0] < interval[1]]
            if len(gates) == 2 and gates[0]["coordinate"] != gates[1]["coordinate"]:
                return gates, joint
    axis = evidence.get("gate_evidence", []) if isinstance(evidence, dict) else evidence
    supported = [item for item in axis if item.get("gain", 0.) > 1e-10
                 and item.get("control") in spec.feature_names and item.get("threshold_interval")]
    supported.sort(key=lambda item: item.get("conditional_loss", float("inf")))
    gates = []
    for item in supported:
        gate = Program("feature", name=item["control"])
        if any(saved["coordinate"] == gate for saved in gates):
            continue
        interval = list(item["threshold_interval"])
        for other in supported:
            if other.get("control") == item["control"]:
                bounds = other["threshold_interval"]
                common = [max(interval[0], bounds[0]), min(interval[1], bounds[1])]
                if common[0] < common[1]:
                    interval = common
        gates.append({"coordinate": gate, "interval": interval})
        if len(gates) == 2:
            break
    widths = [width for item in supported for width in
              item.get("width_candidates", [item.get("width", 1)])]
    return gates, {"width_candidates": list(dict.fromkeys(widths))}


def _search(spec, rows, budget, *, basis, gates, support, max_candidates, evidence,
            reused=False):
    count, start = len(rows), budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    _, basis = _coordinates(spec, rows, basis)
    basis.append(Program("const", value=1.))
    if len(gates) != 2 or 2 * len(basis) + 3 >= count:
        return []
    history_bound = max(1, max(len(row.history) + 1 for row in rows))
    prepared = []
    for item in gates:
        gate = item["coordinate"]
        column = fitter.column(gate)
        # A historical gate reads measured historical controls too. Pay this
        # scan before using those values to narrow its boundary.
        scan_cost = sum(len(row.history) for row in rows)
        if column is None or not budget.spend(scan_cost):
            return []
        historical = [gate.predict(Observation(point, 0., row.history[:index]))
                      for row in rows for index, point in enumerate(row.history)]
        values = sorted(set([*column, *historical]))
        lower, upper = item["interval"]
        brackets = [(max(left, lower), min(right, upper))
                    for left, right in zip(values, values[1:])
                    if min(right, upper) > max(left, lower)]
        if not brackets:
            return []
        prepared.append({**item, "brackets": brackets})

    widths = [int(width) for width in support.get("width_candidates", [])
              if width is not None and 1 <= int(width) <= history_bound]
    widths = list(dict.fromkeys(widths)) or [1]
    # Evidence supplies a small measured set of scales, rather than a Cartesian
    # product of every recorded history length with both gate boundaries.
    width_counts = Counter(support.get("width_candidates", widths))
    widths.sort(key=lambda width: (-width_counts[width], width))
    logics = [logic for logic in support.get("logic_candidates", []) if logic in LOGICS]
    placements = [placement for placement in support.get("placement_candidates", [])
                  if placement in PLACEMENTS]
    if support.get("status") == "conditional":
        # A crossed paid experiment can exclude a truth table or temporal
        # order before the much larger response-basis fit is attempted.
        logics = list(dict.fromkeys(logics)) or list(LOGICS)
        placements = list(dict.fromkeys(placements)) or list(PLACEMENTS)
    else:
        # Old procedure parameters only order the hypotheses on a new task.
        logics = list(dict.fromkeys([*logics, *LOGICS]))
        placements = list(dict.fromkeys([*placements, *PLACEMENTS]))
    tested, trials = {}, []
    price = count * (2 * len(basis) + 2)
    reserve = 3 * count

    def evaluate(indices, width, logic, placement):
        if width == 1:
            placement = "both_inside"
        key = tuple(indices), width, logic, placement
        if key in tested:
            return tested[key]
        before = budget.spent
        intervals = [prepared[index]["brackets"][position]
                     for index, position in enumerate(indices)]
        thresholds = [(left + right) / 2. for left, right in intervals]
        conditions = [Program("ge", children=(item["coordinate"], Program("const", value=value)))
                      for item, value in zip(prepared, thresholds)]
        terms = [compose_condition_response(term, conditions, logic, width, placement, side=side)
                 for term in basis for side in (True, False)]
        fit = fitter.fit(terms)
        if fit is None:
            return None
        record = {"gate_coordinates": [item["coordinate"].to_dict() for item in prepared],
            "thresholds": thresholds, "threshold_intervals": [list(pair) for pair in intervals],
            "threshold_indices": list(indices), "width": width, "logic": logic,
            "placement": placement, "loss": fit["loss"], "terms": len(fit["fit_terms"]),
            "compute_cost": budget.spent - before}
        tested[key] = fit, record
        trials.append(record)
        return fit, record

    middle = tuple((len(item["brackets"]) - 1) // 2 for item in prepared)
    # Compare every temporal/Boolean family before spending the remaining
    # fitting budget on the strongest boundary revision.
    for width in widths[:3]:
        for logic in logics:
            for placement in placements:
                if budget.remaining < price + reserve:
                    break
                evaluate(middle, width, logic, placement)
            if tested and min(fit["loss"] for fit, _ in tested.values()) <= 1e-12:
                break
        if tested and min(fit["loss"] for fit, _ in tested.values()) <= 1e-12:
            break
    if not tested:
        return []

    def family_key(key):
        return key[1:]

    families = []
    for key in sorted(tested, key=lambda key: _rank(tested[key][0])):
        if family_key(key) not in families:
            families.append(family_key(key))
    # A boundary error can make two neighboring temporal explanations compete.
    # Keep both until their actual full response has resolved the uncertainty.
    for family in families[:2]:
        rounds = 0
        while budget.remaining >= price + reserve:
            candidates = [key for key in tested if family_key(key) == family]
            winner = min(candidates, key=lambda key: _rank(tested[key][0]))
            if tested[winner][0]["loss"] <= 1e-12:
                break
            before = budget.spent
            for axis in range(2):
                candidates = [key for key in tested if family_key(key) == family]
                winner = min(candidates, key=lambda key: _rank(tested[key][0]))
                indices = list(winner[0])
                last = len(prepared[axis]["brackets"]) - 1
                sampled = sorted({0, last, *(key[0][axis] for key in candidates
                    if key[0][1 - axis] == indices[1 - axis])})
                position = sampled.index(indices[axis])
                adjacent = sampled[max(0, position - 1):position + 2]
                proposed = list(dict.fromkeys([
                    *((left + right) // 2 for left, right in zip(adjacent, adjacent[1:])
                      if right - left > 1),
                    *(index for index in (indices[axis] - 1, indices[axis] + 1)
                      if 0 <= index <= last)]))
                for index in proposed:
                    if budget.remaining < price + reserve:
                        break
                    revised = list(indices)
                    revised[axis] = index
                    evaluate(revised, *family)
                if min(fit["loss"] for fit, _ in tested.values()) <= 1e-12:
                    break
            rounds += 1
            if budget.spent == before or rounds >= 6:
                break
        if min(fit["loss"] for fit, _ in tested.values()) <= 1e-12:
            break

    winners = sorted(tested.values(), key=lambda result: _rank(result[0]))
    sparse = fitter.fit(winners[0][0]["terms"])
    if sparse is not None:
        winners.insert(0, (sparse, winners[0][1]))
    claims, programs = [], set()
    for fit, record in winners:
        if fit["program"] in programs:
            continue
        programs.add(fit["program"])
        step = {"operation": "discover_composed_conditions",
            "question": "Jointly discover measured gate composition, temporal placement and both response laws",
            "program": fit["program"], "loss": fit["loss"],
            **{key: record[key] for key in ("gate_coordinates", "thresholds", "threshold_intervals",
                "width", "logic", "placement")},
            "history_bound": history_bound,
            "branch_basis": [term.to_dict() for term in basis if term.op != "const"],
            "parameter_terms": [term.to_dict() for term in fit["fit_terms"]],
            "experiment_evidence": deepcopy(evidence), "parameter_trials": trials}
        curve_steps = []
        for context in evidence.get("active_contexts", []) if isinstance(evidence, dict) else []:
            if context.get("status") != "supported_response_curve":
                continue
            from .conditional_coordinates import revise_conditional_threshold
            curve = revise_conditional_threshold(Program.from_dict(context["response_program"]),
                context["control"], sum(context["threshold_interval"]) / 2.)
            curve_steps.append({"operation": "discover_conditional_response_curve",
                "question": "Replace contradicted constant branch responses with paid continuous response curves",
                "program": curve, "loss": context["curve_loss"],
                "control": context["control"], "background": deepcopy(context["background"]),
                "threshold_interval": list(context["threshold_interval"]),
                "branch_responses": deepcopy(context["branch_responses"]),
                "fit_rows": context["curve_rows"]})
        entry = _entry(fit, [*curve_steps, step], count, budget.spent - start, len(claims))
        entry["candidate"] = replace(entry["candidate"], candidate_id=f"composed_condition:{len(claims)}",
            rationale="Paid controls jointly select Boolean composition and temporal gate placement")
        if reused:
            entry["research_program"]["reused_procedure"] = True
        claims.append(entry)
        if len(claims) >= max_candidates:
            break
    return claims


def propose_composed_conditions(spec, rows, budget, *, basis, evidence, max_candidates=8):
    """Construct executable hypotheses from two actually supported gates."""
    rows = list(rows)
    if len(rows) < 8 or max_candidates < 1:
        return []
    gates, support = _supported_gates(spec, evidence)
    return _search(spec, rows, budget, basis=basis, gates=gates, support=support,
                   max_candidates=max_candidates, evidence=evidence)


def propose_measured_window_response(spec, rows, budget, *, basis, evidence, max_candidates=8):
    """Jointly fit a common duration supported by purchased reference startups.

    A smooth intervention need not propose a gate. Its matched reference can
    still reveal temporal support, which the complete response coordinates
    must then explain together. The branch-free hypothesis competes with the
    conditional explanations on the same actual rows.
    """
    rows = list(rows)
    if len(rows) < 8 or max_candidates < 1:
        return []
    axis = evidence.get("gate_evidence", []) if isinstance(evidence, dict) else evidence
    upper = max(1, max(len(row.history) + 1 for row in rows))
    widths = Counter(int(width) for item in axis
        for width in item.get("reference_width_candidates", [])
        if width is not None and 1 <= int(width) <= upper)
    if not widths:
        return []
    _, measured_basis = _coordinates(spec, rows, basis)
    measured_basis.append(Program("const", value=1.))
    if len(measured_basis) + 3 >= len(rows):
        return []
    start = budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    measured, trials = [], []
    for width in sorted(widths, key=lambda item: (-widths[item], item)):
        before = budget.spent
        terms = [Program("window_sum", lag=width, children=(term,)) if width > 1 else term
                 for term in measured_basis]
        fit = fitter.fit(terms)
        if fit is None:
            continue
        trials.append({"width": width, "loss": fit["loss"],
            "terms": len(fit["fit_terms"]), "supporting_references": widths[width],
            "compute_cost": budget.spent - before})
        measured.append((fit, width))
    if not measured:
        return []
    measured.sort(key=lambda item: _rank(item[0]))
    sparse = fitter.fit(measured[0][0]["terms"])
    if sparse is not None:
        measured.insert(0, (sparse, measured[0][1]))
    claims, programs = [], set()
    for fit, width in measured:
        if fit["program"] in programs:
            continue
        programs.add(fit["program"])
        step = {"operation": "discover_measured_window_response",
            "question": "Explain all measured response coordinates at a reference-supported duration",
            "program": fit["program"], "loss": fit["loss"], "width": width,
            "history_bound": upper,
            "branch_basis": [term.to_dict() for term in measured_basis if term.op != "const"],
            "parameter_terms": [term.to_dict() for term in fit["fit_terms"]],
            "experiment_evidence": deepcopy(evidence), "duration_trials": trials}
        entry = _entry(fit, [step], len(rows), budget.spent - start, len(claims))
        entry["candidate"] = replace(entry["candidate"],
            candidate_id=f"measured_window_response:{len(claims)}",
            rationale="Purchased reference startup supports a jointly fitted common response duration")
        claims.append(entry)
        if len(claims) >= max_candidates:
            break
    return claims


def replay_composed_conditions(payload, rows, budget, *, risk_quantile=.5,
                               feature_mapping=None):
    """Refit a retained computation and retest its gate composition on paid rows."""
    from .research_graph import ResearchGraph
    from .scale_bindings import bind_measurement_arguments

    rows = list(rows)
    saved = next((step for step in reversed(payload.get("selection_steps", []))
                  if step["operation"] == "discover_composed_conditions"), None)
    if saved is None or len(rows) < 8:
        return None
    basis = [Program.from_dict(term) for term in saved["branch_basis"]]
    gates = [Program.from_dict(term) for term in saved["gate_coordinates"]]
    graph = ResearchGraph.from_dict(payload["graph"])
    terms = [graph.nodes[key].answer for key in payload["term_nodes"]
             if graph.nodes[key].answer is not None]
    if feature_mapping:
        basis = [bind_measurement_arguments(term, feature_mapping) for term in basis]
        gates = [bind_measurement_arguments(term, feature_mapping) for term in gates]
        terms = [bind_measurement_arguments(term, feature_mapping) for term in terms]
    start = budget.spent
    fitter = _Fitter(rows, budget, risk_quantile)
    retained_fit = fitter.fit(terms) if terms else None
    names = tuple(sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"}))
    spec = TaskSpec("composed_method_replay", "Retest measured gate composition", names, (), (),
                    risk_quantile=risk_quantile)
    claims = []
    if retained_fit is None or retained_fit["loss"] > 1e-12:
        current_gates = []
        for gate in gates:
            column = fitter.column(gate)
            if column:
                current_gates.append({"coordinate": gate, "interval": [min(column), max(column)]})
        old_width = saved["width"]
        upper = max(1, max(len(row.history) + 1 for row in rows))
        widths = [width for width in (old_width, old_width - 2, old_width + 2)
                  if 1 <= width <= upper]
        support = {"width_candidates": widths,
            "logic_candidates": [saved["logic"]], "placement_candidates": [saved["placement"]]}
        claims = _search(spec, rows, budget, basis=basis, gates=current_gates, support=support,
            max_candidates=1, evidence={"origin": "retained_procedure"}, reused=True)
    if retained_fit is not None:
        steps = []
        for original in payload.get("selection_steps", []):
            step = deepcopy(original)
            step["program"] = Program.from_dict(step["program"])
            step["historical"] = True
            if feature_mapping:
                step["program"] = bind_measurement_arguments(step["program"], feature_mapping)
                for key in ("gate_coordinates", "branch_basis", "parameter_terms"):
                    if key in step:
                        step[key] = [bind_measurement_arguments(Program.from_dict(term),
                            feature_mapping).to_dict() for term in step[key]]
            steps.append(step)
        steps.append({"operation": "refit_research_program",
            "question": "Retest the retained composed response on newly paid observations",
            "program": retained_fit["program"], "loss": retained_fit["loss"]})
        retained = _entry(retained_fit, steps, len(rows), budget.spent - start)
        retained["research_program"]["reused_procedure"] = True
        claims.append(retained)
    if not claims:
        return None
    selected = min(claims, key=lambda item: (0. if item["training_loss"] <= 1e-12
                    else item["training_loss"], item["candidate"].program.size))
    selected["research_program"]["search_compute"] = budget.spent - start
    selected["research_program"]["graph"]["nodes"][selected["research_program"]["output_node"]][
        "compute_cost"] = budget.spent - start
    return selected
