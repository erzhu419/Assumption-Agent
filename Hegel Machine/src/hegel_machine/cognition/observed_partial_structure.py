"""Carry local temporal support into competing complete response programs.

A fixed control background can identify a duration while leaving conditioning
inside, outside, or absent observationally equivalent. Its response polynomial
does not predict other backgrounds. These nominees refit all response terms on
the same purchased, varied observations before an experiment may use them.
"""

from bisect import bisect_left
from dataclasses import replace
from itertools import combinations_with_replacement

from .conditional_coordinates import _masked
from .language import Program
from .models import SearchBudget
from .research_programs import _entry, _Fitter, _rank


def partial_structure_nominations(paid_events, tolerance=1e-7):
    """Retain actual structural improvements without promoting their support.

    Exact local fits and measured conditional gains nominate a duration. All
    recorded explanations at that duration then compete, including the smooth
    deletion and poorer gate placement. Their original failures stay failures.
    """
    families, experiments, widths = {}, [], {}
    plan = {}
    for index, event in enumerate(paid_events):
        if event.get("action") == "plan_observed_condition_experiment":
            plan = event
        if event.get("action") != "distinguish_paid_condition_duration":
            continue
        experiments.append((index, event, plan))
        for item in event.get("competing_structures", []):
            if not ((item.get("response_supported") and item["loss"] <= tolerance)
                    or (item.get("proposed_structure") == "conditional_duration"
                        and item.get("gain", 0.) > tolerance)):
                continue
            key = item["control"], item["width"]
            widths[key] = min(widths.get(key, float("inf")), item["loss"])
    for index, event, _ in experiments:
        for item in event.get("competing_structures", []):
            if (item.get("control"), item.get("width")) not in widths:
                continue
            structure = item.get("proposed_structure")
            if structure not in {"smooth_duration", "conditional_duration"}:
                continue
            placement = item.get("placement") if structure == "conditional_duration" else None
            key = item["control"], item["width"], placement, structure
            if key not in families:
                families[key] = {**{name: value for name, value in item.items()
                    if name not in {"parameter_trials", "duration_trials"}},
                    "support_scope": "fixed_control_background",
                    "observed_experiment_index": index, "partial_support": []}
    for index, event, previous_plan in experiments:
        for item in event.get("competing_structures", []):
            structure = item.get("proposed_structure")
            placement = item.get("placement") if structure == "conditional_duration" else None
            key = item.get("control"), item.get("width"), placement, structure
            if key not in families:
                continue
            support = {"observed_experiment_index": index,
                "probe_id": event.get("probe_id", previous_plan.get("probe_id")),
                "background": dict(item.get("background",
                    event.get("fixed_control_background", {}))),
                "fit_rows": item.get("fit_rows", event.get("measured_rows", 0)),
                "local_loss": item["loss"], "width": item["width"],
                "placement": placement, "proposed_structure": structure,
                "support_scope": "fixed_control_background",
                "response_supported": bool(item.get("response_supported")),
                "condition_identified": bool(item.get("condition_identified")),
                "status": item.get("status"), "gain": item.get("gain", 0.),
                **{name: item[name] for name in ("threshold_interval", "local_program")
                   if name in item}}
            families[key]["partial_support"].append(support)
    # The best actual local fit chooses which duration gets a complete common
    # response comparison first. It does not label the other explanations true.
    placement_order = {None: 0, "inside_window": 1, "outside_window": 2}
    return sorted(families.values(), key=lambda item: (
        0. if widths[item["control"], item["width"]] <= 1e-12
        else widths[item["control"], item["width"]],
        item["width"], placement_order.get(item.get("placement"), 3)))


def partial_structure_competitors(spec, training, paid_events, budget,
        max_candidates=8, *, tolerance=1e-7):
    """Fit each partial family on all purchased original-response rows.

    A shared fitter gives smooth and conditional rivals one initial comparison
    before refining thresholds in turns. The bounded fit does not purchase
    observations or assert fresh confirmation; failed attempts remain charged.
    """
    nominations = partial_structure_nominations(paid_events, tolerance)
    rows = list(training)
    if not nominations or len(rows) < 8 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows)) - {"__target__"}
    if not any(len({row.features[name] for row in rows}) > 1
               for name in set(spec.feature_bounds) & names):
        return []
    current = [Program("feature", name=name) for name in spec.feature_names if name in names]
    basis = [Program("const", value=1.), *current,
        *(Program("mul", children=pair)
          for pair in combinations_with_replacement(current, 2))]
    if 2 * len(basis) + 3 >= len(rows):
        return []
    local = SearchBudget(min(20000, budget.remaining))
    fitter = _Fitter(rows, local, spec.risk_quantile)
    upper = max(len(row.history) + 1 for row in rows)
    families, brackets, measured, trials = [], {}, {}, []
    for nominee in nominations:
        control, width = nominee["control"], int(nominee["width"])
        if control not in names or not 1 <= width <= upper:
            continue
        placement = nominee.get("placement")
        if nominee["proposed_structure"] == "smooth_duration":
            family = control, width, None
        elif placement in {"inside_window", "outside_window"}:
            family = control, width, placement
            if control not in brackets:
                column = fitter.column(Program("feature", name=control))
                if column is None or not local.spend(len(rows)):
                    continue
                values = sorted({*column, *(past[control] for row in rows
                    for past in row.history if control in past)})
                brackets[control] = tuple(zip(values, values[1:]))
            if not brackets[control]:
                continue
        else:
            continue
        if family not in families:
            families.append(family)

    def evaluate(family, index=None):
        key = family, index
        if key in measured:
            return measured[key]
        control, width, placement = family
        record = {"control": control, "width": width, "placement": placement}
        if placement is None:
            terms = [Program("window_sum", lag=width, children=(term,))
                if width > 1 else term for term in basis]
        else:
            left, right = brackets[control][index]
            condition = Program("ge", children=(Program("feature", name=control),
                Program("const", value=(left + right) / 2.)))
            terms = [_masked(term, condition, side, width, placement)
                     for term in basis for side in (True, False)]
            record.update(threshold=(left + right) / 2., threshold_interval=[left, right],
                          threshold_index=index)
        before = local.spent
        fit = fitter.fit(terms)
        if fit is None:
            return None
        record.update(loss=fit["loss"], compute_cost=local.spent - before)
        trials.append(record)
        measured[key] = fit, record
        return fit, record

    proposals = {}
    for control, width, placement in families:
        if placement is None:
            continue
        family = control, width, placement
        actual = brackets[control]
        values = [left for left, _ in actual]
        local_indices = []
        for nominee in nominations:
            if (nominee["control"], nominee["width"], nominee.get("placement")) != family:
                continue
            for support in nominee["partial_support"]:
                if "threshold_interval" in support:
                    threshold = sum(support["threshold_interval"]) / 2.
                    local_indices.append(max(0, min(len(actual) - 1,
                        bisect_left(values, threshold) - 1)))
        proposals[family] = list(dict.fromkeys([*local_indices, 0, len(actual) - 1]))
    duration_groups = {}
    for family in families:
        duration_groups.setdefault(family[:2], []).append(family)
    # Finish competing placements at the informative duration before spending
    # the shared allowance on every weaker window's coarse fit. A local gate
    # improvement supplies a trial cutoff, not a confidence interval or a law.
    for (control, width), group in duration_groups.items():
        hints = [support for nominee in nominations
            if (nominee["control"], nominee["width"]) == (control, width)
            for support in nominee["partial_support"]
            if support.get("gain", 0.) > tolerance and "threshold_interval" in support]
        first_index = None
        if control in brackets:
            actual = brackets[control]
            first_index = (len(actual) - 1) // 2
            if hints:
                hint = min(hints, key=lambda item: item["local_loss"])
                threshold = sum(hint["threshold_interval"]) / 2.
                first_index = max(0, min(len(actual) - 1,
                    bisect_left([left for left, _ in actual], threshold) - 1))
        for family in group:
            evaluate(family, first_index if family[2] else None)
        # Failed local brackets stay alternatives, never hard restrictions.
        while local.remaining >= len(rows) * (2 * len(basis) + 2):
            before = local.spent
            if any(fit["loss"] <= 1e-12 for (family, _), (fit, _) in measured.items()
                   if family in group):
                break
            for family in group:
                if family not in proposals:
                    continue
                known = [(index, result) for (other, index), result in measured.items()
                         if other == family]
                if not known:
                    continue
                proposals[family] = [index for index in proposals[family]
                                     if (family, index) not in measured]
                if not proposals[family]:
                    winner = min(known, key=lambda item: _rank(item[1][0]))[0]
                    tested = sorted({0, len(brackets[family[0]]) - 1,
                                     *(index for index, _ in known)})
                    pos = tested.index(winner)
                    adjacent = tested[max(0, pos - 1):pos + 2]
                    proposals[family] = [(left + right) // 2
                        for left, right in zip(adjacent, adjacent[1:]) if right - left > 1]
                    proposals[family].extend(index for index in (winner - 1, winner + 1)
                        if 0 <= index < len(brackets[family[0]]))
                next_index = next((index for index in proposals[family]
                                  if (family, index) not in measured), None)
                if next_index is not None:
                    proposals[family].remove(next_index)
                    evaluate(family, next_index)
            if local.spent == before:
                break
        if any(fit["loss"] <= 1e-12 for fit, _ in measured.values()):
            break
    budget.spend(local.spent)
    best = {}
    for (family, _), result in measured.items():
        if family not in best or _rank(result[0]) < _rank(best[family][0]):
            best[family] = result
    claims = []
    for family, (fit, record) in sorted(best.items(), key=lambda item: _rank(item[1][0])):
        control, width, placement = family
        support = [item for nominee in nominations
            if (nominee["control"], nominee["width"], nominee.get("placement")) == family
            for item in nominee["partial_support"]]
        step = {"operation": "discover_conditional_coordinates" if placement else "discover_temporal_coordinates",
            "question": "Compare measured temporal alternatives on all purchased responses",
            "program": fit["program"], "loss": fit["loss"], "width": width,
            "history_bound": max(len(row.history) for row in rows),
            "partial_support": support, "parameter_trials": trials}
        if placement:
            step.update(gate_coordinate=Program("feature", name=control).to_dict(),
                placement=placement, threshold=record["threshold"],
                threshold_interval=record["threshold_interval"],
                branch_basis=[term.to_dict() for term in basis if term.op != "const"],
                parameter_terms=[term.to_dict() for term in fit["fit_terms"]],
                experiment_evidence=[], structure_nominations=nominations, duration_screen=[])
        else:
            step.update(basis_terms=len(basis), selected_terms=len(fit["terms"]), width_trials=trials)
        claim = _entry(fit, [step], len(rows), local.spent, len(claims))
        claim["candidate"] = replace(claim["candidate"], candidate_id=f"partial_structure:{len(claims)}",
            rationale="Measured temporal alternatives compete as complete fitted response programs")
        claim.update(partial_support=support, structure_nominations=nominations,
                     fit_features=list(spec.feature_names), fit_rows=len(rows))
        claims.append(claim)
    return claims[:max_candidates]
