"""Discover a measured condition jointly with branch response and duration.

The search compares conditioning each input before aggregation with selecting a
windowed response at its current condition. Both branches are fitted together;
neither a previously fitted response nor an old threshold is treated as truth.
"""

from dataclasses import replace
from collections import Counter

from .language import Program
from .research_programs import _entry, _Fitter, _rank


def conditional_search_allowance(feature_count, row_count, history_length=0):
    """Quote a bounded joint attempt in the same candidate-row search units."""
    gates = min(feature_count, 8)
    widths = 1 if history_length < 2 else 8
    branches = min(24, max(2, row_count // 3))
    return row_count * ((gates * 7 + 8 * widths + 24) * (branches + 2)
                        + branches * branches)


def _masked(term, condition, side, width, placement):
    zero = Program("const", value=0.)
    if placement == "outside_window" and width > 1:
        term = Program("window_sum", lag=width, children=(term,))
    branches = (term, zero) if side else (zero, term)
    masked = Program("if", children=(condition, *branches))
    return (Program("window_sum", lag=width, children=(masked,))
            if placement == "inside_window" and width > 1 else masked)


def _coordinates(spec, rows, basis):
    names = set.intersection(*(set(row.features) for row in rows)) - {"__target__"}
    available = [Program("feature", name=name) for name in spec.feature_names if name in names]
    measured = list(dict.fromkeys([*available, *(term for term in basis
        if term.history_depth == 0 and term.required_features()
        and term.required_features() <= names)]))
    return available, measured


def _search(spec, rows, budget, *, basis, gates, max_candidates, reused=False,
            width_candidates=None, placement_candidates=None):
    count, start = len(rows), budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    available, basis = _coordinates(spec, rows, basis)
    if not basis or not gates:
        return []
    gates = list(dict.fromkeys(term for term in gates if term.history_depth == 0
        and term.required_features() <= set.intersection(*(set(row.features) for row in rows))))
    basis.append(Program("const", value=1.))
    upper = max(len(row.history) for row in rows)
    upper = max(1, upper)
    placements = (list(dict.fromkeys(placement for placement in placement_candidates
        if placement in {"inside_window", "outside_window"}))
        if placement_candidates is not None else
        ["inside_window", "outside_window"] if upper > 1 else ["inside_window"])
    if not placements:
        return []
    capacity = min(24, max(2, count // 3))
    trials, measured, thresholds, values = [], {}, {}, {}

    for gate in gates:
        column = fitter.column(gate)
        if column is None or not budget.spend(count):
            continue
        occurrences = Counter(column)
        distinct = sorted(occurrences)
        # Each present side needs several measured examples to fit a response.
        # The resulting interval, rather than a precise physical cutoff, is the
        # threshold information available from these observations.
        intervals, seen = [], 0
        for left, right in zip(distinct, distinct[1:]):
            seen += occurrences[left]
            if seen >= 3 and count - seen >= 3:
                intervals.append((left, right))
        if intervals:
            values[gate], thresholds[gate] = column, intervals
    gates = [gate for gate in gates if gate in thresholds]
    if not gates:
        return []

    def evaluate(gate, index, width, placement):
        key = gate, index, width, placement
        if key in measured:
            return measured[key]
        before = budget.spent
        left, right = thresholds[gate][index]
        threshold = (left + right) / 2.
        condition = Program("ge", children=(gate, Program("const", value=threshold)))
        pairs = [(term, side) for term in basis for side in (True, False)]
        terms = [_masked(term, condition, side, width, placement) for term, side in pairs]
        if len(terms) + 3 < count:
            fit = fitter.fit(terms)
        else:
            # When the complete basis is not identifiable, compare its paid
            # conditional columns instead of truncating by grammar order.
            columns = [(term, fitter.column(term)) for term in terms]
            columns = [(term, column) for term, column in columns if column is not None]
            if not budget.spend(count * (len(columns) + 1)):
                return None
            center = sum(fitter.targets) / count
            ranked = []
            for term, column in columns:
                mean = sum(column) / count
                norm = sum((value - mean) ** 2 for value in column)
                if norm > 1e-12:
                    covariance = sum((value - mean) * (target - center)
                        for value, target in zip(column, fitter.targets))
                    ranked.append((covariance * covariance / norm, term))
            fit = fitter.fit([term for _, term in sorted(ranked,
                key=lambda item: item[0], reverse=True)[:capacity]])
        if fit is None:
            return None
        record = {"gate_coordinate": gate, "threshold": threshold,
            "threshold_interval": [left, right], "threshold_index": index,
            "width": width, "placement": placement, "loss": fit["loss"],
            "terms": len(fit["fit_terms"]), "compute_cost": budget.spent - before}
        trials.append(record)
        measured[key] = fit, record
        return fit, record

    if width_candidates is None:
        widths = [1]
        width = 2
        while width < upper:
            widths.append(width)
            width *= 2
        if upper > 1:
            widths.append(upper)
    else:
        widths = list(dict.fromkeys(int(width) for width in width_candidates
                                   if 1 <= int(width) <= upper))
        if not widths:
            return []
    coarse = {gate: sorted({round(fraction * (len(thresholds[gate]) - 1))
                           for fraction in (.25, .5, .75)}) for gate in gates}
    # A branch inside a window can have little correlation with the current
    # response. First compare unconditional durations on the purchased rows,
    # then screen conditions at that measured scale instead of assuming W=1.
    plain, duration_trials = [], []
    for width in widths:
        before = budget.spent
        terms = [Program("window_sum", lag=width, children=(term,))
                 if width > 1 else term for term in basis]
        if len(terms) + 3 < count:
            fit = fitter.fit(terms)
            if fit is not None:
                plain.append((width, fit))
                duration_trials.append({"width": width, "loss": fit["loss"],
                    "terms": len(fit["fit_terms"]), "compute_cost": budget.spent - before})
    pilot_width = min(plain, key=lambda item: _rank(item[1]))[0] if plain else 1

    def plain_entry():
        width, fit = min(plain, key=lambda item: _rank(item[1]))
        step = {"operation": "discover_temporal_coordinates",
            "question": "Retain a common measured response duration when a condition is unnecessary",
            "program": fit["program"], "loss": fit["loss"], "width": width,
            "history_bound": upper, "basis_terms": len(basis),
            "selected_terms": len(fit["terms"]), "width_trials": duration_trials}
        entry = _entry(fit, [step], count, budget.spent - start)
        entry["candidate"] = replace(entry["candidate"], candidate_id="conditional_plain",
            rationale="Measured duration explains the response without an additional condition")
        if reused:
            entry["research_program"]["reused_procedure"] = True
        return entry

    if plain and min(fit["loss"] for _, fit in plain) <= 1e-12:
        return [plain_entry()]
    for position in range(3):
        for gate in gates:
            if position < len(coarse[gate]):
                for placement in (placements if placement_candidates is not None else
                        ("inside_window", "outside_window") if pilot_width > 1
                        else ("inside_window",)):
                    evaluate(gate, coarse[gate][position], pilot_width, placement)
    if not measured:
        return [plain_entry()] if plain else []

    nominated = []
    for placement in placements:
        keys = [key for key in measured if key[3] == placement]
        if not keys:
            keys = list(measured)
        key = min(keys, key=lambda key: _rank(measured[key][0]))
        nominated.append((key[0], placement))
    nominated_indices = {}
    for gate, placement in nominated:
        keys = sorted((key for key in measured if key[0] == gate and key[3] == placement),
                      key=lambda key: _rank(measured[key][0]))
        if not keys:
            keys = sorted((key for key in measured if key[0] == gate),
                          key=lambda key: _rank(measured[key][0]))
        # The other coarse brackets remain measured alternatives. Carry only
        # the leading one through the duration screen, then spend the saved
        # passes refining its boundary jointly with the preferred duration.
        nominated_indices[gate, placement] = list(dict.fromkeys(key[1] for key in keys))[:1]
    for width in (width for width in widths if width != pilot_width):
        for gate, placement in nominated:
            if (width == 1 and placement != "inside_window"
                    and placement_candidates is None):
                continue
            for index in nominated_indices[gate, placement]:
                evaluate(gate, index, width, placement)
        if min(fit["loss"] for fit, _ in measured.values()) <= 1e-12:
            break

    # Coarse comparisons have already covered the full measured history range.
    # Alternate parameter revisions now: a wrong threshold must not force an
    # exhaustive width search before the condition can ever be corrected.
    fit_price = count * (len(basis) * 2 + 2)
    reserve = fit_price
    families = []
    for placement in placements:
        keys = [key for key in measured if key[3] == placement]
        if keys:
            key = min(keys, key=lambda key: _rank(measured[key][0]))
            families.append((key[0], placement))
    families.sort(key=lambda family: min(_rank(fit) for key, (fit, _) in measured.items()
                                        if key[0] == family[0] and key[3] == family[1]))
    def family_best(gate, placement):
        return min((key for key in measured if key[0] == gate and key[3] == placement),
                   key=lambda key: _rank(measured[key][0]))

    round_index = 0
    while families and budget.remaining >= reserve + fit_price:
        before_round = budget.spent
        for gate, placement in families:
            best_key = family_best(gate, placement)
            tested_widths = sorted({key[2] for key in measured
                                   if key[0] == gate and key[3] == placement})
            if width_candidates is not None:
                proposed_widths = []
            elif round_index == 0:
                proposed_widths = [(left + right) // 2 for left, right in
                    zip(tested_widths, tested_widths[1:]) if right - left > 1]
            else:
                position = tested_widths.index(best_key[2])
                adjacent = tested_widths[max(0, position - 1):position + 2]
                proposed_widths = list(dict.fromkeys([*adjacent,
                    *((left + right) // 2 for left, right in zip(adjacent, adjacent[1:])
                      if right - left > 1)]))
            for width in proposed_widths:
                if budget.remaining < reserve + fit_price:
                    break
                evaluate(gate, best_key[1], width, placement)
            # Immediately test a boundary revision at the newly preferred
            # scale, before descending another level into window intervals.
            best_key = family_best(gate, placement)
            for _ in range(2):
                tested_indices = sorted({0, len(thresholds[gate]) - 1,
                    *(key[1] for key in measured if key[0] == gate
                      and key[3] == placement and key[2] == best_key[2])})
                position = tested_indices.index(best_key[1])
                adjacent = tested_indices[max(0, position - 1):position + 2]
                proposed_indices = list(dict.fromkeys([
                    *(index for index in (best_key[1] - 1, best_key[1] + 1)
                      if 0 <= index < len(thresholds[gate])),
                    *((left + right) // 2 for left, right in zip(adjacent, adjacent[1:])
                      if right - left > 1)]))
                for index in proposed_indices:
                    if budget.remaining < reserve + fit_price:
                        break
                    evaluate(gate, index, best_key[2], placement)
                best_key = family_best(gate, placement)
                if measured[best_key][0]["loss"] <= 1e-12:
                    break
            if min(fit["loss"] for fit, _ in measured.values()) <= 1e-12:
                break
        if (budget.spent == before_round
                or min(fit["loss"] for fit, _ in measured.values()) <= 1e-12):
            break
        round_index += 1

    ranked = sorted(measured.values(), key=lambda item: _rank(item[0]))
    # Equivalent gated fits at neighbouring brackets compete once as executable
    # programs. Distinct measured hypotheses remain available to fresh evidence.
    winners = list({fit["program"]: (fit, record) for fit, record in reversed(ranked)}.values())
    winners.sort(key=lambda item: _rank(item[0]))
    # Offer a deletion path from the strongest measured response alongside
    # the full fit. Fresh validation can retain a smaller branch law when
    # weak noisy coefficients only appeared valuable on the fitting rows.
    simplified = []
    for full, record in winners[:2]:
        sparse = fitter.fit(full["terms"])
        if sparse is not None:
            full = sparse
        if not budget.spend(count * len(full["fit_terms"])):
            break
        ordered = sorted(zip(full["fit_terms"], full["coefficients"][1:]),
            key=lambda item: abs(item[1]) *
                (max(fitter.columns[item[0]]) - min(fitter.columns[item[0]])))
        path = [full]
        for term, _ in ordered:
            remaining = [item for item in path[-1]["fit_terms"] if item != term]
            if len(remaining) < 2:
                break
            fit = fitter.fit(remaining)
            if fit is None:
                break
            path.append(fit)
        def adjusted(fit):
            return max(fit["loss"], 1e-12) / (1. - (len(fit["fit_terms"]) + 1) / count) ** 2
        simplified.append((min(path, key=adjusted), record))
        target = len(full["fit_terms"])
        while target > 2:
            target = (target + 1) // 2
            simplified.append((min(path, key=lambda fit: abs(len(fit["fit_terms"]) - target)), record))
    options = [*winners[:1], *simplified, *winners[1:]]
    options = list({fit["program"]: (fit, record) for fit, record in options}.values())
    # Keep several response complexities, then fill remaining places with the
    # strongest distinct parameter hypotheses from the joint search.
    claims = [plain_entry()] if plain and max_candidates > 1 else []
    for index, (fit, record) in enumerate(options[:max_candidates - len(claims)]):
        step = {"operation": "discover_conditional_coordinates",
            "question": "Jointly discover a condition, both branch responses and their temporal support",
            "program": fit["program"], "loss": fit["loss"],
            "gate_coordinate": record["gate_coordinate"].to_dict(),
            "threshold": record["threshold"], "threshold_interval": record["threshold_interval"],
            "width": record["width"], "placement": record["placement"],
            "history_bound": upper, "branch_basis": [term.to_dict() for term in basis
                                                       if term.op != "const"],
            "parameter_terms": [term.to_dict() for term in fit["fit_terms"]],
            "duration_screen": duration_trials,
            "parameter_trials": [{**trial, "gate_coordinate": trial["gate_coordinate"].to_dict()}
                                  for trial in trials]}
        entry = _entry(fit, [step], count, budget.spent - start, index)
        entry["candidate"] = replace(entry["candidate"],
            candidate_id=f"conditional_coordinate:{index}",
            rationale="Measured conditional responses jointly select gate placement and duration")
        if reused:
            entry["research_program"]["reused_procedure"] = True
        claims.append(entry)
    return claims


def propose_conditional_coordinates(spec, rows, budget, *, basis, max_candidates=8,
                                    gate_names=None, width_candidates=None,
                                    placement_candidates=None):
    """Fit conditional response hypotheses using purchased rows alone."""
    rows = list(rows)
    if len(rows) < 8 or max_candidates < 1:
        return []
    available, measured = _coordinates(spec, rows, basis)
    gates = available[:8]
    if gate_names is not None:
        gates = [gate for gate in gates if gate.op == "feature" and gate.name in gate_names]
    return _search(spec, rows, budget, basis=measured, gates=gates,
                   max_candidates=max_candidates, width_candidates=width_candidates,
                   placement_candidates=placement_candidates)


def propose_experimental_conditional_coordinates(spec, rows, budget, *, basis,
                                                 evidence, nominations=(), max_candidates=8):
    """Refit measured condition proposals against the complete paid response.

    Matched experiments nominate a coordinate and its temporal support. Their
    measured boundary brackets guide the search; all purchased fitting rows
    still have to agree with the resulting joint response.
    """
    rows = list(rows)
    if len(rows) < 8:
        return []
    start, count = budget.spent, len(rows)
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    names = set.intersection(*(set(row.features) for row in rows)) - {"__target__"}
    measured_basis = list(dict.fromkeys(term for term in basis
        if term.history_depth == 0 and term.required_features() <= names))
    measured_basis.append(Program("const", value=1.))
    if 2 * len(measured_basis) + 3 >= count:
        return []
    trials, tested, families = [], {}, []
    exploratory_families = set()
    supported = [item for item in evidence if item.get("gain", 0.) > 1e-10]
    supported.sort(key=lambda item: (
        len(item.get("width_candidates", [item.get("width")])) *
        len(item.get("placement_candidates", [item.get("placement")])),
        item.get("conditional_loss", float("inf"))))
    nominated_controls = set()
    measured_brackets = {}
    for item in supported:
        control = item.get("control")
        interval = item.get("threshold_interval")
        width = item.get("width")
        if (control not in spec.feature_names or not interval or width is None
                or item.get("gain", 0.) <= 1e-10):
            continue
        if control in nominated_controls:
            continue
        nominated_controls.add(control)
        interval = list(interval)
        for other in supported:
            other_interval = other.get("threshold_interval")
            if other.get("control") == control and other_interval:
                intersection = [max(interval[0], other_interval[0]),
                                min(interval[1], other_interval[1])]
                if intersection[0] < intersection[1]:
                    interval = intersection
        gate = Program("feature", name=control)
        column = fitter.column(gate)
        if column is None or not budget.spend(count):
            break
        # Include history controls too: an inside-window gate is evaluated at
        # past input values, not only at the present comparison row.
        values = sorted({*column, *(past[control] for row in rows for past in row.history
                                   if control in past)})
        measured_brackets[control] = tuple(zip(values, values[1:]))
        brackets = [(left, right) for left, right in measured_brackets[control]
                    if right > interval[0] and left < interval[1]]
        if not brackets:
            continue
        placements = item.get("placement_candidates", [item.get("placement", "inside_window")])
        widths = item.get("width_candidates", [width])
        for proposed_width in widths:
            for placement in placements:
                family = (gate, int(proposed_width), placement, tuple(brackets))
                if family not in families:
                    families.append(family)
    current_family_count = len(families)
    # A background can expose a useful temporal structure without explaining
    # its whole response. Its cutoff remains a proposal: varied controls and
    # measured history may revise it anywhere in their actual support.
    for item in sorted(nominations, key=lambda item: item.get("loss", float("inf"))):
        control, width = item.get("control"), item.get("width")
        if (control not in spec.feature_names or width is None
                or (item.get("gain", 0.) <= 1e-10 and not item.get("response_supported"))):
            continue
        gate = Program("feature", name=control)
        if control not in measured_brackets:
            column = fitter.column(gate)
            if column is None or not budget.spend(count):
                break
            values = sorted({*column, *(past[control] for row in rows for past in row.history
                                       if control in past)})
            measured_brackets[control] = tuple(zip(values, values[1:]))
        brackets = measured_brackets[control]
        for proposed_width in item.get("width_candidates", [width]):
            for placement in item.get("placement_candidates", [item.get("placement", "inside_window")]):
                family = (gate, int(proposed_width), placement, brackets)
                if brackets and family not in families:
                    families.append(family)
                    exploratory_families.add(family)
    if not families:
        return []

    plain = []

    def evaluate(gate, width, placement, brackets, index):
        left, right = brackets[index]
        key = gate, width, placement, left, right
        if key in tested:
            return tested[key]
        before = budget.spent
        condition = Program("ge", children=(gate, Program("const", value=(left + right) / 2.)))
        terms = [_masked(term, condition, side, width, placement)
                 for term in measured_basis for side in (True, False)]
        fit = fitter.fit(terms)
        if fit is None:
            return None
        record = {"gate_coordinate": gate.to_dict(), "threshold": (left + right) / 2.,
            "threshold_interval": [left, right], "width": width, "placement": placement,
            "loss": fit["loss"], "terms": len(fit["fit_terms"]),
            "compute_cost": budget.spent - before}
        trials.append(record)
        tested[key] = fit, record
        return fit, record

    price = count * (2 * len(measured_basis) + 2)
    reserve = 3 * count
    # Give current measured support a complete original-response fit before
    # paying for old alternative windows. Failed support continues into the
    # historical competition with the same fitter, cached trials and budget.
    for family_group in (families[:current_family_count], families[current_family_count:]):
        for width in dict.fromkeys(family[1] for family in family_group):
            if any(previous_width == width for previous_width, _ in plain):
                continue
            terms = [Program("window_sum", lag=width, children=(term,))
                     if width > 1 else term for term in measured_basis]
            fit = fitter.fit(terms)
            if fit is not None:
                plain.append((width, fit))
        searches = []
        for gate, width, placement, brackets in family_group:
            indices = ([(len(brackets) - 1) // 2]
                       if (gate, width, placement, brackets) in exploratory_families
                       else sorted({0, len(brackets) - 1, (len(brackets) - 1) // 2}))
            found = []
            for index in indices:
                result = evaluate(gate, width, placement, brackets, index)
                if result is not None:
                    found.append((index, result))
            if found:
                searches.append((gate, width, placement, brackets, found))
        searches.sort(key=lambda family: min(_rank(result[0]) for _, result in family[4]))
        for gate, width, placement, brackets, found in searches:
            while budget.remaining >= price + reserve:
                winner = min(found, key=lambda pair: _rank(pair[1][0]))[0]
                sampled = sorted({0, len(brackets) - 1, *(index for index, _ in found)})
                pos = sampled.index(winner)
                adjacent = sampled[max(0, pos - 1):pos + 2]
                midpoints = [(left + right) // 2 for left, right in zip(adjacent, adjacent[1:])
                             if right - left > 1]
                neighbors = [index for index in (winner - 1, winner + 1)
                             if 0 <= index < len(brackets)]
                proposed = list(dict.fromkeys([*midpoints,
                    *(neighbors if (gate, width, placement, brackets) not in exploratory_families
                      or not midpoints else [])]))
                before = budget.spent
                for index in proposed:
                    if any(previous == index for previous, _ in found):
                        continue
                    result = evaluate(gate, width, placement, brackets, index)
                    if result is not None:
                        found.append((index, result))
                    if result is not None and result[0]["loss"] <= 1e-12:
                        break
                if (budget.spent == before or min(result[0]["loss"] for _, result in found) <= 1e-12):
                    break
            if min(result[0]["loss"] for _, result in found) <= 1e-12:
                break

        if any(fit["loss"] <= 1e-12 for fit, _ in tested.values()):
            break

    winners = sorted(tested.values(), key=lambda result: _rank(result[0]))
    if not winners and not plain:
        return []
    # The full quadratic basis is a search language. Keep the actually used
    # response terms in the reusable graph when the paid refit can support it.
    sparse = fitter.fit(winners[0][0]["terms"]) if winners else None
    if sparse is not None:
        winners.insert(0, (sparse, winners[0][1]))
    claims, programs = [], set()
    for fit, record in winners:
        if fit["program"] in programs:
            continue
        programs.add(fit["program"])
        step = {"operation": "discover_conditional_coordinates",
            "question": "Resolve a measured branch experiment jointly with both response laws",
            "program": fit["program"], "loss": fit["loss"],
            **{key: record[key] for key in ("gate_coordinate", "threshold", "threshold_interval",
                                          "width", "placement")},
            "history_bound": max(len(row.history) for row in rows),
            "branch_basis": [term.to_dict() for term in measured_basis if term.op != "const"],
            "parameter_terms": [term.to_dict() for term in fit["fit_terms"]],
            "experiment_evidence": list(evidence), "duration_screen": [],
            "structure_nominations": list(nominations),
            "parameter_trials": trials}
        claim = _entry(fit, [step], count, budget.spent - start, len(claims))
        claim["candidate"] = replace(claim["candidate"],
            candidate_id=f"conditional_experiment:{len(claims)}",
            rationale="Paid matched trajectories resolve conditional response and temporal support")
        claims.append(claim)
        if len(claims) >= max_candidates:
            break
    # Keep deletion available on the same varied responses. A conditional
    # nomination cannot by itself make an unnecessary branch worth retaining.
    for width, fit in plain:
        step = {"operation": "discover_temporal_coordinates",
            "question": "Compare the nominated duration with a common response",
            "program": fit["program"], "loss": fit["loss"], "width": width,
            "history_bound": max(len(row.history) for row in rows),
            "basis_terms": len(measured_basis), "selected_terms": len(fit["terms"]),
            "width_trials": []}
        claim = _entry(fit, [step], count, budget.spent - start, len(claims))
        claim["candidate"] = replace(claim["candidate"], candidate_id=f"experimental_plain:{width}",
            rationale="The measured temporal response can omit an unnecessary condition")
        claims.append(claim)
    claims.sort(key=lambda item: (0. if item["training_loss"] < 1e-12 else item["training_loss"],
                                 item["candidate"].program.size))
    return claims[:max_candidates]


def revise_conditional_threshold(program, control, threshold):
    """Share one threshold revision across the fitted branch computations."""
    children = tuple(revise_conditional_threshold(child, control, threshold)
                     for child in program.children)
    if (program.op == "ge" and children[0] == Program("feature", name=control)
            and children[1].op == "const"):
        children = children[0], Program("const", value=threshold)
    return replace(program, children=children)


def replay_conditional_coordinates(payload, rows, budget, *, risk_quantile=.5,
                                   feature_mapping=None):
    """Rewrite a saved conditional method's parameters on a changed task.

    Keep its executable branch coordinates, then jointly retest gate position,
    gate placement and window support. Previously used threshold/width values
    are information about the method, not fixed constraints on its next task.
    """
    from .models import TaskSpec
    from .research_graph import ResearchGraph
    from .scale_bindings import bind_measurement_arguments
    from copy import deepcopy

    rows = list(rows)
    if len(rows) < 8:
        return None
    saved = next((step for step in reversed(payload.get("selection_steps", []))
                  if step["operation"] == "discover_conditional_coordinates"), None)
    if saved is None:
        return None
    basis = [Program.from_dict(term) for term in saved["branch_basis"]]
    gate = Program.from_dict(saved["gate_coordinate"])
    graph = ResearchGraph.from_dict(payload["graph"])
    retained_terms = [graph.nodes[key].answer for key in payload["term_nodes"]]
    retained_terms = [term for term in retained_terms if term is not None]
    if feature_mapping:
        basis = [bind_measurement_arguments(term, feature_mapping) for term in basis]
        gate = bind_measurement_arguments(gate, feature_mapping)
        retained_terms = [bind_measurement_arguments(term, feature_mapping) for term in retained_terms]
    names = tuple(sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"}))
    spec = TaskSpec("conditional_method_replay", "Retest measured conditional response", names,
                    (), (), risk_quantile=risk_quantile)
    available, measured = _coordinates(spec, rows, basis)
    start = budget.spent
    fitter = _Fitter(rows, budget, risk_quantile)
    retained_fit = fitter.fit(retained_terms) if retained_terms else None
    claims = _search(spec, rows, budget, basis=measured,
        gates=list(dict.fromkeys([gate, *available]))[:8], max_candidates=1, reused=True)
    if retained_fit is not None:
        steps = []
        for original in payload.get("selection_steps", []):
            step = deepcopy(original)
            step["program"] = Program.from_dict(step["program"])
            step["historical"] = True
            if feature_mapping:
                step["program"] = bind_measurement_arguments(step["program"], feature_mapping)
                for key in ("gate_coordinate",):
                    if key in step:
                        step[key] = bind_measurement_arguments(
                            Program.from_dict(step[key]), feature_mapping).to_dict()
                for key in ("branch_basis", "parameter_terms"):
                    if key in step:
                        step[key] = [bind_measurement_arguments(Program.from_dict(term),
                            feature_mapping).to_dict() for term in step[key]]
            steps.append(step)
        steps.append({"operation": "refit_research_program",
            "question": "Retest the current revised computation dependencies on fresh observations",
            "program": retained_fit["program"], "loss": retained_fit["loss"]})
        retained = _entry(retained_fit, steps, len(rows), budget.spent - start)
        retained["research_program"]["reused_procedure"] = True
        claims.append(retained)
    if not claims:
        return None
    selected = min(claims, key=lambda item: (
        0. if item["training_loss"] < 1e-12 else item["training_loss"],
        item["candidate"].program.size))
    selected["research_program"]["search_compute"] = budget.spent - start
    output = selected["research_program"]["graph"]["nodes"]
    # The selected answer carries
    # the full current comparison cost, including an unsuccessful alternative.
    output[selected["research_program"]["output_node"]]["compute_cost"] = budget.spent - start
    return selected


def adopt_conditional_research_program(payload, program, *, measured_loss,
                                      fit_rows, compute_cost):
    """Rebuild a retained conditional method from its actually adopted answer.

    The caller has already measured this adoption. Its current coefficients and
    computation dependencies are recorded directly, while prior discovery steps
    remain research history. No new fit or observation is performed here.
    """
    from copy import deepcopy
    from .research_programs import _plan

    weights, intercept = {}, 0.

    def collect(term, weight):
        nonlocal intercept
        if term.op == "add":
            for child in term.children:
                collect(child, weight)
        elif term.op == "sub":
            collect(term.children[0], weight)
            collect(term.children[1], -weight)
        elif term.op == "mul" and any(child.op == "const" for child in term.children):
            constant = 0 if term.children[0].op == "const" else 1
            collect(term.children[1 - constant], weight * term.children[constant].value)
        elif term.op == "const":
            intercept += weight * term.value
        else:
            weights[term] = weights.get(term, 0.) + weight

    collect(program, 1.)
    terms = tuple(term for term, weight in weights.items() if weight != 0.)
    fit = {"program": program, "loss": measured_loss, "terms": terms,
           "fit_terms": terms,
           "coefficients": [intercept, *(weights[term] for term in terms)]}
    steps = []
    for original in payload.get("selection_steps", []):
        step = deepcopy(original)
        step["program"] = Program.from_dict(step["program"])
        step["historical"] = True
        steps.append(step)
    operation = "adopt_composed_response" if any(step["operation"] == "discover_composed_conditions"
        for step in steps) else "adopt_conditional_response"
    steps.append({"operation": operation,
        "question": "Retain the computation dependencies of the measured adopted response",
        "program": program, "loss": measured_loss})
    return _plan(fit, steps, fit_rows, compute_cost)
