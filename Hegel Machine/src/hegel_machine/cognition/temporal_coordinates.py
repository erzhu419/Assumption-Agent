"""Jointly discover a response basis and its finite temporal support."""

from dataclasses import replace

from .language import Program
from .research_programs import _entry, _Fitter, _rank


def propose_temporal_coordinates(spec, rows, budget, *, basis, max_candidates=8):
    """Fit current intermediate terms inside a data-selected common window.

    Coefficients are estimated again at every width. The fitted current response
    is not treated as a known inner mechanism. Width trials use only purchased
    history; independent response validation remains the caller's responsibility.
    """
    rows = list(rows)
    if len(rows) < 6 or max_candidates < 1:
        return []
    # At least one acquired row must extend beyond the window. A window as
    # long as the complete trajectory is indistinguishable from accumulation.
    upper = max(len(row.history) for row in rows)
    if upper < 2:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    basis = list(dict.fromkeys(term for term in basis if term.history_depth == 0
        and term.required_features() and term.required_features() <= names))
    if not basis:
        return []
    # A window's internal offset also changes during a trajectory's startup.
    # A global regression intercept alone cannot express this contribution.
    basis.append(Program("const", value=1.))
    start, count = budget.spent, len(rows)
    fitter, trials, measured = _Fitter(rows, budget, spec.risk_quantile), [], {}
    capacity = max(2, min(24, count // 2))

    def evaluate(width):
        if width in measured or not 2 <= width <= upper:
            return
        before = budget.spent
        terms = tuple(Program("window_sum", lag=width, children=(term,)) for term in basis)
        fit = None
        if len(terms) < count - 2:
            fit = fitter.fit(terms)
            if fit is not None and len(fit["fit_terms"]) > capacity:
                # A full but identifiable pilot estimates joint contributions;
                # retain the strongest before comparing this width. This avoids
                # truncating the generated basis by its enumeration order.
                if not budget.spend(count * len(fit["fit_terms"])):
                    return
                contributions = [(abs(coefficient) *
                    (max(fitter.columns[term]) - min(fitter.columns[term])), term)
                    for term, coefficient in zip(fit["fit_terms"], fit["coefficients"][1:])]
                retained = [term for _, term in sorted(contributions,
                    key=lambda item: item[0], reverse=True)[:capacity]]
                fit = fitter.fit(retained)
        else:
            # Very small samples cannot identify the complete generated basis.
            # Screen paid columns by evidence, then fit a smaller joint pilot.
            columns = [(term, fitter.column(term)) for term in terms]
            columns = [(term, column) for term, column in columns if column is not None]
            if not budget.spend(count * (len(columns) + 1)):
                return
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
        if fit is not None:
            measured[width] = fit
            trials.append({"width": width, "loss": fit["loss"],
                "terms": len(fit["fit_terms"]), "compute_cost": budget.spent - before})

    width = 2
    while width < upper:
        evaluate(width)
        if measured and min(fit["loss"] for fit in measured.values()) <= 1e-12:
            break
        width *= 2
    if not measured or min(fit["loss"] for fit in measured.values()) > 1e-12:
        evaluate(upper)
    if not measured:
        return []
    tested = sorted({1, *measured})
    intervals = list(zip(tested, tested[1:]))
    # Finite-history fits need not have a unimodal loss. Refine every coarse
    # interval before descending another level into any one of them.
    # Leave enough of the same budget to compare response complexities at the
    # winning width; otherwise a wider search merely returns an overfit basis.
    refinement_floor = count * (len(basis) + 2 + 3 * min(len(basis), capacity))
    while intervals and min(fit["loss"] for fit in measured.values()) > 1e-12:
        following = []
        for left, right in intervals:
            if right - left < 2:
                continue
            if budget.remaining < refinement_floor:
                break
            middle = (left + right) // 2
            evaluate(middle)
            following.extend(((left, middle), (middle, right)))
            if min(fit["loss"] for fit in measured.values()) <= 1e-12:
                break
        intervals = following if budget.remaining >= refinement_floor else []
    winner = min(measured, key=lambda width: _rank(measured[width]))
    full = measured[winner]
    # Follow the deletion path beyond the fit preferred by training evidence.
    # Independent response validation must be able to choose a smaller basis
    # when noisy training coefficients give spurious terms apparent value.
    path = [full]
    def adjusted(fit):
        parameters = len(fit["terms"]) + 1
        return max(fit["loss"], 1e-12) / (1. - parameters / count) ** 2

    if budget.spend(count * len(full["fit_terms"])):
        ordered = sorted(zip(full["fit_terms"], full["coefficients"][1:]),
            key=lambda item: abs(item[1]) *
                (max(fitter.columns[item[0]]) - min(fitter.columns[item[0]])))
        for term, _ in ordered:
            remaining = [item for item in path[-1]["fit_terms"] if item != term]
            if not remaining:
                break
            trial = fitter.fit(remaining)
            if trial is not None:
                path.append(trial)
    fits = [min(path, key=adjusted), full]
    # Geometric coverage gives validation a range of response complexities
    # without supplying a task-specific number of true physical terms.
    target = len(full["fit_terms"])
    while target > 1:
        target = (target + 1) // 2
        fit = min(path, key=lambda item: abs(len(item["fit_terms"]) - target))
        fits.append(fit)
    fits = list({fit["program"]: fit for fit in fits}.values())
    cost = budget.spent - start
    claims = []
    for index, fit in enumerate(fits[:max_candidates]):
        step = {"operation": "discover_temporal_coordinates",
            "question": "Jointly choose a response basis and the duration of its observed influence",
            "program": fit["program"], "loss": fit["loss"], "width": winner,
            "history_bound": upper, "basis_terms": len(basis),
            "selected_terms": len(fit["terms"]), "width_trials": trials,
            "simplification_trials": [{"terms": len(item["fit_terms"]),
                "loss": item["loss"]} for item in path]}
        entry = _entry(fit, [step], count, cost, index)
        entry["candidate"] = replace(entry["candidate"],
            candidate_id=f"temporal_coordinate:{index}",
            rationale="Measured intermediate terms jointly identify response and temporal support")
        claims.append(entry)
    return claims
