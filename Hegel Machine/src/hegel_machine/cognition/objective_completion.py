"""Finish a goal experiment while its remaining observations are still usable."""

from copy import deepcopy
from itertools import combinations
from statistics import mean

from .decision_experiments import _predict
from .models import SearchBudget
from .validation_continuation import evidence_summary


def validation_compute_reserve(spec, row_count, remaining_calls=4):
    """Price one public pool and four complete future validation scans."""
    dimension = len(spec.feature_names)
    pool = 1 + 33 * dimension + row_count + 4
    return 5 * pool + dimension ** 3 + 8 + max(0, remaining_calls - 4)


def fit_compute_reserve(spec, row_count):
    """Bound design, full fits, and five whole-control validation folds."""
    dimension = len(spec.feature_names)
    widths = [dimension + 1, 2 * dimension + 1]
    if dimension > 1:
        widths.append(2 * dimension + 1 + dimension * (dimension - 1) // 2)
    return row_count * (5 * sum(widths) + len(widths) + widths[-1])


def _remaining_identification(spec, controls, readiness):
    """Lower bound on new controls for the next unidentifiable representation."""
    dimension = len(spec.feature_names)
    widths = [dimension + 1, 2 * dimension + 1]
    if dimension > 1:
        widths.append(2 * dimension + 1 + dimension * (dimension - 1) // 2)
    missing = [width for width, ready in zip(widths, readiness or []) if not ready]
    if not missing:
        return None
    count, width = len(controls), min(missing)
    required = count
    while required - (required + 4) // 5 < width:
        required += 1
    return max(1, required - count)


def _paid_improvement(spec, rows, model, budget):
    """Compare stored held-out predictions with the same fold's constant fit."""
    if not budget.spend(2 * len(rows)):
        return None
    keys = [tuple(row["observation"]["features"][name] for name in spec.feature_names)
            for row in rows]
    groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
    targets = [float(row["context"]["objective"]) for row in rows]
    baselines = {group: mean(target for key, target in zip(keys, targets)
                            if groups[key] != group) for group in set(groups.values())}
    by_control = {}
    for key, row, target in zip(keys, rows, targets):
        group = groups[key]
        prediction = _predict(spec, model["folds"][group], row["observation"]["features"])
        by_control.setdefault(key, []).append({
            "candidate_error": (prediction - target) ** 2,
            "previous_error": (baselines[group] - target) ** 2})
    return evidence_summary([{name: mean(row[name] for row in values)
                              for name in ("candidate_error", "previous_error")}
                             for values in by_control.values()])


def _readiness(spec, rows, keys, budget):
    import numpy as np

    dimension = len(spec.feature_names)
    widths = [dimension + 1, 2 * dimension + 1]
    if dimension > 1:
        widths.append(2 * dimension + 1 + dimension * (dimension - 1) // 2)
    if not budget.spend(len(rows) * widths[-1]):
        return None
    design = []
    for row in rows:
        point = row["observation"]["features"]
        values = [(point[name] - spec.feature_bounds[name][0]) /
                  (spec.feature_bounds[name][1] - spec.feature_bounds[name][0])
                  for name in spec.feature_names]
        design.append([1., *values, *(value * value for value in values),
                       *(values[i] * values[j] for i, j in combinations(range(dimension), 2))])
    groups = {tuple(key): index % 5 for index, key in enumerate(keys)}
    assigned = [groups[tuple(row["observation"]["features"][name]
                             for name in spec.feature_names)] for row in rows]
    matrix = np.asarray(design)
    return [all(np.linalg.matrix_rank(matrix[np.asarray(assigned) != group, :width]) == width
                for group in set(assigned)) for width in widths]


def objective_completion(spec, rows, budget, *, fit, reference_loss,
                         remaining_calls, state):
    """Assess paid evidence before buying another identification control.

    ``fit`` is the caller's grouped-CV candidate fitter. ``state`` belongs to
    this collecting episode and avoids fitting the same paid prefix twice.
    Reference loss is the frozen pre-trigger model's error scale, not an
    incumbent's error against these prospective validation labels.

    A low-order model may start future validation when its held-out error is
    already within the existing monitoring tolerance. When richer comparison
    cannot finish before the validation deadline, positive paid improvement may
    instead justify testing an imperfect candidate on future labels. No model
    is adopted by this helper.
    """
    indices = [row["context"]["global_index"] for row in rows]
    controls = list(dict.fromkeys(tuple(row["observation"]["features"][name]
                                       for name in spec.feature_names) for row in rows))
    control_keys = [list(point) for point in controls]
    folds = min(5, len(controls))
    validation_reserve = validation_compute_reserve(spec, len(rows), remaining_calls)
    next_validation_reserve = validation_compute_reserve(spec, len(rows) + 1, remaining_calls - 1)
    next_fit_reserve = fit_compute_reserve(spec, len(rows) + 1)
    threshold = 9 * max(float(reference_loss), 1e-7)
    result = {"candidate": None, "ready": False, "need_interactions": False,
        "status": "needs_identification", "error_threshold": threshold,
        "validation_reserve": validation_reserve,
        "next_validation_reserve": next_validation_reserve,
        "next_fit_reserve": next_fit_reserve, "fit_compute": 0, "readiness_compute": 0,
        "identification_allowance": 0}
    if remaining_calls < 4:
        return {**result, "status": "insufficient_validation_calls"}
    candidate = deepcopy(state.get("model"))
    # Replicates cannot make an otherwise rank-deficient held-out fit possible.
    identifiable = (folds >= 3 and len(controls) - (len(controls) + folds - 1) // folds
                    >= len(spec.feature_names) + 1)
    affordable_fit = budget.remaining >= validation_reserve + fit_compute_reserve(spec, len(rows))
    readiness = state.get("checked_readiness")
    if identifiable and state.get("checked_controls") != control_keys:
        if not affordable_fit:
            if candidate is None:
                return {**result, "status": "insufficient_completion_compute"}
            # The cached candidate can still face future labels. Do not spend
            # its validation reserve on a fresh comparison we cannot finish.
        else:
            dimension = len(spec.feature_names)
            design_cost = len(rows) * (2 * dimension + 1 + dimension * (dimension - 1) // 2)
            checking = SearchBudget(design_cost)
            readiness = _readiness(spec, rows, control_keys, checking)
            budget.spend(checking.spent)
            result["readiness_compute"] = checking.spent
            state.update(checked_controls=control_keys, checked_readiness=readiness)
    refit = (identifiable and affordable_fit and ("fit_control_count" not in state
             or readiness != state.get("fit_readiness")
             or len(controls) >= state.get("fit_control_count", 0) + 4))
    if refit:
        work = SearchBudget(max(0, budget.remaining - validation_reserve))
        try:
            fitted = fit(spec, rows, work)
        finally:
            budget.spend(work.spent)
        result["fit_compute"] = work.spent
        # An unsuccessful richer fit does not erase the earlier hypothesis.
        # Its held-out folds still belong only to the prefix that fitted it.
        state.update(fit_readiness=readiness, fit_control_count=len(controls))
        if fitted is not None:
            candidate = fitted
            state.update(training_indices=indices, model=deepcopy(candidate))
            state.pop("residual_evidence", None)
        state.setdefault("fits", []).append({"training_indices": indices,
            "compute_cost": work.spent,
            "degree": fitted["degree"] if fitted else None,
            "interactions": deepcopy(fitted.get("interactions", [])) if fitted else [],
            "cv_loss": fitted["cv_loss"] if fitted else None})
    if candidate is not None:
        adequate = (candidate["cv_loss"] <= threshold
                    and candidate["cv_loss"] < candidate["reference_loss"])
        compared = bool(candidate.get("interaction_comparison"))
        relational = bool(candidate.get("interactions"))
        if (adequate or compared or relational) and budget.remaining >= validation_reserve:
            result.update(candidate=deepcopy(candidate), ready=True,
                status="adequate_paid_residual" if adequate else "representations_compared")
        else:
            result.update(need_interactions=True, status="unresolved_low_order_residual")
            needed = _remaining_identification(spec, controls, readiness)
            limited_by = []
            if needed is not None and needed + 4 > remaining_calls:
                limited_by.append("observations")
            richer_rows = len(rows) + (needed or 0)
            richer_reserve = (fit_compute_reserve(spec, richer_rows)
                + validation_compute_reserve(spec, richer_rows, remaining_calls - (needed or 0)))
            if not affordable_fit or (needed is not None and budget.remaining < richer_reserve):
                limited_by.append("compute")
            if limited_by and readiness and readiness[0]:
                result["identification_calls_required"] = needed
                result["completion_limited_by"] = limited_by
                if "residual_evidence" not in state:
                    fitted_indices = set(state["training_indices"])
                    fitted_rows = [row for row in rows
                                   if row["context"]["global_index"] in fitted_indices]
                    if budget.remaining >= validation_reserve + 2 * len(fitted_rows):
                        before = budget.spent
                        state["residual_evidence"] = _paid_improvement(
                            spec, fitted_rows, candidate, budget)
                        result["fit_compute"] += budget.spent - before
                evidence = state.get("residual_evidence")
                result["residual_evidence"] = deepcopy(evidence)
                if (evidence and evidence["mean_gain"] > 1e-7
                        and budget.remaining >= validation_reserve):
                    result.update(candidate=deepcopy(candidate), ready=True,
                        status="budgeted_low_order_validation", residual_unresolved=True)
    elif readiness and readiness[1]:
        result["need_interactions"] = True
    if not result["ready"] and remaining_calls > 4:
        result["identification_allowance"] = max(
            0, budget.remaining - next_validation_reserve - next_fit_reserve)
    return result
