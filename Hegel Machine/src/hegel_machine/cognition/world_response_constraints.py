"""Finite, response-constrained refits of an already declared expression basis.

A successful setting is an inequality, not an equality label for the latent
program.  Each LP jointly fits that program and a shared connected activation
band.  Negative settings choose which side of the band they occupy; observed
positive settings fix that choice, otherwise we enumerate their ordered gaps.
The target search is conditional on an untried reachable activation setting.
Failure here does not exclude models whose activation lies outside the grid.
"""

from itertools import islice, product
from math import ceil, floor, inf, prod

import numpy as np
from scipy.optimize import linprog

from .language import Program
from .models import Observation


BRANCH_LIMIT = 128


def _target_gaps(negatives, observed, grid):
    step, origin = grid["step"], grid["origin"]
    bounds = [-inf, *negatives, inf]
    gaps = []
    for left, right in zip(bounds, bounds[1:]):
        first = ceil((grid["lower"] - origin) / step)
        last = floor((grid["upper"] - origin) / step)
        if left != -inf:
            first = max(first, floor((left - origin) / step) + 1)
        if right != inf:
            last = min(last, ceil((right - origin) / step) - 1)
        while first <= last and origin + step * first in observed:
            first += 1
        while first <= last and origin + step * last in observed:
            last -= 1
        if first <= last:
            gaps.append((left, right, first, last))
    return gaps


def fit_response_constraints(features, responses, estimators, target_number, grid,
                             seed_programs):
    """Return at most two exactly checked representative models per basis.

    ``seed_programs`` maps a family id to its prior ``Program``; a seed guides
    representative selection but is not a coefficient bound. Search records
    include the branch limit, solver outcomes and executable verification
    failures. ``compute_cost`` counts basis evaluations, LP constraint rows
    submitted and executable response checks, not wall time or FLOPs.
    """
    result = {"models": [], "search": [], "compute_cost": 0}
    by_crystal = {}
    for item in responses:
        by_crystal.setdefault(item["crystal"], []).append(item)
    positives = {key: sorted({item["frequency"] for item in values if item["activated"]})
                 for key, values in by_crystal.items()}
    negatives = {key: sorted({item["frequency"] for item in values if not item["activated"]})
                 for key, values in by_crystal.items()}
    conflicts = [item for key, values in by_crystal.items() if positives[key]
                 for item in values if not item["activated"]
                 and positives[key][0] <= item["frequency"] <= positives[key][-1]]
    observed = {item["frequency"] for item in by_crystal.get(target_number, [])}
    target_gaps = _target_gaps(negatives.get(target_number, []), observed, grid)
    other_gaps = {key: list(zip([-inf, *negatives[key]], [*negatives[key], inf]))
                  for key in by_crystal if not positives[key] and key != target_number}
    numbers = sorted(features)
    frequency_scale = max(1., abs(grid["lower"]), abs(grid["upper"]),
                          *(abs(item["frequency"]) for item in responses))
    for estimator in estimators:
        family = estimator["id"]
        audit = {"family": family, "status": "unavailable", "branch_count": 0,
                 "branches_attempted": 0, "branches_feasible": 0, "lp_calls": 0,
                 "lp_iterations": 0, "solver_statuses": {}, "verification_failures": 0,
                 "branch_limit": BRANCH_LIMIT, "limited_by_branch_budget": False,
                 "scope": "reachable_untried_target_support_in_declared_basis"}
        result["search"].append(audit)
        if conflicts:
            audit.update(status="inconsistent_response_band", conflicts=conflicts)
            continue
        if not any(positives.values()):
            audit["reason"] = "positive_response_needed"
            continue
        terms = [Program.from_dict(term) for term in estimator["basis"]]
        required = set().union(*(term.required_features() for term in terms))
        if target_number not in features or any(key not in features for key in by_crystal) \
                or any(not required <= set(features[key]) for key in numbers):
            audit["reason"] = "missing_public_measurements"
            continue
        raw = np.array([[term.predict(Observation(features[key], 0.)) for term in terms]
                        for key in numbers], dtype=float).reshape(len(numbers), len(terms))
        result["compute_cost"] += len(numbers) * len(terms)
        means = raw.mean(axis=0)
        scales = np.max(abs(raw - means), axis=0)
        scales[scales == 0.] = 1.
        design = (raw - means) / scales
        rows = dict(zip(numbers, design))
        width = len(terms)
        lower_index, upper_index, margin_index, setting_index = range(width, width + 4)
        dimensions = width + 4
        seed = seed_programs.get(family)
        if seed is not None:
            seeds = {key: seed.predict(Observation(features[key], 0.)) for key in numbers}
            residuals = [item["frequency"] - seeds[item["crystal"]]
                         for item in responses if item["activated"]]
            seed_offset = (min(residuals) + max(residuals)) / 2.
            desired = {key: (value + seed_offset) / frequency_scale for key, value in seeds.items()}
        else:
            desired = None
        desired_setting = (desired[target_number] * frequency_scale if desired else
                           (grid["lower"] + grid["upper"]) / 2.)
        gaps = sorted(target_gaps, key=lambda gap: max(
            grid["origin"] + grid["step"] * gap[2] - desired_setting,
            desired_setting - grid["origin"] - grid["step"] * gap[3], 0.))
        audit["branch_count"] = len(gaps) * prod(len(values) for values in other_gaps.values())
        audit["limited_by_branch_budget"] = audit["branch_count"] > BRANCH_LIMIT
        base_a, base_b = [], []

        def constraint(number, setting, boundary, sign, strict=False):
            row = np.zeros(dimensions)
            row[:width] = sign * rows[number]
            row[boundary] = sign
            if strict:
                row[margin_index] = 1.
            return row, sign * setting / frequency_scale

        for number, values in by_crystal.items():
            for item in values:
                if item["activated"]:
                    for boundary, sign in ((lower_index, 1.), (upper_index, -1.)):
                        a, b = constraint(number, item["frequency"], boundary, sign)
                        base_a.append(a); base_b.append(b)
                elif positives[number]:
                    below = item["frequency"] < positives[number][0]
                    a, b = constraint(number, item["frequency"],
                        lower_index if below else upper_index, -1. if below else 1., True)
                    base_a.append(a); base_b.append(b)
        order = np.zeros(dimensions)
        order[lower_index], order[upper_index] = 1., -1.
        base_a.append(order); base_b.append(0.)
        # The continuous witness is constrained to activate.  Its nearest grid
        # neighbours are subsequently solved as exact physical settings.
        for boundary, sign in ((lower_index, 1.), (upper_index, -1.)):
            a, b = constraint(target_number, 0., boundary, sign)
            a[setting_index] = -sign
            base_a.append(a); base_b.append(b)
        feasible_models = []
        branches = product(gaps, *(other_gaps.values()))
        for branch in islice(branches, BRANCH_LIMIT):
            audit["branches_attempted"] += 1
            gap, *others = branch
            assignments = {target_number: gap[:2], **dict(zip(other_gaps, others))}
            a, b = list(base_a), list(base_b)
            for number, (left, _) in assignments.items():
                if positives.get(number):
                    continue
                for setting in negatives.get(number, []):
                    below = setting <= left
                    row, bound = constraint(number, setting,
                        lower_index if below else upper_index, -1. if below else 1., True)
                    a.append(row); b.append(bound)
            matrix, rhs = np.asarray(a), np.asarray(b)
            first = grid["origin"] + grid["step"] * gap[2]
            last = grid["origin"] + grid["step"] * gap[3]
            bounds = [(None, None)] * (width + 2) + [(0., 1.),
                         (first / frequency_scale, last / frequency_scale)]
            objective = np.zeros(dimensions)
            objective[margin_index] = -1.

            def solve(cost, a_ub=matrix, b_ub=rhs, variable_bounds=bounds):
                fit = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=variable_bounds,
                              method="highs", options={"dual_feasibility_tolerance": 1e-9,
                                                       "primal_feasibility_tolerance": 1e-9})
                audit["lp_calls"] += 1
                audit["lp_iterations"] += int(fit.nit or 0)
                status = str(fit.status)
                audit["solver_statuses"][status] = audit["solver_statuses"].get(status, 0) + 1
                result["compute_cost"] += len(b_ub)
                return fit

            fit = solve(objective)
            if not fit.success or fit.x[margin_index] <= 0.:
                continue
            continuous_setting = fit.x[setting_index] * frequency_scale
            index = (continuous_setting - grid["origin"]) / grid["step"]
            witnesses = {min(gap[3], max(gap[2], index_value))
                         for index_value in (floor(index), ceil(index),
                             round((desired_setting - grid["origin"]) / grid["step"]))}
            branch_models = []
            for witness in sorted(witnesses, key=lambda value: abs(
                    grid["origin"] + grid["step"] * value - desired_setting)):
                setting = grid["origin"] + grid["step"] * witness
                if setting in observed:
                    continue
                fixed_bounds = list(bounds)
                fixed_bounds[setting_index] = (setting / frequency_scale,) * 2
                fixed = solve(objective, variable_bounds=fixed_bounds)
                if not fixed.success or fixed.x[margin_index] <= 0.:
                    continue
                margin = fixed.x[margin_index]
                fixed_bounds[margin_index] = (margin / 2., 1.)
                # L1 closeness to previous predicted activation centres (or L1
                # coefficients when no seed exists) chooses a representative.
                goals = [(np.r_[rows[key], .5, .5, 0., 0.], desired[key])
                         for key in numbers] if desired else [
                    (np.eye(dimensions)[index], 0.) for index in range(width)]
                expanded = np.pad(matrix, ((0, 0), (0, len(goals))))
                extra_a, extra_b = [], []
                for index, (row, value) in enumerate(goals):
                    for sign in (1., -1.):
                        extra = np.r_[sign * row, np.zeros(len(goals))]
                        extra[dimensions + index] = -1.
                        extra_a.append(extra); extra_b.append(sign * value)
                cost = np.r_[np.zeros(dimensions), np.ones(len(goals))]
                chosen = solve(cost, np.vstack([expanded, *extra_a]) if extra_a else expanded,
                    np.r_[rhs, extra_b], fixed_bounds + [(0., None)] * len(goals))
                if not chosen.success:
                    continue
                slopes = chosen.x[:width] * frequency_scale / scales
                program = Program("const", value=-float(np.dot(slopes, means)))
                for term, slope in zip(terms, slopes):
                    if slope != 0.:
                        program = Program("add", children=(program, Program("mul", children=(
                            Program("const", value=float(slope)), term))))
                program = program.simplified()
                predictions = {key: program.predict(Observation(features[key], 0.)) for key in numbers}
                actual_positive = [item["frequency"] - predictions[item["crystal"]]
                                   for item in responses if item["activated"]]
                actual_positive.append(setting - predictions[target_number])
                lower, upper = min(actual_positive), max(actual_positive)
                actual_negative = [item["frequency"] - predictions[item["crystal"]]
                                   for item in responses if not item["activated"]]
                # Re-express the fitted band using executable residuals.  This
                # prevents floating-point Program evaluation from dropping a
                # positive boundary; negatives still receive exact checks.
                below = max((value for value in actual_negative if value < lower), default=None)
                above = min((value for value in actual_negative if value > upper), default=None)
                if below is not None:
                    lower = (lower + below) / 2.
                if above is not None:
                    upper = (upper + above) / 2.
                result["compute_cost"] += len(numbers) + len(responses)
                if any((lower <= item["frequency"] - predictions[item["crystal"]] <= upper)
                       != bool(item["activated"]) for item in responses):
                    audit["verification_failures"] += 1
                    continue
                model = {"family": family, "program": program, "lower": lower, "upper": upper,
                         "predictions_by_crystal": {str(key): value for key, value in predictions.items()},
                         "target_setting": setting, "fit_provenance": {
                             "origin": "joint_response_inequality_lp", "branch": audit["branches_attempted"],
                             "negative_gaps": {str(key): [left if left != -inf else None,
                                 right if right != inf else None] for key, (left, right) in assignments.items()},
                             "solver_status": "optimal", "response_count": len(responses),
                             "strict_margin": margin * frequency_scale,
                             "representative_distance": float(chosen.fun),
                             "selection": "nearest_seed_centres" if desired else "minimum_normalized_slope_l1"}}
                branch_models.append(model)
                break
            if branch_models:
                audit["branches_feasible"] += 1
                feasible_models.extend(branch_models)
        if feasible_models:
            best = min(feasible_models, key=lambda item: item["fit_provenance"]["representative_distance"])
            retained = [best]
            alternative = max(feasible_models, key=lambda item: abs(item["target_setting"] - best["target_setting"]))
            if alternative["target_setting"] != best["target_setting"]:
                retained.append(alternative)
            result["models"].extend(retained)
            audit["models_retained"] = len(retained)
        uncertain = audit["limited_by_branch_budget"] or audit["verification_failures"] or any(
            status not in {"0", "2"} for status in audit["solver_statuses"])
        audit["status"] = "incomplete" if uncertain else "available" if feasible_models else "infeasible"
    return result
