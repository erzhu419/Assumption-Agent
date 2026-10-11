"""Finite, response-constrained refits of an already declared expression basis.

A successful setting is an inequality, not an equality label for the latent
program.  Each LP jointly fits that program and a shared connected activation
band.  Negative settings choose which side of the band they occupy; observed
positive settings fix that choice, otherwise we enumerate their ordered gaps.
The legacy target refit is conditional on an untried reachable activation
setting. The family query instead checks history plus explicit hypothetical
responses, including models whose activation lies outside a target grid.
"""

from itertools import islice, product
from fractions import Fraction
from math import ceil, floor, inf, isfinite, prod

import numpy as np
from scipy.optimize import linprog

from .language import Program
from .models import Observation


BRANCH_LIMIT = 128


def _response_extremes(raw_rows, values, *, minimum=True, maximum=True):
    """Keep the strongest residual bounds for each identical basis vector.

    Subtracting the same model prediction preserves frequency order. Positive
    responses therefore need only their two extremes; a below-band negative
    needs only the largest setting, and an above-band negative the smallest.
    The original observations remain available for executable witness checks.
    """
    groups = {}
    for item in values:
        key = tuple(raw_rows[item["crystal"]])
        low, high = groups.setdefault(key, (item, item))
        groups[key] = (item if item["frequency"] < low["frequency"] else low,
                       item if item["frequency"] > high["frequency"] else high)
    extremes = []
    for low, high in groups.values():
        if minimum:
            extremes.append(low)
        if maximum and (not minimum or high["frequency"] != low["frequency"]):
            extremes.append(high)
    return extremes


def _excluded_low_dimensional_branch(raw_rows, positives, below, above, comparison_limit=None):
    """Exactly eliminate band bounds and up to two varying basis coefficients.

    Below-band residuals must be strictly smaller than every positive residual;
    positive residuals must be strictly smaller than above-band residuals. With
    no positives the below/above pairs still constrain the possible band. Each
    comparison is a strict linear inequality on the slopes. With two slopes,
    Fourier--Motzkin elimination removes one before intersecting intervals for
    the other. There are no coefficient or bandwidth limits. Fractions represent
    the supplied floating-point basis values exactly. Each original comparison
    and each elimination pair consumes one unit of the supplied allowance.
    """
    vectors = list(raw_rows.values())
    varying = [index for index in range(len(vectors[0]))
               if any(row[index] != vectors[0][index] for row in vectors[1:])]
    if len(varying) > 2:
        return None, 0, True
    inequalities = {}
    slope_lower = slope_upper = None
    comparisons = 0

    def excludes_interval(coefficient, bound):
        nonlocal slope_lower, slope_upper
        if not coefficient:
            return bound <= 0
        cutoff = bound / coefficient
        if coefficient > 0:
            slope_upper = cutoff if slope_upper is None else min(slope_upper, cutoff)
        else:
            slope_lower = cutoff if slope_lower is None else max(slope_lower, cutoff)
        return slope_lower is not None and slope_upper is not None and slope_lower >= slope_upper

    # With any positive residual, below < positive < above already implies
    # below < above. Repeating those pairs used to exhaust the query allowance
    # on long histories before reaching another negative-response branch.
    pairs = ((below, positives), (positives, above)) if positives else ((below, above),)
    for lefts, rights in pairs:
        for left in lefts:
            for right in rights:
                if comparison_limit is not None and comparisons >= comparison_limit:
                    return None, comparisons, False
                comparisons += 1
                coefficients = tuple(Fraction(float(raw_rows[right["crystal"]][column]))
                                     - Fraction(float(raw_rows[left["crystal"]][column]))
                                     for column in varying)
                bound = Fraction(float(right["frequency"])) - Fraction(float(left["frequency"]))
                if len(varying) < 2:
                    if excludes_interval(coefficients[0] if coefficients else Fraction(0), bound):
                        return True, comparisons, True
                else:
                    # Duplicate inequalities need only their strongest bound.
                    inequalities[coefficients] = min(bound, inequalities.get(coefficients, bound))
    if len(varying) < 2:
        return False, comparisons, True
    # Eliminate the cheaper coordinate. The intermediate size is at most the
    # number of positive/negative coefficient pairs, bounded by the allowance.
    pivot = min(range(2), key=lambda column:
                sum(row[column] > 0 for row in inequalities)
                * sum(row[column] < 0 for row in inequalities))
    positive, negative = [], []
    for coefficients, bound in inequalities.items():
        coefficient, remaining = coefficients[pivot], coefficients[1 - pivot]
        if not coefficient:
            if excludes_interval(remaining, bound):
                return True, comparisons, True
        else:
            (positive if coefficient > 0 else negative).append(
                (remaining / abs(coefficient), bound / abs(coefficient)))
    for upper_coefficient, upper_bound in positive:
        for lower_coefficient, lower_bound in negative:
            if comparison_limit is not None and comparisons >= comparison_limit:
                return None, comparisons, False
            comparisons += 1
            if excludes_interval(upper_coefficient + lower_coefficient, upper_bound + lower_bound):
                return True, comparisons, True
    return False, comparisons, True


def query_response_family(features, responses, estimator, queries=(), branch_limit=BRANCH_LIMIT,
                          compute_limit=None):
    """Ask whether a declared response family permits explicit consequences.

    ``queries`` are hypothetical ``{crystal, frequency, activated}`` conditions,
    never newly observed evidence. An empty query checks the paid history alone,
    without requiring a positive response or a reachable target setting.

    SAT has an executable witness. UNSAT is restricted to exact equal-basis
    connected-band conflicts or exhaustive exact elimination for at most two
    varying terms. In higher dimensions an LP failure is UNKNOWN. A truncated
    search can establish SAT with a checked witness, but cannot establish UNSAT.
    ``compute_limit`` bounds basis evaluations, endpoint-reduction scans, exact
    comparisons, LP rows and executable checks; work is charged before execution.
    Dominated same-basis response bounds are removed before comparisons and LPs;
    every original response is still checked against a returned witness.
    """
    result = {"status": "UNKNOWN", "witness": None,
              "scope": "declared_basis_shared_connected_band",
              "exclusion_scope": "equal_basis_conflict_or_up_to_two_varying_terms_exact_elimination",
              "basis": estimator.get("basis", []), "family": estimator.get("id"),
              "query_role": "hypothetical_conditions_not_observed_evidence",
              "queries": [dict(item) for item in queries],
              "paid_response_count": len(responses), "compute_cost": 0,
              "lp_calls": 0, "lp_iterations": 0, "solver_statuses": {},
              "branch_count": 0, "branches_attempted": 0,
              "branches_excluded_exactly": 0, "exact_comparisons": 0,
              "constraint_rows_scanned": 0,
              "branch_limit": branch_limit, "limited_by_branch_budget": False,
              "verification_failures": 0, "compute_limit": compute_limit}

    def spend(cost):
        if compute_limit is not None and result["compute_cost"] + cost > compute_limit:
            result.update(status="UNKNOWN", reason="query_compute_limit")
            return False
        result["compute_cost"] += cost
        return True

    try:
        if not features or "basis" not in estimator or branch_limit < 0:
            raise ValueError("features, declared basis and nonnegative branch_limit are required")
        terms = [Program.from_dict(term) for term in estimator["basis"]]
        required = set().union(*(term.required_features() for term in terms))
        if any(term.history_depth for term in terms):
            raise ValueError("response-family rows do not contain historical observations")
        if any(not required <= set(values) for values in features.values()):
            raise ValueError("missing_public_measurements")
        observations = [dict(item) for item in responses] + result["queries"]
        if any(item["crystal"] not in features or not isfinite(item["frequency"])
               or not isinstance(item["activated"], bool) for item in observations):
            raise ValueError("response conditions need measured objects, finite settings and Boolean outcomes")
        numbers = list(features)
        if not spend(len(numbers) * len(terms)):
            return result
        raw = np.array([[term.predict(Observation(features[key], 0.)) for term in terms]
                        for key in numbers], dtype=float).reshape(len(numbers), len(terms))
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite_basis_evaluation")
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result.update(status="INVALID_BINDING", reason=str(exc))
        return result
    raw_rows = dict(zip(numbers, raw))
    result["varying_basis_terms"] = sum(any(row[index] != raw[0][index] for row in raw[1:])
                                         for index in range(len(terms)))
    by_representation = {}
    by_crystal = {}
    for item in observations:
        by_representation.setdefault(tuple(raw_rows[item["crystal"]]), []).append(item)
        by_crystal.setdefault(item["crystal"], []).append(item)
    for values in by_representation.values():
        positive_settings = [item["frequency"] for item in values if item["activated"]]
        if positive_settings and any(not item["activated"]
                and min(positive_settings) <= item["frequency"] <= max(positive_settings)
                for item in values):
            result.update(status="UNSAT_VERIFIED", reason="equal_basis_connected_band_conflict")
            return result
    if not spend(len(observations)):
        return result
    result["constraint_rows_scanned"] += len(observations)
    positive_rows = _response_extremes(raw_rows, [item for item in observations if item["activated"]])
    positive_settings = {key: [item["frequency"] for item in values if item["activated"]]
                         for key, values in by_crystal.items()}
    gaps = {key: list(zip([-inf, *settings], [*settings, inf]))
            for key, values in by_crystal.items() if not positive_settings[key]
            for settings in [sorted({item["frequency"] for item in values})]}
    result["branch_count"] = prod(len(values) for values in gaps.values())
    result["limited_by_branch_budget"] = result["branch_count"] > branch_limit
    means = raw.mean(axis=0)
    scales = np.max(abs(raw - means), axis=0)
    scales[scales == 0.] = 1.
    rows = dict(zip(numbers, (raw - means) / scales))
    frequency_scale = max([1., *(abs(item["frequency"]) for item in observations)])
    width = len(terms)
    lower_index, upper_index, margin_index = range(width, width + 3)
    for branch in islice(product(*gaps.values()), branch_limit):
        result["branches_attempted"] += 1
        assigned = dict(zip(gaps, branch))
        below, above = [], []
        if not spend(len(observations)):
            return result
        result["constraint_rows_scanned"] += len(observations)
        for item in observations:
            if not item["activated"]:
                settings = positive_settings[item["crystal"]]
                is_below = item["frequency"] < min(settings) if settings \
                    else item["frequency"] <= assigned[item["crystal"]][0]
                (below if is_below else above).append(item)
        below = _response_extremes(raw_rows, below, minimum=False)
        above = _response_extremes(raw_rows, above, maximum=False)
        remaining = None if compute_limit is None else compute_limit - result["compute_cost"]
        excluded, comparisons, finished = _excluded_low_dimensional_branch(
            raw_rows, positive_rows, below, above, comparison_limit=remaining)
        result["exact_comparisons"] += comparisons
        spend(comparisons)
        if not finished:
            result.update(status="UNKNOWN", reason="query_compute_limit")
            return result
        if excluded:
            result["branches_excluded_exactly"] += 1
            continue
        matrix, rhs = [], []
        for values, constraints in ((positive_rows, ((lower_index, 1.), (upper_index, -1.))),
                                   (below, ((lower_index, -1.),)),
                                   (above, ((upper_index, 1.),))):
            for item in values:
                for boundary, sign in constraints:
                    row = np.zeros(width + 3)
                    row[:width] = sign * rows[item["crystal"]]
                    row[boundary] = sign
                    row[margin_index] = 0. if item["activated"] else 1.
                    matrix.append(row); rhs.append(sign * item["frequency"] / frequency_scale)
        row = np.zeros(width + 3)
        row[lower_index], row[upper_index] = 1., -1.
        matrix.append(row); rhs.append(0.)
        objective = np.zeros(width + 3)
        objective[margin_index] = -1.
        if not spend(len(rhs)):
            return result
        fit = linprog(objective, A_ub=np.asarray(matrix), b_ub=np.asarray(rhs),
                      bounds=[(None, None)] * (width + 2) + [(0., 1.)],
                      method="highs", options={"dual_feasibility_tolerance": 1e-9,
                                               "primal_feasibility_tolerance": 1e-9})
        result["lp_calls"] += 1
        result["lp_iterations"] += int(fit.nit or 0)
        status = str(fit.status)
        result["solver_statuses"][status] = result["solver_statuses"].get(status, 0) + 1
        if not fit.success or (below or above) and fit.x[margin_index] <= 0.:
            continue
        if not spend(len(numbers) + len(observations)):
            return result
        slopes = fit.x[:width] * frequency_scale / scales
        program = Program("const", value=-float(np.dot(slopes, means)))
        for term, slope in zip(terms, slopes):
            if slope != 0.:
                program = Program("add", children=(program, Program("mul", children=(
                    Program("const", value=float(slope)), term))))
        program = program.simplified()
        predictions = {key: program.predict(Observation(features[key], 0.)) for key in numbers}
        residual = lambda item: item["frequency"] - predictions[item["crystal"]]
        left = max((residual(item) for item in below), default=-inf)
        right = min((residual(item) for item in above), default=inf)
        if positive_rows:
            lower = min(residual(item) for item in positive_rows)
            upper = max(residual(item) for item in positive_rows)
            if left != -inf:
                lower = (left + lower) / 2.
            if right != inf:
                upper = (upper + right) / 2.
        else:
            lower = upper = ((left / 2. + right / 2.) if isfinite(left) and isfinite(right)
                             else left + max(1., abs(left)) if isfinite(left)
                             else right - max(1., abs(right)) if isfinite(right) else 0.)
        if not isfinite(lower) or not isfinite(upper) or lower > upper or any(
                (lower <= residual(item) <= upper) != item["activated"] for item in observations):
            result["verification_failures"] += 1
            continue
        result.update(status="SAT_WITNESS", witness={"program": program.to_dict(),
            "lower": lower, "upper": upper,
            "predictions_by_crystal": {str(key): value for key, value in predictions.items()},
            "checked_responses": len(responses), "checked_queries": len(queries)})
        return result
    if not result["limited_by_branch_budget"] and \
            result["branches_excluded_exactly"] == result["branch_count"]:
        result.update(status="UNSAT_VERIFIED", reason="exact_one_dimensional_branch_exclusion"
                      if result["varying_basis_terms"] <= 1 else "exact_two_dimensional_branch_exclusion")
    else:
        result["reason"] = "branch_limit" if result["limited_by_branch_budget"] \
            else "no_executable_witness_and_no_exact_exclusion"
    return result


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
