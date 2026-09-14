"""Compile a finite set of world hypotheses from paid observations.

The ontology identifiers name possible structural assumptions, not established
facts. Fitted programs make their predictions executable; the research session
must challenge them on later observations before retaining a claim.
"""

from itertools import combinations, combinations_with_replacement
from math import sqrt

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import loss
from .language import EvaluationContext, Program
from .models import Candidate


def _multiply(left, right):
    return Program("mul", children=(left, right))


def _basis(vectors, tolerance):
    """Row-space directions, including an independent affine response term."""
    basis = []
    for source in vectors:
        vector = list(source)
        for direction in basis:
            weight = sum(a * b for a, b in zip(vector, direction))
            vector = [a - weight * b for a, b in zip(vector, direction)]
        norm = sqrt(sum(value * value for value in vector))
        if norm > tolerance:
            basis.append([value / norm for value in vector])
    return basis


def compile_claims(spec, observations, budget, *, seed_programs=()):
    """Fit sparse, separable, coupled, quotient and finite-history hypotheses.

    Each newly evaluated coordinate, complete least-squares fit and complete
    prediction comparison spends one unit per applicable row. Cached columns
    are shared across hypotheses. Projection discovery also pays for its small
    matrix decomposition. All returned losses concern training evidence only.

    A seed is an already executable learned computation. Its fitted coefficient
    and nonlinear readout can change while its callable body remains intact.
    """
    rows = list(observations)
    if not rows:
        return []
    names = sorted(set(spec.feature_names).intersection(*(set(row.features) for row in rows)))
    current = [Program("feature", name=name) for name in names]
    targets = [row.target for row in rows]
    count, columns, contexts, results = len(rows), {}, {}, []
    all_indices = tuple(range(count))
    fallback = sum(targets) / count

    def fit(terms, family, kind, claim, counter_prediction, *, indices=None,
            startup=None, extra=None):
        indices = all_indices if indices is None else tuple(indices)
        if not indices:
            return None
        terms = list(dict.fromkeys(terms))
        missing = {term: [i for i in indices if i not in columns.get(term, {})] for term in terms}
        needed = sum(map(len, missing.values())) + len(indices) + count
        if needed > budget.remaining:
            return None
        start = budget.spent
        for term, unseen in missing.items():
            if not unseen:
                continue
            budget.spend(len(unseen))
            cached = columns.setdefault(term, {})
            for index in unseen:
                context = contexts.setdefault(index, EvaluationContext(rows[index]))
                cached[index] = term.predict(rows[index], context=context)
        budget.spend(len(indices))
        coefficients = _linear_fit([[columns[term][i] for i in indices] for term in terms],
                                   [targets[i] for i in indices])
        program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
        if startup is not None:
            depth, initial_program = startup
            program = Program("if", children=(Program("available", lag=depth), program, initial_program))
        budget.spend(count)
        measured = loss(program, rows, fallback, spec.risk_quantile)
        result = {"family_id": family, "kind": kind,
            "candidate": Candidate(f"meta:{kind}:{len(results)}:{budget.spent}", program,
                                   "meta_prior", claim),
            "terms": terms, "claim": claim, "counter_prediction": counter_prediction,
            "training_loss": measured, "compute_cost": budget.spent - start,
            "support_rows": len(indices), "coefficients": coefficients,
            **(extra or {})}
        results.append(result)
        return result

    # Learned calls receive a small fit first; callers can test them without
    # paying for discovery of every raw interaction again.
    for seed in dict.fromkeys(seed_programs):
        if not seed.required_features().issubset(names):
            continue
        eligible = [i for i, row in enumerate(rows) if len(row.history) >= seed.minimum_history_depth]
        if len(eligible) < 3:
            continue
        seed_terms = [seed, _multiply(seed, seed)]
        fit(seed_terms, "T03", "reuse_compiled",
            "A previously learned computation remains a sufficient coordinate for this response",
            {"type": "retained_coordinate", "coordinate": seed.to_dict(),
             "condition": "Changing raw inputs while retaining this coordinate preserves the response"},
            indices=eligible, startup=(seed.minimum_history_depth, Program("const", value=fallback))
            if len(eligible) < count else None,
            extra={"source_program": seed, "basis": [seed]})

    affine = fit(current, "T02", "affine", "An affine response of the measured coordinates is sufficient",
        {"type": "constant_slope", "features": names,
         "condition": "The response increment of one coordinate is independent of its level and other coordinates"})
    # A nonzero fitted coefficient nominates support; omitting it remains a
    # competing simplification and is checked against all acquired outcomes.
    if affine is not None and len(current) > 1:
        ordered = sorted(zip(current, affine["coefficients"][1:]), key=lambda item: -abs(item[1]))
        for width in dict.fromkeys((1, len(current) - 1)):
            retained = [term for term, _ in ordered[:width]]
            omitted = [name for name in names if name not in {term.name for term in retained}]
            fit(retained, "T02", "sparse_affine", "Some measured coordinates can be omitted from the response",
                {"type": "omitted_coordinates", "features": omitted,
                 "condition": "Intervening on omitted coordinates at fixed retained coordinates leaves the response unchanged"})

    squares = [_multiply(term, term) for term in current]
    fit([*current, *squares], "T05", "additive_quadratic",
        "Separate univariate responses suffice without interactions between coordinates",
        {"type": "mixed_difference_zero", "pairs": [list(pair) for pair in combinations(names, 2)],
         "condition": "The effect of changing one coordinate is unchanged when another coordinate changes"})
    pairs = list(combinations_with_replacement(range(len(current)), 2))
    quadratic_terms = [_multiply(current[i], current[j]) for i, j in pairs]
    coupled = fit([*current, *quadratic_terms], "T05", "coupled_quadratic",
        "Pairwise interactions can explain changes that separate responses miss",
        {"type": "quadratic_response", "features": names,
         "condition": "Mixed response differences are constant within the observed control range"})

    if coupled is not None and coupled["training_loss"] > 1e-8:
        # A persistent error can be a change in slope, rather than a smooth
        # higher-order interaction. The possible equality planes come from the
        # public coordinates; their contribution is fitted only from outcomes.
        kink_pairs = list(combinations(range(len(current)), 2))
        kinks = [Program("abs", children=(Program("sub", children=(current[i], current[j])),))
                 for i, j in kink_pairs]
        if kinks:
            fit([*current, *quadratic_terms, *kinks], "T11", "piecewise_coordinate_difference",
                "Observed coordinate differences can define response boundaries with a change in slope",
                {"type": "coordinate_difference_kinks", "features": names,
                 "pairs": [[names[i], names[j]] for i, j in kink_pairs],
                 "condition": "Crossing a proposed equality plane changes the local response slope while preserving continuity"})

    if coupled is not None and len(current) > 1:
        # y = intercept + linear.x + x'Qx. The row space of Q and
        # the affine slope supply the response-relevant subspace. No apparatus
        # family or hidden generating coefficients enter this construction.
        dimension = len(current)
        start = budget.spent
        if budget.spend(dimension ** 3):
            coefficients = coupled["coefficients"]
            matrix = [[0.] * dimension for _ in current]
            for (i, j), coefficient in zip(pairs, coefficients[1 + dimension:]):
                matrix[i][j] = matrix[j][i] = coefficient if i == j else coefficient / 2
            vectors = [*matrix, coefficients[1:1 + dimension]]
            magnitude = max((abs(value) for vector in vectors for value in vector), default=0.)
            directions = _basis(vectors, max(1e-8, 1e-6 * magnitude))
            if 0 < len(directions) < dimension:
                basis = [_sum_terms(current, direction).simplified() for direction in directions]
                units = [[float(i == j) for i in range(dimension)] for j in range(dimension)]
                null_directions = _basis([*directions, *units], 1e-7)[len(directions):]
                projected = [*basis, *[_multiply(a, b) for a, b in combinations_with_replacement(basis, 2)]]
                result = fit(projected, "T03", "sufficient_projection",
                    "The response depends only on a lower-dimensional projection of the measured coordinates",
                    {"type": "projection_nullspace", "features": names, "directions": directions,
                     "null_directions": null_directions,
                     "condition": "Moving inside the inferred projection nullspace leaves the response unchanged"},
                    extra={"basis": basis, "directions": directions, "raw_dimension": dimension,
                           "effective_dimension": len(directions)})
                if result is not None:
                    result["compute_cost"] = budget.spent - start

    # History is a proposed finite sufficiency boundary, never an assertion that
    # arbitrarily remote history is irrelevant. Startup predictions retain an
    # already fitted current-coordinate model on every initial row.
    initial_program = (affine["candidate"].program if affine else Program("const", value=fallback))
    depth_limit = min(4, max((len(row.history) for row in rows), default=0))
    for depth in range(1, depth_limit + 1):
        eligible = [i for i, row in enumerate(rows) if len(row.history) >= depth]
        if len(eligible) < 3:
            continue
        history_terms = []
        for lag in range(1, depth + 1):
            available = set(names).intersection(*(set(rows[i].history[-lag]) for i in eligible))
            history_terms.extend(Program("lag", name=name, lag=lag) for name in sorted(available))
        if not history_terms:
            continue
        fit([*current, *history_terms], "T08", "finite_history",
            f"The latest {depth} recorded input steps suffice for the current response",
            {"type": "finite_memory", "depth": depth, "features": names,
             "condition": "Trajectories with the same retained suffix but different earlier inputs give the same response"},
            indices=eligible, startup=(depth, initial_program), extra={"history_depth": depth})
    return results
