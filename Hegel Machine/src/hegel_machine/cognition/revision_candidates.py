"""Compare executable explanations and design measurements that separate them."""

from itertools import combinations
from math import sqrt
from random import Random
from statistics import mean, variance

from .continuous_state import _linear_fit, _sum_terms
from .decision_experiments import (fit_objective, objective_pool, objective_prediction,
                                   point_distance, select_objective_controls)
from .evaluation import loss
from .language import Program
from .meta_prior_claims import compile_claims
from .models import Candidate, Observation


def _claims(spec, rows, budget):
    claims = compile_claims(spec, rows, budget)
    # A noisy full quadratic need not have exactly zero nuisance coefficients.
    # Test each measured coordinate as a sufficient nonlinear explanation.
    for name in spec.feature_names:
        feature = Program("feature", name=name)
        terms = [feature, Program("mul", children=(feature, feature))]
        if not budget.spend(4 * len(rows)):
            break
        columns = [[term.predict(row) for row in rows] for term in terms]
        coefficients = _linear_fit(columns, [row.target for row in rows])
        program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
        claims.append({"family_id": "T05", "kind": "single_coordinate_quadratic",
            "retained_feature": name, "terms": terms,
            "candidate": Candidate(f"univariate:{name}", program, "meta_prior_revision",
                "One measured coordinate may suffice after a nonlinear readout"),
            "claim": "One measured coordinate may suffice after a nonlinear readout",
            "counter_prediction": {"type": "omitted_coordinates",
                "features": [item for item in spec.feature_names if item != name],
                "retained_feature": name,
                "condition": "Other coordinates do not change the response at a fixed retained coordinate"},
            "training_loss": loss(program, rows, mean(row.target for row in rows), spec.risk_quantile)})
    for claim in claims:
        claim["method_id"] = ":".join((claim["kind"], str(len(claim["terms"])),
                                          claim.get("retained_feature", "")))
    return claims


def propose_competitors(spec, rows, budget, *, limit=3):
    """Rank fitting procedures by leave-one-out forecasts of already paid rows.

    Recompile each fold: variable selection and discovered coordinates cannot
    inherit the held-out target. A method missing in a fold has incomplete
    evidence and does not enter the frozen competition. The later actual
    measurements are separate from this internal comparison.
    """
    full = _claims(spec, rows, budget)
    if not full:
        return []
    errors = {claim["method_id"]: [] for claim in full}
    for index, row in enumerate(rows):
        fitted = _claims(spec, rows[:index] + rows[index + 1:], budget)
        for claim in fitted:
            key = claim["method_id"]
            if key in errors and budget.spend(1):
                errors[key].append((claim["candidate"].program.predict(row) - row.target) ** 2)
        if not budget.remaining:
            break
    ranked = []
    for claim in full:
        heldout = errors[claim["method_id"]]
        if len(heldout) != len(rows):
            continue
        ranked.append({"candidate_id": claim["method_id"],
            **{key: claim[key] for key in ("family_id", "kind", "claim", "counter_prediction", "training_loss")},
            "program": claim["candidate"].program.to_dict(),
            "cv_loss": mean(heldout), "cv_errors": heldout,
            "cv_standard_error": sqrt(variance(heldout) / len(heldout)) if len(heldout) > 1 else 0.})
    ranked.sort(key=lambda item: (item["cv_loss"], Program.from_dict(item["program"]).size))
    chosen, programs = [], set()
    for item in ranked:
        program = Program.from_dict(item["program"])
        if program not in programs:
            programs.add(program)
            chosen.append(item)
            if len(chosen) == limit:
                break
    return chosen


def distinguishing_controls(spec, candidates, budget, *, seed, count, decision_rows=(), details=None):
    """Measure candidate disagreements and useful opportunities for the goal."""
    rng = Random(seed)
    bounds = spec.feature_bounds
    pool = [{name: rng.uniform(*bounds[name]) for name in spec.feature_names} for _ in range(64)]
    midpoint = {name: sum(bounds[name]) / 2 for name in spec.feature_names}
    pool.extend({**midpoint, name: edge} for name in spec.feature_names for edge in bounds[name])
    pool.extend({name: bounds[name][side] for name in spec.feature_names} for side in (0, 1))
    starting_cost = budget.spent
    objective = fit_objective(spec, decision_rows, budget)
    if objective:
        pool.extend(objective_pool(spec, objective))
    programs = [Program.from_dict(item["program"]) for item in candidates]
    scored = []
    for point in pool:
        if any(point == item["point"] for item in scored):
            continue
        if not budget.spend(len(programs)):
            break
        values = [program.predict(Observation(point, 0.)) for program in programs]
        disagreement = max(((a - b) ** 2 for a, b in combinations(values, 2)), default=0.)
        goal = objective_prediction(spec, objective, point, budget) if objective else None
        scored.append({"point": point, "disagreement": disagreement,
                       "predictions": values, "objective": goal})
    planned = select_objective_controls(spec, scored, objective, count)
    selected = [item["point"] for item, _ in planned]
    scored = [item for item in scored if item["point"] not in selected]
    while scored and len(selected) < count:
        def utility(item):
            distance = point_distance(spec, item["point"], selected)
            return item["disagreement"] * distance, distance
        item = max(scored, key=utility)
        selected.append(item["point"])
        planned.append((item, "hypothesis_disagreement"))
        scored.remove(item)
    if details is not None:
        details.update(objective_model={key: value for key, value in objective.items() if key != "folds"}
                       if objective else None,
                       compute_cost=budget.spent - starting_cost,
                       selected=[{**item, "purpose": purpose} for item, purpose in planned])
    return selected


def select_competitor(candidates, rows, *, noise_variance, minimum_error, maximum_loss):
    """Use new paired errors, then CV evidence to resolve uncertain differences."""
    measured = []
    for candidate in candidates:
        key = candidate["candidate_id"]
        if not all(key in row.get("candidate_predictions", {}) for row in rows):
            continue
        predictions = [row["candidate_predictions"][key] for row in rows]
        errors = [(value - row["observation"]["target"]) ** 2 for value, row in zip(predictions, rows)]
        measured.append({**candidate, "validation_loss": mean(errors),
                         "validation_errors": errors, "validation_predictions": predictions})
    eligible = [item for item in measured if max(item["training_loss"], item["validation_loss"]) <= maximum_loss]
    if not eligible:
        return None, measured
    best = min(eligible, key=lambda item: item["validation_loss"])
    plausible = []
    for item in eligible:
        gains = [a - b for a, b in zip(item["validation_errors"], best["validation_errors"])]
        n = len(gains)
        sampling = variance(gains) / n if n > 1 else 0.
        noise = sum(4 * (a - b) ** 2 * noise_variance for a, b in
                    zip(item["validation_predictions"], best["validation_predictions"])) / n ** 2
        item["comparison_radius"] = 2 * sqrt(max(sampling, noise))
        if mean(gains) <= item["comparison_radius"] + minimum_error:
            plausible.append(item)
    chosen = min(plausible, key=lambda item: (item["cv_loss"], Program.from_dict(item["program"]).size))
    return chosen, measured
