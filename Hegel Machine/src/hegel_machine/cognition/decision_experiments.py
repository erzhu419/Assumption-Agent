"""Use paid objective observations to propose useful validation experiments."""

from math import erf, exp, pi, sqrt
from statistics import mean

from .continuous_state import _linear_fit


def _design(spec, point, degree, interactions=()):
    values = [(point[name] - spec.feature_bounds[name][0]) /
              (spec.feature_bounds[name][1] - spec.feature_bounds[name][0])
              for name in spec.feature_names]
    return [1., *values, *([value * value for value in values] if degree == 2 else []),
            *(values[first] * values[second] for first, second in interactions)]


def _predict(spec, model, point):
    design = _design(spec, point, model["degree"], model.get("interactions", ()))
    if "concept" in model:
        from .language import Program
        from .models import Observation
        design.append(Program.from_dict(model["concept"]).predict(Observation(point, 0.)))
    return sum(a * b for a, b in zip(model["coefficients"], design))


def fit_objective(spec, rows, budget):
    """Compare affine and additive quadratic goal models on the current scope."""
    if spec.context.get("objective_direction") != "minimize" or not rows:
        return None
    if any("objective" not in row["context"] for row in rows):
        return None
    targets = [row["context"]["objective"] for row in rows]
    points = [row["observation"]["features"] for row in rows]
    models = []
    for degree in (1, 2):
        design = [_design(spec, point, degree) for point in points]

        def fit(indices):
            if not budget.spend(len(indices) * len(design[0])):
                return None
            columns = [[design[i][j] for i in indices] for j in range(1, len(design[0]))]
            return {"degree": degree,
                    "coefficients": _linear_fit(columns, [targets[i] for i in indices])}

        fitted = fit(list(range(len(rows))))
        if fitted is None:
            break
        folds, errors = [], []
        for index, point in enumerate(points):
            fold = fit([i for i in range(len(rows)) if i != index])
            if fold is None or not budget.spend(1):
                break
            folds.append(fold)
            errors.append((_predict(spec, fold, point) - targets[index]) ** 2)
        if len(folds) == len(rows):
            models.append({**fitted, "cv_loss": mean(errors), "folds": folds})
    if not models:
        return None
    chosen = min(models, key=lambda item: (item["cv_loss"], item["degree"]))
    bound = float(spec.context["constraint_upper_bound"])
    feasible_targets = [target for target, row in zip(targets, rows)
                        if row.get("observation_status") != "disputed"
                        and row["observation"]["target"] <= bound]
    return {**chosen, "constraint_upper_bound": bound,
            "incumbent": min(feasible_targets) if feasible_targets else None,
            "training_rows": len(rows),
            "compared_models": [{key: model[key] for key in ("degree", "coefficients", "cv_loss")}
                                for model in models]}


def objective_pool(spec, model, budget=None):
    """Derive an optimum and coordinate sweeps from the learned goal model."""
    coefficients, dimension = model["coefficients"], len(spec.feature_names)
    optimum = {}
    for index, name in enumerate(spec.feature_names):
        slope = coefficients[index + 1]
        curvature = coefficients[dimension + index + 1] if model["degree"] == 2 else 0.
        locations = [0., 1.]
        if curvature > 0:
            locations.append(min(1., max(0., -slope / (2 * curvature))))
        value = min(locations, key=lambda x: slope * x + curvature * x * x)
        low, high = spec.feature_bounds[name]
        optimum[name] = low + value * (high - low)
    if model.get("interactions") and (budget is None or budget.spend(dimension ** 3)):
        import numpy as np

        hessian = np.diag(2 * np.asarray(coefficients[1 + dimension:1 + 2 * dimension]))
        for coefficient, (first, second) in zip(coefficients[1 + 2 * dimension:], model["interactions"]):
            hessian[first, second] = hessian[second, first] = coefficient
        # A positive definite fitted quadratic has one joint minimum. If it
        # lies outside the apparatus, the existing coordinate sweeps retain
        # feasible alternatives around its clipped control.
        if np.linalg.eigvalsh(hessian).min() > 1e-10:
            normalized = np.clip(np.linalg.solve(hessian, -np.asarray(coefficients[1:1 + dimension])), 0., 1.)
            optimum = {name: spec.feature_bounds[name][0] + float(value) *
                       (spec.feature_bounds[name][1] - spec.feature_bounds[name][0])
                       for name, value in zip(spec.feature_names, normalized)}
    points = [optimum, *({**optimum, name: low + index * (high - low) / 32}
                        for name, (low, high) in spec.feature_bounds.items() for index in range(33))]
    if "concept" in model:
        points.extend(dict(point) for probe in spec.available_probes for point in probe.sequence)
    return points


def objective_prediction(spec, model, point, budget):
    if not budget.spend(1 + len(model["folds"])):
        return None
    prediction = _predict(spec, model, point)
    folds = [_predict(spec, fold, point) for fold in model["folds"]]
    center = mean(folds)
    uncertainty = model["cv_loss"] + (len(folds) - 1) / len(folds) * sum(
        (value - center) ** 2 for value in folds)
    return {"mean": prediction, "variance": uncertainty}


def _improvement(prediction, incumbent):
    if incumbent is None:
        return 1.
    gain = incumbent - prediction["mean"]
    deviation = sqrt(prediction["variance"])
    if deviation == 0:
        return max(0., gain)
    z = gain / deviation
    return max(0., gain * (1 + erf(z / sqrt(2))) / 2
               + deviation * exp(-z * z / 2) / sqrt(2 * pi))


def point_distance(spec, point, selected):
    return min((sum(((point[name] - previous[name]) /
                    (spec.feature_bounds[name][1] - spec.feature_bounds[name][0])) ** 2
                    for name in spec.feature_names) / len(spec.feature_names)
                for previous in selected), default=1.)


def select_objective_controls(spec, scored, model, count):
    """Reserve goal opportunities while leaving at least two separating tests."""
    if count <= 2 or not model:
        return []
    available = [item for item in scored if item["objective"] is not None]
    offers = []
    for index in range(len(scored[0]["predictions"]) if scored else 0):
        feasible = [item for item in available
                    if item["predictions"][index] <= model["constraint_upper_bound"]]
        if feasible:
            offer = min(feasible, key=lambda item: item["objective"]["mean"])
            if offer not in offers:
                offers.append(offer)
    if not offers:
        return []
    first = max(offers, key=lambda item: _improvement(item["objective"], model["incumbent"]))
    if _improvement(first["objective"], model["incumbent"]) <= 0:
        return []
    selected = [(first, "objective_improvement")]
    feasible = [item for item in available if item != first and
                min(item["predictions"]) <= model["constraint_upper_bound"]]
    if count > 3 and feasible:
        def opportunity(item):
            return (_improvement(item["objective"], model["incumbent"])
                    * point_distance(spec, item["point"], [first["point"]]))
        second = max(feasible, key=opportunity)
        if opportunity(second) > 0:
            selected.append((second, "objective_coverage"))
    return selected
