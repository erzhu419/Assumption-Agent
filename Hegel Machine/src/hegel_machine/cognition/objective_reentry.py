"""Repair a goal representation and choose its next distinguishing control."""

from copy import deepcopy
from math import sqrt
from statistics import mean

from .continuous_state import _linear_fit
from .decision_experiments import _design, _predict, objective_pool
from .language import Program
from .models import Observation


def fit_repaired_objective(spec, rows, concept, budget):
    """Compare one learned coordinate against identifiable base goal models.

    The concept's scope determines which already paid rows may fit its goal
    relation. Whole controls stay together in the same five-way validation fold,
    including repeated observations of a control.
    """
    import numpy as np

    eligible = [row for row in rows
                if row["context"]["global_index"] >= concept["scope_start"]]
    if not eligible:
        return None
    points = [row["observation"]["features"] for row in eligible]
    targets = [float(row["context"]["objective"]) for row in eligible]
    indices = [row["context"]["global_index"] for row in eligible]
    keys = [tuple(point[name] for name in spec.feature_names) for point in points]
    groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
    folds = sorted(set(groups.values()))
    if len(folds) < 3 or not budget.spend(len(points)):
        return None
    program = Program.from_dict(concept["program"])
    coordinate = [program.predict(Observation(point, 0.)) for point in points]
    designs = {}
    for degree in (1, 2):
        width = 1 + degree * len(spec.feature_names)
        if not budget.spend(len(points) * width):
            return None
        designs[degree] = np.asarray([_design(spec, point, degree) for point in points])
    fitted_models = {False: [], True: []}
    for repaired in (False, True):
        for degree in (1, 2):
            design = (np.column_stack((designs[degree], coordinate))
                      if repaired else designs[degree])
            exhausted = False

            def fit(selected):
                nonlocal exhausted
                if not budget.spend(len(selected) * design.shape[1]):
                    exhausted = True
                    return None
                if np.linalg.matrix_rank(design[selected]) < design.shape[1]:
                    return None
                columns = [design[selected, j].tolist() for j in range(1, design.shape[1])]
                model = {"degree": degree,
                    "coefficients": _linear_fit(columns, [targets[i] for i in selected])}
                if repaired:
                    model["concept"] = deepcopy(concept["program"])
                return model

            fitted = fit(list(range(len(eligible))))
            if exhausted:
                return None
            if fitted is None:
                continue
            models, errors, reference_errors = [], [], []
            for group in folds:
                training = [i for i, key in enumerate(keys) if groups[key] != group]
                testing = [i for i, key in enumerate(keys) if groups[key] == group]
                fold = fit(training)
                if exhausted:
                    return None
                if fold is None:
                    break
                if not budget.spend(len(testing)):
                    return None
                models.append(fold)
                reference = mean(targets[i] for i in training)
                errors.extend((sum(a * b for a, b in zip(fold["coefficients"], design[i]))
                               - targets[i]) ** 2 for i in testing)
                reference_errors.extend((reference - targets[i]) ** 2 for i in testing)
            if len(models) == len(folds):
                fitted_models[repaired].append({**fitted, "folds": models,
                    "cv_loss": mean(errors), "reference_loss": mean(reference_errors),
                    "training_rows": len(eligible), "training_indices": list(indices),
                    "training_end": max(indices)})
    if not fitted_models[False] or not fitted_models[True]:
        return None
    base_loss = min(model["cv_loss"] for model in fitted_models[False])
    candidate = min(fitted_models[True], key=lambda model: (model["cv_loss"], model["degree"]))
    if candidate["cv_loss"] >= min(base_loss, candidate["reference_loss"]):
        return None
    return {**candidate, "base_cv_loss": base_loss}


def select_reentry_control(spec, research, rows, forecast, budget):
    """Scan the public pool for a feasible, untested model disagreement."""
    model = research["candidate"]
    validation = [*research.get("incumbent_validation", []),
                  *research["validation"][research.get("comparison_start", 0):]]

    def key(point):
        return tuple(point[name] for name in spec.feature_names)

    compared = {key(row["features"]) for row in validation}
    points = [*objective_pool(spec, model, budget),
              *(point for probe in spec.available_probes for point in probe.sequence),
              *(row["observation"]["features"] for row in rows)]
    requested, actual_points, best = set(), set(), None
    for point in points:
        if key(point) in requested or key(point) in compared:
            continue
        requested.add(key(point))
        if not budget.spend(1):
            break
        prediction = forecast(point)
        actual = prediction["features"]
        actual_key = key(actual)
        if actual_key in actual_points or actual_key in compared:
            continue
        actual_points.add(actual_key)
        if not prediction["feasible"]:
            continue
        if not budget.spend(1):
            break
        candidate_mean = _predict(spec, model, actual)
        gap = abs(candidate_mean - prediction["objective_mean"])
        disagreement = gap / sqrt(max(float(prediction["objective_noise"]), 1e-12))
        if gap > 1e-7 and (best is None or disagreement > best["disagreement"]):
            best = {**deepcopy(prediction), "candidate_mean": candidate_mean,
                    "disagreement": disagreement}
    return best
