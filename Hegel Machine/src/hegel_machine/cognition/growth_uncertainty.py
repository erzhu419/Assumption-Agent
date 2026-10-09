"""Estimate response uncertainty from independent repetitions at fixed controls.

Between-control fit error is not counted as stochastic variance. A conditional
variance proposal must predict held-out repeat groups better than a constant
variance; the final response confirmation remains a separate acquisition.
"""

from collections import defaultdict
from statistics import mean, variance

from .language import Program
from .models import Observation
from .synthesis import _fit_linear


def estimate_uncertainty(rows, program, controls, budget, *, bounds=None):
    if not budget.spend(len(rows)):
        return None
    groups = defaultdict(list)
    for row in rows:
        key = tuple((name, row.features[name]) for name in controls)
        groups[key].append(row.target - program.predict(row))
    grouped = [Observation(dict(key), variance(values)) for key, values in groups.items()
               if len(values) >= 3]
    dof = sum(len(values) - 1 for values in groups.values() if len(values) >= 3)
    pooled = (sum(variance(values) * (len(values) - 1) for values in groups.values()
                  if len(values) >= 3) / dof if dof else 0.)
    result = {"kind": "stochastic" if pooled > 1e-6 else "deterministic",
              "variance": pooled, "repeat_degrees_of_freedom": dof,
              "variance_program": None, "variance_selection": [],
              "repeat_groups": len(grouped)}
    if len(grouped) < 8 or pooled <= 1e-6:
        return result
    candidates = [("constant", [])]
    for name in controls:
        coordinate = Program("feature", name=name)
        if bounds and name in bounds:
            low, high = bounds[name]
            if high > low:
                coordinate = Program("add", children=(Program("mul", children=(
                    Program("const", value=2. / (high - low)), coordinate)),
                    Program("const", value=-(low + high) / (high - low))))
        candidates.append((f"absolute:{name}", [Program("abs", children=(coordinate,))]))
        candidates.append((f"quadratic:{name}", [Program("mul", children=(coordinate, coordinate))]))
    best = None
    for label, terms in candidates:
        # Leave a whole repeated setting out. Individual replicates at a shared
        # setting cannot independently assess variance generalization.
        charge = len(grouped) * (len(grouped) * (len(terms) + 1) + 1)
        if not budget.spend(charge):
            break
        errors = []
        for i, held in enumerate(grouped):
            training = grouped[:i] + grouped[i + 1:]
            fitted = _fit_linear(terms, training)
            prediction = max(mean(row.target for row in training) * .05, fitted.predict(held))
            errors.append((held.target - prediction) ** 2)
        error = mean(errors)
        result["variance_selection"].append({"model": label, "group_cv_mse": error})
        if best is None or error < best[0] * .9:
            best = error, terms, label
    if best and best[1] and budget.spend(len(grouped) * (len(best[1]) + 1)):
        fitted = _fit_linear(best[1], grouped).simplified()
        bounded = Program("max", children=(Program("const", value=pooled * .05), fitted))
        result["variance_program"] = bounded.to_dict()
    return result


def predictive_variance(uncertainty, row):
    payload = uncertainty.get("variance_program")
    return (Program.from_dict(payload).predict(row) if payload else uncertainty.get("variance", 0.))


def calibration(uncertainty, program, rows):
    """Score a frozen mean/variance model; this function acquires no evidence."""
    if not rows:
        return {}
    errors = [(program.predict(row) - row.target) ** 2 for row in rows]
    variances = [max(1e-12, predictive_variance(uncertainty, row)) for row in rows]
    return {"standardized_mse": mean(error / value for error, value in zip(errors, variances)),
            "normal_95_coverage": mean(float(error <= 1.96 ** 2 * value)
                                       for error, value in zip(errors, variances)),
            "mean_predicted_variance": mean(variances)}
