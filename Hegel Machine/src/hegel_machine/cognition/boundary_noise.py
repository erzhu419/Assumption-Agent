"""Price boundary measurements with uncertainty learned from paid repeats."""

from copy import deepcopy
from math import exp, log, sqrt
from statistics import mean


def repeated_noise_scales(rows, noise_prediction, budget, *, scope_start, feature_names):
    """Approximate the unknown scale of the current HVD noise shape.

    Within-control differences eliminate the unknown constraint mean. Their
    standardized sum of squares and replication degrees of freedom define an
    inverse-chi-square scale distribution. The fitted HVD shape is held fixed
    while later repeat contrasts update the scale weights. The predictive
    mixture informs experiment choice and terminal chance constraints when a
    boundary question activates it. Single observations supply no variance
    degrees of freedom, and obsolete/disputed constraint labels are excluded.
    """
    groups = {}
    if not budget.spend(len(rows)):
        return None
    for row in rows:
        if (row["context"]["global_index"] < scope_start
                or row.get("observation_status") == "disputed"):
            continue
        key = tuple(row["observation"]["features"][name] for name in feature_names)
        groups.setdefault(key, []).append(row)
    repeated = [values for values in groups.values() if len(values) > 1]
    if not repeated:
        return None
    summaries, standardized_sum, degrees = [], 0., 0
    for values in repeated:
        if not budget.spend(len(values) + 1):
            return None
        features = values[0]["observation"]["features"]
        noise = float(noise_prediction(features))
        targets = [row["observation"]["target"] for row in values]
        center = mean(targets)
        squares = sum((target - center) ** 2 for target in targets)
        standardized_sum += squares / max(noise, 1e-12)
        degrees += len(values) - 1
        summaries.append({"features": dict(features), "replicates": len(values),
            "indices": [row["context"]["global_index"] for row in values],
            "sum_squares": squares, "predicted_noise": noise})
    if not budget.spend(4):
        return None
    from scipy.stats import chi2

    probabilities = [.125, .375, .625, .875]
    scales = [float(standardized_sum / chi2.ppf(1. - q, degrees)) for q in probabilities]
    return {"degrees_of_freedom": degrees, "standardized_sum_squares": standardized_sum,
        "scale_quantiles": probabilities, "scales": scales, "weights": [.25] * 4,
        "groups": summaries, "constraint_scope_start": int(scope_start)}


def scale_sensitive_repeats(actions, evidence, *, chance_z):
    """Identify purchased controls whose certificate depends on unknown noise."""
    result = []
    for action in actions:
        prediction = action["prediction"]
        if not action["replicate_count"] or "constraint_noise" not in prediction:
            continue
        noise = max(0., prediction["constraint_noise"])
        margins = [prediction["constraint_margin"] + chance_z * sqrt(noise) * (sqrt(scale) - 1.)
                   for scale in evidence["scales"]]
        if min(margins) <= 0. < max(margins):
            result.append({"point": list(action["point"]), "margins": margins,
                           "replicate_count": action["replicate_count"]})
    return result


def scaled_predictive_innovation(innovation, prediction, scale):
    """Marginalize latent noise through the label, never through learner state.

    The native fantasy routine expects a standardized predictive innovation.
    Rescaling its constraint component changes the generated label only. Its
    GP/HVD update still sees its ordinary current noise estimate and the drawn
    vector, exactly as it would on a physical observation.
    """
    noise = max(0., prediction["constraint_noise"])
    epistemic = max(0., prediction["constraint_variance"])
    ratio = sqrt((epistemic + noise * scale) / max(epistemic + noise, 1e-12))
    return [float(innovation[0]), float(innovation[1]) * ratio]


def update_noise_weights(evidence, *, previous_values, target, predicted_noise):
    """Update the scale mixture with one new, independent within-control contrast.

    Previous values determine the centering contrast; their old scatter was
    already counted and must not be scored again. ``updated_degrees`` counts
    only these new contrasts, separately from the initial replication fit.
    """
    updated = deepcopy(evidence)
    count = len(previous_values)
    if not count:
        return updated
    contrast = sqrt(count / (count + 1)) * (float(target) - mean(previous_values))
    scores = []
    for weight, scale in zip(evidence["weights"], evidence["scales"]):
        variance = max(float(predicted_noise) * scale, 1e-12)
        scores.append((log(weight) if weight > 0 else -float("inf"))
                      - .5 * (log(variance) + contrast * contrast / variance))
    normalizer = max(scores)
    weights = [exp(score - normalizer) for score in scores]
    updated.update(weights=[weight / sum(weights) for weight in weights],
                   updated_degrees=evidence.get("updated_degrees", 0) + 1,
                   contrast=contrast)
    return updated


def noise_tail_factor(evidence, *, alpha):
    """One upper-tail quantile after mixing noise explanations at one control."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    scales, weights = evidence["scales"], evidence["weights"]
    if max(scales) == 0:
        return {"quantile": 0., "variance_factor": 0., "alpha": alpha,
                "tail_probability": 0., "cdf_evaluations": 0, "component_evaluations": 0}
    evaluations = 0

    def cdf(value):
        nonlocal evaluations
        evaluations += 1
        return sum(weight * float(norm.cdf(value / sqrt(scale)))
                   for weight, scale in zip(weights, scales))

    ordinary = float(norm.ppf(1 - alpha))
    quantile = brentq(lambda value: cdf(value) - (1 - alpha),
                      0., ordinary * sqrt(max(scales)))
    probability = 1 - cdf(quantile)
    return {"quantile": quantile, "variance_factor": (quantile / ordinary) ** 2,
            "alpha": alpha, "tail_probability": probability,
            "cdf_evaluations": evaluations, "component_evaluations": evaluations * len(scales)}
