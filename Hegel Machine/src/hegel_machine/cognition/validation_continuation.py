"""Price a small additional validation round by its terminal decision value.

The evidence rule and model weights are planning approximations. They do not
provide a calibrated model probability or a frequentist coverage guarantee.
"""

from itertools import combinations, product
from math import erf, sqrt
from statistics import mean, variance


def evidence_summary(validation):
    """Summarize paired, already observed squared prediction errors."""
    count = len(validation)
    gains = [row["previous_error"] - row["candidate_error"] for row in validation]
    gain = mean(gains) if gains else 0.0
    radius = 2 * sqrt(variance(gains) / count) if count > 1 else 0.0
    decision = ("inconclusive" if count < 4 else
                "supported" if gain - radius > 1e-7 else
                "rejected" if gain <= 1e-7 else "inconclusive")
    return {"count": count, "mean_gain": gain, "radius": radius,
            "candidate_mse": mean(row["candidate_error"] for row in validation) if count else 0.0,
            "previous_mse": mean(row["previous_error"] for row in validation) if count else 0.0,
            "decision": decision}


def plan_validation_round(validation, controls, *, target_count, remaining_calls, budget):
    """Choose one or two fresh labels that can change the final objective choice.

    Controls contain feasible predictions and observation *variance*. Previously
    used validation controls remain terminal choices but cannot be new probes.
    Other paid positions remain eligible for a fresh label. A complete combination
    costs two model branches times three innovations per new observation.
    """
    evidence = evidence_summary(validation)
    missing = target_count - evidence["count"]
    if (evidence["count"] < 4 or evidence["mean_gain"] <= 1e-7
            or missing not in (1, 2)
            or remaining_calls < missing):
        return None

    feasible = []
    for control in controls:
        if control["feasible"] and not any(
                control["features"] == row["features"] for row in feasible):
            feasible.append(control)
    if len(feasible) < missing:
        return None
    previous_best = min(feasible, key=lambda row: row["objective_mean"])
    candidate_best = min(feasible, key=lambda row: row["candidate_mean"])
    if previous_best["features"] == candidate_best["features"]:
        return None

    used = [row["features"] for row in validation if "features" in row]
    eligible = [row for row in feasible if row["features"] not in used]
    if len(eligible) < missing:
        return None
    # Keep both available recommendations, then fill by disagreement. Terminal
    # choices above still include controls that supplied earlier validation.
    pool = []
    for control in [min(eligible, key=lambda row: row["objective_mean"]),
                    min(eligible, key=lambda row: row["candidate_mean"]), *sorted(eligible,
            key=lambda row: abs(row["candidate_mean"] - row["objective_mean"]),
            reverse=True)]:
        if (abs(control["candidate_mean"] - control["objective_mean"]) > 1e-12
                and not any(control["features"] == row["features"] for row in pool)):
            pool.append(control)
            if len(pool) == 6:
                break
    if len(pool) < missing:
        return None

    se = evidence["radius"] / 2
    candidate_weight = (0.5 * (1 + erf(evidence["mean_gain"] / (se * sqrt(2))))
                        if se else 1.0)
    innovations = (-sqrt(1.5), 0.0, sqrt(1.5))
    scenarios_per_group = 2 * 3 ** missing
    evaluated_scenarios = 0
    best = None
    for group in combinations(pool, missing):
        if not budget.spend(scenarios_per_group):
            break
        evaluated_scenarios += scenarios_per_group
        expected_gain = resolution_probability = 0.0
        for model, weight in (("candidate_mean", candidate_weight),
                              ("objective_mean", 1 - candidate_weight)):
            scenario_weight = weight / (3 ** missing)
            for innovation in product(innovations, repeat=missing):
                future = list(validation)
                for control, z in zip(group, innovation):
                    observed = control[model] + sqrt(max(0.0, control["objective_noise"])) * z
                    future.append({
                        "previous_error": (control["objective_mean"] - observed) ** 2,
                        "candidate_error": (control["candidate_mean"] - observed) ** 2})
                decision = evidence_summary(future)["decision"]
                future_choice = candidate_best if decision == "supported" else previous_best
                expected_gain += scenario_weight * (previous_best[model] - future_choice[model])
                resolution_probability += scenario_weight * (decision != "inconclusive")
        if expected_gain > 0 and (best is None or expected_gain > best["expected_gain"]):
            best = {"controls": list(group), "expected_gain": expected_gain,
                    "expected_gain_per_call": expected_gain / missing,
                    "resolution_probability": resolution_probability,
                    "candidate_weight": candidate_weight,
                    "target_count": target_count, "new_calls": missing}
    if best is not None:
        best["evaluated_scenarios"] = evaluated_scenarios
    return best
