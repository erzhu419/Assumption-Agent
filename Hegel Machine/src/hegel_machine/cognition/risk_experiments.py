"""Purchase evidence that distinguishes already proposed risk explanations.

The proposals and their sample-size forecasts come from paid evidence. Their
comparison uses only responses acquired after all compared programs are frozen.
Uncontrolled instrument readings are supplied by the apparatus, never forecast
as if they were controllable inputs.
"""

from copy import deepcopy
from math import ceil, sqrt
from statistics import mean, stdev

from .evaluation import point_loss, predict
from .language import Program
from .models import Candidate


def _composed(baseline, correction, fallback):
    def at_start(program, value):
        depth = program.minimum_history_depth
        return (Program("if", children=(Program("available", lag=depth), program,
                Program("const", value=value))) if depth else program)
    return Program("add", children=(at_start(baseline, fallback),
                                    at_start(correction, 0.))).simplified()


def distinguish_risk_explanations(support, *, spec, acquire, channels, remaining,
                                 budget, minimum_improvement, measurement_rate,
                                 reserve_rows, reserve_compute=0):
    """Resolve affordable positive residual rivals through independent exposure.

    ``acquire`` is the ordinary budgeted acquisition callback. The caller retains
    its existing control-probe policy, including inactive-probe ablations. The
    returned support uses the same residual-program contract as noisy adequacy;
    a supported correction therefore enters the normal revision/confirmation
    path. No laboratory-private information is needed.
    """
    start = remaining()
    original, revised, events = deepcopy(support), deepcopy(support), []
    accumulated, baseline_losses, covered = [], [], []
    baseline_payload = support.get("counterexample_baseline_program")
    comparisons = [item for item in support.get("residual_comparisons", [])
                   if item.get("gain", 0.) > minimum_improvement]
    if baseline_payload is None or not comparisons or support.get("structural_counterexample"):
        return {"support": revised, "events": events, "status": "not_needed",
                "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0}

    baseline = Program.from_dict(baseline_payload)
    fallback = support.get("counterexample_fallback", 0.)
    q, frozen, seen = spec.risk_quantile, [], []
    for item in comparisons:
        correction = Program.from_dict(item["program"])
        if correction in seen:
            continue
        seen.append(correction)
        frozen.append({"comparison": deepcopy(item), "program":
            _composed(baseline, correction, fallback)})
    gains = [[] for _ in frozen]
    candidates = [{"candidate": Candidate(f"risk_experiment:{index}", item["program"],
                    "residual_explanation"), "kind": item["comparison"]["kind"],
                    "fallback": fallback}
                  for index, item in enumerate(frozen)]
    candidates.append({"candidate": Candidate("risk_experiment:incumbent", baseline,
                                              "retained_incumbent"), "kind": "incumbent",
                       "fallback": fallback})
    unit_compute = len(frozen) + 1
    minimum_rows = min(16, spec.max_probe_length)

    def finish(status, reason=None):
        if reason:
            revised.update(adequate=False, reason=reason)
        revised["risk_experiment"] = {"status": status, "new_rows": len(accumulated),
            "selection_rows": support.get("validation_rows", 0),
            "selection_validation_loss": support.get("validation_loss"),
            "selection_noise_loss_limit": support.get("noise_loss_limit"),
            "selection_empirical_quantile_coverage": support.get("empirical_quantile_coverage"),
            "selection_comparisons": [{key: item.get(key) for key in ("kind", "gain", "margin")}
                                      for item in comparisons],
            "frozen_alternatives": len(frozen), "selection_evidence_reused": False}
        after = remaining()
        costs = {resource + "_cost": start[resource + "_remaining"] - after[resource + "_remaining"]
                 for resource in ("observation", "measurement", "compute")}
        events.append({"action": "finish_risk_experiment", "status": status,
            "reason": revised.get("reason"), "new_rows": len(accumulated), **costs})
        return {"support": revised, "events": events, "status": status, **costs}

    while True:
        available = remaining()
        capacity = min(max(0, available["observation_remaining"] - reserve_rows),
                       max(0, budget.remaining - reserve_compute) // unit_compute)
        if measurement_rate:
            capacity = min(capacity, max(0, available["measurement_remaining"]
                           - reserve_rows * measurement_rate) // measurement_rate)
        metrics, eligible = [], []
        for index, item in enumerate(frozen):
            compared = item["comparison"]
            if accumulated:
                gain = mean(gains[index])
                margin = max(minimum_improvement, 3 * stdev(gains[index]) / sqrt(len(accumulated)))
                count = len(accumulated)
            else:
                gain, margin = compared["gain"], compared["margin"]
                count = support["validation_rows"]
            measured = {**deepcopy(compared), "gain": gain, "margin": margin}
            metrics.append(measured)
            if gain <= minimum_improvement:
                continue
            # The initial forecast requests a wholly NEW sample. After that,
            # only observations of these unchanged frozen rivals accumulate.
            required = max(minimum_rows, ceil(count * (margin / gain) ** 2)
                           - (len(accumulated) if accumulated else 0))
            if required <= capacity:
                eligible.append((required, -gain, index))

        if accumulated:
            revised.update(residual_comparisons=metrics, validation_rows=len(accumulated),
                validation_loss=mean(baseline_losses),
                predictable_residual_gain=max(item["gain"] for item in metrics))
            supported = [item for item in metrics if item["gain"] > item["margin"]]
            if supported:
                revised["structural_counterexample"] = max(supported, key=lambda item: item["gain"])
                return finish("supported", "predictable_residual")
            if not any(item["gain"] > minimum_improvement for item in metrics):
                revised.update(adequate=original["adequate"], reason=original["reason"])
                # More observations can reject the incumbent's noise account
                # even when none of the frozen alternatives improves it.
                if q != .5:
                    coverage = mean(covered)
                    allowance = 3 * sqrt(q * (1 - q) / len(accumulated)) + 1 / len(accumulated)
                    revised.update(empirical_quantile_coverage=coverage,
                                   quantile_coverage_allowance=allowance)
                    if abs(coverage - q) > allowance:
                        return finish("baseline_miscalibrated", "quantile_miscalibration")
                training_count = original.get("training_rows", 0)
                training_loss, old_limit = original.get("training_loss"), original.get("noise_loss_limit")
                if training_count and training_loss is not None and old_limit is not None:
                    factor = sqrt((1 / training_count + 1 / len(accumulated)) /
                                  (1 / training_count + 1 / original["validation_rows"]))
                    limit = max(minimum_improvement, training_loss + (old_limit - training_loss) * factor)
                    revised["noise_loss_limit"] = limit
                    if revised["validation_loss"] > limit:
                        return finish("baseline_loss_shift", "residual_loss_shift")
                return finish("no_positive_measured_gain")
        if not eligible:
            if not accumulated:
                return finish("deferred")
            return finish("unresolved", "risk_experiment_unfunded")

        required, _, selected = min(eligible)
        count = min(spec.max_probe_length, required)
        events.append({"action": "plan_risk_experiment", "kind": frozen[selected]["comparison"]["kind"],
            "predicted_gain": metrics[selected]["gain"], "gain_margin": metrics[selected]["margin"],
            "required_additional_rows": required, "planned_rows": count,
            "channels": list(channels), "frozen_alternatives": len(frozen),
            "evidence_basis": "new_frozen_comparison" if accumulated else "prior_paid_comparison",
            "observation_cost": count, "measurement_cost": count * measurement_rate,
            "compute_cost": count * unit_compute,
            "reserve_rows": reserve_rows, "reserve_compute": reserve_compute})
        rows = acquire(tuple(channels), "distinguish_risk_explanations", candidates,
                       rows_count=count)
        if not rows:
            return finish("unresolved", "risk_experiment_unfunded")
        if not budget.spend(len(rows) * unit_compute):
            return finish("unresolved", "risk_experiment_unfunded")
        predictions = [predict(baseline, row, fallback) for row in rows]
        losses = [point_loss(value, row.target, q) for value, row in zip(predictions, rows)]
        baseline_losses.extend(losses)
        covered.extend(row.target <= value for value, row in zip(predictions, rows))
        for index, item in enumerate(frozen):
            gains[index].extend(before - point_loss(predict(item["program"], row, fallback), row.target, q)
                                for before, row in zip(losses, rows))
        accumulated.extend(rows)
        events.append({"action": "compare_risk_explanations", "observation_rows": len(rows),
            "new_row_start": len(accumulated) - len(rows), "new_row_end": len(accumulated),
            "accumulated_new_rows": len(accumulated), "fixed_quantile": q,
            "baseline_program": baseline_payload, "baseline_fallback": fallback,
            "comparisons": [{"kind": item["comparison"]["kind"],
                "program": item["comparison"]["program"], "gain": mean(values),
                "margin": max(minimum_improvement, 3 * stdev(values) / sqrt(len(values)))}
                for item, values in zip(frozen, gains)]})
