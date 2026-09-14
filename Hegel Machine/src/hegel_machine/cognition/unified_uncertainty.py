"""Residual risk revisions and finite-sample noisy-model assessment.

These routines never query an apparatus. Residual fits use paid training rows;
their structural counterexamples are assessed on the supplied later trajectory.
The task's quantile remains fixed, including the existing squared-loss meaning
of q=0.5. Noise compatibility is an empirical result, not a claim that hidden
mechanisms do not exist.
"""

from dataclasses import replace
from math import sqrt
from statistics import mean, stdev

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, loss, point_loss, predict
from .language import Program
from .meta_prior_claims import compile_claims
from .models import Candidate, Observation, SearchBudget
from .risk_search import empirical_quantile, trajectory_blocks


def _residuals(program, rows, budget, fallback):
    if not budget.spend(len(rows)):
        return None
    return [row.target - predict(program, row, fallback) for row in rows]


def _scale_models(rows, residuals, budget):
    """Competing global, absolute-coordinate and squared-coordinate scales."""
    count = len(rows)
    if not budget.spend(count):
        return []
    center = mean(residuals)
    magnitudes = [abs(value - center) for value in residuals]
    magnitude = mean(magnitudes)
    floor = max(1e-12, .05 * magnitude)
    result = [("global", Program("const", value=max(floor, magnitude)), center)]
    names = sorted(set.intersection(*(set(row.features) for row in rows)))
    current = [Program("feature", name=name) for name in names]
    for kind, terms in (("absolute", [Program("abs", children=(term,)) for term in current]),
                        ("quadratic", [Program("mul", children=(term, term)) for term in current])):
        if not terms or not budget.spend(count * (len(terms) + 1)):
            continue
        columns = [[term.predict(row) for row in rows] for term in terms]
        coefficients = _linear_fit(columns, magnitudes)
        spread = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
        spread = Program("max", children=(Program("const", value=floor), spread))
        result.append((kind, spread, center))
    return result


def residual_risk_candidates(spec, program, training, budget, *, fallback=0.):
    """Return executable residual revisions in the unified candidate format.

    The arbitrary original computation remains an actual child expression.
    Global and conditional tail corrections are fitted only from its residuals.
    Conditional scales include squared public coordinates without assuming a
    particular instrument or disturbance distribution.
    """
    rows = list(training)
    if not rows:
        return []
    q = spec.risk_quantile
    residuals = _residuals(program, rows, budget, fallback)
    if residuals is None or not budget.spend(len(rows)):
        return []
    offset = mean(residuals) if q == .5 else empirical_quantile(residuals, q)
    corrected = Program("add", children=(program, Program("const", value=offset))).simplified()
    proposals = [("residual_mean" if q == .5 else "residual_global_quantile", corrected,
                  {"kind": "global_residual", "offset": offset})]
    if q != .5:
        for kind, spread, center in _scale_models(rows, residuals, budget):
            if kind == "global" or not budget.spend(2 * len(rows)):
                continue
            standardized = [(residual - center) / spread.predict(row)
                            for row, residual in zip(rows, residuals)]
            quantile = empirical_quantile(standardized, q)
            correction = Program("add", children=(Program("const", value=center),
                Program("mul", children=(Program("const", value=quantile), spread))))
            revised = Program("add", children=(program, correction)).simplified()
            proposals.append((f"residual_{kind}_scale_quantile", revised,
                {"kind": "conditional_scale", "scale_program": spread.to_dict(),
                 "residual_center": center, "standardized_quantile": quantile}))
    results = []
    for kind, revised, model in proposals:
        if not budget.spend(len(rows)):
            break
        results.append({"candidate": Candidate(f"uncertainty:{kind}:{budget.spent}", revised,
                           "uncertainty_revision", "Fit the task's fixed risk target around the retained computation"),
            "kind": kind, "family_id": "T22" if q != .5 else "T18",
            "training_loss": loss(revised, rows, fallback, q), "fixed_quantile": q,
            "uncertainty": {**model, "support_rows": len(rows), "fixed_quantile": q}})
    return results


def assess_noisy_adequacy(spec, program, training, validation, budget, *,
                          fallback=0., minimum_improvement=1e-7):
    """Assess a frozen predictor without forgiving predictable structural error.

    A residual-noise tolerance is learned from training losses. Before it can
    support adoption, ordinary current, nonlinear and history residual models
    compete on the later validation trajectory. Public object schemas also
    nominate direct structural models of the original response. Their paired
    gain must exceed three empirical standard errors to count as a structural counterexample.
    This finite check deliberately reports unresolved evidence when calibration
    rows or fitting budget are insufficient.
    """
    training, validation = list(training), list(validation)
    start, q = budget.spent, spec.risk_quantile
    result = {"adequate": False, "reason": "insufficient_calibration_rows",
              "fixed_quantile": q, "training_rows": len(training), "validation_rows": len(validation),
              "training_loss": None, "validation_loss": None, "noise_loss_limit": None,
              "predictable_residual_gain": None, "structural_counterexample": None,
              "empirical_quantile_coverage": None, "compute_cost": 0}

    def finish(reason, adequate=False):
        result.update(reason=reason, adequate=adequate, compute_cost=budget.spent - start)
        return result

    if not training or not validation:
        return finish("insufficient_calibration_rows")
    values = _residuals(program, [*training, *validation], budget, fallback)
    if values is None or not budget.spend(len(values)):
        return finish("insufficient_budget")
    residuals, later = values[:len(training)], values[len(training):]
    training_losses = [point_loss(0., value, q) for value in residuals]
    validation_losses = [point_loss(0., value, q) for value in later]
    training_loss, validation_loss = mean(training_losses), mean(validation_losses)
    result.update(training_loss=training_loss, validation_loss=validation_loss)
    if max(training_loss, validation_loss) <= minimum_improvement:
        result["noise_loss_limit"] = minimum_improvement
        return finish("deterministic_fit", True)
    if len(training) < 32 or len(validation) < 16 or len(trajectory_blocks(training)) < 2:
        return finish("insufficient_calibration_rows")

    # The tolerance is fixed by training residuals and validation row count;
    # a large new validation error cannot enlarge its own allowance.
    variation = stdev(training_losses)
    limit = training_loss + 3 * variation * sqrt(1 / len(training) + 1 / len(validation))
    result["noise_loss_limit"] = max(minimum_improvement, limit)
    calibration_ok = True
    if q != .5:
        coverage = mean(value <= 0. for value in later)
        allowance = 3 * sqrt(q * (1 - q) / len(validation)) + 1 / len(validation)
        result.update(empirical_quantile_coverage=coverage, quantile_coverage_allowance=allowance)
        calibration_ok = abs(coverage - q) <= allowance

    # Fit residual explanations rather than widening a band around a wrong
    # history or current-coordinate model. The compiler has no access to the
    # later observations used below to challenge these explanations.
    rows = [Observation(row.features, value, row.history) for row, value in zip(training, residuals)]
    reserve = len(validation) * 8
    if budget.remaining < reserve + len(training) * 6:
        return finish("insufficient_budget")
    compared = []
    if spec.context.get("object_schema"):
        from .unified_structure import propose_structures

        structural_budget = SearchBudget(budget.remaining - reserve)
        structures = propose_structures(replace(spec, risk_quantile=.5), training, structural_budget)
        budget.spend(structural_budget.spent)
        structure_complete = bool(structures) and structures[0]["structure"]["search_complete"]
        for claim in structures:
            direct = claim["candidate"].program
            if q != .5:
                if not budget.spend(2 * len(training)):
                    break
                remaining = [row.target - predict(direct, row, fallback) for row in training]
                direct = Program("add", children=(direct,
                    Program("const", value=empirical_quantile(remaining, q)))).simplified()
            if not budget.spend(len(validation)):
                break
            corrected_losses = [point_loss(predict(direct, row, fallback), row.target, q)
                                for row in validation]
            gains = [before - after for before, after in zip(validation_losses, corrected_losses)]
            # Downstream research composes counterexamples with the incumbent.
            # Preserve that residual-program contract for a direct-target fit.
            correction = Program("sub", children=(direct, program)).simplified()
            compared.append({"kind": claim["kind"], "gain": mean(gains),
                "margin": max(minimum_improvement, 3 * stdev(gains) / sqrt(len(gains))),
                "program": correction.to_dict(), "comparison_basis": "direct_target"})
        result["residual_comparisons"] = compared
        if compared:
            result["predictable_residual_gain"] = max(item["gain"] for item in compared)
            supported = [item for item in compared if item["gain"] > item["margin"]]
            if supported:
                result["structural_counterexample"] = max(supported, key=lambda item: item["gain"])
                return finish("predictable_residual")
        if not structure_complete or len(compared) != len(structures):
            return finish("insufficient_budget")
    structure_comparisons = len(compared)
    if budget.remaining < reserve + len(training) * 6:
        return finish("insufficient_budget")
    fitting = SearchBudget(min(18000, budget.remaining - reserve))
    claims = compile_claims(replace(spec, risk_quantile=.5), rows, fitting)
    budget.spend(fitting.spent)
    if not claims:
        return finish("insufficient_budget")
    required_depth = min(4, max(len(row.history) for row in training))
    complete_search = (any(claim["kind"] == "coupled_quadratic" for claim in claims)
        and (not required_depth or any(claim.get("history_depth") == required_depth for claim in claims)))
    for claim in claims:
        correction = claim["candidate"].program
        # Squared-loss residual fits are proposals. Under a fixed quantile they
        # receive a training-only quantile offset before the independent test.
        if q != .5:
            if not budget.spend(2 * len(training)):
                break
            remaining = [value - predict(correction, row, 0.) for row, value in zip(training, residuals)]
            correction = Program("add", children=(correction,
                Program("const", value=empirical_quantile(remaining, q)))).simplified()
        if not budget.spend(len(validation)):
            break
        corrected_losses = [point_loss(predict(correction, row, 0.), value, q)
                            for row, value in zip(validation, later)]
        gains = [before - after for before, after in zip(validation_losses, corrected_losses)]
        gain = mean(gains)
        margin = max(minimum_improvement, 3 * stdev(gains) / sqrt(len(gains)))
        compared.append({"kind": claim["kind"], "gain": gain, "margin": margin,
                         "program": correction.to_dict()})
    result["residual_comparisons"] = compared
    if not compared:
        return finish("insufficient_budget")
    supported = [item for item in compared if item["gain"] > item["margin"]]
    best = max(compared, key=lambda item: item["gain"])
    result["predictable_residual_gain"] = best["gain"]
    if supported:
        result["structural_counterexample"] = max(supported, key=lambda item: item["gain"])
        return finish("predictable_residual")
    if not complete_search or len(compared) - structure_comparisons != len(claims):
        return finish("insufficient_budget")
    if not calibration_ok:
        return finish("quantile_miscalibration")
    if validation_loss > result["noise_loss_limit"]:
        return finish("residual_loss_shift")
    return finish("noise_compatible", True)
