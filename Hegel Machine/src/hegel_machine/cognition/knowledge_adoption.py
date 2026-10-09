"""Screen costly risk knowledge against current, independently checked evidence.

Cheap retained/refitted responses remain proposals. Only measured marginal
benefit opens their expensive research procedures; a weak result can still be
confirmed or challenged by the ordinary risk researcher.
"""

from statistics import mean, stdev
from math import sqrt

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, point_loss, predict
from .meta_prior_research import _sources
from .models import Candidate
from .language import Program
from .risk_search import empirical_quantile, trajectory_blocks


def screen_risk_knowledge(spec, training, validation, baseline, calls, budget, *,
                          tolerance, priors=()):
    """Return affordable response proposals and sources worth deeper research.

    Each source competes against the best already retained response on the
    same unfitted rows. Redundant sources cannot each claim its full gain.
    Every invocation uses current evidence, so a past rejection is reversible.
    """
    if not calls:
        return [], [], []
    independent = bool(validation) and all(row not in training for row in validation)
    if baseline is None or not independent:
        # Without a comparison there is no measured reason to replay every
        # source. Leave one exploratory procedure available, with no reward.
        return calls[:1], [], [{"status": "comparison_unavailable", "source_ids":
            list(dict.fromkeys(source for call in calls for source in _sources(call))),
            "evidence_basis": "no_independent_validation", "compute_cost": 0}]
    q, n, m = spec.risk_quantile, len(training), len(validation)
    fallback = initial_prediction(training, q)
    columns = {}

    def column(program):
        if program not in columns:
            if not budget.spend(n + m):
                return None
            columns[program] = ([predict(program, row, fallback) for row in training],
                                [predict(program, row, fallback) for row in validation])
        return columns[program]

    def calibrated(program, fit_values, check_values):
        if not budget.spend(2 * n + m):
            return None
        offset = empirical_quantile([row.target - value for row, value in zip(training, fit_values)], q)
        result = Program("add", children=(program, Program("const", value=offset))).simplified()
        fit_values = [value + offset for value in fit_values]
        check_values = [value + offset for value in check_values]
        return {"program": result, "fit_values": fit_values, "check_values": check_values,
                "training_loss": mean(point_loss(value, row.target, q) for value, row in zip(fit_values, training)),
                "validation_loss": mean(point_loss(value, row.target, q) for value, row in zip(check_values, validation))}

    start = budget.spent
    base = baseline["candidate"].program
    values = column(base)
    incumbent = calibrated(base, *values) if values is not None else None
    if incumbent is None:
        return [], [], [{"status": "comparison_unavailable", "source_ids": [],
                         "compute_cost": budget.spent - start}]
    base_cost = budget.spent - start
    retained, claims, events, retained_sources = [], [], [], []
    # Existing recalled order is a hypothesis order, not a private task label.
    for call in calls:
        start = budget.spent
        sources = _sources(call)
        values = column(call)
        if values is None:
            break
        alternatives = []
        frozen = calibrated(call, *values)
        if frozen is not None:
            alternatives.append(frozen)
        # A source can contribute jointly with current coordinates even when
        # its old coefficients are wrong. Materialized predictions are shared.
        if budget.spend(2 * n):
            coefficients = _linear_fit([incumbent["fit_values"], values[0]],
                                       [row.target for row in training])
            terms = (incumbent["program"], call)
            program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
            train_values = [coefficients[0] + coefficients[1] * left + coefficients[2] * right
                            for left, right in zip(incumbent["fit_values"], values[0])]
            check_values = [coefficients[0] + coefficients[1] * left + coefficients[2] * right
                            for left, right in zip(incumbent["check_values"], values[1])]
            refit = calibrated(program, train_values, check_values)
            if refit is not None:
                alternatives.append(refit)
        if not alternatives:
            break
        best = min(alternatives, key=lambda item: item["validation_loss"])
        candidate_sources = _sources(best["program"])
        gains = [point_loss(left, row.target, q) - point_loss(right, row.target, q)
                 for row, left, right in zip(validation, incumbent["check_values"], best["check_values"])]
        blocks = trajectory_blocks(validation)
        block_gains, position = [], 0
        for block in blocks:
            block_gains.append(mean(gains[position:position + len(block)]))
            position += len(block)
        # Multiple trajectories, when present, are the independent units.
        # With one trajectory use its paired dispersion as a conservative
        # screening margin, not as final confirmation of a reusable method.
        units = block_gains if len(block_gains) > 1 else gains
        margin = max(tolerance, 3 * stdev(units) / sqrt(len(units))) if len(units) > 1 else tolerance
        gain = mean(gains)
        status = ("supported_for_research" if gain > margin else
                  "weak_evidence" if gain > tolerance else "no_marginal_gain")
        event = {"source_ids": sources, "against_source_ids": list(retained_sources),
                 "candidate_source_ids": candidate_sources,
                 "status": status, "baseline_loss": incumbent["validation_loss"],
                 "source_loss": best["validation_loss"], "gain": gain, "margin": margin,
                 "training_rows": n, "validation_rows": m, "validation_trajectories": len(blocks),
                 "fixed_quantile": q, "compute_cost": budget.spent - start,
                 "baseline_compute_cost": base_cost if not events else 0,
                 "evidence_basis": "current_unfitted_validation", "confirmed": False}
        events.append(event)
        if gain > tolerance:
            claims.append({"candidate": Candidate(f"risk_knowledge:{len(claims)}", best["program"],
                "knowledge_screen", "Compare a retained response before replaying its research procedure", tuple(candidate_sources)),
                "kind": "screened_risk_knowledge", "family_id": "T03", "method_source_ids": candidate_sources,
                "training_loss": best["training_loss"], "validation_loss": best["validation_loss"],
                "knowledge_adoption": event})
        if status == "supported_for_research":
            retained.append(call)
            retained_sources = candidate_sources
            incumbent = best
        # Keep observations about suitability with their source, without
        # turning a screening gain into a confirmed research-action reward.
        for prior in priors:
            if prior.get("source_concept_id") in sources:
                prior.setdefault("adoption_evidence", []).append(dict(event))
    return retained, claims, events
