"""Let a frozen interaction explanation share the base model's future evidence."""

from copy import deepcopy

from .decision_experiments import _predict
from .validation_continuation import evidence_summary


def prepare_interaction_prediction(spec, research, point, budget):
    """Price a challenger prediction before the shared control is observed."""
    challenger = (research.get("candidate") or {}).get("interaction_challenger")
    if challenger is None or not budget.spend(1):
        return {}
    return {"interaction_mean": float(_predict(spec, challenger, point))}


def supported_interaction(research, validation, stage, *, base_evidence):
    """Compare the challenger with the incumbent and any supported alternative.

    Missing predictions remain missing. In particular, rebuilding the incumbent
    cannot create historical forecasts for this frozen challenger.
    """
    candidate = research.get("candidate") or {}
    challenger = candidate.get("interaction_challenger")
    rows = [row for row in validation if "interaction_error" in row]
    if challenger is None or len(rows) < 4:
        return None
    against_incumbent = evidence_summary([
        {"candidate_error": row["interaction_error"], "previous_error": row["previous_error"]}
        for row in rows])
    against_base = evidence_summary([
        {"candidate_error": row["interaction_error"], "previous_error": row["candidate_error"]}
        for row in rows])
    model = deepcopy(challenger)
    if "training_end" in candidate:
        model["training_end"] = candidate["training_end"]
    return {
        "supported": (against_incumbent["decision"] == "supported"
                      and (base_evidence["decision"] != "supported"
                           or against_base["decision"] == "supported")),
        "model": model,
        "validation": [{**deepcopy(row), "candidate_error": row["interaction_error"]}
                       for row in rows],
        "comparison": {"stage": int(stage), "indices": [row["global_index"] for row in rows],
                       "comparison_rows": 2 * len(rows), "against_base": against_base,
                       "against_incumbent": against_incumbent,
                       "base_evidence": deepcopy(base_evidence)},
    }
