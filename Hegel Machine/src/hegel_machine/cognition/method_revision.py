"""Turn paid, unsuccessful parameter repairs into a different research step."""

from copy import deepcopy


def repair_support(profile):
    """Evidence relevant to another fit of the *same* executable coordinates.

    Extending a trajectory beyond a fixed lag does not add a parameter to
    that program. New control support or a changed executable basis can make
    another fit informative. Continuous states still use the full history.
    """
    data, representation = profile.get("data", {}), profile.get("representation", {})
    depth = data.get("max_history", 0)
    if "continuous" not in representation.get("families", []):
        depth = min(depth, representation.get("history_depth", 0))
    return {"objective": deepcopy(profile.get("objective")),
            "structure": deepcopy(representation.get("structure")),
            "history_depth": depth,
            "data": {key: deepcopy(data.get(key)) for key in (
                "feature_dimensions", "control_dimensions", "coverage",
                "linear_data_support", "object_fields", "independent_trajectories")}}


def source_revision(profile, trace, source_id):
    """Summarize actual failed source fits, without importing their source ID.

    A failed source fit followed by a completed unproductive ordinary refit
    is already a repeated parameter attempt. Empty searches and unfinished
    work supply no evidence that the current representation needs replacing.
    """
    if not source_id:
        return None
    residual = profile.get("residual", {})
    if residual.get("state") != "unresolved" and not residual.get("new_counterexample"):
        return None
    support = repair_support(profile)
    failures, source_seen = 0, False
    for step in trace:
        action = step.get("action")
        if action == "refit_transfer" and step.get("source_id") == source_id:
            if step.get("residual_resolved"):
                failures, source_seen = 0, False
                continue
            if (not step.get("execution_complete", True) or not step.get("candidate_count")
                    or not step.get("profile") or repair_support(step["profile"]) != support):
                continue
            failures += 1
            source_seen = True
        elif (action == "refit" and source_seen and step.get("execution_complete", True)
              and (step.get("execution_profile") or step.get("profile"))
              and repair_support(step.get("execution_profile") or step["profile"]) == support):
            if step.get("gain", 0.) > 1e-8:
                failures, source_seen = 0, False
            else:
                failures += 1
    if not failures:
        return None
    return {"status": "parameter_repair_exhausted" if failures >= 2 else "parameter_repair_unresolved",
            "failed_attempts": failures, "support": support}


def revision_condition(profile):
    state = (profile or {}).get("source_revision") or {}
    if state.get("status") != "parameter_repair_exhausted":
        return None
    # Attempt counts and source IDs are not transfer conditions. The reusable
    # method requires the same failed coordinate family and acquired support.
    return {"status": state["status"], "support": deepcopy(state["support"])}


def distinguishing_experiment(option):
    evidence = option.payload[1] if option.payload else {}
    return bool(evidence.get("candidate_disagreement", 0) > 0
                or evidence.get("expected_risk_reduction", 0) > 0
                or evidence.get("boundary_support") and evidence.get("unresolved_boundary_width", 0) > 0)
