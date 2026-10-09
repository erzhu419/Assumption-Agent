"""Offer a paid instrument question against a session's unused resources."""

from copy import deepcopy
from dataclasses import asdict

from .condition_marginal_value import (CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
                                       instrument_cycle_cost_estimate)
from .condition_research_returns import consider_condition_research
from .language import Program
from .models import Candidate, Probe
from .observed_condition_experiments import plan_observed_condition_probe
from .observed_condition_uncertainty import assess_observed_condition_uncertainty
from .research_resources import research_resource_share
from .unified_continuation import paid_view


def condition_experiment_history(actions):
    """Retain the actual experiments, including unsuccessful confirmation reads."""
    experiments, pending, confirmation = [], None, None
    for action in actions:
        if action["action"] == "plan_competing_observed_conditions":
            pending = action
        elif (action["action"] == "revise_competing_observed_conditions"
                and pending is not None and action.get("measured_rows", 0)):
            experiments.append({**pending, "research_return": action.get("research_return")})
            pending = None
        elif action["action"] == "revise_condition_on_paid_response" and action["changed_program"]:
            experiments.append({"research_return": {"changed_program": True}})
        elif action["action"] == "plan_condition_confirmation":
            confirmation = action
        elif action["action"] == "allocate_condition_confirmation" and confirmation is not None:
            experiments.extend({**confirmation, "research_return": outcome}
                               for outcome in action["outcomes"])
            confirmation = None
    return experiments


def screen_instrument_condition_revisit(agent, laboratory, previous, budget):
    """Plan once from purchased responses; a zero point loss does not erase risk."""
    tolerance = agent.config.minimum_improvement
    state = previous.get("condition_uncertainty")
    if (previous["status"] != "unresolved" or not previous["selected_channels"]
            or laboratory.spec.risk_quantile != .5 or not agent.config.active_probes
            or agent.config.strategy == "fixed" or state is None
            or state["status"] == "supported"
            or (state.get("expected_prediction_ambiguity") or 0.) <= tolerance
            or getattr(laboratory, "measurement_views", None)):
        return None
    channels = tuple(previous["selected_channels"])
    spec = agent._spec(laboratory, channels)
    rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
    selected = {"candidate": Candidate("instrument_condition_incumbent",
        Program.from_dict(previous["program"]), "retained_incumbent"),
        "research_program": deepcopy(previous.get("research_program")),
        "condition_uncertainty": deepcopy(state)}
    if state.get("program") != previous["program"] or state.get("fit_rows") != len(rows):
        state = assess_observed_condition_uncertainty(spec, rows, selected, budget,
            previous=state, tolerance=tolerance)
        selected["condition_uncertainty"] = state
    if (state is None or state["status"] != "unresolved"
            or state.get("original_target_loss") is None
            or state["original_target_loss"] > tolerance):
        return None
    alternatives = []
    for item in previous.get("continuation_state", {}).get("alternatives", []):
        if tuple(item["channels"]) != channels or item["program"] == previous["program"]:
            continue
        alternatives.append({**deepcopy(item), "candidate": Candidate(
            "retained_condition_competitor", Program.from_dict(item["program"]), "retained_alternative")})
    experiments = condition_experiment_history(previous["actions"])
    length = min(24, spec.max_probe_length)
    plan = plan_observed_condition_probe(spec, rows, selected, budget, finalists=alternatives,
        serial=len(previous["batches"]), length=length,
        previous_experiments=experiments, tolerance=tolerance)
    if plan is None:
        return None
    policy = condition_followup_policy(spec, rows, selected,
        agent.library.action_experience if agent.config.use_memory
        and agent.config.learn_research_policy else {})
    rate = sum(channel.cost_per_row for channel in laboratory.channels if channel.name in channels)
    # The existing measured-condition executor caps response reconstruction at
    # 80k. Preserve that cap, all paid-candidate comparisons, and a NEW frozen
    # response confirmation with its instrument-uncertainty assessment.
    revision_quote = 80000
    confirmation_quote = 30000 + 4 * length
    comparison_quote = len(rows) * (2 + len(alternatives))
    return {"program": deepcopy(previous["program"]), "paid_batches": len(previous["batches"]),
        "selected_channels": list(channels), "paid_rows": len(rows),
        "condition_uncertainty": deepcopy(state),
        "condition_research_returns": deepcopy(previous.get("condition_research_returns")),
        "previous_experiments": deepcopy(experiments),
        "plan": {**plan, "probe": asdict(plan["probe"])},
        "candidate": {"program": deepcopy(previous["program"]), "channels": list(channels),
            "fallback": previous["fallback"], "kind": "instrument_condition_revisit",
            "family_id": "T03", "source_concept_ids": [],
            "research_program": deepcopy(previous.get("research_program")),
            "condition_uncertainty": deepcopy(state),
            "condition_research_returns": deepcopy(previous.get("condition_research_returns"))},
        "measurement_rate": rate, "revision_compute_quote": revision_quote,
        "confirmation_compute_quote": confirmation_quote,
        "comparison_compute_quote": comparison_quote,
        "compute_cost": budget.spent, "tolerance": tolerance,
        "marginal_context": policy.action_context(CONDITION_FOLLOWUP_ACTION),
        "marginal_applicability": deepcopy(
            policy.action_record(CONDITION_FOLLOWUP_ACTION)["knowledge_applicability"]),
        "use_experience": agent.config.use_memory and agent.config.learn_research_policy}


def instrument_condition_offer(screen, remaining, *, experience=None, admission_resources=None):
    """Reprice cached evidence when the pool changes, without another screen."""
    if screen is None:
        return None
    plan = {**screen["plan"], "probe": Probe(screen["plan"]["probe"]["probe_id"],
        tuple(screen["plan"]["probe"]["sequence"]))}
    length, rate = plan["probe"].cost, screen["measurement_rate"]
    revision = screen["revision_compute_quote"] + screen["comparison_compute_quote"]
    confirmation = screen["confirmation_compute_quote"]
    # Deleted experience can reuse a numbered context for a different law.
    # Match its actual research content, including after JSON restoration.
    prefix = screen["marginal_context"].rsplit(":knowledge:", 1)[0] + ":knowledge:"
    record = next((actions[CONDITION_FOLLOWUP_ACTION]
        for context, actions in (experience or {}).items()
        if context.startswith(prefix) and CONDITION_FOLLOWUP_ACTION in actions
        and actions[CONDITION_FOLLOWUP_ACTION].get("knowledge_applicability")
            == screen["marginal_applicability"]), {}) if screen["use_experience"] else {}
    resources = {"observation_cost": 2 * length, "measurement_cost": 2 * length * rate,
                 "compute_cost": revision + confirmation}
    expected_costs = instrument_cycle_cost_estimate(record, plan, resources)
    decision = consider_condition_research(screen["condition_uncertainty"], plan,
        screen["condition_research_returns"], observation_remaining=remaining["observation_cost"],
        measurement_remaining=remaining["measurement_cost"], compute_remaining=remaining["compute_cost"],
        measurement_rate=rate, compute_quote=revision, observation_reserve=length,
        measurement_reserve=length * rate, compute_reserve=confirmation,
        previous_experiments=screen["previous_experiments"], marginal_record=record,
        compute_spent_before_decision=screen["compute_cost"], tolerance=screen["tolerance"],
        expected_cycle_costs=expected_costs)
    if (admission_resources is not None and not decision["buy"]
            and decision["reason"] == "retain_question_for_more_valuable_future_evidence"):
        # A chosen pair already justified both visits against this pool. Spending
        # on the first visit alone does not refute the second. Recheck its NEW
        # value and expected costs against that admission, then let the global
        # allocator compare it with current rivals using the actual remainder.
        admitted_share = research_resource_share(expected_costs, admission_resources)
        if decision["expected_marginal_value"] > admitted_share:
            decision.update(buy=True, local_buy=False,
                reason="selected_portfolio_companion", admission_resource_share=admitted_share)
    if not decision["buy"] or decision["expected_marginal_value"] <= 0.:
        return None
    risk = decision["predicted_response_risk_reduction"]
    # Express the existing forecast in response squared-error units. This is
    # prospective value, not an independently measured prediction improvement.
    expected_loss_reduction = max(risk or 0.,
        decision["empirical_marginal_prediction_gain"]
        * (decision["response_risk_normalizer"] or 0.))
    candidate = deepcopy(screen["candidate"])
    candidate["instrument_condition_experiment"] = {
        "plan": deepcopy(screen["plan"]), "decision": decision,
        "paid_batches": screen["paid_batches"], "program": deepcopy(screen["program"]),
        "confirmation_compute_quote": confirmation,
        "screening_compute_cost": screen["compute_cost"]}
    return {"candidate": candidate, "basis": "unresolved_instrument_condition",
        "old_evidence_gain": 0., "old_evidence_rows": screen["paid_rows"],
        "predicted_response_risk_reduction": risk,
        "expected_loss_reduction": expected_loss_reduction, "loss_kind": "squared_error",
        "relative_gain": decision["expected_marginal_value"] / max(decision["resource_share"], 1e-12),
        "minimum_resources": resources, "expected_resource_costs": expected_costs,
        "investment_decision": decision}


def execute_instrument_condition_revisit(agent, spec, selected, training, validation,
        finalists, budget, actions, experiment, **kwargs):
    """Execute the already priced probe and return for fresh confirmation."""
    from .condition_response_actions import research_observed_condition_competition

    funded = {**experiment, "plan": {**experiment["plan"],
        "probe": Probe(experiment["plan"]["probe"]["probe_id"],
                       tuple(experiment["plan"]["probe"]["sequence"]))}}
    return research_observed_condition_competition(agent, spec, selected, training,
        validation, finalists, budget, actions, funded_experiment=funded, **kwargs)
