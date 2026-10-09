"""Buy an affordable amount of evidence for a frozen response's reading cut."""

from copy import deepcopy

from .condition_research_returns import (consider_condition_research,
    plan_condition_research_allocation, record_condition_research)
from .condition_marginal_value import (CONDITION_FOLLOWUP_ACTION,
    condition_followup_policy, record_condition_information)
from .language import Program
from .models import Candidate, Probe, SearchBudget
from .observed_condition_experiments import (plan_observed_condition_probe,
                                             refresh_condition_sampling)
from .observed_condition_uncertainty import assess_observed_condition_uncertainty


def pair_confirmation_reopen_required(actions, probe_id=None):
    """Return whether a fresh pair batch missed both frozen responses.

    This is deliberately read from the paid action receipt.  A persisted
    ``pair_confirmation_returns`` flag alone would make an old mismatch look
    like a new one after JSON restoration and could repeatedly reopen the same
    response question without buying evidence.
    """
    for event in reversed(actions or ()):
        if event.get("action") != "allocate_pair_confirmation":
            continue
        for outcome in reversed(event.get("outcomes", ())):
            if (outcome.get("common_response_mismatch")
                    and (probe_id is None or outcome.get("probe_id") == probe_id)):
                return True
        if probe_id is not None:
            # A newer pair allocation has superseded older receipts.
            return False
    return False


def _acquire_pair_confirmation(spec, selected, plan, budget, actions, *,
        acquire, remaining, rows_for, channels, measurement_rate, tolerance):
    """Buy a funded first batch, then price fresh evidence for this same pair."""
    from .pair_confirmation_investment import (consider_pair_confirmation,
                                               record_pair_confirmation)

    pair = deepcopy(plan["confirmation_pair"])
    baseline = {"candidate": Candidate("confirmation_baseline",
        Program.from_dict(pair["baseline_program"]), "explanation_before_research"),
        "fallback": pair["baseline_fallback"], "channels": channels}
    frozen = {**selected, "candidate": Candidate("frozen_confirmation",
        Program.from_dict(pair["program"]), "response_revision"),
        "fallback": pair["fallback"]}
    length = plan["probe"].cost
    original_id = plan["probe"].probe_id
    history = selected.get("pair_confirmation_returns")
    event = {"action": "allocate_pair_confirmation", "confirmation_pair": pair,
        "acquired_rows": 0, "probe_ids": [], "outcomes": [], "decisions": [],
        "planning_compute_cost": 0, "evaluation_compute_cost": 0}
    actions.append(event)
    validation = None
    while True:
        current = acquire(channels, "confirm_continuation_program", probe=plan["probe"])
        if current is None:
            event["stop_reason"] = "remaining_resources"
            break
        validation = current
        event["acquired_rows"] += len(current)
        event["probe_ids"].append(plan["probe"].probe_id)
        # The caller still compares its incumbent and settles this batch. Do
        # not consume that already funded evaluation when scoring the pair.
        work = SearchBudget(max(0, budget.remaining - 2 * len(current)))
        measured = record_pair_confirmation(history, plan, current, work,
            measurement_rate=measurement_rate, tolerance=tolerance)
        budget.spend(work.spent)
        event["evaluation_compute_cost"] += work.spent
        if measured is None:
            event["stop_reason"] = "return_batch_for_funded_settlement"
            break
        history = measured
        selected["pair_confirmation_returns"] = history
        outcome = {**deepcopy(history["last_outcome"]), "probe_id": plan["probe"].probe_id}
        event["outcomes"].append(outcome)
        if outcome["common_response_mismatch"]:
            event["stop_reason"] = "reopen_response_model_after_common_mismatch"
            event["reopen_required"] = True
            break
        if outcome["distinguished"] or outcome["response_counterexample"]:
            event["stop_reason"] = "return_measured_pair_evidence"
            break
        # Leave the final independent comparison and the existing condition
        # assessment affordable. Other tasks' reserves are outside this visit.
        reserve = 15000 + 2 * length
        paid_loss = max(selected.get("training_loss") or 0.,
                        selected.get("validation_loss") or 0., outcome["selected_loss"])
        # Replanning is itself a prospective investment. Price its complete
        # work before spending it, using the currently paid best experiment.
        investment = consider_pair_confirmation(plan, history, paid_loss=paid_loss,
            **remaining(), measurement_rate=measurement_rate,
            compute_quote=15000 + 2 * length, compute_reserve=reserve,
            tolerance=tolerance)
        event["decisions"].append({**investment, "stage": "before_replanning"})
        if not investment["buy"]:
            event["stop_reason"] = investment["reason"]
            break
        work = SearchBudget(min(15000, budget.remaining - reserve - 2 * length))
        updated = plan_observed_condition_probe(spec, rows_for(channels), frozen, work,
            serial=f"{original_id}:pair{len(event['probe_ids'])}", length=length,
            tolerance=tolerance, confirmation_baseline=baseline)
        budget.spend(work.spent)
        event["planning_compute_cost"] += work.spent
        if updated is None:
            event["stop_reason"] = "no_funded_pair_experiment"
            break
        decision = consider_pair_confirmation(updated, history, paid_loss=paid_loss,
            **remaining(), measurement_rate=measurement_rate, compute_quote=2 * length,
            compute_reserve=reserve, tolerance=tolerance)
        decision.update(stage="after_replanning", planning_compute_cost=work.spent,
            changed_background=updated["probe"].sequence != plan["probe"].sequence)
        event["decisions"].append(decision)
        if not decision["buy"]:
            event["stop_reason"] = decision["reason"]
            break
        plan = updated
        actions.append({"action": "plan_condition_confirmation",
            **{key: value for key, value in plan.items() if key != "probe"},
            "probe_id": plan["probe"].probe_id, "planning_compute_cost": work.spent})
    return validation


def acquire_condition_confirmation(spec, selected, plan, budget, actions, *,
        acquire, remaining, rows_for, channels, measurement_rate, tolerance,
        experience=None):
    """Keep the law frozen across batches; return the last independent batch.

    Earlier batches remain paid evidence. A counterexample or narrower bracket
    ends this allocation so the caller can revise before a new confirmation.
    """
    if plan.get("confirmation_pair") is not None:
        return _acquire_pair_confirmation(spec, selected, plan, budget, actions,
            acquire=acquire, remaining=remaining, rows_for=rows_for, channels=channels,
            measurement_rate=measurement_rate, tolerance=tolerance)
    length = plan["probe"].cost
    state = selected["condition_uncertainty"]
    policy = condition_followup_policy(spec, rows_for(channels), selected,
                                      experience if experience is not None else {})
    available = remaining()
    # Reserve a new confirmation and the existing condition-refit allowance.
    # The current frozen law still receives its funded first batch if that
    # complete extra research cycle cannot be afforded.
    reserve_compute = 20000 + 4 * length
    allocation = plan_condition_research_allocation(state, plan,
        observation_remaining=available["observation_remaining"],
        measurement_remaining=available["measurement_remaining"],
        compute_remaining=available["compute_remaining"],
        measurement_rate=measurement_rate, setup_observation_cost=length,
        setup_compute_cost=reserve_compute, batch_compute_cost=15000 + 2 * length,
        previous=selected.get("condition_research_returns"), tolerance=tolerance)
    planned_rows = allocation["selected_rows"] if allocation is not None else length
    event = {"action": "allocate_condition_confirmation", "planned_rows": planned_rows,
        "allocation": allocation, "program": selected["candidate"].program.to_dict(),
        "confirmation_reserve": length, "revision_compute_reserve": reserve_compute,
        "acquired_rows": 0, "probe_ids": [], "outcomes": [], "decisions": [],
        "sampling_updates": [],
        "marginal_context": policy.action_context(CONDITION_FOLLOWUP_ACTION)}
    actions.append(event)
    validation = None
    for batch_index in range(planned_rows // length):
        sampling_cost = 0
        if batch_index:
            # A miss is also a paid reading of the instrument. Update its
            # sampling model before pricing another batch at the same controls.
            if plan.get("sampling_context") is not None:
                work = SearchBudget(min(15000, max(0, budget.remaining
                    - reserve_compute - 15000 - 2 * length)))
                updated = refresh_condition_sampling(spec, rows_for(channels), plan, work)
                sampling_cost = work.spent
                budget.spend(sampling_cost)
                event["sampling_updates"].append({
                    "compute_cost": sampling_cost, "updated": updated is not None,
                    "paid_reading_rows": len(rows_for(channels)),
                    "interval_exposure": deepcopy(updated["interval_exposure"]) if updated else []})
                if updated is None:
                    event["stop_reason"] = "sampling_model_update_unfunded"
                    break
                plan = updated
            available = remaining()
            decision = consider_condition_research(state, plan,
                selected.get("condition_research_returns"),
                observation_remaining=available["observation_remaining"],
                measurement_remaining=available["measurement_remaining"],
                compute_remaining=available["compute_remaining"],
                measurement_rate=measurement_rate, compute_quote=15000 + 2 * length,
                observation_reserve=length, measurement_reserve=length * measurement_rate,
                compute_reserve=reserve_compute, tolerance=tolerance,
                marginal_record=policy.action_record(CONDITION_FOLLOWUP_ACTION),
                compute_spent_before_decision=sampling_cost)
            event["decisions"].append(decision)
            if not decision["buy"]:
                event["stop_reason"] = decision["reason"]
                break
        probe = (plan["probe"] if batch_index == 0 else Probe(
            f"{plan['probe'].probe_id}:batch{batch_index}", plan["probe"].sequence))
        current = acquire(channels, "confirm_continuation_program", probe=probe)
        if current is None:
            event["stop_reason"] = "remaining_resources"
            break
        validation = current
        event["acquired_rows"] += len(current)
        event["probe_ids"].append(probe.probe_id)
        # Assess all original responses, preserving every earlier contradiction.
        work = SearchBudget(min(15000, max(0, budget.remaining - reserve_compute
            - 2 * len(current))))
        after = assess_observed_condition_uncertainty(spec, rows_for(channels),
            selected, work, previous=state, tolerance=tolerance) if work.remaining else None
        budget.spend(work.spent)
        if after is None or after.get("original_target_loss") is None:
            event["stop_reason"] = "condition_assessment_unfunded"
            break
        returns = record_condition_research(selected.get("condition_research_returns"),
            state, after, plan, observation_cost=len(current),
            measurement_cost=len(current) * measurement_rate,
            compute_cost=work.spent + sampling_cost,
            tolerance=tolerance)
        selected["condition_research_returns"] = returns
        record_condition_information(policy, returns["last_outcome"])
        selected["condition_uncertainty"] = after
        event["outcomes"].append(deepcopy(returns["last_outcome"]))
        state = after
        if returns["last_outcome"]["informative"]:
            event["stop_reason"] = "new_condition_evidence"
            break
        if after["status"] == "supported":
            event["stop_reason"] = "condition_supported"
            break
    event.setdefault("stop_reason", "priced_allocation_exhausted")
    return validation
