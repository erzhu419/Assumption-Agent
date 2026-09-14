"""Accumulate paid trials of two frozen prediction and calibration plans."""

from copy import deepcopy
from math import sqrt
from statistics import mean, stdev

from .language import Program
from .risk_search import trajectory_blocks
from .uncertainty import action_and_interval, decision_loss


def _summarize(state):
    comparison = state["comparison"]
    trials = comparison["trajectory_comparisons"]
    count = len(trials)
    comparison.update(rows=sum(trial["rows"] for trial in trials), trajectories=count,
        compute_cost=sum(trial["compute_cost"] for trial in trials),
        selected="previous", adopted=False, before_loss=None, after_loss=None,
        gain=None, standard_error=None, support_margin=None)
    state["status"] = "continue"
    if not trials:
        comparison["reason"] = (state.get("last_update_reason") or "no_compared_trajectories")
        return state
    gains = [trial["gain"] for trial in trials]
    gain = mean(gains)
    comparison.update(before_loss=mean(trial["before_loss"] for trial in trials),
        after_loss=mean(trial["after_loss"] for trial in trials), gain=gain)
    if count < 2:
        comparison["reason"] = "insufficient_trajectory_support"
        return state
    standard_error = stdev(gains) / sqrt(count)
    margin = max(comparison["minimum_improvement"], 2 * standard_error)
    comparison.update(standard_error=standard_error, support_margin=margin)
    if gain > margin:
        comparison.update(selected="candidate", adopted=True, reason="supported_paired_gain")
        state["status"] = "adopted"
    elif all(trial.get("maximum_action_difference", float("inf")) <= 1e-12 for trial in trials):
        comparison["reason"] = "no_observed_action_difference"
        state["status"] = "retained"
    elif max(gains) <= comparison["minimum_improvement"]:
        comparison["reason"] = "no_useful_trajectory_gain"
        state["status"] = "retained"
    elif gain + 2 * standard_error <= comparison["minimum_improvement"]:
        comparison["reason"] = "no_supported_positive_possibility"
        state["status"] = "retained"
    else:
        comparison["reason"] = "unresolved_paired_gain"
    return state


def begin_decision_evidence(previous_program, previous_model, candidate_program, candidate_model,
                            *, fallback, costs, minimum_improvement=.002, comparison=None):
    """Freeze both complete plans and optionally import an already paid trial.

    ``comparison`` is a completed ``select_decision_plan`` result. Its actual
    trajectory summaries are sufficient: importing them performs no prediction
    and spends no new computation. Reported rows that were never scored because
    the earlier comparison lacked budget do not become measured evidence.
    """
    imported = comparison or {}
    trials = [{**deepcopy(trial), "compute_cost": 2 * trial["rows"]}
              for trial in imported.get("trajectory_comparisons", ())]
    state = {
        "previous_program": previous_program.to_dict(), "previous_model": deepcopy(previous_model),
        "candidate_program": candidate_program.to_dict(), "candidate_model": deepcopy(candidate_model),
        "fallback": fallback, "costs": list(costs),
        "comparison": {"trajectory_comparisons": trials,
            "fixed_costs": list(costs), "minimum_improvement": minimum_improvement,
            "comparison_scope": "paired_independent_followup_trajectories",
            "program_changed": previous_program != candidate_program, "observation_cost": 0},
        "last_compute_cost": 0, "last_rows": 0, "last_trajectory_comparisons": [],
        "last_update_reason": imported.get("reason", "no_compared_trajectories"),
        "uncompared_rows": max(0, imported.get("rows", 0) - sum(trial["rows"] for trial in trials)),
    }
    return _summarize(state)


def extend_decision_evidence(state, new_rows, *, budget):
    """Score only the newly supplied, complete trajectories of the frozen pair.

    The caller passes observations acquired after freezing these plans and
    excludes rows already represented by the saved trajectory summaries.
    An unaffordable batch is not evaluated or charged; the caller retains those
    acquired rows until their comparison can be funded.
    """
    updated = deepcopy(state)
    updated.update(last_compute_cost=0, last_rows=0, last_trajectory_comparisons=[])
    blocks = trajectory_blocks(new_rows)
    if not blocks:
        updated["last_update_reason"] = "no_new_trajectories"
        return _summarize(updated)
    cost = 2 * len(new_rows)
    if not budget.spend(cost):
        updated["last_update_reason"] = "insufficient_budget"
        return _summarize(updated)
    previous = Program.from_dict(updated["previous_program"])
    candidate = Program.from_dict(updated["candidate_program"])
    costs, fallback = updated["costs"], updated["fallback"]
    measured = []
    for block in blocks:
        before_losses, after_losses, action_differences = [], [], []
        for row in block:
            before = action_and_interval(previous, row, fallback, updated["previous_model"], costs)[0]
            after = action_and_interval(candidate, row, fallback, updated["candidate_model"], costs)[0]
            before_losses.append(decision_loss(before, row.target, costs))
            after_losses.append(decision_loss(after, row.target, costs))
            action_differences.append(abs(before - after))
        before, after = mean(before_losses), mean(after_losses)
        measured.append({"rows": len(block), "before_loss": before, "after_loss": after,
            "gain": before - after, "compute_cost": 2 * len(block),
            "action_disagreement": mean(action_differences),
            "maximum_action_difference": max(action_differences)})
    updated["comparison"]["trajectory_comparisons"].extend(measured)
    updated.update(last_compute_cost=cost, last_rows=len(new_rows),
        last_trajectory_comparisons=deepcopy(measured), last_update_reason="measured_new_trajectories",
        uncompared_rows=max(0, updated["uncompared_rows"] - len(new_rows)))
    return _summarize(updated)


def decision_evidence_quote(state, *, trajectory_length, observation_remaining, compute_remaining):
    """Price another two paired trajectories only while the choice is unresolved.

    This quote covers observation acquisition and paired comparison. The caller
    adds its actual reporting or feedback costs. A pending question retains
    exactly the same frozen plans and paid evidence for a later funded visit.
    """
    unresolved = state["status"] == "continue"
    quote = {"status": state["status"], "needs_evidence": unresolved,
        "reason": state["comparison"]["reason"], "payable": False,
        "observation_cost": 0, "compute_cost": 0, "probe_lengths": []}
    if not unresolved:
        return quote
    if trajectory_length <= 0:
        quote.update(status="pending", reason="no_available_trajectory")
        return quote
    rows = 2 * trajectory_length
    payable = rows <= observation_remaining and 2 * rows <= compute_remaining
    quote.update(status="continue" if payable else "pending", payable=payable,
        observation_cost=rows, compute_cost=2 * rows,
        probe_lengths=[trajectory_length, trajectory_length])
    if not payable:
        quote["reason"] = "insufficient_budget"
    return quote
