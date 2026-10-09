"""Use the existing research-action policy on paid temporal inquiry evidence.

The policy chooses among executable directions, while the LLM still decides
whether to follow its recommendation. A recommendation is neither an executed
trial nor a new reward. Response and instrument plans are the existing planners'
plans and can be purchased and assessed through their ordinary interfaces.
"""

from copy import deepcopy
from math import isfinite

from .action_policy import ActionOption, ResearchActionPolicy
from .language import MissingHistory, Program
from .models import Observation, SearchBudget
from .temporal_instrument_research import (
    _specs, assess_temporal_instrument, select_temporal_instrument_probe,
    temporal_instrument_program,
)
from .temporal_response_experiments import plan_response_experiment


def _paid_predictions(records, hypotheses, budget, tolerance):
    """Compare frozen laws on actual measured fields, including measured history."""
    observations = []
    for record in records:
        readings = {item["observation_index"]: item["values"]
                    for item in record.get("measurements", ())}
        history = []
        for index, row in zip(record["row_indices"], record["rows"]):
            fields = {**row["features"], **readings.get(index, {})}
            observations.append(Observation(fields, row["target"], tuple(history)))
            history.append({**fields, "__target__": row["target"]})
    scored = []
    for hypothesis in hypotheses:
        program = Program.from_dict(hypothesis["program"])
        errors, missing = [], 0
        for row in observations:
            if not budget.spend():
                return scored, observations, False
            try:
                prediction = program.predict(row)
            except (MissingHistory, KeyError):
                missing += 1
                continue
            errors.append((prediction - row.target) ** 2 if isfinite(prediction) else float("inf"))
        mse = sum(errors) / len(errors) if errors else None
        scored.append({"hypothesis_id": hypothesis["hypothesis_id"],
            "path": hypothesis.get("path"), "rows": len(errors), "missing_rows": missing,
            "mse": mse, "max_squared_error": max(errors) if errors else None,
            "status": ("refuted" if errors and max(errors) > tolerance else
                       "unassessed" if not errors or missing else "fits_paid_evidence")})
    return scored, observations, True


def choose_temporal_research(records, hypotheses, public_spec, *, instrument=None,
                             methods=None, experience=None, observation_remaining,
                             compute_remaining, confirmation_rows=48, task_id=None,
                             serial=0, length=4, tolerance=1e-10,
                             finish_scoring_rows=None):
    """Diagnose paid evidence and quote the next direction through v0.93's policy.

    All records are learning evidence. Current instrument residuals are checked
    on a copy and cannot renew its prospective support. A paid-data tie excludes
    another fit as a resolution of that tie; the old competition planner must
    find a genuinely different prediction. Every acquisition quote retains the
    caller's separate final-confirmation allowance.
    """
    command, _ = _specs(public_spec)
    budget = SearchBudget(max(0, compute_remaining))
    scores, rows, complete = _paid_predictions(records, hypotheses, budget, tolerance)
    result = {"diagnosis": "unassessed", "hypotheses": scores,
        "paid_rows": len(rows), "recommendation": None, "response_plan": None,
        "instrument_plan": None, "compute_cost": budget.spent,
        "evidence_scope": "Paid learning evidence; no new observation or confirmation."}
    if not complete:
        result["diagnosis"] = "insufficient_compute_for_paid_comparison"
        return result
    if not rows:
        result["diagnosis"] = "no_paid_evidence"
        return result

    # The existing assessor separates failed calibration from failed response.
    # Check every current survivor; do not mutate the saved instrument's status.
    meter_records = [record for record in records if record.get("measurements")]
    meter_status = "unavailable"
    if instrument is not None and meter_records:
        measured_rows = sum(len(record["measurements"]) for record in meter_records)
        live = [hypothesis for field in instrument.get("fields", {}).values()
                for hypothesis in field.get("hypotheses", [])
                if hypothesis.get("status") not in {"refuted", "counterexample"}]
        if measured_rows * len(live) > budget.remaining:
            result.update(diagnosis="insufficient_compute_for_calibration_comparison",
                          compute_cost=budget.spent)
            return result
        current = deepcopy(instrument)
        assessment = assess_temporal_instrument(current, meter_records, public_spec,
            compute_budget=budget.remaining, tolerance=tolerance)
        budget.spend(assessment["compute_cost"])
        losses = [loss for field in assessment.get("field_losses", {}).values()
                  for loss in field.values()]
        meter_status = ("counterexample" if "meter" in assessment["refuted_fields"]
                        else "consistent" if losses else "unavailable")
        result["paid_meter_assessment"] = assessment
    result["paid_meter_status"] = meter_status

    by_id = {item["hypothesis_id"]: item for item in hypotheses}
    survivors = [item for item in scores if item["status"] == "fits_paid_evidence"]
    unassessed = any(item["status"] == "unassessed" for item in scores)
    options, recommendations = [], {}
    finish_rows = len(rows) if finish_scoring_rows is None else finish_scoring_rows

    def offer(name, recommendation, *, probe=None, compute=0):
        probe_rows = len(probe["sequence"]) if probe else 0
        observation_cost = probe_rows + confirmation_rows
        options.append(ActionOption(name, observation_cost,
            compute + finish_rows + probe_rows + confirmation_rows, payload=(None, {})))
        recommendations[name] = recommendation

    if meter_status == "counterexample":
        result["diagnosis"] = "instrument_counterexample"
        offer("remeasure_inputs", {"operator": "instrument",
            "reason": "Refit the instrument on the already purchased counterexample before binding responses."},
            compute=len(rows))
    elif (instrument is not None and temporal_instrument_program(instrument) is None
          and meter_status == "consistent"):
        result["diagnosis"] = "instrument_ambiguity"
        plan = select_temporal_instrument_probe(instrument, public_spec, methods=methods,
            max_length=min(public_spec["max_probe_length"],
                           max(0, observation_remaining - confirmation_rows)),
            serial=serial, compute_budget=budget.remaining, tolerance=tolerance)
        budget.spend(plan["compute_cost"])
        result["instrument_plan"] = plan
        if plan.get("action") is not None:
            offer("discriminate_inputs", {"operator": "disagreement"}, probe=plan["action"],
                  compute=len(plan["action"]["sequence"]) * len(live))
    elif len(survivors) > 1:
        result["diagnosis"] = "response_ambiguity"
        plan = plan_response_experiment([by_id[item["hypothesis_id"]] for item in survivors],
            public_spec, methods=methods, instrument=instrument, serial=serial, length=length,
            compute_budget=budget.remaining, task_id=task_id)
        budget.spend(plan["compute_cost"])
        result["response_plan"] = plan
        if plan.get("action") is not None:
            name = ("discriminate_response" if plan["forecast_status"] == "predicted_distinction"
                    else "remeasure_inputs")
            offer(name, {"operator": "response_disagreement"}, probe=plan["action"],
                  compute=len(plan["action"]["sequence"]) * (len(survivors) + len(plan["binding"])))
    elif len(survivors) == 1 and not unassessed:
        result["diagnosis"] = "one_response_survives_paid_evidence"
        survivor = survivors[0]
        offer("retain_response", {"operator": "retain_response",
            "hypothesis_id": survivor["hypothesis_id"], "path": survivor["path"],
            "reason": "Retain this explanation and use the separate final confirmation."})
    elif unassessed:
        result["diagnosis"] = "response_evidence_unavailable"
    elif scores:
        result["diagnosis"] = "response_counterexample"
        offer("revise_mechanism", {"operator": "response",
            "reason": "All supplied response laws fail paid evidence; revise the response family."},
            compute=len(rows))
    else:
        result["diagnosis"] = "no_response_candidates"

    policy = ResearchActionPolicy(rows, command, experience if experience is not None else {})
    policy.set_condition("temporal:" + result["diagnosis"])
    choice = policy.choose(options, observation_remaining=observation_remaining,
        compute_remaining=budget.remaining, minimum_improvement=tolerance)
    result.update(decision={**policy.last_decision,
        "options": [{"name": option.name, "observation_cost": option.observation_cost,
                     "compute_cost": option.compute_cost} for option in options]},
        recommendation=recommendations[choice.name] if choice is not None else None,
        compute_cost=budget.spent)
    return result
