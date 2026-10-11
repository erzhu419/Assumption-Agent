"""Read paid root01 position7 traces only, after every formal chain has learned."""
import gzip
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / "artifacts/background_research_investment_20261003"
POSITION = 7
ACTION = "condition_followup_research"


def read(path):
    contents = path.read_bytes()
    return json.loads(gzip.decompress(contents) if path.suffix == ".gz" else contents)


def scalar_fields(item):
    return {key: value for key, value in item.items() if not isinstance(value, (dict, list))}


def first_task_state(directory, final):
    # Only a logarithmic number of local snapshots is read; no new experiment.
    lower, upper = 1, final["study"]["steps"]
    cache = {upper: final}
    while lower < upper:
        middle = (lower + upper) // 2
        snapshot = cache.setdefault(middle, read(directory / f"visit{middle:02d}.json.gz"))
        if len(snapshot["results"]) > POSITION:
            upper = middle
        else:
            lower = middle + 1
    first = cache.get(lower) or read(directory / f"visit{lower:02d}.json.gz")
    before = cache.get(lower - 1) or read(directory / f"visit{lower - 1:02d}.json.gz")
    return lower, first, before


def questions(report):
    state = report.get("condition_uncertainty") or {}
    return {"status": report["status"], "costs": report["costs"],
        "paid_validation_loss": report.get("validation_loss"),
        "paid_confirmation_loss": report.get("confirmation_loss"),
        "condition_status": state.get("status"),
        "paid_response_loss": state.get("original_target_loss"),
        "paid_prediction_ambiguity": state.get("expected_prediction_ambiguity"),
        "remaining_brackets": [{"field": item["condition_field"],
            "intervals": item.get("threshold_intervals", [])}
            for item in state.get("conditions", [])]}


def purchases(report):
    return [{"action_index": index, **scalar_fields(action),
        "fixed_control_background": action.get("fixed_control_background")}
        for index, action in enumerate(report["actions"])
        if action["action"] == "choose_condition_research_investment"]


def paid_outcomes(report):
    return [{"action_index": index, "measured_rows": action.get("measured_rows"),
        **scalar_fields(action.get("research_return") or {})}
        for index, action in enumerate(report["actions"])
        if action["action"] == "revise_competing_observed_conditions"]


def confirmations(report):
    return [{"action_index": index, **scalar_fields(action)}
        for index, action in enumerate(report["actions"])
        if ("confirm" in action["action"] or action["action"] == "validate_continued_program")
        and action.get("probe_id")]


def extract(base, arm, sampling_applicability):
    directory = base / "successor_regression/runs" / arm / "root01"
    final = read(directory / "checkpoint.json.gz")
    if not final["study"]["learning_complete"]:
        raise RuntimeError(f"Learning incomplete: {directory}")
    step, first, before = first_task_state(directory, final)
    initial_report = first["results"][POSITION]["research"]
    final_report = final["results"][POSITION]["research"]
    prior = ({} if arm == "no_marginal_history" else before["knowledge"]["action_experience"])
    weights = []
    decision = None
    for index, action in enumerate(initial_report["actions"]):
        if action["action"] == "choose_condition_research_investment":
            decision = action
        if action["action"] != "plan_competing_observed_conditions" or decision is None:
            continue
        current = action.get("sampling_context")
        record = prior.get(decision["marginal_context"], {}).get(ACTION, {})
        entries = [{"weight": sampling_applicability(source["sampling_context"], current),
            "past_events": source["information_event_sum"],
            "past_predicted_events": source["predicted_information_event_sum"],
            "batches": source["batches"],
            "normalized_background": (source["sampling_context"] or {}).get("normalized_background"),
            "reading_forecast": (source["sampling_context"] or {}).get("reading_forecast")}
            for source in record.get("sampling_information", [])]
        weights.append({"action_index": index,
            "fixed_control_background": action["fixed_control_background"],
            "paid_reading_rows": action.get("paid_reading_rows"),
            "recorded_information_yield_including_current_task": decision.get("information_yield"),
            "reading_forecast": (current or {}).get("reading_forecast"),
            "prior_task_entries_only": entries})
    revisits = [visit for visit in final["revisits"] if visit["task_position"] == POSITION]
    return {"first_task_visit_step": step, "first_visit": questions(initial_report),
        "final": questions(final_report), "first_visit_investments": purchases(initial_report),
        "first_visit_paid_outcomes": paid_outcomes(initial_report),
        "first_visit_confirmations": confirmations(initial_report),
        "actual_prior_task_applicability_for_purchased_plans": weights,
        "revisits": [{"basis": visit.get("basis"), "costs": visit["visit_costs"],
            "before": questions(visit["previous"])} for visit in revisits],
        "revisit_investments": [item for item in purchases(final_report)
            if item["action_index"] >= len(initial_report["actions"])],
        "revisit_confirmations": [item for item in confirmations(final_report)
            if item["action_index"] >= len(initial_report["actions"])],
        "final_condition_followup_comparisons": [{**scalar_fields(item),
            "stage_costs": item.get("stage_costs")} for item in
            (final_report.get("continuation") or {}).get("method_marginal_comparisons", [])],
        "chain_costs": final["costs"]}


def main():
    jobs = read(BASE / "learn_candidate_jobs.json")
    if len(jobs) != 36 or any(job["returncode"] for job in jobs):
        raise RuntimeError("Wait for all 36 formal learning jobs to finish")
    sys.path.insert(0, str(BASE / "runtime/src"))
    from hegel_machine.cognition.condition_marginal_value import sampling_applicability
    output = {"case": "successor_regression/root01/position7",
        "evidence": "Paid observations and saved decisions only; no terminal evaluator or new acquisitions.",
        "weight_scope": "Exact saved contexts of purchased plans versus knowledge present before this task. Rejected-plan decisions retain their recorded aggregate yield; their unsaved contexts are not invented.",
        "arms": {label: extract(base, arm, sampling_applicability) for label, base, arm in (
            ("old_163_keep", OLD, "new_cumulative"),
            ("old_163_delete", OLD, "no_marginal_history"),
            ("new_164_keep", BASE, "new_cumulative"),
            ("new_164_delete", BASE, "no_marginal_history"))}}
    path = BASE / "root01_paid_knowledge.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print({"output": str(path), "arms": len(output["arms"])})


if __name__ == "__main__":
    main()
