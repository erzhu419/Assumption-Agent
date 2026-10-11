#!/usr/bin/env python3
"""Summarize fixed-quantile risk scores and actual method visits separately from numeric MSE."""
from collections import Counter
import json
from pathlib import Path
import sys

base = Path(sys.argv[1])
output = base / "risk"
protocol = json.loads((output / "protocol.json").read_text())
cost_keys = ("observation_cost", "measurement_cost", "compute_cost")
result = {"risk_quantile": .9, "primary_endpoint": "pinball", "solved_threshold": None,
    "external_model_calls": 0, "cohorts": {}}
for cohort in protocol["cohorts"]:
    summary = json.loads((output / cohort / "summary.json").read_text())["families"]["risk"]
    entry = {"development_only": cohort == "development", "arms": {}, "comparisons": {}}
    for arm, data in summary["arms"].items():
        actions, replays, rewrites, methods, quotes, choices = Counter(), [], [], [], [], []
        feedback, adopted_methods = [], []
        terminal_cpu, saved_methods = 0., 0
        cold_costs, cold_cpu = Counter(), 0.
        for seed in protocol["cohorts"][cohort]:
            path = output / cohort / "runs/risk" / arm / str(seed) / "checkpoint.json"
            state = json.loads(path.read_text())
            terminal_cpu += state["study_terminal_cpu_seconds"]
            for visit in state["study_visits"]:
                if visit["first_visit"] and visit["position"] == 0:
                    cold_costs.update(visit["incremental_costs"])
                    cold_cpu += visit["cpu_seconds"]
            drafts = {item["draft_id"]: item for item in state["knowledge"]["research_drafts"]}
            saved_methods += sum(item.get("risk_method") is not None for item in drafts.values())
            for record in state["results"]:
                recorded_actions = record["research"]["actions"]
                for index, action in enumerate(recorded_actions):
                    actions[action["action"]] += 1
                    if action["action"] == "risk_research_feedback":
                        feedback.append({"root_seed": seed, "position": record["task_position"],
                            "action_index": index, **{key: action[key] for key in
                            ("direction", "status", "confirmed", "risk_quantile", "probe_id",
                             "before_loss", "after_loss", "observation_rows", "observation_cost",
                             "measurement_cost", "compute_cost")}})
                    elif action["action"] == "choose_risk_method_on_paid_response":
                        adopted_methods.append({"root_seed": seed, "position": record["task_position"],
                            "action_index": index, **{key: action[key] for key in
                            ("draft_id", "source_task_id", "training_loss", "validation_loss") if key in action}})
                    if action["action"] == "choose_risk_research":
                        choices.append({"root_seed": seed, "position": record["task_position"],
                            "action_index": index, **{key: action[key] for key in
                            ("selected_action", "reason", "paid_rows", "observation_remaining",
                             "measurement_remaining", "compute_remaining", "options")}})
                    if action["action"] in ("replay_risk_method", "revise_risk_research_method"):
                        target = replays if action["action"] == "replay_risk_method" else rewrites
                        item = {"root_seed": seed, "position": record["task_position"],
                            "action_index": index, **action}
                        if action["action"] == "revise_risk_research_method":
                            later = recorded_actions[index + 1:]
                            stop = next((offset for offset, event in enumerate(later)
                                if event["action"] in ("resume_measured_research", "revise_risk_research_method")), len(later))
                            next_quote = next((event for event in later[:stop]
                                if event["action"] == "choose_risk_research"), None)
                            next_fit = next((event for event in later[:stop]
                                if event["action"] == "propose_risk_research" and event["compute_cost"] > 0), None)
                            item["subsequent_action_quote"] = ({key: next_quote[key] for key in
                                ("selected_action", "reason", "paid_rows", "options") if key in next_quote}
                                if next_quote else None)
                            item["subsequent_paid_fit_attempt"] = next_fit
                            checked = next((event for event in later[later.index(next_fit) + 1:stop]
                                if event["action"] == "risk_research_feedback"), None) if next_fit else None
                            item["subsequent_fresh_feedback"] = ({key: checked[key] for key in
                                ("status", "confirmed", "risk_quantile", "probe_id", "before_loss",
                                 "after_loss", "observation_rows")} if checked else None)
                        target.append(item)
            for event_index, event in enumerate(state["scheduler_events"]):
                if event["action"] == "try_retained_method":
                    quotes.append({"root_seed": seed, "event_index": event_index,
                        **{key: event[key] for key in ("task_position", "source_task_position", "compute_cost",
                            "scheduled", "basis", "reason") if key in event}})
                for field in ("screening", "method_screening"):
                    for index, action in enumerate(event.get(field, [])):
                        if action["action"] == "replay_risk_method":
                            replays.append({"root_seed": seed, "position": event["task_position"],
                                "location": "scheduler", "event_index": event_index,
                                "action_index": index, **action})
            for index, visit in enumerate(state["revisits"]):
                if visit["candidate"].get("kind") != "successor_risk_method":
                    continue
                outcome = visit.get("method_visit_outcome") or {}
                fresh = outcome.get("fresh_comparison")
                draft = drafts.get(visit["candidate"].get("research_draft_id"), {})
                source_seed = draft.get("source_seed")
                source = next((position for position, task in enumerate(state["tasks"])
                    if task["seed"] == source_seed), None) if source_seed is not None else None
                direction = ("unknown_source" if source is None else
                    "predecessor_return" if source > visit["task_position"] else
                    "successor_transfer" if source < visit["task_position"] else "same_task")
                screening = visit.get("screening_compute_cost", 0)
                costs = {key: visit["visit_costs"][key] for key in cost_keys}
                whole = {**costs, "compute_cost": costs["compute_cost"] + screening}
                methods.append({"root_seed": seed, "visit_index": index,
                    "source_seed": source_seed, "source_position": source,
                    "recorded_source_position": visit.get("source_task_position"),
                    "destination_position": visit["task_position"], "direction": direction,
                    "draft_id": visit["candidate"].get("research_draft_id"),
                    "fresh_rows": fresh["observation_rows"] if fresh else 0,
                    "before_loss": fresh["before_loss"] if fresh else None,
                    "after_loss": fresh["after_loss"] if fresh else None,
                    "visit_costs": costs, "screening_compute_cost": screening,
                    "whole_visit_costs": whole, "status": outcome.get("status")})
        tasks = data["tasks"]
        entry["arms"][arm] = {"tasks": len(tasks), "mean_pinball_loss": data["mean_loss"],
            "mean_coverage": sum(item["evaluation"]["final_coverage"] for item in tasks) / len(tasks),
            "mean_action_loss": sum(item["evaluation"]["final_action_loss"] for item in tasks) / len(tasks),
            "statuses": data["statuses"], "learning_cpu_seconds": data["cpu_seconds"],
            "terminal_cpu_seconds": terminal_cpu, "all_cpu_seconds": data["cpu_seconds"] + terminal_cpu,
            "learning_costs": data["learning_costs"], "terminal_costs": data["terminal_costs"],
            "empty_library_first_visit_costs": dict(cold_costs), "empty_library_first_visit_cpu_seconds": cold_cpu,
            "all_costs": {key: data["learning_costs"][key] + data["terminal_costs"][key] for key in cost_keys},
            "actual_learning_action_counts": dict(actions), "final_library_risk_methods": saved_methods,
            "risk_research_choice_reasons": dict(Counter(item["reason"] for item in choices)),
            "risk_research_actual_choices": sum(item["selected_action"] is not None for item in choices),
            "risk_research_choice_records": choices,
            "risk_methods_adopted_on_paid_response": len(adopted_methods),
            "risk_research_proposal_attempts": actions["propose_risk_research"],
            "risk_research_new_paired_feedback": sum(item["before_loss"] is not None
                and item["after_loss"] is not None and item["observation_rows"] > 0 for item in feedback),
            "risk_research_feedback_fresh_improved": sum(item["before_loss"] is not None
                and item["after_loss"] is not None and item["observation_rows"] > 0
                and item["after_loss"] + 1e-7 < item["before_loss"] for item in feedback),
            "risk_research_feedback_records": feedback,
            "risk_method_adoption_records": adopted_methods,
            "risk_method_replays": len(replays), "risk_method_replay_compute": sum(item["compute_cost"] for item in replays),
            "risk_method_rewrites": len(rewrites),
            "risk_method_true_step_rewrites": sum(item["before_steps"] != item["after_steps"] for item in rewrites),
            "risk_method_rewrites_with_subsequent_action_quote": sum(item["subsequent_action_quote"] is not None for item in rewrites),
            "risk_method_rewrites_with_subsequent_paid_fit_attempt": sum(item["subsequent_paid_fit_attempt"] is not None for item in rewrites),
            "risk_method_rewrites_with_subsequent_fit_candidate": sum(item["subsequent_paid_fit_attempt"] is not None
                and item["subsequent_paid_fit_attempt"]["status"] == "requires_fresh_confirmation" for item in rewrites),
            "risk_method_rewrites_with_subsequent_fresh_feedback": sum(item["subsequent_fresh_feedback"] is not None
                and item["subsequent_fresh_feedback"]["observation_rows"] > 0 for item in rewrites),
            "risk_method_rewrites_with_subsequent_fresh_improved": sum(item["subsequent_fresh_feedback"] is not None
                and item["subsequent_fresh_feedback"]["before_loss"] is not None
                and item["subsequent_fresh_feedback"]["after_loss"] is not None
                and item["subsequent_fresh_feedback"]["observation_rows"] > 0
                and item["subsequent_fresh_feedback"]["after_loss"] + 1e-7
                    < item["subsequent_fresh_feedback"]["before_loss"] for item in rewrites),
            "risk_method_rewrite_reasons": dict(Counter(item["reason"] for item in rewrites)),
            "method_visits": len(methods), "method_visits_with_fresh_confirmation": sum(item["fresh_rows"] > 0 for item in methods),
            "method_visits_unscored": sum(item["fresh_rows"] == 0 for item in methods),
            "whole_method_costs": {key: sum(item["whole_visit_costs"][key] for item in methods) for key in cost_keys},
            "method_directions": {direction: {"visits": sum(item["direction"] == direction for item in methods),
                "fresh_improved": sum(item["direction"] == direction and item["fresh_rows"] > 0
                    and item["after_loss"] + 1e-7 < item["before_loss"] for item in methods)}
                for direction in ("predecessor_return", "successor_transfer", "same_task", "unknown_source")},
            "retained_method_quotes": len(quotes), "retained_method_quotes_scheduled": sum(item["scheduled"] for item in quotes),
            "retained_method_quote_compute": sum(item["compute_cost"] for item in quotes),
            "risk_method_replay_records": replays, "risk_method_rewrite_records": rewrites,
            "executed_risk_method_chain": methods, "retained_method_quote_records": quotes}
    for arm, chain_comparison in summary["paired"].items():
        current = summary["arms"]["new_cumulative"]
        reference = summary["arms"][arm]
        old = {(item["seed"], item["position"]): item["evaluation"]["final_loss"] for item in reference["tasks"]}
        paired = [{"seed": item["seed"], "position": item["position"],
            "candidate_loss": item["evaluation"]["final_loss"],
            "reference_loss": old[item["seed"], item["position"]],
            "difference": item["evaluation"]["final_loss"] - old[item["seed"], item["position"]]}
            for item in current["tasks"]]
        entry["comparisons"][arm] = {"chain_pairs": chain_comparison,
            "task_pairs": {"wins": sum(item["difference"] < 0 for item in paired),
                "ties": sum(item["difference"] == 0 for item in paired),
                "losses": sum(item["difference"] > 0 for item in paired)},
            "worst_deteriorations": sorted(paired, key=lambda item: item["difference"], reverse=True)[:5]}
    result["cohorts"][cohort] = entry
(base / "risk_compact_results.json").write_text(json.dumps(result, indent=2) + "\n")
for cohort, entry in result["cohorts"].items():
    print({"cohort": cohort, "arms": {arm: {key: record[key] for key in
        ("mean_pinball_loss", "all_cpu_seconds", "all_costs", "method_visits", "risk_method_rewrites")}
        for arm, record in entry["arms"].items()}})
