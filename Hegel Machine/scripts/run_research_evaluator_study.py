"""Prospective small paired study of a revisable research-value evaluator.

Protocol: artifacts/research_evaluator_protocol_20260909.md.
Old complete-session feedback is retrospective training; fresh task outcomes
are acquired only after the two continuation arms have stopped.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import __version__
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.research_opportunities import ResearchOpportunities
from hegel_machine.cognition.session import ResearchSession
from evaluate_cognition_session import evaluate_session

SEEDS = (12391, 15427)
DATE = "20260909"
HISTORY = ROOT / f"artifacts/research_evaluator_retrospective_history_{DATE}.json"
CONTINUATIONS = 6
OBSERVATIONS = 23 * 64
COMPUTATION = 23 * 16000
BOOTSTRAP_COMPUTATION = 24000


@contextmanager
def fixed_estimator(book):
    """Use the old rule while retaining identical real observation updates."""
    estimate, observe = book.estimate, book.observe

    def old_estimate(offer, *, evaluator_budget=None):
        return estimate(offer)

    def old_observe(offer, feedback, *, observation_cost, compute_cost, evaluator_budget=None):
        return observe(offer, feedback, observation_cost=observation_cost, compute_cost=compute_cost)

    book.estimate, book.observe = old_estimate, old_observe
    try:
        yield
    finally:
        book.estimate, book.observe = estimate, observe


def feedback_from_history(trial):
    outcome = trial["outcome"]
    return {"execution_complete": outcome["execution_complete"],
        "reference_rows": outcome["reference_rows"],
        "normalized_gain": outcome["reported_normalized_gain"], "gain": outcome["raw_gain"],
        **{key: outcome[key] for key in ("interval_reduction", "boundary_width_before", "boundary_width_after")
           if key in outcome}}


def historical_book(history, *, budget=None):
    book = ResearchOpportunities()
    for trial in history["trials"]:
        selected = book.estimate(deepcopy(trial["offer"]), evaluator_budget=budget)
        book.observe(selected, feedback_from_history(trial),
            observation_cost=trial["outcome"]["observation_cost"],
            compute_cost=trial["outcome"]["compute_cost"], evaluator_budget=budget)
    return book


def save(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def shadow_selection(session, config, seed):
    """Read the old rule's next executable action on this same paid state."""
    state = deepcopy(session.state)
    state["knowledge"] = session.library.to_dict()
    shadow = ResearchSession(config, make_suite(seed), state=state,
        observation_budget=OBSERVATIONS, compute_budget=COMPUTATION, max_visits=2)
    with fixed_estimator(shadow.library.research_opportunities):
        next_item = next((task for task in shadow._pending() if shadow._visit_config(task) is not None), None)
    if next_item is None:
        return None
    offer = next_item.get("next_research") or {}
    return {"task_index": next_item["task_index"], "task_id": next_item["task_id"],
        "kind": offer.get("kind"), "action": offer.get("action"), "method_id": offer.get("method_id"),
        "expected_gain": offer.get("expected_gain")}


def chosen_action(episode):
    offer = episode["scheduler"].get("research_opportunity") or {}
    return {"task_index": episode["task_index"], "task_id": episode["task_id"],
        "kind": offer.get("kind"), "action": offer.get("action"), "method_id": offer.get("method_id"),
        "expected_gain": offer.get("expected_gain")}


def matching_action(left, right):
    keys = ("task_index", "kind", "action", "method_id")
    return left is not None and right is not None and all(left[key] == right[key] for key in keys)


def prediction_records(episodes):
    records = []
    for episode in episodes:
        offer = episode["scheduler"].get("research_opportunity") or {}
        actual = episode["scheduler"].get("research_return") or {}
        records.append({"task_index": episode["task_index"], "task_id": episode["task_id"],
            "kind": offer.get("kind"), "action": offer.get("action"),
            "forecast": deepcopy(offer.get("evaluator_forecast")),
            "baseline_expected_gain": offer.get("baseline_expected_gain", offer.get("expected_gain")),
            "selected_expected_gain": offer.get("expected_gain"),
            "actual": actual, "observation_cost": episode["observation_cost"],
            "compute_cost": episode["compute_cost"]})
    return records


def forecast_summary(records):
    completed = [row for row in records if row["actual"].get("gain") is not None]
    baseline, learned, paired = [], [], []
    candidates = {}
    for row in completed:
        target = row["actual"]["gain"]
        old = row["baseline_expected_gain"]
        new = (row["forecast"] or {}).get("expected_gain")
        if old is not None:
            baseline.append((max(-1., min(1., old))-target) ** 2)
        if new is not None:
            learned.append((new-target) ** 2)
        if old is not None and new is not None:
            paired.append(((max(-1., min(1., old))-target) ** 2, (new-target) ** 2))
        for prediction in ((row["forecast"] or {}).get("forecast") or {}).get("predictions", []):
            entry = candidates.setdefault(prediction["model_id"], {"errors": [], "common": []})
            error = (prediction["value"]-target) ** 2
            entry["errors"].append(error)
            if old is not None:
                entry["common"].append(((max(-1., min(1., old))-target) ** 2, error))
    return {"selected_actions": len(records), "completed_returns": len(completed),
        "unknown_returns": len(records)-len(completed),
        "old_numeric_predictions": len(baseline), "model_numeric_predictions": len(learned),
        "old_mse_when_numeric": mean(baseline) if baseline else None,
        "model_mse_when_numeric": mean(learned) if learned else None,
        "common_numeric_predictions": len(paired),
        "common_old_mse": mean(item[0] for item in paired) if paired else None,
        "common_model_mse": mean(item[1] for item in paired) if paired else None,
        "candidate_forecasts": [{"model_id": identifier, "predictions": len(item["errors"]),
            "mse": mean(item["errors"]), "common_numeric_predictions": len(item["common"]),
            "common_old_mse": mean(pair[0] for pair in item["common"]) if item["common"] else None,
            "common_model_mse": mean(pair[1] for pair in item["common"]) if item["common"] else None}
            for identifier, item in candidates.items()]}


def evaluation_summary(evaluation):
    risks = [task for task in evaluation["tasks"] if task.get("visits") and "decision" in task["latest"]]
    return {"task_count": evaluation["task_count"], "metrics_by_loss": evaluation["metrics_by_loss"],
        "mean_decision_loss": mean(task["latest"]["decision"]["mean_decision_loss"] for task in risks) if risks else None,
        "decision_tasks": len(risks), "evaluation_rows": evaluation["evaluation_rows"]}


def run_seed(seed, label, common_label=None):
    output = ROOT / f"artifacts/{label}_seed{seed}_{DATE}.json"
    common_path = ROOT / f"artifacts/{common_label or label}_seed{seed}_common_state_{DATE}.json"
    if output.exists() or (common_label is None and common_path.exists()):
        raise FileExistsError("Preserve the fixed experiment; choose a different label for a diagnosed repair")
    history = json.loads(HISTORY.read_text())
    config = ResearchConfig(seed=seed, observation_budget=64, compute_budget=16000, max_rounds=2)
    environments = make_suite(seed)
    if len(environments) != 17:
        raise ValueError("The frozen roster requires every development task")
    if common_label is None:
        library = KnowledgeLibrary()
        library.research_opportunities = historical_book(history)
        common = ResearchSession(config, environments, library=library,
            observation_budget=OBSERVATIONS, compute_budget=COMPUTATION, max_visits=1)
        with fixed_estimator(common.library.research_opportunities):
            common_state = common.run()
        save(common_path, common_state)
    else:
        common_state = json.loads(common_path.read_text())
    if common_state["visited_tasks"] != 17 or common_state["completed_visits"] != 17:
        raise RuntimeError("The common first-visit state did not cover the fixed roster")
    print(json.dumps({"seed": seed, "phase": "common_first_visits_complete",
        "observation_cost": common_state["total_observation_cost"],
        "compute_cost": common_state["total_compute_cost"]}), flush=True)

    sessions, arms = {}, {}
    for name in ("fixed", "learned"):
        session = ResearchSession(config, make_suite(seed), state=common_state,
            observation_budget=OBSERVATIONS, compute_budget=COMPUTATION, max_visits=2)
        bootstrap = SearchBudget(BOOTSTRAP_COMPUTATION)
        if name == "learned":
            learned_book = historical_book(history, budget=bootstrap)
            actual_outcomes = [{key: value for key, value in outcome.items() if key != "evaluator"}
                               for outcome in learned_book.outcomes]
            if actual_outcomes != session.library.research_opportunities.outcomes:
                raise RuntimeError("Evaluator arms must retain exactly the same historical outcomes")
            session.library.research_opportunities = learned_book
            session.state["total_compute_cost"] += bootstrap.spent
        initial_evaluator_events = len(session.library.research_opportunities.to_dict().get("evaluator", {}).get("events", []))
        shadow_choices = []

        def continue_research():
            for step in range(CONTINUATIONS):
                before = len(session.state["episodes"])
                shadow = shadow_selection(session, config, seed) if name == "learned" else None
                session.run(max_steps=1)
                if len(session.state["episodes"]) == before:
                    break
                episode = session.state["episodes"][-1]
                choice = chosen_action(episode)
                if name == "learned":
                    shadow_choices.append({"step": step, "fixed_rule": shadow, "actual": choice,
                        "changed": not matching_action(shadow, choice)})
                print(json.dumps({"seed": seed, "arm": name, "continuation": step + 1,
                    "selected": choice, "total_compute_cost": session.state["total_compute_cost"]}), flush=True)

        if name == "fixed":
            with fixed_estimator(session.library.research_opportunities):
                continue_research()
        else:
            continue_research()
        state = session.to_dict()
        sessions[name] = state
        episodes = state["episodes"][17:]
        records = prediction_records(episodes)
        evaluator_state = session.library.research_opportunities.to_dict()
        arms[name] = {"status": state["status"], "bootstrap_compute_cost": bootstrap.spent,
            "online_evaluator_compute_cost": state.get("evaluator_compute_cost", 0),
            "total_evaluator_compute_cost": bootstrap.spent + state.get("evaluator_compute_cost", 0),
            "total_observation_cost": state["total_observation_cost"], "total_compute_cost": state["total_compute_cost"],
            "new_observation_cost": state["total_observation_cost"] - common_state["total_observation_cost"],
            "new_compute_cost": state["total_compute_cost"] - common_state["total_compute_cost"],
            "continuation_visits": len(episodes), "selection_comparisons": shadow_choices,
            "prediction_records": records, "forecast_summary": forecast_summary(records), "episodes": episodes,
            "evaluator_state": evaluator_state,
            "new_evaluator_events": evaluator_state.get("evaluator", {}).get("events", [])[initial_evaluator_events:],
            "scheduler_events": state["scheduler_events"][len(common_state["scheduler_events"]):]}
    # Neither arm has accessed independent evaluation before this point.
    for name, state in sessions.items():
        evaluation = evaluate_session(state)
        arms[name]["evaluation"] = evaluation
        arms[name]["evaluation_summary"] = evaluation_summary(evaluation)
    report = {"experiment": "paired_research_evaluator_learning", "version": __version__, "seed": seed,
        "config": asdict(config), "limits": {"observation_budget": OBSERVATIONS, "compute_budget": COMPUTATION,
            "max_continuations": CONTINUATIONS, "max_visits_per_task": 2, "bootstrap_compute_budget": BOOTSTRAP_COMPUTATION},
        "history": {"source": str(HISTORY), "trials": len(history["trials"]),
            "completed_numeric": sum(trial["outcome"]["gain"] is not None for trial in history["trials"]),
            "prior_full_session_cost": history["prior_full_session_cost"]},
        "common_state": str(common_path), "common_state_reused": common_label is not None,
        "common_cost": {
            "observation_cost": common_state["total_observation_cost"], "compute_cost": common_state["total_compute_cost"]},
        "arms": arms,
        "interpretation": "Retrospective research history precedes fresh paid task feedback. Both arms share their whole first-visit state; unknown gains remain unknown. Every model-fit and forecast charge is included in the learned arm's finite compute budget. Final task outcomes are accessed only after both arms stop."}
    save(output, report)
    print(json.dumps({"seed": seed, "completed": str(output), "arms": {name: {
        key: arm[key] for key in ("bootstrap_compute_cost", "total_observation_cost", "total_compute_cost",
                                  "continuation_visits", "evaluation_summary")} for name, arm in arms.items()}}), flush=True)
    return report


def summarize(seeds, label):
    reports = [json.loads((ROOT / f"artifacts/{label}_seed{seed}_{DATE}.json").read_text()) for seed in seeds]
    result = {"seeds": seeds, "label": label, "per_seed": [], "aggregate": {}}
    for report in reports:
        row = {"seed": report["seed"], "arms": {}}
        for name, arm in report["arms"].items():
            row["arms"][name] = {key: arm[key] for key in ("total_observation_cost", "total_compute_cost",
                "bootstrap_compute_cost", "online_evaluator_compute_cost", "continuation_visits", "evaluation_summary")}
            row["arms"][name].update(forecast_summary=forecast_summary(arm["prediction_records"]),
                changed_selections=sum(item["changed"] for item in arm["selection_comparisons"]),
                new_evaluator_events=arm["new_evaluator_events"],
                current_evaluator=arm["evaluator_state"].get("evaluator", {}).get("current"))
        result["per_seed"].append(row)
    for name in ("fixed", "learned"):
        arms = [report["arms"][name] for report in reports]
        result["aggregate"][name] = {key: sum(arm[key] for arm in arms) for key in (
            "total_observation_cost", "total_compute_cost", "bootstrap_compute_cost",
            "online_evaluator_compute_cost", "continuation_visits")}
        result["aggregate"][name].update(
            mean_squared_loss=mean(arm["evaluation_summary"]["metrics_by_loss"]["squared_error"]["latest_loss"] for arm in arms),
            mean_pinball_loss=mean(arm["evaluation_summary"]["metrics_by_loss"]["pinball"]["latest_loss"] for arm in arms),
            mean_decision_loss=mean(arm["evaluation_summary"]["mean_decision_loss"] for arm in arms),
            changed_selections=sum(item["changed"] for arm in arms for item in arm["selection_comparisons"]),
            forecast_summary=forecast_summary([record for arm in arms for record in arm["prediction_records"]]))
    save(ROOT / f"artifacts/{label}_summary_{DATE}.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    parser.add_argument("--label", default="research_evaluator_v024")
    parser.add_argument("--common-label", help="Reuse a preserved common first-visit checkpoint for a wiring repair")
    parser.add_argument("--summarize", action="store_true", help="Summarize saved runs without performing research")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.seed, args.label)
        return
    for seed in args.seed:
        run_seed(seed, args.label, args.common_label)


if __name__ == "__main__":
    main()
