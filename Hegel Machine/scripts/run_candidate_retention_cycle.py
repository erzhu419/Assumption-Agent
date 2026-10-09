#!/usr/bin/env python3
"""Measure saved-candidate reuse across a response change and its return."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


ARMS = ("retained", "latest_only")
PHASES = ("source", "changed_response", "original_response_return")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost",
             "python_execution_seconds", "model_calls", "reported_tokens")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    return {"study": "paid_candidate_retention_response_cycle", "arms": list(ARMS),
        "cases": [{"apparatus_seed": 761006001 + 1009 * index,
                   "changed_response_seed": 861006001 + 1009 * index,
                   "seed": 771006001 + 1009 * index} for index in range(12)],
        "phases": list(PHASES),
        "limits": asdict(InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            confirmation_rows=24, execution_seconds=120., max_model_calls=0, token_budget=0)),
        "prefix": {"trajectories": 3, "length": 24, "channels": []},
        "evaluation": {"length": 128, "seed_offset": 1000003, "mse_threshold": 1e-10},
        "task": "Existing TemporalInquiryLab finite2, task_index=1. The instrument remains fixed; "
            "only response_seed changes, then returns to its original value.",
        "controller": "A fixed offline controller purchases three trajectories, asks existing "
            "research_choice to compare available executable predictors on those paid rows, and "
            "follows its finish recommendation. Without that recommendation it invokes existing "
            "history(base_lags=2) once, compares again and explicitly finishes if recommended. "
            "The source starts without any candidate. No numerical law or private condition label "
            "is supplied to research_choice or history.",
        "intervention": "After one common confirmed source, retain all confirmed predictor files "
            "or carry only the most recently selected confirmed predictor. All candidate paths "
            "are generic and are rescored on the current paid evidence. Failed confirmation does "
            "not publish a new cross-task candidate.",
        "persistence": "Only frozen numerical Python predictors from the history operator transfer. "
            "Preserve each original history export, then set fit=None before "
            "any candidate evaluation or confirmation, retaining its actually fitted coefficients "
            "and prediction code. This prevents silent refitting on later tasks. Copy "
            "the actually selected evaluated_model.source_model code; the task-local _evaluated_model "
            "snapshot and its evidence never become cross-task memory.",
        "accounting": "Charge source preparation once in each arm's total and report it separately; "
            "the physical source is learned once per apparatus. Include paid comparisons, fitting "
            "and independent finish confirmation. Report training wall time, including candidate "
            "snapshot copying, separately from Python execution seconds and host scoring units. "
            "Scoring units are not FLOPs. Terminal scoring is separate.",
        "evaluation_order": "Complete all 12 cases, both arms and both target stages before any "
            "128-row source or target terminal evaluation. Keep failures without replacing cases.",
        "scope": "Controlled offline candidate-retention mechanism experiment. No LLM chooses "
            "these actions; this is not evidence that the live LLM decision gap was repaired. "
            "The cases vary parameters within the existing finite-history family.",
        "external_model_calls": 0}


def add_costs(items):
    return {key: sum(item[key] for item in items) for key in COST_KEYS}


def task_for(case, position):
    return {"task_index": 1, "instrument_mode": "finite2",
        "seed": case["seed"] + 10007 * position,
        "apparatus_seed": case["apparatus_seed"],
        "response_seed": case["changed_response_seed"] if position == 1 else case["apparatus_seed"]}


def train_episode(case, position, directory, pool, plan):
    """Select using public evidence; position only constructs the private task."""
    started = monotonic()
    task = task_for(case, position)
    agent = TemporalOperatorAgent(None, directory, arm="code_react", source_task=False,
        knowledge={"files": deepcopy(pool)}, config=InquiryLLMConfig(**plan["limits"]),
        confirmation_seed=task["seed"] + 7919)
    lab = TemporalInquiryLab(**task)
    initial_paths = list(pool)
    trace = []

    def action(name, arguments):
        feedback = agent._execute(name, arguments, lab)
        trace.append({"action": name, "arguments": deepcopy(arguments), "feedback": deepcopy(feedback)})
        return feedback

    rng = Random(task["seed"])
    prefix = plan["prefix"]
    action("query", {"actions": [{"kind": "trajectory", "channels": list(prefix["channels"]),
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(prefix["length"])]}
        for _ in range(prefix["trajectories"])]})
    paths = list(pool)
    choices, refits = [], []
    if paths:
        choices.append(action("run", {"operator": "research_choice", "models": paths}))
    if not choices or choices[-1].get("next_action", {}).get("action") != "finish":
        path = f"shared/predictor_{position:03d}.py"
        raw_path = f"history_exports/fitted_{position:03d}.py"
        refits.append(action("run", {"operator": "history", "base_lags": 2, "path": raw_path}))
        agent._save(path, (agent.workspace / raw_path).read_text()
            + "\n# Keep the measured coefficients fixed for cross-task prediction.\nfit = None\n")
        refits[-1]["frozen_path"] = path
        # No new evidence was bought: the retained incumbent already has its
        # paid score, so the existing retention path compares only this export.
        choices.append(action("run", {"operator": "research_choice", "models": [path]}))
    choice = choices[-1]
    selected = (choice.get("evaluated_model") or {}).get("source_model")
    confirmation = None
    next_action = choice.get("next_action") or {}
    if next_action.get("action") == "finish":
        confirmation = action("finish", next_action["arguments"])
    passed = bool(confirmation and confirmation.get("fresh_confirmation", {}).get("passed"))
    # The frozen export contains its fitted coefficients, with no active fit
    # callback or files needed from the task-local evaluated-model snapshot.
    archive = ({selected: (agent.workspace / selected).read_text()} if passed and selected else {})
    result = {"phase": PHASES[position], "position": position,
        "status": "completed" if passed else "not_confirmed", "workspace": str(agent.workspace),
        "task": task, "initial_candidate_paths": initial_paths,
        "selected_source_model": selected, "submitted_model": agent.model,
        "selected_original_source": passed and selected == "shared/predictor_000.py",
        "selected_inherited_candidate": passed and selected in initial_paths,
        "refit_count": len(refits), "history_results": refits, "choices": choices,
        "confirmation": confirmation, "fresh_confirmed": passed,
        "training_wall_seconds": monotonic() - started,
        "costs": deepcopy(agent.costs), "controller_trace": trace,
        "paid_observations": deepcopy(agent.observations), "confirmed_model_files": archive}
    write_json(Path(directory) / "training.json", result)
    return {"agent": agent, "laboratory": lab, "result": result}


def train_case(case, directory, plan):
    directory = Path(directory)
    source = train_episode(case, 0, directory / "source", {}, plan)
    pool = source["result"]["confirmed_model_files"]
    arms = {}
    for arm in plan["arms"]:
        current, episodes = deepcopy(pool), []
        for position in (1, 2):
            episode = train_episode(case, position, directory / arm / f"stage_{position:02d}", current, plan)
            episodes.append(episode)
            published = episode["result"]["confirmed_model_files"]
            if published:
                current = {**current, **published} if arm == "retained" else deepcopy(published)
        arms[arm] = episodes
    return {"case": case, "directory": directory, "source": source, "arms": arms}


def score_episode(episode, plan):
    agent, lab, result = episode["agent"], episode["laboratory"], episode["result"]
    rows = lab.evaluation_rows(length=plan["evaluation"]["length"],
        seed=result["task"]["seed"] + plan["evaluation"]["seed_offset"])
    terminal = (evaluate_python_model(agent.workspace, agent.model, rows,
                    timeout_seconds=plan["limits"]["execution_seconds"])
                if agent.model else {"status": "no_submitted_model"})
    terminal.pop("predictions", None)
    terminal["rows"] = len(rows)
    correct = terminal.get("returncode") == 0 and terminal.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"]
    result = {**result, "terminal": terminal, "terminal_correct": correct,
        "confirmed_and_correct": result["fresh_confirmed"] and correct,
        "false_confirmation": result["fresh_confirmed"] and not correct}
    write_json(agent.workspace / "evaluation.json", result)
    return result


def score_case(trained, plan):
    source = score_episode(trained["source"], plan)
    arms = {}
    for arm, episodes in trained["arms"].items():
        stages = [score_episode(episode, plan) for episode in episodes]
        arms[arm] = {"stages": stages, "source_costs": source["costs"],
            "source_training_wall_seconds": source["training_wall_seconds"],
            "target_training_wall_seconds": sum(stage["training_wall_seconds"] for stage in stages),
            "target_costs": add_costs([stage["costs"] for stage in stages]),
            "total_costs": add_costs([source["costs"], *[stage["costs"] for stage in stages]]),
            "refit_count": sum(stage["refit_count"] for stage in stages)}
    result = {"case": trained["case"], "source": source, "arms": arms,
        "paired_retained_minus_latest_total": {key: arms["retained"]["total_costs"][key]
            - arms["latest_only"]["total_costs"][key] for key in COST_KEYS}}
    write_json(trained["directory"] / "results.json", result)
    return result


def summarize(results, failures, plan):
    arms = {}
    for arm in plan["arms"]:
        selected = [result["arms"][arm] for result in results]
        stages = [stage for item in selected for stage in item["stages"]]
        arms[arm] = {"confirmed_and_correct": sum(stage["confirmed_and_correct"] for stage in stages),
            "false_confirmations": sum(stage["false_confirmation"] for stage in stages),
            "changed_confirmed_and_correct": sum(item["stages"][0]["confirmed_and_correct"] for item in selected),
            "return_confirmed_and_correct": sum(item["stages"][1]["confirmed_and_correct"] for item in selected),
            "return_selected_original_source": sum(item["stages"][1]["selected_original_source"] for item in selected),
            "changed_refits": sum(item["stages"][0]["refit_count"] for item in selected),
            "return_refits": sum(item["stages"][1]["refit_count"] for item in selected),
            "source_costs": add_costs([item["source_costs"] for item in selected]),
            "target_costs": add_costs([item["target_costs"] for item in selected]),
            "total_costs": add_costs([item["total_costs"] for item in selected])}
        arms[arm]["source_training_wall_seconds"] = sum(item["source_training_wall_seconds"] for item in selected)
        arms[arm]["target_training_wall_seconds"] = sum(item["target_training_wall_seconds"] for item in selected)
    return {"study": plan["study"], "registered_cases": len(plan["cases"]),
        "completed_cases": len(results), "failures": failures,
        "source_confirmed_and_correct": sum(result["source"]["confirmed_and_correct"] for result in results),
        "arms": arms, "external_model_calls": 0, "scope": plan["scope"]}


def run_study(output, *, study_protocol=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src" / "hegel_machine" / "cognition",
        output / "source_at_run" / "src" / "hegel_machine" / "cognition",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    target = output / "source_at_run" / "scripts" / Path(__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, target)
    trained, failures = [], []
    for case in plan["cases"]:
        path = output / f"case_{case['apparatus_seed']}"
        try:
            trained.append(train_case(case, path, plan))
            print(json.dumps({"phase": "training", "apparatus_seed": case["apparatus_seed"]}), flush=True)
        except Exception:
            failure = {"case": case, "phase": "training", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(path / "failure.json", failure)
    results = []
    for item in trained:
        try:
            results.append(score_case(item, plan))
        except Exception:
            failure = {"case": item["case"], "phase": "evaluation", "error": traceback.format_exc()}
            failures.append(failure)
            write_json(item["directory"] / "evaluation_failure.json", failure)
    summary = summarize(results, failures, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(args.output), indent=2))


if __name__ == "__main__":
    main()
