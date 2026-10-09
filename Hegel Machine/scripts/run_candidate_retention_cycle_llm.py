#!/usr/bin/env python3
"""Run actual LLM research through a change and return with branch-local memory."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import (
    CodexCodeProvider, archive_confirmed_python_model, evaluate_python_model,
    python_model_context_matches,
)
from hegel_machine.cognition.online_inquiry_llm import COMMON_INTERFACE, InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import OPERATOR_INTERFACE, TemporalOperatorAgent
from scripts.run_candidate_retention_cycle import protocol as offline_protocol, task_for
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription


CONDITIONS = ("retained", "latest_only")
BANK_PATH = "shared/confirmed_models"
CATALOG_PATH = "shared/candidate_catalog.json"
TASK_INTERFACE = """
Three paid control trajectories are already available in your ordinary evidence
files. Infer a controls-only predictor for this apparatus and submit it with finish.
Earlier independently confirmed predictors, when available, are listed in
shared/candidate_catalog.json and session_knowledge.candidates. Their archived
files preserve their original training view: evaluating an archived model scores
that learned predictor on current observations without fitting it to current labels.
Earlier confirmation describes earlier evidence, not current applicability. You
may assess, reuse, revise or work without these candidates using ordinary Python
and the same shared tools. All additional observations and execution are charged.
Choose your research actions yourself. No knowledge manifest is required here.
If a new predictor depends on an archived candidate, name that candidate's catalog
model path in finish.reused_files. Only declared archive dependencies accompany
the new predictor into later tasks. Submitting an archived model itself preserves
that model and its existing dependencies without an extra declaration.
"""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    previous = offline_protocol()
    return {"study": "same_llm_confirmed_candidate_retention_cycle", "model": "gpt-6-luna",
        "conditions": list(CONDITIONS),
        "cases": [{"apparatus_seed": 791006001 + 1009 * index,
                   "changed_response_seed": 891006001 + 1009 * index,
                   "seed": 801006001 + 1009 * index} for index in range(2)],
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 180},
        "limits": asdict(InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            confirmation_rows=24, execution_seconds=120., max_model_calls=12, token_budget=400000)),
        "prefix": previous["prefix"], "evaluation": previous["evaluation"],
        "task": previous["task"],
        "source": "One live LLM source per apparatus, learned once from three paid control "
            "trajectories. Fork only its actual independently confirmed predictor and required files. "
            "No supplied law, offline source fit or researcher-selected source predictor.",
        "intervention": "Each branch continues through a response change and then its return. "
            "retained keeps every confirmed candidate; latest_only keeps only its most recently "
            "confirmed candidate. A target inherits its own branch's preceding confirmed state, "
            "rather than resetting to the original source. Equal model, tools, limits, evidence "
            "prefix and confirmation seeds. The chooser receives generic candidate IDs, never "
            "private phase, response seed or mechanism-change labels.",
        "controller": "The host only purchases the common three-trajectory prefix and carries "
            "confirmed archives between episodes. Every research, fitting, comparison and finish "
            "action is selected by agent.run. Ordinary fit callbacks retain their actual original "
            "training files when archived; no host coefficient edit or fit=None substitution.",
        "archive_publication": "New predictors retain ordinary code, data and binary dependencies. "
            "Imported confirmed-model archives accompany them only when declared in the successful "
            "finish.reused_files. Runner logs and the undeclared candidate catalog are excluded. "
            "An unchanged directly submitted archive keeps its existing dependencies. Revisions to "
            "an inherited model publish a new candidate. A latest predictor explicitly depending "
            "on an older candidate retains that dependency, which is reported.",
        "failure_policy": "No replacement or automatic retry. A source without confirmed submission "
            "skips all four dependent target episodes. A failed target publishes no new candidate; "
            "the next target keeps that branch's last confirmed state. Keep service failures separate "
            "from research failure and compare complete service-available pairs separately.",
        "accounting": "Report source costs separately and charge the common source once in each "
            "counterfactual condition. Distinguish actual source-plus-target model calls from these "
            "counterfactual totals. Host scoring units are not FLOPs; preserve Python execution, "
            "archive import/export time, and total episode wall time separately.",
        "terminal_order": "Every source and target stops learning before any 128-row terminal "
            "evaluation. Source admission uses fresh confirmation only, never terminal outcomes.",
        "scope": "Two new parameter realizations within the existing finite-history family. "
            "This is a small live decision experiment, not an independent population estimate. "
            "Both conditions retain the same powerful shared tools. Explicit candidate execution "
            "is observable; an untracked conceptual influence is not inferred from file availability.",
        "model_identity": "gpt-6-luna is requested through the existing subscription provider. "
            "Usage metadata repeating that name does not independently establish server routing."}


def public_candidate(candidate):
    return {"candidate_id": candidate["candidate_id"], "model": candidate["model"],
            "training_rows": candidate["training_rows"],
            "declared_dependencies": candidate.get("declared_dependencies", [])}


def prepare_episode(case, position, workspace, knowledge, plan, provider=None, *, laboratory=None,
                    agent_class=TemporalOperatorAgent):
    started = monotonic()
    workspace = Path(workspace).resolve()
    candidates = [public_candidate(item) for item in knowledge]
    agent = agent_class(provider, workspace, arm="code_react", source_task=False,
        config=InquiryLLMConfig(**plan["limits"]),
        knowledge={"candidates": candidates, "files": {CATALOG_PATH: json.dumps(candidates, indent=2)}},
        confirmation_seed=task_for(case, position)["seed"] + 7919)
    agent.common_interface = (COMMON_INTERFACE.split("For the hegel condition,")[0]
                              + OPERATOR_INTERFACE + TASK_INTERFACE)
    imported = monotonic()
    for candidate in knowledge:
        destination = workspace / BANK_PATH / candidate["candidate_id"] / "_evaluated_model"
        shutil.copytree(candidate["snapshot_workspace"], destination)
        for path in destination.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
                agent.files[str(path.relative_to(workspace))] = path.read_text(encoding="utf-8")
    import_seconds = monotonic() - imported
    agent.knowledge["files"] = deepcopy(agent.files)
    agent.inherited_files = set(agent.files)
    agent.candidate_paths = [candidate["model"] for candidate in candidates]
    lab = laboratory if laboratory is not None else TemporalInquiryLab(**task_for(case, position))
    rng, prefix = Random(task_for(case, position)["seed"]), plan["prefix"]
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": list(prefix["channels"]),
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(prefix["length"])]}
        for _ in range(prefix["trajectories"])]}, lab)
    payload = {"case": deepcopy(case), "position": position, "path": workspace,
        "agent": agent, "laboratory": lab, "inherited_candidates": deepcopy(knowledge),
        "prefix_costs": deepcopy(agent.costs), "archive_import_seconds": import_seconds,
        "preparation_wall_seconds": monotonic() - started}
    write_json(workspace / "preparation.json", {"position": position,
        "candidate_catalog": candidates, "archive_provenance": knowledge,
        "prefix_costs": payload["prefix_costs"], "archive_import_seconds": import_seconds,
        "agent": agent.result("prepared")})
    return payload


def execute_episode(payload):
    agent = payload["agent"]

    def checkpoint(result):
        write_json(payload["path"] / "active.json", result)
        event = result["events"][-1] if result["events"] else {}
        print(json.dumps({"workspace": str(payload["path"]), "calls": result["costs"]["model_calls"],
            "action": event.get("action"), "operator": event.get("arguments", {}).get("operator")}), flush=True)

    started = monotonic()
    try:
        result = agent.run(payload["laboratory"], on_checkpoint=checkpoint)
    except Exception:
        result = agent.result("execution_failed")
        result["execution_error"] = traceback.format_exc()
    payload["research_wall_seconds"] = monotonic() - started
    payload["training"] = result
    write_json(payload["path"] / "training.json", result)
    return payload


def selected_candidate(payload):
    if "selected_inherited_candidate" in payload:
        return payload["selected_inherited_candidate"]
    result = payload["training"]
    submitted = result.get("model")
    source = (result.get("evaluated_model") or {}).get("source_model")
    paths = [submitted, source] if submitted and submitted.startswith("_evaluated_model/") else [submitted]
    located = next((candidate for candidate in payload["inherited_candidates"]
        if any(path and (path == candidate["model"] or path.endswith("/" + candidate["model"]))
               for path in paths)), None)
    unchanged = bool(located and python_model_context_matches(payload["agent"].workspace,
        submitted, located["snapshot_workspace"], located["inner_model"]))
    payload["inherited_candidate_comparison"] = {"path_candidate_id": located["candidate_id"] if located else None,
        "unchanged": unchanged if located else None}
    payload["selected_inherited_candidate"] = located if unchanged else None
    return payload["selected_inherited_candidate"]


def clean_published_archive(payload, archive):
    snapshot = Path(archive["snapshot_workspace"])
    finish = next((event for event in reversed(payload["training"]["events"])
        if event.get("action") == "finish" and event.get("feedback", {}).get("fresh_confirmation", {}).get("passed")), {})
    declared = finish.get("arguments", {}).get("reused_files", [])
    dependencies = [item for item in payload["inherited_candidates"]
        if any(str(path) == item["model"] or str(path).startswith(str(Path(BANK_PATH) / item["candidate_id"]) + "/")
               for path in declared)]
    located_id = payload.get("inherited_candidate_comparison", {}).get("path_candidate_id")
    located = next((item for item in payload["inherited_candidates"]
                    if item["candidate_id"] == located_id), None)
    if located is not None:
        dependencies += [item for item in located.get("declared_dependencies", [])
                         if item["candidate_id"] not in {entry["candidate_id"] for entry in dependencies}]
    keep = {item["candidate_id"] for item in dependencies}
    bank = snapshot / BANK_PATH
    if bank.exists():
        for directory in bank.iterdir():
            if directory.name not in keep:
                if directory.is_dir():
                    shutil.rmtree(directory)
                else:
                    directory.unlink()
        if not any(bank.iterdir()):
            bank.rmdir()
    catalog = snapshot / CATALOG_PATH
    if dependencies:
        write_json(catalog, [public_candidate(item) for item in dependencies])
    else:
        catalog.unlink(missing_ok=True)
    for name in ("preparation.json", "active.json", "training.json", "published_candidate.json",
                 "evaluation.json", "research_context.json", "research_state.json"):
        (snapshot / name).unlink(missing_ok=True)
    return [public_candidate(item) for item in dependencies]


def publish_candidate(payload, bank):
    result = payload["training"]
    if not result.get("model") or not (result.get("fresh_confirmation") or {}).get("passed"):
        payload["archive_export_seconds"] = 0.
        payload["published_candidate"] = None
        return None
    inherited = selected_candidate(payload)
    if inherited is not None:
        candidate, elapsed = deepcopy(inherited), 0.
    else:
        identifier = f"candidate_{payload['position']:03d}"
        started = monotonic()
        archive = archive_confirmed_python_model(payload["agent"], Path(bank) / identifier / "_evaluated_model")
        dependencies = clean_published_archive(payload, archive)
        elapsed = monotonic() - started
        candidate = {**archive, "candidate_id": identifier,
            "model": str(Path(BANK_PATH) / identifier / archive["model"]),
            "declared_dependencies": dependencies,
            "created_position": payload["position"], "confirmed_at": str(payload["path"] / "training.json")}
    payload["archive_export_seconds"] = elapsed
    payload["published_candidate"] = candidate
    write_json(payload["path"] / "published_candidate.json", {"candidate": candidate,
        "reconfirmed_inherited_candidate": inherited is not None,
        "inherited_candidate_comparison": payload.get("inherited_candidate_comparison"),
        "archive_export_seconds": elapsed})
    return candidate


def train_source(case, path, plan, provider_factory):
    path = Path(path)
    episode = prepare_episode(case, 0, path / "episode_000", [], plan,
                              provider=provider_factory(path / "episode_000"))
    execute_episode(episode)
    candidate = publish_candidate(episode, path / "bank")
    return {"case": deepcopy(case), "episode": episode, "candidate": candidate}


def train_branch(source, condition, path, plan, provider_factory):
    path = Path(path)
    pool, episodes = [deepcopy(source["candidate"])], []
    for position in (1, 2):
        workspace = path / f"episode_{position:03d}"
        payload = prepare_episode(source["case"], position, workspace, pool, plan,
                                  provider=provider_factory(workspace))
        execute_episode(payload)
        candidate = publish_candidate(payload, path / "bank")
        episodes.append(payload)
        if candidate is not None:
            if condition == "latest_only":
                pool = [candidate]
            else:
                pool = [item for item in pool if item["candidate_id"] != candidate["candidate_id"]] + [candidate]
    return {"case": deepcopy(source["case"]), "condition": condition,
            "source": source, "episodes": episodes, "final_candidates": pool}


def action_matches(event, proposed):
    return (event.get("action") == proposed.get("action") and all(
        event.get("arguments", {}).get(key) == value for key, value in proposed.get("arguments", {}).items()))


def decision_evidence(payload):
    result, inherited = payload["training"], payload["inherited_candidates"]
    events, recommendations, candidate_scores = result["events"], [], []
    for index, event in enumerate(events):
        arguments, feedback = event.get("arguments", {}), event.get("feedback", {})
        if event.get("action") == "run" and arguments.get("operator") == "research_choice":
            proposed = feedback.get("next_action")
            if proposed:
                followed = [next_index for next_index in range(index + 1, len(events))
                            if action_matches(events[next_index], proposed)]
                recommendations.append({"step": index, "next_action": proposed,
                    "followed_steps": followed, "followed_immediately": index + 1 in followed})
            evaluations = feedback.get("evaluations", [])
        elif event.get("action") in {"evaluate", "finish"}:
            evaluations = [{"path": arguments.get("model"), **feedback}]
        else:
            evaluations = []
        for evaluation in evaluations:
            path = evaluation.get("path")
            candidate = next((item for item in inherited if path and
                (path == item["model"] or path.endswith("/" + item["model"]))), None)
            if candidate:
                candidate_scores.append({"step": index, "candidate_id": candidate["candidate_id"],
                    "path": path, "mse": evaluation.get("mse"), "returncode": evaluation.get("returncode"),
                    "rejected_on_paid_rows": evaluation.get("returncode") == 0
                        and evaluation.get("mse") is not None and evaluation["mse"] > 1e-10,
                    "fresh_confirmation": {key: value for key, value in evaluation.get("fresh_confirmation", {}).items()
                        if key in {"passed", "mse", "rows"}}})
    chosen = selected_candidate(payload)
    return {"research_choice_recommendations": recommendations,
        "inherited_candidate_scores": candidate_scores,
        "selected_inherited_candidate_id": chosen["candidate_id"] if chosen else None,
        "inherited_candidate_comparison": payload.get("inherited_candidate_comparison"),
        "selected_original_candidate": bool(chosen and chosen["candidate_id"] == "candidate_000"),
        "native_fit_operator_calls": dict(Counter(event.get("arguments", {}).get("operator") for event in events
            if event.get("action") == "run" and event.get("arguments", {}).get("operator") in {"history", "instrument", "response"})),
        "ordinary_python_runs": sum(event.get("action") == "run" and not event.get("arguments", {}).get("operator")
                                    for event in events)}


def score_episode(payload, plan):
    result, agent = payload["training"], payload["agent"]
    rows = payload["laboratory"].evaluation_rows(length=plan["evaluation"]["length"],
        seed=task_for(payload["case"], payload["position"])["seed"] + plan["evaluation"]["seed_offset"])
    terminal = (evaluate_python_model(agent.workspace, result["model"], rows,
                    timeout_seconds=plan["limits"]["execution_seconds"]) if result.get("model")
                else {"status": "no_submitted_model"})
    terminal.pop("predictions", None)
    terminal["rows"] = len(rows)
    confirmed = bool((result.get("fresh_confirmation") or {}).get("passed"))
    correct = terminal.get("returncode") == 0 and terminal.get("mse", float("inf")) <= plan["evaluation"]["mse_threshold"]
    usage = [event.get("usage", {}) for event in result["events"]]
    scored = {"position": payload["position"], "status": result["status"], "model": result.get("model"),
        "confirmed_and_correct": confirmed and correct, "fresh_confirmed": confirmed,
        "false_confirmation": confirmed and not correct, "terminal": terminal,
        "costs": deepcopy(result["costs"]), "prefix_costs": payload["prefix_costs"],
        "research_wall_seconds": payload["research_wall_seconds"],
        "preparation_wall_seconds": payload["preparation_wall_seconds"],
        "archive_import_seconds": payload["archive_import_seconds"],
        "archive_export_seconds": payload.get("archive_export_seconds", 0.),
        "candidate_catalog": [public_candidate(item) for item in payload["inherited_candidates"]],
        "published_candidate": payload.get("published_candidate"),
        "known_tokens": sum(item["total_tokens"] for item in usage if isinstance(item.get("total_tokens"), int)),
        "missing_usage_calls": sum(not isinstance(item.get("total_tokens"), int) for item in usage),
        "input_tokens": sum(item.get("input_tokens") or 0 for item in usage),
        "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage),
        **decision_evidence(payload)}
    write_json(payload["path"] / "evaluation.json", scored)
    return scored


def summarize(sources, branches, skipped, plan):
    by_seed = {item["case"]["apparatus_seed"]: item for item in sources}
    conditions = {}
    for condition in plan["conditions"]:
        selected = [branch for branch in branches if branch["condition"] == condition]
        stages = [stage for branch in selected for stage in branch["stages"]]
        conditions[condition] = {"attempted_sequences": len(selected),
            "registered_target_episodes": len(plan["cases"]) * 2,
            "attempted_target_episodes": len(stages),
            "target_statuses": dict(Counter(stage["status"] for stage in stages)),
            "targets_confirmed_and_correct": sum(stage["confirmed_and_correct"] for stage in stages),
            "complete_sequences_correct": sum(all(stage["confirmed_and_correct"] for stage in branch["stages"])
                                               for branch in selected),
            "return_selected_original": sum(branch["stages"][1]["selected_original_candidate"] for branch in selected),
            "false_confirmations": sum(stage["false_confirmation"] for stage in stages),
            "source_costs": total_costs([source["episode"]["costs"] for source in sources]),
            "target_costs": total_costs([stage["costs"] for stage in stages]),
            "source_plus_target_costs": total_costs([source["episode"]["costs"] for source in sources]
                + [stage["costs"] for stage in stages]),
            "recommendations": sum(len(stage["research_choice_recommendations"]) for stage in stages),
            "recommendations_followed": sum(bool(item["followed_steps"]) for stage in stages
                for item in stage["research_choice_recommendations"])}
    complete_pairs = []
    for case in plan["cases"]:
        pair = {branch["condition"]: branch for branch in branches if branch["case"] == case}
        if len(pair) != len(plan["conditions"]):
            continue
        all_stages = [by_seed[case["apparatus_seed"]]["episode"]] + [stage for branch in pair.values() for stage in branch["stages"]]
        if all(stage["status"] not in {"provider_failed", "token_usage_unavailable", "execution_failed"} for stage in all_stages):
            complete_pairs.append(case["apparatus_seed"])
    actual = [source["episode"] for source in sources] + [stage for branch in branches for stage in branch["stages"]]
    return {"study": plan["study"], "registered_source_episodes": len(plan["cases"]),
        "registered_target_episodes": len(plan["cases"]) * 2 * len(plan["conditions"]),
        "attempted_episodes": len(actual), "sources": sources, "branches": branches,
        "skipped": skipped, "conditions": conditions, "service_available_matched_cases": complete_pairs,
        "actual_costs": total_costs([item["costs"] for item in actual]),
        "known_tokens": sum(item["known_tokens"] for item in actual),
        "missing_usage_calls": sum(item["missing_usage_calls"] for item in actual),
        "input_tokens": sum(item["input_tokens"] for item in actual),
        "cached_input_tokens": sum(item["cached_input_tokens"] for item in actual),
        "scope": plan["scope"], "model_identity": plan["model_identity"]}


def run_study(output, *, provider_factory, auth_mode=None, max_workers=2, study_protocol=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    shutil.copytree(ROOT / "src" / "hegel_machine" / "cognition",
        output / "source_at_run" / "src" / "hegel_machine" / "cognition",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_candidate_retention_cycle.py", "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)

    def source_job(case):
        return train_source(case, output / f"case_{case['apparatus_seed']}" / "source", plan, provider_factory)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        sources = list(pool.map(source_job, plan["cases"]))
        jobs, skipped = [], []
        for index, source in enumerate(sources):
            if source["candidate"] is None:
                skipped.extend({"case": source["case"], "condition": condition, "position": position,
                    "reason": "source_not_confirmed", "source_status": source["episode"]["training"]["status"]}
                    for condition in plan["conditions"] for position in (1, 2))
                continue
            order = plan["conditions"] if index % 2 == 0 else plan["conditions"][::-1]
            jobs.extend((source, condition) for condition in order)

        def branch_job(job):
            source, condition = job
            return train_branch(source, condition,
                output / f"case_{source['case']['apparatus_seed']}" / condition, plan, provider_factory)

        branches = list(pool.map(branch_job, jobs))
    scored_sources = [{"case": source["case"], "episode": score_episode(source["episode"], plan)} for source in sources]
    scored_branches = [{"case": branch["case"], "condition": branch["condition"],
        "stages": [score_episode(stage, plan) for stage in branch["episodes"]],
        "final_candidates": branch["final_candidates"]} for branch in branches]
    summary = summarize(scored_sources, scored_branches, skipped, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    result = run_study(args.output, auth_mode=auth, provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps({key: value for key, value in result.items() if key not in {"sources", "branches"}}, indent=2))


if __name__ == "__main__":
    main()
