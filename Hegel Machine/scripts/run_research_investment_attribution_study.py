#!/usr/bin/env python3
"""Mixed long streams with paid-prefix source deletion and target-withdrawal ablation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import time

from run_knowledge_lineage_study import ROOT, COSTS, read, save, load_runtime
from run_nonlinear_mechanism_revision import laboratory as nonlinear_laboratory
from run_condition_response_action_study import laboratory as conditional_laboratory


def laboratory(task):
    task = dict(task)
    family = task.pop("family")
    task.pop("research_limits", None)
    return (nonlinear_laboratory if family == "noise" else conditional_laboratory)(task)


def directory(output, arm, root):
    return output / "runs" / arm / root


def checkpoint(output, arm, root):
    return directory(output, arm, root) / "checkpoint.json.gz"


def runtime(protocol, arm):
    module = load_runtime(protocol, arm)
    if arm == "no_target_withdrawal":
        from hegel_machine.cognition.memory import KnowledgeLibrary
        class NoTargetWithdrawal(KnowledgeLibrary):
            def record_transfer_counterexample(self, spec, program, source_ids, *, evidence, probe_id):
                return None
        # Only the session's concrete restoration constructor is replaced.
        # Old counterexamples, retrieval, revision, and the predicate remain intact.
        module.KnowledgeLibrary = NoTargetWithdrawal
    return module


def session_for(module, protocol, root, state=None):
    if state is not None:
        return module.UnifiedResearchSession(state=state, lab_factory=laboratory)
    return module.UnifiedResearchSession(module.ResearchConfig(**protocol["config"]),
        tasks=protocol["roots"][root]["tasks"], measurement_budget=0, lab_factory=laboratory)


def persist(output, arm, root, state):
    save(directory(output, arm, root) / f"visit{state['study']['steps']:02d}.json.gz", state)
    save(checkpoint(output, arm, root), state)


def one_step(session, output, arm, root):
    first = len(session.state["results"]) < len(session.state["tasks"])
    cpu, wall = time.process_time(), time.monotonic()
    try:
        state = session.run(max_tasks=1)
    except Exception as error:
        save(directory(output, arm, root) / "failure.json", {"error": repr(error),
            "steps": session.state["study"]["steps"],
            "interrupted_cpu_seconds": time.process_time() - cpu})
        raise
    state["study"]["steps"] += 1
    state["study"]["cpu_seconds"] += time.process_time() - cpu
    state["study"]["elapsed_seconds"] += time.monotonic() - wall
    if first and len(state["results"]) == len(state["tasks"]):
        save(directory(output, arm, root) / "first_visits.json.gz", state)
    state["study"]["learning_complete"] = (len(state["results"]) == len(state["tasks"])
        and not state["pending_revisits"])
    persist(output, arm, root, state)
    print({"arm": arm, "root": root, "steps": state["study"]["steps"],
        "first_visits": len(state["results"]), "revisits": len(state["revisits"]),
        "complete": state["study"]["learning_complete"], "costs": state["costs"]}, flush=True)
    return state


def learn(output, protocol, arm, root):
    if arm == "delete_consumed_sources":
        return ablate(output, protocol, root)
    module = runtime(protocol, arm)
    path = checkpoint(output, arm, root)
    if path.exists():
        state = read(path)
        if state["study"]["learning_complete"]:
            return
        session = session_for(module, protocol, root, state)
    else:
        session = session_for(module, protocol, root)
        session.state["study"] = {"arm": arm, "root": root, "steps": 0,
            "cpu_seconds": 0., "elapsed_seconds": 0., "learning_complete": False}
        persist(output, arm, root, session.to_dict())
    while len(session.state["results"]) < len(session.state["tasks"]) or session.state["pending_revisits"]:
        state = one_step(session, output, arm, root)
        session = session_for(module, protocol, root, state)


def first_consumption(output, root, task_count):
    for position in range(1, task_count):
        before = read(directory(output, "candidate", root) / f"visit{position:02d}.json.gz")
        after = read(directory(output, "candidate", root) / f"visit{position + 1:02d}.json.gz")
        report = after["results"][position]["research"]
        present = {item["concept_id"] for item in before["knowledge"].get("concepts", [])}
        sources = sorted(present & set(report.get("used_sources", []) + report.get("method_source_ids", [])))
        if sources and report["status"] == "completed":
            evidence = {"position": position, "source_ids": sources,
                "basis": "first first-visit confirmed adopted cross-task source set",
                "status": report["status"], "program": report["program"],
                "used_sources": report.get("used_sources", []),
                "method_source_ids": report.get("method_source_ids", []),
                "actions": [{"index": i, **action} for i, action in enumerate(report["actions"])
                    if any(key in action["action"] for key in ("select_", "replay_", "confirm"))]}
            return before, evidence
    return None, {"source_ids": [], "basis": "no confirmed adopted cross-task source in first visits"}


def prune_deleted_offers(state, removed_source_ids, removed_draft_ids):
    """Withdraw unexecuted deleted-source proposals, keeping their paid evidence."""
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import _called_definitions
    removed_source_ids, removed_draft_ids = set(removed_source_ids), set(removed_draft_ids)
    def program_sources(payload):
        return {definition.source_concept_id for definition in
            _called_definitions(Program.from_dict(payload))} if payload is not None else set()
    def question_sources(question):
        sources = set()
        for name in ("parent", "candidate"):
            entry = question[name]
            candidate = entry["candidate"]
            sources.update(entry.get("method_source_ids", []))
            sources.update(candidate.get("parents", []))
            sources.add(candidate.get("memory_id"))
            sources.update(program_sources(candidate.get("program")))
        return sources & removed_source_ids
    def dependencies(offer):
        candidate = offer.get("candidate", {})
        sources = set(candidate.get("source_concept_ids", []))
        sources.update(program_sources(candidate.get("program")))
        if candidate.get("response_question") is not None:
            sources.update(question_sources(candidate["response_question"]))
        drafts = set(candidate.get("method_opportunity", {}).get("draft_ids", []))
        drafts.update(candidate.get("method_experiment", {}).get("source_draft_ids", []))
        return sources & removed_source_ids, drafts & removed_draft_ids
    removed, kept = [], []
    # The session regenerates question offers from these saved, unexecuted
    # hypotheses. Withdraw the proposal itself; retain both programs and rows.
    for result in state.get("results", []):
        for question in result["research"].get("response_questions", []):
            if question["status"] != "pending":
                continue
            sources = question_sources(question)
            if sources:
                removed.append({"queue": "response_questions", "task_position": result["task_position"],
                    "question_id": question["question_id"], "previous_status": question["status"],
                    "source_ids": sorted(sources), "draft_ids": []})
                question["status"] = "withdrawn_source_deletion"
    for offer in state["pending_revisits"]:
        sources, drafts = dependencies(offer)
        if sources or drafts:
            removed.append({"queue": "pending_revisits", "task_position": offer["task_position"],
                "source_ids": sorted(sources), "draft_ids": sorted(drafts)})
        else:
            kept.append(offer)
    state["pending_revisits"] = kept
    for attempt in state.get("method_revisit_attempts", []):
        if "deferred_offer" in attempt:
            sources, drafts = dependencies(attempt["deferred_offer"])
            if sources or drafts:
                removed.append({"queue": "deferred_offer", "task_position": attempt["task_position"],
                    "source_ids": sorted(sources), "draft_ids": sorted(drafts)})
                attempt.pop("deferred_offer")
    return removed


def ablate(output, protocol, root):
    arm = "delete_consumed_sources"
    module = runtime(protocol, arm)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.knowledge_changes import knowledge_snapshot, knowledge_changes
    paired = read(checkpoint(output, "candidate", root))
    if not paired["study"]["learning_complete"]:
        raise ValueError("finish the full candidate learning schedule before matched source deletion")
    path = checkpoint(output, arm, root)
    if path.exists():
        state = read(path)
        if state["study"]["learning_complete"]:
            return
    else:
        state, evidence = first_consumption(output, root, len(paired["tasks"]))
        save(path.parent / "deletion.json", evidence)
        if state is None:
            state = deepcopy(paired)
            state["study"].update(arm=arm, deletion=evidence, no_intervention=True,
                shared_prefix_cpu_seconds=paired["study"]["cpu_seconds"], incremental_cpu_seconds=0.)
            save(path, state)
            return
        library = KnowledgeLibrary.from_dict(state["knowledge"])
        reduced = library.without(evidence["source_ids"])
        evidence["removed_concept_ids"] = sorted({c.concept_id for c in library.concepts} -
            {c.concept_id for c in reduced.concepts})
        evidence["retained_concept_ids"] = [c.concept_id for c in reduced.concepts]
        removed_sources = set(reduced.deleted_ids) - set(library.deleted_ids)
        removed_drafts = {d["draft_id"] for d in library.research_drafts} - {d["draft_id"] for d in reduced.research_drafts}
        evidence["pruned_unexecuted_offers"] = prune_deleted_offers(state, removed_sources, removed_drafts)
        save(path.parent / "deletion.json", evidence)
        state["knowledge"] = reduced.to_dict()
        state["study"].update(arm=arm, deletion=evidence, matched=[], learning_complete=False,
            shared_prefix_cpu_seconds=state["study"]["cpu_seconds"], no_intervention=False)
        persist(output, arm, root, state)
    while len(state["results"]) < len(state["tasks"]):
        session = session_for(module, protocol, root, state)
        state = one_step(session, output, arm, root)
        state["study"]["learning_complete"] = False
        persist(output, arm, root, state)
    for index in range(len(state["study"]["matched"]), len(paired["revisits"])):
        visit = paired["revisits"][index]
        position = visit["task_position"]
        session = session_for(module, protocol, root, state)
        previous = deepcopy(session.state["results"][position]["research"])
        requested = {key: visit["resource_limits"][key] - visit["previous"]["costs"][key] for key in COSTS}
        allocation = {key: min(requested[key], session._remaining()[key]) for key in COSTS}
        receipt = {"matched_index": index, "task_position": position,
            "requested": requested, "allocation": allocation, "executed": bool(allocation["compute_cost"])}
        session._offer_response_questions()
        own_offers = [offer for offer in session.state["pending_revisits"]
            if offer["task_position"] == position and all(
                allocation[key] >= amount for key, amount in offer["minimum_resources"].items())]
        own_offer = max(own_offers, key=lambda item: item["relative_gain"]) if own_offers else None
        if allocation["compute_cost"] and own_offer is not None:
            cpu, wall = time.process_time(), time.monotonic()
            session.state["pending_revisits"].remove(own_offer)
            receipt["executed"] = session._perform_revisit(own_offer, allocation_limits=allocation)
            receipt["own_offer_basis"] = own_offer.get("basis")
            session.state["study"]["cpu_seconds"] += time.process_time() - cpu
            session.state["study"]["elapsed_seconds"] += time.monotonic() - wall
        elif allocation["compute_cost"]:
            limits = {key: previous["costs"][key] + allocation[key] for key in COSTS}
            task = state["tasks"][position]
            available = {concept.concept_id for concept in session.library.concepts}
            models = {model["model_id"] for model in session.library.object_models}
            before = knowledge_snapshot(session.library)
            cpu, wall = time.process_time(), time.monotonic()
            with laboratory(task) as lab:
                report = UnifiedResearchAgent(replace(session.config, seed=task["seed"],
                    observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
                    measurement_budget=limits["measurement_cost"], library=session.library).run(
                        lab, previous=previous, continuation_candidate=None)
            session.state["results"][position]["research"] = report
            session.state["revisits"].append({"task_position": position, "source_task_position": None,
                "candidate": None, "previous": previous, "resource_limits": limits,
                "visit_costs": report["visit_costs"], "status": report["status"], "basis": "matched_own_research"})
            session._offer_new_knowledge(position,
                [c.concept_id for c in session.library.concepts if c.concept_id not in available],
                [m["model_id"] for m in session.library.object_models if m["model_id"] not in models],
                knowledge_changes(before, knowledge_snapshot(session.library)))
            session._offer_response_questions()
            session.state["study"]["cpu_seconds"] += time.process_time() - cpu
            session.state["study"]["elapsed_seconds"] += time.monotonic() - wall
        session.state["study"]["matched"].append(receipt)
        session.state["study"]["steps"] += 1
        state = session.to_dict()
        persist(output, arm, root, state)
    state["study"]["learning_complete"] = True
    state["study"]["unmatched_own_revisits"] = state["pending_revisits"]
    state["pending_revisits"] = []
    state["study"]["incremental_cpu_seconds"] = state["study"]["cpu_seconds"] - state["study"]["shared_prefix_cpu_seconds"]
    state["status"] = "completed"
    save(path, state)
    print({"arm": arm, "root": root, "complete": True, "deletion": state["study"]["deletion"]["source_ids"],
        "costs": state["costs"]}, flush=True)


def confirm_learning_complete(output, protocol):
    outputs = [Path(path).resolve() for path in protocol.get("learning_barrier_outputs", [str(output)])]
    expected = {"arms": list(protocol["arms"]), "roots": list(protocol["roots"]), "complete": True}
    if "learning_barrier_outputs" in protocol:
        expected["learning_barrier_outputs"] = protocol["learning_barrier_outputs"]
    receipt = output / "learning_completion.json"
    if receipt.exists() and read(receipt) == expected:
        return
    cohorts = [(path, read(path / "protocol.json") if path != output.resolve() else protocol)
        for path in outputs]
    for path, cohort in cohorts:
        for name in cohort["arms"]:
            for key in cohort["roots"]:
                if not read(checkpoint(path, name, key))["study"]["learning_complete"]:
                    raise ValueError("all protocol cohorts and arms must finish learning before terminal scoring")
    for path, cohort in cohorts:
        completion = {"arms": list(cohort["arms"]), "roots": list(cohort["roots"]), "complete": True}
        if "learning_barrier_outputs" in cohort:
            completion["learning_barrier_outputs"] = cohort["learning_barrier_outputs"]
        save(path / "learning_completion.json", completion)


def evaluate(output, protocol, arm, root):
    confirm_learning_complete(output, protocol)
    runtime(protocol, arm)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = directory(output, arm, root) / "evaluation.json"
    scores = read(path) if path.exists() else []
    state, settings = read(checkpoint(output, arm, root)), protocol["terminal"]
    for position in range(len(scores), len(state["tasks"])):
        task, report = state["tasks"][position], state["results"][position]["research"]
        program, cpu = Program.from_dict(report["program"]), time.process_time()
        offset = settings["seed_offsets"][task["family"]]
        with laboratory({**task, "noise_seed": task["seed"] + offset}) as lab:
            errors, mean_errors = [], []
            for index in range(settings["trajectories"]):
                probe = lab.random_probe(task["seed"] + offset + 10007 * index,
                    length=settings["length"], name=f"terminal_{index}")
                rows = lab.acquire(probe).observations
                errors.append(loss(program, rows, report["fallback"], .5))
                if task["family"] == "noise":
                    with laboratory({**task, "noise_sigma": 0.}) as oracle:
                        expected = oracle.acquire(probe).observations
                    mean_rows = [replace(row, target=truth.target) for row, truth in zip(rows, expected)]
                    mean_errors.append(loss(program, mean_rows, report["fallback"], .5))
            scores.append({"position": position, "family": task["family"], "stage": protocol["stages"][position],
                "mse": sum(mean_errors or errors) / len(errors), "predictive_mse": sum(errors) / len(errors),
                "trajectories": errors, "mean_response_trajectories": mean_errors, "status": report["status"],
                "costs": {**lab.costs, "compute_cost": settings["length"] * settings["trajectories"] * (2 if mean_errors else 1)},
                "cpu_seconds": time.process_time() - cpu})
        save(path, scores)


def summarize(output, protocol):
    rows, streams, totals = [], [], {}
    for arm in protocol["arms"]:
        for root in protocol["roots"]:
            state = read(checkpoint(output, arm, root))
            scores = read(directory(output, arm, root) / "evaluation.json")
            visits = state["revisits"]
            follow = {key: sum(v["visit_costs"][key] for v in visits) for key in COSTS}
            first = {key: state["costs"][key] - follow[key] -
                (state["screening_compute_cost"] if key == "compute_cost" else 0) for key in COSTS}
            streams.append({"arm": arm, "root": root, "learning_costs": state["costs"],
                "first_visit_costs": first, "revisit_costs": follow, "screening_compute": state["screening_compute_cost"],
                "cpu_seconds": state["study"]["cpu_seconds"], "revisits": len(visits),
                "shared_prefix_cpu_seconds": state["study"].get("shared_prefix_cpu_seconds", 0.),
                "physical_incremental_cpu_seconds": state["study"]["cpu_seconds"] - state["study"].get("shared_prefix_cpu_seconds", 0.),
                "deletion": state["study"].get("deletion"),
                "target_bound_refutation_records": sum(record.get("kind") == "transfer_counterexample"
                    for concept in state["knowledge"].get("concepts", []) for record in concept.get("reuse_history", [])),
                "unmatched_own_offers": len(state["study"].get("unmatched_own_revisits", [])),
                "scheduler_actions": dict(Counter(a["action"] for a in state["scheduler_events"]))})
            for score in scores:
                report = state["results"][score["position"]]["research"]
                rows.append({"arm": arm, "root": root, **score, "learning_costs": report["costs"],
                    "used_sources": report.get("used_sources", []), "method_source_ids": report.get("method_source_ids", []),
                    "confirmation_rows": sum(a["observation_cost"] for a in report["actions"]
                        if a["action"] == "acquire" and a.get("purpose") == "confirm_frozen_program"),
                    "confirmation_visits": sum(a["action"] == "acquire" and a.get("purpose") == "confirm_frozen_program"
                        for a in report["actions"]),
                    "accepted_noisy_refits": sum(a["action"] == "confirm_noisy_response_refit" and a.get("accepted", False)
                        for a in report["actions"]),
                    "research_confirmation_outcomes": dict(Counter(a["status"] for a in report["actions"]
                        if a["action"] == "research_action_confirmation")),
                    "actions": dict(Counter(a["action"] for a in report["actions"]))})
        selected, chains = [r for r in rows if r["arm"] == arm], [s for s in streams if s["arm"] == arm]
        total = {"tasks": len(selected), "completed": sum(r["status"] == "completed" for r in selected),
            "learning_costs": {key: sum(s["learning_costs"][key] for s in chains) for key in COSTS},
            "terminal_costs": {key: sum(r["costs"][key] for r in selected) for key in COSTS},
            "first_visit_costs": {key: sum(s["first_visit_costs"][key] for s in chains) for key in COSTS},
            "revisit_costs": {key: sum(s["revisit_costs"][key] for s in chains) for key in COSTS},
            "screening_compute": sum(s["screening_compute"] for s in chains),
            "revisits": sum(s["revisits"] for s in chains),
            "target_bound_refutation_records": sum(s["target_bound_refutation_records"] for s in chains),
            "actions": dict(sum((Counter(r["actions"]) for r in selected), Counter())),
            "confirmation_rows": sum(r["confirmation_rows"] for r in selected),
            "confirmation_visits": sum(r["confirmation_visits"] for r in selected),
            "accepted_noisy_refits": sum(r["accepted_noisy_refits"] for r in selected),
            "research_confirmation_outcomes": dict(sum((Counter(r["research_confirmation_outcomes"]) for r in selected), Counter())),
            "learning_cpu_seconds": sum(s["cpu_seconds"] for s in chains),
            "terminal_cpu_seconds": sum(r["cpu_seconds"] for r in selected),
            "shared_prefix_cpu_seconds": sum(s["shared_prefix_cpu_seconds"] for s in chains),
            "physical_incremental_cpu_seconds": sum(s["physical_incremental_cpu_seconds"] for s in chains), "families": {}}
        for family in ("noise", "condition"):
            group = [r for r in selected if r["family"] == family]
            total["families"][family] = {"tasks": len(group), "mean_mse": sum(r["mse"] for r in group) / len(group),
                "mean_predictive_mse": sum(r["predictive_mse"] for r in group) / len(group),
                "completed": sum(r["status"] == "completed" for r in group),
                "solved": sum(r["mse"] <= 1e-6 for r in group) if family == "condition" else None,
                "false_completed": sum(r["status"] == "completed" and r["mse"] > 1e-6 for r in group)
                    if family == "condition" else None}
        total["all_costs"] = {key: total["learning_costs"][key] + total["terminal_costs"][key] for key in COSTS}
        total["all_cpu_seconds"] = total["learning_cpu_seconds"] + total["terminal_cpu_seconds"]
        totals[arm] = total
    paired = []
    for control in (arm for arm in protocol["arms"] if arm != "candidate"):
        for family in ("noise", "condition"):
            groups = {arm: {(r["root"], r["position"]): r for r in rows if r["arm"] == arm and r["family"] == family}
                for arm in ("candidate", control)}
            counts = Counter()
            for key, new in groups["candidate"].items():
                old = groups[control][key]
                tie = math.isclose(new["mse"], old["mse"], rel_tol=1e-9, abs_tol=1e-12 if family == "condition" else 0.)
                counts["tie" if tie else "win" if new["mse"] < old["mse"] else "loss"] += 1
            paired.append({"candidate": "candidate", "control": control, "family": family, "roundoff_counts": dict(counts)})
    chain_comparisons = []
    for control in (arm for arm in protocol["arms"] if arm != "candidate"):
        for root in protocol["roots"]:
            costs, means = {}, {}
            for arm in ("candidate", control):
                chain = next(s for s in streams if s["arm"] == arm and s["root"] == root)
                group = [r for r in rows if r["arm"] == arm and r["root"] == root]
                costs[arm] = {key: chain["learning_costs"][key] + sum(r["costs"][key] for r in group) for key in COSTS}
                means[arm] = {family: sum(r["mse"] for r in group if r["family"] == family) /
                    sum(r["family"] == family for r in group) for family in ("noise", "condition")}
            chain_comparisons.append({"control": control, "root": root, "means": means,
                "all_costs": costs, "candidate_minus_control_costs": {
                    key: costs["candidate"][key] - costs[control][key] for key in COSTS}})
    chain_counts = []
    for control in (arm for arm in protocol["arms"] if arm != "candidate"):
        for family in ("noise", "condition"):
            counts = Counter()
            for pair in chain_comparisons:
                if pair["control"] != control:
                    continue
                new, old = pair["means"]["candidate"][family], pair["means"][control][family]
                tie = math.isclose(new, old, rel_tol=1e-9, abs_tol=1e-12 if family == "condition" else 0.)
                counts["tie" if tie else "win" if new < old else "loss"] += 1
            chain_counts.append({"control": control, "family": family, "roundoff_counts": dict(counts)})
    save(output / "summary.json", {"totals": totals, "comparisons": paired,
        "chain_counts": chain_counts, "chain_comparisons": chain_comparisons, "streams": streams, "tasks": rows})
    print({arm: {key: total[key] for key in ("tasks", "families", "all_costs", "all_cpu_seconds")}
        for arm, total in totals.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("learn", "evaluate", "summarize"), required=True)
    parser.add_argument("--arm")
    parser.add_argument("--root")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "summarize":
        summarize(output, protocol)
    elif args.root:
        (learn if args.phase == "learn" else evaluate)(output, protocol, args.arm, args.root)
    else:
        if args.phase == "evaluate":
            confirm_learning_complete(output, protocol)
        def child(root):
            log = directory(output, args.arm, root) / f"{args.phase}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                result = subprocess.run([sys.executable, __file__, "--output", str(output), "--phase", args.phase,
                    "--arm", args.arm, "--root", root], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            print({"phase": args.phase, "arm": args.arm, "root": root, "returncode": result.returncode}, flush=True)
            return result.returncode
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            codes = list(executor.map(child, protocol["roots"]))
        if any(codes):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
