#!/usr/bin/env python3
"""Compare retained versus removed learned workflows in the legacy researcher."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_current_offline_ablation import _compact
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for

COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("retained_workflows", "removed_workflows")


def protocol(seed=61007001, apparatus_seed=None, instrument_mode="recurrent"):
    apparatus_seed = seed if apparatus_seed is None else apparatus_seed
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=5, max_history=4, active_probes=True, seed=seed)
    arms = list(ARMS)
    Random(seed+17011).shuffle(arms)
    return {"experiment": "learned_workflow_legacy_session_bridge", "seed": seed,
        "instrument_mode": instrument_mode, "prefix_tasks": 2,
        "tasks": [{"environment": "instrument_methods", "task_index": i,
            "seed": seed + 10007*i, "apparatus_seed": apparatus_seed,
            "instrument_mode": instrument_mode} for i in range(6)],
        "research_config": asdict(config), "measurement_budget": 1536,
        "evaluation": {"length": 24, "trajectories": 4, "seed_offset": 1000003},
        "impulse_amplitudes": [-.85, .85], "solved_threshold": 1e-6,
        "arms": arms, "arm_order_seed": seed+17011,
        "information_contract": "Learn stages 0 and 1 once from empty knowledge. Clone the complete "
            "session and clear only research_workflows in one arm. Preserve predictive knowledge, "
            "legacy methods and experience. Both arms may learn new workflows and revisit tasks. "
            "Both complete learning before any terminal scores in that paired instance.",
        "accounting": "Identical initial caps and summed sequence ceilings; both may pool unused "
            "resources. Full costs include the common prefix once per counterfactual arm, screening, "
            "failed attempts and confirmation. Physical prefix learning happens once. Terminal "
            "random and impulse predictions are charged separately.",
        "scope": "Existing finite/recurrent instrument tasks, not new physical families. A replay "
            "must actually execute to count. Resource allocation and feedback are reported only "
            "when their recorded actions occur; no trigger or no benefit remains an outcome."}


def cohort_protocol():
    return {"experiment": "workflow_bridge_12_new_apparatuses", "bootstrap_samples": 10000,
        "bootstrap_seed": 61007019,
        "cases": [protocol(61107001 + 100003*i, instrument_mode="finite" if i < 6 else "recurrent")
            for i in range(12)],
        "case_policy": "Retain every registered case, including execution failures and inactive interventions."}


def _fork_state(prefix, arm):
    state = deepcopy(prefix)
    if arm == "removed_workflows":
        state["knowledge"]["research_workflows"]["methods"] = []
    return state


def _impulse_score(factory, task, research, plan):
    with factory(task) as laboratory:
        rows = []
        for index, amplitude in enumerate(plan["impulse_amplitudes"]):
            bounds = laboratory.spec.feature_bounds
            baseline = {name: (low+high)/2 for name, (low,high) in bounds.items()}
            point = {name: low+(amplitude+1)*(high-low)/2 for name,(low,high) in bounds.items()}
            sequence = (point, *(dict(baseline) for _ in range(
                min(plan["evaluation"]["length"], laboratory.spec.max_probe_length)-1)))
            rows.extend(laboratory.acquire(Probe(f"terminal_impulse_{index}",sequence),
                research["selected_channels"]).observations)
        rate = sum(channel.cost_per_row for channel in laboratory.channels
            if channel.name in research["selected_channels"])
        measured = loss(Program.from_dict(research["program"]),rows,research["fallback"],
            laboratory.spec.risk_quantile) if rows else None
    return {"mse": measured, "costs": {"observation_cost":len(rows),
        "measurement_cost":len(rows)*rate,"compute_cost":len(rows)}}


def run_study(plan, lab_factory=None):
    factory = lab_factory or laboratory_for
    config = ResearchConfig(**plan["research_config"])
    start = process_time()
    initial = UnifiedResearchSession(config,tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"],library=KnowledgeLibrary(),lab_factory=factory)
    prefix = initial.run(max_tasks=plan["prefix_tasks"])
    prefix_cpu = process_time()-start
    methods = prefix["knowledge"]["research_workflows"]["methods"]
    sessions, training_cpu, initial_books = {}, {}, {}
    for arm in plan["arms"]:
        state = _fork_state(prefix,arm)
        initial_books[arm] = deepcopy(state["knowledge"]["research_workflows"])
        session = UnifiedResearchSession(state=state,lab_factory=factory)
        start = process_time()
        session.run()
        sessions[arm],training_cpu[arm] = session,process_time()-start
    results = {}
    for arm,session in sessions.items():
        start = process_time()
        report = session.evaluate(**plan["evaluation"])
        result = _compact(report,plan)
        counts = Counter()
        for row,entry,score in zip(result["tasks"],report["results"],report["evaluation"]):
            research,position = entry["research"],entry["task_position"]
            prior_actions = (len(prefix["results"][position]["research"]["actions"])
                if position < plan["prefix_tasks"] else 0)
            actions = deepcopy(research["actions"][prior_actions:])
            counts.update(action["action"] for action in actions)
            impulse = _impulse_score(factory,plan["tasks"][position],research,plan)
            row.update(random_mse=row["mse"],impulse_mse=impulse["mse"],actions=actions,
                research_workflow_id=research.get("research_workflow_id"),
                concepts_added=research.get("concepts_added",[]),program=research["program"],
                nested_call_sources=sorted({d.source_concept_id for d in
                    _called_definitions(Program.from_dict(research["program"])) if d.source_concept_id}),
                transfer=deepcopy(research.get("transfer",[])),
                knowledge_return=deepcopy(research.get("knowledge_return")),
                fresh_confirmation_passed=research["status"] == "completed"
                    and research.get("confirmation_loss") is not None
                    and research["confirmation_loss"] <= config.minimum_improvement,
                evaluation_costs={key:score[key]+impulse["costs"][key] for key in COST_KEYS})
            row["mse"] = max(row["random_mse"],impulse["mse"] or 0.)
            row["solved"] = row["mse"] <= plan["solved_threshold"]
            row["confirmed_and_solved"] = row["solved"] and row["fresh_confirmation_passed"]
        result.update(solved=sum(row["solved"] for row in result["tasks"]),
            mean_mse=mean(row["mse"] for row in result["tasks"]),
            fresh_confirmed=sum(row["fresh_confirmation_passed"] for row in result["tasks"]),
            confirmed_and_solved=sum(row["confirmed_and_solved"] for row in result["tasks"]),
            suffix_costs={key:result["costs"][key]-prefix["costs"][key] for key in COST_KEYS},
            evaluation_costs={key:sum(row["evaluation_costs"][key] for row in result["tasks"]) for key in COST_KEYS},
            training_cpu_seconds=prefix_cpu+training_cpu[arm],suffix_training_cpu_seconds=training_cpu[arm],
            evaluation_cpu_seconds=process_time()-start,continuation_action_counts=dict(counts),
            initial_workflows=initial_books[arm],final_workflows=report["knowledge"]["research_workflows"],
            workflow_knowledge_outcomes=[deepcopy(outcome)
                for method in report["knowledge"]["research_workflows"]["methods"]
                for outcome in method.get("knowledge_outcomes",[])],
            scheduler_events=deepcopy(report["scheduler_events"][len(prefix["scheduler_events"]):]),
            revisits=deepcopy(report["revisits"][len(prefix["revisits"]):]))
        results[arm] = result
    retained,removed = (results[arm] for arm in ARMS)
    return {"protocol":deepcopy(plan),"common_prefix_costs":prefix["costs"],
        "common_prefix_training_cpu_seconds":prefix_cpu,
        "intervention_active":bool(methods),"prefix_workflow_ids":[item["method_id"] for item in methods],
        "results":results,"paired_retained_minus_removed":{
            "costs":{key:retained["costs"][key]-removed["costs"][key] for key in COST_KEYS},
            "solved":retained["solved"]-removed["solved"],
            "confirmed_and_solved":retained["confirmed_and_solved"]-removed["confirmed_and_solved"],
            "mean_mse":retained["mean_mse"]-removed["mean_mse"]},
        "external_model_calls":0}


def summarize(completed, failures=(), *, bootstrap_samples=10000, bootstrap_seed=61007019):
    totals = {}
    for arm in ARMS:
        rows = [case["results"][arm] for case in completed]
        action_counts,scheduler_counts = Counter(),Counter()
        for row in rows:
            action_counts.update(row["continuation_action_counts"])
            scheduler_counts.update(event["action"] for event in row["scheduler_events"])
        totals[arm] = {"tasks":sum(len(row["tasks"]) for row in rows),
            "solved":sum(row["solved"] for row in rows),
            "fresh_confirmed":sum(row["fresh_confirmed"] for row in rows),
            "confirmed_and_solved":sum(row["confirmed_and_solved"] for row in rows),
            "costs":{key:sum(row["costs"][key] for row in rows) for key in COST_KEYS},
            "evaluation_costs":{key:sum(row["evaluation_costs"][key] for row in rows) for key in COST_KEYS},
            "continuation_action_counts":dict(action_counts),"scheduler_action_counts":dict(scheduler_counts),
            "revisit_count":sum(row["revisit_count"] for row in rows),
            "training_cpu_seconds":sum(row["training_cpu_seconds"] for row in rows),
            "evaluation_cpu_seconds":sum(row["evaluation_cpu_seconds"] for row in rows),
            "workflow_knowledge_outcomes":sum(len(row["workflow_knowledge_outcomes"]) for row in rows),
            "failed_tasks":[{"seed":case["protocol"]["seed"],"task_position":task["task_position"],
                "mse":task["mse"],"status":task["status"],"confirmation_loss":task["confirmation_loss"]}
                for case in completed for task in case["results"][arm]["tasks"] if not task["confirmed_and_solved"]]}
    groups = [[case for case in completed if case["protocol"]["instrument_mode"] == mode]
        for mode in ("finite","recurrent")]
    random,uncertainty = Random(bootstrap_seed),{}
    for key in (*COST_KEYS,"solved","confirmed_and_solved","mean_mse"):
        value = lambda case: (case["paired_retained_minus_removed"]["costs"][key]
            if key in COST_KEYS else case["paired_retained_minus_removed"][key])
        sampled = sorted(mean(value(random.choice(group)) for group in groups for _ in group)
            for _ in range(bootstrap_samples)) if completed else []
        uncertainty[key] = {"mean_difference":mean(value(case) for case in completed) if completed else None,
            "interval_95":[sampled[int(.025*(len(sampled)-1))],sampled[int(.975*(len(sampled)-1))]]
                if sampled else None}
    return {"planned_cases":len(completed)+len(failures),"completed_cases":len(completed),
        "execution_failures":list(failures),"intervention_active_cases":sum(case["intervention_active"] for case in completed),
        "totals":totals,"paired_retained_minus_removed":uncertainty,"external_model_calls":0}


def run_cohort(plan, output):
    """Run a registered roster; save each success or exception immediately."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/"protocol.json").write_text(json.dumps(plan,indent=2)+"\n")
    completed,failures = [],[]
    for case in plan["cases"]:
        path = output/f"case_{case['seed']}.json"
        try:
            record = run_study(case)
            completed.append(record)
            receipt = {"seed":case["seed"],"status":"completed", "joint":{
                arm:record["results"][arm]["confirmed_and_solved"] for arm in ARMS}}
        except Exception:
            record = {"seed":case["seed"],"status":"execution_failed","traceback":traceback.format_exc()}
            failures.append(record)
            receipt = {"seed":case["seed"],"status":"execution_failed"}
        path.write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(receipt),flush=True)
    summary = summarize(completed,failures,bootstrap_samples=plan["bootstrap_samples"],
        bootstrap_seed=plan["bootstrap_seed"])
    summary["protocol"] = deepcopy(plan)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def run_return_regression():
    """Replay the existing paid task-8 continuation fixture without supplied laws."""
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    config = ResearchConfig(observation_budget=192,compute_budget=120000,
        minimum_improvement=1e-7,max_rounds=4,max_candidates=16,active_probes=False,seed=11811)
    tasks = [{"task_index":8,"seed":11811},{"task_index":8,"seed":11812}]
    session = UnifiedResearchSession(config,tasks=tasks,measurement_budget=768)
    with laboratory_for(tasks[0]) as laboratory:
        previous = UnifiedResearchAgent(replace(config,compute_budget=512),
            measurement_budget=768,library=session.library).run(laboratory)
    session.state["results"].append({"task_position":0,"available_concepts":[],"research":previous})
    session.state["status"] = "paused"
    first = {key:deepcopy(previous.get(key)) for key in ("status","costs","confirmation_loss","research_workflow_id")}
    session = UnifiedResearchSession(state=json.loads(json.dumps(session.to_dict())))
    queued = session.run(max_tasks=1)
    offers = deepcopy(queued["pending_revisits"])
    producers = [{"concept_id":concept.concept_id,"production_methods":deepcopy(concept.production_methods)}
        for concept in session.library.concepts]
    session = UnifiedResearchSession(state=json.loads(json.dumps(queued)))
    session.run()
    report = session.evaluate(length=24,trajectories=2)
    return {"fixture":"tests/test_cognition_unified_revisits.py::revisited_stream",
        "research_config":asdict(config),"tasks":tasks,"initial_visit_compute_cap":512,
        "initial":first,"queued_offers":offers,"produced_concepts":producers,
        "costs":report["costs"],"screening_compute_cost":report["screening_compute_cost"],
        "scheduler_events":report["scheduler_events"],"revisit_count":len(report["revisits"]),
        "results":[{"task_position":entry["task_position"],"status":entry["research"]["status"],
            "confirmation_loss":entry["research"].get("confirmation_loss"),
            "costs":entry["research"]["costs"],"knowledge_return":entry["research"].get("knowledge_return"),
            "continuation_actions":[event for event in entry["research"]["actions"]
                if event["action"] in {"compare_continuation_candidates","validate_continued_program",
                    "compile_research_workflow","replay_research_workflow","revise_research_workflow"}]}
            for entry in report["results"]],
        "evaluation":report["evaluation"],"workflows":session.library.research_workflows.to_dict(),
        "external_model_calls":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=61007001)
    parser.add_argument("--apparatus-seed",type=int)
    parser.add_argument("--instrument-mode",choices=("finite","recurrent"),default="recurrent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohort",action="store_true",help="run the registered 12-case roster; output is a new directory")
    mode.add_argument("--return-regression",action="store_true",help="run the fixed existing partial-visit regression")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output")
    if args.cohort:
        run_cohort(cohort_protocol(),args.output)
        return
    result = (run_return_regression() if args.return_regression else
        run_study(protocol(args.seed,args.apparatus_seed,args.instrument_mode)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    compact = ({"revisit_count":result["revisit_count"],"costs":result["costs"]} if args.return_regression else
        {arm:{key:report[key] for key in ("solved","fresh_confirmed","confirmed_and_solved","costs")}
            for arm,report in result["results"].items()})
    print(json.dumps(compact),flush=True)


if __name__ == "__main__":
    main()
