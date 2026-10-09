#!/usr/bin/env python3
"""Compare conditional retention, unmarked retention, and discard of a failed response."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_composed_local_cycle import protocol as cycle_protocol
from run_current_offline_ablation import _compact
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for

COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("conditional_retention", "unmarked_retention", "global_discard")


def protocol(seed=10078, *, apparatus_seed=9013, profile="development"):
    plan = cycle_protocol(seed, apparatus_seed=apparatus_seed, profile=profile)
    plan.update(experiment="observed_failure_scoped_retention", arms={arm: {} for arm in ARMS},
        challenge={"length": 24, "seed_offset": 700001, "task_position": 3})
    plan["information_contract"] = "Learn the first three tasks from an empty library once. " \
        "Challenge the newly learned, confirmed complete third response on the fourth task. " \
        "After measured failure, clone conditional retention, unmarked retention, and global discard. " \
        "Each suffix task uses a fresh one-task session carrying only its arm's library. " \
        "Complete all learning before terminal scoring."
    plan["budget_contract"] = "Same six initial task caps; no suffix pooling or cached reports. " \
        "Deduct the common challenge's observation, measurement and prediction costs from task 3. " \
        "Charge the common prefix and challenge once to each arm's total."
    plan["acceptance"] = "Include every case, including absent sources and challenges that pass. " \
        "Compare terminal quality, return-task source calls, and actual costs across all three arms."
    plan["scope"] = "A fixed random challenge in one finite object-exchange family. Tests retention " \
        "of a measured-failing complete response for later reassessment, and the effect of its " \
        "conditional-prior record. Does not test autonomous challenge selection or a learned universal validity predicate."
    return plan


def _exact_response_source(prefix, library, tolerance):
    research = prefix["results"][2]["research"]
    if research["status"] != "completed" or research.get("confirmation_loss") is None \
            or research["confirmation_loss"] > tolerance:
        return None
    response = Program.from_dict(research.get("measurement_program") or research["program"])
    canonical = response.expanded().simplified()
    return next((concept for concept in library.concepts
        if concept.concept_id in research["concepts_added"]
        and concept.program.expanded().simplified() == canonical), None)


def _challenge(plan, prefix, library, factory):
    source = _exact_response_source(prefix, library, plan["research_config"]["minimum_improvement"])
    result = {"status": "intervention_not_triggered", "activated": False, "reason": "no_exact_confirmed_new_response",
        "source_id": source.concept_id if source else None,
        "program": source.program.to_dict() if source else None, "loss": None,
        "costs": dict.fromkeys(COST_KEYS, 0), "channels": [], "probe": None, "task_id": None}
    if source is None:
        return result
    task = plan["tasks"][plan["challenge"]["task_position"]]
    with factory(task) as laboratory:
        required = {name for name, _ in source.program.observed_variables() if name != "__target__"}
        views = getattr(laboratory, "measurement_views",
            ((), tuple(channel.name for channel in laboratory.channels)))
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        channels = min((tuple(view) for view in views
            if required <= set(laboratory.spec_for(view).feature_names)),
            key=lambda view: (sum(rates[name] for name in view), len(view)))
        probe = laboratory.random_probe(task["seed"] + plan["challenge"]["seed_offset"],
            length=plan["challenge"]["length"], name="retained_response_challenge")
        batch = laboratory.acquire(probe, channels)
        measured = loss(source.program, batch.observations, prefix["results"][2]["research"]["fallback"],
            laboratory.spec.risk_quantile)
        failed = measured > plan["research_config"]["minimum_improvement"]
        result.update(status="counterexample" if failed else "intervention_not_triggered",
            activated=failed,
            reason="measured_response_failure" if failed else "challenge_passed", loss=measured,
            channels=list(channels), probe=asdict(probe), task_id=laboratory.spec.task_id,
            costs={"observation_cost": batch.observation_cost, "measurement_cost": batch.measurement_cost,
                "compute_cost": len(batch.observations)})
    return result


def _conditional_library(library, challenge):
    result = KnowledgeLibrary.from_dict(library.to_dict())
    if challenge["activated"]:
        for prior in result.meta_priors:
            if prior["source_concept_id"] == challenge["source_id"]:
                prior.setdefault("counterexamples", []).append({"task_id": challenge["task_id"],
                    "loss": challenge["loss"], "kind": "failed_retained_call",
                    "program": challenge["program"], "replacement_concept_id": None})
                prior["status"] = "conditional"
    return result


def run_benchmark(plan, *, lab_factory=None):
    factory, config = lab_factory or laboratory_for, ResearchConfig(**plan["research_config"])
    start = process_time()
    initial = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary(), lab_factory=factory)
    prefix = initial.run(max_tasks=3)
    prefix_cpu = process_time() - start
    unmarked = KnowledgeLibrary.from_dict(prefix["knowledge"])
    start = process_time()
    challenge = _challenge(plan, prefix, unmarked, factory)
    conditional = _conditional_library(unmarked, challenge)
    triggered = challenge["activated"]
    libraries = {"conditional_retention": conditional, "unmarked_retention": unmarked,
        "global_discard": conditional.without({challenge["source_id"]}) if triggered
            else KnowledgeLibrary.from_dict(unmarked.to_dict())}
    challenge_cpu = process_time() - start
    sources = {name: prefix["results"][index]["research"]["concepts_added"]
        for index, name in enumerate(("a", "b"))}
    sources["c"] = [challenge["source_id"]] if challenge["source_id"] else []
    available = {arm: [c.concept_id for c in library.concepts] for arm, library in libraries.items()}
    feedback = {arm: [{"status": prior["status"],
        "counterexample_count": len(prior.get("counterexamples", []))}
        for prior in library.meta_priors if prior["source_concept_id"] == challenge["source_id"]]
        for arm, library in libraries.items()}
    original = {c.concept_id for c in unmarked.concepts}
    removed = {arm: sorted(original - set(ids)) for arm, ids in available.items()}
    sessions, suffix_cpu = {}, {}
    for arm, library in libraries.items():
        start = process_time()
        combined = UnifiedResearchSession(config, tasks=plan["tasks"],
            measurement_budget=plan["measurement_budget"], library=library, lab_factory=factory)
        combined.state["results"] = deepcopy(prefix["results"])
        combined.state["screening_compute_cost"] = prefix["screening_compute_cost"]
        for position in range(3, len(plan["tasks"])):
            charged = challenge["costs"] if position == 3 else dict.fromkeys(COST_KEYS, 0)
            local_config = replace(config, observation_budget=config.observation_budget - charged["observation_cost"],
                compute_budget=config.compute_budget - charged["compute_cost"])
            local = UnifiedResearchSession(local_config, tasks=[plan["tasks"][position]], library=library,
                measurement_budget=plan["measurement_budget"] - charged["measurement_cost"], lab_factory=factory)
            local.run()
            entry = deepcopy(local.state["results"][0])
            entry["task_position"] = position
            combined.state["results"].append(entry)
            combined.state["screening_compute_cost"] += local.state["screening_compute_cost"]
            combined.state["revisits"].extend({**visit, "task_position": position}
                for visit in local.state["revisits"])
        combined.state["status"] = "completed"
        sessions[arm], suffix_cpu[arm] = combined, process_time() - start
    results = {}
    for arm, session in sessions.items():
        start = process_time()
        report = session.evaluate(**plan["evaluation"])
        evaluation_cpu = process_time() - start
        report["costs"] = {key: report["costs"][key] + challenge["costs"][key] for key in COST_KEYS}
        result = _compact(report, plan)
        for row, entry, score in zip(result["tasks"], report["results"], report["evaluation"]):
            research = entry["research"]
            definitions = _called_definitions(Program.from_dict(research["program"]))
            row["nested_call_sources"] = sorted({d.source_concept_id for d in definitions if d.source_concept_id})
            row["concepts_added"] = research["concepts_added"]
            row["calls_original_sources"] = {name: bool(set(ids).intersection(row["nested_call_sources"]))
                for name, ids in sources.items()}
            row["fresh_confirmation_passed"] = research["status"] == "completed" \
                and research.get("confirmation_loss") is not None \
                and research["confirmation_loss"] <= config.minimum_improvement
            row["evaluation_costs"] = {key: score[key] for key in COST_KEYS}
            row["resource_limits"] = {
                "observation_cost": config.observation_budget,
                "measurement_cost": plan["measurement_budget"], "compute_cost": config.compute_budget}
            if row["task_position"] == 3:
                row["resource_limits"] = {key: value - challenge["costs"][key]
                    for key, value in row["resource_limits"].items()}
        result.update(fresh_confirmed=sum(row["fresh_confirmation_passed"] for row in result["tasks"]),
            challenge_costs=deepcopy(challenge["costs"]),
            evaluation_costs={key: sum(row["evaluation_costs"][key] for row in result["tasks"]) for key in COST_KEYS},
            cpu_seconds={"common_prefix": prefix_cpu, "common_challenge": challenge_cpu,
                "suffix_training": suffix_cpu[arm], "terminal_evaluation": evaluation_cpu,
                "total_training": prefix_cpu + challenge_cpu + suffix_cpu[arm]},
            training_cpu_seconds=prefix_cpu + challenge_cpu + suffix_cpu[arm],
            evaluation_cpu_seconds=evaluation_cpu,
            available_source_ids=[c.concept_id for c in session.library.concepts])
        results[arm] = result
    retained = results["conditional_retention"]
    paired = {arm: {"cost_difference": {key: retained["costs"][key] - other["costs"][key] for key in COST_KEYS},
        "mse_difference": [a["mse"] - b["mse"] for a, b in zip(retained["tasks"], other["tasks"])],
        "return_task_cost_difference": {key: retained["tasks"][4]["costs"][key] - other["tasks"][4]["costs"][key]
            for key in COST_KEYS}, "both_solve_all": retained["solved"] == other["solved"] == len(plan["tasks"])}
        for arm, other in results.items() if arm != "conditional_retention"}
    return {"protocol": deepcopy(plan), "source_ids": sources, "challenge": challenge,
        "common_prefix_costs": prefix["costs"], "initial_available_sources": available,
        "initial_source_feedback": feedback, "removed_concept_ids": removed, "results": results,
        "paired_conditional_vs_control": paired, "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10078)
    parser.add_argument("--apparatus-seed", type=int, default=9013)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output")
    result = run_benchmark(protocol(args.seed, apparatus_seed=args.apparatus_seed, profile=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"challenge": result["challenge"]["status"],
        "arms": {arm: {key: item[key] for key in ("solved", "fresh_confirmed", "costs")}
            for arm, item in result["results"].items()}}), flush=True)


if __name__ == "__main__":
    main()
