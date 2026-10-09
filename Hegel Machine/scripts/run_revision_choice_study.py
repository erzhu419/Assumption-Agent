#!/usr/bin/env python3
"""Observe research decisions after no change, response reweighting, or relation change."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_composed_local_cycle import protocol as cycle_protocol
from run_current_offline_ablation import _compact
from run_scoped_retention_comparison import _exact_response_source
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for

COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")


def protocol(seed=10078, apparatus_seed=9013, profile="development"):
    base = cycle_protocol(seed, apparatus_seed=apparatus_seed, profile=profile)
    first = {**base["tasks"][2], "seed": base["tasks"][3]["seed"]}
    returned = {**base["tasks"][2], "seed": base["tasks"][4]["seed"]}
    return {"experiment": "component_revision_choice", "tasks": base["tasks"][:3],
        "branches": {name: [{**first, **changes}, deepcopy(returned)] for name, changes in (
            ("unchanged", {}), ("change_a", {"total_weight": 1.35}),
            ("change_b", {"relation_radius": .45}))},
        "research_config": base["research_config"], "measurement_budget": base["measurement_budget"],
        "evaluation": base["evaluation"], "solved_threshold": base["solved_threshold"],
        "information_contract": "Learn A, B, and their response C from an empty library in three initial visits. "
            "Clone the acquired library into three branches. Every suffix task uses a new one-task session "
            "carrying only its branch's library, with no cross-task budget borrowing or cached reports. "
            "No artificial challenge is inserted. All branches finish learning before terminal evaluation.",
        "budget_contract": "Every initial visit has the same observation, measurement and compute caps. "
            "Report suffix costs and costs including the common prefix separately. Evaluate the prefix once.",
        "reference_contract": "After all learning, score the frozen third response on each suffix task's "
            "terminal controls using the cheapest legal view supporting its variables. Record its evaluation "
            "costs separately. This fixed prediction reference is not a research algorithm control.",
        "scope": "A finite component-change study. Branch labels and intervention parameters configure "
            "the laboratory, not the researcher. Report actual actions and retained definitions without "
            "inferring a stopping reason that the researcher did not emit."}


def _frozen_reference(program, fallback, task, evaluation, factory):
    with factory(task) as laboratory:
        required = {name for name, _ in program.observed_variables() if name != "__target__"}
        views = getattr(laboratory, "measurement_views",
            ((), tuple(channel.name for channel in laboratory.channels)))
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        channels = min((tuple(view) for view in views
            if required <= set(laboratory.spec_for(view).feature_names)),
            key=lambda view: (sum(rates[name] for name in view), len(view)))
        seed = task["seed"] + evaluation["seed_offset"]
        rows = laboratory.evaluation_rows(channels, seed=seed,
            length=evaluation["length"], trajectories=evaluation["trajectories"])
        measured = loss(program, rows, fallback, laboratory.spec.risk_quantile)
    return {"mse": measured, "rows": len(rows), "channels": list(channels), "seed": seed,
        "costs": {"observation_cost": len(rows), "measurement_cost": len(rows) * sum(rates[name] for name in channels),
            "compute_cost": len(rows)}}


def _details(session, plan, original_definitions, *, full_trace=False):
    start = process_time()
    report = session.evaluate(**plan["evaluation"])
    evaluation_cpu = process_time() - start
    result = _compact(report, plan)
    for row, entry, score in zip(result["tasks"], report["results"], report["evaluation"]):
        research = entry["research"]
        definitions = _called_definitions(Program.from_dict(research["program"]))
        row.update(program=research["program"], measurement_program=research.get("measurement_program"),
            measurement_channels=research.get("measurement_channels"), concepts_added=research["concepts_added"],
            nested_call_sources=sorted({d.source_concept_id for d in definitions if d.source_concept_id}),
            unchanged_source_definitions_called={name: any(d in originals for d in definitions)
                for name, originals in original_definitions.items()},
            fresh_confirmation_passed=research["status"] == "completed"
                and research.get("confirmation_loss") is not None
                and research["confirmation_loss"] <= plan["research_config"]["minimum_improvement"],
            evaluation_costs={key: score[key] for key in COST_KEYS})
        if full_trace:
            row.update(actions=deepcopy(research["actions"]), transfer=deepcopy(research["transfer"]),
                research_program=deepcopy(research.get("research_program")),
                mechanism_revision=deepcopy(research.get("mechanism_revision")),
                stopping_reason=research.get("stopping_reason", research.get("stop_reason", "unknown")))
    result.update(fresh_confirmed=sum(row["fresh_confirmation_passed"] for row in result["tasks"]),
        evaluation_costs={key: sum(row["evaluation_costs"][key] for row in result["tasks"]) for key in COST_KEYS},
        evaluation_cpu_seconds=evaluation_cpu,
        available_source_ids=[c.concept_id for c in session.library.concepts])
    return result


def run_study(plan, lab_factory=None):
    factory, config = lab_factory or laboratory_for, ResearchConfig(**plan["research_config"])
    start = process_time()
    initial = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary(), lab_factory=factory)
    prefix_state = initial.run(max_tasks=3)
    prefix_cpu = process_time() - start
    prefix_library = KnowledgeLibrary.from_dict(prefix_state["knowledge"])
    source_ids = {name: prefix_state["results"][i]["research"]["concepts_added"]
        for i, name in enumerate(("a", "b"))}
    source_c = _exact_response_source(prefix_state, prefix_library, config.minimum_improvement)
    source_ids["c"] = [source_c.concept_id] if source_c else []
    original_definitions = {name: [d for d in prefix_library.definitions.values()
        if d.source_concept_id in source_ids[name]] for name in ("a", "b")}
    frozen = prefix_state["results"][2]["research"]
    frozen_program = Program.from_dict(frozen["program"])
    sessions, training_cpu = {}, {}
    for name, tasks in plan["branches"].items():
        start = process_time()
        library = KnowledgeLibrary.from_dict(prefix_library.to_dict())
        combined = UnifiedResearchSession(config, tasks=tasks, measurement_budget=plan["measurement_budget"],
            library=library, lab_factory=factory)
        for position, task in enumerate(tasks):
            local = UnifiedResearchSession(config, tasks=[task], measurement_budget=plan["measurement_budget"],
                library=library, lab_factory=factory)
            local.run()
            entry = deepcopy(local.state["results"][0])
            entry["task_position"] = position
            combined.state["results"].append(entry)
            combined.state["screening_compute_cost"] += local.state["screening_compute_cost"]
            combined.state["revisits"].extend({**visit, "task_position": position}
                for visit in local.state["revisits"])
        combined.state["status"] = "completed"
        sessions[name], training_cpu[name] = combined, process_time() - start
    # These containers only score frozen programs after every branch finishes.
    prefix_session = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=prefix_library, lab_factory=factory)
    prefix_session.state.update(results=prefix_state["results"], status="completed",
        screening_compute_cost=prefix_state["screening_compute_cost"], revisits=prefix_state["revisits"])
    prefix = _details(prefix_session, plan, original_definitions)
    prefix.update(training_cpu_seconds=prefix_cpu, frozen_response_program=frozen_program.to_dict(),
        frozen_response_confirmed=frozen["status"] == "completed" and frozen.get("confirmation_loss") is not None
            and frozen["confirmation_loss"] <= config.minimum_improvement)
    results = {}
    for name, session in sessions.items():
        branch_plan = {**plan, "tasks": plan["branches"][name]}
        result = _details(session, branch_plan, original_definitions, full_trace=True)
        start = process_time()
        for row, task in zip(result["tasks"], branch_plan["tasks"]):
            row["frozen_response_reference"] = _frozen_reference(frozen_program, frozen["fallback"],
                task, plan["evaluation"], factory)
        result.update(reference_evaluation_cpu_seconds=process_time() - start,
            reference_evaluation_costs={key: sum(row["frozen_response_reference"]["costs"][key]
                for row in result["tasks"]) for key in COST_KEYS},
            training_cpu_seconds=training_cpu[name],
            total_training_cpu_seconds_with_prefix=prefix_cpu + training_cpu[name],
            total_with_prefix={key: prefix["costs"][key] + result["costs"][key] for key in COST_KEYS})
        results[name] = result
    return {"protocol": deepcopy(plan), "prefix": prefix, "source_ids": source_ids,
        "results": results, "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10078)
    parser.add_argument("--apparatus-seed", type=int, default=9013)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output")
    result = run_study(protocol(args.seed, args.apparatus_seed, args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("solved", "fresh_confirmed", "costs", "total_with_prefix")}
        for name, row in result["results"].items()}), flush=True)


if __name__ == "__main__":
    main()
