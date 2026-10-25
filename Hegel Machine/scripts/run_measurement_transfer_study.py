#!/usr/bin/env python3
"""Compare persistent measurement knowledge with a fresh library on 3 x 24 tasks.

Run learning to completion before evaluation. Checkpoints resume the active
research visit as well as the shared library; neither arm calls an LLM.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import __version__
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import MeasurementResearchAgent
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig


COSTS = ("observation_cost", "measurement_cost", "compute_cost")
BASES = (241003, 451003, 661003)
ARMS = ("shared", "cold")


def write_json(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=False, allow_nan=True)


def protocol():
    config = ResearchConfig(observation_budget=96, compute_budget=60000, max_rounds=3)
    return {
        "experiment": "measurement_knowledge_long_sequence_transfer",
        "cohorts": {f"sequence_{index + 1}": [
            {"task_index": position % 4, "seed": base + 10007 * position}
            for position in range(24)] for index, base in enumerate(BASES)},
        "arms": list(ARMS), "config": asdict(config), "measurement_budget": 384,
        "total_limits_per_arm_per_sequence": {
            "observation_cost": 24 * 96, "measurement_cost": 24 * 384,
            "compute_cost": 24 * 60000},
        "backend": "process",
        "evaluation": {"seed_offset": 1000003, "length": 64, "trajectories": 4},
        "prefixes": [4, 8, 16, 24],
        "learning": "Shared retains one KnowledgeLibrary across the sequence; cold starts a new library for every task. Both retain within-task learning and use identical ResearchConfig values.",
        "stopping": "Task budgets and the sum of those budgets are fixed. An incomplete or inaccurate model remains a result; seeds and limits are not replaced after seeing outcomes.",
        "evaluation_timing": "All three sequences and both arms must finish learning before any independent evaluation rows are requested.",
        "scope": "Repeated instances of four finite process families; no claim about open-world transfer or equivalence to LLM compute.",
        "accounting": "Pilot and later research costs are separately recorded and both included. Final evaluation is additional paid scoring: 256 rows per available model. Model calls and reported tokens are zero.",
    }


def empty_state(cohort, arm, tasks):
    return {"cohort": cohort, "arm": arm, "tasks": tasks,
            "implementation_version": __version__, "status": "paused", "results": [],
            "costs": {key: 0 for key in COSTS}, "library": KnowledgeLibrary().to_dict(),
            "active": None}


def learn(spec, cohort, arm, directory, max_tasks):
    path = directory / f"{cohort}_{arm}_learning.json"
    state = json.loads(path.read_text()) if path.exists() else empty_state(cohort, arm, spec["cohorts"][cohort])
    if state["implementation_version"] != __version__:
        raise ValueError("Resume learning with the saved implementation version or use a new output directory")
    completed = len(state["results"])
    target = len(state["tasks"]) if max_tasks is None else min(len(state["tasks"]), completed + max_tasks)
    while len(state["results"]) < target:
        position = len(state["results"])
        task = state["tasks"][position]
        active = state["active"]
        library = (KnowledgeLibrary.from_dict(active["library"]) if active is not None else
                   KnowledgeLibrary.from_dict(state["library"]) if arm == "shared" else KnowledgeLibrary())
        if active is None:
            active = {"research_state": None, "library": library.to_dict(), "stage_costs": [],
                      "initial_costs": None,
                      "inherited_history_lengths": {item.concept_id: len(item.reuse_history)
                                                     for item in library.concepts}}
            state["active"] = active
        remaining = {key: spec["total_limits_per_arm_per_sequence"][key] - state["costs"][key]
                     for key in COSTS}
        config = replace(ResearchConfig(**spec["config"]), seed=task["seed"],
                         observation_budget=min(spec["config"]["observation_budget"], remaining["observation_cost"]),
                         compute_budget=min(spec["config"]["compute_budget"], remaining["compute_cost"]))

        def checkpoint(result):
            previous = active["stage_costs"][-1]["cumulative_costs"] if active["stage_costs"] else dict.fromkeys(COSTS, 0)
            paid = {key: result["costs"][key] for key in COSTS}
            active["stage_costs"].append({"next_stage": result["research_state"]["stage"],
                "cumulative_costs": paid, "incremental_costs": {key: paid[key] - previous[key] for key in COSTS}})
            if active["initial_costs"] is None:
                active["initial_costs"] = paid
            active["research_state"] = result["research_state"]
            active["library"] = library.to_dict()
            write_json(path, state)

        with make_lab(task["task_index"], task["seed"], backend=spec["backend"]) as lab:
            result = MeasurementResearchAgent(config, library=library,
                measurement_budget=min(spec["measurement_budget"], remaining["measurement_cost"])).run(
                    lab, research_state=active["research_state"], on_checkpoint=checkpoint)
        histories = active["inherited_history_lengths"]
        reuse = [{"concept_id": item.concept_id, **event}
                 for item in library.concepts if item.concept_id in histories
                 for event in item.reuse_history[histories[item.concept_id]:]]
        result = deepcopy(result)
        # Completed tasks need their executable model and research decisions,
        # while only the active task needs its full observations for resumption.
        result.pop("research_state", None)
        result.pop("pilot_observations", None)
        if result.get("episode"):
            result["episode"].pop("research_state", None)
        result.update(**task, task_position=position, initial_costs=active["initial_costs"],
            later_costs={key: result["costs"][key] - active["initial_costs"][key] for key in COSTS},
            stage_costs=active["stage_costs"], inherited_reuse_events=reuse,
            inherited_executed_reuses=sum(event.get("kind") == "executed_program" for event in reuse))
        state["results"].append(result)
        state["costs"] = {key: sum(row["costs"][key] for row in state["results"]) for key in COSTS}
        state["library"] = library.to_dict() if arm == "shared" else KnowledgeLibrary().to_dict()
        state["active"] = None
        state["status"] = "completed" if len(state["results"]) == len(state["tasks"]) else "paused"
        write_json(path, state)
        print(json.dumps({"cohort": cohort, "arm": arm, "completed": len(state["results"]),
              "selected_channels": result["selected_channels"], "costs": result["costs"],
              "inherited_executed_reuses": result["inherited_executed_reuses"]}), flush=True)
    write_json(path, state)


def summarize_rows(rows):
    scores = [row["evaluation"]["mse"] for row in rows if row["evaluation"]["mse"] is not None]
    initial_reuse = [view.get("reuse", {}) for row in rows for view in row["selection"]]
    measurement_events = [event for row in rows for event in row["measurement_events"]]
    return {"tasks": len(rows), "completed_models": len(scores),
        "mean_mse": mean(scores) if scores else None, "maximum_mse": max(scores) if scores else None,
        "costs": {key: sum(row["costs"][key] for row in rows) for key in COSTS},
        "initial_costs": {key: sum(row["initial_costs"][key] for row in rows) for key in COSTS},
        "later_costs": {key: sum(row["later_costs"][key] for row in rows) for key in COSTS},
        "inherited_executed_reuses": sum(row["inherited_executed_reuses"] for row in rows),
        "tasks_with_inherited_executed_reuse": sum(row["inherited_executed_reuses"] > 0 for row in rows),
        "initial_whole_call_trials": sum(trial.get("call", False) for reuse in initial_reuse for trial in reuse.get("trials", [])),
        "initial_views_using_inherited_model": sum(reuse.get("selected_source") is not None for reuse in initial_reuse),
        "initial_views_skipping_search": sum(reuse.get("skipped_search", False) for reuse in initial_reuse),
        "tasks_with_initial_search_reuse": sum(any(view.get("reuse", {}).get("skipped_search", False)
                                                  for view in row["selection"]) for row in rows),
        "initial_reuse_compute_cost": sum(reuse.get("compute_cost", 0) for reuse in initial_reuse),
        "measurement_comparisons": sum(event["action"] == "recompare_measurements" for event in measurement_events),
        "adopted_measurement_changes": sum(event.get("adopted", False)
            and set(event["before_channels"]) != set(event["selected_channels"]) for event in measurement_events),
        "representation_revision_attempts": sum(event["action"] == "revise_measurement_representation"
                                                 for event in measurement_events),
        "adopted_representation_changes": sum(event.get("representation_changed", False)
                                                for event in measurement_events),
        "same_view_representation_changes": sum(event.get("adopted", False)
            and set(event["before_channels"]) == set(event["selected_channels"]) for event in measurement_events),
        "evaluation_rows": sum(row["evaluation"]["rows"] for row in rows),
        "model_calls": 0, "reported_tokens": 0}


def difference(shared, cold):
    return {"mean_mse": shared["mean_mse"] - cold["mean_mse"] if shared["mean_mse"] is not None and cold["mean_mse"] is not None else None,
            **{name: {key: shared[name][key] - cold[name][key] for key in COSTS}
               for name in ("costs", "initial_costs", "later_costs")}}


def evaluate(spec, directory, baseline_path=None):
    learned = {(cohort, arm): json.loads((directory / f"{cohort}_{arm}_learning.json").read_text())
               for cohort in spec["cohorts"] for arm in ARMS}
    if any(state["status"] != "completed" for state in learned.values()):
        raise ValueError("Finish all declared learning sequences before final evaluation")
    cohorts, aggregate = {}, {arm: [] for arm in ARMS}
    for cohort in spec["cohorts"]:
        arms = {}
        for arm in ARMS:
            rows = deepcopy(learned[cohort, arm]["results"])
            for result in rows:
                episode = result["episode"]
                evaluation = {"mse": None, "error": "no_completed_model", "rows": 0}
                if episode is not None:
                    options = spec["evaluation"]
                    with make_lab(result["task_index"], result["seed"], backend=spec["backend"]) as lab:
                        observations = lab.evaluation_rows(result["selected_channels"],
                            seed=result["seed"] + options["seed_offset"],
                            length=options["length"], trajectories=options["trajectories"])
                    evaluation = {"mse": loss(Program.from_dict(episode["program"]), observations, episode["fallback"], .5),
                        "rows": len(observations), "scoring_compute_cost": len(observations),
                        "seed": result["seed"] + options["seed_offset"]}
                result["evaluation"] = evaluation
            arms[arm] = {**summarize_rows(rows), "results": rows}
            aggregate[arm].extend(rows)
        prefixes = []
        for length in spec["prefixes"]:
            prefix = {arm: summarize_rows(arms[arm]["results"][:length]) for arm in ARMS}
            prefixes.append({"tasks": length, **prefix, "shared_minus_cold": difference(prefix["shared"], prefix["cold"])})
        cohorts[cohort] = {"arms": arms, "prefixes": prefixes,
                           "shared_minus_cold": difference(arms["shared"], arms["cold"])}
    totals = {arm: summarize_rows(rows) for arm, rows in aggregate.items()}
    summary = {"status": "completed", "protocol": spec,
        "implementation_version": next(iter(learned.values()))["implementation_version"],
        "cohorts": cohorts, "aggregate": totals,
        "shared_minus_cold": difference(totals["shared"], totals["cold"])}
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text())
        if baseline["protocol"] != spec:
            raise ValueError("Compare implementations only under the same saved protocol")
        previous = {"path": str(baseline_path.resolve()), "version": baseline["implementation_version"],
            "aggregate_shared_difference": difference(totals["shared"], baseline["aggregate"]["shared"]),
            "aggregate_cold_difference": difference(totals["cold"], baseline["aggregate"]["cold"]),
            "cohorts": {}}
        for cohort, payload in cohorts.items():
            old = baseline["cohorts"][cohort]
            previous["cohorts"][cohort] = {
                "shared_difference": difference(payload["arms"]["shared"], old["arms"]["shared"]),
                "prefixes": [{"tasks": new["tasks"],
                    "shared_difference": difference(new["shared"], past["shared"])}
                    for new, past in zip(payload["prefixes"], old["prefixes"])]}
        summary["previous_implementation"] = previous
    write_json(directory / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"protocol", "cohorts"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("learn", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, help="Maximum additional tasks per arm and sequence")
    parser.add_argument("--baseline", type=Path, help="Completed previous-implementation summary for evaluation comparison")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    spec = protocol()
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Resume with the saved study inputs or choose a new output directory")
    write_json(path, spec)
    if args.phase == "learn":
        for cohort in spec["cohorts"]:
            for arm in ARMS:
                learn(spec, cohort, arm, args.output, args.steps)
    else:
        evaluate(spec, args.output, args.baseline)


if __name__ == "__main__":
    main()
