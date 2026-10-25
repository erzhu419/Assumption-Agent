#!/usr/bin/env python3
"""Compare the saved v2 algorithm, live v3 and bounded local ablations.

Examples:
  python3 scripts/run_cognition_v3_research.py --partitions development confirmation
  python3 scripts/run_cognition_v3_research.py --partitions forward --output-dir artifacts/cognition_v3_forward

Forward is evaluated last when several partitions are requested. Run it only
after fixing the development implementation. No model service is called.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "artifacts"))

from cognition_v2_reference import experiments as reference_experiments
from cognition_v2_reference.models import ResearchConfig as ReferenceConfig
from hegel_machine import __version__ as live_version
from hegel_machine.cognition import experiments as live_experiments
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_protocol import summarize_research


PARTITIONS = ("development", "confirmation", "forward")
DEFAULT_SEEDS = {"development": (7, 11, 19), "confirmation": (23, 47, 83),
                 "forward": (101, 211, 307)}
SYSTEMS = ("reference_v2", "cognitive_v3", "raw_disagreement",
           "retrospective_selection", "without_warm_start", "without_knowledge")
SYSTEM_CHANGES = {
    "cognitive_v3": {},
    "raw_disagreement": {"experiment_policy": "disagreement"},
    "retrospective_selection": {"selection_policy": "retrospective"},
    "without_warm_start": {"use_warm_start": False},
    "without_knowledge": {"use_memory": False},
    "erase_specific_concept": {},
}


def run_system(system, config, *, remove_concept_id=None):
    """Keep the old label adaptation entirely in the experiment script."""
    started = perf_counter()
    if system == "reference_v2":
        shared = {field.name: getattr(config, field.name) for field in fields(ReferenceConfig)}
        actual_partition = config.partition
        if actual_partition == "forward":
            shared["partition"] = "confirmation"
        reference_config = ReferenceConfig(**shared)
        original_suite = reference_experiments.make_suite
        try:
            if actual_partition == "forward":
                reference_experiments.make_suite = lambda seed, partition: make_suite(seed, partition="forward")
            result = reference_experiments.run_sequence(reference_config)
        finally:
            reference_experiments.make_suite = original_suite
        result["reference_config_partition"] = reference_config.partition
        result["partition"] = actual_partition
        result["algorithm_version"] = "0.4.0_saved_cognition_package"
    else:
        changed = replace(config, **SYSTEM_CHANGES[system])
        result = live_experiments.run_sequence(
            changed, remove_concept_ids={remove_concept_id} if remove_concept_id else None)
        result["algorithm_version"] = live_version
    result["system"] = system
    result["execution_wall_time_seconds"] = perf_counter() - started
    return result


def compact_run(run):
    result = {key: value for key, value in run.items() if key not in {"episodes", "knowledge"}}
    result["episodes"] = []
    for episode in run["episodes"]:
        compact = {key: value for key, value in episode.items()
                   if key not in {"actions", "questions", "plans", "program"}}
        compact["action_counts"] = dict(Counter(action["action"] for action in episode.get("actions", [])))
        compact["evaluation"] = {key: value for key, value in episode["evaluation"].items()
                                 if key != "prediction_examples"}
        if "final_counterfactual" in compact["evaluation"]:
            compact["evaluation"]["final_counterfactual"] = {
                key: value for key, value in compact["evaluation"]["final_counterfactual"].items()
                if key != "events"}
        result["episodes"].append(compact)
    return result


def selected_cases(run):
    """Save two largest prediction losses and one risk case from the first seed."""
    completed = [episode for episode in run["episodes"] if episode["status"] == "completed"]
    prediction = sorted((row for row in completed if row["evaluation"]["loss_kind"] == "squared_error"),
                        key=lambda row: row["evaluation"]["loss"], reverse=True)[:2]
    risk = sorted((row for row in completed if row["evaluation"]["loss_kind"] != "squared_error"),
                  key=lambda row: row["evaluation"]["loss"], reverse=True)[:1]
    return [{"system": run["system"], "seed": run["config"]["seed"], **row}
            for row in (*prediction, *risk)]


def acquired_concept(run):
    for episode in run["episodes"]:
        if episode["evaluation"]["task_metadata"]["phase"] == "transfer":
            break
        if episode.get("concepts_added"):
            return episode["concepts_added"][0]
    return None


def partition_summary(payload):
    # Reuse the seed-level statistics while preserving the actual arm names.
    by_seed = {seed: {} for seed in payload["seeds"]}
    for run in payload["runs"]:
        name = "cognitive_agent" if run["system"] == "cognitive_v3" else run["system"]
        by_seed[run["config"]["seed"]][name] = run
    result = summarize_research([{"runs": runs} for runs in by_seed.values()],
                                payload["seeds"], payload["partition"])
    if "cognitive_agent" in result["systems"]:
        result["systems"]["cognitive_v3"] = result["systems"].pop("cognitive_agent")
    for row in result["seed_results"]:
        if "cognitive_agent" in row["systems"]:
            row["systems"]["cognitive_v3"] = row["systems"].pop("cognitive_agent")
    result["paired_loss_alternative_minus_v3"] = result.pop("paired_loss_alternative_minus_cognitive")
    result["inference"]["paired_sign"] = "positive favors cognitive_v3"
    result["experiment"] = "v2_v3_matched_local_research"
    result["completed_runs"] = len(payload["runs"])
    result["planned_runs"] = len(payload["seeds"]) * len(payload["systems"])
    result["algorithm_versions"] = {run["system"]: run["algorithm_version"] for run in payload["runs"]}
    result["design"] = payload["design"]
    result["limitations"] = payload["limitations"]
    return result


def write_json(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=list(PARTITIONS[:2]))
    parser.add_argument("--systems", nargs="+", choices=(*SYSTEMS, "erase_specific_concept"), default=list(SYSTEMS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Override the three fixed seeds for each requested partition")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "cognition_v3_research")
    parser.add_argument("--observation-budget", type=int, default=96)
    parser.add_argument("--compute-budget", type=int, default=24000)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Retain already completed runs with the same design")
    args = parser.parse_args()
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not repeat")
    if len(set(args.systems)) != len(args.systems):
        parser.error("systems must not repeat")
    if "erase_specific_concept" in args.systems and "cognitive_v3" not in args.systems:
        parser.error("erase_specific_concept needs cognitive_v3 to choose an actually acquired concept")
    systems = [name for name in (*SYSTEMS, "erase_specific_concept") if name in args.systems]
    base = ResearchConfig(observation_budget=args.observation_budget, compute_budget=args.compute_budget,
                          max_rounds=args.max_rounds, proposer="search")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for partition in PARTITIONS:
        if partition not in args.partitions:
            continue
        seeds = args.seeds or list(DEFAULT_SEEDS[partition])
        design = {"base_config": asdict(replace(base, partition=partition)),
                  "seeds": seeds, "systems": systems, "reference_version": "0.4.0",
                  "live_version": live_version,
                  "system_changes": {name: SYSTEM_CHANGES[name] for name in systems if name in SYSTEM_CHANGES},
                  "pairing": "same suite seed, initial and catalog probes, final evaluation and budget caps; adaptive queries may differ",
                  "expression_language": "v3 adds learned startup branches with history availability; without_warm_start removes that option",
                  "specific_deletion": "first actually learned acquisition concept and its recorded derivatives, before first transfer",
                  "reference_forward_adapter": "saved v2 config uses confirmation label; experiment injects the live forward environment generator"}
        artifact = args.output_dir / f"{partition}_runs.json"
        summary_path = args.output_dir / f"{partition}_summary.json"
        if artifact.exists():
            if not args.resume:
                parser.error(f"{artifact} exists; choose another output directory or use --resume")
            payload = json.loads(artifact.read_text())
            if payload["design"] != design:
                parser.error(f"{artifact} uses a different experiment design")
        else:
            payload = {"experiment": "v2_v3_matched_local_research", "partition": partition,
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "seeds": seeds, "systems": systems, "design": design, "runs": [], "case_traces": [],
                       "limitations": [
                           "Development and earlier confirmation outcomes informed v3 development; they are retrospective comparisons.",
                           "Forward is a public synthetic prospective partition; its generator is visible and not held by an independent custodian.",
                           "v3 changes both experiment/selection policy and the available startup expression language; ablations separate these contributions.",
                           "Three suite seeds produce coarse bootstrap intervals; failures and unknown costs remain explicit.",
                           "Only the first seed retains up to three v2 and three v3 case traces, selected after evaluation by prediction and risk loss.",
                           "Compute units and learning wall time remain separate; final counterfactual evaluation has its own cost and never updates learning.",
                       ]}
        completed = {(run["config"]["seed"], run["system"]): run for run in payload["runs"]}
        for seed in seeds:
            for system in systems:
                if (seed, system) in completed:
                    continue
                concept_id = None
                if system == "erase_specific_concept":
                    concept_id = acquired_concept(completed[(seed, "cognitive_v3")])
                    if concept_id is None:
                        raise RuntimeError("specific deletion has no acquired concept; no unchanged run is relabeled as a deletion")
                print(json.dumps({"event": "start", "partition": partition, "seed": seed, "system": system}), flush=True)
                run = run_system(system, replace(base, seed=seed, partition=partition), remove_concept_id=concept_id)
                if seed == seeds[0] and system in {"reference_v2", "cognitive_v3"}:
                    payload["case_traces"].extend(selected_cases(run))
                compact = compact_run(run)
                payload["runs"].append(compact)
                completed[(seed, system)] = compact
                write_json(artifact, payload)
                summary = partition_summary(payload)
                write_json(summary_path, summary)
                print(json.dumps({"event": "complete", "partition": partition, "seed": seed, "system": system,
                                  "completed_tasks": run["completed_tasks"], "task_count": run["task_count"],
                                  "loss": run["metrics_by_loss"], "wall_seconds": run["execution_wall_time_seconds"]}), flush=True)
        all_summaries[partition] = partition_summary(payload)
    write_json(args.output_dir / "summary.json", all_summaries)
    print(json.dumps({"summary": str(args.output_dir / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
