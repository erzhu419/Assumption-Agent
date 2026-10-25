#!/usr/bin/env python3
"""Compare the saved v3 algorithm, live v4 and bounded local ablations.

Examples:
  python3 scripts/run_cognition_v4_research.py --partitions development confirmation forward
  python3 scripts/run_cognition_v4_research.py --partitions continual --output-dir artifacts/cognition_v4_continual

Continual is evaluated last when several partitions are requested. Run it only
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
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "artifacts"))

from cognition_v3_reference import experiments as reference_experiments
from cognition_v3_reference.models import Observation as ReferenceObservation, ResearchConfig as ReferenceConfig
from hegel_machine import __version__ as live_version
from hegel_machine.cognition import experiments as live_experiments
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_protocol import confidence_summary, summarize_research


PARTITIONS = ("development", "confirmation", "forward", "continual")
DEFAULT_SEEDS = {"development": (7, 11, 19), "confirmation": (23, 47, 83),
                 "forward": (101, 211, 307), "continual": (401, 503, 607)}
SYSTEMS = ("reference_v3", "cognitive_v4", "greedy_joint_search", "short_horizon",
           "without_knowledge", "without_risk_learning")
SYSTEM_CHANGES = {
    "cognitive_v4": {},
    "greedy_joint_search": {"joint_search": "greedy"},
    "short_horizon": {"adaptive_horizon": False},
    "without_knowledge": {"use_memory": False},
    "without_risk_learning": {"risk_learning": False},
}
HORIZON_BINS = (("0-7", 0, 8), ("8-31", 8, 32), ("32+", 32, None))


class _ReferenceEnvironment:
    """Present continual observations using the saved algorithm's value type."""

    def __init__(self, environment):
        self._environment = environment
        self.spec = environment.spec
        self.evaluation_metadata = environment.evaluation_metadata

    @staticmethod
    def _row(row):
        return ReferenceObservation(dict(row.features), row.target,
                                    tuple(dict(previous) for previous in row.history))

    def observe(self, probe):
        return [self._row(row) for row in self._environment.observe(probe)]

    def evaluation_rows(self):
        return tuple(self._row(row) for row in self._environment.evaluation_rows())


def run_system(system, config):
    """Keep the old label adaptation entirely in the experiment script."""
    started = perf_counter()
    if system == "reference_v3":
        shared = {field.name: getattr(config, field.name) for field in fields(ReferenceConfig)}
        actual_partition = config.partition
        if actual_partition == "continual":
            shared["partition"] = "forward"
        reference_config = ReferenceConfig(**shared)
        original_suite = reference_experiments.make_suite
        try:
            if actual_partition == "continual":
                reference_experiments.make_suite = lambda seed, partition: [
                    _ReferenceEnvironment(environment) for environment in make_suite(seed, partition="continual")]
            result = reference_experiments.run_sequence(reference_config)
        finally:
            reference_experiments.make_suite = original_suite
        result["reference_config_partition"] = reference_config.partition
        result["partition"] = actual_partition
        result["algorithm_version"] = "0.5.0_saved_cognition_package"
    else:
        changed = replace(config, **SYSTEM_CHANGES[system])
        result = live_experiments.run_sequence(changed)
        result["algorithm_version"] = live_version
    result["system"] = system
    result["horizon_evaluation"] = evaluate_horizons(result, config)
    result["execution_wall_time_seconds"] = perf_counter() - started
    return result


def evaluate_horizons(run, config):
    """Score recorded final programs only after the entire learning run returns."""
    started = perf_counter()
    implementation = reference_experiments if run["system"] == "reference_v3" else live_experiments
    environments = make_suite(config.seed, partition=config.partition)
    groups, units = {}, 0
    for episode, environment in zip(run["episodes"], environments):
        if episode["status"] != "completed":
            continue
        program = implementation.Program.from_dict(episode["program"])
        rows = environment.evaluation_rows()
        buckets = {name: [] for name, _, _ in HORIZON_BINS}
        for row in rows:
            predicted = implementation.predict(program, row, episode["fallback"])
            measured = implementation.point_loss(predicted, row.target, environment.spec.risk_quantile)
            for name, low, high in HORIZON_BINS:
                if low <= len(row.history) and (high is None or len(row.history) < high):
                    buckets[name].append(measured)
                    break
        units += len(rows)
        episode["evaluation"]["history_length_bins"] = {
            name: {"rows": len(values), "mean_test_loss": mean(values) if values else None}
            for name, values in buckets.items()}
        kind = episode["evaluation"]["loss_kind"]
        by_kind = groups.setdefault(kind, {name: [] for name, _, _ in HORIZON_BINS})
        for name, values in buckets.items():
            if values:
                by_kind[name].append((len(values), mean(values)))
    return {"additional_evaluation_units": units, "wall_time_seconds": perf_counter() - started,
            "by_loss": {kind: {name: {"tasks": len(values), "rows": sum(value[0] for value in values),
                                      "mean_test_loss": mean(value[1] for value in values) if values else None}
                               for name, values in buckets.items()} for kind, buckets in groups.items()},
            "definition": "history lengths 0-7, 8-31 and 32+; mean over completed task means, using final predictions after learning"}


def compact_run(run):
    result = {key: value for key, value in run.items() if key not in {"episodes", "knowledge"}}
    result["episodes"] = []
    for episode in run["episodes"]:
        compact = {key: value for key, value in episode.items()
                   if key not in {"actions", "questions", "plans", "program"}}
        compact["action_counts"] = dict(Counter(action["action"] for action in episode.get("actions", [])))
        compact["observed_query_lengths"] = [action["details"]["rows"] for action in episode.get("actions", [])
                                             if action["action"] == "observe"]
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


def partition_summary(payload):
    # Reuse the seed-level statistics while preserving the actual arm names.
    by_seed = {seed: {} for seed in payload["seeds"]}
    for run in payload["runs"]:
        name = "cognitive_agent" if run["system"] == "cognitive_v4" else run["system"]
        by_seed[run["config"]["seed"]][name] = run
    result = summarize_research([{"runs": runs} for runs in by_seed.values()],
                                payload["seeds"], payload["partition"])
    if "cognitive_agent" in result["systems"]:
        result["systems"]["cognitive_v4"] = result["systems"].pop("cognitive_agent")
    for row in result["seed_results"]:
        if "cognitive_agent" in row["systems"]:
            row["systems"]["cognitive_v4"] = row["systems"].pop("cognitive_agent")
    result["paired_loss_alternative_minus_v4"] = result.pop("paired_loss_alternative_minus_cognitive")
    result["inference"]["paired_sign"] = "positive favors cognitive_v4"
    result["experiment"] = "v3_v4_matched_local_research"
    result["completed_runs"] = len(payload["runs"])
    result["planned_runs"] = len(payload["seeds"]) * len(payload["systems"])
    result["algorithm_versions"] = {run["system"]: run["algorithm_version"] for run in payload["runs"]}
    for system, summary in result["systems"].items():
        runs = [run for run in payload["runs"] if run["system"] == system]
        kinds = sorted({kind for run in runs for kind in run["horizon_evaluation"]["by_loss"]})
        summary["horizon_loss"] = {
            kind: {name: confidence_summary([
                run["horizon_evaluation"]["by_loss"][kind][name]["mean_test_loss"]
                for run in runs if kind in run["horizon_evaluation"]["by_loss"]
                and run["horizon_evaluation"]["by_loss"][kind][name]["mean_test_loss"] is not None])
                   for name, _, _ in HORIZON_BINS} for kind in kinds}
        summary["horizon_evaluation_costs"] = {
            key: confidence_summary([run["horizon_evaluation"][key] for run in runs])
            for key in ("additional_evaluation_units", "wall_time_seconds")}
    for seed_result in result["seed_results"]:
        for run in payload["runs"]:
            if run["config"]["seed"] == seed_result["seed"]:
                seed_result["systems"][run["system"]]["horizon_evaluation"] = run["horizon_evaluation"]
    result["design"] = payload["design"]
    result["limitations"] = payload["limitations"]
    return result


def write_json(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=list(PARTITIONS[:-1]))
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Override the three fixed seeds for each requested partition")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "cognition_v4_research")
    parser.add_argument("--observation-budget", type=int, default=96)
    parser.add_argument("--compute-budget", type=int, default=24000)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Retain already completed runs with the same design")
    args = parser.parse_args()
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not repeat")
    if len(set(args.systems)) != len(args.systems):
        parser.error("systems must not repeat")
    systems = [name for name in SYSTEMS if name in args.systems]
    base = ResearchConfig(observation_budget=args.observation_budget, compute_budget=args.compute_budget,
                          max_rounds=args.max_rounds, proposer="search")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for partition in PARTITIONS:
        if partition not in args.partitions:
            continue
        seeds = args.seeds or list(DEFAULT_SEEDS[partition])
        design = {"base_config": asdict(replace(base, partition=partition)),
                  "seeds": seeds, "systems": systems, "reference_version": "0.5.0",
                  "live_version": live_version,
                  "system_changes": {name: SYSTEM_CHANGES[name] for name in systems if name in SYSTEM_CHANGES},
                  "pairing": "same suite seed, initial and catalog probes, final evaluation and budget caps; adaptive queries may differ",
                  "short_horizon": "fixed representative trajectory length at most eight; all arms retain the same longer catalog queries",
                  "continual_horizons": "maximum training probe length 32; eight independent final trajectories of length 64",
                  "reference_continual_adapter": "saved v3 config uses forward label; continual environment values are converted to the saved Observation type"}
        artifact = args.output_dir / f"{partition}_runs.json"
        summary_path = args.output_dir / f"{partition}_summary.json"
        if artifact.exists():
            if not args.resume:
                parser.error(f"{artifact} exists; choose another output directory or use --resume")
            payload = json.loads(artifact.read_text())
            if payload["design"] != design:
                parser.error(f"{artifact} uses a different experiment design")
        else:
            payload = {"experiment": "v3_v4_matched_local_research", "partition": partition,
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "seeds": seeds, "systems": systems, "design": design, "runs": [], "case_traces": [],
                       "limitations": [
                           "Development, confirmation and forward outcomes informed v4 development; all three are retrospective comparisons.",
                           "Continual is a public synthetic prospective partition; using its results to change learning turns it into development evidence.",
                           "Continual evaluates history lengths beyond the maximum acquired trajectory; history bins distinguish startup, trained horizons and temporal extrapolation.",
                           "short_horizon fixes representative experiments at at most eight rows; it does not truncate the shared catalog or prohibit longer targeted queries.",
                           "Three suite seeds produce coarse bootstrap intervals; failures and unknown costs remain explicit.",
                           "Only the first seed retains up to three v3 and three v4 case traces, selected after evaluation by prediction and risk loss.",
                           "Compute units and learning wall time remain separate; final counterfactual evaluation has its own cost and never updates learning.",
                       ]}
        completed = {(run["config"]["seed"], run["system"]): run for run in payload["runs"]}
        for seed in seeds:
            for system in systems:
                if (seed, system) in completed:
                    continue
                print(json.dumps({"event": "start", "partition": partition, "seed": seed, "system": system}), flush=True)
                run = run_system(system, replace(base, seed=seed, partition=partition))
                if seed == seeds[0] and system in {"reference_v3", "cognitive_v4"}:
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
