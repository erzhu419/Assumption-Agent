#!/usr/bin/env python3
"""Compare relation-type revision with one shared directed-transport laboratory."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import run_inherited_relation_intervals as inherited


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/directed_relation_composition_20260914"
PAIR = ("hegel_v082", "hegel_v081")


def serialized_program_nodes(program):
    pending, count = [program], 0
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            count += "op" in value
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return count


def compact_assessment(evidence):
    return {**{key: evidence[key] for key in
        ("status", "observation_rows", "compute_cost", "calibration_status")},
        "hypotheses": [{key: item[key] for key in
            ("hypothesis_id", "status", "measured_loss", "forecast_loss", "rows")}
            for item in evidence["hypotheses"]]}


def summarize(output, protocol):
    inherited.PAIR = PAIR
    inherited.summary.PAIRS = (PAIR,)
    reports, mechanisms = {}, {arm: [] for arm in protocol["arms"]}
    for seed in protocol["paired_seeds"]:
        reports[seed], paths = {}, {}
        for arm in protocol["arms"]:
            terminal = output / "runs" / arm / str(seed) / "evaluation_report.json"
            path = terminal if terminal.exists() else terminal.with_name("hegel_checkpoint.json")
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            reports[seed][arm] = report
            if terminal.exists():
                paths[arm] = str(terminal.relative_to(output))
            for entry in report["session"]["results"]:
                research, position = entry["research"], entry["task_position"]
                actions = research["actions"]
                steps = (research.get("research_program") or {}).get("selection_steps", [])
                mechanisms[arm].append({
                    "seed": seed, "task_position": position, "stage": protocol["stages"][position],
                    "research_status": research["status"], "used_sources": research.get("used_sources", []),
                    "method_source_ids": research.get("method_source_ids", []),
                    "serialized_program_nodes": serialized_program_nodes(research.get("program")),
                    "learning_wall_seconds": entry.get("learning_wall_seconds"),
                    "action_counts": dict(Counter(action["action"] for action in actions)),
                    "acquisition_purposes": dict(Counter(action.get("purpose", "unspecified")
                        for action in actions if action["action"] == "acquire")),
                    "relation_type_tests": [{**{key: action[key] for key in
                        ("relation", "previous_relation", "direction", "observation_rows")},
                        "evidence": compact_assessment(action["evidence"]),
                        "plan": {key: action["plan"][key] for key in
                            ("forecast_status", "proposal_source", "compute_cost") if key in action["plan"]}}
                        for action in actions if action["action"] == "test_relation_type"],
                    "inherited_interval_actions": [action for action in actions if action["action"] in
                        ("inherit_relation_intervals", "condition_inherited_relation_intervals")],
                    "inherited_probe_results": [{"probe_id": action["probe_id"],
                        **compact_assessment(action["evidence"])} for action in actions
                        if action["action"] == "test_inherited_relation_predictions"],
                    "saved_relation_steps": [{**{key: step[key] for key in
                        ("operation", "relation", "previous_relation", "radius", "radius_interval",
                         "parameters", "parameter_index") if key in step},
                        "has_parameter_template": "parameter_template" in step} for step in steps
                        if step["operation"] in ("revise_relation_type", "resolve_relation_intervals",
                                                  "revise_nested_relation")]})
        if len(paths) == len(protocol["arms"]):
            inherited.study.save(output / "comparisons" / f"{seed}.json", {
                "seed": seed, "status": "evaluated", "arm_report_paths": paths})
    result = inherited.summary.summarize(protocol, reports)
    result["stage_mechanisms"] = mechanisms
    inherited.add_work_comparison(result, reports, protocol)
    for position in range(len(protocol["stages"])):
        inherited.add_work_comparison(result, reports, protocol, position)
    inherited.study.save(output / "summary.json", result)
    print(json.dumps({arm: {key: values[key] for key in ("solved", "failed", "pending_terminal",
        "fully_solved_sequences", "recorded_session_costs", "recorded_queries")}
        for arm, values in result["arms"].items()}, indent=2))


def load_runtime(protocol, arm):
    module, source = inherited.study.load_runtime(protocol, arm)
    directory = ROOT / protocol["shared_laboratory_source"]
    for name in ("structure_lab", "unified_object_lab"):
        canonical = f"hegel_machine.cognition.{name}"
        spec = importlib.util.spec_from_file_location(canonical, directory / f"{name}.py")
        laboratory_module = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = laboratory_module
        spec.loader.exec_module(laboratory_module)

    def laboratory_for(item):
        arguments = dict(item)
        if arguments.pop("environment") != "objects":
            raise ValueError("this protocol uses the public object-schema laboratory")
        return laboratory_module.UnifiedObjectLab(**arguments)

    # Both learning's recorded-acquisition factory and evaluation's default
    # factory resolve here; no learner module is replaced on disk.
    module.laboratory_for = laboratory_for
    return module, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "summarize"))
    parser.add_argument("--arm", choices=PAIR)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    if args.mode == "summarize":
        summarize(args.output, protocol)
        return
    if args.arm is None or not args.seeds or any(seed not in protocol["paired_seeds"] for seed in args.seeds):
        parser.error("learning/evaluation needs an arm and seeds declared in the fixed protocol")
    module, source = load_runtime(protocol, args.arm)
    for seed in args.seeds:
        if args.mode == "learn":
            inherited.study.learn(args.output, protocol, args.arm, seed, module, source)
        else:
            inherited.study.evaluate(args.output, protocol, args.arm, seed, module)


if __name__ == "__main__":
    main()
