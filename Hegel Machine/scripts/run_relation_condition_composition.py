#!/usr/bin/env python3
"""Compare composed relation conditions with the common public object laboratory."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import run_directed_relation_composition as study


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/relation_condition_composition_20260914"
PAIR = ("hegel_v083", "hegel_v082")


def relation_test(action):
    return {**{key: action[key] for key in ("relation", "previous_relation", "direction",
        "predicate", "previous_predicate", "observation_rows") if key in action},
        "evidence": study.compact_assessment(action["evidence"]),
        "plan": {key: action["plan"][key] for key in
            ("forecast_status", "proposal_source", "compute_cost") if key in action["plan"]}}


def mechanism(seed, position, research, entry, protocol):
    actions = research["actions"]
    steps = (research.get("research_program") or {}).get("selection_steps", [])
    condition_tests = [action for action in actions if action["action"] == "test_relation_condition"]
    return {"seed": seed, "task_position": position, "stage": protocol["stages"][position],
        "research_status": research["status"], "costs": research["costs"],
        "actual_costs": entry.get("actual_costs"), "learning_wall_seconds": entry.get("learning_wall_seconds"),
        "used_sources": research.get("used_sources", []),
        "method_source_ids": research.get("method_source_ids", []),
        "input_preservation_actions": [action for action in actions
            if action["action"] == "preserve_revisable_input_dependencies"],
        "serialized_program_nodes": study.serialized_program_nodes(research.get("program")),
        "action_counts": dict(Counter(action["action"] for action in actions)),
        "acquisition_purposes": dict(Counter(action.get("purpose", "unspecified")
            for action in actions if action["action"] == "acquire")),
        "condition_paid_acquisitions": sum(action["action"] == "acquire" and
            action.get("purpose") == "distinguish_relation_conditions" for action in actions),
        "condition_test_observation_rows": sum(action["observation_rows"] for action in condition_tests),
        "relation_condition_tests": [relation_test(action) for action in condition_tests],
        "relation_type_tests": [relation_test(action) for action in actions
            if action["action"] == "test_relation_type"],
        "inherited_interval_actions": [action for action in actions if action["action"] in
            ("inherit_relation_intervals", "condition_inherited_relation_intervals")],
        "inherited_probe_results": [{"probe_id": action["probe_id"],
            **study.compact_assessment(action["evidence"])} for action in actions
            if action["action"] == "test_inherited_relation_predictions"],
        "saved_relation_steps": [{**{key: step[key] for key in ("operation", "relation",
            "previous_relation", "predicate", "previous_predicate", "condition_questions",
            "compact_replacement", "radius", "radius_interval", "parameters", "parameter_index")
            if key in step}, "has_parameter_template": "parameter_template" in step,
            "parameter_template_nodes": study.serialized_program_nodes(step.get("parameter_template"))}
            for step in steps if step["operation"] in ("revise_relation_condition", "revise_relation_type",
                "resolve_relation_intervals", "revise_nested_relation")]}


def summarize(output, protocol):
    paired = all(arm in protocol["arms"] for arm in PAIR)
    study.inherited.PAIR = PAIR
    study.inherited.summary.PAIRS = (PAIR,) if paired else ()
    reports, mechanisms = {}, {arm: [] for arm in protocol["arms"]}
    saved_inputs = {arm: {} for arm in protocol["arms"]}
    for seed in protocol["paired_seeds"]:
        reports[seed], paths = {}, {}
        for arm in protocol["arms"]:
            terminal = output / "runs" / arm / str(seed) / "evaluation_report.json"
            path = terminal if terminal.exists() else terminal.with_name("hegel_checkpoint.json")
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            reports[seed][arm] = report
            saved_inputs[arm][seed] = [{"source_concept_id": prior["source_concept_id"],
                "uses": prior.get("uses", 0), "method_uses": prior.get("method_uses", 0),
                "required_fields": prior["input_program"].get("required_fields", []),
                "roundtrip_loss": prior["input_program"].get("roundtrip_loss"),
                "support_rows": prior["input_program"].get("support_rows"),
                "definition_id": prior["input_program"].get("definition", {}).get("definition_id"),
                "input_research_program_preserved": bool(prior.get("input_research_program"))}
                for prior in report["session"].get("knowledge", {}).get("meta_priors", [])
                if prior.get("input_program")]
            if terminal.exists():
                paths[arm] = str(terminal.relative_to(output))
            for entry in report["session"]["results"]:
                mechanisms[arm].append(mechanism(seed, entry["task_position"], entry["research"], entry, protocol))
        if len(paths) == len(protocol["arms"]):
            study.inherited.study.save(output / "comparisons" / f"{seed}.json", {
                "seed": seed, "status": "evaluated", "arm_report_paths": paths})
    result = study.inherited.summary.summarize(protocol, reports)
    result["stage_mechanisms"] = mechanisms
    result["saved_input_programs"] = saved_inputs
    if paired:
        study.inherited.add_work_comparison(result, reports, protocol)
        for position in range(len(protocol["stages"])):
            study.inherited.add_work_comparison(result, reports, protocol, position)
    else:
        for arm in protocol["arms"]:
            durations = [report[arm]["learning_wall_seconds"] for report in reports.values() if arm in report]
            result["arms"][arm]["recorded_learning_wall_seconds"] = sum(durations) if durations else None
            for stage in protocol["stages"]:
                durations = [row["learning_wall_seconds"] for row in mechanisms[arm]
                             if row["stage"] == stage and row["learning_wall_seconds"] is not None]
                result["stages"][stage]["arms"][arm]["recorded_learning_wall_seconds"] = sum(durations) if durations else None
    study.inherited.study.save(output / "summary.json", result)
    print(json.dumps({arm: {key: values[key] for key in ("solved", "failed", "pending_terminal",
        "fully_solved_sequences", "recorded_session_costs", "recorded_queries")}
        for arm, values in result["arms"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "summarize", "development"))
    parser.add_argument("--arm", choices=PAIR)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (OUTPUT / "development_v1" if args.mode == "development" else OUTPUT)
    protocol = json.loads((output / "protocol.json").read_text())
    if args.mode == "summarize":
        summarize(output, protocol)
        return
    arm = "hegel_v083" if args.mode == "development" else args.arm
    seeds = protocol["paired_seeds"] if args.mode == "development" else args.seeds
    if arm is None or not seeds or any(seed not in protocol["paired_seeds"] for seed in seeds):
        parser.error("learning/evaluation needs an arm and seeds declared in the selected protocol")
    module, source = study.load_runtime(protocol, arm)
    if args.mode == "development":
        import hegel_machine

        study.inherited.study.save(output / "runtime_info.json", {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "package_version": hegel_machine.__version__,
            "implementation_target": "0.83 condition composition development",
            "formal_release_snapshot": False,
            "common_laboratory": protocol["shared_laboratory_source"]})
    for seed in seeds:
        if args.mode in ("learn", "development"):
            study.inherited.study.learn(output, protocol, arm, seed, module, source)
        else:
            study.inherited.study.evaluate(output, protocol, arm, seed, module)
    if args.mode == "development":
        # Both fixed development sequences finish learning before terminal
        # measurements; each task was checkpointed and its full library restored.
        for seed in seeds:
            study.inherited.study.evaluate(output, protocol, arm, seed, module)
        summarize(output, protocol)


if __name__ == "__main__":
    main()
