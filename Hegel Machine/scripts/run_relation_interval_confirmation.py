#!/usr/bin/env python3
"""Run the fixed two-version relation-interval study, preserving task checkpoints."""

import argparse
import importlib
import json
from pathlib import Path
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
OUTPUT = ROOT / "artifacts/relation_interval_confirmation_20260913"


def save(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=True, allow_nan=True)


def load_runtime(protocol, arm):
    source = (ROOT / protocol["arms"][arm]["runtime_source"]).resolve()
    sys.path.insert(0, str(source))
    module = importlib.import_module("hegel_machine.cognition.unified_session")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"loaded {module.__file__}, expected frozen source {source}")
    return module, source


def new_session(module, protocol, seed, state=None, lab_factory=None):
    return module.UnifiedResearchSession(module.ResearchConfig(**protocol["hegel_config"], seed=seed),
        tasks=protocol["tasks_by_seed"][str(seed)], measurement_budget=protocol["measurement_budget"],
        state=state, lab_factory=lab_factory)


def learn(output, protocol, arm, seed, module, source):
    path = output / "runs" / arm / str(seed) / "hegel_checkpoint.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved and len(saved["session"]["results"]) == len(protocol["stages"]):
        print(json.dumps({"arm": arm, "seed": seed, "status": "already_learned"}), flush=True)
        return
    started, prior_seconds = monotonic(), saved.get("learning_wall_seconds", 0.)
    laboratories = []

    def laboratory(item):
        result = module.laboratory_for(item)
        laboratories.append(result)
        return result

    def checkpoint(state):
        payload = {"arm": arm, "seed": seed, "runtime_source": str(source),
            "learning_wall_seconds": prior_seconds + monotonic() - started, "session": state}
        save(path, payload)
        return payload

    session = new_session(module, protocol, seed, saved.get("session"), laboratory)
    while len(session.state["results"]) < len(protocol["stages"]):
        task_started = monotonic()
        state = session.run(max_tasks=1, on_checkpoint=checkpoint)
        state["results"][-1]["actual_costs"] = laboratories[-1].costs
        state["results"][-1]["learning_wall_seconds"] = monotonic() - task_started
        checkpoint(state)
        print(json.dumps({"arm": arm, "seed": seed, "tasks_completed": len(state["results"])}), flush=True)
        session = new_session(module, protocol, seed, json.loads(json.dumps(state)), laboratory)


def evaluate(output, protocol, arm, seed, module):
    reports = {name: json.loads((output / "runs" / name / str(seed) / "hegel_checkpoint.json").read_text())
               for name in protocol["arms"]}
    if any(len(report["session"]["results"]) != len(protocol["stages"]) for report in reports.values()):
        raise ValueError("both complete learning sequences must stop before terminal evaluation")
    destination = output / "runs" / arm / str(seed) / "evaluation_report.json"
    if destination.exists():
        print(json.dumps({"arm": arm, "seed": seed, "status": "already_evaluated"}), flush=True)
        return
    report = reports[arm]
    session = new_session(module, protocol, seed, report["session"])
    session.evaluate(**{key: protocol["terminal"][key] for key in ("length", "trajectories", "seed_offset")})
    report["session"] = session.to_dict()
    save(destination, report)
    print(json.dumps({"arm": arm, "seed": seed, "status": "evaluated"}), flush=True)


def summarize(output, protocol):
    import summarize_dynamic_research_replication as summary

    summary.PAIRS = (("hegel_v0802", "hegel_v0801"),)
    for seed in protocol["paired_seeds"]:
        paths = {arm: output / "runs" / arm / str(seed) / "evaluation_report.json"
                 for arm in protocol["arms"]}
        if all(path.exists() for path in paths.values()):
            save(output / "comparisons" / f"{seed}.json", {"seed": seed, "status": "evaluated",
                "arms": {arm: json.loads(path.read_text()) for arm, path in paths.items()}})
    reports = summary.read_reports(output, protocol)
    result = summary.summarize(protocol, reports)
    mechanism = {arm: [] for arm in protocol["arms"]}
    for seed, arms in reports.items():
        for arm, report in arms.items():
            entry = next((item for item in report["session"]["results"] if item["task_position"] == 4), None)
            if entry is None:
                continue
            research = entry["research"]
            actions = research["actions"]
            updates = [{key: action.get(key) for key in ("evidence_rows", "before_intervals", "after_intervals",
                       "candidate_count", "compute_cost")} for action in actions
                       if action["action"] == "refine_relation_parameter_space"]
            intervals = [step["radius_interval"] for step in
                (research.get("research_program") or {}).get("selection_steps", [])
                if step["operation"] == "revise_nested_relation" and step.get("radius_interval")]
            mechanism[arm].append({"seed": seed, "research_status": research["status"],
                "confirmation_loss": research.get("confirmation_loss"),
                "condition_acquisitions": sum(action["action"] == "acquire" and
                    action.get("purpose") == "distinguish_program_conditions" for action in actions),
                "directed_program_experiments": sum(action["action"] == "test_competing_program_predictions"
                    for action in actions),
                "interval_refinements": len(updates), "refinement_history": updates,
                "final_intervals": intervals or None,
                "final_interval_widths": [high - low for low, high in intervals] if intervals else None})
    result["relation_revision_mechanism"] = mechanism
    # The two Hegel versions share a search-operation unit, so its paired change
    # is meaningful here in addition to the shared observation/measurement costs.
    a, b = summary.PAIRS[0]
    result["paired"][f"{a}_minus_{b}"]["compute_cost"] = summary._deltas(
        result["arms"][a]["sequences"][seed]["recorded_learning_costs"]["compute_cost"] -
        result["arms"][b]["sequences"][seed]["recorded_learning_costs"]["compute_cost"]
        for seed in protocol["paired_seeds"]
        if all(result["arms"][arm]["sequences"][seed]["costed_tasks"] == len(protocol["stages"])
               for arm in (a, b)))
    save(output / "summary.json", result)
    print(json.dumps({arm: {key: values[key] for key in ("solved", "failed", "pending_terminal",
        "fully_solved_sequences", "recorded_session_costs", "recorded_queries")}
        for arm, values in result["arms"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "summarize"))
    parser.add_argument("--arm", choices=("hegel_v0801", "hegel_v0802"))
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
            learn(args.output, protocol, args.arm, seed, module, source)
        else:
            evaluate(args.output, protocol, args.arm, seed, module)


if __name__ == "__main__":
    main()
