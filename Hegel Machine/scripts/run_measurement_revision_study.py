#!/usr/bin/env python3
"""Revisit the eight-task measurement gap, then score only stopped learners.

The known cohort reuses the previously paid ordinary-code comparison. The
fresh cohort uses eight predefined new instances and has no new code-agent arm.
Run --phase learn followed by --phase evaluate; neither phase calls a model.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_session import CodeComparisonSession
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab


FRESH_TASKS = [{"task_index": index % 4, "seed": 127003 + 10007 * index}
               for index in range(8)]
COSTS = ("observation_cost", "measurement_cost", "compute_cost")
DEFAULT_BASELINE = ROOT / "artifacts/code_session_v026_completion_extension_20260909.json"


def write_json(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=False, allow_nan=True)


def protocol_for(baseline, baseline_path, cohorts):
    return {
        "experiment": "measurement_revision_known_and_fresh_instances",
        "baseline": str(baseline_path.resolve()),
        "cohorts": {name: baseline["code"]["tasks"] if name == "known" else FRESH_TASKS
                    for name in cohorts},
        "hegel_config": baseline["hegel_config"],
        "code_config_for_resource_limits": baseline["code"]["config"],
        "totals": baseline["code"]["totals"],
        "backend": baseline["code"]["backend"],
        "evaluation": {"seed_offset": 1000003, "length": 64, "trajectories": 4},
        "learning": "Each cohort starts with an empty Hegel library; successors reuse its paid knowledge.",
        "comparison": "Known instances reuse the completed code extension and old Hegel scores; fresh instances have no code or old Hegel arm.",
        "scope": "Known tasks are development comparisons. Fresh seeds test the same four apparatus families, not open-world generality.",
        "accounting": "Model calls and reported tokens are zero in this study. Historical Python scored rows and Hegel search compute are different resources.",
    }


def learn(protocol, name, directory, max_tasks):
    path = directory / f"{name}_learning.json"
    session = CodeComparisonSession(directory / f"{name}_session",
        tasks=protocol["cohorts"][name],
        config=protocol["code_config_for_resource_limits"],
        hegel_config=protocol["hegel_config"], totals=protocol["totals"],
        backend=protocol["backend"])
    if path.exists():
        saved = json.loads(path.read_text())
        state = session.to_dict()
        state["hegel"] = saved["hegel"]
        session = CodeComparisonSession(directory / f"{name}_session", state=state)

    def checkpoint(payload):
        hegel = payload["hegel"]
        completed = len(hegel["results"]) == len(protocol["cohorts"][name])
        write_json(path, {"cohort": name, "status": "completed" if completed else "paused",
                         "tasks": protocol["cohorts"][name], "hegel": hegel})
        last = hegel["results"][-1] if hegel["results"] else {}
        print(json.dumps({"cohort": name, "completed": len(hegel["results"]),
            "task_index": last.get("task_index"), "selected_channels": last.get("selected_channels"),
            "measurement_events": len(last.get("measurement_events", [])),
            "costs": hegel["costs"]}), flush=True)

    # This public method advances only Hegel; CodeResearchSession.run is never
    # called. No old model, independent score or old knowledge reaches learning.
    checkpoint(session.run_hegel(max_tasks=max_tasks, on_checkpoint=checkpoint))


def task_summary(result, evaluation, old=None, old_evaluation=None):
    episode = result["episode"]
    row = {"task_position": result["task_position"], "task_index": result["task_index"],
        "seed": result["seed"], "status": result["status"],
        "selected_channels": result["selected_channels"],
        "validation_loss": episode["validation_loss"] if episode else None,
        "expression": episode["expression"] if episode else None,
        "costs": result["costs"], "measurement_events": result["measurement_events"],
        "measurement_question": result["measurement_question"], "evaluation": evaluation}
    if old is not None:
        row["previous_hegel"] = {"selected_channels": old["selected_channels"],
            "validation_loss": old["episode"]["validation_loss"],
            "costs": old["costs"], "evaluation": old_evaluation["hegel"],
            "measurement_events": old["measurement_events"]}
        row["previous_code_evaluation"] = old_evaluation["code"]
        old_mse = old_evaluation["hegel"]["mse"]
        row["difference_new_minus_previous_hegel"] = {
            "mse": evaluation["mse"] - old_mse if evaluation["mse"] is not None else None,
            **{key: result["costs"][key] - old["costs"][key] for key in COSTS}}
    return row


def evaluate(protocol, baseline, directory):
    # Require every declared learning sequence to stop before opening any final
    # evaluation rows. Fresh-cohort outcomes therefore cannot steer known tasks.
    learned = {name: json.loads((directory / f"{name}_learning.json").read_text())
               for name in protocol["cohorts"]}
    if any(item["status"] != "completed" for item in learned.values()):
        raise ValueError("Complete all declared learning sequences before final evaluation")
    summaries = {}
    for name, payload in learned.items():
        rows = []
        for result in payload["hegel"]["results"]:
            episode = result["episode"]
            evaluation = {"mse": None, "error": "no_completed_model", "rows": 0}
            if episode is not None:
                with make_lab(result["task_index"], result["seed"], backend=protocol["backend"]) as lab:
                    options = protocol["evaluation"]
                    observations = lab.evaluation_rows(result["selected_channels"],
                        seed=result["seed"] + options["seed_offset"], length=options["length"],
                        trajectories=options["trajectories"])
                evaluation = {"mse": loss(Program.from_dict(episode["program"]), observations,
                                           episode["fallback"], .5),
                    "rows": len(observations), "scoring_compute_cost": len(observations),
                    "seed": result["seed"] + options["seed_offset"]}
            position = result["task_position"]
            old = baseline["hegel"]["results"][position] if name == "known" else None
            old_evaluation = baseline["evaluation"][position] if name == "known" else None
            rows.append(task_summary(result, evaluation, old, old_evaluation))
        valid = [row["evaluation"]["mse"] for row in rows if row["evaluation"]["mse"] is not None]
        summary = {"completed_models": len(valid), "tasks": len(rows),
            "mean_mse": mean(valid) if valid else None, "maximum_mse": max(valid) if valid else None,
            "mean_mse_by_kind": {str(kind): mean(row["evaluation"]["mse"] for row in rows
                if row["task_index"] == kind and row["evaluation"]["mse"] is not None)
                for kind in sorted({row["task_index"] for row in rows if row["evaluation"]["mse"] is not None})},
            "costs": payload["hegel"]["costs"], "model_calls": 0, "reported_tokens": 0,
            "measurement_comparisons": sum(len(row["measurement_events"]) for row in rows),
            "adopted_measurement_changes": sum(event.get("adopted", False) for row in rows
                for event in row["measurement_events"]),
            "evaluation_rows": sum(row["evaluation"]["rows"] for row in rows), "results": rows}
        if name == "known":
            previous_hegel = mean(row["hegel"]["mse"] for row in baseline["evaluation"])
            summary.update(previous_hegel_mean_mse=previous_hegel,
                previous_code_mean_mse=mean(row["code"]["mse"] for row in baseline["evaluation"]),
                previous_hegel_costs=baseline["hegel"]["costs"],
                previous_code_costs=baseline["code"]["costs"],
                difference_new_minus_previous_hegel={
                    "mean_mse": summary["mean_mse"] - previous_hegel if valid else None,
                    **{key: payload["hegel"]["costs"][key] - baseline["hegel"]["costs"][key]
                       for key in COSTS}})
        summaries[name] = summary
    report = {"protocol": protocol, "status": "completed", "cohorts": summaries}
    write_json(directory / "summary.json", report)
    print(json.dumps({name: {key: value for key, value in summary.items() if key != "results"}
                      for name, summary in summaries.items()}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("learn", "evaluate"))
    parser.add_argument("--cohort", choices=("known", "fresh", "both"), default="both")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="Maximum additional tasks per cohort in this invocation")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    cohorts = ("known", "fresh") if args.cohort == "both" else (args.cohort,)
    protocol = protocol_for(baseline, args.baseline, cohorts)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume with the saved study inputs, or choose a new output directory")
    else:
        write_json(protocol_path, protocol)
    if args.phase == "learn":
        for name in cohorts:
            learn(protocol, name, args.output, args.steps)
    else:
        evaluate(protocol, baseline, args.output)


if __name__ == "__main__":
    main()
