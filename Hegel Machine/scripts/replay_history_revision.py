#!/usr/bin/env python3
"""Develop a history repair on frozen paid evidence, without new LLM calls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.history_revision import revise_history
from scripts.run_online_inquiry_llm import tasks_for
from scripts.run_temporal_inquiry_llm import laboratory


def replay(run, output):
    run, output = Path(run), Path(output)
    plan = json.loads((run / "protocol.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("finish the original training run before development replay")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted(run.glob("*/*/result.json")):
        chain = json.loads(path.read_text())
        spec = next(case for case in plan["cases"] if case["chain_id"] == chain["chain_id"])
        for task, result in zip(tasks_for(spec)[1:], chain["tasks"][1:]):
            rows = json.loads((Path(result["workspace"]) / "model_observations.json").read_text())
            trajectories = sum(not row["history"] for row in rows)
            if trajectories < 3:
                raise ValueError("replay requires training trajectories and two complete validation trajectories")
            # Same development rule in both worlds. These are already purchased
            # rows, including failed confirmation where present, not fresh evidence.
            proposal = revise_history(rows, base_lags=2, holdout_fraction=2 / trajectories)
            workspace = output / chain["chain_id"] / chain["arm"]
            workspace.mkdir(parents=True)
            (workspace / "model.py").write_text(proposal.to_code())
            (workspace / "model_observations.json").write_text(json.dumps(rows))
            # Freeze the exported predictor before drawing a new diagnostic set.
            seed = task["evaluation_seed"] + 10000019
            independent = laboratory(task).evaluation_rows(length=256, seed=seed)
            score = evaluate_python_model(workspace, "model.py", independent, timeout_seconds=120)
            score.pop("predictions", None)
            record = {"chain_id": chain["chain_id"], "arm": chain["arm"],
                "original_status": result["status"], "paid_rows": len(rows),
                "proposal": proposal.resultdict, "new_evaluation_seed": seed,
                "independent_diagnostic": score,
                "independent_correct": score.get("returncode") == 0 and score.get("mse", float("inf")) <= 1e-10}
            (workspace / "revision.json").write_text(json.dumps(record, indent=2) + "\n")
            cases.append(record)
    report = {"evidence_kind": "developer_authored_operator_on_frozen_paid_evidence",
        "source_run": str(run.resolve()), "new_model_calls": 0, "new_training_observations": 0,
        "registered_llm_results_changed": False,
        "selection": "Extend two history coordinates to three while retaining every quadratic interaction; select on the last two complete paid trajectories, then refit on all paid rows.",
        "scope": "A bounded repair procedure; not autonomous LLM invention or a same-model performance result.",
        "cases": cases}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.run, args.output)
    print(json.dumps([{key: case[key] for key in ("chain_id", "independent_correct")}
        | {"status": case["proposal"]["status"], "mse": case["independent_diagnostic"].get("mse")}
        for case in result["cases"]], indent=2))


if __name__ == "__main__":
    main()
