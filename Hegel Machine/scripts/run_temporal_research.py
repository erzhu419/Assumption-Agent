#!/usr/bin/env python3
"""Run a prospective offline temporal research cohort; no LLM/API calls."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_research import fit_history_program, mse, run_temporal_research


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output, pairs=12, seed=85105007):
    output.mkdir(parents=True, exist_ok=False)
    plan = {"study": "prospective_temporal_research", "apparatus_pairs": pairs,
            "apparatus_seed_start": seed, "seed_stride": 1009,
            "modes": ["finite2", "recurrent"], "terminal_rows": 256,
            "mse_threshold": 1e-10, "model_calls": 0,
            "procedure": "short history; one-coordinate quadratic revision; "
                "if unresolved, measured instrument child and composed quadratic response; fresh confirmation",
            "evidence_kind": "prospective_offline_bounded_controller",
            "controls": ["frozen_source", "history_revision", "history_same_paid_rows"],
            "control_scope": "Mechanism diagnostics. Same-paid-row control refits a fixed three-lag "
                "quadratic on all preconfirmation responses, with no instrument representation. "
                "These are not matched-budget flat/LLM agents."}
    write(output / "plan.json", plan)
    results = []
    for index in range(pairs):
        apparatus_seed = seed + 1009 * index
        for mode in plan["modes"]:
            started = perf_counter()
            case_id = f"{index:02d}_{mode}"
            case_dir = output / case_id
            case_dir.mkdir()
            source = TemporalInquiryLab(task_index=0, apparatus_seed=apparatus_seed, instrument_mode=mode)
            target = TemporalInquiryLab(task_index=1, apparatus_seed=apparatus_seed, instrument_mode=mode)
            inquiry = run_temporal_research(source, target, seed=apparatus_seed + 100003)
            paid_rows = [row for record in inquiry.records for row in record["rows"]]
            training_rows = paid_rows[:inquiry.resultdict.get("fit_rows", len(paid_rows))]
            controls = {"frozen_source": inquiry.source_model, "history_revision": inquiry.history_model,
                        "history_same_paid_rows": fit_history_program(training_rows, 3)}
            programs = {"research": inquiry.model, **controls}
            # All programs are fixed and saved before independently generated labels.
            write(case_dir / "models.json", {name: model.to_dict() for name, model in programs.items()})
            write(case_dir / "research.json", inquiry.resultdict)
            write(case_dir / "paid_records.json", inquiry.records)
            terminal = target.evaluation_rows(length=plan["terminal_rows"], seed=apparatus_seed + 200003)
            scores = {name: mse(model, terminal) for name, model in programs.items()}
            result = {"case_id": case_id, "apparatus_seed": apparatus_seed, "mode": mode,
                      "status": inquiry.resultdict["status"],
                      "confirmation_mse": inquiry.resultdict["final_confirmation_mse"],
                      "terminal_mse": scores,
                      "child_opened": any(event["event"] == "child_opened" for event in inquiry.resultdict["events"]),
                      "correct": inquiry.resultdict["status"] == "confirmed" and scores["research"] <= 1e-10,
                      "costs": inquiry.resultdict["costs"],
                      "instrument_compute_cost": inquiry.resultdict["instrument_compute_cost"],
                      "wall_seconds": perf_counter() - started}
            results.append(result)
            write(case_dir / "evaluation.json", result)
            print(json.dumps(result, separators=(",", ":")), flush=True)
    summary = {**plan, "case_count": len(results), "results": results,
               "correct": sum(item["correct"] for item in results),
               "by_mode": {mode: {"cases": pairs,
                   "confirmed_and_correct": sum(item["correct"] for item in results if item["mode"] == mode),
                   "child_opened": sum(item["child_opened"] for item in results if item["mode"] == mode),
                   "prediction_correct": {name: sum(item["terminal_mse"][name] <= 1e-10
                       for item in results if item["mode"] == mode) for name in programs}}
                   for mode in plan["modes"]}}
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=85105007)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.output.exists():
        parser.error("output must be new")
    result = run(args.output, args.pairs, args.seed)
    print(json.dumps({key: result[key] for key in ("case_count", "correct", "by_mode")}, indent=2))
    return 0 if result["correct"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
