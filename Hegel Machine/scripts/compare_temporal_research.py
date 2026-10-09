#!/usr/bin/env python3
"""Compare hierarchical and flat response selection with a shared repertoire."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_flat_research import run_flat_temporal_research
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_research import TEMPORAL_LIMITS, mse, run_temporal_research


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def run(output, *, pairs=30, seed=86106007):
    output.mkdir(parents=True, exist_ok=False)
    runners = {"recursive": run_temporal_research, "flat": run_flat_temporal_research}
    modes = ("finite2", "recurrent")
    plan = {"study": "temporal_matched_repertoire_comparison", "apparatus_pairs": pairs,
            "seed_start": seed, "seed_stride": 1009, "arms": list(runners), "modes": modes,
            "limits": TEMPORAL_LIMITS, "terminal_rows": 256, "mse_threshold": 1e-10,
            "shared_repertoire": ["complete quadratic history expansion", "finite and recurrent instrument fitting",
                "instrument disagreement probes", "quadratic response composition", "fresh response confirmation"],
            "contrast": "Separately confirm an instrument child before composing the response, versus "
                "select whole response candidates on paid held-out trajectories without requiring a unique instrument.",
            "pairing": "Independent apparatus copies, identical source/scope/random-seed probes and confirmation controls.",
            "compute_scope": "Shared instrument-search row-pass cap. NumPy response/history fitting and wall time "
                "are not charged to this cap; wall time is descriptive, not a total-compute comparison.",
            "model_calls": 0, "evidence_kind": "offline_bounded_policy_comparison"}
    write(output / "plan.json", plan)
    trained = []
    for index in range(pairs):
        apparatus_seed = seed + 1009 * index
        for mode in modes:
            for arm, researcher in runners.items():
                case_dir = output / f"{index:02d}_{mode}" / arm
                case_dir.mkdir(parents=True)
                source = TemporalInquiryLab(task_index=0, apparatus_seed=apparatus_seed, instrument_mode=mode)
                target = TemporalInquiryLab(task_index=1, apparatus_seed=apparatus_seed, instrument_mode=mode)
                started = perf_counter()
                inquiry = researcher(source, target, seed=apparatus_seed + 100003)
                wall_seconds = perf_counter() - started
                write(case_dir / "model.json", inquiry.model.to_dict())
                write(case_dir / "research.json", inquiry.resultdict)
                write(case_dir / "paid_records.json", inquiry.records)
                case = {"case_id": f"{index:02d}_{mode}", "apparatus_seed": apparatus_seed,
                        "mode": mode, "arm": arm, "status": inquiry.resultdict["status"],
                        "confirmation_mse": inquiry.resultdict["final_confirmation_mse"],
                        "costs": inquiry.resultdict["costs"],
                        "instrument_compute_cost": inquiry.resultdict["instrument_compute_cost"],
                        "response_fit_calls": inquiry.resultdict["response_fit_calls"],
                        "wall_seconds": wall_seconds}
                trained.append(case)
        print(json.dumps({"apparatus_pair": index + 1, "trained_tasks": len(trained)}), flush=True)
    # Both policies finish the entire fixed cohort before terminal labels exist.
    for case in trained:
        case_dir = output / case["case_id"] / case["arm"]
        model = Program.from_dict(json.loads((case_dir / "model.json").read_text()))
        target = TemporalInquiryLab(task_index=1, apparatus_seed=case["apparatus_seed"], instrument_mode=case["mode"])
        rows = target.evaluation_rows(length=plan["terminal_rows"], seed=case["apparatus_seed"] + 200003)
        error = mse(model, rows)
        case.update(terminal_mse=error, correct=case["status"] == "confirmed" and error <= plan["mse_threshold"],
                    false_confirmation=case["status"] == "confirmed" and error > plan["mse_threshold"])
        write(case_dir / "evaluation.json", case)
    totals = {}
    for arm in runners:
        cases = [case for case in trained if case["arm"] == arm]
        totals[arm] = {"cases": len(cases), "correct": sum(case["correct"] for case in cases),
                      "false_confirmations": sum(case["false_confirmation"] for case in cases),
                      "by_mode": {mode: sum(case["correct"] for case in cases if case["mode"] == mode) for mode in modes},
                      "target_observations": sum(case["costs"]["target"]["observation_cost"] for case in cases),
                      "target_measurement_units": sum(case["costs"]["target"]["measurement_cost"] for case in cases),
                      "instrument_compute_cost": sum(case["instrument_compute_cost"] for case in cases),
                      "response_fit_calls": sum(case["response_fit_calls"] for case in cases),
                      "max_terminal_mse": max(case["terminal_mse"] for case in cases)}
    paired = []
    for recursive, flat in zip(trained[::2], trained[1::2]):
        paired.append({"case_id": recursive["case_id"], "mode": recursive["mode"],
                       "correct_difference_recursive_minus_flat": int(recursive["correct"]) - int(flat["correct"]),
                       "target_rows_recursive_minus_flat": recursive["costs"]["target"]["observation_cost"]
                           - flat["costs"]["target"]["observation_cost"],
                       "meter_units_recursive_minus_flat": recursive["costs"]["target"]["measurement_cost"]
                           - flat["costs"]["target"]["measurement_cost"]})
    summary = {**plan, "totals": totals, "paired": paired, "results": trained}
    write(output / "summary.json", summary)
    print(json.dumps(totals, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=86106007)
    args = parser.parse_args()
    if args.pairs < 1 or args.output.exists():
        parser.error("use a positive pair count and a new output directory")
    result = run(args.output, pairs=args.pairs, seed=args.seed)
    return 0 if all(item["correct"] == item["cases"] for item in result["totals"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
