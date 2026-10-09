#!/usr/bin/env python3
"""Offline identifiability and known-form recoverability; no agent/model calls.

The reference fitter is deliberately told the one-pole/three-tap instrument
families and quadratic response form. This is a feasibility upper bound, not a
baseline supplied to an agent and not evidence that recursive research wins.
Only constructors receive physical family labels; fitting uses paid records.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


TOLERANCE = 1e-10
MODES = ("finite2", "recurrent")


def purchase(lab, controls):
    return lab.execute({"kind": "trajectory",
                        "sequence": [{"u": value} for value in controls],
                        "channels": ["meter"]})


def readings(record):
    by_index = {item["observation_index"]: item["values"]["meter"]
                for item in record["measurements"]}
    return np.asarray([by_index[index] for index in record["row_indices"]])


def response(record):
    return np.asarray([row["target"] for row in record["rows"]])


@dataclass
class ReferenceModel:
    impulse: tuple[float, ...]
    decay: float | None
    coefficients: np.ndarray

    def meter(self, controls):
        if not controls:
            return 0.0
        if self.decay is None:
            return sum(value * controls[-lag - 1]
                       for lag, value in enumerate(self.impulse)
                       if lag < len(controls))
        return (self.impulse[0] * controls[-1]
                + self.impulse[1] * sum(self.decay ** (lag - 1) * controls[-lag - 1]
                                        for lag in range(1, len(controls))))

    def predict(self, row):
        # Accept only controls, even though the scorer also holds target labels.
        controls = [point["u"] for point in row["history"]]
        previous = self.meter(controls)
        controls.append(row["features"]["u"])
        current = self.meter(controls)
        return float(self.coefficients @ [1.0, controls[-1], current * current, previous])


def fit_reference(impulse_record, records, *, short_scope=False):
    """Infer instrument parameters and regress response from purchased data."""
    impulse = readings(impulse_record)
    recurrent = not short_scope and bool(np.max(np.abs(impulse[3:])) > TOLERANCE)
    decay = float(impulse[2] / impulse[1]) if recurrent else None
    matrix, targets = [], []
    for record in records:
        previous = 0.0
        for row, current in zip(record["rows"], readings(record)):
            matrix.append([1.0, row["features"]["u"], current * current, previous])
            targets.append(row["target"])
            previous = current
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, targets, rcond=None)
    if rank != 4:
        raise ValueError("paid response design does not identify the four reference coefficients")
    return ReferenceModel(tuple(float(value) for value in impulse[:3]), decay, coefficients)


def mse(model, rows):
    return float(np.mean([(model.predict(row) - row["target"]) ** 2 for row in rows]))


def controls_only(rows):
    return all(set(row["features"]) == {"u"}
               and all(set(point) == {"u"} for point in row["history"])
               for row in rows)


def inspect_apparatus(seed):
    # Both worlds use exactly the same apparatus draw and differ only in family.
    source = {mode: TemporalInquiryLab(task_index=0, apparatus_seed=seed,
                                       instrument_mode=mode) for mode in MODES}
    target = {mode: TemporalInquiryLab(task_index=1, apparatus_seed=seed,
                                       instrument_mode=mode) for mode in MODES}
    rng = Random(seed + 70001)
    source_sequences = [[1.0, 0.0, 0.0]] + [
        [rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(8)]
    source_records = {mode: [purchase(source[mode], sequence) for sequence in source_sequences]
                      for mode in MODES}
    source_gap = max(float(np.max(np.abs(readings(left) - readings(right))))
                     for left, right in zip(source_records[MODES[0]], source_records[MODES[1]]))
    source_response_gap = max(float(np.max(np.abs(response(left) - response(right))))
                              for left, right in zip(source_records[MODES[0]], source_records[MODES[1]]))
    long_impulse = {mode: purchase(target[mode], [1.0] + [0.0] * 23) for mode in MODES}
    meter_difference = readings(long_impulse[MODES[0]]) - readings(long_impulse[MODES[1]])
    response_difference = response(long_impulse[MODES[0]]) - response(long_impulse[MODES[1]])
    long_random = [rng.uniform(-1.0, 1.0) for _ in range(24)]
    checks = {
        "public_interfaces_equal": source[MODES[0]].public_spec() == source[MODES[1]].public_spec()
            and target[MODES[0]].public_spec() == target[MODES[1]].public_spec(),
        "three_impulse_coefficients_equal": bool(np.max(np.abs(meter_difference[:3])) <= TOLERANCE),
        "sampled_short_sources_equal": source_gap <= TOLERANCE and source_response_gap <= TOLERANCE,
        "fourth_impulse_point_distinguishes": abs(float(meter_difference[3])) > TOLERANCE
            and abs(float(response_difference[3])) > TOLERANCE,
    }
    cases = []
    for mode in MODES:
        short_model = fit_reference(source_records[mode][0], source_records[mode], short_scope=True)
        target_records = [long_impulse[mode], purchase(target[mode], long_random)]
        long_model = fit_reference(target_records[0], target_records)
        source_rows = [asdict(row) for row in source[mode].evaluation_rows(length=48, seed=seed + 90001)]
        target_rows = [asdict(row) for row in target[mode].evaluation_rows(length=96, seed=seed + 190001)]
        source_train_rows = [row for record in source_records[mode] for row in record["rows"]]
        source_loss = mse(short_model, source_rows)
        shortcut_loss = mse(short_model, target_rows)
        recovered_loss = mse(long_model, target_rows)
        inferred_mode = "recurrent" if long_model.decay is not None else "finite2"
        case_checks = {
            "short_fit_valid_in_source_scope": source_loss <= TOLERANCE,
            "long_reference_recovers": recovered_loss <= TOLERANCE and inferred_mode == mode,
            "shortcut_has_expected_scope": ((shortcut_loss <= TOLERANCE) if mode == "finite2"
                                             else (shortcut_loss > TOLERANCE)),
            "evaluation_inputs_controls_only": controls_only(source_rows) and controls_only(target_rows),
            "paid_costs_match": (source[mode].costs["observation_cost"] == 27
                                  and source[mode].costs["measurement_cost"] == 81
                                  and target[mode].costs["observation_cost"] == 48
                                  and target[mode].costs["measurement_cost"] == 144),
        }
        cases.append({"apparatus_seed": seed, "instrument_mode": mode,
                      "source_training_mse": mse(short_model, source_train_rows),
                      "source_heldout_mse": source_loss,
                      "frozen_short_model_long_mse": shortcut_loss,
                      "long_reference_heldout_mse": recovered_loss,
                      "inferred_mode": inferred_mode,
                      "source_costs": dict(source[mode].costs),
                      "target_recovery_costs": dict(target[mode].costs),
                      "checks": case_checks})
    return {"apparatus_seed": seed, "short_meter_max_difference": source_gap,
            "short_response_max_difference": source_response_gap,
            "fourth_meter_difference": abs(float(meter_difference[3])),
            "fourth_response_difference": abs(float(response_difference[3])),
            "checks": checks}, cases


def run(cases=12, seed=84051001):
    pairs, results = [], []
    for index in range(cases):
        pair, modes = inspect_apparatus(seed + 1009 * index)
        pairs.append(pair)
        results.extend(modes)
    passed = all(all(item["checks"].values()) for item in [*pairs, *results])
    return {
        "experiment": "temporal_inquiry_offline_design", "passed": passed,
        "evidence_kind": "known_form_reference_upper_bound_not_agent_performance",
        "scope": "Same physical apparatus, source horizon 3 then target horizon 24.",
        "short_equivalence_basis": "The first three impulse coefficients determine every length<=3 "
            "meter trajectory in these linear reset instruments; equal readings and previous "
            "readings also give equal responses. Random short trajectories check the adapter.",
        "reference_prior": "One-pole or three-tap instrument; response linear in 1,u,meter^2,previous_meter. "
            "All coefficients and the long-scope family are fitted from paid records.",
        "apparatus_count": cases, "case_count": len(results),
        "source_heldout_rows_per_case": 48, "target_heldout_rows_per_case": 96,
        "mse_threshold": TOLERANCE,
        "counts": {
            "ambiguous_short_pairs": sum(item["checks"]["sampled_short_sources_equal"] for item in pairs),
            "distinguished_long_pairs": sum(item["checks"]["fourth_impulse_point_distinguishes"] for item in pairs),
            "source_short_model_correct": sum(item["source_heldout_mse"] <= TOLERANCE for item in results),
            "long_reference_correct": sum(item["long_reference_heldout_mse"] <= TOLERANCE for item in results),
            "finite_shortcut_correct": sum(item["frozen_short_model_long_mse"] <= TOLERANCE
                                           for item in results if item["instrument_mode"] == "finite2"),
            "recurrent_shortcut_rejected": sum(item["frozen_short_model_long_mse"] > TOLERANCE
                                               for item in results if item["instrument_mode"] == "recurrent"),
        },
        "pairs": pairs, "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSON output file; never overwritten.")
    parser.add_argument("--cases", type=int, default=12, help="Number of paired apparatuses (default: 12).")
    parser.add_argument("--seed", type=int, default=84051001)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = run(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("experiment", "passed", "evidence_kind", "apparatus_count", "case_count", "counts")}
                     | {"output": str(args.output.resolve())}, separators=(",", ":")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
