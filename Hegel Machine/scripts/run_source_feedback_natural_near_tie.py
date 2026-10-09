#!/usr/bin/env python3
"""Run a natural near-tie source-attribution study.

The two frozen source offers keep their original expected gains.  The target
transition is chosen at their numerical crossing, so the scheduler's gain
resolution—not a hand-edited quote—decides whether a confirmed source return
can affect the next purchase.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_natural_near_tie_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_source_feedback_attribution import (
    _config, _delete_returns, _earn_positive_return, _prepare, _run_arm,
)
from run_source_feedback_task_family import _fixtures
from hegel_machine.cognition.memory import KnowledgeLibrary


SEED = 20261006
NEAR_TIE = {"current_weight": 0.5697373812934383, "lag_weight": 1.19, "noise": 0.0}
VARIANTS = (
    {"label": "near_tie_0", **NEAR_TIE},
    {"label": "shifted", "current_weight": .6, "lag_weight": 1.2, "noise": 0.},
    {"label": "near_tie_1", **NEAR_TIE},
    {"label": "noisy", "current_weight": NEAR_TIE["current_weight"],
     "lag_weight": NEAR_TIE["lag_weight"], "noise": .02},
)


def _natural_gains(prepared):
    return [{
        "task_position": position,
        "variant": variant["label"],
        "gain_by_source": {
            offer["candidate"]["source_concept_ids"][0]: offer["expected_loss_reduction"]
            for offer in prepared[2] if offer["task_position"] == position
        },
    } for position, (_spec, variant) in enumerate(prepared[1])]


def _arm_rows(arm):
    return {row["variant"]: row for row in arm["tasks"]}


def main():
    case, good, warm, _cold = _fixtures()
    positive_return = _earn_positive_return(case, good, warm)

    # Remove the known negative return before either arm.  This isolates the
    # good-return attribution; the two arms differ only in retaining that
    # confirmed record.
    isolated = KnowledgeLibrary.from_dict(deepcopy(warm.to_dict()))
    _delete_returns(isolated, "delete_bad", case["source_id"], good)
    prepared = _prepare(case, good, isolated, variants=VARIANTS)
    allocation_limits = {}
    for position in range(len(VARIANTS)):
        offers = [item for item in prepared[2] if item["task_position"] == position]
        if len(offers) != 2:
            raise RuntimeError(f"expected two frozen source offers at task {position}")
        allocation_limits[position] = {
            key: max(item["minimum_resources"][key] for item in offers)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}

    arms = {}
    for tolerance_label, tolerance in (("raw_gain", 0.0), ("measured_resolution", 1e-8)):
        for mode in ("retain_good", "delete_good"):
            arms[f"{tolerance_label}_{mode}"] = _run_arm(
                case, good, isolated, mode, prepared, allocation_limits,
                knowledge_tie_tolerance=tolerance)

    rows = {name: _arm_rows(arm) for name, arm in arms.items()}
    bad, source = case["source_id"], good
    raw_retain = rows["raw_gain_retain_good"]
    raw_delete = rows["raw_gain_delete_good"]
    measured_retain = rows["measured_resolution_retain_good"]
    measured_delete = rows["measured_resolution_delete_good"]

    for label in ("near_tie_0", "near_tie_1"):
        # The raw quote is naturally higher for the good source by only a few
        # floating-point units, so the raw arm chooses it even after deletion.
        assert raw_retain[label]["selected_source"] == [source]
        assert raw_delete[label]["selected_source"] == [source]
        assert measured_retain[label]["selected_source"] == [source]
        # With the good return deleted, the explicit resolution policy exposes
        # the remaining source instead of letting a 3e-15 quote difference win.
        assert measured_delete[label]["selected_source"] == [bad]

    for label in ("shifted", "noisy"):
        # The response condition does not transfer the good score across a
        # changed or unresolved response regime.
        assert measured_retain[label]["selected_source"] == measured_delete[label]["selected_source"]

    gain_rows = _natural_gains(prepared)
    near_gaps = [abs(item["gain_by_source"][bad] - item["gain_by_source"][source])
        for item in gain_rows if item["variant"].startswith("near_tie")]
    assert max(near_gaps) < 1e-10
    for arm in arms.values():
        assert arm["total_costs"] == arms["raw_gain_retain_good"]["total_costs"]
        assert all(row["external_model_calls"] == 0 for row in arm["tasks"])

    contrasts = {}
    for left, right in (
            ("raw_gain_retain_good", "raw_gain_delete_good"),
            ("measured_resolution_retain_good", "measured_resolution_delete_good")):
        contrasts[f"{left}_vs_{right}"] = {
            "action_divergence_count": sum(
                rows[left][label]["selected_source"] != rows[right][label]["selected_source"]
                for label in rows[left]),
            "heldout_loss_delta": {
                label: rows[left][label]["heldout_loss"] - rows[right][label]["heldout_loss"]
                for label in rows[left]
                if rows[left][label]["heldout_loss"] is not None
                and rows[right][label]["heldout_loss"] is not None
            },
        }

    payload = {
        "status": "complete",
        "lineage": {"package_version": "0.181.0",
            "parent": "source-specific attribution and withdrawal",
            "stage": "natural near-tie gain resolution"},
        "protocol": {
            "seed": SEED, "variants": list(VARIANTS), "arms": list(arms),
            "same_prepared_offers": True, "same_execution_limits": True,
            "bad_return_removed_before_arms": True, "external_model_calls": 0,
            "knowledge_tie_tolerance_is_relative": True,
            "tolerances": {"raw_gain": 0.0, "measured_resolution": 1e-8},
        },
        "positive_return_acquisition": positive_return,
        "natural_gain_rows": gain_rows,
        "contrasts": contrasts,
        "source_ids": {"bad": bad, "good": source},
        "arms": arms,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "arms": list(arms),
        "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
