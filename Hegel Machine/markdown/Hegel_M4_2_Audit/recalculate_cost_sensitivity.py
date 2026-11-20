"""Recompute the explicitly post-hoc M4.2 cost sensitivity from committed receipts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--evidence", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
rows = [json.loads(s) for s in (args.evidence / "part4_unified_execution_receipts.jsonl").read_text().splitlines()]
report = json.loads((args.evidence / "part4_unified_cross_pillar_lifecycle_and_breakeven.json").read_text())
by_arm = defaultdict(list)
for row in rows:
    if not row["is_out_of_scope"]:
        by_arm[row["arm"]].append(row)
result = {"description": "Post hoc accounting sensitivity only. Preserve actual decisions, observations and original costs. All Pi observation events repriced to 4 or 8; remove charge-only rediscovery entry. Not a corrected benchmark and not a full fixed baseline rerun.", "arms": {}}
for arm, runs in by_arm.items():
    values = {4: [], 8: []}
    charge_total = 0.0
    pi_observations = []
    for row in runs:
        cost = row["cost"]
        phase = cost["phase_breakdown"].get("open_compositional_rediscovery_diag", {})
        diag_meas = float(phase.get("meas", 0))
        diag_compute = float(phase.get("compute", 0))
        diag_obs = float(phase.get("obs", 0))
        charge_total += diag_meas + 0.05 * diag_compute
        base_cost = float(cost["composite_cost"]) - diag_meas - 0.05 * diag_compute
        is_pi = row["mechanism_family"] == "PiS_Adiabatic_CrossDomain_2D"
        for unit in (4, 8):
            normalized = base_cost
            if is_pi:
                normalized += unit * (float(cost["observation_cost"]) - diag_obs) - (float(cost["measurement_cost"]) - diag_meas)
            values[unit].append(normalized)
        if is_pi:
            pi_observations.append(cost["observation_cost"] - diag_obs)
    summary = report["stage3_arms"][arm]
    result["arms"][arm] = {
        "original_mean": summary["mean_composite_cost"],
        "upfront": summary["upfront_lifecycle_composite_cost"],
        "charge_only_diag_total": round(charge_total, 6),
        "pi_actual_observations": pi_observations,
        "same_pi_unit4_mean": sum(values[4]) / len(runs),
        "same_pi_unit8_mean": sum(values[8]) / len(runs),
        "normalized_costs_unit4": values[4],
        "normalized_costs_unit8": values[8],
    }
hegel = result["arms"]["Hegel_Unified_Continual_Forest"]
result["accounting_only_breakeven"] = {}
for arm in ("Frozen_After_Stage0_PoolFallback", "Frozen_After_Stage0_OpenRediscovery"):
    baseline = result["arms"][arm]
    cases = {}
    for unit in (4, 8):
        hc, bc, crossing = hegel["upfront"], baseline["upfront"], None
        key = "normalized_costs_unit" + str(unit)
        for k, (h, b) in enumerate(zip(hegel[key], baseline[key]), 1):
            hc += h
            bc += b
            if crossing is None and hc < bc:
                crossing = k
        cases[str(unit)] = {"first_crossing": crossing, "hegel_total80": round(hc, 6),
                            "baseline_total80": round(bc, 6), "savings80": round(bc-hc, 6)}
    result["accounting_only_breakeven"][arm] = cases
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result["accounting_only_breakeven"], indent=2))

