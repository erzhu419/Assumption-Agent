"""Run v10 Phase 5C controlled study: Symbolic Algebraic Elimination, Certified Proofs & AWM §4.3 Premise Repair.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase5c_algebraic_proofs_and_impossibility_20261116.json``

Evaluates all three Phase 5C / AWM Stage 03 capabilities:
  1. Cohort 5C-1 — Cross-Commitment Symbolic Flux Elimination & Emergent Invariants (6 theorems):
     Automatic discovery via exact ``sympy.Matrix.nullspace()`` over ``QQ`` + ``SymbolicProofKernel``
     multi-arity polynomial identity verification + negative tamper-rejection audit + Lean 4 emission.
  2. Cohort 5C-2 — Certified Control Solvability (8 tasks: 4 Winnable + 4 Sealed-Budget Impossible):
     Compares ``bounded_rollout_search_no_proof`` vs ``v10_phase5c_symbolic_proof_kernel``
     (exact rational ``ReachabilityWitnessCertificate`` + ``PotentialLowerBoundCertificate`` via Farkas LP).
  3. Cohort 5C-3 — AWM §4.3 Discrepancy-Triggered Premise Audit & Re-Proof Loop (4 hidden-port levels):
     Compares ``untargeted_full_premise_scan`` vs ``v10_phase5c_sufficient_cover_repair`` when an overly
     restrictive world commitment certifies ``B_hat* > B*`` and receives binary discrepancy feedback.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from random import Random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.algebraic_proof_certificates import (
    BudgetedControlLab,
    SymbolicProofKernel,
    _to_frac,
    prove_task_solvability,
    repair_proof_from_discrepancy,
    synthesize_emergent_invariants,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_rollout_baseline(lab: BudgetedControlLab, task: dict, *, max_trials: int = 250, horizon: int = 5) -> dict:
    """Empirical rollout baseline without symbolic LP/Farkas proof capability."""
    rng = Random(4242)
    u_lo, u_hi = task["control_bounds"]
    budget = float(task["budget"])
    n = lab.objects
    for trial in range(max_trials):
        steps = (trial % horizon) + 1
        per_step_cap = min(u_hi, budget / max(1, steps * n))
        seq = [
            [rng.uniform(-per_step_cap, per_step_cap) for _ in range(n)]
            for _ in range(steps)
        ]
        out = lab.execute_trace(task["initial_state"], seq, task)
        if out["winnable_verified_on_lab"]:
            return {
                "verdict": "winnable",
                "certified_proof": False,
                "trials_spent": trial + 1,
                "witness_cost": out["total_cost"],
            }
    return {
        "verdict": "unproved_timeout",
        "certified_proof": False,
        "trials_spent": max_trials,
        "witness_cost": None,
    }


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cohort 5C-1: Cross-Commitment Symbolic Flux Elimination & Emergent Invariants
    # ------------------------------------------------------------------
    invariant_specs = [
        {"id": "inv_pomdp_1", "system_class": "coupled_state_reservoir_pomdp", "alpha": "3/4", "eta": "1/2", "c_u": "1"},
        {"id": "inv_pomdp_2", "system_class": "coupled_state_reservoir_pomdp", "alpha": "2/3", "eta": "2/5", "c_u": "1/2"},
        {"id": "inv_stoich_2x", "system_class": "two_species_reaction_transport", "stoichiometric_ratio": "2"},
        {"id": "inv_stoich_3x", "system_class": "two_species_reaction_transport", "stoichiometric_ratio": "3"},
        {"id": "inv_neumann", "system_class": "manifold_neumann_conservation", "diffusion": "1/7", "advection": "1/11"},
        {"id": "inv_dirichlet_gg", "system_class": "manifold_dirichlet_gauss_green", "diffusion": "1/7", "advection": "1/11"},
        {"id": "inv_right_moment", "system_class": "directional_moment_monotonicity", "gamma": "1/4"},
    ]

    cohort_5c1_records = []
    for sp_item in invariant_specs:
        certs = synthesize_emergent_invariants(sp_item)
        cert = certs[0]
        # Negative soundness audit: perturb a weight or parameter and confirm kernel REJECTS!
        tampered = deepcopy(cert)
        if "sum_z" in tampered["invariant_weights"]:
            tampered["invariant_weights"]["sum_z"] = "999/100"
        elif "sum_a" in tampered["invariant_weights"]:
            tampered["invariant_weights"]["sum_a"] = "999/100"
        elif "sum_q" in tampered["invariant_weights"]:
            tampered["exogenous_input_weights"]["sum_u"] = "999/100"
        else:
            tampered["exact_parameters"]["gamma"] = "-1/4"
        tamper_res = SymbolicProofKernel.verify_emergent_invariant_certificate(tampered)
        cohort_5c1_records.append(
            {
                "theorem_id": sp_item["id"],
                "system_class": sp_item["system_class"],
                "formula": cert["formula"],
                "used_premise_ids": cert["used_premise_ids"],
                "verified_arities": cert["kernel_verification"]["checked_arities"],
                "kernel_verified": cert["kernel_verification"]["verified"],
                "tampered_certificate_rejected": not tamper_res["verified"],
                "lean4_emitted_lines": len(cert["lean4_module"].splitlines()),
            }
        )

    # ------------------------------------------------------------------
    # Cohort 5C-2: Certified Solvability Benchmark (4 Winnable + 4 Impossible)
    # ------------------------------------------------------------------
    solvability_tasks = [
        # 1. Global conserved charge reachability (winnable, B = 1.60 >= B* = 1.50)
        {
            "task_id": "ctrl_5401_winnable_global",
            "ground_truth": "winnable",
            "model": {"objects": 4, "positions": [-1.8, -0.5, 0.4, 1.8], "coupling": 0.25, "radius": 1.0, "direction": "right"},
            "task": {
                "initial_state": [0.0, 0.0, 0.0, 0.0],
                "control_bounds": [-0.5, 0.5],
                "state_bounds": [-10.0, 10.0],
                "budget": 1.60,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
                "goal_rhs": [-1.5],
            },
        },
        # 2. Global conserved charge impossibility (impossible, B = 1.20 < B* = 1.50)
        {
            "task_id": "ctrl_5402_impossible_global",
            "ground_truth": "impossible",
            "model": {"objects": 4, "positions": [-1.8, -0.5, 0.4, 1.8], "coupling": 0.25, "radius": 1.0, "direction": "right"},
            "task": {
                "initial_state": [0.0, 0.0, 0.0, 0.0],
                "control_bounds": [-0.5, 0.5],
                "state_bounds": [-10.0, 10.0],
                "budget": 1.20,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
                "goal_rhs": [-1.5],
            },
        },
        # 3. Right-transport passive conveyor assist (winnable within tiny B = 0.10!)
        {
            "task_id": "ctrl_5403_winnable_right_conveyor",
            "ground_truth": "winnable",
            "model": {"objects": 4, "positions": [-1.8, -0.4, 1.1, 1.8], "coupling": 0.30, "radius": 1.0, "direction": "right"},
            "task": {
                "initial_state": [0.0, 0.0, 2.0, 0.0],
                "control_bounds": [-0.25, 0.25],
                "state_bounds": [0.0, 10.0],
                "budget": 0.10,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
                "goal_rhs": [-0.85],
            },
        },
        # 4. Left-transport directional obstruction on identical setup (impossible over physical nonnegative density cone!)
        {
            "task_id": "ctrl_5404_impossible_left_obstruction",
            "ground_truth": "impossible",
            "model": {"objects": 4, "positions": [-1.8, -0.4, 1.1, 1.8], "coupling": 0.30, "radius": 1.0, "direction": "left"},
            "task": {
                "initial_state": [0.0, 0.0, 2.0, 0.0],
                "control_bounds": [-0.25, 0.25],
                "state_bounds": [0.0, 10.0],
                "budget": 0.50,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
                "goal_rhs": [-0.85],
            },
        },
        # 5. Isolated site concentration within budget (winnable, B = 1.05 >= B* = 1.00)
        {
            "task_id": "ctrl_5405_winnable_isolated_target",
            "ground_truth": "winnable",
            "model": {"objects": 4, "positions": [-2.0, -0.6, 0.6, 2.0], "coupling": 0.25, "radius": 0.9, "direction": "symmetric"},
            "task": {
                "initial_state": [0.5, 0.5, 0.5, 0.0],
                "control_bounds": [-0.35, 0.35],
                "state_bounds": [-10.0, 10.0],
                "budget": 1.05,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
                "goal_rhs": [-1.0],
            },
        },
        # 6. Isolated site concentration under locality gap (impossible, B = 0.75 < B* = 1.00 despite sum q_0 = 1.5!)
        {
            "task_id": "ctrl_5406_impossible_locality_gap",
            "ground_truth": "impossible",
            "model": {"objects": 4, "positions": [-2.0, -0.6, 0.6, 2.0], "coupling": 0.25, "radius": 0.9, "direction": "symmetric"},
            "task": {
                "initial_state": [0.5, 0.5, 0.5, 0.0],
                "control_bounds": [-0.35, 0.35],
                "state_bounds": [-10.0, 10.0],
                "budget": 0.75,
                "step_cost": 0.0,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
                "goal_rhs": [-1.0],
            },
        },
        # 7. Step-overhead horizon winnable (step_cost = 0.10, u_hi = 0.5, goal = 1.8 in 1 step -> cost = 1.9 <= 2.0)
        {
            "task_id": "ctrl_5407_winnable_step_overhead",
            "ground_truth": "winnable",
            "model": {"objects": 4, "positions": [-1.8, -0.5, 0.5, 1.8], "coupling": 0.20, "radius": 0.8, "direction": "right"},
            "task": {
                "initial_state": [0.0, 0.0, 0.0, 0.0],
                "control_bounds": [-0.5, 0.5],
                "state_bounds": [-10.0, 10.0],
                "budget": 2.00,
                "step_cost": 0.10,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
                "goal_rhs": [-1.8],
            },
        },
        # 8. Step-overhead + actuator box limit impossibility (u_hi = 0.20 on 1 target site, step_cost = 0.25, goal q_3 >= 0.8 -> min 4 steps -> min cost 0.8 + 4*0.25 = 1.8 > 1.2)
        {
            "task_id": "ctrl_5408_impossible_step_box_bound",
            "ground_truth": "impossible",
            "model": {"objects": 4, "positions": [-2.0, -0.7, 0.5, 2.0], "coupling": 0.20, "radius": 0.8, "direction": "right"},
            "task": {
                "initial_state": [0.0, 0.0, 0.0, 0.0],
                "control_bounds": [-0.20, 0.20],
                "state_bounds": [-10.0, 10.0],
                "budget": 1.20,
                "step_cost": 0.25,
                "fuel_weights": [1.0, 1.0, 1.0, 1.0],
                "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
                "goal_rhs": [-0.8],
            },
        },
    ]

    cohort_5c2_records = []
    for item in solvability_tasks:
        ms = item["model"]
        tk = item["task"]
        lab = BudgetedControlLab(
            objects=ms["objects"],
            positions=tuple(ms["positions"]),
            coupling=ms["coupling"],
            radius=ms["radius"],
            direction=ms["direction"],
            control_bounds=tuple(tk["control_bounds"]),
            step_cost=tk["step_cost"],
        )
        rollout_res = _bounded_rollout_baseline(lab, tk)
        cert = prove_task_solvability(ms, tk, certificate_id=item["task_id"])
        lab_exec = None
        if cert["verdict"] == "winnable":
            acts_f = [[float(_to_frac(v)) for v in st] for st in cert["action_sequence"]]
            lab_exec = lab.execute_trace(tk["initial_state"], acts_f, tk)

        cohort_5c2_records.append(
            {
                "task_id": item["task_id"],
                "ground_truth": item["ground_truth"],
                "rollout_baseline": {
                    **rollout_res,
                    "correct_certified_decision": bool(
                        rollout_res["verdict"] == item["ground_truth"] and rollout_res["certified_proof"]
                    ),
                    "correct_heuristic_winnable_only": bool(rollout_res["verdict"] == item["ground_truth"]),
                },
                "v10_symbolic_proof_kernel": {
                    "certificate_kind": cert["kind"],
                    "verdict": cert["verdict"],
                    "kernel_verified": cert["kernel_verification"]["verified"],
                    "certified_min_budget": cert.get("certified_min_budget"),
                    "used_premise_ids": cert.get("used_premise_ids"),
                    "lab_replay_verified": lab_exec["winnable_verified_on_lab"] if lab_exec else None,
                    "exact_match_ground_truth": bool(
                        cert["verdict"] == item["ground_truth"] and cert["kernel_verification"]["verified"]
                    ),
                },
            }
        )

    # ------------------------------------------------------------------
    # Cohort 5C-3: AWM §4.3 Discrepancy-Triggered Premise Repair (4 hidden-port levels)
    # ------------------------------------------------------------------
    discrepancy_levels = [
        {"level_id": "sc25_port_gain_0.60", "objects": 4, "gain": 0.60, "goal_sum": 2.40, "budget": 1.65, "true_b_star": 1.50},
        {"level_id": "sc25_port_gain_1.00", "objects": 4, "gain": 1.00, "goal_sum": 2.00, "budget": 1.25, "true_b_star": 1.00},
        {"level_id": "sc25_port_gain_1.25", "objects": 5, "gain": 1.25, "goal_sum": 2.25, "budget": 1.15, "true_b_star": 1.00},
        {"level_id": "sc25_port_gain_1.50", "objects": 5, "gain": 1.50, "goal_sum": 3.00, "budget": 1.35, "true_b_star": 1.20},
    ]

    cohort_5c3_records = []
    for lv in discrepancy_levels:
        n_obj = lv["objects"]
        pos_list = [-1.8, *([-0.5 + 0.4 * k for k in range(n_obj - 2)]), 1.8]
        lab = BudgetedControlLab(
            objects=n_obj,
            positions=tuple(pos_list),
            coupling=0.25,
            radius=1.0,
            direction="right",
            boundary_port_gain=lv["gain"],
            boundary_port_threshold=1.5,
            control_bounds=(-0.5, 0.5),
        )
        init_model = {
            "objects": n_obj,
            "positions": pos_list,
            "coupling": 0.25,
            "radius": 1.0,
            "direction": "right",
            "boundary_port_gain": 0.0,
            "boundary_port_threshold": 1.5,
            "premise_ids": ["P_conservation", "P_interaction_direction", "P_locality_radius", "P_control_input_gain"],
        }
        tk = {
            "initial_state": [0.0] * n_obj,
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-10.0, 10.0],
            "budget": lv["budget"],
            "step_cost": 0.0,
            "fuel_weights": [1.0] * n_obj,
            "goal_matrix": [[-1.0] * n_obj],
            "goal_rhs": [-lv["goal_sum"]],
        }
        p_graph = {
            "nodes": [
                {"id": "P_conservation", "status": "supported"},
                {"id": "P_control_input_gain", "status": "supported"},
                {"id": "P_interaction_direction", "status": "supported"},
                {"id": "P_position_ownership", "status": "supported"},
                {"id": "P_locality_radius", "status": "supported"},
                {"id": "P_focus_membership", "status": "supported"},
                {"id": "P_temporal_readout", "status": "supported"},
            ]
        }
        cert_init = prove_task_solvability(init_model, tk, certificate_id=f"{lv['level_id']}_init")
        rep = repair_proof_from_discrepancy(
            lab,
            init_model,
            p_graph,
            tk,
            cert_init,
            discrepancy_feedback={"oracle_winnable": True},
        )
        # Full untargeted re-exploration baseline requires single-control + pairwise x-q + selector sweeps: 2*(3N+1) + N*(N-1)
        untargeted_probes = 2 * (3 * n_obj + 1) + n_obj * (n_obj - 1)
        cohort_5c3_records.append(
            {
                "level_id": lv["level_id"],
                "objects": n_obj,
                "initial_overstrict_min_budget": rep["initial_certified_min_budget"],
                "task_budget": lv["budget"],
                "true_min_budget": lv["true_b_star"],
                "repaired_certified_min_budget": rep["true_certified_min_budget"],
                "min_budget_abs_error": abs(float(rep["true_certified_min_budget"]) - lv["true_b_star"]),
                "audited_suspect_premises": rep["audited_suspect_premises"],
                "untouched_premises_count": len(rep["untouched_premises"]),
                "targeted_probes_used": rep["targeted_probe_count"],
                "untargeted_reexploration_probes_baseline": untargeted_probes,
                "probe_reduction_pct": round(
                    100.0 * (1.0 - rep["targeted_probe_count"] / untargeted_probes), 2
                ),
                "repaired_witness_kernel_verified": rep["repaired_reachability_certificate"][
                    "kernel_verification"
                ]["verified"],
                "repaired_witness_lab_verified": rep["real_lab_execution_verification"][
                    "winnable_verified_on_lab"
                ],
            }
        )

    summary = {
        "cohort_5c1_emergent_invariants": {
            "theorems_total": len(cohort_5c1_records),
            "kernel_verified_count": sum(1 for r in cohort_5c1_records if r["kernel_verified"]),
            "tamper_rejected_count": sum(1 for r in cohort_5c1_records if r["tampered_certificate_rejected"]),
        },
        "cohort_5c2_solvability_certification": {
            "tasks_total": len(cohort_5c2_records),
            "rollout_baseline_certified_count": sum(
                1 for r in cohort_5c2_records if r["rollout_baseline"]["correct_certified_decision"]
            ),
            "rollout_baseline_heuristic_winnable_hits": sum(
                1 for r in cohort_5c2_records if r["rollout_baseline"]["correct_heuristic_winnable_only"]
            ),
            "v10_kernel_certified_exact_count": sum(
                1
                for r in cohort_5c2_records
                if r["v10_symbolic_proof_kernel"]["exact_match_ground_truth"]
            ),
        },
        "cohort_5c3_discrepancy_premise_repair": {
            "levels_total": len(cohort_5c3_records),
            "repaired_and_lab_verified_count": sum(
                1
                for r in cohort_5c3_records
                if r["repaired_witness_kernel_verified"] and r["repaired_witness_lab_verified"]
            ),
            "max_true_min_budget_abs_error": max(r["min_budget_abs_error"] for r in cohort_5c3_records),
            "mean_probe_reduction_pct_vs_untargeted": round(
                sum(r["probe_reduction_pct"] for r in cohort_5c3_records) / len(cohort_5c3_records), 2
            ),
        },
    }

    receipt = {
        "experiment_id": "v10_phase5c_algebraic_proofs_and_impossibility_20261116",
        "generated_at": _utc_now(),
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t_start, 3),
        "summary": summary,
        "cohort_5c1_emergent_invariants": cohort_5c1_records,
        "cohort_5c2_solvability_tasks": cohort_5c2_records,
        "cohort_5c3_discrepancy_repairs": cohort_5c3_records,
    }
    out_path = EVIDENCE_DIR / "v10_phase5c_algebraic_proofs_and_impossibility_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
