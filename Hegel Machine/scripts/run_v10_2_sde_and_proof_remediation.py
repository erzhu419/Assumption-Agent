"""Run v10.2 Deliverable A empirical remediation receipt (model_calls = 0, zero overwrite of v10 receipts).

Writes new receipt:
  ``docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json``

Covers:
  1. Phase 5B SDE Re-Audit (F08, Acceptance #17):
     Separates ``native_completed_count`` (prominently surfaced), ``live_noisy_trajectory_mse``,
     ``noisy_history_conditional_drift_mse`` (exact analytical + 64-rollout Monte Carlo on realized
     noisy history ``F_t ~ E_noisy``), ``noiseless_twin_transfer_mse`` (distribution-shifted twin),
     and ``information_set_contract``.
  2. Phase 5C Soundness, Budget, Shared Witness Checker & Learned-Model Bridge (F01, F02, F05,
     F07, F10, F11, F12, F14, Acceptance #1-#8, #20):
     - Verifies GPT-6's exact rational counterexample across all three ``state_bounds_mode`` regimes
       (``physical_legality``, ``analysis_region``, ``unbounded``);
     - Verifies ``SearchBudget`` deduction across all invariant/solvability/rollout/repair calls;
     - Gives ``certify_rollout_baseline`` access to the exact same ``SymbolicProofKernel`` witness checker;
     - Demonstrates an end-to-end ``learned_model -> certificate`` pipeline from anonymous ``AnonymousObjectLab``
       data (``model_origin = "learned_model"``) distinct from ``supplied_model`` solver checks.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.run_v10_phase3a_higher_order_and_coupled_pomdp import (
    _make_markov_right_prior,
    _schema,
)
from hegel_machine.cognition.algebraic_proof_certificates import (
    BudgetedControlLab,
    SymbolicProofKernel,
    _to_frac,
    certify_rollout_baseline,
    extract_control_model_from_learned_commitment,
    prove_task_solvability,
    synthesize_emergent_invariants,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.stochastic_premise_audit import (
    evaluate_sde_conditional_drift_and_twin_metrics,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
)

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_sde_remediation() -> dict:
    v4 = _schema(4, "src4")
    base_te_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]
    sde_arms = (
        ("deterministic_v10_no_sde_audit", False, "2sls_iv"),
        ("stochastic_v10_naive_ols", True, "naive_ols"),
        ("v10_phase5b_stochastic_2sls_iv", True, "2sls_iv"),
    )
    sde_chains = [
        {
            "chain_id": "sde_ar1_n4_to_n5_chain",
            "tasks": [
                {
                    "task_id": "530401",
                    "stage": "n4_sde_discovery",
                    "objects": 4,
                    "seed": 530401,
                    "latent_mode": "transition_memory",
                    "pole": 0.74,
                    "freq": 0.0,
                    "weight": 0.48,
                    "radius": 0.95,
                    "sigma_p": 0.030,
                },
                {
                    "task_id": "530501",
                    "stage": "n4_to_n5_sde_transfer",
                    "objects": 5,
                    "seed": 530501,
                    "latent_mode": "transition_memory",
                    "pole": 0.68,
                    "freq": 0.0,
                    "weight": 0.52,
                    "radius": 1.06,
                    "sigma_p": 0.026,
                },
            ],
        },
        {
            "chain_id": "sde_high_memory_and_ar2_chain",
            "tasks": [
                {
                    "task_id": "530402",
                    "stage": "n4_high_pole_sde",
                    "objects": 4,
                    "seed": 530402,
                    "latent_mode": "transition_memory",
                    "pole": 0.80,
                    "freq": 0.0,
                    "weight": 0.46,
                    "radius": 0.93,
                    "sigma_p": 0.035,
                },
                {
                    "task_id": "530403",
                    "stage": "n4_oscillatory_ar2_sde",
                    "objects": 4,
                    "seed": 530403,
                    "latent_mode": "oscillatory_transition_memory",
                    "pole": 0.76,
                    "freq": 0.70,
                    "weight": 0.46,
                    "radius": 0.94,
                    "sigma_p": 0.025,
                },
            ],
        },
    ]

    episodes = []
    for chain in sde_chains:
        libs = {}
        for arm_name, _, _ in sde_arms:
            lb = KnowledgeLibrary()
            lb.meta_priors = deepcopy(base_te_priors)
            libs[arm_name] = lb

        for sp in chain["tasks"]:
            ep = {"chain_id": chain["chain_id"], **sp, "arms": {}}
            for arm_name, use_sde, est_method in sde_arms:
                t0 = time.monotonic()
                cfg = ResearchConfig(
                    observation_budget=520,
                    compute_budget=680000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=True,
                    transition_premise_audit=True,
                    expanded_premise_revisions=True,
                    latent_state_discovery=True,
                    higher_order_latent_discovery=True,
                    stochastic_premise_audit=use_sde,
                    stochastic_estimation_method=est_method,
                )
                lab_common = dict(
                    seed=sp["seed"],
                    objects=sp["objects"],
                    readout="total_event",
                    relation_direction="right",
                    relation_radius=sp["radius"],
                    latent_mode=sp["latent_mode"],
                    latent_decay=sp["pole"],
                    latent_freq=sp["freq"] if sp["freq"] > 0 else None,
                    latent_weight=sp["weight"],
                    name_seed=sp["seed"] % 997 + 17,
                    wiring_seed=sp["seed"] % 991 + 31,
                )
                with AnonymousObjectLab(**lab_common, process_noise_std=sp["sigma_p"]) as noisy_lab:
                    res = UnifiedResearchAgent(
                        cfg,
                        measurement_budget=12500 if sp["objects"] >= 5 else 10500,
                        library=libs[arm_name],
                    ).run(noisy_lab)
                    prog = Program.from_dict(res["program"]) if res.get("program") else None
                    sde_eval = evaluate_sde_conditional_drift_and_twin_metrics(
                        noisy_lab,
                        prog,
                        res["fallback"],
                        tuple(res.get("selected_channels") or ()),
                        eval_seed=sp["seed"] + 100000,
                        length=48,
                        trajectories=2,
                        mc_rollouts=64,
                    )

                sde_audits = [
                    a for a in res["actions"] if a.get("action") == "audit_stochastic_transition_premises"
                ]
                lat_st = (sde_audits[-1].get("latent_transition_state") or {}) if sde_audits else {}
                est_pole = lat_st.get("pole")
                ep["arms"][arm_name] = {
                    "native_status": res["status"],
                    "native_completed": bool(res["status"] == "completed"),
                    "selected_channels": res.get("selected_channels"),
                    "live_noisy_trajectory_mse": sde_eval["live_noisy_trajectory_mse"],
                    "noisy_history_conditional_drift_mse": sde_eval["noisy_history_conditional_drift_mse"],
                    "noisy_history_mc_conditional_drift_mse": sde_eval["noisy_history_mc_conditional_drift_mse"],
                    "noiseless_twin_transfer_mse": sde_eval["noiseless_twin_transfer_mse"],
                    "information_set_contract": sde_eval["information_set_contract"],
                    "estimated_pole": est_pole,
                    "pole_abs_error": (abs(float(est_pole) - sp["pole"]) if est_pole is not None else None),
                    "costs": res["costs"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            episodes.append(ep)

    summary_by_arm = {}
    for arm_name, _, _ in sde_arms:
        rows = [ep["arms"][arm_name] for ep in episodes]
        live_mses = [r["live_noisy_trajectory_mse"] for r in rows if r["live_noisy_trajectory_mse"] is not None]
        cond_mses = [
            r["noisy_history_conditional_drift_mse"]
            for r in rows
            if r["noisy_history_conditional_drift_mse"] is not None
        ]
        mc_mses = [
            r["noisy_history_mc_conditional_drift_mse"]
            for r in rows
            if r["noisy_history_mc_conditional_drift_mse"] is not None
        ]
        twin_mses = [
            r["noiseless_twin_transfer_mse"] for r in rows if r["noiseless_twin_transfer_mse"] is not None
        ]
        p_errs = [r["pole_abs_error"] for r in rows if r["pole_abs_error"] is not None]
        summary_by_arm[arm_name] = {
            "study_type": "estimation_method_ablation_not_memory_attribution",
            "episodes_total": len(rows),
            "native_completed_count": sum(1 for r in rows if r["native_completed"]),
            "native_unresolved_count": sum(1 for r in rows if not r["native_completed"]),
            "noisy_history_conditional_drift_under_2e3_count": sum(
                1
                for r in rows
                if r["noisy_history_conditional_drift_mse"] is not None
                and r["noisy_history_conditional_drift_mse"] <= 2e-3
            ),
            "noiseless_twin_transfer_under_2e3_count": sum(
                1
                for r in rows
                if r["noiseless_twin_transfer_mse"] is not None
                and r["noiseless_twin_transfer_mse"] <= 2e-3
            ),
            "mean_live_noisy_trajectory_mse": float(mean(live_mses)) if live_mses else None,
            "mean_noisy_history_conditional_drift_mse": float(mean(cond_mses)) if cond_mses else None,
            "mean_noisy_history_mc_conditional_drift_mse": float(mean(mc_mses)) if mc_mses else None,
            "mean_noiseless_twin_transfer_mse": float(mean(twin_mses)) if twin_mses else None,
            "mean_pole_abs_error": float(mean(p_errs)) if p_errs else None,
            "mean_compute_cost": float(mean(r["costs"]["compute_cost"] for r in rows)),
        }

    return {"summary_by_arm": summary_by_arm, "episodes": episodes}


def _run_proof_remediation() -> dict:
    # 1. GPT-6 exact rational counterexample across all 3 state_bounds_mode regimes (F01)
    model_ce = dict(objects=2, positions=["0", "1/2"], coupling="1/2", radius="1", direction="right")
    base_task_ce = dict(
        initial_state=["0", "0"],
        control_bounds=["0", "1"],
        budget="1/2",
        step_cost="0",
        fuel_weights=["1/10", "1"],
        goal_matrix=[["0", "-1"]],
        goal_rhs=["-1"],
    )
    pot_ce = dict(
        kind="potential_lower_bound_certificate",
        certificate_id="audit_domain_counterexample",
        alpha_0="1",
        alpha=["0", "-1"],
        farkas_mu_step=["1/2", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"],
        farkas_mu_goal=["1"],
        used_premise_ids=["audit_unverified_state_box"],
    )
    wit_ce = dict(
        kind="reachability_witness_certificate",
        action_sequence=[["1", "0"], ["1", "0"], ["0", "0"]],
    )

    ce_regimes = {}
    for regime_name, task_variant in (
        ("physical_legality", {**base_task_ce, "state_bounds": ["-1", "0"], "state_bounds_mode": "physical_legality"}),
        ("analysis_region", {**base_task_ce, "state_bounds": ["-1", "0"], "state_bounds_mode": "analysis_region"}),
        ("unbounded", {**base_task_ce, "state_bounds": None}),
    ):
        b_ce = SearchBudget(10000)
        p_res = SymbolicProofKernel.verify_solvability_certificate(pot_ce, model_ce, task_variant, budget=b_ce)
        w_res = SymbolicProofKernel.verify_solvability_certificate(wit_ce, model_ce, task_variant, budget=b_ce)
        solver_res = prove_task_solvability(
            model_ce, task_variant, certificate_id=f"ce_{regime_name}", budget=b_ce
        )
        ce_regimes[regime_name] = {
            "potential_verified": bool(p_res["verified"]),
            "potential_status": p_res["status"],
            "witness_verified": bool(w_res["verified"]),
            "witness_status": w_res["status"],
            "both_opposite_certificates_accepted": bool(p_res["verified"] and w_res["verified"]),
            "solver_verdict": solver_res["verdict"],
            "solver_status": solver_res["status"],
            "budget_spent": b_ce.spent,
        }

    # 2. Cohort 5C-1 with SearchBudget accounting, finite-arity labeling & cone obligations (F02, F10, F11, F12)
    invariant_specs = [
        {"id": "inv_pomdp_1", "system_class": "coupled_state_reservoir_pomdp", "alpha": "3/4", "eta": "1/2", "c_u": "1"},
        {"id": "inv_pomdp_2", "system_class": "coupled_state_reservoir_pomdp", "alpha": "2/3", "eta": "2/5", "c_u": "1/2"},
        {"id": "inv_stoich_2x", "system_class": "two_species_reaction_transport", "stoichiometric_ratio": "2"},
        {"id": "inv_stoich_3x", "system_class": "two_species_reaction_transport", "stoichiometric_ratio": "3"},
        {"id": "inv_neumann", "system_class": "manifold_neumann_conservation", "diffusion": "1/7", "advection": "1/11"},
        {"id": "inv_dirichlet_gg", "system_class": "manifold_dirichlet_gauss_green", "diffusion": "1/7", "advection": "1/11"},
        {"id": "inv_right_moment", "system_class": "directional_moment_monotonicity", "gamma": "1/4", "direction": "right"},
    ]
    inv_records = []
    inv_budget = SearchBudget(200000)
    for sp_item in invariant_specs:
        before_b = inv_budget.spent
        certs = synthesize_emergent_invariants(sp_item, budget=inv_budget)
        cert = certs[0]
        inv_records.append(
            {
                "theorem_id": sp_item["id"],
                "model_origin": cert["model_origin"],
                "verified_on_finite_arities": cert["verified_on_finite_arities"],
                "all_arity_proved": cert["all_arity_proved"],
                "lean_status": cert["lean_status"],
                "lean_compiled": cert["lean_compiled"],
                "proof_verified_by": cert["proof_verified_by"],
                "kernel_status": cert["kernel_verification"]["status"],
                "formal_obligations": cert["kernel_verification"]["formal_obligations"],
                "measured_compute_cost": inv_budget.spent - before_b,
            }
        )

    # 3. Shared witness checker on Cohort 5C-2 solvability tasks (F07, F10)
    from scripts.run_v10_phase5c_algebraic_proofs_and_impossibility import (
        main as _unused_5c_ref,  # noqa: F401
    )

    solvability_tasks = [
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
    ]

    solv_records = []
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
            state_bounds=tuple(tk["state_bounds"]),
            step_cost=tk["step_cost"],
        )
        b_roll = SearchBudget(200000)
        roll_res = certify_rollout_baseline(lab, ms, tk, budget=b_roll)
        b_lp = SearchBudget(200000)
        lp_res = prove_task_solvability(ms, tk, certificate_id=item["task_id"], budget=b_lp)
        solv_records.append(
            {
                "task_id": item["task_id"],
                "ground_truth": item["ground_truth"],
                "rollout_with_shared_witness_checker": {
                    "verdict": roll_res["verdict"],
                    "certified_proof": roll_res["certified_proof"],
                    "status": roll_res["status"],
                    "measured_compute_cost": b_roll.spent,
                },
                "lp_farkas_solver_with_checker": {
                    "verdict": lp_res["verdict"],
                    "status": lp_res["status"],
                    "model_origin": lp_res["model_origin"],
                    "lean_status": lp_res["lean_status"],
                    "measured_compute_cost": b_lp.spent,
                },
            }
        )

    # 4. End-to-End Learned-Model -> Certificate Bridge (F14, Acceptance #8)
    from scripts.run_v10_phase2b_open_grammar_llm import (
        _acquire_shared_discovery_bundle,
        _build_parent_commitment_from_empty_library,
    )
    from hegel_machine.cognition.premise_dependency_audit import audit_transition_premises

    _, learned_comm, learn_info = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    with AnonymousObjectLab(
        seed=600402,
        objects=4,
        readout="window_total_event",
        relation_direction="right",
        relation_radius=0.94,
        actuator_curvature=0.16,
        name_seed=600419,
        wiring_seed=600431,
    ) as learn_lab:
        b_aud = _acquire_shared_discovery_bundle(learn_lab, seed=600402)
        last_audit = audit_transition_premises(
            b_aud["model"], b_aud["records"], SearchBudget(50000), commitments=[learned_comm]
        )

    learned_spec = extract_control_model_from_learned_commitment(
        learned_comm,
        last_audit,
        positions=[-1.8, -0.5, 1.0, 1.8],
    )
    res_learn = {"status": learn_info["s1_status"]}
    b_learned_cert = SearchBudget(50000)
    learned_winnable_cert = prove_task_solvability(
        learned_spec,
        {
            "initial_state": [0.0, 0.0, 2.0, 0.0],
            "control_bounds": [-0.25, 0.25],
            "state_bounds": [0.0, 10.0],
            "budget": 0.15,
            "step_cost": 0.0,
            "fuel_weights": [1.0, 1.0, 1.0, 1.0],
            "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
            "goal_rhs": [-0.45],
        },
        certificate_id="learned_model_winnable",
        budget=b_learned_cert,
    )
    learned_imposs_cert = prove_task_solvability(
        learned_spec,
        {
            "initial_state": [0.0, 0.0, 0.0, 0.0],
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-10.0, 10.0],
            "budget": 1.10,
            "step_cost": 0.0,
            "fuel_weights": [1.0, 1.0, 1.0, 1.0],
            "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
            "goal_rhs": [-1.50],
        },
        certificate_id="learned_model_impossible",
        budget=b_learned_cert,
    )

    return {
        "gpt6_domain_counterexample_regimes": ce_regimes,
        "emergent_invariants_budgeted_recheck": inv_records,
        "shared_witness_checker_solvability_comparison": solv_records,
        "learned_commitment_to_certificate_bridge": {
            "learning_status": res_learn["status"],
            "learned_model_spec": learned_spec,
            "model_origin": learned_winnable_cert["model_origin"],
            "empirically_supported_premises": learned_winnable_cert["empirically_supported_premises"],
            "winnable_certificate_status": learned_winnable_cert["status"],
            "impossibility_certificate_status": learned_imposs_cert["status"],
            "environment_coverage_verified": learned_imposs_cert["environment_coverage_verified"],
            "certificate_compute_cost": b_learned_cert.spent,
        },
    }


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sde_part = _run_sde_remediation()
    proof_part = _run_proof_remediation()

    receipt = {
        "experiment_id": "v10_2_sde_and_proof_remediation_20261116",
        "generated_at": _utc_now(),
        "audited_anchor_commit": "a66cd82b22f444044dbda98f43782c96f833a4f0",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "phase5b_sde_remediation": sde_part,
        "phase5c_proof_remediation": proof_part,
    }
    out_path = OUT_DIR / "sde_and_proof_remediation_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("SDE Summary:", json.dumps(sde_part["summary_by_arm"], indent=2))
    print(
        "Counterexample Regimes:",
        json.dumps(proof_part["gpt6_domain_counterexample_regimes"], indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
