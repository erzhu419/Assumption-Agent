"""Run v10.4 Occam Premise-Minimality Isolation & Combined M=72 x N=8 Full-Stack Study (model_calls = 0).

Writes new receipt (zero modifications to any v10, v10.2, or v10.3 receipts):
  ``docs/evidence/v10_4_occam_and_gpt6_bundle_20261117/occam_minimality_ablation_20261117.json``

Contains three strictly separated studies:
  1. **Study 1A (Strict Single-Factor Occam Minimality Ablation at M in {36, 72})**:
     Uses the exact 72-prior library and 6 N=5 target tasks from v10.3 Track 1 with cold sensor roles
     (``sensor_role_calibration_priors=False``), comparing 4 equal-tool arms:
       - ``archive_no_occam`` (``indexed=False, occam_premise_minimality=False``)
       - ``archive_with_occam`` (``indexed=False, occam_premise_minimality=True`` — equal-tool fairness!)
       - ``structured_indexed_no_occam`` (``indexed=True, occam_premise_minimality=False``)
       - ``structured_indexed_with_occam`` (``indexed=True, occam_premise_minimality=True``)
  2. **Study 1B (Adversarial Insertion-Order Stress Test at M = 72)**:
     Places all 68 conflicting open-grammar priors *before* the 4 foundational priors
     (``hetero_68 + foundational_4``) on 4 target tasks (2 conditional ``donor_positive_gap``,
     2 unconditional ``always``), isolating ``occam_premise_minimality=False`` vs ``True``.
  3. **Study 2 (Combined v10.3 + v10.4 Full-Stack M = 72 Conflicting Library x N = 8 High-Arity Targets)**:
     Evaluates all 3 N=8 anonymous target apparatuses (24 controls, 32 sensors, (8!)^2 = 1.63e9
     role assignments) directly against the full M=72 heterogeneous conflicting library with
     ``indexed_commitment_store=True + occam_premise_minimality=True + sensor_role_calibration_priors=True``.
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

from scripts.run_v10_3_high_arity_calibration_study import (
    _study_2a_high_arity_tasks,
)
from scripts.run_v10_3_library_scaling_study import (
    _build_master_72_prior_library,
    _target_tasks,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equal_tool_attribution import (
    reconstruct_commitments_from_raw_archive,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import ensure_library_commitment_index
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_4_occam_and_gpt6_bundle_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_scaling_arm(
    *,
    arm_name: str,
    task_spec: dict,
    priors_slice: list[dict],
    preindexed_lib: KnowledgeLibrary | None,
    target_compute_cap: int,
    indexed_flag: bool,
    occam_flag: bool,
) -> dict:
    t0 = time.monotonic()
    kw = {k: v for k, v in task_spec.items() if k != "task_id"}
    seed = int(kw["seed"])
    eval_seed = seed + 1000003

    archive_rec_compute = 0
    if not indexed_flag:
        lib = KnowledgeLibrary(meta_priors=deepcopy(priors_slice))
        rec_b = SearchBudget(target_compute_cap // 2)
        _, archive_rec_compute = reconstruct_commitments_from_raw_archive(
            lib.meta_priors, rec_b, tolerance=1e-7
        )
    else:
        assert preindexed_lib is not None
        lib = KnowledgeLibrary(meta_priors=deepcopy(preindexed_lib.meta_priors))
        lib._indexed_prior_fingerprints = deepcopy(preindexed_lib._indexed_prior_fingerprints)
        lib._indexed_group_commitments = deepcopy(preindexed_lib._indexed_group_commitments)
        lib._premise_commitment_index = deepcopy(preindexed_lib._premise_commitment_index)

    rem_cap = max(1000, target_compute_cap - archive_rec_compute)
    cfg = ResearchConfig(
        seed=seed,
        observation_budget=340,
        compute_budget=rem_cap,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=indexed_flag,
        sensor_role_calibration_priors=False,
        occam_premise_minimality=occam_flag,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=8400, library=lib)
    with AnonymousObjectLab(**kw) as lab:
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")
        mse = (
            float(
                loss(
                    prog,
                    lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1),
                    res["fallback"],
                    0.5,
                )
            )
            if prog is not None
            else None
        )

    consolidation_or_retrieval_in_run = sum(
        int(a.get("compute_cost", 0))
        for a in res.get("actions", ())
        if a.get("action")
        in {"consolidate_world_structure_commitments", "retrieve_indexed_world_commitments"}
    )
    audit_actions = [
        a for a in res.get("actions", ()) if a.get("action") == "audit_transition_premises"
    ]
    audit_in_run = sum(int(a.get("compute_cost", 0)) for a in audit_actions)
    first_audit = audit_actions[0] if audit_actions else {}
    tot_target_compute = archive_rec_compute + int(res["costs"]["compute_cost"])
    passed = bool(confirmed and mse is not None and mse <= 1e-7)

    return {
        "task_id": task_spec["task_id"],
        "arm": arm_name,
        "indexed_commitment_store": indexed_flag,
        "occam_premise_minimality": occam_flag,
        "confirmed": confirmed,
        "passed_1e7": passed,
        "selected_channels": list(ch),
        "heldout_fresh48_mse": mse,
        "first_audit_verified_condition": first_audit.get("verified_edge_condition"),
        "first_audit_occam_gate": first_audit.get("occam_minimality_gate"),
        "costs": {
            "observation_cost": res["costs"]["observation_cost"],
            "measurement_cost": res["costs"]["measurement_cost"],
            "archive_on_demand_reconstruction_compute": archive_rec_compute,
            "in_run_consolidation_or_index_lookup_compute": consolidation_or_retrieval_in_run,
            "in_run_premise_audit_compute": audit_in_run,
            "agent_run_compute_cost": res["costs"]["compute_cost"],
            "total_target_compute_cost": tot_target_compute,
        },
        "model_calls": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def _summarize_arm_eps(eps: list[dict], write_once: int = 0) -> dict:
    tot_sum = sum(e["costs"]["total_target_compute_cost"] for e in eps)
    return {
        "passes_1e7": sum(1 for e in eps if e["passed_1e7"]),
        "total_tasks": len(eps),
        "upfront_write_consolidation_compute_once": write_once,
        "mean_on_demand_reconstruction_compute": float(
            mean(e["costs"]["archive_on_demand_reconstruction_compute"] for e in eps)
        ),
        "mean_in_run_consolidation_or_lookup_compute": float(
            mean(e["costs"]["in_run_consolidation_or_index_lookup_compute"] for e in eps)
        ),
        "mean_in_run_audit_compute": float(
            mean(e["costs"]["in_run_premise_audit_compute"] for e in eps)
        ),
        "mean_total_target_compute": float(
            mean(e["costs"]["total_target_compute_cost"] for e in eps)
        ),
        "sum_total_target_compute": tot_sum,
        "amortized_cumulative_compute_incl_write": tot_sum + write_once,
        "episodes": eps,
    }


def _run_m72_n8_episode(
    *,
    arm_name: str,
    task_info: dict,
    preindexed_m72_lib: KnowledgeLibrary,
) -> dict:
    t0 = time.monotonic()
    lab_kwargs = dict(task_info["lab_kwargs"])
    seed = int(lab_kwargs["seed"])
    n_obj = int(lab_kwargs["objects"])
    eval_seed = seed + 1000003
    obs_cap = 400
    comp_cap = 480000
    meas_cap = 3 * n_obj * obs_cap

    if arm_name == "cold_full_tools":
        lib = KnowledgeLibrary()
        use_mem = False
        use_idx = False
        use_occam = True
        use_cal = False
    elif arm_name == "m72_indexed_cold_roles_with_occam":
        lib = KnowledgeLibrary(meta_priors=deepcopy(preindexed_m72_lib.meta_priors))
        lib._indexed_prior_fingerprints = deepcopy(preindexed_m72_lib._indexed_prior_fingerprints)
        lib._indexed_group_commitments = deepcopy(preindexed_m72_lib._indexed_group_commitments)
        lib._premise_commitment_index = deepcopy(preindexed_m72_lib._premise_commitment_index)
        use_mem = True
        use_idx = True
        use_occam = True
        use_cal = False
    elif arm_name == "m72_full_stack_indexed_calibrated_occam":
        lib = KnowledgeLibrary(meta_priors=deepcopy(preindexed_m72_lib.meta_priors))
        lib._indexed_prior_fingerprints = deepcopy(preindexed_m72_lib._indexed_prior_fingerprints)
        lib._indexed_group_commitments = deepcopy(preindexed_m72_lib._indexed_group_commitments)
        lib._premise_commitment_index = deepcopy(preindexed_m72_lib._premise_commitment_index)
        use_mem = True
        use_idx = True
        use_occam = True
        use_cal = True
    else:
        raise ValueError(arm_name)

    cfg = ResearchConfig(
        seed=seed,
        observation_budget=obs_cap,
        compute_budget=comp_cap,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=use_mem,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=use_idx,
        sensor_role_calibration_priors=use_cal,
        occam_premise_minimality=use_occam,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=meas_cap, library=lib)
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")
        mse = (
            float(
                loss(
                    prog,
                    lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1),
                    res["fallback"],
                    0.5,
                )
            )
            if prog is not None
            else None
        )

    role_act = next(
        (
            a
            for a in res.get("actions", ())
            if a.get("action") in {"adopt_discovered_objects", "retain_object_ambiguity"}
        ),
        {},
    )
    cal_act = next(
        (
            a
            for a in res.get("actions", ())
            if "sensor_role_calibration" in str(a.get("action", ""))
        ),
        None,
    )
    audit_acts = [
        a for a in res.get("actions", ()) if a.get("action") == "audit_transition_premises"
    ]
    passed = bool(confirmed and mse is not None and mse <= 1e-7)
    return {
        "task_id": task_info["task_id"],
        "objects_N": n_obj,
        "library_size_M": 72 if use_mem else 0,
        "arm": arm_name,
        "confirmed": confirmed,
        "passed_1e7": passed,
        "selected_channels": list(ch),
        "heldout_fresh48_mse": mse,
        "role_discovery_compute": int(role_act.get("compute_cost", 0)),
        "in_run_audit_compute": sum(int(a.get("compute_cost", 0)) for a in audit_acts),
        "first_audit_occam_gate": audit_acts[0].get("occam_minimality_gate") if audit_acts else None,
        "calibration_action": cal_act.get("action") if cal_act else None,
        "costs": dict(res["costs"]),
        "model_calls": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building 72-prior heterogeneous conflicting library...", flush=True)
    master_72, lib_meta = _build_master_72_prior_library()
    tasks_6 = _target_tasks()

    # ------------------------------------------------------------------
    # Study 1A: Strict Single-Factor Occam Premise-Minimality Ablation
    # ------------------------------------------------------------------
    study_1a = []
    for m_size in (36, 72):
        slice_m = deepcopy(master_72[:m_size])
        indexed_lib = KnowledgeLibrary(meta_priors=deepcopy(slice_m))
        comms_m, _, write_once = ensure_library_commitment_index(
            indexed_lib, budget=SearchBudget(500000), tolerance=1e-7
        )

        arm_configs = [
            ("archive_no_occam", False, False),
            ("archive_with_occam", False, True),
            ("structured_indexed_no_occam", True, False),
            ("structured_indexed_with_occam", True, True),
        ]
        arms_out = {}
        for arm_name, idx_flag, occ_flag in arm_configs:
            eps = [
                _run_scaling_arm(
                    arm_name=arm_name,
                    task_spec=t,
                    priors_slice=slice_m,
                    preindexed_lib=indexed_lib if idx_flag else None,
                    target_compute_cap=380000,
                    indexed_flag=idx_flag,
                    occam_flag=occ_flag,
                )
                for t in tasks_6
            ]
            arms_out[arm_name] = _summarize_arm_eps(eps, write_once if idx_flag else 0)
            print(
                f"[Study 1A | M={m_size} | {arm_name}] "
                f"passes={arms_out[arm_name]['passes_1e7']}/6 | "
                f"mean_audit={arms_out[arm_name]['mean_in_run_audit_compute']:.1f} | "
                f"mean_total_compute={arms_out[arm_name]['mean_total_target_compute']:.1f}",
                flush=True,
            )

        per_task_saving_with_occam = (
            arms_out["archive_with_occam"]["sum_total_target_compute"]
            - arms_out["structured_indexed_with_occam"]["sum_total_target_compute"]
        ) / len(tasks_6)
        breakeven_with_occam = (
            round(write_once / per_task_saving_with_occam, 2)
            if per_task_saving_with_occam > 0
            else None
        )
        study_1a.append(
            {
                "library_priors_count_M": m_size,
                "consolidated_commitments_count": len(comms_m),
                "target_compute_cap": 380000,
                "upfront_write_consolidation_compute_once": write_once,
                "equal_tool_with_occam_per_task_compute_saving_indexed_vs_archive": round(
                    per_task_saving_with_occam, 2
                ),
                "equal_tool_with_occam_breakeven_tasks_K": breakeven_with_occam,
                "arms": arms_out,
            }
        )

    # ------------------------------------------------------------------
    # Study 1B: Adversarial Reversed Insertion-Order Stress Test (M = 72)
    # ------------------------------------------------------------------
    reversed_72 = deepcopy(master_72[4:] + master_72[:4])
    rev_indexed = KnowledgeLibrary(meta_priors=deepcopy(reversed_72))
    _, _, rev_write_once = ensure_library_commitment_index(
        rev_indexed, budget=SearchBudget(500000), tolerance=1e-7
    )
    stress_tasks = [tasks_6[0], tasks_6[2], tasks_6[4], tasks_6[5]]
    study_1b_arms = {}
    for arm_name, occ_flag in (
        ("reversed_m72_indexed_no_occam", False),
        ("reversed_m72_indexed_with_occam", True),
    ):
        eps = [
            _run_scaling_arm(
                arm_name=arm_name,
                task_spec=t,
                priors_slice=reversed_72,
                preindexed_lib=rev_indexed,
                target_compute_cap=380000,
                indexed_flag=True,
                occam_flag=occ_flag,
            )
            for t in stress_tasks
        ]
        study_1b_arms[arm_name] = _summarize_arm_eps(eps, rev_write_once)
        print(
            f"[Study 1B | Reversed M=72 | {arm_name}] "
            f"passes={study_1b_arms[arm_name]['passes_1e7']}/4 | "
            f"mean_audit={study_1b_arms[arm_name]['mean_in_run_audit_compute']:.1f} | "
            f"mean_total_compute={study_1b_arms[arm_name]['mean_total_target_compute']:.1f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Study 2: Combined Full-Stack M = 72 Library x N = 8 High-Arity Targets
    # ------------------------------------------------------------------
    print("Running Study 2: M=72 conflicting library x N=8 high-arity targets...", flush=True)
    indexed_72_full = KnowledgeLibrary(meta_priors=deepcopy(master_72))
    ensure_library_commitment_index(indexed_72_full, budget=SearchBudget(500000), tolerance=1e-7)

    n8_tasks = [t for t in _study_2a_high_arity_tasks() if int(t["lab_kwargs"]["objects"]) == 8]
    study_2_arms = {}
    for arm_name in (
        "cold_full_tools",
        "m72_indexed_cold_roles_with_occam",
        "m72_full_stack_indexed_calibrated_occam",
    ):
        eps_n8 = [
            _run_m72_n8_episode(
                arm_name=arm_name,
                task_info=t,
                preindexed_m72_lib=indexed_72_full,
            )
            for t in n8_tasks
        ]
        study_2_arms[arm_name] = {
            "passes_1e7": sum(1 for e in eps_n8 if e["passed_1e7"]),
            "total_tasks": len(eps_n8),
            "mean_role_discovery_compute": float(
                mean(e["role_discovery_compute"] for e in eps_n8)
            ),
            "mean_in_run_audit_compute": float(mean(e["in_run_audit_compute"] for e in eps_n8)),
            "mean_total_compute": float(mean(e["costs"]["compute_cost"] for e in eps_n8)),
            "episodes": eps_n8,
        }
        print(
            f"[Study 2 | M=72 x N=8 | {arm_name}] "
            f"passes={study_2_arms[arm_name]['passes_1e7']}/3 | "
            f"mean_role_compute={study_2_arms[arm_name]['mean_role_discovery_compute']:.1f} | "
            f"mean_total_compute={study_2_arms[arm_name]['mean_total_compute']:.1f}",
            flush=True,
        )

    receipt = {
        "experiment_id": "v10_4_occam_minimality_and_fullstack_20261117",
        "generated_at": _utc_now(),
        "base_commit": "c0bad267a4f211cda03f388e9386908e73c27016",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "library_construction": lib_meta,
        "study_1a_single_factor_occam_ablation": study_1a,
        "study_1b_adversarial_insertion_order_m72": {
            "description": "All 68 conflicting open-grammar priors placed BEFORE the 4 foundational priors.",
            "arms": study_1b_arms,
        },
        "study_2_combined_m72_x_n8_fullstack": {
            "description": "Full M=72 conflicting library combined with SensorRoleCalibrationPrior on N=8 anonymous apparatuses (24 controls, 32 sensors, (8!)^2 = 1.63e9 permutations).",
            "arms": study_2_arms,
        },
    }
    out_path = OUT_DIR / "occam_minimality_ablation_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
