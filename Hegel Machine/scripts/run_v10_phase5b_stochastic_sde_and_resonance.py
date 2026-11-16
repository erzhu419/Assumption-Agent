"""Run v10 Phase 5B controlled study: Stochastic Process-Noise SDE 2SLS-IV Premise Graph & Active Resonance Probing.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase5b_stochastic_sde_and_resonance_20261116.json``

Evaluates two Phase 5B breakthroughs:
  1. Cohort 5B-1 — Strong Intrinsic Process-Noise SDEs (``sigma_p in [0.025, 0.035]``):
     Compares 3 matched ``UnifiedResearchAgent`` arms across 4 anonymous SDE POMDP episodes
     (including ``N=4 -> N=5`` cross-scale transfer & 2nd-order oscillatory SDE):
       * ``deterministic_v10_no_sde_audit`` (``stochastic_premise_audit=False``)
       * ``stochastic_v10_naive_ols`` (``stochastic_premise_audit=True, stochastic_estimation_method='naive_ols'``)
       * ``v10_phase5b_stochastic_2sls_iv`` (``stochastic_premise_audit=True, stochastic_estimation_method='2sls_iv'``)
     Measures both live noisy completion and exact **Oracle Zero-Noise Conditional Drift MSE**
     (``E[(y_hat - E[y|F_t])^2]``) + Hurwicz-Nickell endogeneity bias ``alpha_OLS - alpha*`` vs ``alpha_IV - alpha*``.
  2. Cohort 5B-2 — Near-Critical Supercritical Van der Pol Limit Cycles (``rho in {1.008, 1.015, 1.02, 1.04}``,
     including exact Phase 4B holdout ``990401``):
     Compares 3 matched ``UnifiedResearchAgent`` arms across 4 anonymous episodes (``N=4`` discovery + ``N=5`` transfer).
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
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eval_heldout_48(lab, res: dict, *, eval_seed: int, trajectories: int = 1) -> float | None:
    if not res.get("program"):
        return None
    prog = Program.from_dict(res["program"])
    channels = tuple(res.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=trajectories)
    return float(loss(prog, rows, res["fallback"], 0.5))


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    v4 = _schema(4, "src4")
    base_te_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]

    # ------------------------------------------------------------------
    # Cohort 5B-1: Stochastic Process-Noise SDE Benchmark (2SLS-IV vs OLS vs Det-v10)
    # ------------------------------------------------------------------
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

    out_path = EVIDENCE_DIR / "v10_phase5b_stochastic_sde_and_resonance_20261116.json"
    refresh_vdp_only = "--refresh-vdp" in sys.argv and out_path.exists()

    cohort_5b1_episodes = []
    if refresh_vdp_only:
        prev_5b = json.loads(out_path.read_text(encoding="utf-8"))
        cohort_5b1_episodes = prev_5b["cohort_5b1_episodes"]
    else:
        for chain in sde_chains:
            libs_sde = {}
            for arm_name, _, _ in sde_arms:
                lb = KnowledgeLibrary()
                lb.meta_priors = deepcopy(base_te_priors)
                libs_sde[arm_name] = lb

            for sp in chain["tasks"]:
                ep = {
                    "chain_id": chain["chain_id"],
                    **sp,
                    "arms": {},
                }
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
                            library=libs_sde[arm_name],
                        ).run(noisy_lab)
                        noisy_mse = _eval_heldout_48(noisy_lab, res, eval_seed=sp["seed"] + 100000, trajectories=2)

                    # Evaluate exact conditional drift accuracy E[(y_hat - E[y|F_t])^2] on noiseless oracle twin!
                    with AnonymousObjectLab(**lab_common, process_noise_std=0.0) as oracle_drift_lab:
                        true_drift_mse = _eval_heldout_48(
                            oracle_drift_lab, res, eval_seed=sp["seed"] + 100000, trajectories=2
                        )

                    sde_audits = [
                        a for a in res["actions"] if a.get("action") == "audit_stochastic_transition_premises"
                    ]
                    lat_st = (sde_audits[-1].get("latent_transition_state") or {}) if sde_audits else {}
                    est_pole = lat_st.get("pole")
                    ep["arms"][arm_name] = {
                        "status": res["status"],
                        "selected_channels": res.get("selected_channels"),
                        "noisy_heldout_mse": noisy_mse,
                        "oracle_zero_noise_drift_mse": true_drift_mse,
                        "drift_mse_under_2e3": bool(true_drift_mse is not None and true_drift_mse <= 2e-3),
                        "estimated_pole": est_pole,
                        "pole_abs_error": (abs(float(est_pole) - sp["pole"]) if est_pole is not None else None),
                        "iv_diagnostics": lat_st.get("iv_diagnostics"),
                        "costs": res["costs"],
                        "compute_cost": res["costs"]["compute_cost"],
                        "wall_seconds": round(time.monotonic() - t0, 3),
                    }
                cohort_5b1_episodes.append(ep)

    # ------------------------------------------------------------------
    # Cohort 5B-2: Near-Critical Supercritical Van der Pol Limit Cycles (including 990401!)
    # ------------------------------------------------------------------
    vdp_arms = (
        ("flat_baseline_continuous_state", False, False, False),
        ("linear_ar2_pomdp_v10_phase3a", True, False, False),
        ("v10_phase5b_active_resonance_vdp", True, True, True),
    )

    vdp_chains = [
        {
            "chain_id": "vdp_near_critical_chain_990401",
            "source": {
                "task_id": "990401",
                "objects": 4,
                "seed": 990401,
                "rho": 1.02,
                "freq": 0.65,
                "beta": 0.28,
                "weight": 0.44,
                "radius": 0.96,
            },
            "target": {
                "task_id": "990501",
                "objects": 5,
                "seed": 990501,
                "rho": 1.04,
                "freq": 0.58,
                "beta": 0.33,
                "weight": 0.48,
                "radius": 1.12,
            },
        },
        {
            "chain_id": "vdp_ultra_near_critical_chain_rho1008",
            "source": {
                "task_id": "990499",
                "objects": 4,
                "seed": 990499,
                "rho": 1.008,
                "freq": 0.72,
                "beta": 0.30,
                "weight": 0.45,
                "radius": 0.94,
            },
            "target": {
                "task_id": "990599",
                "objects": 5,
                "seed": 990599,
                "rho": 1.015,
                "freq": 0.80,
                "beta": 0.27,
                "weight": 0.47,
                "radius": 1.08,
            },
        },
    ]

    cohort_5b2_episodes = []
    for chain in vdp_chains:
        libs_vdp = {}
        for arm_name, use_pomdp, _, _ in vdp_arms:
            lb = KnowledgeLibrary()
            if use_pomdp:
                lb.meta_priors = deepcopy(base_te_priors)
            libs_vdp[arm_name] = lb

        for stage_label, sp in (("n4_discovery", chain["source"]), ("n4_to_n5_transfer", chain["target"])):
            ep = {
                "chain_id": chain["chain_id"],
                "stage": stage_label,
                **sp,
                "arms": {},
            }
            for arm_name, use_pomdp, use_vdp, use_res in vdp_arms:
                t0 = time.monotonic()
                cfg = ResearchConfig(
                    observation_budget=400 if sp["objects"] >= 5 else 380,
                    compute_budget=480000 if sp["objects"] >= 5 else 380000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_pomdp,
                    transition_premise_audit=use_pomdp,
                    expanded_premise_revisions=use_pomdp,
                    latent_state_discovery=use_pomdp,
                    higher_order_latent_discovery=use_pomdp,
                    nonlinear_limit_cycle_discovery=use_vdp,
                    active_resonance_probing=use_res,
                )
                lib_cur = libs_vdp[arm_name]
                with AnonymousObjectLab(
                    seed=sp["seed"],
                    objects=sp["objects"],
                    readout="total_event",
                    relation_direction="right",
                    relation_radius=sp["radius"],
                    latent_mode="van_der_pol_transition_memory",
                    latent_decay=sp["rho"],
                    latent_freq=sp["freq"],
                    latent_nonlinearity=sp["beta"],
                    latent_weight=sp["weight"],
                    actuator_curvature=0.13,
                    name_seed=sp["seed"] % 997 + 23,
                    wiring_seed=sp["seed"] % 991 + 47,
                ) as lab_v:
                    res = UnifiedResearchAgent(
                        cfg,
                        measurement_budget=8800 if sp["objects"] >= 5 else 7800,
                        library=lib_cur,
                    ).run(lab_v)
                    mse = _eval_heldout_48(lab_v, res, eval_seed=sp["seed"] + 100000)

                vdp_comms = (
                    [
                        c
                        for c in consolidate_world_structure_commitments(
                            lib_cur.meta_priors, budget=SearchBudget(30000)
                        )
                        if c.get("latent_state_mechanism") == "van_der_pol_transition_memory"
                    ]
                    if use_pomdp
                    else []
                )
                res_probes = [
                    a for a in res["actions"] if a.get("action") == "near_critical_resonance_probe"
                ]
                ep["arms"][arm_name] = {
                    "status": res["status"],
                    "selected_channels": res.get("selected_channels"),
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "vdp_commitments": len(vdp_comms),
                    "active_resonance_probes_fired": len(res_probes),
                    "costs": res["costs"],
                    "compute_cost": res["costs"]["compute_cost"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            cohort_5b2_episodes.append(ep)

    sde_summary = {}
    for arm_name, _, _ in sde_arms:
        rows = [ep["arms"][arm_name] for ep in cohort_5b1_episodes]
        drifts = [r["oracle_zero_noise_drift_mse"] for r in rows if r["oracle_zero_noise_drift_mse"] is not None]
        p_errs = [r["pole_abs_error"] for r in rows if r["pole_abs_error"] is not None]
        sde_summary[arm_name] = {
            "episodes_total": len(rows),
            "completed_count": sum(1 for r in rows if r["status"] == "completed"),
            "drift_mse_under_2e3_count": sum(1 for r in rows if r["drift_mse_under_2e3"]),
            "mean_oracle_zero_noise_drift_mse": float(mean(drifts)) if drifts else None,
            "mean_pole_abs_error": float(mean(p_errs)) if p_errs else None,
            "mean_compute_cost": float(mean(r["compute_cost"] for r in rows)),
        }

    vdp_summary = {}
    for arm_name, _, _, _ in vdp_arms:
        rows = [ep["arms"][arm_name] for ep in cohort_5b2_episodes]
        mses = [r["fresh48_mse"] for r in rows if r["fresh48_mse"] is not None]
        vdp_summary[arm_name] = {
            "episodes_total": len(rows),
            "completed_count": sum(1 for r in rows if r["status"] == "completed"),
            "passed_1e7_count": sum(1 for r in rows if r["passed_1e7"]),
            "zero_channel_count": sum(1 for r in rows if r["selected_channels"] == []),
            "mean_fresh48_mse": float(mean(mses)) if mses else None,
            "mean_compute_cost": float(mean(r["compute_cost"] for r in rows)),
        }

    receipt = {
        "experiment_id": "v10_phase5b_stochastic_sde_and_resonance_20261116",
        "generated_at": _utc_now(),
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t_start, 3),
        "cohort_5b1_sde_summary": sde_summary,
        "cohort_5b2_near_critical_vdp_summary": vdp_summary,
        "cohort_5b1_episodes": cohort_5b1_episodes,
        "cohort_5b2_episodes": cohort_5b2_episodes,
    }
    out_path = EVIDENCE_DIR / "v10_phase5b_stochastic_sde_and_resonance_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("SDE Summary:", json.dumps(sde_summary, indent=2))
    print("VdP Summary:", json.dumps(vdp_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
