"""Run v11.1 Prospective Remediation Study: Candidate-Generator & Scale-Free Version-Space M1 (2x2 KxM).

Preserves ALL 25 prior receipts (including v11 `deliverable_c_knowledge_vs_method_2x2_20261117.json`
on seeds `971000..976000`) 100% byte-for-byte untouched, and writes a new prospective receipt on a
strictly disjoint seed block (`981xxx` dev_formation, `982xxx` dev_selection, `985000..990000` unseen):
  ``docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_v11_1_method_remediation_2x2_20261117.json``

Tests the two explicit root-cause remediations diagnosed from v11 `deliverable_c` (§4):
  1. **Candidate-generator co-located synchronous bystander excitation (`u != 0, q != 0`)**:
     Eliminates curved-actuator (`q + c * q^3`) calibration starvation on `_fit_instruments`
     validation batches while keeping zero bystander cross-talk (`dx == 0, dq == 0`).
  2. **Public version-space interval-bisecting candidate probes + scale-free disagreement coverage**:
     Replaces `v11`'s global fixed `preferred_displacement_scale` penalty (`scale = 0.0718`) with
     observable version-space disagreement features (`disagreement_step_fraction`, `ensemble_std`,
     `pairwise_disagreement_margin`, `control_sparsity`, `sign_contrast`) induced from realistic
     post-role-confirmation `dev_formation` rollouts.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import log
from pathlib import Path
from random import Random
from statistics import mean
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.run_v11_knowledge_vs_method_2x2 import (
    _bootstrap_paired_ci_95,
    _build_chain_k1_and_archive,
    _build_shared_calibration_only_prior,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.discriminating_experiment_policy import (
    build_generic_schema_candidate_hypotheses,
    compute_observable_probe_features,
    fit_empirical_q_actuator_maps,
    frozen_default_method_m0,
    generate_candidate_probe_menu,
    induce_discriminating_experiment_method,
)
from hegel_machine.cognition.equal_tool_attribution import (
    UnifiedResourceAccount,
    extract_real_consumption_events_from_actions,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import (
    OrdinaryPersistentCommitmentCache,
    ensure_library_commitment_index,
)
from hegel_machine.cognition.premise_dependency_audit import audit_transition_premises
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v11_branch_lifecycle_and_km_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ambiguity_bits_score(aud: dict) -> float:
    """Measure log-volume ambiguity of direction, Occam boundary, radius interval, and threshold interval."""
    dir_unc = 0.0 if aud.get("verified_direction") else 1.0
    r_iv = aud.get("radius_interval")
    if isinstance(r_iv, (list, tuple)) and len(r_iv) == 2:
        r_w = max(0.0, min(1.8, float(r_iv[1]) - float(r_iv[0])))
        r_unc = log((r_w + 1e-4) / 1e-4) / log(1.8e4)
    else:
        r_unc = 1.0

    gate = aud.get("occam_minimality_gate") or {}
    t_iv = aud.get("verified_threshold_interval")
    if isinstance(aud.get("verified_edge_condition"), dict) and isinstance(t_iv, (list, tuple)) and len(t_iv) == 2:
        t_w = max(0.0, min(1.6, float(t_iv[1]) - float(t_iv[0])))
        t_unc = log((t_w + 1e-4) / 1e-4) / log(1.6e4)
    elif isinstance(aud.get("verified_edge_condition"), dict):
        t_unc = 0.9
    else:
        t_unc = 0.85 if gate.get("boundary_test_needed", True) else 0.0
    return float(dir_unc + r_unc + t_unc)


def _measure_empirical_elimination_per_row_v11_1(
    lab: AnonymousObjectLab,
    model: dict,
    base_records: list[dict],
    probe: Probe,
    k0_commitments: list[dict],
    aud_before: dict,
) -> float:
    """Execute ``probe`` on ``lab`` from post-role-confirmation state and measure log-volume contraction/row."""
    unc_before = _ambiguity_bits_score(aud_before)
    batch = lab.acquire(probe, ("fine",))
    new_recs = [
        {"controls": dict(pt), **tr, "target": float(obs.target)}
        for pt, obs, tr in zip(probe.sequence, batch.observations, batch.transitions)
    ]
    aud_after = audit_transition_premises(
        model,
        [*base_records, *new_recs],
        SearchBudget(35_000),
        commitments=k0_commitments,
        allow_expanded_revisions=True,
        occam_premise_minimality=True,
    )
    unc_after = _ambiguity_bits_score(aud_after)
    gain = max(0.0, unc_before - unc_after)
    return float(gain / max(1, probe.cost))


def _collect_dev_formation_rollouts_v11_1(
    cal_prior: list[dict],
    seeds: list[int],
    *,
    guiding_method: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect real post-role-confirmation probe rollouts on disjoint v11.1 `dev_formation` seeds."""
    rollout_records: list[dict[str, Any]] = []
    tot_comp = 0
    tot_obs = 0

    k0_lib_tmp = KnowledgeLibrary(meta_priors=deepcopy(cal_prior))
    ensure_library_commitment_index(k0_lib_tmp, budget=SearchBudget(20_000), tolerance=1e-7)
    k0_comms = list(k0_lib_tmp._premise_commitment_index.commitments)

    configs = [
        ("right", "donor_positive_gap", 0.23, 0.89),
        ("left", "donor_positive_gap", 0.31, 0.85),
        ("right", "donor_positive_gap", 0.41, 0.93),
        ("right", "source_greater", None, 0.84),
        ("left", "donor_positive_gap", 0.36, 0.91),
        ("right", "donor_positive_gap", 0.27, 0.86),
    ]
    for idx, sd in enumerate(seeds):
        d_mode, c_mode, c_th, r_val = configs[idx % len(configs)]
        kw: dict[str, Any] = dict(
            seed=sd,
            apparatus_seed=sd + 13,
            objects=6,
            readout="window_total_event",
            variant="exchange",
            relation_direction=d_mode,
            relation_condition=c_mode,
            relation_radius=r_val,
            actuator_curvature=0.15,
            name_seed=sd + 17,
            wiring_seed=sd + 41,
        )
        if c_th is not None:
            kw["condition_threshold"] = c_th

        lib_role = KnowledgeLibrary(meta_priors=deepcopy(cal_prior))
        cfg_role = ResearchConfig(
            seed=sd,
            observation_budget=210,
            compute_budget=110_000,
            minimum_improvement=1e-7,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=False,
            transition_premise_audit=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
        )
        with AnonymousObjectLab(**kw) as lab_disc:
            res_disc = UnifiedResearchAgent(cfg_role, measurement_budget=6600, library=lib_role).run(
                lab_disc
            )
            tot_comp += int(res_disc["costs"]["compute_cost"])
            tot_obs += int(res_disc["costs"]["observation_cost"])

        obj_res = res_disc.get("object_research") or {}
        if not obj_res.get("complete"):
            continue
        sc_dict = obj_res["schema"]
        sc = ObjectSchema.from_dict(sc_dict)
        # Use realistic post-role-confirmation fine batches (~68 rows) matching runtime invocation point!
        base_recs: list[dict] = []
        for b in res_disc.get("batches") or ():
            if tuple(b.get("channels") or ()) == ("fine",):
                for pt, r_d, tr in zip(
                    b["probe"]["sequence"], b["observations"], b.get("transitions") or ()
                ):
                    base_recs.append({"controls": dict(pt), **tr, "target": float(r_d["target"])})
                if len(base_recs) >= 72:
                    break

        aud_before = audit_transition_premises(
            obj_res,
            base_recs,
            SearchBudget(35_000),
            commitments=k0_comms,
            allow_expanded_revisions=True,
            occam_premise_minimality=True,
        )
        q_fwd, q_inv = fit_empirical_q_actuator_maps(base_recs, sc)

        with AnonymousObjectLab(**kw) as lab_probe:
            cmd_spec = lab_probe.spec_for(())
            cand_progs = build_generic_schema_candidate_hypotheses(sc, audit_hint=aud_before)
            menu = generate_candidate_probe_menu(
                cmd_spec,
                sc,
                sc.controls,
                seed=sd + 503,
                probe_length=6,
                audit_hint=aud_before,
                q_inverse_fns=q_inv,
            )
            if guiding_method is not None and guiding_method.get("mode") == "learned_linear_probe_scorer":
                gw = guiding_method.get("feature_weights") or {}
                scored_m = []
                for p in menu:
                    fts = compute_observable_probe_features(
                        p, cand_progs, cmd_spec, sc, sc.controls, q_forward_fns=q_fwd
                    )
                    sc_val = (
                        float(gw.get("pairwise_disagreement_margin", 0.0)) * fts["pairwise_disagreement_margin"]
                        + float(gw.get("ensemble_std", 0.0)) * fts["ensemble_std"]
                        + float(gw.get("disagreement_step_fraction", 0.0)) * fts["disagreement_step_fraction"]
                        + float(gw.get("control_sparsity", 0.0)) * fts["control_sparsity"]
                        + float(gw.get("sign_contrast", 0.0)) * fts["sign_contrast"]
                    )
                    scored_m.append((sc_val, p, fts))
                scored_m.sort(key=lambda item: item[0], reverse=True)
                top_guided = [(p, fts) for _, p, fts in scored_m[:5]]
                contrastive = [
                    (
                        p,
                        compute_observable_probe_features(
                            p, cand_progs, cmd_spec, sc, sc.controls, q_forward_fns=q_fwd
                        ),
                    )
                    for p in (menu[0], menu[4], menu[9], menu[10])
                    if p in menu
                ]
                seen_pids: set[str] = set()
                sampled_items = []
                for p, fts in [*top_guided, *contrastive]:
                    if p.probe_id not in seen_pids:
                        seen_pids.add(p.probe_id)
                        sampled_items.append((p, fts))
            else:
                # Sample evenly across random_fullbound, random_subbound, broad fixed-grid pairs, and interval_bisect!
                pick_indices = [0, 1, 4, 7, 9, 10, 15, len(menu) - 3, len(menu) - 2, len(menu) - 1]
                seen_pids = set()
                sampled_items = []
                for pi in pick_indices:
                    if 0 <= pi < len(menu):
                        p = menu[pi]
                        if p.probe_id not in seen_pids:
                            seen_pids.add(p.probe_id)
                            sampled_items.append(
                                (
                                    p,
                                    compute_observable_probe_features(
                                        p, cand_progs, cmd_spec, sc, sc.controls, q_forward_fns=q_fwd
                                    ),
                                )
                            )

            for prb, feats in sampled_items:
                with AnonymousObjectLab(**kw) as fresh_lab:
                    elim_rate = _measure_empirical_elimination_per_row_v11_1(
                        fresh_lab, obj_res, base_recs, prb, k0_comms, aud_before
                    )
                    tot_obs += prb.cost
                    tot_comp += 350
                    rollout_records.append(
                        {
                            "episode_id": f"dev_form_v11_1_{sd}",
                            "probe_id": prb.probe_id,
                            "probe_features": feats,
                            "empirical_elimination_per_row": elim_rate,
                        }
                    )

    return rollout_records, {
        "formation_rollout_compute": tot_comp,
        "formation_rollout_observations": tot_obs,
    }


def _run_episode_with_km_v11_1(
    *,
    task_kw: dict[str, Any],
    cal_prior: list[dict],
    k1_template_lib: KnowledgeLibrary | None,
    ord_archive_cache: OrdinaryPersistentCommitmentCache | None,
    ord_raw_priors: list[dict] | None,
    method_spec: dict[str, Any],
    compute_cap: int = 315_000,
    obs_cap: int = 300,
    meas_cap: int = 6800,
) -> dict[str, Any]:
    seed = int(task_kw["seed"])
    eval_seed = seed + 900001
    acc = UnifiedResourceAccount(
        compute_limit=compute_cap, observation_limit=obs_cap, measurement_limit=meas_cap
    )
    t0 = time.perf_counter()

    if k1_template_lib is not None:
        lib = KnowledgeLibrary(meta_priors=deepcopy(k1_template_lib.meta_priors))
        lib._indexed_prior_fingerprints = deepcopy(k1_template_lib._indexed_prior_fingerprints)
        lib._indexed_group_commitments = deepcopy(k1_template_lib._indexed_group_commitments)
        lib._premise_commitment_index = deepcopy(k1_template_lib._premise_commitment_index)
        lib._index_build_status = k1_template_lib._index_build_status
        lib._indexed_stale_dep_cids = deepcopy(k1_template_lib._indexed_stale_dep_cids)
        lib._generalized_branch_commitments = deepcopy(
            getattr(k1_template_lib, "_generalized_branch_commitments", [])
        )
        lib._active_reconciled_branches = deepcopy(
            getattr(k1_template_lib, "_active_reconciled_branches", [])
        )
        wsc_flag = True
        idx_flag = True
        br_flag = True
    elif ord_archive_cache is not None and ord_raw_priors is not None:
        lib = KnowledgeLibrary(meta_priors=deepcopy(ord_raw_priors))
        lib._ordinary_persistent_cache = deepcopy(ord_archive_cache)
        wsc_flag = True
        idx_flag = False
        br_flag = False
    else:
        lib = KnowledgeLibrary(meta_priors=deepcopy(cal_prior))
        ensure_library_commitment_index(lib, budget=SearchBudget(20_000), tolerance=1e-7)
        wsc_flag = True
        idx_flag = True
        br_flag = False

    m_cfg = deepcopy(method_spec)
    m_cfg.setdefault("max_probes_per_episode", 4)
    m_cfg.setdefault("probe_length", 6)
    lib._learned_experiment_method = m_cfg

    cfg = ResearchConfig(
        seed=seed,
        observation_budget=acc.remaining_observations,
        compute_budget=max(1000, acc.remaining_compute),
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=wsc_flag,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=idx_flag,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=br_flag,
    )
    with AnonymousObjectLab(**task_kw) as lab:
        res = UnifiedResearchAgent(
            cfg, measurement_budget=acc.remaining_measurements, library=lib
        ).run(lab)
        c_used = int(res["costs"]["compute_cost"])
        o_used = int(res["costs"]["observation_cost"])
        m_used = int(res["costs"]["measurement_cost"])
        in_b = acc.request_compute("episode_compute", c_used) and acc.request_observations(
            "episode_obs", o_used, m_used
        )
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        int_conf = bool(res["status"] == "completed")
        ext_mse = (
            float(
                loss(
                    prog,
                    lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1),
                    res["fallback"],
                    0.5,
                )
            )
            if prog is not None
            else float("inf")
        )

    actions = res.get("actions") or []
    method_actions = [
        a for a in actions if a.get("action") == "execute_discriminating_experiment_method"
    ]
    all_cons = extract_real_consumption_events_from_actions(actions)
    m_cons = [
        ce
        for ce in all_cons
        if ce.get("consumption_site") == "select_discriminating_experiment_probe"
    ]
    changed_probes = sum(1 for a in method_actions if a.get("changed_probe_vs_m0"))

    return {
        "strict_pass": bool(in_b and int_conf and ext_mse <= 1e-7),
        "internally_confirmed": int_conf,
        "pure_control_channel_compiled": ch == (),
        "in_unified_budget": in_b,
        "external_fresh48_mse": ext_mse,
        "compute_cost": acc.compute_spent,
        "observation_cost": acc.observation_spent,
        "measurement_cost": acc.measurement_spent,
        "method_invocations": len(method_actions),
        "method_changed_probe_count": changed_probes,
        "method_consumption_events": len(m_cons),
        "total_consumption_events": len(all_cons),
        "wall_seconds": round(time.perf_counter() - t0, 4),
    }


def _build_dev_selection_evaluator_v11_1(cal_prior: list[dict], sel_seeds: list[int]):
    sel_specs = [
        ("right", "donor_positive_gap", 0.28, 0.88),
        ("left", "donor_positive_gap", 0.34, 0.90),
        ("right", "donor_positive_gap", 0.38, 0.86),
        ("left", "donor_positive_gap", 0.25, 0.92),
    ]

    def _eval(cand_method: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for i, sd in enumerate(sel_seeds):
            dm, cm, th, rv = sel_specs[i % len(sel_specs)]
            tkw = dict(
                seed=sd,
                apparatus_seed=sd + 13,
                objects=6,
                readout="window_total_event",
                variant="exchange",
                relation_direction=dm,
                relation_condition=cm,
                condition_threshold=th,
                relation_radius=rv,
                actuator_curvature=0.15,
                name_seed=sd + 23,
                wiring_seed=sd + 59,
            )
            r = _run_episode_with_km_v11_1(
                task_kw=tkw,
                cal_prior=cal_prior,
                k1_template_lib=None,
                ord_archive_cache=None,
                ord_raw_priors=None,
                method_spec=cand_method,
            )
            rows.append(r)
        return {
            "passes": sum(1 for r in rows if r["strict_pass"]),
            "total_episodes": len(rows),
            "mean_observations": round(mean(r["observation_cost"] for r in rows), 2),
            "mean_compute": round(mean(r["compute_cost"] for r in rows), 1),
        }

    return _eval


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-register brand-new unseen evaluation chains (985000..990000), strictly disjoint from v11!
    v11_historic_seeds = {
        961001, 961002, 961003, 961004, 961005, 961006,
        962001, 962002, 962003, 962004,
        963001, 963002, 963003, 963004,
        964001, 964002, 964003, 964004,
        971000, 972000, 973000, 974000, 975000, 976000,
    }
    dev_form_seeds = [981001, 981002, 981003, 981004, 981005, 981006]
    dev_sel_seeds = [982001, 982002, 982003, 982004]
    dev_gen2_form_seeds = [983001, 983002, 983003, 983004]
    dev_gen2_sel_seeds = [984001, 984002, 984003, 984004]
    eval_chain_seeds = [985000, 986000, 987000, 988000, 989000, 990000]

    evaluator_contract = {
        "eval_chain_seeds": eval_chain_seeds,
        "compute_cap": 315_000,
        "obs_cap": 300,
        "meas_cap": 6800,
        "max_probes_per_episode": 4,
        "probe_length": 6,
        "mse_pass_threshold": 1e-7,
        "eval_trajectory_length": 48,
    }
    evaluator_sha256_before = hashlib.sha256(
        json.dumps(evaluator_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cal_prior, cal_comm, cal_formation_compute = _build_shared_calibration_only_prior()

    # =========================================================================
    # STEP 1: Induce M1_v11_1 on disjoint `dev_formation` & `dev_selection`
    # =========================================================================
    rollouts_m1, form_ledger_m1 = _collect_dev_formation_rollouts_v11_1(cal_prior, dev_form_seeds)
    sel_eval_fn = _build_dev_selection_evaluator_v11_1(cal_prior, dev_sel_seeds)
    m1_induction = induce_discriminating_experiment_method(
        rollouts_m1,
        sel_eval_fn,
        generation_tag="M1_v11_1",
        budget=SearchBudget(50_000),
    )
    m0_policy = frozen_default_method_m0()
    m1_policy = deepcopy(m1_induction["promoted_method"])

    # =========================================================================
    # STEP 2: Bounded M1 -> M2 Second-Generation Improver Check (§3.3)
    # =========================================================================
    sel_eval_gen2 = _build_dev_selection_evaluator_v11_1(cal_prior, dev_gen2_sel_seeds)
    rollouts_from_m0, _ = _collect_dev_formation_rollouts_v11_1(
        cal_prior, dev_gen2_form_seeds, guiding_method=m0_policy
    )
    m2_from_m0 = induce_discriminating_experiment_method(
        rollouts_from_m0,
        sel_eval_gen2,
        generation_tag="M2_v11_1_from_M0",
        parent_method_id=m0_policy["method_id"],
        budget=SearchBudget(50_000),
    )

    rollouts_from_m1, _ = _collect_dev_formation_rollouts_v11_1(
        cal_prior, dev_gen2_form_seeds, guiding_method=m1_policy
    )
    m2_from_m1 = induce_discriminating_experiment_method(
        rollouts_from_m1,
        sel_eval_gen2,
        generation_tag="M2_v11_1_from_M1",
        parent_method_id=m1_policy["method_id"],
        budget=SearchBudget(50_000),
    )

    # =========================================================================
    # STEP 3: Single-Shot Unseen Pre-Registered 2x2 Evaluation (985000..990000)
    # =========================================================================
    arm_keys = (
        "A_K0_M0",
        "B_K1_M0",
        "C_K0_M1",
        "D_K1_M1",
        "E_optimized_ordinary_archive_M0",
    )
    chain_reports: list[dict[str, Any]] = []

    for base_sd in eval_chain_seeds:
        cid = f"unseen_km_v11_1_chain_{base_sd}"
        k1_lib, ord_cache, raw_priors, chain_ledger = _build_chain_k1_and_archive(
            cid, base_sd, cal_prior, cal_comm
        )
        targets = [
            {
                "target_id": f"{cid}:T1_right_theta_0p29_N6",
                "seed": base_sd + 801,
                "apparatus_seed": base_sd + 101,
                "objects": 6,
                "readout": "window_total_event",
                "variant": "exchange",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": 0.29,
                "relation_radius": 0.87,
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 1801,
                "wiring_seed": base_sd + 2801,
            },
            {
                "target_id": f"{cid}:T2_left_theta_0p35_N6",
                "seed": base_sd + 802,
                "apparatus_seed": base_sd + 102,
                "objects": 6,
                "readout": "window_total_event",
                "variant": "exchange",
                "relation_direction": "left",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": 0.35,
                "relation_radius": 0.90,
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 1802,
                "wiring_seed": base_sd + 2802,
            },
        ]

        per_arm_targets: dict[str, list[dict[str, Any]]] = {k: [] for k in arm_keys}
        for tgt in targets:
            tid = tgt["target_id"]
            tkw = {k: v for k, v in tgt.items() if k != "target_id"}

            per_arm_targets["A_K0_M0"].append(
                {
                    "target_id": tid,
                    **_run_episode_with_km_v11_1(
                        task_kw=tkw,
                        cal_prior=cal_prior,
                        k1_template_lib=None,
                        ord_archive_cache=None,
                        ord_raw_priors=None,
                        method_spec=m0_policy,
                    ),
                }
            )
            per_arm_targets["B_K1_M0"].append(
                {
                    "target_id": tid,
                    **_run_episode_with_km_v11_1(
                        task_kw=tkw,
                        cal_prior=cal_prior,
                        k1_template_lib=k1_lib,
                        ord_archive_cache=None,
                        ord_raw_priors=None,
                        method_spec=m0_policy,
                    ),
                }
            )
            per_arm_targets["C_K0_M1"].append(
                {
                    "target_id": tid,
                    **_run_episode_with_km_v11_1(
                        task_kw=tkw,
                        cal_prior=cal_prior,
                        k1_template_lib=None,
                        ord_archive_cache=None,
                        ord_raw_priors=None,
                        method_spec=m1_policy,
                    ),
                }
            )
            per_arm_targets["D_K1_M1"].append(
                {
                    "target_id": tid,
                    **_run_episode_with_km_v11_1(
                        task_kw=tkw,
                        cal_prior=cal_prior,
                        k1_template_lib=k1_lib,
                        ord_archive_cache=None,
                        ord_raw_priors=None,
                        method_spec=m1_policy,
                    ),
                }
            )
            per_arm_targets["E_optimized_ordinary_archive_M0"].append(
                {
                    "target_id": tid,
                    **_run_episode_with_km_v11_1(
                        task_kw=tkw,
                        cal_prior=cal_prior,
                        k1_template_lib=None,
                        ord_archive_cache=ord_cache,
                        ord_raw_priors=raw_priors,
                        method_spec=m0_policy,
                    ),
                }
            )

        chain_reports.append(
            {
                "chain_id": cid,
                "base_seed": base_sd,
                "formation_ledger": chain_ledger,
                "arms": per_arm_targets,
            }
        )

    evaluator_sha256_after = hashlib.sha256(
        json.dumps(evaluator_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert evaluator_sha256_before == evaluator_sha256_after

    arm_summaries: dict[str, Any] = {}
    chain_pass_rates_by_arm: dict[str, list[float]] = {}
    chain_obs_by_arm: dict[str, list[float]] = {}

    for ak in arm_keys:
        all_eps = [ep for ch in chain_reports for ep in ch["arms"][ak]]
        c_pass_rates = [
            mean(1.0 if ep["strict_pass"] else 0.0 for ep in ch["arms"][ak])
            for ch in chain_reports
        ]
        c_obs = [mean(ep["observation_cost"] for ep in ch["arms"][ak]) for ch in chain_reports]
        chain_pass_rates_by_arm[ak] = c_pass_rates
        chain_obs_by_arm[ak] = c_obs

        arm_summaries[ak] = {
            "independent_chains": len(chain_reports),
            "total_unseen_target_episodes": len(all_eps),
            "strict_passes": sum(1 for ep in all_eps if ep["strict_pass"]),
            "pure_control_channel_passes": sum(
                1 for ep in all_eps if ep["strict_pass"] and ep["pure_control_channel_compiled"]
            ),
            "failed_episodes": sum(1 for ep in all_eps if not ep["strict_pass"]),
            "uninvoked_method_episodes": sum(1 for ep in all_eps if ep["method_invocations"] == 0),
            "episodes_where_method_changed_probe_vs_m0": sum(
                1 for ep in all_eps if ep["method_changed_probe_count"] > 0
            ),
            "total_method_consumption_events": sum(
                ep["method_consumption_events"] for ep in all_eps
            ),
            "chain_mean_pass_rate": round(float(mean(c_pass_rates)), 4),
            "mean_target_compute": round(float(mean(ep["compute_cost"] for ep in all_eps)), 1),
            "mean_target_observations": round(
                float(mean(ep["observation_cost"] for ep in all_eps)), 2
            ),
            "mean_target_measurements": round(
                float(mean(ep["measurement_cost"] for ep in all_eps)), 2
            ),
            "mean_wall_seconds": round(float(mean(ep["wall_seconds"] for ep in all_eps)), 4),
        }

    def _paired_contrast(arm_hi: str, arm_lo: str) -> dict[str, Any]:
        p_diffs = [
            hi - lo
            for hi, lo in zip(chain_pass_rates_by_arm[arm_hi], chain_pass_rates_by_arm[arm_lo])
        ]
        o_diffs = [
            hi - lo
            for hi, lo in zip(chain_obs_by_arm[arm_hi], chain_obs_by_arm[arm_lo])
        ]
        return {
            "contrast": f"{arm_hi} minus {arm_lo}",
            "pass_rate_paired_diff": _bootstrap_paired_ci_95(p_diffs),
            "observation_cost_paired_diff": _bootstrap_paired_ci_95(o_diffs),
            "per_chain_pass_rate_diffs": [round(x, 4) for x in p_diffs],
        }

    contrasts = {
        "B_minus_A_world_knowledge_at_M0": _paired_contrast("B_K1_M0", "A_K0_M0"),
        "C_minus_A_learned_method_at_K0": _paired_contrast("C_K0_M1", "A_K0_M0"),
        "D_minus_B_learned_method_at_K1": _paired_contrast("D_K1_M1", "B_K1_M0"),
        "D_minus_C_world_knowledge_at_M1": _paired_contrast("D_K1_M1", "C_K0_M1"),
        "B_minus_E_structured_K1_vs_ordinary_archive_at_M0": _paired_contrast(
            "B_K1_M0", "E_optimized_ordinary_archive_M0"
        ),
        "D_minus_E_full_K1_M1_vs_ordinary_archive": _paired_contrast(
            "D_K1_M1", "E_optimized_ordinary_archive_M0"
        ),
    }

    all_v11_1_seeds = (
        set(dev_form_seeds)
        | set(dev_sel_seeds)
        | set(dev_gen2_form_seeds)
        | set(dev_gen2_sel_seeds)
        | set(eval_chain_seeds)
    )
    payload = {
        "schema_version": "hegel_v11_1_deliverable_c_remediated_2x2_km_v1",
        "generated_at_utc": _utc_now(),
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
        "llm_usage": {"model_calls": 0, "total_tokens": 0},
        "remediation_lineage": {
            "preserved_v11_diagnostic_receipt": (
                "docs/evidence/v11_branch_lifecycle_and_km_20261117/"
                "deliverable_c_knowledge_vs_method_2x2_20261117.json"
            ),
            "remediated_root_causes": [
                "Co-located synchronous bystander full-range q/u excitation eliminates curved-actuator calibration starvation without cross-talk.",
                "Public audit-interval bisecting candidate probes + scale-free disagreement_step_fraction replace global fixed preferred_displacement_scale=0.0718 penalty.",
                "dev_formation rollouts start from realistic post-role-confirmation state (~68 rows) matching runtime invocation context.",
            ],
        },
        "partition_hygiene_and_evaluator_immutability": {
            "dev_formation_seeds": dev_form_seeds,
            "dev_selection_seeds": dev_sel_seeds,
            "dev_gen2_formation_seeds": dev_gen2_form_seeds,
            "dev_gen2_selection_seeds": dev_gen2_sel_seeds,
            "unseen_evaluation_chain_seeds": eval_chain_seeds,
            "zero_internal_seed_overlap_verified": (
                len(
                    set(dev_form_seeds) & set(dev_sel_seeds)
                    | (set(dev_form_seeds) | set(dev_sel_seeds)) & set(eval_chain_seeds)
                )
                == 0
            ),
            "zero_overlap_with_v11_971000_976000_verified": (
                len(all_v11_1_seeds & v11_historic_seeds) == 0
            ),
            "evaluator_contract_sha256_before": evaluator_sha256_before,
            "evaluator_contract_sha256_after": evaluator_sha256_after,
            "evaluator_immutable_verified": evaluator_sha256_before == evaluator_sha256_after,
        },
        "cost_ledgers_separated": {
            "neutral_calibration_anchor_compute": cal_formation_compute,
            "m1_dev_formation_rollouts": form_ledger_m1,
            "m1_induction_and_selection_compute": m1_induction["formation_compute_cost"],
            "mean_chain_k1_source_formation_compute": round(
                mean(
                    ch["formation_ledger"]["source_world_formation_compute"]
                    for ch in chain_reports
                ),
                1,
            ),
        },
        "method_induction_M0_to_M1_v11_1": {
            "promoted": m1_induction["promoted"],
            "promoted_method": m1_policy,
            "baseline_m0_dev_selection": m1_induction["baseline_m0_dev_selection"],
            "promoted_dev_selection": m1_induction["promoted_dev_selection"],
            "learned_regression_weights": m1_induction["learned_regression_weights"],
            "learned_preferred_displacement_scale": m1_induction[
                "learned_preferred_displacement_scale"
            ],
            "exploration_archive_count": len(m1_induction["exploration_archive"]),
            "exploration_archive": m1_induction["exploration_archive"],
        },
        "second_generation_improver_check_M1_to_M2_v11_1": {
            "m2_from_m0_dev_selection": m2_from_m0["promoted_dev_selection"],
            "m2_from_m1_dev_selection": m2_from_m1["promoted_dev_selection"],
            "m2_from_m0_weights": m2_from_m0["learned_regression_weights"],
            "m2_from_m1_weights": m2_from_m1["learned_regression_weights"],
        },
        "unseen_2x2_arm_summaries": arm_summaries,
        "paired_chain_contrasts_and_ci95": contrasts,
        "unseen_chain_details": chain_reports,
    }
    out_path = OUT_DIR / "deliverable_c_v11_1_method_remediation_2x2_20261117.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    for ak, av in arm_summaries.items():
        print(
            f"  {ak}: passes={av['strict_passes']}/{av['total_unseen_target_episodes']} "
            f"(chain_rate={av['chain_mean_pass_rate']}, obs={av['mean_target_observations']}, "
            f"comp={av['mean_target_compute']}, m_changed={av['episodes_where_method_changed_probe_vs_m0']})"
        )


if __name__ == "__main__":
    main()
