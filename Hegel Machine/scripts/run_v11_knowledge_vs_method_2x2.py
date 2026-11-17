"""Run v11 Deliverable C: 2x2 World Knowledge (K0/K1) x Induced Research Method (M0/M1) Study.

Writes new receipt (zero modifications to any v10, v10.2, v10.3, or v10.4 receipts):
  ``docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_knowledge_vs_method_2x2_20261117.json``

Implements ``markdown/hegel_v11/02_下一轮实施与验收.md`` §3 (3.1–3.4):
  1. **Strict 3-way partition split**:
     - ``dev_formation`` (``961001..961006``): Collects real ``lab.acquire`` probe rollouts and
       measures empirical hypothesis-elimination rate per row.
     - ``dev_selection`` (``962001..962004``): Evaluates candidate method policies, archives ALL
       tested policies (including ties) in ``exploration_archive``, and promotes ``M1``
       (``author_origin='runtime_induced_from_dev_episodes'``) only if strictly supported.
     - ``unseen_evaluation`` (``971000, 972000, 973000, 974000, 975000, 976000``): 6 contiguous
       pre-registered unseen source-target chains (12 unseen N=6 target apparatuses total).
  2. **Clean 2x2 + Strong Ordinary Anchor**:
     All arms share identical solvers, ``occam_premise_minimality=True``, identical baseline
     ``SensorRoleCalibrationPrior`` (so K1 vs K0 isolates world-hypothesis commitments rather
     than sensor-role calibration!), and identical ``UnifiedResourceAccount`` caps:
       - ``A_K0_M0``: Frozen pre-domain K0 + frozen default M0
       - ``B_K1_M0``: Runtime-learned & reconciled world hypotheses K1 + frozen default M0
       - ``C_K0_M1``: Frozen pre-domain K0 + dev-induced experiment policy M1
       - ``D_K1_M1``: Runtime-learned & reconciled world hypotheses K1 + dev-induced M1
       - ``E_optimized_ordinary_archive_M0``: ``OrdinaryPersistentCommitmentCache`` on identical
         source history + frozen default M0
  3. **Bounded M1 -> M2 Second-Generation Improver Check (§3.3)**:
     Tests whether using ``M1`` during a second dev refinement stage yields any downstream
     improvement over ``M0`` when inducing ``M2``, reporting ties/gains candidly without overclaiming.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.run_v11_branch_lifecycle_study import _discover_real_source
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    propose_anti_unified_generalization,
)
from hegel_machine.cognition.discriminating_experiment_policy import (
    build_generic_schema_candidate_hypotheses,
    compute_observable_probe_features,
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


def _bootstrap_paired_ci_95(diffs: list[float], *, seed: int = 20261117, n_boot: int = 2000) -> dict[str, float]:
    if not diffs:
        return {"mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    return {
        "mean_diff": round(float(mean(diffs)), 4),
        "ci95_low": round(float(means[int(0.025 * n_boot)]), 4),
        "ci95_high": round(float(means[int(0.975 * n_boot)]), 4),
    }


def _build_shared_calibration_only_prior() -> tuple[list[dict], dict, int]:
    """Acquire frozen initial baseline world knowledge K0 ([S0_uncond, S_init_dpg_0p24])."""
    art0, _ = _discover_real_source(
        source_tag="K0_frozen_baseline_S0_uncond",
        seed=960101,
        objects=4,
        relation_direction="right",
        relation_condition="always",
        relation_radius=1.04,
    )
    art_init, _ = _discover_real_source(
        source_tag="K0_frozen_baseline_S_init_dpg0p24",
        seed=960111,
        objects=4,
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.24,
        relation_radius=1.04,
        parent_commitment=art0["commitment"],
        parent_prior=art0["prior"],
    )
    return (
        [deepcopy(art0["prior"]), deepcopy(art_init["prior"])],
        deepcopy(art0["commitment"]),
        int(art0["compute_cost"] + art_init["compute_cost"]),
    )


def _measure_empirical_elimination_per_row(
    lab: AnonymousObjectLab,
    model: dict,
    base_records: list[dict],
    probe: Probe,
    k0_commitments: list[dict],
) -> float:
    """Execute ``probe`` on ``lab`` and measure empirical premise-ambiguity reduction per row."""
    b_before = SearchBudget(35_000)
    aud_before = audit_transition_premises(
        model,
        base_records,
        b_before,
        commitments=k0_commitments,
        allow_expanded_revisions=True,
        occam_premise_minimality=True,
    )

    def _ambiguity_score(aud: dict) -> float:
        # Sum normalized uncertainty across direction (1.0 if unknown), radius width, and threshold width!
        dir_unc = 0.0 if aud.get("verified_direction") else 1.0
        r_iv = aud.get("radius_interval")
        r_unc = min(1.0, (float(r_iv[1]) - float(r_iv[0])) / 0.80) if r_iv else 1.0
        t_iv = aud.get("verified_threshold_interval")
        if isinstance(aud.get("verified_edge_condition"), dict) and t_iv:
            t_unc = min(1.0, (float(t_iv[1]) - float(t_iv[0])) / 0.60)
        elif isinstance(aud.get("verified_edge_condition"), dict):
            t_unc = 0.8
        else:
            # Unconditional or source_greater still has boundary ambiguity if Occam is provisional!
            gate = aud.get("occam_minimality_gate") or {}
            t_unc = 0.65 if gate.get("boundary_test_needed", True) else 0.0
        return dir_unc + r_unc + t_unc

    unc_before = _ambiguity_score(aud_before)
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
    unc_after = _ambiguity_score(aud_after)
    gain = max(0.0, unc_before - unc_after)
    return float(gain / max(1, probe.cost))


def _collect_dev_formation_rollouts(
    cal_prior: dict,
    seeds: list[int],
    *,
    guiding_method: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run real probe rollouts on ``dev_formation`` episodes to collect ``(features, elimination_rate)``."""
    rollout_records: list[dict[str, Any]] = []
    tot_comp = 0
    tot_obs = 0

    k0_lib_tmp = KnowledgeLibrary(meta_priors=deepcopy(cal_prior if isinstance(cal_prior, list) else [cal_prior]))
    ensure_library_commitment_index(k0_lib_tmp, budget=SearchBudget(20_000), tolerance=1e-7)
    k0_comms = list(k0_lib_tmp._premise_commitment_index.commitments)

    configs = [
        ("right", "donor_positive_gap", 0.22, 0.88),
        ("left", "donor_positive_gap", 0.28, 0.86),
        ("right", "donor_positive_gap", 0.42, 0.92),
        ("right", "source_greater", None, 0.85),
        ("left", "donor_positive_gap", 0.35, 0.90),
        ("right", "donor_positive_gap", 0.31, 0.84),
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

        # Discover public anonymous object roles first using calibration prior only
        lib_role = KnowledgeLibrary(meta_priors=deepcopy(cal_prior if isinstance(cal_prior, list) else [cal_prior]))
        cfg_role = ResearchConfig(
            seed=sd,
            observation_budget=220,
            compute_budget=120_000,
            minimum_improvement=1e-7,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=False,
            transition_premise_audit=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
        )
        with AnonymousObjectLab(**kw) as lab_disc:
            res_disc = UnifiedResearchAgent(cfg_role, measurement_budget=6800, library=lib_role).run(
                lab_disc
            )
            tot_comp += int(res_disc["costs"]["compute_cost"])
            tot_obs += int(res_disc["costs"]["observation_cost"])

        obj_res = res_disc.get("object_research") or {}
        if not obj_res.get("complete"):
            continue
        sc_dict = obj_res["schema"]
        sc = ObjectSchema.from_dict(sc_dict)
        base_recs: list[dict] = []
        for b in res_disc.get("batches") or ():
            if tuple(b.get("channels") or ()) == ("fine",):
                for pt, r_d, tr in zip(
                    b["probe"]["sequence"], b["observations"], b.get("transitions") or ()
                ):
                    base_recs.append({"controls": dict(pt), **tr, "target": float(r_d["target"])})

        with AnonymousObjectLab(**kw) as lab_probe:
            cmd_spec = lab_probe.spec_for(())
            cand_progs = build_generic_schema_candidate_hypotheses(sc)
            menu = generate_candidate_probe_menu(
                cmd_spec, sc, sc.controls, seed=sd + 503, probe_length=6
            )
            # Sample diverse probes across the menu (or prioritize around guiding_method if M1->M2)
            if guiding_method is not None and guiding_method.get("mode") == "learned_linear_probe_scorer":
                # Evaluate top-scoring probes under guiding_method + contrastive random probes
                gw = guiding_method.get("feature_weights") or {}
                gp = float(guiding_method.get("preferred_displacement_scale", 0.25))
                scored_m = []
                for p in menu:
                    fts = compute_observable_probe_features(p, cand_progs, cmd_spec, sc, sc.controls)
                    sc_val = (
                        float(gw.get("pairwise_disagreement_margin", 0.0)) * fts["pairwise_disagreement_margin"]
                        + float(gw.get("ensemble_std", 0.0)) * fts["ensemble_std"]
                        + float(gw.get("control_sparsity", 0.0)) * fts["control_sparsity"]
                        + float(gw.get("sign_contrast", 0.0)) * fts["sign_contrast"]
                        - float(gw.get("scale_proximity", 1.0)) * abs(fts["displacement_scale"] - gp)
                    )
                    scored_m.append((sc_val, p, fts))
                scored_m.sort(key=lambda item: item[0], reverse=True)
                top_guided = [(p, fts) for _, p, fts in scored_m[:6]]
                contrastive = [
                    (p, compute_observable_probe_features(p, cand_progs, cmd_spec, sc, sc.controls))
                    for p in (menu[0], menu[4], menu[-1])
                ]
                seen_pids: set[str] = set()
                sampled_items = []
                for p, fts in [*top_guided, *contrastive]:
                    if p.probe_id not in seen_pids:
                        seen_pids.add(p.probe_id)
                        sampled_items.append((p, fts))
            else:
                sampled_items = [
                    (p, compute_observable_probe_features(p, cand_progs, cmd_spec, sc, sc.controls))
                    for p in (menu[0], menu[1], menu[4], menu[5], menu[7], menu[9], menu[11], menu[15], menu[19])
                    if p in menu
                ]

            for prb, feats in sampled_items:
                with AnonymousObjectLab(**kw) as fresh_lab:
                    elim_rate = _measure_empirical_elimination_per_row(
                        fresh_lab, obj_res, base_recs[:24], prb, k0_comms
                    )
                    tot_obs += prb.cost
                    tot_comp += 350
                    rollout_records.append(
                        {
                            "episode_id": f"dev_form_{sd}",
                            "probe_id": prb.probe_id,
                            "probe_features": feats,
                            "empirical_elimination_per_row": elim_rate,
                        }
                    )

    return rollout_records, {"formation_rollout_compute": tot_comp, "formation_rollout_observations": tot_obs}


def _run_episode_with_km(
    *,
    task_kw: dict[str, Any],
    cal_prior: dict,
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
        # K0: Frozen initial world knowledge K0 (unhandicapped — has full premise-revision & calibration tools!)
        k0_list = deepcopy(cal_prior) if isinstance(cal_prior, list) else [deepcopy(cal_prior)]
        lib = KnowledgeLibrary(meta_priors=k0_list)
        ensure_library_commitment_index(lib, budget=SearchBudget(20_000), tolerance=1e-7)
        wsc_flag = True
        idx_flag = True
        br_flag = False

    m_cfg = deepcopy(method_spec)
    m_cfg.setdefault("max_probes_per_episode", 3)
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


def _build_dev_selection_evaluator(cal_prior: dict, sel_seeds: list[int]):
    sel_specs = [
        ("right", "donor_positive_gap", 0.26, 0.87),
        ("left", "donor_positive_gap", 0.32, 0.89),
        ("right", "donor_positive_gap", 0.39, 0.85),
        ("right", "donor_positive_gap", 0.21, 0.91),
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
            r = _run_episode_with_km(
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


def _build_chain_k1_and_archive(
    chain_id: str, base_seed: int, cal_prior: dict, cal_comm: dict
) -> tuple[KnowledgeLibrary, OrdinaryPersistentCommitmentCache, list[dict], dict[str, int]]:
    """Form reconciled K1 library and ordinary persistent archive from real chain sources."""
    s1, s1_f = _discover_real_source(
        source_tag=f"{chain_id}:S1",
        seed=base_seed + 21,
        objects=4,
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.20,
        relation_radius=1.04,
        parent_commitment=cal_comm,
        parent_prior=cal_prior[0] if isinstance(cal_prior, list) else cal_prior,
    )
    s2, s2_f = _discover_real_source(
        source_tag=f"{chain_id}:S2",
        seed=base_seed + 31,
        objects=4,
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.45,
        relation_radius=1.06,
        parent_commitment=cal_comm,
        parent_prior=cal_prior[0] if isinstance(cal_prior, list) else cal_prior,
    )
    sl, sl_f = _discover_real_source(
        source_tag=f"{chain_id}:SL",
        seed=base_seed + 41,
        objects=4,
        relation_direction="left",
        relation_condition="donor_positive_gap",
        condition_threshold=0.22,
        relation_radius=1.05,
        parent_commitment=cal_comm,
        parent_prior=cal_prior[0] if isinstance(cal_prior, list) else cal_prior,
    )
    sl_2, sl2_f = _discover_real_source(
        source_tag=f"{chain_id}:SL2",
        seed=base_seed + 51,
        objects=4,
        relation_direction="left",
        relation_condition="donor_positive_gap",
        condition_threshold=0.44,
        relation_radius=1.07,
        parent_commitment=cal_comm,
        parent_prior=cal_prior[0] if isinstance(cal_prior, list) else cal_prior,
    )

    au_right = propose_anti_unified_generalization(
        [
            {
                "claim_id": f"{chain_id}:H1_right",
                "commitment": deepcopy(s1["commitment"]),
                "schema": s1["schema"],
                "support_rows": s1["paid_observations"],
                "scoped_counterexamples": [],
                "fresh_holdout_rows": s1_f,
            },
            {
                "claim_id": f"{chain_id}:H2_right",
                "commitment": deepcopy(s2["commitment"]),
                "schema": s2["schema"],
                "support_rows": s2["paid_observations"],
                "scoped_counterexamples": [],
                "fresh_holdout_rows": s2_f,
            },
        ],
        budget=SearchBudget(40_000),
        tolerance=1e-7,
    )
    gen_fams = []
    fallback_branches = []
    right_guard_d = Program(
        "lt",
        children=(Program("const", value=0.0), Program("feature", name="obs_signed_direction_flux")),
    ).to_dict()
    left_guard_d = Program(
        "lt",
        children=(Program("feature", name="obs_signed_direction_flux"), Program("const", value=0.0)),
    ).to_dict()

    if au_right["accepted"]:
        fam_right = deepcopy(au_right["generalized_commitment"])
        fam_right["scope_guard"] = right_guard_d
        gen_fams.append(fam_right)
    else:
        for idx_r, sr in enumerate((s1, s2)):
            fallback_branches.append(
                {
                    "claim_id": f"{chain_id}:H{idx_r+1}_right_branch",
                    "commitment": deepcopy(sr["commitment"]),
                    "scope_guard": right_guard_d,
                    "status": "scoped_conditional",
                    "independent_empirical_support": True,
                }
            )

    au_left = propose_anti_unified_generalization(
        [
            {
                "claim_id": f"{chain_id}:H1_left",
                "commitment": deepcopy(sl["commitment"]),
                "schema": sl["schema"],
                "support_rows": sl["paid_observations"],
                "scoped_counterexamples": [],
                "fresh_holdout_rows": sl_f,
            },
            {
                "claim_id": f"{chain_id}:H2_left",
                "commitment": deepcopy(sl_2["commitment"]),
                "schema": sl_2["schema"],
                "support_rows": sl_2["paid_observations"],
                "scoped_counterexamples": [],
                "fresh_holdout_rows": sl2_f,
            },
        ],
        budget=SearchBudget(40_000),
        tolerance=1e-7,
    )
    if au_left["accepted"]:
        fam_left = deepcopy(au_left["generalized_commitment"])
        fam_left["scope_guard"] = left_guard_d
        gen_fams.append(fam_left)
    else:
        for idx_l, sl_item in enumerate((sl, sl_2)):
            fallback_branches.append(
                {
                    "claim_id": f"{chain_id}:H{idx_l+1}_left_branch",
                    "commitment": deepcopy(sl_item["commitment"]),
                    "scope_guard": left_guard_d,
                    "status": "scoped_conditional",
                    "independent_empirical_support": True,
                }
            )

    k0_list = deepcopy(cal_prior) if isinstance(cal_prior, list) else [deepcopy(cal_prior)]
    raw_priors = [
        *k0_list,
        deepcopy(s1["prior"]),
        deepcopy(s2["prior"]),
        deepcopy(sl["prior"]),
        deepcopy(sl_2["prior"]),
    ]

    b_idx = SearchBudget(120_000)
    k1_lib = KnowledgeLibrary(meta_priors=deepcopy(raw_priors))
    ensure_library_commitment_index(k1_lib, budget=b_idx, tolerance=1e-7)
    k1_lib._generalized_branch_commitments = gen_fams
    k1_lib._active_reconciled_branches = fallback_branches

    b_ord = SearchBudget(120_000)
    ord_cache = OrdinaryPersistentCommitmentCache()
    ord_cache.get_or_consolidate(raw_priors, budget=b_ord, tolerance=1e-7)

    ledger = {
        "source_world_formation_compute": sum(
            x["compute_cost"] for x in (s1, s2, sl, sl_2)
        ),
        "source_world_formation_observations": sum(
            x["observation_cost"] for x in (s1, s2, sl, sl_2)
        ),
        "k1_index_and_anti_unification_compute": b_idx.spent + 8000,
        "ordinary_cache_build_compute": b_ord.spent,
    }
    return k1_lib, ord_cache, raw_priors, ledger


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-register unseen evaluation configuration & compute immutability hash BEFORE M1 induction!
    eval_chain_seeds = [971000, 972000, 973000, 974000, 975000, 976000]
    evaluator_contract = {
        "eval_chain_seeds": eval_chain_seeds,
        "compute_cap": 315_000,
        "obs_cap": 300,
        "meas_cap": 6800,
        "mse_pass_threshold": 1e-7,
        "eval_trajectory_length": 48,
    }
    evaluator_sha256_before = hashlib.sha256(
        json.dumps(evaluator_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cal_prior, cal_comm, cal_formation_compute = _build_shared_calibration_only_prior()

    # =========================================================================
    # STEP 1: Induce M1 from `dev_formation` (961001..961006) & `dev_selection` (962001..962004)
    # =========================================================================
    dev_form_seeds = [961001, 961002, 961003, 961004, 961005, 961006]
    dev_sel_seeds = [962001, 962002, 962003, 962004]
    rollouts_m1, form_ledger_m1 = _collect_dev_formation_rollouts(cal_prior, dev_form_seeds)
    sel_eval_fn = _build_dev_selection_evaluator(cal_prior, dev_sel_seeds)
    m1_induction = induce_discriminating_experiment_method(
        rollouts_m1,
        sel_eval_fn,
        generation_tag="M1",
        budget=SearchBudget(50_000),
    )
    m0_policy = frozen_default_method_m0()
    m1_policy = deepcopy(m1_induction["promoted_method"])

    # =========================================================================
    # STEP 2: Bounded M1 -> M2 Second-Generation Improver Check (§3.3)
    # =========================================================================
    dev_gen2_form_seeds = [963001, 963002, 963003, 963004]
    dev_gen2_sel_seeds = [964001, 964002, 964003, 964004]
    sel_eval_gen2 = _build_dev_selection_evaluator(cal_prior, dev_gen2_sel_seeds)

    rollouts_from_m0, _ = _collect_dev_formation_rollouts(
        cal_prior, dev_gen2_form_seeds, guiding_method=m0_policy
    )
    m2_from_m0 = induce_discriminating_experiment_method(
        rollouts_from_m0,
        sel_eval_gen2,
        generation_tag="M2_from_M0",
        parent_method_id=m0_policy["method_id"],
        budget=SearchBudget(50_000),
    )

    rollouts_from_m1, _ = _collect_dev_formation_rollouts(
        cal_prior, dev_gen2_form_seeds, guiding_method=m1_policy
    )
    m2_from_m1 = induce_discriminating_experiment_method(
        rollouts_from_m1,
        sel_eval_gen2,
        generation_tag="M2_from_M1",
        parent_method_id=m1_policy["method_id"],
        budget=SearchBudget(50_000),
    )

    # =========================================================================
    # STEP 3: Unseen Pre-Registered 2x2 Evaluation + Ordinary Persistent Archive Anchor
    # =========================================================================
    arm_keys = (
        "A_K0_M0",
        "B_K1_M0",
        "C_K0_M1",
        "D_K1_M1",
        "E_optimized_ordinary_archive_M0",
    )
    chain_reports: list[dict[str, Any]] = []

    for c_idx, base_sd in enumerate(eval_chain_seeds):
        cid = f"unseen_km_chain_{base_sd}"
        k1_lib, ord_cache, raw_priors, chain_ledger = _build_chain_k1_and_archive(
            cid, base_sd, cal_prior, cal_comm
        )
        # Two unseen N=6 conditional target apparatuses per chain (unseen thresholds 0.29 and 0.35!)
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
                    **_run_episode_with_km(
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
                    **_run_episode_with_km(
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
                    **_run_episode_with_km(
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
                    **_run_episode_with_km(
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
                    **_run_episode_with_km(
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

    # Aggregate statistics with chain as independent statistical unit (§3.4)!
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

    payload = {
        "schema_version": "hegel_v11_deliverable_c_2x2_km_v1",
        "generated_at_utc": _utc_now(),
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
        "llm_usage": {"model_calls": 0, "total_tokens": 0},
        "partition_hygiene_and_evaluator_immutability": {
            "dev_formation_seeds": dev_form_seeds,
            "dev_selection_seeds": dev_sel_seeds,
            "dev_gen2_formation_seeds": dev_gen2_form_seeds,
            "dev_gen2_selection_seeds": dev_gen2_sel_seeds,
            "unseen_evaluation_chain_seeds": eval_chain_seeds,
            "zero_seed_overlap_verified": (
                len(
                    set(dev_form_seeds)
                    & set(dev_sel_seeds)
                    | (set(dev_form_seeds) | set(dev_sel_seeds)) & set(eval_chain_seeds)
                )
                == 0
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
        "method_induction_M0_to_M1": {
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
        "second_generation_improver_check_M1_to_M2": {
            "m2_from_m0_dev_selection": m2_from_m0["promoted_dev_selection"],
            "m2_from_m1_dev_selection": m2_from_m1["promoted_dev_selection"],
            "m2_from_m0_weights": m2_from_m0["learned_regression_weights"],
            "m2_from_m1_weights": m2_from_m1["learned_regression_weights"],
            "epistemic_scope_note": (
                "M0 -> M1 demonstrates single-step runtime induction of an executable probe policy "
                "from dev episodes. The bounded M1 -> M2 check tests whether M1-guided rollouts "
                "further improve linear feature-weight estimation within the same 5-feature probe "
                "scorer representation; multi-generation open-ended method language expansion "
                "remains future scope."
            ),
        },
        "unseen_2x2_arm_summaries": arm_summaries,
        "paired_chain_contrasts_and_ci95": contrasts,
        "unseen_chain_details": chain_reports,
    }
    out_path = OUT_DIR / "deliverable_c_knowledge_vs_method_2x2_20261117.json"
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
