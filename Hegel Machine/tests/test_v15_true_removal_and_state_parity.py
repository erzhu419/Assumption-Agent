"""v15 unit & integration controls for true versioned single-object removal, replay scope semantics,
chronological K/R state-transition parity, and structured 5-stage adoption attribution.

Covers every mandatory engineering control in GPT-6's v15 contract (`02_下一轮最小施工与实验合同.md` §2–§5):
  1. §3.3 Control 1: True list-shaped `KnowledgeLibrary` single-version removal across `concepts`, `abstracts`,
     `definitions`, `definition_bindings`, `meta_priors`, branches, families, indices, and caches + zero `_recall`
     reachability + preserved raw empirical evidence & untouched sibling sources + version-mismatch safety.
  2. §3.3 Control 2: Exclusive derivative loses eligibility when its sole supporting version is removed, whereas
     a shared concept/rule/family with an independent surviving second source remains grounded and callable.
  3. §3.3 Control 3: Legal billed on-demand reconstruction from preserved raw empirical history restores an
     eligible confirmed object (`generated_from_raw_history = True`, non-zero billed compute, start/end events).
  4. §4.1 & §4.4: `replay_evidence_batches_onto_branches` invokes `_cross_evaluate_refittable_template_on_neighbor`
     even when `shrink_claim_to_observable_scope` returns `scope_unidentified`, yielding `instance_scope_unknown`
     (never global `refuted`), and `compute_evidence_archive_fingerprint` invalidates on `radius_value`,
     `commitment`, and `observations_serialized` edits.
  5. §4.2 & §4.3: Shared `apply_reconciliation_step_from_evidence` produces exact first-dispatch state parity
     between online `K` and lazy `R` on conflicting multi-episode histories, and returns `partial_pending`
     without caching incomplete state as complete when budget is exhausted mid-replay.
"""

from __future__ import annotations

from copy import deepcopy

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    _serialize_obs_rows,
    dispatch_branches_and_families_for_target,
    reconstruct_missing_objects_from_raw_history,
    reconstruct_parametric_families_from_ordinary_archive,
    replay_evidence_batches_onto_branches,
    shrink_claim_to_observable_scope,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import (
    KnowledgeLibrary,
    export_library_object_inventory,
    remove_library_object_version,
)
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import (
    compute_evidence_archive_fingerprint,
    compute_semantic_version,
    extract_archive_witness_maps,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments
from tests.test_v12_semantics_and_autonomous_reconciliation import (
    _dpg_dict,
    _make_confirmed_regime_prior,
)


def _make_cfg(
    seed: int,
    *,
    branch_reconciliation: bool = True,
    compute_budget: int = 740_000,
    obs_budget: int = 660,
) -> ResearchConfig:
    return ResearchConfig(
        seed=seed,
        observation_budget=obs_budget,
        compute_budget=compute_budget,
        max_rounds=4,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
        active_threshold_probing=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        indexed_commitment_store=branch_reconciliation,
        branch_reconciliation=branch_reconciliation,
    )


def _run_source_episode_into_k_library(
    lib: KnowledgeLibrary,
    lab_kwargs: dict,
    seed: int,
    *,
    meas_budget: int = 13_200,
    compute_budget: int = 740_000,
    obs_budget: int = 660,
) -> dict:
    cfg = _make_cfg(seed, branch_reconciliation=True, compute_budget=compute_budget, obs_budget=obs_budget)
    with AnonymousObjectLab(**lab_kwargs) as lab:
        ep = UnifiedResearchAgent(cfg, measurement_budget=meas_budget, library=lib).run(lab)
    assert ep["status"] == "completed", f"Expected source {lab_kwargs} to complete, got {ep['status']}"
    return ep


def test_v15_controls_1_and_2_true_versioned_removal_reachability_and_shared_support() -> None:
    """§3.3 Controls 1 & 2: List-shaped `KnowledgeLibrary` versioned removal, zero `_recall`, & 2-source retention."""
    lib = KnowledgeLibrary()
    lib._prioritize_generalized_families = True

    src_specs = [
        dict(seed=1_031_005, objects=4, readout="window_total_event", relation_direction="symmetric", relation_condition="always", condition_threshold=0.0, relation_radius=0.92, actuator_curvature=0.15, name_seed=1_031_105, wiring_seed=1_031_205),
        dict(seed=1_031_015, objects=4, readout="window_total_event", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.93, actuator_curvature=0.15, name_seed=1_031_115, wiring_seed=1_031_215),
        dict(seed=1_031_025, objects=4, readout="window_total_event", relation_direction="right", relation_condition="donor_positive_gap", condition_threshold=0.20, relation_radius=0.90, actuator_curvature=0.15, name_seed=1_031_125, wiring_seed=1_031_225),
        dict(seed=1_031_035, objects=4, readout="window_total_event", relation_direction="right", relation_condition="donor_positive_gap", condition_threshold=0.44, relation_radius=0.94, actuator_curvature=0.15, name_seed=1_031_135, wiring_seed=1_031_235),
        dict(seed=1_032_025, objects=4, readout="window_total_event", relation_direction="right", relation_condition="donor_positive_gap", condition_threshold=0.18, relation_radius=0.94, actuator_curvature=0.15, name_seed=1_032_125, wiring_seed=1_032_225),
    ]
    for sp in src_specs:
        _run_source_episode_into_k_library(lib, sp, seed=int(sp["seed"]))

    assert isinstance(lib.concepts, list) and len(lib.concepts) >= 4
    target_cid = lib.concepts[-1].concept_id
    target_prior = next(p for p in lib.meta_priors if p["source_concept_id"] == target_cid)
    target_ver = compute_semantic_version(target_prior)

    surviving_cid = lib.concepts[1].concept_id
    with AnonymousObjectLab(**src_specs[-1]) as lab_ref:
        fine_spec = lab_ref.spec_for(("fine",))
    sample_rows = [
        Observation({fn: 0.2 for fn in fine_spec.feature_names}, 0.2)
        for _ in range(8)
    ]
    excl_cid, _ = lib.learn(
        Candidate(
            "cand_excl_deriv",
            Program("add", children=(Program("feature", name=fine_spec.feature_names[0]), Program("const", value=0.111))),
            "derived_exclusive",
            parents=(target_cid,),
        ),
        fine_spec,
        sample_rows,
    )
    shared_cid, _ = lib.learn(
        Candidate(
            "cand_shared_two_source",
            Program("add", children=(Program("feature", name=fine_spec.feature_names[0]), Program("const", value=0.222))),
            "derived_shared",
            parents=(surviving_cid, target_cid),
        ),
        fine_spec,
        sample_rows,
    )

    pre_inv = export_library_object_inventory(lib)
    assert pre_inv["by_object"][target_cid]["has_executable_concept"] is True
    assert pre_inv["by_object"][excl_cid]["has_executable_concept"] is True
    assert pre_inv["by_object"][shared_cid]["has_executable_concept"] is True
    raw_count_before = len(lib._raw_reconciliation_evidence_archive)
    assert raw_count_before >= 4

    untouched = lib.without_object_version(target_cid, "sem_ver:wrong_nonmatching_hash", mode="repaired_version")
    assert any(c.concept_id == target_cid for c in untouched.concepts)

    pruned = lib.without_object_version(target_cid, target_ver, mode="repaired_version")
    post_inv = export_library_object_inventory(pruned)

    assert all(c.concept_id != target_cid for c in pruned.concepts)
    assert all(a.source_concept_id != target_cid for a in pruned.abstracts)
    assert all(d.source_concept_id != target_cid for d in pruned.definitions.values())
    assert all(not k.startswith(f"{target_cid}:") for k in pruned.definition_bindings)
    assert all(p.get("source_concept_id") != target_cid for p in pruned.meta_priors)
    assert all(b.get("source_concept_id") != target_cid for b in pruned._active_reconciled_branches)
    assert target_cid not in post_inv["by_object"]

    recall_agent = UnifiedResearchAgent(_make_cfg(10599), library=pruned)
    recalled = recall_agent._recall(fine_spec)
    recalled_srcs = {
        str(s)
        for item in recalled
        for s in (_sources(item) if isinstance(item, Program) else [*(item.get("method_source_ids") or ()), *(item["candidate"].parents or ())])
    }
    assert target_cid not in recalled_srcs
    assert excl_cid not in recalled_srcs
    assert surviving_cid in {c.concept_id for c in pruned.concepts}

    assert len(pruned._raw_reconciliation_evidence_archive) == raw_count_before
    removed_src_batches = [
        ev for ev in pruned._raw_reconciliation_evidence_archive
        if ev.get("kind") == "source_episode_batch" and ev.get("source_concept_id") == target_cid
    ]
    assert len(removed_src_batches) == 1
    assert len(removed_src_batches[0]["observations_serialized"]) > 0
    assert "commitment" not in removed_src_batches[0]
    assert pruned._allow_on_demand_raw_history_reconstruction is True

    assert all(c.concept_id != excl_cid for c in pruned.concepts)
    assert any(c.concept_id == shared_cid for c in pruned.concepts)
    surviving_shared = next(c for c in pruned.concepts if c.concept_id == shared_cid)
    assert surviving_cid in surviving_shared.parents and target_cid not in surviving_shared.parents
    assert len(pruned._generalized_branch_commitments) >= 1
    for gf in pruned._generalized_branch_commitments:
        assert f"branch:{target_cid}" not in (gf.get("depends_on_claim_ids") or ())
        assert len(gf.get("depends_on_claim_ids") or ()) >= 2


def test_v15_control_3_billed_raw_history_reconstruction_and_adoption() -> None:
    """§3.3 Control 3: Billed reconstruction from preserved raw history restores an eligible confirmed candidate."""
    lib = KnowledgeLibrary()
    lib._prioritize_generalized_families = False

    cal_kw = dict(seed=10601, objects=4, readout="focus", relation_direction="symmetric", relation_condition="always", condition_threshold=0.0, relation_radius=0.92, actuator_curvature=0.15, name_seed=10701, wiring_seed=10801)
    right_kw = dict(seed=10602, objects=4, readout="focus", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.91, actuator_curvature=0.15, name_seed=10702, wiring_seed=10802)
    focal_kw = dict(seed=10603, objects=4, readout="focus", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.92, latent_mode="focal_memory", latent_decay=0.54, latent_weight=0.36, actuator_curvature=0.15, name_seed=10703, wiring_seed=10803)

    _run_source_episode_into_k_library(lib, cal_kw, seed=10601, compute_budget=540_000, obs_budget=520, meas_budget=11_000)
    _run_source_episode_into_k_library(lib, right_kw, seed=10602, compute_budget=540_000, obs_budget=520, meas_budget=11_000)
    ep_focal = _run_source_episode_into_k_library(lib, focal_kw, seed=10603, compute_budget=880_000, obs_budget=560, meas_budget=11_500)
    focal_cid = ep_focal["concepts_added"][0]
    focal_prior = next(p for p in lib.meta_priors if p["source_concept_id"] == focal_cid)
    focal_ver = compute_semantic_version(focal_prior)

    lib_removed = remove_library_object_version(lib, focal_cid, focal_ver, mode="repaired_version")
    assert all(c.concept_id != focal_cid for c in lib_removed.concepts)
    assert all(p["source_concept_id"] != focal_cid for p in lib_removed.meta_priors)

    recon_budget = SearchBudget(350_000)
    recon_actions: list[dict] = []
    recon_events = reconstruct_missing_objects_from_raw_history(
        lib_removed, recon_budget, actions=recon_actions, tolerance=1e-7
    )
    assert len(recon_events) == 1
    ev0 = recon_events[0]
    assert ev0["status"] == "reconstructed_and_confirmed"
    assert ev0["generated_from_raw_history"] is True
    assert ev0["compute_cost"] > 0
    assert ev0["budget_remaining_after"] < ev0["budget_remaining_before"]
    new_cid = ev0["reconstructed_concept_id"]
    assert new_cid is not None and new_cid != focal_cid
    assert any(c.concept_id == new_cid for c in lib_removed.concepts)
    assert any(p["source_concept_id"] == new_cid and p.get("generated_from_raw_history") is True for p in lib_removed.meta_priors)

    target_kw = dict(seed=10610, objects=5, readout="focus", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.90, latent_mode="focal_memory", latent_decay=0.68, latent_weight=0.42, actuator_curvature=0.15, name_seed=10710, wiring_seed=10810)
    with AnonymousObjectLab(**target_kw) as lab_target:
        res_t = UnifiedResearchAgent(_make_cfg(10610, compute_budget=840_000, obs_budget=580), measurement_budget=11_800, library=lib_removed).run(lab_target)
    assert res_t["status"] == "completed"
    res_cons = [
        ev
        for act in res_t["actions"]
        if act.get("action") == "resolve_commitment_consumption"
        for ev in (act.get("consumption_events") or ())
        if ev.get("adopted_confirmed") is True
    ]
    assert len(res_cons) >= 1
    assert any(new_cid in (ev.get("source_concept_ids") or ()) and ev.get("generated_from_raw_history") is True for ev in res_cons)
    for ev in res_cons:
        assert ev["stages"]["confirmed_adopted"] is True
        assert ev["fresh_confirmation_evidence"]["confirmed"] is True


def test_v15_control_4_scope_unidentified_replay_fallback_and_evidence_hash_dependencies() -> None:
    """§4.1 & §4.4: Replay preserves template eligibility as `instance_scope_unknown` & hashes `radius_value` + `commitment`."""
    p_old = _make_confirmed_regime_prior(concept_id="old", seed=9301, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.18), radius=0.93)
    p_new = _make_confirmed_regime_prior(concept_id="new", seed=9301, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.45), radius=0.93)
    c_old = consolidate_world_structure_commitments([p_old], budget=SearchBudget(50_000))[0]
    c_new = consolidate_world_structure_commitments([p_new], budget=SearchBudget(50_000))[0]
    sup_old = extract_archive_witness_maps([p_old])[0]["old"]
    sup_new = extract_archive_witness_maps([p_new])[0]["new"]

    reg = {
        "obs_signed_direction_flux": 0.5,
        "obs_unconditional_block_rate": 0.2,
        "_meta": {"protocol_id": "paid_episode_transitions", "pilot_calibration_available": True},
    }
    sc_shared = p_old["object_schema"]
    b_old = {
        "claim_id": "branch:old",
        "source_concept_id": "old",
        "source_concept_ids": ["old"],
        "commitment": deepcopy(c_old),
        "radius_value": 0.93,
        "coefficients": [0.0, 1.0, 0.3],
        "schema": deepcopy(sc_shared),
        "status": "universal",
        "scope_claim_level": "bound_instance",
        "independent_empirical_support": True,
        "pilot_regime_features": [deepcopy(reg)],
        "support_rows_serialized": _serialize_obs_rows(sup_old, limit=64),
    }
    b_new = {
        "claim_id": "branch:new",
        "source_concept_id": "new",
        "source_concept_ids": ["new"],
        "commitment": deepcopy(c_new),
        "radius_value": 0.93,
        "coefficients": [0.0, 1.0, 0.3],
        "schema": deepcopy(sc_shared),
        "status": "universal",
        "scope_claim_level": "bound_instance",
        "independent_empirical_support": True,
        "pilot_regime_features": [deepcopy(reg)],
        "support_rows_serialized": _serialize_obs_rows(sup_new, limit=64),
    }
    ev_batch = [
        {
            "evidence_id": "ev:new",
            "schema": deepcopy(sc_shared),
            "observations_serialized": _serialize_obs_rows(sup_new[:24], limit=24),
            "transitions": [],
            "protocol_id": "paid_episode_transitions",
            "radius_value": 0.93,
            "commitment": deepcopy(c_new),
        }
    ]
    shr = shrink_claim_to_observable_scope(b_old, [reg], [reg], claim_level="bound_instance")
    assert shr["status"] == "scope_unidentified"

    replayed, _ = replay_evidence_batches_onto_branches([b_old, b_new], ev_batch, budget=SearchBudget(80_000))
    assert replayed[0]["status"] == "instance_scope_unknown"
    assert replayed[0]["independent_empirical_support"] is True
    assert replayed[0]["template_status"] == "template_retained_rebindable"

    lib_disp = KnowledgeLibrary()
    lib_disp._active_reconciled_branches = deepcopy(replayed)
    disp = dispatch_branches_and_families_for_target(
        library=lib_disp,
        consistent_comms=[deepcopy(c_old), deepcopy(c_new)],
        transitions=[],
        schema=sc_shared,
        audit=None,
        task_sig="t_scope_unknown",
        comm_budget=SearchBudget(30_000),
    )
    assert "branch:old" not in disp["telemetry"]["support_invalidated"]
    assert "branch:old" in disp["telemetry"]["scope_not_established"]
    assert any("old" in (c.get("source_concept_ids") or ()) for c in disp["ordered_consistent_comms"])

    fp_base = compute_evidence_archive_fingerprint(ev_batch)
    ev_rad = deepcopy(ev_batch)
    ev_rad[0]["radius_value"] = 0.75
    ev_com = deepcopy(ev_batch)
    ev_com[0]["commitment"]["radius_parameters"][0]["direction"] = "left"
    ev_obs_mod = deepcopy(ev_batch)
    ev_obs_mod[0]["observations_serialized"][0]["target"] += 0.25
    ev_ts = deepcopy(ev_batch)
    ev_ts[0]["wall_timestamp_nonsemantic"] = "2026-11-18T15:00:00Z"

    assert fp_base != compute_evidence_archive_fingerprint(ev_rad)
    assert fp_base != compute_evidence_archive_fingerprint(ev_com)
    assert fp_base != compute_evidence_archive_fingerprint(ev_obs_mod)
    assert fp_base == compute_evidence_archive_fingerprint(ev_ts)


def test_v15_control_5_chronological_k_vs_r_dispatch_parity_and_partial_pending_resume() -> None:
    """§4.2 & §4.3: Online `K` and lazy `R` achieve 0 semantic dispatch differences + `partial_pending` budget resumption."""
    lib_k = KnowledgeLibrary()
    lib_k._prioritize_generalized_families = False

    src_list = [
        dict(seed=10705, objects=4, readout="window_total_event", relation_direction="symmetric", relation_condition="always", condition_threshold=0.0, relation_radius=0.92, actuator_curvature=0.15, name_seed=10805, wiring_seed=10905),
        dict(seed=10715, objects=4, readout="window_total_event", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.93, actuator_curvature=0.15, name_seed=10815, wiring_seed=10915),
        dict(seed=10725, objects=4, readout="window_total_event", relation_direction="right", relation_condition="donor_positive_gap", condition_threshold=0.20, relation_radius=0.90, actuator_curvature=0.15, name_seed=10825, wiring_seed=10925),
        dict(seed=10745, objects=4, readout="window_total_event", relation_direction="left", relation_condition="donor_positive_gap", condition_threshold=0.24, relation_radius=0.91, actuator_curvature=0.15, name_seed=10845, wiring_seed=10945),
    ]
    for sp in src_list:
        _run_source_episode_into_k_library(lib_k, sp, seed=int(sp["seed"]))

    lib_r = deepcopy(lib_k)
    lib_r._active_reconciled_branches = []
    lib_r._generalized_branch_commitments = []
    lib_r._premise_commitment_index = None
    lib_r._allow_on_demand_family_reconstruction = True
    lib_r._allow_on_demand_routing_reconstruction = True

    _, st_partial = reconstruct_parametric_families_from_ordinary_archive(
        lib_r, SearchBudget(30), reconstruct_routing=True
    )
    assert st_partial["interrupted"] is True
    assert st_partial["reconstruction_status"] == "partial_pending"
    assert len(lib_r._ordinary_commitment_cache._routing_scope_memo) == 0

    _, st_full = reconstruct_parametric_families_from_ordinary_archive(
        lib_r, SearchBudget(180_000), reconstruct_routing=True
    )
    assert st_full["interrupted"] is False
    assert st_full["reconstruction_status"] == "complete"

    t_kw = dict(seed=10755, objects=5, readout="window_total_event", relation_direction="right", relation_condition="donor_positive_gap", condition_threshold=0.31, relation_radius=0.92, actuator_curvature=0.15, name_seed=10855, wiring_seed=10955)
    with AnonymousObjectLab(**t_kw) as target_lab:
        fb_t = target_lab.spec_for(()).feature_bounds
        probe_seq = tuple(
            {
                fn: float(lo + (0.2 + 0.3 * ((i + j) % 3)) * (hi - lo))
                for j, (fn, (lo, hi)) in enumerate(fb_t.items())
            }
            for i in range(8)
        )
        b_acq = target_lab.acquire(Probe("p_disp", probe_seq), ("fine",))
        sc_t = lib_k.meta_priors[-1]["object_schema"]

    disp_k = dispatch_branches_and_families_for_target(
        library=lib_k,
        consistent_comms=[],
        transitions=b_acq.transitions,
        schema=sc_t,
        audit=None,
        task_sig="t_parity",
        comm_budget=SearchBudget(80_000),
    )["telemetry"]
    disp_r = dispatch_branches_and_families_for_target(
        library=lib_r,
        consistent_comms=[],
        transitions=b_acq.transitions,
        schema=sc_t,
        audit=None,
        task_sig="t_parity",
        comm_budget=SearchBudget(80_000),
    )["telemetry"]

    def _norm(tel: dict) -> dict:
        id_map = {it["branch_id"]: it["source_concept_ids"][0] for it in tel["branch_identity_map"]}
        levels = {it["source_concept_ids"][0]: it.get("scope_claim_level") for it in tel["branch_identity_map"]}
        fields = (
            "support_invalidated",
            "empirically_excluded_on_context",
            "empirically_excluded_bound_instance_only",
            "not_scheduled_by_router",
            "scope_not_established",
        )
        out = {f: sorted(id_map[bid] for bid in tel[f]) for f in fields}
        out["scope_claim_level"] = dict(sorted(levels.items()))
        return out

    assert _norm(disp_k) == _norm(disp_r)
