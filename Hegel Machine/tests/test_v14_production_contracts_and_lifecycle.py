"""Tests for Hegel Machine v14: Production Contracts, Replayable Evidence, Pending Resumption, & Boundary Probes.

Covers all 6 Replay Gates (Replays A–F) in ``markdown/hegel_v14/02_下一轮最小施工与实验合同.md`` §3–§6
plus all 3 GPT-6 V14 audit probe invariants (strict JSON ``allow_nan=False`` compliant, zero API calls).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    _cross_evaluate_fixed_instance_on_neighbor,
    _cross_evaluate_refittable_family_on_neighbor,
    _cross_evaluate_refittable_template_on_neighbor,
    _refine_coexisting_branch_scopes_and_routing,
    _serialize_obs_rows,
    autonomous_reconcile_library_episode,
    dispatch_branches_and_families_for_target,
    evaluate_scope_guard,
    instantiate_aligned_claim_program,
    reconstruct_parametric_families_from_ordinary_archive,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.premise_commitment_index import (
    compute_evidence_archive_fingerprint,
    extract_archive_witness_maps,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior


def _dpg_dict(donor_min: float, gap_min: float) -> dict:
    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    gap = Program("sub", children=(q_src, q_dst))
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=donor_min), q_src)),
            Program("lt", children=(Program("const", value=gap_min), gap)),
        ),
    ).to_dict()


def _prior_prog(pr: dict) -> dict | None:
    fits = (pr.get("structure") or {}).get("fits") or ()
    if fits and isinstance(fits[0], dict) and "program" in fits[0]:
        return fits[0]["program"]
    return pr.get("program")


def test_v14_replay_a_real_acquire_schema_exact_protocol_reuse() -> None:
    """Replay A (§3.1, §6.A): Real `acquire(...)` nested batch schema -> `_exec_reconcile_probe` exact reuse & real measurement_cost."""
    with AnonymousObjectLab(
        seed=1001001,
        objects=3,
        readout="focus",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=0.92,
    ) as lab:
        cfg = ResearchConfig(
            seed=1001001,
            observation_budget=96,
            compute_budget=180_000,
            max_rounds=1,
            minimum_improvement=1e-7,
            use_memory=True,
            world_structure_commitments=True,
            transition_premise_audit=True,
            branch_reconciliation=True,
        )
        agent = UnifiedResearchAgent(cfg, measurement_budget=3_000, library=KnowledgeLibrary())
        res = agent.run(lab)
        assert res["batches"], "Expected real production batches from acquire(...)"
        first_fine = next(b for b in res["batches"] if tuple(b["channels"]) == ("fine",))
        assert isinstance(first_fine["probe"], dict) and "sequence" in first_fine["probe"]
        assert isinstance(first_fine["observations"][0], dict)
        assert first_fine["measurement_cost"] > 0
        assert first_fine["episode_uid"] == res["episode_uid"]

        ur_path = Path(__file__).resolve().parents[1] / "src" / "hegel_machine" / "cognition" / "unified_research.py"
        tree = ast.parse(ur_path.read_text(encoding="utf-8"))
        fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_exec_reconcile_probe")

        acq_calls: list[str] = []

        def _mock_acquire(channels, purpose, *, probe=None, **kw):
            acq_calls.append(probe.probe_id)
            b_real = lab.acquire(probe, channels)
            ns["batches"].append(
                {
                    "probe": asdict(probe),
                    "channels": list(channels),
                    "observations": [asdict(r) for r in b_real.observations],
                    "transitions": getattr(b_real, "transitions", []),
                    "observation_cost": int(b_real.observation_cost),
                    "measurement_cost": int(b_real.measurement_cost),
                    "episode_uid": res["episode_uid"],
                    "history_preparation": "reset",
                }
            )
            return b_real.observations

        ns = {
            "reconcile_channels": ("fine",),
            "batches": [deepcopy(first_fine)],
            "acquire": _mock_acquire,
            "episode_uid": res["episode_uid"],
            "Observation": Observation,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn_node], type_ignores=[])), str(ur_path), "exec"), ns)

        # 1) Exact matching protocol on nested production batch -> `reused_exact_protocol` with zero new acquire!
        same_probe = Probe(
            "new_request_same_seq",
            tuple(dict(pt) for pt in first_fine["probe"]["sequence"]),
        )
        reuse_ret = ns["_exec_reconcile_probe"](same_probe, ["prog_v1:frozen_a", "prog_v1:frozen_b"])
        assert reuse_ret["status"] == "reused_exact_protocol"
        assert reuse_ret["actual_probe_id"] == first_fine["probe"]["probe_id"]
        assert reuse_ret["cost_ref"]["new_observations"] == 0
        assert reuse_ret["cost_ref"]["new_measurements"] == 0
        assert len(acq_calls) == 0
        assert isinstance(reuse_ret["rows"][0], Observation)

        # 2) Mismatched `episode_uid` -> must NOT reuse; executes new probe and reports real non-zero `new_measurements`!
        ns["episode_uid"] = "other_episode_uid_999"
        exec_ret = ns["_exec_reconcile_probe"](same_probe, ["prog_v1:frozen_a"])
        assert exec_ret["status"] == "executed_new"
        assert len(acq_calls) == 1
        assert exec_ret["cost_ref"]["new_measurements"] > 0
        assert exec_ret["cost_ref"]["cost_accounting_status"] == "recorded_from_production_batch"


def test_v14_replay_b_budget_denial_halt_and_instance_vs_template_layering() -> None:
    """Replay B (§3.2, §6.B): Immediate halt on `SearchBudget(0)` + `bound_instance` vs `refittable_template` separation."""
    p_r050 = _make_confirmed_regime_prior(
        concept_id="src_r050",
        seed=9401,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=None,
        radius=0.50,
    )
    p_r075 = _make_confirmed_regime_prior(
        concept_id="src_r075",
        seed=9402,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=None,
        radius=0.75,
    )
    p_left = _make_confirmed_regime_prior(
        concept_id="src_left",
        seed=9403,
        objects=3,
        readout="focused_next",
        direction="left",
        edge_pred_dict=None,
        radius=0.75,
    )
    sup_map, _ = extract_archive_witness_maps([p_r050, p_r075, p_left])

    def _mk_branch(p: dict, cid: str, r_val: float, reg: dict) -> dict:
        c = consolidate_world_structure_commitments([p])[0]
        c["commitment_id"] = f"branch_comm:{cid}"
        c["source_concept_ids"] = [cid]
        return {
            "claim_id": f"branch:{cid}",
            "source_concept_id": cid,
            "source_concept_ids": [cid],
            "commitment": c,
            "radius_value": r_val,
            "schema": p["object_schema"],
            "status": "universal",
            "independent_empirical_support": True,
            "pilot_regime_features": [reg],
            "support_rows_serialized": _serialize_obs_rows(sup_map[cid], limit=64),
        }

    b_r050 = _mk_branch(
        p_r050,
        "src_r050",
        0.50,
        {"obs_signed_direction_flux": 0.12, "obs_unconditional_block_rate": 0.10, "_meta": {"protocol_id": "prot_A", "pilot_calibration_available": True}},
    )
    b_r075 = _mk_branch(
        p_r075,
        "src_r075",
        0.75,
        {"obs_signed_direction_flux": 0.28, "obs_unconditional_block_rate": 0.45, "_meta": {"protocol_id": "prot_A", "pilot_calibration_available": True}},
    )
    b_left = _mk_branch(
        p_left,
        "src_left",
        0.75,
        {"obs_signed_direction_flux": -0.25, "obs_unconditional_block_rate": 0.45, "_meta": {"protocol_id": "prot_A", "pilot_calibration_available": True}},
    )

    # 1) `SearchBudget(0)` halts immediately with zero mutations and strict-JSON-safe `cross_mse = None`
    zero_fixed = _cross_evaluate_fixed_instance_on_neighbor(b_r050, b_r075, budget=SearchBudget(0))
    assert zero_fixed["evaluated"] is False
    assert zero_fixed["execution_status"] == "budget_deferred"
    assert zero_fixed["empirically_refuted"] is False
    assert zero_fixed["cross_mse"] is None

    fam_stub = deepcopy(b_r050["commitment"])
    fam_stub["commitment_id"] = "fam:gap"
    fam_stub["parameter_slots"] = ["theta_0"]
    fam_stub["anti_unified_predicate_skeleton"] = Program(
        "lt", children=(Program("arg", name="theta_0"), Program("arg", name="edge_source_quantity"))
    ).to_dict()
    zero_fam = _cross_evaluate_refittable_family_on_neighbor(fam_stub, b_r075, budget=SearchBudget(0))
    assert zero_fam["evaluated"] is False
    assert zero_fam["execution_status"] == "budget_deferred"
    assert zero_fam["empirically_refuted"] is False
    assert zero_fam["cross_mse"] is None
    json.dumps({"zero_fixed": zero_fixed, "zero_fam": zero_fam}, allow_nan=False)

    # 2) `b_r050` vs `b_r075`: Fixed instance `r=0.50` fails on `r=0.75` rows, while refittable template rebinding `r->0.75` FITS!
    fixed_chk = _cross_evaluate_fixed_instance_on_neighbor(b_r050, b_r075, budget=SearchBudget(20_000))
    tmpl_chk = _cross_evaluate_refittable_template_on_neighbor(b_r050, b_r075, budget=SearchBudget(20_000))
    assert fixed_chk["empirically_refuted"] is True and fixed_chk["verdict"] == "refuted_instance_on_batch"
    assert tmpl_chk["empirically_refuted"] is False and tmpl_chk["verdict"] == "supported_on_batch"

    refined_pair = _refine_coexisting_branch_scopes_and_routing([b_r050, b_r075], budget=SearchBudget(50_000))
    ref_050 = next(b for b in refined_pair if b["claim_id"] == "branch:src_r050")
    assert ref_050["scope_claim_level"] == "bound_instance"
    assert ref_050.get("template_scope_guard") is None

    lib_pair = KnowledgeLibrary()
    lib_pair._active_reconciled_branches = [ref_050]
    lib_pair._prioritize_generalized_families = False
    agg_template = deepcopy(b_r050["commitment"])
    agg_template["commitment_id"] = "world_commitment:0000"
    agg_template["source_concept_ids"] = ["src_r050"]
    sc_75 = p_r075["object_schema"]
    o0, o1 = sc_75["objects"][0], sc_75["objects"][1]
    # Ensure regime has `protocol_id == 'prot_A'` matching `ref_050`'s support!
    disp_r = dispatch_branches_and_families_for_target(
        library=lib_pair,
        consistent_comms=[agg_template],
        transitions=[
            {
                "before": {sc_75["fields"][o0]["q"]: 0.8, sc_75["fields"][o0]["x"]: 0.0, sc_75["fields"][o1]["q"]: 0.1, sc_75["fields"][o1]["x"]: 0.6},
                "after": {sc_75["fields"][o0]["q"]: 0.8, sc_75["fields"][o0]["x"]: 0.0, sc_75["fields"][o1]["q"]: 0.1, sc_75["fields"][o1]["x"]: 0.6},
            }
        ],
        schema=sc_75,
        audit=None,
        task_sig="target_r075",
        comm_budget=SearchBudget(10_000),
    )
    # Also test with matching protocol (`paid_episode_transitions`) so `scope_guard` returns `'reject'`:
    b_r050_std = deepcopy(b_r050)
    b_r050_std["pilot_regime_features"] = [{"obs_signed_direction_flux": 0.12, "obs_unconditional_block_rate": 0.10, "_meta": {"protocol_id": "paid_episode_transitions", "pilot_calibration_available": True}}]
    b_r075_std = deepcopy(b_r075)
    b_r075_std["pilot_regime_features"] = [{"obs_signed_direction_flux": 0.28, "obs_unconditional_block_rate": 0.45, "_meta": {"protocol_id": "paid_episode_transitions", "pilot_calibration_available": True}}]
    ref_std = _refine_coexisting_branch_scopes_and_routing([b_r050_std, b_r075_std], budget=SearchBudget(50_000))
    lib_pair._active_reconciled_branches = [next(b for b in ref_std if b["claim_id"] == "branch:src_r050")]
    o0, o1, o2 = sc_75["objects"][0], sc_75["objects"][1], sc_75["objects"][2]
    disp_std = dispatch_branches_and_families_for_target(
        library=lib_pair,
        consistent_comms=[agg_template],
        transitions=[
            {
                "before": {sc_75["fields"][o0]["q"]: 0.8, sc_75["fields"][o0]["x"]: 0.0, sc_75["fields"][o1]["q"]: 0.1, sc_75["fields"][o1]["x"]: 0.4, sc_75["fields"][o2]["q"]: 0.0, sc_75["fields"][o2]["x"]: 1.9},
                "after": {sc_75["fields"][o0]["q"]: 0.8, sc_75["fields"][o0]["x"]: 0.0, sc_75["fields"][o1]["q"]: 0.1, sc_75["fields"][o1]["x"]: 0.4, sc_75["fields"][o2]["q"]: 0.0, sc_75["fields"][o2]["x"]: 1.9},
            }
        ],
        schema=sc_75,
        audit=None,
        task_sig="target_r075",
        comm_budget=SearchBudget(10_000),
    )
    assert "branch:src_r050" in disp_std["telemetry"]["empirically_excluded_bound_instance_only"]
    assert any(c["commitment_id"] == "world_commitment:0000" for c in disp_std["ordered_consistent_comms"])

    # 3) Conversely, `b_r050` (right-directed) vs `b_left` (left-directed): even refittable template is infeasible!
    tmpl_vs_left = _cross_evaluate_refittable_template_on_neighbor(b_r050, b_left, budget=SearchBudget(20_000))
    assert tmpl_vs_left["empirically_refuted"] is True
    assert tmpl_vs_left["verdict"] == "infeasible_on_declared_domain"
    refined_opp = _refine_coexisting_branch_scopes_and_routing([b_r050, b_left], budget=SearchBudget(50_000))
    ref_050_opp = next(b for b in refined_opp if b["claim_id"] == "branch:src_r050")
    assert ref_050_opp["scope_claim_level"] == "refittable_template"
    assert ref_050_opp.get("template_scope_guard") is not None
    assert (
        evaluate_scope_guard(
            ref_050_opp["scope_guard"],
            {"obs_signed_direction_flux": 0.2, "_meta": {"protocol_id": "prot_OTHER", "pilot_calibration_available": True}},
        )
        == "unknown"
    )


def test_v14_replay_c_router_deferral_end_to_end_consumption() -> None:
    """Replay C (§3.3, §6.C): Router deferral preserves `world_commitment:0000` AND consumes `deferred_router_families` on fallback."""
    p_s1 = _make_confirmed_regime_prior(
        concept_id="s1",
        seed=9501,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.18),
        radius=0.92,
    )
    p_s2 = _make_confirmed_regime_prior(
        concept_id="s2",
        seed=9502,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.34),
        radius=0.92,
    )
    lib = KnowledgeLibrary()
    for cid, pr in (("s1", p_s1), ("s2", p_s2)):
        lib.meta_priors.append(deepcopy(pr))
        sup_r = extract_archive_witness_maps([pr])[0][cid]
        autonomous_reconcile_library_episode(
            library=lib,
            task_id=f"src_{cid}",
            spec=TaskSpec(f"src_{cid}", "d", tuple(pr["object_schema"]["fields"][g][k] for g in pr["object_schema"]["objects"] for k in ("q", "x", "u", "f")), (), (), feature_bounds={}),
            schema_dict=pr["object_schema"],
            confirmed=True,
            confirmation_loss=0.0,
            confirmation_rows_list=sup_r[-12:],
            support_rows=sup_r[:-12],
            episode_transitions=[],
            concept_id=cid,
            execute_probe_fn=None,
            budget=SearchBudget(80_000),
            delivered_program=_prior_prog(pr),
        )
    assert len(lib._generalized_branch_commitments) == 1
    defer_guard = Program("lt", children=(Program("feature", name="obs_signed_direction_flux"), Program("const", value=-0.5))).to_dict()
    for b in lib._active_reconciled_branches:
        b["routing_guard"] = deepcopy(defer_guard)
    lib._generalized_branch_commitments[0]["routing_guard"] = deepcopy(defer_guard)
    fam_id = lib._generalized_branch_commitments[0]["commitment_id"]

    p_tgt = _make_confirmed_regime_prior(
        concept_id="tgt",
        seed=9503,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.26),
        radius=0.92,
    )
    tgt_rows = extract_archive_witness_maps([p_tgt])[0]["tgt"]
    sc_t = p_tgt["object_schema"]
    o0, o1 = sc_t["objects"][0], sc_t["objects"][1]
    pos_trans = [
        {
            "before": {sc_t["fields"][o0]["q"]: 0.8, sc_t["fields"][o0]["x"]: 0.0, sc_t["fields"][o1]["q"]: 0.1, sc_t["fields"][o1]["x"]: 0.4},
            "after": {sc_t["fields"][o0]["q"]: 0.6, sc_t["fields"][o0]["x"]: 0.0, sc_t["fields"][o1]["q"]: 0.3, sc_t["fields"][o1]["x"]: 0.4},
        }
    ]
    c_agg = consolidate_world_structure_commitments([p_s1])[0]
    c_agg["commitment_id"] = "world_commitment:0000"
    c_agg["source_concept_ids"] = ["s1"]

    disp = dispatch_branches_and_families_for_target(
        library=lib,
        consistent_comms=[c_agg],
        transitions=pos_trans,
        schema=sc_t,
        audit={"radius_interval": [0.92, 0.92], "verified_direction": "right"},
        task_sig="tgt_026",
        comm_budget=SearchBudget(50_000),
    )
    assert any(c["commitment_id"] == "world_commitment:0000" for c in disp["ordered_consistent_comms"])
    assert disp["scheduled_families"] == []
    assert [f["commitment_id"] for f in disp["deferred_router_families"]] == [fam_id]

    # Primary `meta_priors` has only an unconditional commitment (which fails on conditional `tgt_026`),
    # so consumer MUST fall back to `deferred_router_families` (`fam_id`) to achieve `training_loss <= 1e-7`!
    p_uncond = _make_confirmed_regime_prior(
        concept_id="s0_uncond",
        seed=9500,
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=None,
        radius=0.92,
    )
    lib_fallback = KnowledgeLibrary()
    lib_fallback.meta_priors = [deepcopy(p_uncond)]
    lib_fallback._generalized_branch_commitments = deepcopy(lib._generalized_branch_commitments)
    lib_fallback._active_reconciled_branches = []

    agent = UnifiedResearchAgent(
        ResearchConfig(
            seed=9503,
            use_memory=True,
            world_structure_commitments=True,
            branch_reconciliation=True,
            active_threshold_probing=False,
        ),
        library=lib_fallback,
    )
    actions: list[dict] = []
    spec_t = TaskSpec(
        "tgt_026",
        "d",
        tuple(sc_t["fields"][g][k] for g in sc_t["objects"] for k in ("q", "x", "u", "f")),
        (),
        (),
        feature_bounds={},
        context={"object_schema": json.dumps(sc_t)},
    )
    cands = agent._propose_schema_structures(
        spec_t, tgt_rows[:32], SearchBudget(120_000), actions=actions, tolerance=1e-7, transitions=pos_trans
    )
    assert any(c["training_loss"] <= 1e-7 for c in cands)
    apply_act = next(a for a in actions if a.get("action") == "apply_world_structure_commitments")
    assert fam_id in apply_act["branch_dispatch_telemetry"]["deferred_router_families_tried"]


def test_v14_replay_d_shared_evidence_replay_equivalence_and_content_cache_invalidation() -> None:
    """Replay D (§4.1, §4.2, §6.D): `K_eager` vs `R_lazy_noH` semantic equivalence + content-hash cache invalidation."""
    p1 = _make_confirmed_regime_prior(concept_id="c1", seed=9601, objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.18), radius=0.90)
    p2 = _make_confirmed_regime_prior(concept_id="c2", seed=9602, objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.35), radius=0.90)
    p3 = _make_confirmed_regime_prior(concept_id="c3", seed=9603, objects=3, readout="focused_next", direction="left", edge_pred_dict=None, radius=0.88)

    lib_k = KnowledgeLibrary()
    lib_k._prioritize_generalized_families = False
    for cid, pr, flux in (("c1", p1, 0.18), ("c2", p2, 0.16), ("c3", p3, -0.22)):
        lib_k.meta_priors.append(deepcopy(pr))
        sup_r = extract_archive_witness_maps([pr])[0][cid]
        sc_p = pr["object_schema"]
        o0, o1 = sc_p["objects"][0], sc_p["objects"][1]
        tr_p = [
            {
                "before": {sc_p["fields"][o0]["q"]: 0.7 if flux > 0 else 0.1, sc_p["fields"][o0]["x"]: 0.0, sc_p["fields"][o1]["q"]: 0.1 if flux > 0 else 0.7, sc_p["fields"][o1]["x"]: 0.4},
                "after": {sc_p["fields"][o0]["q"]: 0.55 if flux > 0 else 0.25, sc_p["fields"][o0]["x"]: 0.0, sc_p["fields"][o1]["q"]: 0.25 if flux > 0 else 0.55, sc_p["fields"][o1]["x"]: 0.4},
            }
        ]
        autonomous_reconcile_library_episode(
            library=lib_k,
            task_id=f"src_{cid}",
            spec=TaskSpec(f"src_{cid}", "d", tuple(sc_p["fields"][g][k] for g in sc_p["objects"] for k in ("q", "x", "u", "f")), (), (), feature_bounds={}),
            schema_dict=sc_p,
            confirmed=True,
            confirmation_loss=0.0,
            confirmation_rows_list=sup_r[-12:],
            support_rows=sup_r[:-12],
            episode_transitions=tr_p,
            concept_id=cid,
            execute_probe_fn=None,
            budget=SearchBudget(100_000),
            delivered_program=_prior_prog(pr),
        )

    lib_r = KnowledgeLibrary()
    lib_r.meta_priors = deepcopy(lib_k.meta_priors)
    lib_r._raw_reconciliation_evidence_archive = [
        {
            "evidence_id": "ev_disc:0",
            "task_id": "src_c3",
            "episode_uid": "src_c3",
            "protocol_id": "single_step_transition_ledger_v1",
            "schema": deepcopy(p3["object_schema"]),
            "radius_value": 0.88,
            "commitment": deepcopy(lib_k._active_reconciled_branches[-1]["commitment"]),
            "observations_serialized": deepcopy(lib_k._active_reconciled_branches[-1]["support_rows_serialized"][:16]),
            "transitions": [
                {
                    "before": {p3["object_schema"]["fields"][p3["object_schema"]["objects"][0]]["q"]: 0.1, p3["object_schema"]["fields"][p3["object_schema"]["objects"][0]]["x"]: 0.0, p3["object_schema"]["fields"][p3["object_schema"]["objects"][1]]["q"]: 0.7, p3["object_schema"]["fields"][p3["object_schema"]["objects"][1]]["x"]: 0.4},
                    "after": {p3["object_schema"]["fields"][p3["object_schema"]["objects"][0]]["q"]: 0.25, p3["object_schema"]["fields"][p3["object_schema"]["objects"][0]]["x"]: 0.0, p3["object_schema"]["fields"][p3["object_schema"]["objects"][1]]["q"]: 0.55, p3["object_schema"]["fields"][p3["object_schema"]["objects"][1]]["x"]: 0.4},
                }
            ],
        }
    ]
    lib_k._raw_reconciliation_evidence_archive = deepcopy(lib_r._raw_reconciliation_evidence_archive)
    lib_r._prioritize_generalized_families = False
    lib_r._allow_on_demand_family_reconstruction = True
    lib_r._allow_on_demand_routing_reconstruction = True

    fams_r1, st_r1 = reconstruct_parametric_families_from_ordinary_archive(
        lib_r, SearchBudget(120_000), reconstruct_routing=True
    )
    assert fams_r1 == [] and st_r1["families_skipped_noH"] is True and st_r1["layer2_built"] == 0
    assert st_r1["layer3_routing_built"] is True

    k_by_src = {b["source_concept_id"]: b for b in lib_k._active_reconciled_branches}
    r_by_src = {b["source_concept_id"]: b for b in st_r1["reconstructed_branches"]}
    test_rows_p1 = extract_archive_witness_maps([p1])[0]["c1"][:10]
    probe_regimes = [
        {"obs_signed_direction_flux": 0.20, "_meta": {"protocol_id": "single_step_transition_ledger_v1", "pilot_calibration_available": True}},
        {"obs_signed_direction_flux": -0.20, "_meta": {"protocol_id": "single_step_transition_ledger_v1", "pilot_calibration_available": True}},
        {"obs_signed_direction_flux": 0.20, "_meta": {"protocol_id": "other_protocol_v2", "pilot_calibration_available": True}},
    ]
    for cid in ("c1", "c2", "c3"):
        bk, br = k_by_src[cid], r_by_src[cid]
        assert bk["status"] == br["status"]
        assert bk.get("scope_claim_level") == br.get("scope_claim_level")
        assert abs(float(bk["radius_value"]) - float(br["radius_value"])) <= 1e-9
        alk = instantiate_aligned_claim_program(bk, p1["object_schema"])
        alr = instantiate_aligned_claim_program(br, p1["object_schema"])
        for row in test_rows_p1:
            assert abs(float(alk["program"].predict(row)) - float(alr["program"].predict(row))) <= 1e-11
        for reg in probe_regimes:
            assert evaluate_scope_guard(bk.get("scope_guard"), reg) == evaluate_scope_guard(br.get("scope_guard"), reg)

    _, st_r2 = reconstruct_parametric_families_from_ordinary_archive(
        lib_r, SearchBudget(120_000), reconstruct_routing=True
    )
    assert st_r2["cache_hit"] is True and st_r2["layer3_routing_reused"] is True

    fp_before = compute_evidence_archive_fingerprint(lib_r._raw_reconciliation_evidence_archive)
    lib_r._raw_reconciliation_evidence_archive[0]["observations_serialized"][0]["target"] += 0.12345
    fp_after = compute_evidence_archive_fingerprint(lib_r._raw_reconciliation_evidence_archive)
    assert len(lib_r._raw_reconciliation_evidence_archive) == 1
    assert fp_before != fp_after
    _, st_r3 = reconstruct_parametric_families_from_ordinary_archive(
        lib_r, SearchBudget(120_000), reconstruct_routing=True
    )
    assert st_r3["cache_hit"] is False and st_r3["layer3_routing_built"] is True


def test_v14_replay_e_persistent_pending_resumption_and_post_subresearch_reconcile() -> None:
    """Replay E (§5.1, §5.2, §6.E): Cross-run `pending_unfunded` resumption on same `library`, stale version guard, & local re-reconciliation."""
    lib = KnowledgeLibrary()
    for sd, d_mode in ((9700, "symmetric"), (9701, "right")):
        cfg_src = ResearchConfig(
            seed=sd,
            observation_budget=520,
            compute_budget=520_000,
            max_rounds=4,
            minimum_improvement=1e-7,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            active_threshold_probing=True,
            indexed_commitment_store=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
            branch_reconciliation=True,
        )
        with AnonymousObjectLab(
            seed=sd,
            objects=4,
            readout="focus",
            relation_direction=d_mode,
            relation_condition="always",
            condition_threshold=0.0,
            relation_radius=0.93,
            actuator_curvature=0.15,
            name_seed=sd + 31,
            wiring_seed=sd + 61,
        ) as src_lab:
            src_res = UnifiedResearchAgent(cfg_src, measurement_budget=11_000, library=lib).run(src_lab)
        assert src_res["status"] == "completed"

    assert len(lib._active_reconciled_branches) == 2

    # Run 1 on `focal_memory` conflict task with tight observation cap (`195`) -> `pending_unfunded`!
    fm_kwargs = dict(
        seed=9705,
        objects=4,
        readout="focus",
        relation_direction="right",
        relation_condition="always",
        condition_threshold=0.0,
        relation_radius=0.93,
        latent_mode="focal_memory",
        latent_decay=0.55,
        latent_weight=0.35,
        actuator_curvature=0.15,
        name_seed=9736,
        wiring_seed=9766,
    )
    cfg_run1 = ResearchConfig(
        seed=9705,
        observation_budget=195,
        compute_budget=480_000,
        max_rounds=1,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=False,
        latent_state_discovery=False,
        active_threshold_probing=False,
        state_search=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=True,
    )
    with AnonymousObjectLab(**fm_kwargs) as lab_fm1:
        res_run1 = UnifiedResearchAgent(cfg_run1, measurement_budget=3_600, library=lib).run(lab_fm1)

    assert res_run1["status"] == "unresolved"
    pend_qs = [q for q in lib._pending_reconciliation_questions if q.get("status") == "pending_unfunded"]
    assert len(pend_qs) >= 1
    q1_id = pend_qs[0]["question_id"]
    run1_obs = res_run1["costs"]["observation_cost"]
    run1_comp = res_run1["costs"]["compute_cost"]

    sibling_id = f"sibling_unidentified:{res_run1['episode_uid']}"
    lib._pending_reconciliation_questions.append(
        {
            "question_id": sibling_id,
            "episode_uid": res_run1["episode_uid"],
            "kind": "scope_boundary_unidentified_after_refutation",
            "refuted_claim_id": "nonexistent_branch",
            "status": "pending_unfunded",
        }
    )

    # Parent-version mutation check on cloned library -> marks `stale_parent_version_changed`!
    lib_stale = deepcopy(lib)
    lib_stale._active_reconciled_branches[0]["semantic_version"] = "sem_v1:mutated_parent_999"
    cfg_run2 = ResearchConfig(
        seed=9705,
        observation_budget=560,
        compute_budget=880_000,
        max_rounds=1,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=False,
        latent_state_discovery=False,
        active_threshold_probing=False,
        state_search=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=True,
    )
    with AnonymousObjectLab(**fm_kwargs) as lab_fm_stale:
        res_stale = UnifiedResearchAgent(cfg_run2, measurement_budget=11_500, library=lib_stale).run(lab_fm_stale)
    stale_entry = next(q for q in lib_stale._pending_reconciliation_questions if q["question_id"] == q1_id)
    assert stale_entry["status"] == "stale_parent_version_changed"
    assert any(a.get("action") == "invalidate_stale_pending_subquestion" for a in res_stale["actions"])

    # Run 2 on the SAME `lib` with increased budget (NO `previous=` passed!) -> resumes & resolves `q1_id`!
    with AnonymousObjectLab(**fm_kwargs) as lab_fm2:
        res_run2 = UnifiedResearchAgent(cfg_run2, measurement_budget=11_500, library=lib).run(lab_fm2)

    assert res_run2["status"] == "completed"
    assert res_run2["confirmation_loss"] <= 1e-7
    resolved_q1 = next(q for q in lib._pending_reconciliation_questions if q["question_id"] == q1_id)
    assert resolved_q1["status"] == "answered"
    assert resolved_q1["resolution"]["independent_fresh_confirmation"] is True
    assert res_run2["costs"]["observation_cost"] > run1_obs
    assert res_run2["costs"]["compute_cost"] > run1_comp
    assert any(a.get("is_post_subresearch_re_reconcile") is True for a in res_run2["actions"])
    repaired_cid = resolved_q1["resolution"]["repaired_concept_id"]
    assert any(b["claim_id"] == f"branch:{repaired_cid}" for b in lib._active_reconciled_branches)
    sibling_after = next(q for q in lib._pending_reconciliation_questions if q["question_id"] == sibling_id)
    assert sibling_after["status"] != "answered"


def test_v14_replay_f_ordinary_loop_residual_threshold_boundary_probe() -> None:
    """Replay F (§5.3, §6.F): Exact exposed `1021000/S_left_dpg_N4` chain executes boundary probe before separate fresh confirmation (`0` false confirmations!)."""
    lib = KnowledgeLibrary()
    base_seed = 1_021_000
    res = None
    ext_mse = None
    for tag, off, d_mode, cond, th, rad in (
        ("S_cal", 5, "symmetric", "always", 0.0, 0.92),
        ("S0", 15, "right", "always", 0.0, 0.94),
        ("S1", 25, "right", "donor_positive_gap", 0.19, 0.91),
        ("S2", 35, "right", "donor_positive_gap", 0.45, 0.93),
        ("S_left", 45, "left", "donor_positive_gap", 0.25, 0.90),
    ):
        sd = base_seed + off
        cfg_ep = ResearchConfig(
            seed=sd,
            observation_budget=660,
            compute_budget=740_000,
            max_rounds=4,
            minimum_improvement=1e-7,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            active_threshold_probing=True,
            indexed_commitment_store=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
            branch_reconciliation=True,
        )
        with AnonymousObjectLab(
            seed=sd,
            objects=4,
            readout="window_total_event",
            relation_direction=d_mode,
            relation_condition=cond,
            condition_threshold=th,
            relation_radius=rad,
            actuator_curvature=0.15,
            name_seed=sd + 100,
            wiring_seed=sd + 200,
        ) as ep_lab:
            ep_res = UnifiedResearchAgent(cfg_ep, measurement_budget=13_200, library=lib).run(ep_lab)
            if tag == "S_left":
                res = ep_res
                prog_sleft = Program.from_dict(res["program"])
                req_vars = {name for name, _ in prog_sleft.observed_variables()}
                cmd_vars = set(ep_lab.spec_for(()).feature_names)
                eval_ch = ("fine",) if (res.get("selected_channels") or not (req_vars <= (cmd_vars | {"__target__"}))) else ()
                ext_rows = ep_lab.evaluation_rows(eval_ch, seed=sd + 920_001, length=24, trajectories=2)
                ext_mse = float(loss(prog_sleft, ext_rows, float(res.get("fallback", 0.0)), 0.5))

    assert res is not None and ext_mse is not None
    assert res["status"] == "completed"
    assert res["confirmation_loss"] <= 1e-7
    bp_acts = [a for a in res["actions"] if a.get("action") == "execute_threshold_boundary_probe"]
    assert len(bp_acts) >= 1, "Expected residual threshold boundary probe to fire on 1021000/S_left_dpg_N4!"
    conf_acts = [a for a in res["actions"] if a.get("action") == "acquire" and a.get("purpose") == "confirm_frozen_program"]
    assert conf_acts, "Must confirm on a separate fresh confirmation probe!"
    bp_ids = {a["probe_id"] for a in bp_acts}
    assert conf_acts[-1]["probe_id"] not in bp_ids
    assert ext_mse <= 1e-7, f"Expected strict external MSE <= 1e-7 on 1021000/S_left_dpg_N4, got {ext_mse}"
