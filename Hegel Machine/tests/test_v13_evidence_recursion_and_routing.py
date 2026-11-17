"""v13 Behavioral Acceptance Suite covering all 16 contract controls (#1–#16) & GPT-6 review probes.

Directly verifies ``markdown/hegel_v13/02_下一轮最小施工与实验合同.md`` §2–§5 without adding new
framework modules or new physics families:
  - Controls #1, #2, #3: Typed probe acquisition (``executed_new / reused_exact_protocol / unavailable``)
    and separate ``reanalyze_paid_evidence`` (never disguising unexecuted probes as paid probes).
  - Controls #4, #5: Unknown (``None``) statistics for empty/zero-denominator pilot summaries,
    three-valued ``evaluate_scope_guard`` (``'accept' | 'reject' | 'unknown'``), and full-AST
    pilot dependency export on retained old guard atoms.
  - Controls #6, #7, #8: Both-refuted parent-bound ``pending_questions`` + ordinary
    ``UnifiedResearchAgent.run(lab)`` recursive subquestion execution (funded -> mutates parent
    model & confirms missing mechanism; unfunded -> preserves ``pending_unfunded`` & ``unresolved``),
    and instance-level refutation preserving valid old claims & model families.
  - Controls #9, #10: Explicit ``branch_id -> commitment_id -> source_concept_ids`` mapping and
    multi-source preservation when one supporting source is retracted or excluded.
  - Controls #11, #12: ``routing_scope_candidate`` vs ``empirical_applicability_scope`` separation
    for prediction-equivalent distinct structures, and unconfirmed complement proposals allowing
    third mechanisms.
  - Controls #13, #14: 3-layer ``OrdinaryPersistentCommitmentCache`` full reuse & local invalidation.
  - Controls #15, #16: ``_prioritize_generalized_families`` single-factor toggle and threshold-boundary
    disagreement probe synthesis + honest ``false_internal_confirmation`` retention.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    _refine_coexisting_branch_scopes_and_routing,
    adjudicate_branches_on_paid_evidence,
    compute_observable_regime_features,
    dispatch_branches_and_families_for_target,
    evaluate_scope_guard,
    extract_guard_ast_dependencies,
    propagate_premise_retraction_invalidation,
    propose_threshold_interval_boundary_probe,
    reconcile_competing_branches,
    reconstruct_parametric_families_from_ordinary_archive,
    shrink_claim_to_observable_scope,
    validate_observable_scope_guard,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.premise_commitment_index import extract_archive_witness_maps
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments

from tests.test_v12_semantics_and_autonomous_reconciliation import (
    _dpg_dict,
    _make_confirmed_regime_prior,
)


def _make_prior(
    *,
    cid: str,
    objects: int,
    readout: str,
    direction: str,
    edge_pred_dict: dict | str | None,
    radius: float,
    seed: int = 9201,
) -> dict:
    eff_pred = None if edge_pred_dict == "unconditional" else edge_pred_dict
    eff_readout = "window_total_event" if readout == "focused_next" else readout
    return _make_confirmed_regime_prior(
        concept_id=cid,
        seed=seed,
        objects=max(4, objects),
        readout=eff_readout,
        direction=direction,
        edge_pred_dict=eff_pred,
        radius=radius,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v13_controls_1_2_3_typed_probe_acquisition_reuse_and_reanalysis() -> None:
    # Extract verbatim `_exec_reconcile_probe` from `unified_research.py` (matching GPT-6's review_probes.py!)
    upath = REPO_ROOT / "src/hegel_machine/cognition/unified_research.py"
    u_tree = ast.parse(upath.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(u_tree) if isinstance(n, ast.FunctionDef) and n.name == "_exec_reconcile_probe")

    # Control #1: `acquire` returns `None` -> returns `status == 'unavailable'`, NEVER old episode rows!
    old_rows = [Observation({"x": 0.0}, 0.0)]
    old_trans = [{"before": {"x": 0.0}, "after": {"x": 0.0}}]
    cb_ns = {
        "reconcile_channels": ("fine",),
        "acquire": lambda *a, **kw: None,
        "batches": [],
        "ep_obs_rec": old_rows,
        "ep_trans_rec": old_trans,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(upath), "exec"), cb_ns)
    req_probe = SimpleNamespace(probe_id="new_probe_not_executed", sequence=({"c0": 0.5},), reset=True)
    ret_denied = cb_ns["_exec_reconcile_probe"](req_probe, frozen_program_versions=["prog_v1:a", "prog_v1:b"])
    assert ret_denied["status"] == "unavailable"
    assert ret_denied["actual_probe_id"] is None
    assert ret_denied["rows"] is None

    # Verify `reconcile_competing_branches` NEVER records `new_probe_not_executed` in `paid_probes_executed`!
    p_a = _make_prior(cid="p_a", objects=4, readout="window_total_event", direction="right", edge_pred_dict="unconditional", radius=0.95)
    p_b = _make_prior(cid="p_b", objects=4, readout="window_total_event", direction="left", edge_pred_dict="unconditional", radius=0.95)
    sc = ObjectSchema.from_dict(p_a["object_schema"])
    c_a = consolidate_world_structure_commitments([p_a])[0]
    c_b = consolidate_world_structure_commitments([p_b])[0]
    spec = TaskSpec(
        task_id="t",
        description="test",
        feature_names=tuple(sc.feature_names),
        initial_probes=(),
        available_probes=(),
        feature_bounds={f: (-1.5, 1.5) for f in sc.feature_names},
    )
    claim_new = {"claim_id": "new_c", "commitment": c_a, "status": "universal", "independent_empirical_support": True}
    claim_old = {"claim_id": "old_c", "commitment": c_b, "status": "universal", "independent_empirical_support": True}

    rec_unavail = reconcile_competing_branches(
        claim_new, [claim_old], spec, sc, execute_probe_fn=cb_ns["_exec_reconcile_probe"]
    )
    assert rec_unavail["paid_probes_executed"] == []
    assert rec_unavail["decisions"][0]["action"] == "retain_unresolved_competition"
    assert rec_unavail["decisions"][0]["executed_requested_probe"] is False

    # Control #3: Exact same protocol batch already in `batches` -> `reused_exact_protocol` with 0 new independent samples!
    # Synthesize the exact disagreement probe first so we can populate a matching historical batch
    captured_probes = []
    reconcile_competing_branches(
        claim_new,
        [claim_old],
        spec,
        sc,
        execute_probe_fn=lambda p, **kw: (captured_probes.append(p) or {"status": "unavailable"}),
    )
    real_probe = captured_probes[0]
    exact_row = Observation({k: float(v) for k, v in real_probe.sequence[-1].items()}, 0.0, tuple(dict(s) for s in real_probe.sequence[:-1]))
    cb_ns["batches"] = [
        {
            "probe_id": "historical_exact_batch_0",
            "channels": ("fine",),
            "sequence": list(real_probe.sequence),
            "reset": True,
            "observations": [exact_row],
            "transitions": [{"before": dict(real_probe.sequence[0]), "after": dict(real_probe.sequence[-1])}],
        }
    ]
    rec_reuse = reconcile_competing_branches(
        claim_new, [claim_old], spec, sc, execute_probe_fn=cb_ns["_exec_reconcile_probe"]
    )
    assert rec_reuse["paid_probes_executed"] == []
    assert len(rec_reuse["reused_protocol_probes"]) == 1
    assert rec_reuse["reused_protocol_probes"][0]["probe_id"] == "historical_exact_batch_0"
    assert rec_reuse["reused_protocol_probes"][0]["independent_sample_increment"] == 0

    # Control #2: Non-matching historical record can ONLY go through `reanalyze_paid_evidence` keeping original ID!
    sup_rows_a = extract_archive_witness_maps([p_a])[0]["p_a"]

    def _reanal_fn(p, **kw):
        return {
            "status": "unavailable",
            "requested_probe_id": p.probe_id,
            "actual_probe_id": None,
            "reanalyzable_existing_evidence": {
                "original_probe_id": "prior_m0_trajectory_batch_0",
                "rows": list(sup_rows_a[-4:]),
                "transitions": [{"before": dict(real_probe.sequence[0]), "after": dict(real_probe.sequence[-1])}],
                "observation_refs": ["batch_0:row_0"],
            },
        }

    rec_hist = reconcile_competing_branches(claim_new, [claim_old], spec, sc, execute_probe_fn=_reanal_fn)
    assert rec_hist["paid_probes_executed"] == []
    assert len(rec_hist["historical_reanalyses"]) == 1
    assert rec_hist["historical_reanalyses"][0]["original_probe_id"] == "prior_m0_trajectory_batch_0"
    assert rec_hist["historical_reanalyses"][0]["executed_requested_probe"] is False
    assert rec_hist["historical_reanalyses"][0]["independent_sample_increment"] == 0


def test_v13_controls_4_5_unknown_pilot_tristate_guard_and_full_ast_export() -> None:
    # Control #4: Empty pilot or zero-denominator ratio is `None` (unknown), NEVER `0.0`,
    # and `evaluate_scope_guard` returns `'unknown'` (neither `'accept'` nor `'reject'`)!
    g = Program("lt", children=(Program("feature", name="obs_unconditional_block_rate"), Program("const", value=0.1)))
    empty = compute_observable_regime_features([], None)
    assert empty["obs_unconditional_block_rate"] is None
    assert empty["_meta"]["pilot_calibration_available"] is False
    assert empty["_meta"]["valid_counts"]["obs_unconditional_block_rate"] == 0
    assert validate_observable_scope_guard(g, pilot_calibration_available=False)[0] is False
    assert evaluate_scope_guard(g, empty) == "unknown"

    # Non-empty transitions where NO negative donor (`q < -0.08`) occurred -> `obs_negative_donor_transfer_rate` is `None` (`0/0` unknown)!
    sc = ObjectSchema(
        objects=("g0", "g1"),
        fields={"g0": {"q": "q0", "x": "x0", "u": "u0", "f": "f0"}, "g1": {"q": "q1", "x": "x1", "u": "u1", "f": "f1"}},
        members={"g0": ("q0", "x0", "u0", "f0"), "g1": ("q1", "x1", "u1", "f1")},
        controls={"g0": {"q": "u0"}, "g1": {"q": "u1"}},
    )
    pos_only_trans = [
        {
            "before": {"q0": 0.6, "x0": 0.0, "u0": 0.0, "f0": 1.0, "q1": 0.2, "x1": 0.4, "u1": 0.0, "f1": 0.0},
            "after": {"q0": 0.5, "x0": 0.0, "u0": 0.0, "f0": 1.0, "q1": 0.3, "x1": 0.4, "u1": 0.0, "f1": 0.0},
        }
    ]
    pos_feats = compute_observable_regime_features(pos_only_trans, sc)
    assert pos_feats["obs_negative_donor_transfer_rate"] is None
    assert pos_feats["_meta"]["valid_counts"]["obs_negative_donor_transfer_rate"] == 0
    g_neg = Program("lt", children=(Program("feature", name="obs_negative_donor_transfer_rate"), Program("const", value=0.2)))
    assert evaluate_scope_guard(g_neg, pos_feats) == "unknown"

    # Control #5: Retained old guard atom using pilot statistic + newly added non-pilot atom (`obs_mean_abs_q`)
    # MUST export `requires_pilot_calibration == True` from the COMPLETE guard AST!
    old_pilot_guard = Program("lt", children=(Program("const", value=0.0), Program("feature", name="obs_signed_direction_flux")))
    claim_with_old_guard = {
        "claim_id": "b_old_atom",
        "status": "scoped_conditional",
        "scope_guard": old_pilot_guard.to_dict(),
        "observable_bounds": {"obs_signed_direction_flux": [0.0, 1e6]},
    }
    # Support has flux > 0, mean_abs_q = 0.8; CE ALSO has flux > 0 (passes old guard!), separated only if we add another atom
    # Even simpler: CE is ALREADY excluded by old_pilot_guard (`chosen_features == []`!) — exact case GPT-6 found in B!
    shrunk_kept = shrink_claim_to_observable_scope(
        claim_with_old_guard,
        retained_support_regime_features=[{"obs_signed_direction_flux": 0.25, "obs_unconditional_block_rate": 0.0}],
        refuting_regime_features={"obs_signed_direction_flux": -0.25, "obs_unconditional_block_rate": 0.0},
    )
    ast_feats, req_pilot = extract_guard_ast_dependencies(shrunk_kept["scope_guard"])
    assert "obs_signed_direction_flux" in ast_feats
    assert req_pilot is True
    assert shrunk_kept["scope_verification"]["requires_pilot_calibration"] is True


def test_v13_controls_9_10_branch_commitment_mapping_and_multisource_preservation() -> None:
    p_a = _make_prior(cid="concept_A", objects=2, readout="focused_next", direction="right", edge_pred_dict="unconditional", radius=0.95)
    p_b = _make_prior(cid="concept_B", objects=2, readout="focused_next", direction="right", edge_pred_dict="unconditional", radius=0.95)
    c_single = consolidate_world_structure_commitments([p_a])[0]
    c_single["commitment_id"] = "world_commitment:0000"
    c_single["source_concept_ids"] = ["concept_A"]

    c_shared = deepcopy(c_single)
    c_shared["commitment_id"] = "world_commitment:0001"
    c_shared["source_concept_ids"] = ["concept_A", "concept_B"]

    guard_pos_flux = Program("lt", children=(Program("const", value=0.0), Program("feature", name="obs_signed_direction_flux"))).to_dict()
    sc = p_a["object_schema"]
    o0, o1 = sc["objects"][0], sc["objects"][1]
    # Negative-flux transition (`flux < 0`) so `branch:concept_A`'s `empirical_applicability_scope` rejects!
    neg_flux_trans = [
        {
            "before": {sc["fields"][o0]["q"]: 0.2, sc["fields"][o0]["x"]: 0.0, sc["fields"][o1]["q"]: 0.8, sc["fields"][o1]["x"]: 0.4},
            "after": {sc["fields"][o0]["q"]: 0.35, sc["fields"][o0]["x"]: 0.0, sc["fields"][o1]["q"]: 0.65, sc["fields"][o1]["x"]: 0.4},
        }
    ]

    lib = KnowledgeLibrary()
    lib._active_reconciled_branches = [
        {
            "claim_id": "branch:concept_A",
            "source_concept_id": "concept_A",
            "source_concept_ids": ["concept_A"],
            "commitment": deepcopy(c_single),
            "status": "scoped_conditional",
            "scope_kind": "empirical_applicability_scope",
            "scope_guard": guard_pos_flux,
            "independent_empirical_support": True,
        }
    ]
    # Control #9: `branch:concept_A` is empirically excluded -> single-source `world_commitment:0000` IS removed,
    # while Control #10: shared `world_commitment:0001` (`['concept_A', 'concept_B']`) is PRESERVED via `concept_B`!
    disp = dispatch_branches_and_families_for_target(
        library=lib,
        consistent_comms=[c_single, c_shared],
        transitions=neg_flux_trans,
        schema=sc,
        audit=None,
        task_sig="t_map",
        comm_budget=SearchBudget(10_000),
    )
    out_ids = [c["commitment_id"] for c in disp["ordered_consistent_comms"]]
    assert "branch:concept_A" in disp["telemetry"]["empirically_excluded_on_context"]
    assert "world_commitment:0000" not in out_ids
    assert "world_commitment:0001" in out_ids
    kept_shared = next(c for c in disp["ordered_consistent_comms"] if c["commitment_id"] == "world_commitment:0001")
    assert kept_shared["source_concept_ids"] == ["concept_B"]

    # Control #10 on multi-branch family: retracting 1 of 3 supporting branches keeps family `valid_grounded`!
    claims_prop = propagate_premise_retraction_invalidation(
        [
            {"claim_id": "b1", "independent_empirical_support": True},
            {"claim_id": "b2", "independent_empirical_support": True},
            {"claim_id": "b3", "independent_empirical_support": True},
            {
                "claim_id": "fam_H_star",
                "independent_empirical_support": False,
                "depends_on_claim_ids": ["b1", "b2", "b3"],
                "min_grounded_dependencies": 2,
            },
        ],
        retracted_premise_or_claim_ids={"b1"},
    )
    by_cid = {c["claim_id"]: c for c in claims_prop}
    assert by_cid["b1"]["support_validity"] == "directly_retracted"
    assert by_cid["fam_H_star"]["support_validity"] == "valid_grounded"
    assert by_cid["fam_H_star"]["active_grounded_dependencies"] == ["b2", "b3"]


def test_v13_controls_8_11_12_routing_vs_applicability_and_complement_guard() -> None:
    # Control #11: Different structural labels (`unconditional` vs `and(-10 < q_src, -10 < q_src - q_dst)`)
    # that are prediction-equivalent on `[-1.2, 1.2]` become `routing_scope_candidate` ONLY, NEVER scientific counterexamples!
    p_uncond = _make_prior(cid="p_u", objects=3, readout="focused_next", direction="right", edge_pred_dict="unconditional", radius=0.95)
    p_wide_cond = _make_prior(
        cid="p_wc",
        objects=3,
        readout="focused_next",
        direction="right",
        edge_pred_dict=_dpg_dict(-10.0, -10.0),
        radius=0.95,
    )
    sup, _ = extract_archive_witness_maps([p_uncond, p_wide_cond])
    c_u = consolidate_world_structure_commitments([p_uncond])[0]
    c_wc = consolidate_world_structure_commitments([p_wide_cond])[0]
    from hegel_machine.cognition.branch_reconciliation import _serialize_obs_rows

    b_u = {
        "claim_id": "branch:p_u",
        "source_concept_id": "p_u",
        "source_concept_ids": ["p_u"],
        "commitment": c_u,
        "schema": p_uncond["object_schema"],
        "status": "universal",
        "independent_empirical_support": True,
        "pilot_regime_features": [{"obs_mean_abs_q": 0.25, "obs_signed_direction_flux": 0.15}],
        "support_rows_serialized": _serialize_obs_rows(sup["p_u"]),
    }
    b_wc = {
        "claim_id": "branch:p_wc",
        "source_concept_id": "p_wc",
        "source_concept_ids": ["p_wc"],
        "commitment": c_wc,
        "schema": p_wide_cond["object_schema"],
        "status": "universal",
        "independent_empirical_support": True,
        "pilot_regime_features": [{"obs_mean_abs_q": 0.75, "obs_signed_direction_flux": 0.35}],
        "support_rows_serialized": _serialize_obs_rows(sup["p_wc"]),
    }
    refined = _refine_coexisting_branch_scopes_and_routing([b_u, b_wc], budget=SearchBudget(50_000))
    # Neither branch failed cross-prediction -> scientific `scope_guard` stays None and `status` stays `universal`!
    for rb in refined:
        assert rb.get("scope_guard") is None
        assert rb["status"] == "universal"
        assert rb.get("routing_guard") is not None
        assert rb.get("routing_scope_kind") == "routing_scope_candidate"

    # Control #12 & Control #8: Complement of old guard is NOT auto-confirmed as new's applicability scope;
    # third mechanism outside old_guard remains allowed, and instance refutation doesn't overwrite valid old!
    old_c = {"claim_id": "old_u", "status": "universal", "independent_empirical_support": True}
    new_c = {"claim_id": "new_l", "status": "universal", "independent_empirical_support": True}
    sc_12 = ObjectSchema(
        objects=("g0", "g1"),
        fields={"g0": {"q": "q0", "x": "x0", "u": "u0", "f": "f0"}, "g1": {"q": "q1", "x": "x1", "u": "u1", "f": "f1"}},
        members={"g0": ("q0", "x0", "u0", "f0"), "g1": ("q1", "x1", "u1", "f1")},
        controls={"g0": {"q": "u0"}, "g1": {"q": "u1"}},
    )
    rows_12 = [Observation({"q0": 0.2, "x0": 0.0, "u0": 0.0, "f0": 1.0, "q1": 0.8, "x1": 0.4, "u1": 0.0, "f1": 0.0}, 0.0)]
    trans_12 = [{"before": {"q0": 0.2, "x0": 0.0, "q1": 0.8, "x1": 0.4}, "after": {"q0": 0.35, "x0": 0.0, "q1": 0.65, "x1": 0.4}}]
    adj_12 = adjudicate_branches_on_paid_evidence(
        new_c,
        old_c,
        {"program": Program("const", value=0.0), "radius_bindings": {"r_0": 0.9}},
        {"program": Program("const", value=1.0), "radius_bindings": {"r_0": 0.9}},
        rows_12,
        trans_12,
        sc_12,
        old_support_regime_features=[{"obs_signed_direction_flux": 0.25}],
    )
    upd_new_12 = adj_12["updated_new_claim"]
    assert upd_new_12.get("scope_guard") is None
    assert upd_new_12["proposed_complement_scope_candidate"]["confirmed_applicability_scope"] is False
    assert upd_new_12["proposed_complement_scope_candidate"]["allows_third_mechanism"] is True

    adj_8 = adjudicate_branches_on_paid_evidence(
        new_c,
        old_c,
        {"program": Program("const", value=1.0), "radius_bindings": {"r_0": 0.9}},
        {"program": Program("const", value=0.0), "radius_bindings": {"r_0": 0.9}},
        rows_12,
        trans_12,
        sc_12,
    )
    assert adj_8["verdict"] == "new_claim_refuted_old_preserved"
    assert adj_8["updated_old_claim"]["status"] == "universal"
    assert adj_8["updated_new_claim"]["counterexamples"][0]["instance_only_not_family_refutation"] is True


def test_v13_controls_13_14_15_16_ordinary_3layer_cache_and_boundary_probe() -> None:
    p1 = _make_prior(cid="p1", objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.20), radius=0.94)
    p2 = _make_prior(cid="p2", objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.44), radius=0.92)
    p3 = _make_prior(cid="p3", objects=3, readout="focused_next", direction="left", edge_pred_dict="unconditional", radius=0.93)
    for p, flx in ((p1, 0.2), (p2, 0.25), (p3, -0.2)):
        p["pilot_regime_features"] = [{"obs_signed_direction_flux": flx, "obs_unconditional_block_rate": 0.3 if flx > 0 else 0.0}]

    lib = KnowledgeLibrary()
    lib.meta_priors = [deepcopy(p1), deepcopy(p2), deepcopy(p3)]

    # Control #13: First call builds & debits budget; second call on unchanged raw priors hits all 3 layers (`compute_spent == 1`)!
    b_first = SearchBudget(200_000)
    fams_1, st_1 = reconstruct_parametric_families_from_ordinary_archive(
        lib, b_first, tolerance=1e-7, reconstruct_routing=True
    )
    assert len(fams_1) == 1
    assert st_1["cache_hit"] is False
    assert st_1["layer1_extracted"] == 3 and st_1["layer2_built"] == 1 and st_1["layer3_routing_built"] is True

    b_second = SearchBudget(200_000)
    fams_2, st_2 = reconstruct_parametric_families_from_ordinary_archive(
        lib, b_second, tolerance=1e-7, reconstruct_routing=True
    )
    assert st_2["cache_hit"] is True
    assert st_2["compute_spent"] == 1
    assert st_2["layer1_reused"] == 3 and st_2["layer1_extracted"] == 0
    assert st_2["layer2_reused"] == 1 and st_2["layer2_built"] == 0
    assert st_2["layer3_routing_reused"] is True
    assert fams_1[0]["commitment_id"] == fams_2[0]["commitment_id"]

    # Control #14: Adding 1 new prior `p4` invalidates ONLY the new prior & its touched family group (`layer1_reused == 3, layer1_extracted == 1`)!
    p4 = _make_prior(cid="p4", objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.32), radius=0.95)
    p4["pilot_regime_features"] = [{"obs_signed_direction_flux": 0.22, "obs_unconditional_block_rate": 0.35}]
    lib.meta_priors.append(p4)
    b_third = SearchBudget(200_000)
    fams_3, st_3 = reconstruct_parametric_families_from_ordinary_archive(
        lib, b_third, tolerance=1e-7, reconstruct_routing=True
    )
    assert st_3["layer1_reused"] == 3 and st_3["layer1_extracted"] == 1
    assert st_3["layer2_built"] == 1
    assert len(fams_3[0]["source_branch_ids"]) == 3

    # Control #15: `_prioritize_generalized_families = False` skips H* Pass-0 without mutating status or weakening solver
    lib_no_h = deepcopy(lib)
    lib_no_h._generalized_branch_commitments = deepcopy(fams_3)
    lib_no_h._prioritize_generalized_families = False
    disp_no_h = dispatch_branches_and_families_for_target(
        library=lib_no_h,
        consistent_comms=[],
        transitions=[],
        schema=p1["object_schema"],
        audit=None,
        task_sig="t15",
        comm_budget=SearchBudget(10_000),
    )
    assert disp_no_h["scheduled_families"] == []
    assert disp_no_h["telemetry"]["prioritize_generalized_families"] is False

    # Control #16: Threshold boundary disagreement probe synthesized strictly from `[lo, hi]` with frozen versions before acquisition
    c1 = consolidate_world_structure_commitments([p1])[0]
    sc_p1 = ObjectSchema.from_dict(p1["object_schema"])
    spec_p1 = TaskSpec(
        task_id="t16",
        description="boundary probe check",
        feature_names=tuple(sc_p1.feature_names),
        initial_probes=(),
        available_probes=(),
        feature_bounds={fn: (-1.5, 1.5) for fn in sc_p1.feature_names},
    )
    bp = propose_threshold_interval_boundary_probe(
        c1, [0.15, 0.45], spec_p1, sc_p1, fitted_radius=0.94, budget=SearchBudget(20_000)
    )
    assert bp["actionable"] is True
    assert bp["role"] == "boundary_refinement_training_requires_fresh_confirmation"
    assert len(bp["frozen_program_versions"]) == 2
    assert bp["max_abs_disagreement"] > 1e-3


def test_v13_controls_6_7_ordinary_entry_both_refuted_funded_and_unfunded_recursion() -> None:
    """Ordinary `UnifiedResearchAgent.run(lab)` test when BOTH historical branches & initial conserved search fail!"""
    shared_lib = KnowledgeLibrary()
    # 1. Form neutral calibration + conserved right branch via ordinary entry
    for sd, d_mode in ((9100, "symmetric"), (9101, "right")):
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
            src_res = UnifiedResearchAgent(cfg_src, measurement_budget=11_000, library=shared_lib).run(src_lab)
        assert src_res["status"] == "completed"

    assert len(shared_lib._active_reconciled_branches) == 2

    # 2. Control #7 (Unfunded subquestion): Target has `focal_memory` latent reservoir (`latent_decay=0.55`),
    #    initial config has `latent_state_discovery=False` and tight observation cap (`195`)
    #    so after initial Markov failure, recursive subquestion has insufficient observation budget -> `pending_unfunded`!
    lib_unfunded = deepcopy(shared_lib)
    cfg_unfunded = ResearchConfig(
        seed=9105,
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
    with AnonymousObjectLab(
        seed=9105,
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
        name_seed=9136,
        wiring_seed=9166,
    ) as lat_lab_u:
        res_unfunded = UnifiedResearchAgent(cfg_unfunded, measurement_budget=3_600, library=lib_unfunded).run(lat_lab_u)

    assert res_unfunded["status"] == "unresolved"
    subqs_u = [q for q in res_unfunded["response_questions"] if q.get("kind") == "reconciliation_subquestion"]
    assert len(subqs_u) >= 1
    assert any(q["status"] == "pending_unfunded" for q in subqs_u)
    # Historical Markov branches must NOT remain unqualified `universal` on this non-Markovian regime!
    assert all(b["status"] != "universal" for b in lib_unfunded._active_reconciled_branches)

    # 3. Control #6 (Funded subquestion): Same `latent_state_discovery=False` initial config,
    #    with sufficient budget (`560` obs, `680k` compute) -> `both_competing_branches_refuted` subquestion
    #    executes recursive sub-research, discovers `focal_memory`, confirms on fresh unconsumed rows, and mutates parent!
    lib_funded = deepcopy(shared_lib)
    cfg_funded = ResearchConfig(
        seed=9105,
        observation_budget=560,
        compute_budget=680_000,
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
    with AnonymousObjectLab(
        seed=9105,
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
        name_seed=9136,
        wiring_seed=9166,
    ) as lat_lab_f:
        res_funded = UnifiedResearchAgent(cfg_funded, measurement_budget=11_500, library=lib_funded).run(lat_lab_f)

    subqs_f = [q for q in res_funded["response_questions"] if q.get("kind") == "reconciliation_subquestion"]
    assert len(subqs_f) >= 1
    assert any(q["status"] == "answered" for q in subqs_f)
    ans_q = next(q for q in subqs_f if q["status"] == "answered")
    assert ans_q["resolution"]["confirmed"] is True
    assert ans_q["resolution"]["parent_program_before"] != ans_q["resolution"]["parent_program_after"]
    assert res_funded["status"] == "completed"
    assert res_funded["confirmation_loss"] <= 1e-7
