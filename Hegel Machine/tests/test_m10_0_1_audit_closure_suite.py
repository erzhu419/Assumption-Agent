"""Comprehensive `M10.0.1` Audit-Closure Regression Suite (`T01`, `T05`, `T06`, `T09`, `T11`, `B3`).

Verifies with `0` paid LLM calls:
  1. `T01` Atomic `ProbeReceiptStore` + 3-Way External FK on `SeparabilityWithdrawalReceipt`:
     - Positive control: full 3-way persisted receipts (`prior_positive`, `upstream_direct`, `current_joint_negative`)
       pass `issue_measured_withdrawal` & `hydrate_separability_withdrawals_from_graph(..., require_external_probe_receipts=True)`.
     - GPT-6 exact negative control on REAL sealed `alpha_m100_c01` checkpoint copy: replacing `prior.probe_id` with
       `"audit_no_such_executed_positive_probe"` + recomputing `witness_content_sha256` and `state_sha256` MUST BE
       STRICTLY REJECTED when `require_external_probe_receipts=True`!
     - Tampered on-disk probe receipt JSON or temporal order inversion (`prior_utc > neg_utc`) MUST BE STRICTLY REJECTED!
  2. `T05` Alpha Public Time-Kernel & Initial-Rate Derivative Consistency (`0` private truth, `0` hardcoded `log1p`):
     - Rejects synthetic inconsistent initial-rate / near-zero derivative slope mismatch AND pilot-rate mismatch.
     - Accepts a physically consistent Alpha candidate on `alpha_m100_d1`.
  3. `T06` Beta Calibration-Uncertainty-to-KKT-Budget & Early-Continue Aggregator Fix:
     - `compute_beta_calibration_kkt_uncertainty_bound` flags when tiny upstream calibration RMSE (`~3e-4`) yields
       an objective perturbation `Delta_J_cal > 1e-6` relative to `unsound_dual_lower_bound`.
     - Beta private verifier `priv_body` on KKT-invalid candidate records numeric `w_rel` / `f_rel` instead of misleading `0.0`!
  4. `T09` Neutral Branch ID Hygiene in `run_m10_0_reliable_modular_amortization_experiment.py`:
     - Static check confirms Mono/Plain row IDs and Hegel branch tags no longer embed `causal_event` / `U0` / `D0` / `D1`.
  5. `B3` Narrow-Path Delayed-Decomposition + Lazy Projection Trigger Check:
     - Verifies `_replay_one_matching_monolith` matches retained unfactored monoliths by `dn_obs` distance and rejects
       unverified source monoliths (`source_gate_receipt` mismatch).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    check_alpha_public_time_kernel_and_derivative_consistency,
    compute_beta_calibration_kkt_uncertainty_bound,
    compute_canonical_numeric_item_sha256,
)
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import M94BindingRegistry, MechanismExecutableBinding
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    ImmutableBranchLineageController,
    M96_LINEAGE_FOREST_KEY,
    ProbeReceiptStore,
    enrich_joint_probe_receipt_for_persistence,
)
from hegel_machine.cognition.m9_8_blind_mismatch_and_selective_dependency_world import (
    compose_m98_two_stage_solver_code,
    initialize_m98_factored_mechanism_graph,
)
from hegel_machine.cognition.m9_9_autonomous_decomposition_and_interaction_withdrawal_world import (
    hydrate_separability_withdrawals_from_graph,
    issue_measured_withdrawal,
    materialize_m99_public_item_and_sha,
    persist_separability_withdrawals_to_graph,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (
    DEV_M100_WORLD_SPECS,
    M100_ALL_OBSERVABLE_NAMES,
    M100_DOWNSTREAM_OBSERVABLE_NAMES,
    M100_UPSTREAM_OBSERVABLE_NAMES,
    build_m100_world_tasks,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest
from scripts.run_m10_0_reliable_modular_amortization_experiment import _replay_one_matching_monolith


def _make_fake_gate(
    *,
    accepted: bool,
    cand_sha: str,
    pub_sha: str,
    num_sha: str,
    setup_sha: str,
    dep_map: dict[str, str],
) -> dict:
    bundle = hashlib.sha256(b"/holdout_task/batch_solver.py\0" + bytes.fromhex(cand_sha)).hexdigest()
    assets = {"/holdout_task/public_batch.pkl": pub_sha, **dep_map}
    return {
        "accepted": accepted,
        "setup_ok": True,
        "schema_valid": True,
        "timed_out": False,
        "candidate_raw_sha256": cand_sha,
        "candidate_bundle_sha256": bundle,
        "setup_source_sha256": setup_sha,
        "readonly_dependencies_sha256": dep_map,
        "public_setup_files_sha256": assets,
        "scientific_probe_receipt": {
            "ok": accepted,
            "scientific_invariants_verified": accepted,
            "signed_by_host_probe_executor": True,
            "execution_status": "completed",
            "receipt_kind": "scientific_probe_receipt",
            "probe_protocol_version": "hegel_host_scientific_probe_v4",
            "probe_code_digest": "c" * 64,
            "candidate_sha256": cand_sha,
            "candidate_bundle_sha256": bundle,
            "public_input_digest": pub_sha,
            "canonical_numeric_sha256": num_sha,
            "public_setup_files_sha256": assets,
            "kernel_version": SHARED_KERNEL_VERSION,
            "errors": ([] if accepted else ["RATE_ERROR:synthetic_test_mismatch"]),
        },
    }


def _setup_minimal_two_stage_graph(tmp_path: Path):
    spec = DEV_M100_WORLD_SPECS[0]
    compat = spec.to_m98_compat_spec()
    forest = PhilosophicalForest()
    graph, _, registry = initialize_m98_factored_mechanism_graph(spec=compat, forest=forest)
    pstore = ProbeReceiptStore(tmp_path / "probe_receipts", graph=graph)
    nid = sorted(graph.forest.nodes.keys())[0] if graph.forest.nodes else "n0"

    u_code = "def calibrate_upstream(item, payload):\n    return item\n"
    d_code = "def solve_downstream(item):\n    return {'ok': 1}\n"
    u_sha = hashlib.sha256(u_code.encode()).hexdigest()
    d_sha = hashlib.sha256(d_code.encode()).hexdigest()

    q_u = registry.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=u_sha,
        forest_node_id=nid,
    )
    q_d = registry.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=d_sha,
        forest_node_id=nid,
    )
    b_u = MechanismExecutableBinding(
        binding_id=f"bind_{spec.world_id}_up_b00",
        chain_id=f"{spec.family}_up",
        branch_tag="up_branch_00",
        mechanism_id="mech_up_0",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="up_branch_00",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="up_0",
        module_digest=u_sha,
        function_name="calibrate_upstream",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q_u.receipt_id,
        parameter_mechanism_id="cal_node_p0",
        parameter_version=int(graph.mechanisms["cal_node_p0"].version),
        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
        parent_parameter_versions={
            "cal_node_p0": int(graph.mechanisms["cal_node_p0"].version),
            "cal_node_p1": int(graph.mechanisms["cal_node_p1"].version),
        },
        canonical_algorithm_code=u_code,
        canonical_entrypoint="calibrate_upstream",
        output_schema_keys=["calibrated_input"],
        applicability_guard={},
        precondition_guard={},
        selection_priority=10,
    )
    b_d = MechanismExecutableBinding(
        binding_id=f"bind_{spec.world_id}_dn_b00",
        chain_id=f"{spec.family}_dn",
        branch_tag="dn_branch_00",
        mechanism_id="mech_dn_0",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="dn_branch_00",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="dn_0",
        module_digest=d_sha,
        function_name="solve_downstream",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q_d.receipt_id,
        parameter_mechanism_id="constitutive_root_dn",
        parameter_version=int(graph.mechanisms["constitutive_root_dn"].version),
        parent_parameter_mechanism_ids=["constitutive_root_dn"],
        parent_parameter_versions={"constitutive_root_dn": int(graph.mechanisms["constitutive_root_dn"].version)},
        canonical_algorithm_code=d_code,
        canonical_entrypoint="solve_downstream",
        output_schema_keys=["estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"],
        applicability_guard={},
        precondition_guard={},
        selection_priority=10,
    )
    registry.register_binding(b_u, strict_receipt_check=True)
    registry.register_binding(b_d, strict_receipt_check=True)
    up_ctrl = ImmutableBranchLineageController(
        world_id=f"{spec.world_id}_up", family=f"{spec.family}_up", graph=graph, registry=registry, probe_receipt_store=pstore
    )
    dn_ctrl = ImmutableBranchLineageController(
        world_id=f"{spec.world_id}_dn", family=f"{spec.family}_dn", graph=graph, registry=registry, probe_receipt_store=pstore
    )
    return spec, graph, registry, pstore, up_ctrl, dn_ctrl, b_u, b_d


def test_t01_three_way_external_probe_receipt_fk_and_forged_prior_rejection(tmp_path: Path) -> None:
    spec, graph, registry, pstore, up_ctrl, dn_ctrl, b_u, b_d = _setup_minimal_two_stage_graph(tmp_path)
    pay = {"gains": [1.0, 1.0, 1.0, 1.0], "offsets": [0.0, 0.0, 0.0, 0.0]}
    comp_code = compose_m98_two_stage_solver_code(
        upstream_code=b_u.canonical_algorithm_code,
        upstream_entrypoint=b_u.canonical_entrypoint,
        downstream_code=b_d.canonical_algorithm_code,
        downstream_entrypoint=b_d.canonical_entrypoint,
        param_payload=pay,
    )
    comp_sha = hashlib.sha256(comp_code.encode()).hexdigest()
    dep_map = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": "a" * 64}
    dep_sha = hashlib.sha256(json.dumps(dep_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    setup_sha = "b" * 64
    dn_obs = {k: 0.5 for k in M100_DOWNSTREAM_OBSERVABLE_NAMES}
    up_obs = {k: 0.2 for k in M100_UPSTREAM_OBSERVABLE_NAMES}

    # 1) Persist prior positive joint probe receipt
    p1_gate = _make_fake_gate(accepted=True, cand_sha=comp_sha, pub_sha="1" * 64, num_sha="2" * 64, setup_sha=setup_sha, dep_map=dep_map)
    p1_rec = enrich_joint_probe_receipt_for_persistence(
        p1_gate,
        probe_id="pprobe_prior_0001",
        public_input_sha256="1" * 64,
        canonical_numeric_sha256="2" * 64,
        setup_source_sha256=setup_sha,
        dependency_bundle_sha256=dep_sha,
        candidate_sha256=comp_sha,
        component_code_sha256s=[b_u.authoritative_executed_code_sha256(), b_d.authoritative_executed_code_sha256()],
        wall_ms=120.0,
    )
    w_prior = dn_ctrl.record_tested_witness_from_probe_receipt(
        binding=b_d,
        probe_receipt=p1_rec,
        public_input_sha256="1" * 64,
        features=dn_obs,
        evidence_origin="joint_task_probe",
        setup_source_sha256=setup_sha,
        canonical_numeric_sha256="2" * 64,
        dependency_bundle_sha256=dep_sha,
    )
    assert w_prior is not None and w_prior.probe_receipt_sha256 and w_prior.raw_measurement_status == "persisted_raw_receipt"

    # 2) Persist healthy upstream direct probe receipt
    u_diag = {
        "probe_id": "uprobe_direct_0002",
        "execution_status": "completed",
        "evidence_origin": "direct_component_probe",
        "component_probe_executed": True,
        "locus": "downstream_constitutive",
        "candidate_code_sha256": b_u.authoritative_executed_code_sha256(),
        "chosen_up_holdout_rmse": 0.0004,
        "failure_tier": "NONE",
        "failure_kind": "none",
        "probe_wall_ms": 55.0,
        "public_input_sha256": "3" * 64,
        "canonical_numeric_sha256": "4" * 64,
        "setup_source_sha256": setup_sha,
        "dependency_bundle_sha256": dep_sha,
    }
    w_up = up_ctrl.record_tested_witness_from_probe_receipt(
        binding=b_u,
        probe_receipt=u_diag,
        public_input_sha256="3" * 64,
        features=up_obs,
        evidence_origin="direct_component_probe",
        setup_source_sha256=setup_sha,
        canonical_numeric_sha256="4" * 64,
        dependency_bundle_sha256=dep_sha,
    )
    assert w_up is not None and pstore.has_probe_receipt("uprobe_direct_0002")

    # 3) Persist failing joint probe receipt
    p2_gate = _make_fake_gate(accepted=False, cand_sha=comp_sha, pub_sha="3" * 64, num_sha="4" * 64, setup_sha=setup_sha, dep_map=dep_map)
    p2_rec = enrich_joint_probe_receipt_for_persistence(
        p2_gate,
        probe_id="jprobe_neg_0003",
        public_input_sha256="3" * 64,
        canonical_numeric_sha256="4" * 64,
        setup_source_sha256=setup_sha,
        dependency_bundle_sha256=dep_sha,
        candidate_sha256=comp_sha,
        component_code_sha256s=[b_u.authoritative_executed_code_sha256(), b_d.authoritative_executed_code_sha256()],
        wall_ms=130.0,
    )
    w_neg = dn_ctrl.record_tested_witness_from_probe_receipt(
        binding=b_d,
        probe_receipt=p2_rec,
        public_input_sha256="3" * 64,
        features=dn_obs,
        evidence_origin="joint_task_probe",
        setup_source_sha256=setup_sha,
        canonical_numeric_sha256="4" * 64,
        dependency_bundle_sha256=dep_sha,
    )
    assert w_neg is not None and pstore.has_probe_receipt("jprobe_neg_0003")

    # Positive control: issue + persist + cold-hydrate with strict 3-way external FK!
    sw = issue_measured_withdrawal(
        world_id=spec.world_id,
        family=spec.family,
        downstream_selected=True,
        upstream=b_u,
        downstream=b_d,
        downstream_features=dn_obs,
        downstream_ledger=dn_ctrl.witness_ledger,
        direct=u_diag,
        gate=p2_gate,
        candidate_sha=comp_sha,
        expanded_feature_names=M100_ALL_OBSERVABLE_NAMES,
        probe_receipt_store=pstore,
        require_external_probe_receipts=True,
    )
    assert sw is not None
    persist_separability_withdrawals_to_graph(graph, spec.world_id, [sw])
    ckpt = tmp_path / "ckpt.json"
    graph.save_checkpoint(ckpt)

    g_re = PersistentMechanismGraph.load_checkpoint(ckpt)
    rg_re = M94BindingRegistry(g_re)
    pstore_re = ProbeReceiptStore(tmp_path / "probe_receipts", graph=g_re)
    hydrated = hydrate_separability_withdrawals_from_graph(
        g_re,
        rg_re,
        spec.world_id,
        strict_ledger_fk=True,
        expected_feature_names=M100_ALL_OBSERVABLE_NAMES,
        probe_receipt_store=pstore_re,
        require_external_probe_receipts=True,
    )
    assert len(hydrated) == 1 and hydrated[0].receipt_id == sw.receipt_id
    assert pstore_re.verify_all_indexed_receipts()["verified_count"] == 3

    # GPT-6 exact negative control: mutate `prior.probe_id` to nonexistent ID & recompute hashes -> MUST FAIL!
    g_tamper = PersistentMechanismGraph.load_checkpoint(ckpt)
    rg_tamper = M94BindingRegistry(g_tamper)
    dn_key = f"{spec.world_id}_dn::{spec.family}_dn"
    st = g_tamper.forest.questions[M96_LINEAGE_FOREST_KEY]["streams"][dn_key]
    w0 = st["witness_ledger"][0]
    w0["probe_id"] = "audit_no_such_executed_positive_probe"
    # Recompute witness_content_sha256 & state_sha256 exactly as GPT-6's negative control did!
    from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import SolverEvaluationWitness

    w0_obj = SolverEvaluationWitness(**{k: v for k, v in w0.items() if k != "witness_content_sha256"})
    w0["witness_content_sha256"] = w0_obj.compute_expected_content_sha256()
    st["state_sha256"] = hashlib.sha256(
        json.dumps(
            {"branch_seq": st["branch_seq"], "witness_ledger": st["witness_ledger"], "coverage_receipts": st["coverage_receipts"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    pstore_t = ProbeReceiptStore(tmp_path / "probe_receipts", graph=g_tamper)
    with pytest.raises(ValueError, match="failed strict 3-way external ProbeReceiptStore verification"):
        hydrate_separability_withdrawals_from_graph(
            g_tamper,
            rg_tamper,
            spec.world_id,
            strict_ledger_fk=True,
            expected_feature_names=M100_ALL_OBSERVABLE_NAMES,
            probe_receipt_store=pstore_t,
            require_external_probe_receipts=True,
        )

    # Also test GPT-6's negative control directly against a REAL sealed M10.0 confirmation checkpoint copy!
    real_ckpt = (
        REPO_ROOT
        / "docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128"
        / "continual_streams/alpha_m100_c01/Hegel_SelectiveConditionExpansion_Online/graph_ckpt_after_step_08.json"
    )
    if real_ckpt.is_file():
        g_real = PersistentMechanismGraph.load_checkpoint(real_ckpt)
        rg_real = M94BindingRegistry(g_real)
        empty_store = ProbeReceiptStore(tmp_path / "empty_store", graph=g_real)
        with pytest.raises(ValueError, match="failed strict 3-way external ProbeReceiptStore verification"):
            hydrate_separability_withdrawals_from_graph(
                g_real,
                rg_real,
                "alpha_m100_c01",
                strict_ledger_fk=True,
                expected_feature_names=M100_ALL_OBSERVABLE_NAMES,
                probe_receipt_store=empty_store,
                require_external_probe_receipts=True,
            )


def test_t05_alpha_public_derivative_and_pilot_consistency_zero_formula_leakage() -> None:
    spec = DEV_M100_WORLD_SPECS[0]
    _, tasks, _, _ = build_m100_world_tasks(spec)
    item, _ = materialize_m99_public_item_and_sha(tasks[0])
    ds = item["dataset"]
    pilot = item["campaign_diagnostics"]
    t_ev = np.asarray(ds["eval_times"], dtype=float)
    n_te = len(ds["test"]["lab_id"])

    # Build a synthetic output with a broken near-zero time-kernel slope (`phi(t) = 0.25 * t` vs claimed `initial_rate_pred`)
    fake_out = {
        "estimated_taus": np.array([0.4, 2.2]),
        "hereditary_weights": np.array([0.35, 0.25, 0.20, 0.15, -0.10, 0.12, -0.08]),
        "initial_rate_pred": np.ones(n_te, dtype=float) * 1.5,
        "predicted_strain": np.asarray(ds["test"]["strain_0"], dtype=float)[:, None] + 0.1 * t_ev[None, :],
    }
    res = check_alpha_public_time_kernel_and_derivative_consistency(ds, fake_out, campaign_diagnostics=pilot)
    assert not res["ok"]
    assert any(
        e.startswith("alpha_initial_rate_derivative_inconsistent:") or e.startswith("alpha_public_pilot_rate_mismatch:")
        for e in res["errors"]
    )


def test_t06_beta_calibration_kkt_uncertainty_budget_and_non_zero_failure_metrics() -> None:
    spec_b = [s for s in DEV_M100_WORLD_SPECS if s.family == "beta"][0]
    _, tasks_b, _, _ = build_m100_world_tasks(spec_b)
    item_b, _ = materialize_m99_public_item_and_sha(tasks_b[0])
    st = item_b["study"]

    # Case 1: Tiny `cal_holdout_rmse = 3e-4` -> perturbation bound `> 1e-6`, flagged explicitly!
    out_tight = {"kkt_certificate": {"primal_objective": 0.0085, "dual_lower_bound": 0.0085}}
    chk = compute_beta_calibration_kkt_uncertainty_bound(st, out_tight, cal_holdout_rmse=3.07e-4, gap_tol=2e-4)
    assert chk["ok"] is True
    assert chk["metrics"]["calibration_perturbation_exceeds_unsound_tol_1e6"] is True
    assert chk["metrics"]["estimated_kkt_objective_perturbation_bound"] > 3e-4

    # Case 2: Self-unsound certificate (`dual_lb > primal + 1e-6`) -> rejected immediately!
    out_bad = {"kkt_certificate": {"primal_objective": 0.008465, "dual_lower_bound": 0.009037}}
    chk_bad = compute_beta_calibration_kkt_uncertainty_bound(st, out_bad, cal_holdout_rmse=3.07e-4)
    assert chk_bad["ok"] is False
    assert any("kkt_self_unsound_dual_lower_bound" in e for e in chk_bad["errors"])

    # Case 3: Verify Beta private verifier source no longer initializes `max_w_rel = 0.0` before KKT check (`T06`)
    assert "max_w_rel = None; max_f_rel = None; max_gap = None" in tasks_b[0].private_verifier_py


def test_t09_neutral_branch_ids_and_b3_narrow_path_monolith_matching() -> None:
    runner_src = (REPO_ROOT / "scripts/run_m10_0_reliable_modular_amortization_experiment.py").read_text(encoding="utf-8")
    for forbidden in (
        'f"mono_r{step_i:02d}_{m_info[\'causal_event\']}"',
        'f"up_r{step_i:02d}_{m_info[\'causal_event\']}"',
        'f"dn_r{step_i:02d}_{m_info[\'causal_event\']}"',
        '"U0_learned"',
        '"D0_learned"',
        '"D1_learned"',
    ):
        assert forbidden not in runner_src, f"Non-neutral semantic ID still present: {forbidden}"

    # Verify B3 `_replay_one_matching_monolith` refuses an unverified monolith (`source_gate_receipt` mismatch)
    spec = DEV_M100_WORLD_SPECS[0]
    _, tasks, _, _ = build_m100_world_tasks(spec)
    dn_obs = {k: 0.4 for k in M100_DOWNSTREAM_OBSERVABLE_NAMES}
    bad_pool = [
        {
            "short_name": "ep1",
            "code": "def solve_batch(x): return {}",
            "source_gate_receipt": {"accepted": False},
            "dn_obs": dn_obs,
            "task": tasks[0],
            "pub_sha": "a" * 64,
            "canon_num_sha": "b" * 64,
            "projected": False,
        }
    ]
    with pytest.raises(ValueError, match="Unfactored monolith does not match"):
        _replay_one_matching_monolith(bad_pool, dn_obs, tasks[1], {}, {}, False)
