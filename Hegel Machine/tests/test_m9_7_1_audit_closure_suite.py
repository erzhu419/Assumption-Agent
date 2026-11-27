"""Focused `M9.7.1` Audit Closure & Contract Hardening Suite (`A06, A07, A08, A09, A10, A11, A16`, `0` Paid LLM Calls).

Verifies:
  1. `A07`: Missing expected key in `parent_parameter_versions` fails closed (`PARENT_VERSION_SET_MISMATCH`), and legacy incomplete checkpoint records migrate to `unknown_requires_reverification`.
  2. `A08`: `EmpiricalCoverageReceipt` is strictly unidirectional (`new` covers `old` does NOT permit `old` guard to evaluate `TRUE` on `new` witnesses!), and tampered receipt fields fail content-addressed integrity verification.
  3. `A09`: Strict 4-tier failure classification (`INFRA_FAILURE -> SCHEMA_FAILURE -> DOMAIN_REJECTION -> SCIENTIFIC_FAILURE`) ensures timeouts NEVER become scientific negative witnesses or authorize conflict splits.
  4. `A10`: Cold-restart checkpoint persistence & hydration restores `witness_ledger`, `coverage_receipts`, and `branch_seq` inside `PersistentMechanismGraph`, rejects tampered state digests, and reproduces identical `R1 / U_iso / G_perm / M_couple` branch selections after cold stop at `C2` (`0` LLM calls).
  5. `A11`: Shared stratified per-channel `2/3–1/3` calibration estimator yields exact `0.0` gain/bias difference between `Flat` and `PersistentMechanismGraph` across all 7 dev episodes.
  6. `A06`: Alpha verifier separates `NON_FINITE`, `PARAMETER_ORDER`, `STRAIN_ERROR`, and `RATE_ERROR`.
  7. `A16`: `historical_capabilities` indexes `M6–M9.7.1` modules, entrypoints, and sealed evidence while preserving all `181` historical version entries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_6_autonomous_structure_continual_world as m96,
    m9_7_non_nested_counterfactual_lineage_world as m97,
)
from hegel_machine.cognition.historical_capabilities import historical_capabilities
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def _materialize_public_item(task) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory(prefix="m971_test_") as td:
        exec(compile(task.public_setup_py.replace("/holdout_task", td), "<m971_pub>", "exec"), {})
        raw = (Path(td) / f"public_{task.short_name}.pkl").read_bytes()
        return pickle.loads(raw)[0], hashlib.sha256(raw).hexdigest()


def test_a07_missing_secondary_parent_version_key_fails_closed_and_legacy_hydrates_unverified():
    spec = m97.DEV_M97_WORLD_SPECS[0]
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, _, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)
    code = "def solve(d):\n    return {'v': 0}\n"
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    q = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha,
        forest_node_id=nid,
    )
    incomplete_b = m94.MechanismExecutableBinding(
        binding_id="probe_b00",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id="probe_m00",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="c0",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="m",
        module_digest=sha,
        function_name="solve",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q.receipt_id,
        parameter_mechanism_id="cal_node_p0",
        parameter_version=1,
        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
        parent_parameter_versions={"cal_node_p0": 1},  # Missing `cal_node_p1`!
        canonical_algorithm_code=code,
        canonical_entrypoint="solve",
        output_schema_keys=m97.expected_m97_public_output_keys("alpha"),
    )
    with pytest.raises(ValueError, match="PARENT_VERSION_SET_MISMATCH"):
        reg.register_binding(incomplete_b, strict_receipt_check=True)
    with pytest.raises(ValueError, match="PARENT_VERSION_SET_MISMATCH"):
        m94.verify_binding_identity_and_foreign_keys(
            incomplete_b, graph=graph, registry=reg, require_closure_code=True, strict_receipt_check=True
        )

    # Legacy checkpoint hydration migrates incomplete parent version vector to `unknown_requires_reverification`
    blob = {
        "schema": "hegel_m9_6_1_bindings_v4",
        "bindings": {"probe_b00": incomplete_b.to_dict()},
        "qualification_receipts": {q.receipt_id: q.to_dict()},
        "parameter_epochs": {},
        "binding_events": [],
    }
    graph.forest.questions[m94.M94_BINDINGS_FOREST_KEY] = blob
    reg_reloaded = m94.M94BindingRegistry(graph)
    assert reg_reloaded.bindings["probe_b00"].qualification_status == "unknown_requires_reverification"
    with pytest.raises(ValueError, match="qualification_status"):
        m94.verify_binding_identity_and_foreign_keys(
            reg_reloaded.bindings["probe_b00"], graph=graph, registry=reg_reloaded, require_closure_code=True
        )


def test_a08_directional_empirical_coverage_and_content_addressed_receipt_integrity():
    # `new` covers `old` on `old`'s witness (`f=0.0`), so `new` may be `TRUE` on `old` (`f >= -1`),
    # MUST NOT allow `old` (`{}` = TRUE everywhere) to be `TRUE` on `new`'s exclusive witness (`f=1.0`)!
    sym_attempt = m97.verify_m97_multibranch_witness_partition(
        branch_guards={"old": {}, "new": {}},
        positive_witness_features_by_branch={"old": [{"obs_01": 0.0}], "new": [{"obs_01": 1.0}]},
        covered_branch_pairs={("new", "old")},
    )
    assert sym_attempt["valid"] is False
    assert any("new_pos_0:other_guard_old=TRUE_overlap" in r for r in sym_attempt["reasons"])

    # Proper directional partition: `old` holds ONLY on `obs_01 <= 0.1`, while covering `new` holds on `obs_01 >= -0.1`!
    asym_valid = m97.verify_m97_multibranch_witness_partition(
        branch_guards={
            "old": {"all_of": [{"feature_name": "obs_01", "op": "<=", "threshold": 0.1}]},
            "new": {"all_of": [{"feature_name": "obs_01", "op": ">=", "threshold": -0.1}]},
        },
        positive_witness_features_by_branch={"old": [{"obs_01": 0.0}], "new": [{"obs_01": 1.0}]},
        covered_branch_pairs={("new", "old")},
        negative_witness_features_by_branch={"old": [{"obs_01": 1.0}]},
    )
    assert asym_valid["valid"] is True

    # Content-addressed `EmpiricalCoverageReceipt` tamper check
    rec = m96.EmpiricalCoverageReceipt(
        receipt_id="",
        new_binding_id="b01",
        new_solver_sha256="a" * 64,
        old_binding_id="b00",
        old_solver_sha256="b" * 64,
        witness_ids=["wit_001"],
        parameter_versions_by_witness=[{"cal_node_p0": 1, "cal_node_p1": 1}],
        verifier_version="hegel_shared_math_handoff_kernel_v3_m961_three_layer",
        status="supported_on_witnesses",
        passed_witness_count=1,
        evaluated_witness_count=1,
        total_cpu_probe_wall_ms=12.5,
        mean_probe_wall_ms_ratio=1.0,
        witness_input_sha256s=["c" * 64],
        parameter_payload_sha256s=["d" * 64],
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    rec.verify_integrity()
    tampered = rec.to_dict()
    tampered["status"] = "refuted"
    with pytest.raises(ValueError, match="Tampered EmpiricalCoverageReceipt"):
        m96.EmpiricalCoverageReceipt.from_dict(tampered)


def test_a09_timeout_never_classified_as_scientific_counterexample():
    spec = m97.DEV_M97_WORLD_SPECS[0]
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, _, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)
    code = "def solve(d):\n    return {}\n"
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    q = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha,
        forest_node_id=nid,
    )
    b = m94.MechanismExecutableBinding(
        binding_id="bind_timeout_test_b00",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id="mech_timeout_test_b00",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="c0",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="m",
        module_digest=sha,
        function_name="solve",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q.receipt_id,
        parameter_mechanism_id="cal_node_p0",
        parameter_version=1,
        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
        parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
        canonical_algorithm_code=code,
        canonical_entrypoint="solve",
        output_schema_keys=m97.expected_m97_public_output_keys("alpha"),
    )
    reg.register_binding(b, strict_receipt_check=True)
    ctrl = m96.ImmutableBranchLineageController(world_id="timeout_test", family="alpha", graph=graph, registry=reg)
    # Record a prior positive witness (exact `code_probes.py` scenario!)
    ctrl.record_tested_witness(binding=b, public_input_sha256="prior_ok", passed=True, failure_kind="none", features={"obs_01": 0.0})

    s_task, _, _ = m97.build_m97_world_tasks(spec)
    sample_pub, _ = _materialize_public_item(s_task)
    fake_timeout = {
        "accepted": False,
        "schema_valid": False,
        "schema_stdout": "",
        "schema_stderr": "[TIMEOUT: command exceeded 45s limit]",
        "scientific_probe_receipt": None,
    }
    with patch.object(m97, "execute_single_pre_llm_public_gate_probe_in_container", return_value=fake_timeout):
        trial = m97.run_m97_equal_capability_pre_llm_trial(
            arm="Hegel_DemandGated_Graph",
            spec=spec,
            task=s_task,
            public_item=sample_pub,
            public_input_sha256="timeout_sha",
            prior_archive_entries=[],
            lineage_ctrl=ctrl,
            graph=graph,
            registry=reg,
        )
    assert trial["tested_witness"]["failure_kind"] == "timeout_or_infra"
    assert ctrl.scientific_negative_witnesses_for("bind_timeout_test_b00") == []
    with pytest.raises(ValueError, match="without a genuine tested scientific negative witness"):
        ctrl.mint_immutable_conflict_branch(
            failed_binding_id="bind_timeout_test_b00",
            new_canonical_code="def solve2(d):\n    return {}\n",
            new_entrypoint="solve2",
            forest_node_id=nid,
            qualification_protocol="hegel_public_scientific_gate_v4",
            output_schema_keys=m97.expected_m97_public_output_keys("alpha"),
            validated_parent_ids=["cal_node_p0", "cal_node_p1"],
            new_applicability_guard={"all_of": [{"feature_name": "obs_01", "op": ">", "threshold": 0.1}]},
            updated_failed_branch_applicability_guard=None,
            violated_component_id="c0",
            preserved_component_ids=[],
            selection_priority=20,
        )


def test_a10_cold_restart_hydration_and_post_c2_trajectory_equivalence(tmp_path):
    spec = m97.DEV_M97_WORLD_SPECS[0]
    wid = spec.world_id
    s_task, stream_tasks, manifest = m97.build_m97_world_tasks(spec)
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, planner, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)

    c0_code = m97.build_m97_reference_dev_solver_code("alpha", "solve_c0", "C0")
    c1_code = m97.build_m97_reference_dev_solver_code("alpha", "solve_c1", "C1")
    c2_code = m97.build_m97_reference_dev_solver_code("alpha", "solve_c2", "C2")
    uni_keys = m97.expected_m97_public_output_keys("alpha")

    c0_sha = hashlib.sha256(c0_code.encode("utf-8")).hexdigest()
    q0 = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=c0_sha,
        forest_node_id=nid,
    )
    b00 = m94.MechanismExecutableBinding(
        binding_id=f"bind_{wid}_b00",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id=f"mech_{wid}_b00",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="c0",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="s0",
        module_digest=c0_sha,
        function_name="solve_c0",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q0.receipt_id,
        parameter_mechanism_id="cal_node_p0",
        parameter_version=1,
        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
        parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
        canonical_algorithm_code=c0_code,
        canonical_entrypoint="solve_c0",
        output_schema_keys=uni_keys,
        selection_priority=10,
    )
    reg.register_binding(b00, strict_receipt_check=True)
    ctrl = m96.ImmutableBranchLineageController(world_id=wid, family="alpha", graph=graph, registry=reg)

    s_pub, s_sha = _materialize_public_item(s_task)
    s_feats = m97.compute_m97_primitive_observables("alpha", s_pub, {"stress_gains": [1.0] * 4, "stress_biases": [0.0] * 4})
    ctrl.record_tested_witness(binding=b00, public_input_sha256=s_sha, passed=True, failure_kind="none", features=s_feats)

    # Simulate Steps 1..3 (`C1`, `R0`, `C2`), recording negative witnesses, coverage receipts, and minting `b01` + `b02`
    g_b00 = {
        "all_of": [
            {"feature_name": "obs_01_past_series_exceedance_integral", "op": "<=", "threshold": 0.05},
            {"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50},
        ]
    }
    g_b01 = {
        "all_of": [
            {"feature_name": "obs_01_past_series_exceedance_integral", "op": ">", "threshold": 0.05},
            {"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50},
        ]
    }
    g_b02 = {"all_of": [{"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": ">", "threshold": 0.50}]}

    for step_idx in range(3):
        t_step = stream_tasks[step_idx]
        ev_name = manifest[step_idx]["causal_event"]
        pub_i, sha_i = _materialize_public_item(t_step)
        m97.update_m97_permuted_calibration_graph(
            spec=spec, episode_id=f"{wid}_{t_step.short_name}", public_item=pub_i, graph=graph, planner=planner, registry=reg
        )
        pay_i = m97.assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=["cal_node_p0", "cal_node_p1"])
        feats_i = m97.compute_m97_primitive_observables("alpha", pub_i, pay_i)
        if ev_name == "C1":
            ctrl.record_tested_witness(binding=b00, public_input_sha256=sha_i, passed=False, failure_kind="scientific_or_numerical", features=feats_i)
            cov1 = m96.EmpiricalCoverageReceipt(
                receipt_id="",
                new_binding_id=f"bind_{wid}_b01",
                new_solver_sha256=hashlib.sha256(c1_code.encode("utf-8")).hexdigest(),
                old_binding_id=b00.binding_id,
                old_solver_sha256=c0_sha,
                witness_ids=[ctrl.witness_ledger[0].witness_id],
                parameter_versions_by_witness=[dict(b00.parent_parameter_versions)],
                verifier_version="hegel_shared_math_handoff_kernel_v3_m961_three_layer",
                status="refuted",
                passed_witness_count=0,
                evaluated_witness_count=1,
                total_cpu_probe_wall_ms=10.0,
                mean_probe_wall_ms_ratio=1.0,
                witness_input_sha256s=[s_sha],
                parameter_payload_sha256s=["p0"],
            )
            cov1.receipt_id = cov1.compute_expected_receipt_id()
            ctrl.record_coverage_receipt(cov1)
            b01 = ctrl.mint_immutable_conflict_branch(
                failed_binding_id=b00.binding_id,
                new_canonical_code=c1_code,
                new_entrypoint="solve_c1",
                forest_node_id=nid,
                qualification_protocol="hegel_public_scientific_gate_v4",
                output_schema_keys=uni_keys,
                validated_parent_ids=["cal_node_p0", "cal_node_p1"],
                new_applicability_guard=g_b01,
                updated_failed_branch_applicability_guard=g_b00,
                violated_component_id="c1",
                preserved_component_ids=[],
                selection_priority=11,
            )
            ctrl.record_tested_witness(binding=b01, public_input_sha256=sha_i, passed=True, failure_kind="none", features=feats_i)
        elif ev_name == "R0":
            sel_r0 = ctrl.select_branch_deterministically(
                req_schema_keys=uni_keys,
                feature_evaluator=lambda b: m97.compute_m97_primitive_observables("alpha", pub_i, pay_i),
            )
            assert sel_r0["selected"].binding_id == b00.binding_id
            ctrl.record_tested_witness(binding=b00, public_input_sha256=sha_i, passed=True, failure_kind="none", features=feats_i)
        elif ev_name == "C2":
            ctrl.record_tested_witness(binding=reg.bindings[f"bind_{wid}_b01"], public_input_sha256=sha_i, passed=False, failure_kind="scientific_or_numerical", features=feats_i)
            b02 = ctrl.mint_immutable_conflict_branch(
                failed_binding_id=f"bind_{wid}_b01",
                new_canonical_code=c2_code,
                new_entrypoint="solve_c2",
                forest_node_id=nid,
                qualification_protocol="hegel_public_scientific_gate_v4",
                output_schema_keys=uni_keys,
                validated_parent_ids=["cal_node_p0", "cal_node_p1"],
                new_applicability_guard=g_b02,
                updated_failed_branch_applicability_guard=g_b01,
                violated_component_id="c2",
                preserved_component_ids=[],
                selection_priority=13,
            )
            ctrl.record_tested_witness(binding=b02, public_input_sha256=sha_i, passed=True, failure_kind="none", features=feats_i)

    # Save checkpoint after `C2` (`Step 3`) and simulate full cold stop!
    ckpt_after_c2 = tmp_path / "graph_ckpt_after_step_03.json"
    graph.save_checkpoint(ckpt_after_c2)

    # Branch A: Warm continuation across `R1, U_iso, G_perm, M_couple` (`Steps 4..7`)
    warm_routes = []
    for step_idx in range(3, 7):
        t_step = stream_tasks[step_idx]
        pub_i, sha_i = _materialize_public_item(t_step)
        m97.update_m97_permuted_calibration_graph(
            spec=spec, episode_id=f"{wid}_{t_step.short_name}", public_item=pub_i, graph=graph, planner=planner, registry=reg
        )
        pay_i = m97.assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=["cal_node_p0", "cal_node_p1"])
        sel = ctrl.select_branch_deterministically(
            req_schema_keys=uni_keys,
            feature_evaluator=lambda b: m97.compute_m97_primitive_observables("alpha", pub_i, pay_i),
        )
        chosen = sel["selected"]
        ctrl.record_tested_witness(
            binding=chosen,
            public_input_sha256=sha_i,
            passed=True,
            failure_kind="none",
            features=sel["selected_features"],
        )
        warm_routes.append((manifest[step_idx]["causal_event"], chosen.binding_id, dict(chosen.parent_parameter_versions)))

    # Branch B: Cold-hydrated continuation from `ckpt_after_c2` ONLY!
    cold_graph = PersistentMechanismGraph.load_checkpoint(ckpt_after_c2)
    cold_planner = m97.M8VariationalPlanner(cold_graph)
    cold_reg = m94.M94BindingRegistry(cold_graph)
    cold_ctrl = m96.ImmutableBranchLineageController(world_id=wid, family="alpha", graph=cold_graph, registry=cold_reg)

    assert len(cold_reg.bindings) == 3
    assert len(cold_ctrl.witness_ledger) == 6
    assert len(cold_ctrl.coverage_receipts) == 1

    cold_routes = []
    for step_idx in range(3, 7):
        t_step = stream_tasks[step_idx]
        pub_i, sha_i = _materialize_public_item(t_step)
        m97.update_m97_permuted_calibration_graph(
            spec=spec, episode_id=f"{wid}_{t_step.short_name}", public_item=pub_i, graph=cold_graph, planner=cold_planner, registry=cold_reg
        )
        pay_i = m97.assemble_m97_parameter_payload(spec=spec, graph=cold_graph, parent_ids=["cal_node_p0", "cal_node_p1"])
        sel = cold_ctrl.select_branch_deterministically(
            req_schema_keys=uni_keys,
            feature_evaluator=lambda b: m97.compute_m97_primitive_observables("alpha", pub_i, pay_i),
        )
        chosen = sel["selected"]
        cold_ctrl.record_tested_witness(
            binding=chosen,
            public_input_sha256=sha_i,
            passed=True,
            failure_kind="none",
            features=sel["selected_features"],
        )
        cold_routes.append((manifest[step_idx]["causal_event"], chosen.binding_id, dict(chosen.parent_parameter_versions)))

    assert cold_routes == warm_routes
    assert [ev_b[1] for ev_b in cold_routes] == [
        f"bind_{wid}_b01",  # R1 -> C1
        f"bind_{wid}_b01",  # U_iso -> C1 (counterfactual history!)
        f"bind_{wid}_b02",  # G_perm -> C2
        f"bind_{wid}_b01",  # M_couple -> C1 (counterfactual history + v3 calibration!)
    ]
    assert [w.to_dict() for w in cold_ctrl.witness_ledger] == [w.to_dict() for w in ctrl.witness_ledger]

    # Tampering with persisted witness ledger fails closed on hydration!
    bad_ckpt = tmp_path / "tampered_ckpt.json"
    d_bad = json.loads(ckpt_after_c2.read_text(encoding="utf-8"))
    d_bad["questions"][m96.M96_LINEAGE_FOREST_KEY]["streams"][f"{wid}::alpha"]["witness_ledger"][0]["result"] = "failed"
    bad_ckpt.write_text(json.dumps(d_bad), encoding="utf-8")
    g_bad = PersistentMechanismGraph.load_checkpoint(bad_ckpt)
    r_bad = m94.M94BindingRegistry(g_bad)
    with pytest.raises(ValueError, match="Tampered"):
        m96.ImmutableBranchLineageController(world_id=wid, family="alpha", graph=g_bad, registry=r_bad)


def test_a11_shared_stratified_calibration_estimator_zero_difference_across_dev_worlds():
    for spec in m97.DEV_M97_WORLD_SPECS:
        forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
        graph, planner, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)
        flat = (
            {"stress_gains": [1.0] * 4, "stress_biases": [0.0] * 4}
            if spec.family == "alpha"
            else {"bank_gains": [1.0] * 4, "bank_offsets": [0.0] * 4}
        )
        kg = "stress_gains" if spec.family == "alpha" else "bank_gains"
        kb = "stress_biases" if spec.family == "alpha" else "bank_offsets"
        _, tasks, manifest = m97.build_m97_world_tasks(spec)
        for task, meta in zip(tasks, manifest):
            pub_item, _ = _materialize_public_item(task)
            flat = m97.update_m97_flat_validated_channel_snapshot(family=spec.family, public_item=pub_item, prior_snapshot=flat)
            upd = m97.update_m97_permuted_calibration_graph(
                spec=spec, episode_id=meta["causal_event"], public_item=pub_item, graph=graph, planner=planner, registry=reg
            )
            gp = m97.assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=["cal_node_p0", "cal_node_p1"])
            assert np.max(np.abs(np.asarray(gp[kg]) - np.asarray(flat[kg]))) == 0.0
            assert np.max(np.abs(np.asarray(gp[kb]) - np.asarray(flat[kb]))) == 0.0
            assert upd["sentinel_p3_drift"] == 0.0


def test_a06_verifier_error_codes_separate_non_finite_from_parameter_order():
    spec = m97.DEV_M97_WORLD_SPECS[0]
    s_task, _, _ = m97.build_m97_world_tasks(spec)
    assert '"PARAMETER_ORDER"' in s_task.private_verifier_py
    assert '"NON_FINITE"' in s_task.private_verifier_py
    assert '"STRAIN_ERROR"' in s_task.private_verifier_py
    assert '"RATE_ERROR"' in s_task.private_verifier_py


def test_a16_historical_capabilities_indexes_m6_through_m97_without_breaking_181_versions():
    full = historical_capabilities()
    assert len(full["versions"]) == 181
    m97_hit = historical_capabilities(query="M9.7 ImmutableBranchLineageController EmpiricalCoverageReceipt", compact=True)
    assert m97_hit["count"] >= 1
    route_ids = {r["id"] for r in m97_hit["reuse_routes"]}
    assert "m6_m97_continual_mechanism_lineage_and_verification" in route_ids
    zh_hit = historical_capabilities(query="不可变多分支谱系 经验覆盖回执", compact=True)
    assert "m6_m97_continual_mechanism_lineage_and_verification" in {r["id"] for r in zh_hit["reuse_routes"]}
