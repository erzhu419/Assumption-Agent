"""Regression tests for evidence identity and calibration epochs in M9.9 cold restores."""

from dataclasses import replace

import pytest

from test_m9_9_1_audit_closure_suite import M99_CONF_DIR, _make_fixture, m8p, m94, m96, m98, m99


def test_completed_evidence_is_required_to_classify_a_scientific_failure():
    failed = {"accepted": False, "schema_valid": True}
    for receipt in (None, {}, {"execution_status": "timeout", "scientific_invariants_verified": False},
                    {"execution_status": "infrastructure_error", "scientific_invariants_verified": False}):
        assert m96.classify_public_probe_failure({**failed, "scientific_probe_receipt": receipt})["tier"] == "INFRA_FAILURE"
    completed = {"execution_status": "completed", "scientific_invariants_verified": False}
    assert m96.classify_public_probe_failure({**failed, "scientific_probe_receipt": completed})["tier"] == "SCIENTIFIC_FAILURE"


def test_receipt_writer_rejects_unattributed_or_unexecuted_results():
    _, _, _, up, dn, _, controller = _make_fixture()
    receipt = {
        "probe_id": "probe", "execution_status": "completed", "schema_valid": True, "accepted": True,
        "component_code_sha256s": [dn.authoritative_executed_code_sha256()],
        "public_input_sha256": "a" * 64, "canonical_numeric_sha256": "b" * 64,
    }
    kwargs = dict(binding=dn, public_input_sha256="a" * 64, canonical_numeric_sha256="b" * 64,
                  features={}, evidence_origin="joint_task_probe")
    for edit in ({"execution_status": None}, {"execution_status": "unknown"}, {"accepted": None},
                 {"component_code_sha256s": [up.authoritative_executed_code_sha256()]},
                 {"public_input_sha256": "c" * 64}, {"canonical_numeric_sha256": "d" * 64}):
        with pytest.raises(ValueError, match="Q01"):
            controller.record_tested_witness_from_probe_receipt(probe_receipt={**receipt, **edit}, **kwargs)
    for edit in ({"execution_status": "not_executed"}, {"accepted": False, "scientific_probe_receipt": None}):
        assert controller.record_tested_witness_from_probe_receipt(probe_receipt={**receipt, **edit}, **kwargs) is None
    with pytest.raises(ValueError, match="Q01"):
        controller.record_tested_witness_from_probe_receipt(
            probe_receipt=receipt, **{**kwargs, "evidence_origin": "direct_component_probe"}
        )
    assert controller.witness_ledger == []


def test_withdrawal_binds_historical_candidate_input_and_epochs(tmp_path):
    wid = "alpha_m99_c01"
    step5 = M99_CONF_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online" / "graph_ckpt_after_step_05.json"
    graph = m8p.PersistentMechanismGraph.load_checkpoint(step5)
    registry = m94.M94BindingRegistry(graph)
    original, = m99.inspect_archived_withdrawal_lineage(graph, registry, wid)
    for edit in ({"composed_candidate_sha256": "a" * 64}, {"counterexample_canonical_numeric_sha256": "b" * 64},
                 {"verifier_version": "other_kernel"}, {"expanded_feature_names": ["unknown_feature"]},
                 {"joint_probe_id": "wrong_execution"}, {"prior_witness_id": "unrelated_positive"}):
        invalid = replace(original, **edit)
        invalid.receipt_id = invalid.compute_expected_receipt_id()
        m99.persist_separability_withdrawals_to_graph(graph, wid, [invalid])
        with pytest.raises(ValueError, match="Q07"):
            m99.hydrate_separability_withdrawals_from_graph(graph, registry, wid)
        assert not m99.hydrate_separability_withdrawals_from_graph(graph, registry, wid, quarantine_unverified=True)
    m99.persist_separability_withdrawals_to_graph(graph, wid, [original])
    spec = next(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id == wid)
    _, tasks, _, _ = m99.build_m99_world_tasks(spec)
    public_item, _ = m98.materialize_m98_public_item_and_sha(tasks[5])
    old_version = graph.mechanisms["cal_node_p0"].version
    m99.update_m99_calibration_snapshot_and_graph(
        spec=spec.to_m98_compat_spec(), episode_id="cold_restore", public_item=public_item, graph=graph, registry=registry,
    )
    assert graph.mechanisms["cal_node_p0"].version > old_version
    saved = tmp_path / "checkpoint.json"
    graph.save_checkpoint(saved)
    restored = m8p.PersistentMechanismGraph.load_checkpoint(saved)
    result = m99.inspect_archived_withdrawal_lineage(restored, m94.M94BindingRegistry(restored), wid)
    assert not m99.hydrate_separability_withdrawals_from_graph(restored, m94.M94BindingRegistry(restored), wid, quarantine_unverified=True)
    assert [r.receipt_id for r in result] == [original.receipt_id]


def test_legacy_binding_removed_before_migration_preserves_unverified_witness():
    _, graph, registry, up, _, controller, _ = _make_fixture()
    original = controller.record_tested_witness(
        binding=up, public_input_sha256="a" * 64, passed=True, failure_kind="none", features={},
    )
    graph.forest.questions[m96.M96_LINEAGE_FOREST_KEY]["schema"] = "hegel_m9_6_lineage_state_v1"
    del registry.bindings[up.binding_id]
    args = dict(world_id=controller.world_id, family=controller.family, graph=graph, registry=registry)
    migrated = m96.ImmutableBranchLineageController(**args)
    migrated.sync_to_graph()
    restored = m96.ImmutableBranchLineageController(**args)
    saved, = restored.witness_ledger
    assert (saved.witness_id, saved.solver_code_sha256) == (original.witness_id, original.solver_code_sha256)
    assert saved.verification_trust_state == "unverified_legacy" and saved.result == "not_tested"
    assert restored.positive_witnesses_for(up.binding_id) == []


def test_frozen_ablation_labels_historical_withdrawals_and_missing_prefix(monkeypatch):
    import importlib.util

    script = M99_CONF_DIR.parents[2] / "scripts/run_m9_9_autonomous_decomposition_experiment.py"
    module_spec = importlib.util.spec_from_file_location("m99_frozen_ablation", script)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "grade_m94_artifact_map", lambda *args, **kwargs: {"strict_pass": False})
    worlds = tuple(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id in {"alpha_m99_c01", "alpha_m99_c03"})
    rows = runner.run_m99_factorized_ceiling_ablation_on_sealed_streams(
        world_specs=worlds, built_worlds={s.world_id: m99.build_m99_world_tasks(s) for s in worlds},
        out_dir=M99_CONF_DIR,
    )["rows"]
    assert len(rows) == 12
    assert {row["policy_input_source"] for row in rows} == {"historical_host_reconstruction"}
    assert {row["withdrawal_evidence_basis"] for row in rows if row["world_id"] == "alpha_m99_c01"} == {"historical_lineage_only"}
    assert {row["withdrawal_evidence_basis"] for row in rows if row["world_id"] == "alpha_m99_c03"} == {"absent"}


def test_m98_online_split_requires_the_joint_gate_on_the_measured_input(monkeypatch):
    spec, graph, registry, _, dn, up_controller, dn_controller = _make_fixture()
    task, _, _, logs = m98.build_m98_world_tasks(spec)
    item, raw_sha = m98.materialize_m98_public_item_and_sha(task)
    monkeypatch.setattr(m98, "materialize_sandbox_public_item_and_sha", lambda _: (item, raw_sha))
    # A stored measurement with missing provenance cannot authorize an online split.
    monkeypatch.setattr(m98, "execute_single_pre_llm_public_gate_probe_in_container", lambda **_: {
        "accepted": False, "schema_valid": True,
        "scientific_probe_receipt": {"execution_status": "completed", "errors": ["residual"], "scientific_invariants_verified": False},
    })
    def unexpected_localization(*args, **kwargs):
        pytest.fail("Unbound joint evidence must not initiate component localization")
    monkeypatch.setattr(m98, "diagnose_m98_factored_conflict_locus", unexpected_localization)
    result = m98.run_m98_equal_capability_pre_llm_trial(
        arm="Hegel_FactoredMechanismGraph", spec=spec, task=task, public_item=item, public_input_sha256=raw_sha,
        prefix_apparatus_logs=logs[:1], prior_archive_entries=[], up_ctrl=up_controller, dn_ctrl=dn_controller,
        graph=graph, registry=registry, require_split_evidence=True,
    )
    assert result["conflict_locus_diagnosis"]["locus"] == "unknown_infra"
    assert result["localization_probes_used"] == 0 and result["downstream_tested_witness"] is None
    assert dn_controller.scientific_negative_witnesses_for(dn.binding_id) == []
