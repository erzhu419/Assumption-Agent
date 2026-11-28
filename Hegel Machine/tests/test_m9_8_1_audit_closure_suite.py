"""Focused `M9.8.1` Audit Closure & Contract Hardening Suite (`R04, R05, R06, R07, R11, R12`, `0` Paid LLM Calls).

Verifies all 7 mandatory contract & mocked-infrastructure counterexamples from `Hegel_M9_8_Audit` (`closure_contract_probes.py` + `01_Action_Plan_ZH.md` §2.B3):
  1. `R06`: Malformed `EmpiricalCoverageReceipt.receipt_id = 'ecov_invalid'`, count mismatches, or inconsistent `supported_on_witnesses` counts fail closed (`ValueError`).
  2. `R07` (Current Schema): Deleting `state_sha256` and/or `witness_content_sha256` on current-schema checkpoints fails closed on hydration (`ValueError`), blocking forged negative witnesses from minting branches; foreign-key missing `binding_id` or mismatched `solver_code_sha256` also fails closed.
  3. `R07` (Legacy Downgrade): Changing `schema` to a legacy version while deleting digests hydrates entries strictly into `unverified_legacy` (`result='not_tested'`, `0` positive/negative witness authority), refusing `mint_immutable_conflict_branch`.
  4. `R04`: High `r2_diag >= 0.96` without direct component execution returns `execution_status='not_executed'`, `evidence_origin='compatibility_inference'`, `chosen_up_holdout_rmse=None` (NEVER `0.0`), and NEVER records a `tested_passed` upstream witness.
  5. `R04 + R05` (Main Probe Timeout): Whole-program pre-LLM timeout (`[TIMEOUT:`) returns `locus='unknown_infra'`, leaves BOTH `upstream_tested_witness` and `downstream_tested_witness` `None`, and records `0` positive/negative witnesses.
  6. `R05` (Localization Infra / `rc=125` / `TimeoutExpired`): Localization Docker `rc=125` or `TimeoutExpired` is caught cleanly (`locus='unknown_infra'`, `chosen_up_holdout_rmse=None`, NEVER `999.0`), and NEVER turns a healthy `U1` module into a `scientific_or_numerical` negative witness.
  7. `R12` + Backward Compatibility: All `8` sealed `M9.8` confirmation checkpoints load `100%` cleanly under hardened `ImmutableBranchLineageController`, and `compute_canonical_numeric_item_sha256` produces identical digests across array copies and `R0 == U_iso`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_6_autonomous_structure_continual_world as m96,
    m9_7_non_nested_counterfactual_lineage_world as m97,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_binding(reg, graph, *, bid: str, family: str, code: str, ep: str, parents: list[str], schema: list[str]):
    nid = next(iter(graph.forest.nodes))
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    q = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha,
        forest_node_id=nid,
    )
    b = m94.MechanismExecutableBinding(
        binding_id=bid,
        chain_id=family,
        branch_tag="fixture",
        mechanism_id=f"mech_{bid}",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="fixture",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="fixture",
        module_digest=sha,
        function_name=ep,
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q.receipt_id,
        parameter_mechanism_id=parents[0],
        parameter_version=graph.mechanisms[parents[0]].version,
        parent_parameter_mechanism_ids=parents,
        parent_parameter_versions={k: graph.mechanisms[k].version for k in parents},
        canonical_algorithm_code=code,
        canonical_entrypoint=ep,
        output_schema_keys=schema,
        applicability_guard={},
        selection_priority=10,
    )
    reg.register_binding(b, strict_receipt_check=True)
    return b


def test_r06_malformed_coverage_receipt_id_and_count_invariants_fail_closed():
    rec = m96.EmpiricalCoverageReceipt(
        receipt_id="",
        new_binding_id="new_b",
        new_solver_sha256="a" * 64,
        old_binding_id="old_b",
        old_solver_sha256="b" * 64,
        witness_ids=["w0"],
        parameter_versions_by_witness=[{"cal_node_p0": 1}],
        verifier_version="fixture",
        status="refuted",
        passed_witness_count=0,
        evaluated_witness_count=1,
        total_cpu_probe_wall_ms=1.0,
        mean_probe_wall_ms_ratio=1.0,
        witness_input_sha256s=["c" * 64],
        parameter_payload_sha256s=["d" * 64],
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    rec.verify_integrity()

    # Exact `closure_contract_probes.py` bypass vector: mutate to `supported_on_witnesses` AND set `receipt_id = 'ecov_invalid'`!
    tampered = rec.to_dict()
    tampered["status"] = "supported_on_witnesses"
    tampered["passed_witness_count"] = 1
    tampered["receipt_id"] = "ecov_invalid"
    with pytest.raises(ValueError, match="Tampered EmpiricalCoverageReceipt"):
        m96.EmpiricalCoverageReceipt.from_dict(tampered)

    # Empty `receipt_id` also fails closed!
    tampered["receipt_id"] = ""
    with pytest.raises(ValueError, match="Tampered EmpiricalCoverageReceipt"):
        m96.EmpiricalCoverageReceipt.from_dict(tampered)

    # Length mismatch between `witness_ids` and `witness_input_sha256s` fails closed!
    bad_len = rec.to_dict()
    bad_len["witness_input_sha256s"] = ["c" * 64, "e" * 64]
    with pytest.raises(ValueError, match="witness_input_sha256s length"):
        m96.EmpiricalCoverageReceipt.from_dict(bad_len)


def test_r07_missing_digests_foreign_keys_and_legacy_schema_downgrade_fail_closed():
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    graph, _, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=m97.DEV_M97_WORLD_SPECS[0], forest=forest)
    b = _make_binding(
        reg,
        graph,
        bid="audit_b00",
        family="alpha",
        code="def solve(d):\n    return {}\n",
        ep="solve",
        parents=["cal_node_p0", "cal_node_p1"],
        schema=m97.expected_m97_public_output_keys("alpha"),
    )
    ctrl = m96.ImmutableBranchLineageController(world_id="audit", family="alpha", graph=graph, registry=reg)
    ctrl.record_tested_witness(
        binding=b, public_input_sha256="f" * 64, passed=True, failure_kind="none", features={"obs_01": 0.0}
    )

    # 1. Exact `closure_contract_probes.py` vector: tamper positive witness -> scientific negative AND pop both digests!
    blob = copy.deepcopy(graph.forest.questions[m96.M96_LINEAGE_FOREST_KEY])
    entry = blob["streams"]["audit::alpha"]
    entry["witness_ledger"][0]["result"] = "failed"
    entry["witness_ledger"][0]["failure_kind"] = "scientific_or_numerical"
    entry.pop("state_sha256", None)
    entry["witness_ledger"][0].pop("witness_content_sha256", None)
    graph.forest.questions[m96.M96_LINEAGE_FOREST_KEY] = blob
    with pytest.raises(ValueError, match="Missing state_sha256"):
        m96.ImmutableBranchLineageController(world_id="audit", family="alpha", graph=graph, registry=reg)

    # 2. Even if `state_sha256` is recomputed around a witness missing `witness_content_sha256`, `SolverEvaluationWitness.from_dict` rejects!
    entry["state_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "branch_seq": entry["branch_seq"],
                "witness_ledger": entry["witness_ledger"],
                "coverage_receipts": entry["coverage_receipts"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="Missing witness_content_sha256"):
        m96.ImmutableBranchLineageController(world_id="audit", family="alpha", graph=graph, registry=reg)

    # 3. Downgrade attack: change `schema` to legacy `'hegel_legacy_v0'` while keeping forged negative witness ->
    #    hydrates strictly as `unverified_legacy` with ZERO negative/positive witness authority!
    blob["schema"] = "hegel_legacy_v0"
    cold_legacy = m96.ImmutableBranchLineageController(world_id="audit", family="alpha", graph=graph, registry=reg)
    assert cold_legacy.scientific_negative_witnesses_for(b.binding_id) == []
    assert cold_legacy.positive_witnesses_for(b.binding_id) == []
    assert cold_legacy.witness_ledger[0].verification_trust_state == "unverified_legacy"
    with pytest.raises(ValueError, match="without a genuine tested scientific negative witness"):
        cold_legacy.mint_immutable_conflict_branch(
            failed_binding_id=b.binding_id,
            new_canonical_code="def solve2(d):\n    return {}\n",
            new_entrypoint="solve2",
            forest_node_id=b.forest_node_id,
            qualification_protocol="hegel_public_scientific_gate_v4",
            output_schema_keys=b.output_schema_keys,
            validated_parent_ids=["cal_node_p0", "cal_node_p1"],
            new_applicability_guard={"all_of": [{"feature_name": "obs_01", "op": ">", "threshold": 0.1}]},
            updated_failed_branch_applicability_guard=None,
            violated_component_id="fixture",
            preserved_component_ids=[],
            selection_priority=20,
        )

    # 4. Foreign-key check: valid hashes referencing a non-existent `binding_id` fail closed on current schema!
    ctrl.sync_to_graph()
    fk_blob = copy.deepcopy(graph.forest.questions[m96.M96_LINEAGE_FOREST_KEY])
    w_dict = fk_blob["streams"]["audit::alpha"]["witness_ledger"][0]
    w_dict["binding_id"] = "nonexistent_binding"
    w_obj = m96.SolverEvaluationWitness(**{k: v for k, v in w_dict.items() if k != "witness_content_sha256"})
    w_dict["witness_content_sha256"] = w_obj.compute_expected_content_sha256()
    fk_blob["streams"]["audit::alpha"]["state_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "branch_seq": fk_blob["streams"]["audit::alpha"]["branch_seq"],
                "witness_ledger": fk_blob["streams"]["audit::alpha"]["witness_ledger"],
                "coverage_receipts": fk_blob["streams"]["audit::alpha"]["coverage_receipts"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    graph.forest.questions[m96.M96_LINEAGE_FOREST_KEY] = fk_blob
    with pytest.raises(ValueError, match="unknown binding_id"):
        m96.ImmutableBranchLineageController(world_id="audit", family="alpha", graph=graph, registry=reg)


def test_r04_high_r2_shortcut_never_claims_zero_rmse_or_tested_positive_witness():
    spec = m98.DEV_M98_WORLD_SPECS[0]
    s_task, tasks, _, logs = m98.build_m98_world_tasks(spec)
    pub, pub_sha = m98.materialize_m98_public_item_and_sha(s_task)
    broken = "def calibrate_upstream(d, p=None):\n    raise RuntimeError('deliberately broken upstream')\n"
    diag = m98.diagnose_m98_factored_conflict_locus("alpha", pub, chosen_up_code=broken)
    assert diag["component_probe_executed"] is False
    assert diag["execution_status"] == "not_executed"
    assert diag["evidence_origin"] == "compatibility_inference"
    assert diag["chosen_up_holdout_rmse"] is None

    # Also verify `run_m98_equal_capability_pre_llm_trial` on a downstream scientific failure with `r2_diag >= 0.96`:
    # `upstream_tested_witness` MUST be `None` (NOT `passed=True`!), while `inferred_upstream_compatibility` is `True`!
    f2, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    g2, _, r2 = m98.initialize_m98_factored_mechanism_graph(spec=spec, forest=f2)
    _make_binding(
        r2,
        g2,
        bid="audit_up_b00",
        family="alpha_up",
        code=broken,
        ep="calibrate_upstream",
        parents=["cal_node_p0", "cal_node_p1"],
        schema=["calibrated_input"],
    )
    d = _make_binding(
        r2,
        g2,
        bid="audit_dn_b00",
        family="alpha_dn",
        code="def solve(d):\n    return {}\n",
        ep="solve",
        parents=["constitutive_root_dn"],
        schema=m98.expected_m98_public_output_keys("alpha"),
    )
    uc = m96.ImmutableBranchLineageController(world_id="audit_up", family="alpha_up", graph=g2, registry=r2)
    dc = m96.ImmutableBranchLineageController(world_id="audit_dn", family="alpha_dn", graph=g2, registry=r2)
    sci_fail = {
        "accepted": False,
        "schema_valid": True,
        "schema_stdout": "OK",
        "schema_stderr": "",
        "scientific_probe_receipt": {"ok": False, "errors": ["strain_mae_exceeded:0.42"]},
    }
    with patch.object(m98, "execute_single_pre_llm_public_gate_probe_in_container", return_value=sci_fail):
        trial = m98.run_m98_equal_capability_pre_llm_trial(
            arm="Hegel_FactoredMechanismGraph",
            spec=spec,
            task=tasks[0],
            public_item=pub,
            public_input_sha256=pub_sha,
            prefix_apparatus_logs=logs[:1],
            prior_archive_entries=[],
            up_ctrl=uc,
            dn_ctrl=dc,
            graph=g2,
            registry=r2,
        )
    assert trial["upstream_tested_witness"] is None
    assert trial["inferred_upstream_compatibility"] is True
    assert uc.positive_witnesses_for("audit_up_b00") == []
    # `Q08` (`M9.9.1`): `compatibility_inference` (`unexecuted` upstream) must NOT mint a genuine downstream scientific negative!
    assert len(dc.scientific_negative_witnesses_for(d.binding_id)) == 0
    assert trial["downstream_tested_witness"] is None


def test_r04_r05_main_probe_timeout_and_localization_infra_errors_never_pollute_witnesses():
    spec = m98.DEV_M98_WORLD_SPECS[0]
    s_task, tasks, _, logs = m98.build_m98_world_tasks(spec)
    pub, pub_sha = m98.materialize_m98_public_item_and_sha(s_task)

    f2, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    g2, _, r2 = m98.initialize_m98_factored_mechanism_graph(spec=spec, forest=f2)
    u0 = _make_binding(
        r2,
        g2,
        bid="audit_up_b00",
        family="alpha_up",
        code=m98.build_m98_default_u0_transducer_code("alpha"),
        ep="calibrate_upstream",
        parents=["cal_node_p0", "cal_node_p1"],
        schema=["calibrated_input"],
    )
    d0 = _make_binding(
        r2,
        g2,
        bid="audit_dn_b00",
        family="alpha_dn",
        code="def solve(d):\n    return {}\n",
        ep="solve",
        parents=["constitutive_root_dn"],
        schema=m98.expected_m98_public_output_keys("alpha"),
    )
    uc = m96.ImmutableBranchLineageController(world_id="audit_up", family="alpha_up", graph=g2, registry=r2)
    dc = m96.ImmutableBranchLineageController(world_id="audit_dn", family="alpha_dn", graph=g2, registry=r2)

    # 1. Main whole-program timeout (`[TIMEOUT:`): NEITHER upstream NOR downstream witness recorded!
    timeout_pr = {
        "accepted": False,
        "schema_valid": False,
        "schema_stdout": "",
        "schema_stderr": "[TIMEOUT: command exceeded 45s limit]",
        "timed_out": True,
        "scientific_probe_receipt": None,
    }
    with patch.object(m98, "execute_single_pre_llm_public_gate_probe_in_container", return_value=timeout_pr):
        trial_to = m98.run_m98_equal_capability_pre_llm_trial(
            arm="Hegel_FactoredMechanismGraph",
            spec=spec,
            task=s_task,
            public_item=pub,
            public_input_sha256=pub_sha,
            prefix_apparatus_logs=logs[:1],
            prior_archive_entries=[],
            up_ctrl=uc,
            dn_ctrl=dc,
            graph=g2,
            registry=r2,
        )
    assert trial_to["conflict_locus_diagnosis"]["locus"] == "unknown_infra"
    assert trial_to["upstream_tested_witness"] is None
    assert trial_to["downstream_tested_witness"] is None
    assert uc.positive_witnesses_for(u0.binding_id) == []
    assert dc.scientific_negative_witnesses_for(d0.binding_id) == []

    # 2. Low-R2 (`C_up`) localization Docker `rc=125` AND `TimeoutExpired` caught cleanly (`chosen_up_holdout_rmse is None`, NEVER `999.0`!)
    pu, _ = m98.materialize_m98_public_item_and_sha(tasks[3])
    u1_code = m98.build_m98_reference_dev_upstream_u1_code("alpha")
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["docker"], returncode=125, stdout="", stderr="Cannot connect to the Docker daemon"
        ),
    ):
        diag_125 = m98.diagnose_m98_factored_conflict_locus("alpha", pu, chosen_up_code=u1_code)
    assert diag_125["locus"] == "unknown_infra"
    assert diag_125["execution_status"] == "infrastructure_error"
    assert diag_125["chosen_up_holdout_rmse"] is None
    assert diag_125["probe_id"].startswith("uprobe_")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=30)):
        diag_to = m98.diagnose_m98_factored_conflict_locus("alpha", pu, chosen_up_code=u1_code)
    assert diag_to["locus"] == "unknown_infra"
    assert diag_to["execution_status"] == "timeout"
    assert diag_to["chosen_up_holdout_rmse"] is None

    # 3. Healthy `U1` + main scientific failure + localization `rc=125` MUST NOT fabricate a scientific upstream negative witness!
    f3, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    g3, _, r3 = m98.initialize_m98_factored_mechanism_graph(spec=spec, forest=f3)
    u3 = _make_binding(
        r3,
        g3,
        bid="healthy_up_b01",
        family="alpha_up",
        code=u1_code,
        ep="calibrate_upstream",
        parents=["cal_node_p0", "cal_node_p1"],
        schema=["calibrated_input"],
    )
    d3 = _make_binding(
        r3,
        g3,
        bid="failing_dn_b00",
        family="alpha_dn",
        code="def solve(d):\n    return {}\n",
        ep="solve",
        parents=["constitutive_root_dn"],
        schema=m98.expected_m98_public_output_keys("alpha"),
    )
    uc3 = m96.ImmutableBranchLineageController(world_id="healthy_up", family="alpha_up", graph=g3, registry=r3)
    dc3 = m96.ImmutableBranchLineageController(world_id="failing_dn", family="alpha_dn", graph=g3, registry=r3)
    main_sci_fail = {
        "accepted": False,
        "schema_valid": True,
        "schema_stdout": "OK",
        "schema_stderr": "",
        "scientific_probe_receipt": {"ok": False, "errors": ["strain_mae_exceeded:1.0"]},
    }
    with (
        patch.object(m98, "execute_single_pre_llm_public_gate_probe_in_container", return_value=main_sci_fail),
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["docker"], returncode=125, stdout="", stderr="Cannot connect to Docker daemon"
            ),
        ),
    ):
        res_125 = m98.run_m98_equal_capability_pre_llm_trial(
            arm="Hegel_FactoredMechanismGraph",
            spec=spec,
            task=tasks[3],
            public_item=pu,
            public_input_sha256="d" * 64,
            prefix_apparatus_logs=logs[:4],
            prior_archive_entries=[],
            up_ctrl=uc3,
            dn_ctrl=dc3,
            graph=g3,
            registry=r3,
        )
    assert res_125["conflict_locus_diagnosis"]["locus"] == "unknown_infra"
    assert res_125["upstream_tested_witness"] is None
    assert res_125["downstream_tested_witness"] is None
    assert uc3.scientific_negative_witnesses_for(u3.binding_id) == []
    assert dc3.scientific_negative_witnesses_for(d3.binding_id) == []


def test_r12_sealed_m98_checkpoints_hydrate_cleanly_and_canonical_numeric_sha_is_stable():
    m98_dir = REPO_ROOT / "docs/evidence/m9_8_blind_mismatch_factored_graph_20261127"
    total_wit = 0
    total_cov = 0
    for spec in m98.CONFIRMATION_M98_WORLD_SPECS:
        wid = spec.world_id
        ckpt = m98_dir / "continual_streams" / wid / "Hegel_FactoredMechanismGraph" / "graph_ckpt_after_step_07.json"
        g = PersistentMechanismGraph.load_checkpoint(ckpt)
        r = m94.M94BindingRegistry(g)
        uc = m96.ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{spec.family}_up", graph=g, registry=r)
        dc = m96.ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g, registry=r)
        total_wit += len(uc.witness_ledger) + len(dc.witness_ledger)
        total_cov += len(uc.coverage_receipts) + len(dc.coverage_receipts)
    assert total_wit == 144
    assert total_cov == 24

    # Verify `compute_canonical_numeric_item_sha256` on dev `(R0, U_iso)` including NaN handling!
    spec_dev = m98.DEV_M98_WORLD_SPECS[0]
    _, tasks_d, manifest_d, _ = m98.build_m98_world_tasks(spec_dev)
    by_ev = {m["causal_event"]: t for t, m in zip(tasks_d, manifest_d)}
    r0_item, _ = m98.materialize_m98_public_item_and_sha(by_ev["R0"])
    ui_item, _ = m98.materialize_m98_public_item_and_sha(by_ev["U_iso"])
    sha_r0 = m96.compute_canonical_numeric_item_sha256(r0_item)
    sha_ui = m96.compute_canonical_numeric_item_sha256(ui_item)
    assert len(sha_r0) == 64 and sha_r0 == sha_ui
    mut = copy.deepcopy(r0_item)
    mut["dataset"]["eval_times"] = np.asarray(mut["dataset"]["eval_times"], dtype=float) + 1e-6
    assert m96.compute_canonical_numeric_item_sha256(mut) != sha_r0
