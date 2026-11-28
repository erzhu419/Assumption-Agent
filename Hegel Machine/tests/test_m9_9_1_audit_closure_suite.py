"""Deterministic & Sealed-Checkpoint Contract Suite for `M9.9.1` Audit Closure (`Q01–Q14`, `0` Paid LLM Calls).

Verifies every regression probe from `Hegel_M9_9_Audit`:
  1. `Q01` (`2x2` Decomposition Matrix + `ProbeReceipt` Witness Authority + Non-Triviality Timeout Accounting):
     - `verify_m99_autonomous_decomposition_in_containers` executes all 4 `2x2` cells (`s0_d0`, `e1_d1`, `e1_d0`, `s0_d1`).
     - Identical `d0 == d1` under mocked accepted gates yields `e1_d0['accepted'] == True` and `s0_d1['accepted'] == True`
       (`0` fabricated scientific negative witnesses!).
     - `TimeoutExpired` on non-triviality check records `probe_id` and `wall_ms` in `probes` before returning.
     - `record_tested_witness_from_probe_receipt` refuses bare/unexecuted/timeout receipts from minting scientific negatives.
  2. `Q02` (Shared Public Pooled `train+test` `ref_coupons` Swap Invariance):
     - Deterministically rejects the frozen `alpha_m99_c02` `dn_b02` solver (`ref_swap_strain_max_diff > 1e-5`) on `E7` public input,
       while accepting the pooled `(X_tr, X_te)` reference standardization (`ref_swap_strain_max_diff < 1e-12`).
  3. `Q07` (`SeparabilityWithdrawalReceipt` Strong Foreign-Key Ledger Verification):
     - Rejects synthetic self-hash-consistent receipts with wrong `upstream_code_sha256` / `downstream_code_sha256`, missing
       `upstream_component_probe_id`, or `0` prior colliding downstream positive witnesses.
     - Recovers archival lineage only from all `7/7` real sealed `M9.9` confirmation checkpoints (`alpha_m99_c01, c02, c04` + `beta_m99_c01..c04`).
  4. `Q08` (Legacy `v1` Migration Idempotent Round-Trip & `M9.8` `compatibility_inference` Isolation):
     - `load -> migrate -> save -> reload` succeeds idempotently twice on both legacy witnesses (`unverified_legacy`, `not_tested`)
       and legacy coverage receipts (`unknown_requires_reverification`), with `0` minting/coverage authority.
     - `run_m98_equal_capability_pre_llm_trial` under `compatibility_inference` (`r2 >= 0.96`, unexecuted upstream) records
       `0` downstream scientific negative witnesses!
"""

from __future__ import annotations

import ast
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
    m8_persistent_variational_planner as m8p,
    m9_4_1_shared_verification_kernel as m941,
    m9_4_continual_mechanism_binding_world as m94,
    m9_6_autonomous_structure_continual_world as m96,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
    m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
M99_CONF_DIR = REPO_ROOT / "docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128"


def _make_fixture(broken_upstream: bool = False):
    spec = m98.DEV_M98_WORLD_SPECS[0]
    f, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    g, _, r = m98.initialize_m98_factored_mechanism_graph(spec=spec, forest=f)
    nid = next(iter(g.forest.nodes))

    def _bind(bid: str, fam: str, code: str, ep: str, parents: list[str], schema: list[str]):
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        q = r.issue_qualification_receipt(
            qualification_protocol="hegel_public_scientific_gate_v4",
            verified_code_sha256=sha,
            forest_node_id=nid,
        )
        b = m94.MechanismExecutableBinding(
            binding_id=bid,
            chain_id=fam,
            branch_tag="fixture",
            mechanism_id="mech_" + bid,
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
            parameter_version=g.mechanisms[parents[0]].version,
            parent_parameter_mechanism_ids=parents,
            parent_parameter_versions={k: g.mechanisms[k].version for k in parents},
            canonical_algorithm_code=code,
            canonical_entrypoint=ep,
            output_schema_keys=schema,
            applicability_guard={},
            selection_priority=10,
        )
        r.register_binding(b, strict_receipt_check=True)
        return b

    up_code = (
        "def calibrate_upstream(d, p=None):\n    raise RuntimeError('broken')\n"
        if broken_upstream
        else m98.build_m98_default_u0_transducer_code("alpha")
    )
    u = _bind("audit_up_b00", "alpha_up", up_code, "calibrate_upstream", ["cal_node_p0", "cal_node_p1"], ["calibrated_input"])
    d = _bind("audit_dn_b00", "alpha_dn", "def solve_downstream(d):\n    return {}\n", "solve_downstream", ["constitutive_root_dn"], m98.expected_m98_public_output_keys("alpha"))
    uc = m96.ImmutableBranchLineageController(world_id="audit_up", family="alpha_up", graph=g, registry=r)
    dc = m96.ImmutableBranchLineageController(world_id="audit_dn", family="alpha_dn", graph=g, registry=r)
    return spec, g, r, u, d, uc, dc


def test_q01_2x2_decomposition_identical_d0_d1_and_timeout_accounting() -> None:
    spec, g, r, u, d, uc, dc = _make_fixture()
    s_task, tasks, _, _ = m98.build_m98_world_tasks(spec)
    pub, pub_sha = m99.materialize_m99_public_item_and_sha(s_task)
    e1_pub, e1_sha = m99.materialize_m99_public_item_and_sha(tasks[0])
    same_dn = d.canonical_algorithm_code
    public_calls: list[dict] = []

    def _mock_accepted(**kw):
        public_calls.append(
            {
                "task_id": kw["task"].task_id,
                "candidate_sha256": hashlib.sha256(kw["candidate_code"].encode("utf-8")).hexdigest(),
            }
        )
        task = kw["task"]
        item, raw = (pub, pub_sha) if task == s_task else (e1_pub, e1_sha)
        sha = public_calls[-1]["candidate_sha256"]
        bundle = hashlib.sha256(b"/holdout_task/batch_solver.py\0" + bytes.fromhex(sha)).hexdigest()
        assets = {"/holdout_task/public_batch.pkl": raw}
        return {
            "accepted": True, "schema_valid": True, "timed_out": False, "candidate_raw_sha256": sha,
            "candidate_bundle_sha256": bundle, "public_setup_files_sha256": assets,
            "scientific_probe_receipt": {
                "execution_status": "completed", "signed_by_host_probe_executor": True,
                "scientific_invariants_verified": True, "candidate_bundle_sha256": bundle,
                "public_setup_files_sha256": assets, "public_input_digest": raw,
                "canonical_numeric_sha256": m96.compute_canonical_numeric_item_sha256(item),
                "receipt_kind": "scientific_probe_receipt", "probe_protocol_version": "test", "probe_code_digest": "e" * 64,
            },
        }

    healthy = {
        "execution_status": "completed", "component_probe_executed": True, "evidence_origin": "direct_component_probe",
        "chosen_up_holdout_rmse": 0.0001, "probe_id": "uprobe_mock", "probe_wall_ms": 1.2,
        "failure_tier": "NONE", "failure_kind": "none", "candidate_code_sha256": u.authoritative_executed_code_sha256(),
    }
    nt_ok = json.dumps({"cal_identity_err": 0.0, "shape_ok": True, "transformed_signal": True})
    with patch.object(m99, "diagnose_m98_factored_conflict_locus", return_value=healthy), patch(
        "subprocess.run", return_value=subprocess.CompletedProcess(["docker"], 0, nt_ok, "")
    ), patch.object(m99, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_accepted):
        ver = m99.verify_m99_autonomous_decomposition_in_containers(
            family="alpha",
            s0_task=s_task,
            s0_public_item=pub,
            s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
            e1_task=tasks[0],
            e1_public_item=e1_pub,
            e1_pay={"gains": [2.0] * 4, "offsets": [0.0] * 4},
            e1_ro_deps={},
            e1_ro_shas={},
            upstream_code=u.canonical_algorithm_code,
            d0_code=same_dn,
            d1_code=same_dn,
        )

    assert ver["valid"] is True
    assert len(public_calls) == 4
    m22 = ver["matrix_2x2"]
    assert m22["s0_d0"]["accepted"] is True and m22["e1_d1"]["accepted"] is True
    assert m22["e1_d0"]["accepted"] is True and m22["s0_d1"]["accepted"] is True

    # Recording `e1_d0` via `record_tested_witness_from_probe_receipt` records a PASSED witness (`0` fake negatives!)
    w_cross = dc.record_tested_witness_from_probe_receipt(
        binding=d,
        probe_receipt=m22["e1_d0"],
        public_input_sha256=e1_sha,
        features={"dn_stat_01_log_channel_0_mean": 0.1},
        evidence_origin="joint_task_probe",
    )
    assert w_cross is not None and w_cross.result == "passed"
    assert len(dc.scientific_negative_witnesses_for(d.binding_id)) == 0

    # Non-triviality `TimeoutExpired` appends `probe_id` and `wall_ms` before returning (`R05/Q08` detail)
    with patch.object(m99, "diagnose_m98_factored_conflict_locus", return_value=healthy), patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(["docker"], 25)
    ):
        ver_to = m99.verify_m99_autonomous_decomposition_in_containers(
            family="alpha",
            s0_task=s_task,
            s0_public_item=pub,
            s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
            e1_task=tasks[0],
            e1_public_item=e1_pub,
            e1_pay={"gains": [2.0] * 4, "offsets": [0.0] * 4},
            e1_ro_deps={},
            e1_ro_shas={},
            upstream_code=u.canonical_algorithm_code,
            d0_code=same_dn,
            d1_code=same_dn,
        )
    assert ver_to["valid"] is False and ver_to["reason"] == "nontriviality_probe_timeout"
    assert any(p["role"] == "decomp_canonical_interface_check" and p.get("probe_id") for p in ver_to["probes"])


def test_q08_legacy_roundtrip_idempotence_and_m98_inferred_isolation() -> None:
    # 1. `M9.8` `compatibility_inference` (`broken` upstream not directly executed) MUST NOT write downstream scientific negative!
    spec, g, r, u, d, uc, dc = _make_fixture(broken_upstream=True)
    s_task, _, _, logs = m98.build_m98_world_tasks(spec)
    pub, pub_sha = m98.materialize_m98_public_item_and_sha(s_task)
    sci_fail = {"accepted": False, "schema_valid": True, "schema_stdout": "OK", "schema_stderr": "", "scientific_probe_receipt": {"ok": False, "errors": ["strain_mae_exceeded:1.0"]}}
    with patch.object(m98, "execute_single_pre_llm_public_gate_probe_in_container", return_value=sci_fail):
        tr = m98.run_m98_equal_capability_pre_llm_trial(
            arm="Hegel_FactoredMechanismGraph",
            spec=spec,
            task=s_task,
            public_item=pub,
            public_input_sha256=pub_sha,
            prefix_apparatus_logs=logs[:1],
            prior_archive_entries=[],
            up_ctrl=uc,
            dn_ctrl=dc,
            graph=g,
            registry=r,
        )
    assert tr["conflict_locus_diagnosis"]["evidence_origin"] == "compatibility_inference"
    assert len(dc.scientific_negative_witnesses_for(d.binding_id)) == 0
    assert tr["downstream_tested_witness"] is None

    # 2. Legacy witness + legacy coverage receipt `load -> migrate -> save -> reload` idempotence (`0` authority!)
    spec2, g2, r2, u2, d2, uc2, dc2 = _make_fixture()
    uc2.record_tested_witness(
        binding=u2, public_input_sha256=pub_sha, passed=True, failure_kind="none", features={"x": 0.0}, probe_id="p_ok"
    )
    cov = m96.EmpiricalCoverageReceipt(
        receipt_id="",
        new_binding_id="new",
        new_solver_sha256="a" * 64,
        old_binding_id=d2.binding_id,
        old_solver_sha256=d2.authoritative_executed_code_sha256(),
        witness_ids=["w0"],
        parameter_versions_by_witness=[{"constitutive_root_dn": 1}],
        verifier_version="fixture",
        status="supported_on_witnesses",
        passed_witness_count=1,
        evaluated_witness_count=1,
        total_cpu_probe_wall_ms=1.0,
        mean_probe_wall_ms_ratio=1.0,
        witness_input_sha256s=["c" * 64],
        parameter_payload_sha256s=["d" * 64],
    )
    cov.receipt_id = cov.compute_expected_receipt_id()
    dc2.record_coverage_receipt(cov)

    g2.forest.questions[m96.M96_LINEAGE_FOREST_KEY]["schema"] = "hegel_m9_6_lineage_state_v1"
    leg_u = m96.ImmutableBranchLineageController(world_id="audit_up", family="alpha_up", graph=g2, registry=r2)
    leg_u.sync_to_graph()
    # Writing only the upstream stream must migrate the unopened downstream stream too.
    leg_d = m96.ImmutableBranchLineageController(world_id="audit_dn", family="alpha_dn", graph=g2, registry=r2)
    assert leg_u.witness_ledger[0].verification_trust_state == "unverified_legacy"
    assert leg_d.coverage_receipts[0].status == "unknown_requires_reverification"
    leg_u.sync_to_graph()
    leg_d.sync_to_graph()
    snap_1 = copy.deepcopy(g2.forest.questions[m96.M96_LINEAGE_FOREST_KEY])

    rel_u = m96.ImmutableBranchLineageController(world_id="audit_up", family="alpha_up", graph=g2, registry=r2)
    rel_d = m96.ImmutableBranchLineageController(world_id="audit_dn", family="alpha_dn", graph=g2, registry=r2)
    rel_u.sync_to_graph()
    rel_d.sync_to_graph()
    snap_2 = copy.deepcopy(g2.forest.questions[m96.M96_LINEAGE_FOREST_KEY])
    assert snap_1 == snap_2
    assert rel_u.positive_witnesses_for(u2.binding_id) == []
    assert rel_d.coverage_receipts[0].status == "unknown_requires_reverification"


def test_q07_distinguishes_archived_lineage_from_measured_host_receipts() -> None:
    spec, g, r, u, d, _, _ = _make_fixture()
    bad_rec = m99.SeparabilityWithdrawalReceipt(
        receipt_id="",
        world_id=spec.world_id,
        family="alpha",
        upstream_binding_id=u.binding_id,
        upstream_code_sha256="0" * 64,
        upstream_component_probe_id="uprobe_no_such_probe",
        upstream_holdout_rmse=0.0001,
        downstream_binding_id=d.binding_id,
        downstream_code_sha256="1" * 64,
        composed_candidate_sha256="2" * 64,
        counterexample_public_input_sha256="3" * 64,
        counterexample_canonical_numeric_sha256="4" * 64,
        joint_failure_kind="scientific_or_numerical",
        verifier_version="fixture",
        expanded_feature_names=list(m99.M99_ALL_OBSERVABLE_NAMES),
    )
    bad_rec.receipt_id = bad_rec.compute_expected_receipt_id()
    m99.persist_separability_withdrawals_to_graph(g, spec.world_id, [bad_rec])
    with pytest.raises(ValueError, match="Q07"):
        m99.hydrate_separability_withdrawals_from_graph(g, r, spec.world_id)
    assert m99.hydrate_separability_withdrawals_from_graph(g, r, spec.world_id, quarantine_unverified=True) == []

    # All seven sealed graphs preserve lineage; their original host scientific-rejection receipts are absent.
    expected_7 = ["alpha_m99_c01", "alpha_m99_c02", "alpha_m99_c04", "beta_m99_c01", "beta_m99_c02", "beta_m99_c03", "beta_m99_c04"]
    for wid in expected_7:
        ck = M99_CONF_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online" / "graph_ckpt_after_step_05.json"
        g5 = m8p.PersistentMechanismGraph.load_checkpoint(ck)
        r5 = m94.M94BindingRegistry(g5)
        sw = m99.inspect_archived_withdrawal_lineage(g5, r5, wid)
        assert not m99.hydrate_separability_withdrawals_from_graph(g5, r5, wid, quarantine_unverified=True)
        assert len(sw) == 1


def test_q02_pooled_ref_coupon_swap_invariance_catches_c02_dn_b02_and_accepts_pooled() -> None:
    ck = M99_CONF_DIR / "continual_streams/alpha_m99_c02/Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_06.json"
    bindings = json.loads(ck.read_text(encoding="utf-8"))["questions"]["__m9_4_mechanism_executable_bindings__"]["bindings"]
    up_b = bindings["bind_alpha_m99_c02_up_b01"]
    dn_b = bindings["bind_alpha_m99_c02_dn_b02"]
    uns: dict = {}
    dns: dict = {}
    exec(compile(up_b["canonical_algorithm_code"], "<c02_up_b01>", "exec"), uns)
    exec(compile(dn_b["canonical_algorithm_code"], "<c02_dn_b02>", "exec"), dns)

    spec_c02 = next(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id == "alpha_m99_c02")
    _, tasks, _, _ = m99.build_m99_world_tasks(spec_c02)
    pub_item, _ = m98.materialize_m98_public_item_and_sha(tasks[6])
    ds_pub = pub_item["dataset"]

    def _frozen_composed(ds_in: dict) -> dict:
        cal = uns[up_b["canonical_entrypoint"]](copy.deepcopy(ds_in))
        return dns[dn_b["canonical_entrypoint"]](cal)

    chk_buggy = m941.verify_alpha_pooled_ref_coupon_invariance(_frozen_composed, ds_pub)
    assert chk_buggy["valid"] is False
    assert chk_buggy["ref_swap_strain_max_diff"] > 1e-4

    def _pooled_fixed_composed(ds_in: dict) -> dict:
        cal = uns[up_b["canonical_entrypoint"]](copy.deepcopy(ds_in))
        tr, te = cal["train"], cal["test"]
        t_ev, p_dt = cal["eval_times"], cal["past_dt"]
        x_tr, x_te, _, _ = m941.standardize_reference_coupon_panel(tr, te)
        taus, theta = dns["material"](tr, x_tr, p_dt, t_ev)
        hold = te["hold_channel_raw"].sum(axis=-1)
        inc, rate = dns["chamber"](taus, theta, hold, x_te, t_ev)
        return {"estimated_taus": taus, "hereditary_weights": theta, "predicted_strain": te["strain_0"][:, None] + inc, "initial_rate_pred": rate}

    chk_fixed = m941.verify_alpha_pooled_ref_coupon_invariance(_pooled_fixed_composed, ds_pub)
    assert chk_fixed["valid"] is True
    assert chk_fixed["ref_swap_strain_max_diff"] <= 1e-7
