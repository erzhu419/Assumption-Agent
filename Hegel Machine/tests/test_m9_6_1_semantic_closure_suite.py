"""Regression & Contract Verification Suite for `M9.6.1` (`N01–N11`, `0` Paid LLM Calls)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest

from hegel_machine.cognition import (
    m6_4_executable_wormhole_library as m64,
    m7_autogenous_forest_pretraining as m7a,
    m9_4_1_shared_verification_kernel as m941k,
    m9_4_continual_mechanism_binding_world as m94,
    m9_6_autonomous_structure_continual_world as m96,
)
from hegel_machine.cognition.cost_policy import (
    aggregate_call_usages,
    assert_hierarchical_ledger_conservation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_n01_cost_policy_call_uuid_conservation_and_violation_detection() -> None:
    calls = [
        {"call_uuid": "u1", "event": "T", "world": "w1", "prompt_tokens": 1000, "cached_tokens": 400, "completion_tokens": 50},
        {"call_uuid": "u2", "event": "C", "world": "w1", "prompt_tokens": 1200, "cached_tokens": 600, "completion_tokens": 80},
        {"call_uuid": "u2", "event": "C", "world": "w1", "prompt_tokens": 1200, "cached_tokens": 600, "completion_tokens": 80},
    ]
    arm_ref = aggregate_call_usages(calls, deduplicate_uuid=True)
    assert arm_ref["call_count"] == 2
    ev_t = aggregate_call_usages([c for c in calls if c["event"] == "T"], deduplicate_uuid=True)
    ev_c = aggregate_call_usages([c for c in calls if c["event"] == "C"], deduplicate_uuid=True)
    chk = assert_hierarchical_ledger_conservation(
        reference_arm_ledger=arm_ref,
        named_partition_ledgers={"events": [ev_t, ev_c]},
    )
    assert chk["conserved"] is True

    # Simulate legacy `v1` contaminating `proxy_v2_u_010c_4o` -> MUST fail conservation!
    polluted_t = {**ev_t, "proxy_v2_u_010c_4o": ev_t["proxy_v1_u_025c_1o"]}
    with pytest.raises(AssertionError, match="proxy_v2_u_010c_4o"):
        assert_hierarchical_ledger_conservation(
            reference_arm_ledger=arm_ref,
            named_partition_ledgers={"events": [polluted_t, ev_c]},
        )


def test_n02_n03_f06_three_layer_kernel_and_preallocation_json_bounds() -> None:
    # 1. Pre-allocation JSON ndarray bomb (`shape = [5000, 5000] == 25M > MAX_RPC_ELEMENTS`) rejected BEFORE `np.asarray`!
    with pytest.raises(ValueError, match="Pre-allocation ndarray element budget exceeded"):
        m941k.decode_pure_json_numerical_payload({"__nd__": True, "dtype": "float64", "shape": [5000, 5000], "data": []})

    # 2. `infer_contract_kind` detects dual-channel Beta from `gt["terminal_field_B"]` (`N03` fix!)
    ck = m941k.infer_contract_kind(
        family="beta",
        inp={"source_modes": np.zeros((4, 6, 6))},
        gt={"w_opt": np.zeros(4), "terminal_field_A": np.zeros((6, 6)), "terminal_field_B": np.zeros((6, 6))},
        out={"weights": np.zeros(4), "terminal_field_channel_a": np.zeros((6, 6)), "terminal_field_channel_b": np.zeros((6, 6))},
    )
    assert ck == "beta_dual_channel_kkt"


def test_n04_n07_strict_qualification_receipt_and_immutable_multiconflict_lineage() -> None:
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, _, reg = m96.initialize_m96_multiparent_mechanism_graph(family="alpha", forest=forest)

    code0 = "def solve_0(d):\n    return {'v': 0}\n"
    code1 = "def solve_1(d):\n    return {'v': 1}\n"
    code2 = "def solve_2(d):\n    return {'v': 2}\n"
    sha0 = hashlib.sha256(code0.encode()).hexdigest()

    # Receipt for a DIFFERENT module SHA must be rejected even if `module_digest` matches that other SHA!
    rec_other = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256="a" * 64,
        forest_node_id=nid,
    )
    bad_bind = m94.MechanismExecutableBinding(
        binding_id="bad_or",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id="m_bad_or",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="c",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="m",
        module_digest="a" * 64,
        function_name="solve_0",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=rec_other.receipt_id,
        parameter_mechanism_id="alpha_wing_a_calibration",
        parent_parameter_mechanism_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
        parameter_version=1,
        canonical_algorithm_code=code0,
        canonical_entrypoint="solve_0",
    )
    with pytest.raises(ValueError, match="authoritative executed"):
        reg.register_binding(bad_bind, strict_receipt_check=True)

    rec0 = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha0,
        forest_node_id=nid,
        environment_digest="sha256:env0",
        evidence_digest="sha256:ev0",
    )
    b0 = m94.MechanismExecutableBinding(
        binding_id="bind_t_b00",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id="mech_t_b00",
        mechanism_version=1,
        forest_node_id=nid,
        component_id="c0",
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem="b00",
        module_digest=sha0,
        function_name="solve_0",
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=rec0.receipt_id,
        parameter_mechanism_id="alpha_wing_a_calibration",
        parent_parameter_mechanism_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
        parameter_version=1,
        environment_digest="sha256:env0",
        evidence_receipt_id="sha256:ev0",
        selection_priority=30,
        output_schema_keys=["v"],
        canonical_algorithm_code=code0,
        canonical_entrypoint="solve_0",
    )
    reg.register_binding(b0, strict_receipt_check=True)

    # Mutating receipt content without reissuing ID must fail closed!
    orig_env = rec0.environment_digest
    rec0.environment_digest = "sha256:tampered"
    with pytest.raises(ValueError, match="Tampered QualificationReceipt"):
        m94.verify_binding_identity_and_foreign_keys(b0, graph=graph, registry=reg, require_closure_code=True, strict_receipt_check=True)
    rec0.environment_digest = orig_env

    # `C0 -> C1 -> C2 -> restore C0 -> restore C1`
    ctrl = m96.ImmutableBranchLineageController(world_id="t", family="alpha", graph=graph, registry=reg)
    ctrl.record_tested_witness(binding=b0, public_input_sha256="x1", passed=False, failure_kind="scientific_or_numerical", features={"f": 1.0})
    b1 = ctrl.mint_immutable_conflict_branch(
        failed_binding_id="bind_t_b00",
        new_canonical_code=code1,
        new_entrypoint="solve_1",
        forest_node_id=nid,
        qualification_protocol="hegel_public_scientific_gate_v4",
        output_schema_keys=["v"],
        validated_parent_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
        new_applicability_guard={"feature_name": "f", "op": ">", "threshold": 0.5},
        updated_failed_branch_applicability_guard={"feature_name": "f", "op": "<=", "threshold": 0.5},
        violated_component_id="c0",
        preserved_component_ids=[],
        selection_priority=20,
    )
    ctrl.record_tested_witness(binding=b1, public_input_sha256="x2", passed=False, failure_kind="scientific_or_numerical", features={"f": 2.0})
    b2 = ctrl.mint_immutable_conflict_branch(
        failed_binding_id=b1.binding_id,
        new_canonical_code=code2,
        new_entrypoint="solve_2",
        forest_node_id=nid,
        qualification_protocol="hegel_public_scientific_gate_v4",
        output_schema_keys=["v"],
        validated_parent_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
        new_applicability_guard={"feature_name": "f", "op": ">", "threshold": 1.5},
        updated_failed_branch_applicability_guard={
            "all_of": [
                {"feature_name": "f", "op": ">", "threshold": 0.5},
                {"feature_name": "f", "op": "<=", "threshold": 1.5},
            ]
        },
        violated_component_id="c1",
        preserved_component_ids=[],
        selection_priority=10,
    )
    assert b2.binding_id == "bind_t_b02"
    assert reg.bindings["bind_t_b01"].canonical_algorithm_code == code1
    assert ctrl.select_branch_deterministically(req_schema_keys=["v"], feature_evaluator=lambda _: {"f": 0.2})["selected"].binding_id == "bind_t_b00"
    assert ctrl.select_branch_deterministically(req_schema_keys=["v"], feature_evaluator=lambda _: {"f": 1.0})["selected"].binding_id == "bind_t_b01"
    assert ctrl.select_branch_deterministically(req_schema_keys=["v"], feature_evaluator=lambda _: {"f": 2.0})["selected"].binding_id == "bind_t_b02"


def test_n04_w03_containerized_perturbation_sensitivity_and_n09_funnel_loader(tmp_path: Path) -> None:
    # 1. `N09` funnel loader reads `04d_skill_contract_inspections.jsonl` + `skill_activations` and honors `arm_has_new_source_assets=False`
    ep_d = tmp_path / "ep_t"
    ep_d.mkdir()
    (ep_d / "00_episode_summary.json").write_text(
        json.dumps({"skill_activations": [], "skill_adoption_funnel": {"search_skill_query_count": 1}}),
        encoding="utf-8",
    )
    (ep_d / "04d_skill_contract_inspections.jsonl").write_text(
        json.dumps({"event": "skill_search_executed", "ranked_function_ids": ["autogenous_prim_box_kkt.solve"]}) + "\n"
        + json.dumps({"event": "skill_contract_delivered", "target": "autogenous_prim_box_kkt.solve", "matched_modules": ["autogenous_prim_box_kkt"]}) + "\n",
        encoding="utf-8",
    )
    f_full = m64.load_and_classify_episode_source_aware_funnel(
        episode_dir=ep_d,
        new_source_l2_stems={"l2_other"},
        new_source_l1_stems={"autogenous_prim_box_kkt"},
        arm_has_new_source_assets=True,
    )
    assert f_full["new_source_L1"]["search_disclosed"] is True
    assert f_full["new_source_L1"]["contract_described"] is True
    assert f_full["new_source_L2"]["contract_described"] is False

    f_flat = m64.load_and_classify_episode_source_aware_funnel(
        episode_dir=ep_d,
        new_source_l2_stems={"l2_other"},
        new_source_l1_stems={"autogenous_prim_box_kkt"},
        arm_has_new_source_assets=False,
    )
    assert f_flat["new_source_L1"]["assets_available"] is False

    # 2. `W03` Container-isolated perturbation sensitivity on `dev_alpha_m96_00` (`0` host untrusted candidate execution!)
    dev_alpha = m96.DEV_M96_WORLD_SPECS[0]
    _, stream_tasks, manifest = m96.build_m96_world_tasks(dev_alpha)
    c_task = next(t for t, m in zip(stream_tasks, manifest) if m["causal_event"] == "C")
    with tempfile.TemporaryDirectory(prefix="m961_pub_") as td:
        ns: dict[str, object] = {}
        exec(compile(c_task.public_setup_py.replace("/holdout_task", td), "<pub>", "exec"), ns)
        pub_c = pickle.loads((Path(td) / f"public_{c_task.short_name}.pkl").read_bytes())[0]
    c1_code = m96.build_m96_reference_dev_solver_code("alpha", "solve_dev_c1", "C1")
    sens = m96.verify_upstream_parent_perturbation_sensitivity(
        family="alpha",
        candidate_upstream_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration", "sentinel_shared_acoustic_root"],
        proposed_upstream_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
        c1_canonical_code=c1_code,
        c1_entrypoint="solve_dev_c1",
        public_item=pub_c,
        base_param_payload={"stress_gains": list(dev_alpha.v2_gains), "stress_biases": list(dev_alpha.v2_offsets)},
        run_in_container=True,
    )
    assert sens["executed_in_isolated_container"] is True
    assert sens["proposal_matches_empirical_sensitivity"] is True
