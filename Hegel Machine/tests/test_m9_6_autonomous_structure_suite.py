"""Pre-Freeze Deterministic (`0`-LLM) Verification Suite for `M9.6` (`DEV_M96_WORLD_SPECS` ONLY — Zero Confirmation Seed Touch!)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np
import pytest

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_6_autonomous_structure_continual_world as m96,
)


def _materialize_pub(task: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="t_m96_") as td:
        ns: dict[str, Any] = {}
        exec(compile(task.public_setup_py.replace("/holdout_task", td), "<pub>", "exec"), ns)
        return pickle.loads((Path(td) / f"public_{task.short_name}.pkl").read_bytes())[0]


def test_m96_multiparent_selective_propagation_and_perturbation_sensitivity() -> None:
    """Verify `§4.2` true multi-parent `(wing_a, wing_b)` graph propagation + `0.0` drift on `aux_wing_b_monitor` & `sentinel`!"""
    dev_alpha = m96.DEV_M96_WORLD_SPECS[0]
    assert dev_alpha.split == "dev"
    s_task, stream_tasks, manifest = m96.build_m96_world_tasks(dev_alpha)
    by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}

    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    real_nid = next(iter(forest.nodes.keys()))
    graph, planner, reg = m96.initialize_m96_multiparent_mechanism_graph(family="alpha", forest=forest)

    c0_code = m96.build_m96_reference_dev_solver_code("alpha", s_task.declared_callable_entrypoints[0], "C0")
    c1_code = m96.build_m96_reference_dev_solver_code("alpha", "solve_dev_c1", "C1")
    uni_keys = m96.expected_m96_public_output_keys(s_task)

    for b_tag, b_code, b_ep in (("C0_base", c0_code, s_task.declared_callable_entrypoints[0]), ("C1_extension", c1_code, "solve_dev_c1")):
        q_rec = reg.issue_qualification_receipt(
            qualification_protocol="hegel_public_scientific_gate_v4",
            verified_code_sha256=hashlib.sha256(b_code.encode()).hexdigest(),
            forest_node_id=real_nid,
        )
        reg.register_binding(
            m94.MechanismExecutableBinding(
                binding_id=f"bind_dev_{b_tag}",
                chain_id="alpha",
                branch_tag=b_tag,
                mechanism_id=f"mech_dev_{b_tag}",
                mechanism_version=1,
                forest_node_id=real_nid,
                component_id="comp",
                preserved_component_ids=[],
                violated_component_id=None,
                module_stem="s",
                module_digest=q_rec.verified_code_sha256,
                function_name=b_ep,
                qualification_id="hegel_public_scientific_gate_v4",
                qualification_receipt_id=q_rec.receipt_id,
                parameter_mechanism_id="alpha_wing_a_calibration",
                parent_parameter_mechanism_ids=["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
                parameter_version=1,
                structure_origin="unconstrained_initial_source",
                output_schema_keys=uni_keys,
                canonical_algorithm_code=b_code,
                canonical_entrypoint=b_ep,
                parameterized_closure_code=b_code,
                parameterized_entrypoint=b_ep,
            ),
            strict_receipt_check=True,
        )
        assert graph.mechanisms[f"mech_dev_{b_tag}"].all_parent_ids() == ("alpha_wing_a_calibration", "alpha_wing_b_calibration")

    # Step G1: both wings shift to `v2`
    pub_g1 = _materialize_pub(by_ev["G1"])
    r_g1 = m96.run_m96_multiparent_calibration_update(
        family="alpha", episode_id="dev_g1", public_item=pub_g1, graph=graph, planner=planner, registry=reg
    )
    assert r_g1["multiparent_fanout_count"] == 2
    assert r_g1["sentinel_max_abs_prediction_drift"] == 0.0

    # Step G2: Wing A shifts to `v3`, Wing B has EMI spike (`NoChange` -> `max_weight_drift == 0.0`!)
    pub_g2 = _materialize_pub(by_ev["G2"])
    r_g2 = m96.run_m96_multiparent_calibration_update(
        family="alpha", episode_id="dev_g2", public_item=pub_g2, graph=graph, planner=planner, registry=reg
    )
    assert r_g2["wing_receipts"]["wing_a"]["selected_action"] == "AncestorReweight"
    assert r_g2["wing_receipts"]["wing_b"]["selected_action"] == "NoChange"
    assert r_g2["wing_receipts"]["wing_b"]["max_weight_drift"] == 0.0
    assert r_g2["multiparent_fanout_count"] == 2
    assert r_g2["aux_bench_c_monitor_drift"] == 0.0
    assert r_g2["sentinel_max_abs_prediction_drift"] == 0.0

    # Perturbation sensitivity check rejects `sentinel` & `aux_bench_c`, accepting ONLY `(wing_a, wing_b)`!
    pub_c = _materialize_pub(by_ev["C"])
    cand_up = ["alpha_wing_a_calibration", "alpha_wing_b_calibration", "aux_bench_c_calibration", "sentinel_shared_acoustic_root"]
    s_bad = m96.verify_upstream_parent_perturbation_sensitivity(
        family="alpha",
        candidate_upstream_ids=cand_up,
        proposed_upstream_ids=["alpha_wing_a_calibration", "sentinel_shared_acoustic_root"],
        c1_canonical_code=c1_code,
        c1_entrypoint="solve_dev_c1",
        public_item=pub_c,
        base_param_payload={"stress_gains": list(dev_alpha.v2_gains), "stress_biases": list(dev_alpha.v2_offsets)},
    )
    assert s_bad["proposal_matches_empirical_sensitivity"] is False
    assert set(s_bad["empirically_sensitive_upstream_ids"]) == {"alpha_wing_a_calibration", "alpha_wing_b_calibration"}
    assert s_bad["sensitivity_max_diff_by_upstream_id"]["sentinel_shared_acoustic_root"] == 0.0
    assert s_bad["sensitivity_max_diff_by_upstream_id"]["aux_bench_c_calibration"] == 0.0

    # Mocked LLM `propose_and_contrastively_validate_autonomous_split`: first proposes bad parent (`sentinel`), gets rejected, then proposes valid history guard + `(wing_a, wing_b)`!
    pub_s = _materialize_pub(s_task)
    f_s = m96.compute_generic_public_structural_features(pub_s, {"stress_gains": [1.0]*4, "stress_biases": [0.0]*4})
    f_c = m96.compute_generic_public_structural_features(pub_c, {"stress_gains": list(dev_alpha.v2_gains), "stress_biases": list(dev_alpha.v2_offsets)})

    class _MockSplitClient:
        def __init__(self) -> None:
            self.calls = 0

        def call_structured(self, **kw: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "ok": True,
                    "prompt_tokens": 100,
                    "cached_tokens": 0,
                    "completion_tokens": 40,
                    "parsed": {
                        "mechanistic_diagnosis": "bad attempt including sentinel",
                        "violated_component_id": "linear_prony",
                        "preserved_component_ids": ["panel_alignment"],
                        "dependent_upstream_mechanism_ids": ["alpha_wing_a_calibration", "sentinel_shared_acoustic_root"],
                        "c0_guard": {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": "<=", "threshold": 0.0},
                        "c1_guard": {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": ">", "threshold": 0.0},
                    },
                }
            return {
                "ok": True,
                "prompt_tokens": 120,
                "cached_tokens": 80,
                "completion_tokens": 45,
                "parsed": {
                    "mechanistic_diagnosis": "valid history guard and exact multiparent calibration roots",
                    "violated_component_id": "linear_prony",
                    "preserved_component_ids": ["panel_alignment"],
                    "dependent_upstream_mechanism_ids": ["alpha_wing_a_calibration", "alpha_wing_b_calibration"],
                    "c0_guard": {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": "<=", "threshold": 0.0},
                    "c1_guard": {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": ">", "threshold": 0.0},
                },
            }

    mock_cli = _MockSplitClient()
    ind_out = m96.propose_and_contrastively_validate_autonomous_split(
        client=mock_cli,
        stage_subdir="mock_ind",
        family="alpha",
        task_instruction_text=by_ev["C"].instruction_text,
        c0_code=c0_code,
        c1_code=c1_code,
        c1_entrypoint="solve_dev_c1",
        positive_c0_witness_features=[f_s],
        negative_c0_witness_features=[f_c],
        current_public_item=pub_c,
        current_param_payload={"stress_gains": list(dev_alpha.v2_gains), "stress_biases": list(dev_alpha.v2_offsets)},
        candidate_upstream_ids=cand_up,
    )
    assert ind_out["accepted"] is True and ind_out["attempts_used"] == 2
    assert ind_out["structure_origin"] == "publicly_contrastively_validated"
    assert set(ind_out["validated_parent_mechanism_ids"]) == {"alpha_wing_a_calibration", "alpha_wing_b_calibration"}


def test_m96_non_markovian_U_separates_history_guards_from_myopic_traps_and_shared_kkt_T() -> None:
    """Verify `§4.1 + §4.3 + §4.4` across BOTH `dev_alpha_m96_00` and `dev_beta_m96_00` (`0` LLM calls):
    1. Myopic snapshot guard passes `{S, C}` contrastively at Step `C`, FAILS on non-Markovian `U`, and is REJECTED once `U` enters `W_neg`!
    2. History-inclusive guard passes `{S, C, U, R}` contrastively and routes `U -> C1` and `R -> C0` with `1` probe (`strict_pass=True`)!
    3. Alpha `T_shared_kkt` Box-Simplex KKT solver passes BOTH public isolated-worker probe AND 2-seed private verifier!
    """
    for spec in m96.DEV_M96_WORLD_SPECS:
        assert spec.split == "dev"
        fam = spec.family
        s_task, stream_tasks, manifest = m96.build_m96_world_tasks(spec)
        by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}

        pub_s = _materialize_pub(s_task)
        pub_c = _materialize_pub(by_ev["C"])
        pub_r = _materialize_pub(by_ev["R"])
        pub_u = _materialize_pub(by_ev["U"])

        v1_pay = {"stress_gains": [1.0]*4, "stress_biases": [0.0]*4} if fam == "alpha" else {"bank_gains": [1.0]*4, "bank_offsets": [0.0]*4}
        v2_pay = (
            {"stress_gains": list(spec.v2_gains), "stress_biases": list(spec.v2_offsets)}
            if fam == "alpha"
            else {"bank_gains": list(spec.v2_gains), "bank_offsets": list(spec.v2_offsets)}
        )

        f_s = m96.compute_generic_public_structural_features(pub_s, v1_pay)
        f_c = m96.compute_generic_public_structural_features(pub_c, v2_pay)
        f_r = m96.compute_generic_public_structural_features(pub_r, v2_pay)
        f_u = m96.compute_generic_public_structural_features(pub_u, v2_pay)

        if fam == "alpha":
            myopic_c0 = {"feature_name": "max_hold_signal_over_declared_threshold_excess", "op": "<=", "threshold": 0.0}
            myopic_c1 = {"feature_name": "max_hold_signal_over_declared_threshold_excess", "op": ">", "threshold": 0.0}
            hist_c0 = {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": "<=", "threshold": 0.0}
            hist_c1 = {"feature_name": "max_history_and_hold_plus_residual_state_excess", "op": ">", "threshold": 0.0}
        else:
            myopic_c0 = {"feature_name": "matrix_2d_nonseparable_additive_residual_ratio", "op": "<=", "threshold": 1e-5}
            myopic_c1 = {"feature_name": "matrix_2d_nonseparable_additive_residual_ratio", "op": ">", "threshold": 1e-5}
            hist_c0 = {"feature_name": "combined_nonseparable_or_vortex_history_excess", "op": "<=", "threshold": 1e-4}
            hist_c1 = {"feature_name": "combined_nonseparable_or_vortex_history_excess", "op": ">", "threshold": 1e-4}

        # On `{S}` vs `{C}`, BOTH myopic and history guards pass contrastive validation (proving `{S, C}` does NOT trivially leak `U`!)
        assert m96.verify_contrastive_public_witness_guards(
            c0_guard=myopic_c0, c1_guard=myopic_c1, positive_c0_witness_features=[f_s], negative_c0_witness_features=[f_c]
        )["contrastively_valid"] is True
        assert m96.verify_contrastive_public_witness_guards(
            c0_guard=hist_c0, c1_guard=hist_c1, positive_c0_witness_features=[f_s, f_r], negative_c0_witness_features=[f_c, f_u]
        )["contrastively_valid"] is True

        # On `U` (`same present input as R, different past!`), myopic guard MISROUTES `U -> C0` (`TRUE`), whereas history guard routes `U -> C1`!
        assert m96.evaluate_m96_autonomous_guard(myopic_c0, f_u)["state"] == "TRUE"
        assert m96.evaluate_m96_autonomous_guard(hist_c0, f_u)["state"] == "FALSE"
        assert m96.evaluate_m96_autonomous_guard(hist_c1, f_u)["state"] == "TRUE"
        assert m96.verify_contrastive_public_witness_guards(
            c0_guard=myopic_c0, c1_guard=myopic_c1, positive_c0_witness_features=[f_s], negative_c0_witness_features=[f_c, f_u]
        )["contrastively_valid"] is False

        # Verify in clean Docker container: `C0` adapted to `v2` FAILS `U`, whereas `C1` adapted to `v2` PASSES `U` (public gate AND private verifier)!
        c0_code = m96.build_m96_reference_dev_solver_code(fam, "solve_c0_ref", "C0")
        c1_code = m96.build_m96_reference_dev_solver_code(fam, "solve_c1_ref", "C1")
        u_task = by_ev["U"]
        u_ep = u_task.declared_callable_entrypoints[0]
        u_cand_c0 = m95.render_parameterized_input_adapter(family=fam, source_code=c0_code, source_entrypoint="solve_c0_ref", target_entrypoint=u_ep, param_payload=v2_pay)
        u_cand_c1 = m95.render_parameterized_input_adapter(family=fam, source_code=c1_code, source_entrypoint="solve_c1_ref", target_entrypoint=u_ep, param_payload=v2_pay)
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=u_task, candidate_code=u_cand_c0)["accepted"] is False
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=u_task, candidate_code=u_cand_c1)["accepted"] is True
        assert m94.grade_m94_artifact_map(u_task, {u_task.declared_artifacts[0]: u_cand_c1})["strict_pass"] is True

        if fam == "alpha":
            # Verify Alpha `T_shared_kkt` Box-Simplex KKT reference solver passes public probe + 2-seed private verifier
            t_task = by_ev["T"]
            t_code = m96.build_m96_reference_dev_solver_code("alpha", t_task.declared_callable_entrypoints[0], "T_kkt")
            assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=t_task, candidate_code=t_code)["accepted"] is True
            assert m94.grade_m94_artifact_map(t_task, {t_task.declared_artifacts[0]: t_code})["strict_pass"] is True
