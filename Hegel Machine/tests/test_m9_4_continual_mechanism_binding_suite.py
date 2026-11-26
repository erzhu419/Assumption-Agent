"""Unit & Air-Gapped Container Verification Suite for `M9.4` (`MechanismExecutableBinding` & `S -> T -> G -> C -> R` World)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (
    M94_ALL_TASKS,
    M94_CHAIN_ALPHA_SOURCE,
    M94_CHAIN_ALPHA_STREAM,
    M94_CHAIN_BETA_SOURCE,
    M94_CHAIN_BETA_STREAM,
    M94BindingRegistry,
    MechanismExecutableBinding,
    build_parameterized_source_closure_code,
    evaluate_bound_mechanism_candidate_in_container,
    grade_m94_artifact_map as run_holdout_private_verifier,
    initialize_continual_mechanism_graph_with_sentinel,
    run_m8_shared_calibration_update_from_public_dataset,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def _materialize_public_item(task: Any) -> dict[str, Any]:
    import pickle

    with tempfile.TemporaryDirectory(prefix="m94_pub_") as td:
        td_p = Path(td)
        patched = task.public_setup_py.replace("/holdout_task", str(td_p))
        ns: dict[str, Any] = {}
        exec(compile(patched, f"<pub_{task.short_name}>", "exec"), ns)
        pkl_p = td_p / f"public_{task.short_name}.pkl"
        return pickle.loads(pkl_p.read_bytes())[0]


def test_m94_neutral_task_ids_and_checkpoint_roundtrip() -> None:
    # 1. Hidden schedule check: zero stage/formula giveaways in task IDs!
    forbidden_tokens = ("stage_t", "stage_g", "stage_c", "stage_r", "bingham", "perzyna", "vortex", "recovery")
    for t in M94_ALL_TASKS:
        low_id = t.task_id.lower()
        assert not any(tok in low_id for tok in forbidden_tokens), f"Task ID {t.task_id} leaks schedule/formula hint!"

    # 2. PersistentMechanismGraph + M94BindingRegistry checkpoint round-trip (`0.0` drift!)
    graph, _planner, reg = initialize_continual_mechanism_graph_with_sentinel(PhilosophicalForest())
    b_alpha = MechanismExecutableBinding(
        binding_id="bind_alpha_c0",
        chain_id="alpha",
        branch_tag="C0_base",
        mechanism_id="mech_alpha_c0",
        mechanism_version=1,
        forest_node_id="node_alpha_src",
        component_id="linear_viscoelastic_memory",
        preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
        violated_component_id=None,
        module_stem="autogenous_alpha_hereditary",
        module_digest=hashlib.sha256(b"alpha").hexdigest(),
        function_name="solve_alpha",
        qualification_id="hegel_l1_qual_4tuple_v2",
        parameter_mechanism_id="alpha_shared_loadcell_calibration",
        parameter_version=1,
        precondition_guard=m7a.ObservablePreconditionGuard(
            guard_id="g_alpha_c0",
            observable_extractor="overstress_excess_max_history_and_hold",
            op="<=",
            value=0.0,
            applies_to_component_id="linear_viscoelastic_memory",
        ).to_dict(),
        output_schema_keys=["estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"],
    )
    reg.register_binding(b_alpha, reason="unit_test_init")

    with tempfile.TemporaryDirectory(prefix="m94_ckpt_") as td:
        ckpt_p = Path(td) / "m94_graph.json"
        graph.save_checkpoint(ckpt_p)
        restored_graph = PersistentMechanismGraph.load_checkpoint(ckpt_p)
        restored_reg = M94BindingRegistry(restored_graph)
        assert "bind_alpha_c0" in restored_reg.bindings
        assert restored_reg.bindings["bind_alpha_c0"].parameter_version == 1
        assert "sentinel_acoustic_calibration" in restored_graph.active_task_models


def test_m94_continual_s_t_g_c_r_end_to_end_mechanics_and_private_verifiers() -> None:
    """Verify the full `S -> T -> G -> C -> R` mechanics on BOTH Chain Alpha and Chain Beta:
    - `S` (`s00`) passes held-out private Docker verifier under `param_v1`.
    - `T` (`ep01`) is rejected for direct `C0` reuse due to shifted readout interface (`interface_adaptation_required`).
    - `G` (`ep02`) triggers `M8` `AncestorReweight` (`param_v1 -> param_v2`) with `0.0` drift on `sentinel_acoustic_calibration`:
        * Stale `param_v1` `C0` FAILS held-out private verifier on `ep02` (`strict_pass == False`),
        * Bound `C0` + `AncestorReweight` `param_v2` PASSES guard, public schema, scientific probe, AND held-out private verifier (`strict_pass == True`)!
    - `C` (`ep03`) triggers `ObservablePreconditionGuard == FALSE` on `C0` (rejecting direct `C0` execution!), and after `C1`
      (`> 0` guard) is registered alongside `C0` (`<= 0` guard):
    - `R` (`ep04`) evaluates guards on fresh undamaged / additive-separable `V(x,y)=a(x)+b(y)` data in `G`'s `param_v2` epoch:
        * `guard(C1) == FALSE`, `guard(C0) == TRUE`,
        * Stale `Stage S` (`param_v1`) FAILS on `ep04` (`strict_pass == False`),
        * Bound `C0@param_v2` PASSES guard, scientific probe, AND held-out private verifier (`strict_pass == True`)!
    """
    graph, planner, reg = initialize_continual_mechanism_graph_with_sentinel(PhilosophicalForest())

    reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_alpha_c0",
            chain_id="alpha",
            branch_tag="C0_base",
            mechanism_id="mech_alpha_c0",
            mechanism_version=1,
            forest_node_id="node_alpha_s00",
            component_id="linear_viscoelastic_memory",
            preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
            violated_component_id=None,
            module_stem="autogenous_alpha_c0",
            module_digest=hashlib.sha256(b"a0").hexdigest(),
            function_name="solve_alpha",
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="alpha_shared_loadcell_calibration",
            parameter_version=1,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="g_alpha_c0",
                observable_extractor="overstress_excess_max_history_and_hold",
                op="<=",
                value=0.0,
                applies_to_component_id="linear_viscoelastic_memory",
            ).to_dict(),
            output_schema_keys=["estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"],
        )
    )
    reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_beta_c0",
            chain_id="beta",
            branch_tag="C0_base",
            mechanism_id="mech_beta_c0",
            mechanism_version=1,
            forest_node_id="node_beta_s00",
            component_id="separable_kronecker_transport",
            preserved_component_ids=["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"],
            violated_component_id=None,
            module_stem="autogenous_beta_c0",
            module_digest=hashlib.sha256(b"b0").hexdigest(),
            function_name="solve_beta",
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="beta_shared_sensor_bank_calibration",
            parameter_version=1,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="g_beta_c0",
                observable_extractor="nonseparable_2d_additive_residual_ratio",
                op="<=",
                value=1e-6,
                applies_to_component_id="separable_kronecker_transport",
            ).to_dict(),
            output_schema_keys=["weights", "terminal_field", "kkt_certificate"],
        )
    )

    # 1. Stage S (`alpha_s00` & `beta_s00`) pass private Docker verifiers under `param_v1`
    v1_alpha_src = build_parameterized_source_closure_code(
        "alpha", "solve_alpha_s00", {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_ALPHA_SOURCE, {"/holdout_task/alpha_s00_solver.py": v1_alpha_src})["strict_pass"] is True

    v1_beta_src = build_parameterized_source_closure_code(
        "beta", "solve_beta_s00", {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_BETA_SOURCE, {"/holdout_task/beta_s00_solver.py": v1_beta_src})["strict_pass"] is True

    reg.bindings["bind_alpha_c0"].parameterized_closure_code = v1_alpha_src
    reg.bindings["bind_alpha_c0"].parameterized_entrypoint = "solve_alpha_s00"
    reg.bindings["bind_alpha_c0"].parameterized_closure_sha256 = hashlib.sha256(v1_alpha_src.encode("utf-8")).hexdigest()
    reg.bindings["bind_beta_c0"].parameterized_closure_code = v1_beta_src
    reg.bindings["bind_beta_c0"].parameterized_entrypoint = "solve_beta_s00"
    reg.bindings["bind_beta_c0"].parameterized_closure_sha256 = hashlib.sha256(v1_beta_src.encode("utf-8")).hexdigest()
    reg.sync_to_graph()

    # 2. ep01 (`T`): Sequential M8 check (`NoChange`, `v=1`) + interface adaptation required -> NOT short-circuited by C0
    pub_a_ep01 = _materialize_public_item(M94_CHAIN_ALPHA_STREAM[0])
    upd_a_t = run_m8_shared_calibration_update_from_public_dataset(
        chain_id="alpha", episode_id="alpha_ep01", public_item=pub_a_ep01, graph=graph, planner=planner, registry=reg
    )
    assert upd_a_t["planner_selected_action"] == "NoChange" and upd_a_t["shared_mechanism_version"] == 1
    tr_a_ep01 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_ALPHA_STREAM[0], chain_id="alpha", graph=graph, registry=reg, public_item=pub_a_ep01
    )
    assert tr_a_ep01["accepted_bound_execution"] is False and tr_a_ep01["reason"] == "interface_adaptation_required"

    pub_b_ep01 = _materialize_public_item(M94_CHAIN_BETA_STREAM[0])
    upd_b_t = run_m8_shared_calibration_update_from_public_dataset(
        chain_id="beta", episode_id="beta_ep01", public_item=pub_b_ep01, graph=graph, planner=planner, registry=reg
    )
    assert upd_b_t["planner_selected_action"] == "NoChange" and upd_b_t["shared_mechanism_version"] == 1

    # 3. ep02 (`G`): Stale `param_v1` FAILS private verifier, while `M8` `AncestorReweight` (`param_v2`, `sentinel` drift `0.0`) PASSES!
    stale_a_ep02 = build_parameterized_source_closure_code(
        "alpha", "solve_alpha_ep02", {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_ALPHA_STREAM[1], {"/holdout_task/alpha_ep02_solver.py": stale_a_ep02})["strict_pass"] is False

    pub_a_ep02 = _materialize_public_item(M94_CHAIN_ALPHA_STREAM[1])
    upd_a_g = run_m8_shared_calibration_update_from_public_dataset(
        chain_id="alpha", episode_id="alpha_ep02", public_item=pub_a_ep02, graph=graph, planner=planner, registry=reg
    )
    assert upd_a_g["planner_selected_action"] == "AncestorReweight"
    assert upd_a_g["shared_mechanism_version"] == 2
    assert upd_a_g["sentinel_max_abs_prediction_drift"] == 0.0
    tr_a_ep02 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_ALPHA_STREAM[1], chain_id="alpha", graph=graph, registry=reg, public_item=pub_a_ep02
    )
    assert tr_a_ep02["accepted_bound_execution"] is True
    assert run_holdout_private_verifier(M94_CHAIN_ALPHA_STREAM[1], {"/holdout_task/alpha_ep02_solver.py": tr_a_ep02["candidate_code"]})["strict_pass"] is True

    # Same for Chain Beta ep02 (`G`)
    stale_b_ep02 = build_parameterized_source_closure_code(
        "beta", "solve_beta_ep02", {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_BETA_STREAM[1], {"/holdout_task/beta_ep02_solver.py": stale_b_ep02})["strict_pass"] is False

    pub_b_ep02 = _materialize_public_item(M94_CHAIN_BETA_STREAM[1])
    upd_b_g = run_m8_shared_calibration_update_from_public_dataset(
        chain_id="beta", episode_id="beta_ep02", public_item=pub_b_ep02, graph=graph, planner=planner, registry=reg
    )
    assert upd_b_g["planner_selected_action"] == "AncestorReweight"
    assert upd_b_g["sentinel_max_abs_prediction_drift"] == 0.0
    tr_b_ep02 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_BETA_STREAM[1], chain_id="beta", graph=graph, registry=reg, public_item=pub_b_ep02
    )
    assert tr_b_ep02["accepted_bound_execution"] is True
    assert run_holdout_private_verifier(M94_CHAIN_BETA_STREAM[1], {"/holdout_task/beta_ep02_solver.py": tr_b_ep02["candidate_code"]})["strict_pass"] is True

    # 4. ep03 (`C`): Observable guard evaluates `FALSE` on `C0` for BOTH Alpha & Beta! Register `C1_extension` (`> 0` guard)
    pub_a_ep03 = _materialize_public_item(M94_CHAIN_ALPHA_STREAM[2])
    tr_a_ep03 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_ALPHA_STREAM[2], chain_id="alpha", graph=graph, registry=reg, public_item=pub_a_ep03
    )
    assert tr_a_ep03["accepted_bound_execution"] is False
    assert tr_a_ep03["guard_evaluations"][0]["guard_result"]["state"] == "FALSE"

    pub_b_ep03 = _materialize_public_item(M94_CHAIN_BETA_STREAM[2])
    tr_b_ep03 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_BETA_STREAM[2], chain_id="beta", graph=graph, registry=reg, public_item=pub_b_ep03
    )
    assert tr_b_ep03["accepted_bound_execution"] is False
    assert tr_b_ep03["guard_evaluations"][0]["guard_result"]["state"] == "FALSE"

    # Register C1 extensions so ep04 (`R`) must genuinely arbitrate between `{C0, C1}` via executable guards!
    reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_alpha_c1",
            chain_id="alpha",
            branch_tag="C1_extension",
            mechanism_id="mech_alpha_c1",
            mechanism_version=1,
            forest_node_id="node_alpha_c1",
            component_id="viscoplastic_overstress_extension",
            preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
            violated_component_id="linear_viscoelastic_memory",
            module_stem="autogenous_alpha_c1",
            module_digest=hashlib.sha256(b"a1").hexdigest(),
            function_name="solve_alpha_c1",
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="alpha_shared_loadcell_calibration",
            parameter_version=2,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="g_alpha_c1",
                observable_extractor="overstress_excess_max_history_and_hold",
                op=">",
                value=0.0,
                applies_to_component_id="linear_viscoelastic_memory",
            ).to_dict(),
            output_schema_keys=["estimated_taus", "viscoplastic_weights", "predicted_strain"],
        )
    )
    reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_beta_c1",
            chain_id="beta",
            branch_tag="C1_extension",
            mechanism_id="mech_beta_c1",
            mechanism_version=1,
            forest_node_id="node_beta_c1",
            component_id="nonseparable_channel_b_extension",
            preserved_component_ids=["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"],
            violated_component_id="separable_kronecker_transport",
            module_stem="autogenous_beta_c1",
            module_digest=hashlib.sha256(b"b1").hexdigest(),
            function_name="solve_beta_c1",
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="beta_shared_sensor_bank_calibration",
            parameter_version=2,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="g_beta_c1",
                observable_extractor="nonseparable_2d_additive_residual_ratio",
                op=">",
                value=1e-6,
                applies_to_component_id="separable_kronecker_transport",
            ).to_dict(),
            output_schema_keys=["weights", "terminal_field_channel_a", "terminal_field_channel_b", "kkt_certificate"],
        )
    )

    # 5. ep04 (`R`): Stale `param_v1` FAILS, while guard rejects `C1` (`FALSE`), selects `C0@param_v2` (`TRUE`), and PASSES!
    stale_a_ep04 = build_parameterized_source_closure_code(
        "alpha", "solve_alpha_ep04", {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_ALPHA_STREAM[3], {"/holdout_task/alpha_ep04_solver.py": stale_a_ep04})["strict_pass"] is False

    pub_a_ep04 = _materialize_public_item(M94_CHAIN_ALPHA_STREAM[3])
    run_m8_shared_calibration_update_from_public_dataset(
        chain_id="alpha", episode_id="alpha_ep04", public_item=pub_a_ep04, graph=graph, planner=planner, registry=reg
    )
    tr_a_ep04 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_ALPHA_STREAM[3], chain_id="alpha", graph=graph, registry=reg, public_item=pub_a_ep04
    )
    assert tr_a_ep04["accepted_bound_execution"] is True
    assert tr_a_ep04["selected_binding_id"] == "bind_alpha_c0"
    g_map_a = {ev["binding_id"]: ev["guard_result"]["state"] for ev in tr_a_ep04["guard_evaluations"]}
    assert g_map_a == {"bind_alpha_c0": "TRUE", "bind_alpha_c1": "FALSE"}
    assert run_holdout_private_verifier(M94_CHAIN_ALPHA_STREAM[3], {"/holdout_task/alpha_ep04_solver.py": tr_a_ep04["candidate_code"]})["strict_pass"] is True

    stale_b_ep04 = build_parameterized_source_closure_code(
        "beta", "solve_beta_ep04", {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    assert run_holdout_private_verifier(M94_CHAIN_BETA_STREAM[3], {"/holdout_task/beta_ep04_solver.py": stale_b_ep04})["strict_pass"] is False

    pub_b_ep04 = _materialize_public_item(M94_CHAIN_BETA_STREAM[3])
    run_m8_shared_calibration_update_from_public_dataset(
        chain_id="beta", episode_id="beta_ep04", public_item=pub_b_ep04, graph=graph, planner=planner, registry=reg
    )
    tr_b_ep04 = evaluate_bound_mechanism_candidate_in_container(
        task=M94_CHAIN_BETA_STREAM[3], chain_id="beta", graph=graph, registry=reg, public_item=pub_b_ep04
    )
    assert tr_b_ep04["accepted_bound_execution"] is True
    assert tr_b_ep04["selected_binding_id"] == "bind_beta_c0"
    g_map_b = {ev["binding_id"]: ev["guard_result"]["state"] for ev in tr_b_ep04["guard_evaluations"]}
    assert g_map_b == {"bind_beta_c0": "TRUE", "bind_beta_c1": "FALSE"}
    assert run_holdout_private_verifier(M94_CHAIN_BETA_STREAM[3], {"/holdout_task/beta_ep04_solver.py": tr_b_ep04["candidate_code"]})["strict_pass"] is True
