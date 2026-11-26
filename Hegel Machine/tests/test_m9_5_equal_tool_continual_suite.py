"""Pre-Freeze Deterministic Unit & Integration Suite for `M9.5` (`DEV_WORLD_SPECS` ONLY — Zero Confirmation World Touch!).

Verifies all causal identification invariants BEFORE freezing `Commit 2`:
  1. `DEV_WORLD_SPECS` and `CONFIRMATION_WORLD_SPECS` use 100% disjoint seed ranges and anonymous neutral task IDs.
  2. Unified output contracts across ALL non-`T` steps (`S, N, G1, C, R, C_prime, G2, M`) defeat `RouteBySchema` (`A02`).
  3. On `N` and `G1` (immediate-calibration-sufficient negative controls), ALL THREE ARMS (`Archive_AutoReplay`,
     `Flat_ParameterizedCache`, `Full_Mechanism_Graph`) pass in `0` LLM turns (`A01` closed).
  4. On `C_prime`, `Flat_ParameterizedCache` AND `Full_Mechanism_Graph` (`guard(C1)==TRUE`) pass in `0` turns while stateless
     `Archive_AutoReplay` fails on sparse 2-row calibration (`A03` & `E03` closed).
  5. On `Beta R` (laminar separable recovery under unified schema), `Full_Mechanism_Graph` selects `C0@v2` via physical guard
     and passes in `0` turns while `Flat_ParameterizedCache` (trying most-recent `C1` vortex solver) fails KKT gate (`A02` closed).
  6. On `G2` and `M`, Wing-Scoped `M8VariationalPlanner` updates Wing A (`v2 -> v3`, `fanout = 2` updating `{C0, C1}` simultaneously!),
     selects `NoChange` on EMI-spiked sibling Wing B (`0.0` sibling drift!), maintains `0.0` drift on `sentinel_acoustic_calibration`,
     and passes `G2` and `M` in `0` turns while BOTH `Archive_AutoReplay` and `Flat_ParameterizedCache` are rejected by the shared gate (`A06` closed)!
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import tempfile

from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (
    MechanismExecutableBinding,
    grade_m94_artifact_map,
)
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (
    CONFIRMATION_WORLD_SPECS,
    DEV_WORLD_SPECS,
    build_reference_dev_solver_code,
    build_world_tasks,
    expected_public_output_keys,
    initialize_m95_wing_scoped_mechanism_graph,
    run_equal_tool_pre_llm_trial,
    run_m95_wing_scoped_calibration_update,
    update_flat_parameter_snapshot_from_public_item,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def _mat_pub(task: object) -> dict:
    with tempfile.TemporaryDirectory(prefix="m95_tpub_") as td:
        patched = task.public_setup_py.replace("/holdout_task", td)  # type: ignore[attr-defined]
        ns: dict = {}
        exec(compile(patched, "<pub>", "exec"), ns)
        return pickle.loads((Path(td) / f"public_{task.short_name}.pkl").read_bytes())[0]  # type: ignore[attr-defined]


def test_m95_disjoint_dev_vs_confirmation_seeds_and_unified_anonymous_schemas() -> None:
    dev_seeds = {s.world_seed for s in DEV_WORLD_SPECS}
    conf_seeds = {s.world_seed for s in CONFIRMATION_WORLD_SPECS}
    assert len(CONFIRMATION_WORLD_SPECS) == 8
    assert dev_seeds.isdisjoint(conf_seeds)
    assert max(dev_seeds) < 100_000 and min(conf_seeds) > 900_000

    for spec in DEV_WORLD_SPECS:
        src_t, stream_ts, manifest = build_world_tasks(spec)
        src_keys = expected_public_output_keys(src_t)
        for tk, m_row in zip(stream_ts, manifest):
            # Anonymous neutral task IDs: ZERO stage/event letters leaked in task_id or short_name!
            assert tk.short_name.startswith(f"{spec.world_id}_batch_")
            for leaked_tok in ("stage_", "counterexample", "recovery", "fanout"):
                assert leaked_tok not in tk.task_id.lower()
            if m_row["causal_event"] != "T":
                assert expected_public_output_keys(tk) == src_keys


def test_m95_dev_worlds_three_arm_causal_identification_end_to_end() -> None:
    """End-to-end 0-LLM verification across `dev_alpha_00` and `dev_beta_00` (`DEV_WORLD_SPECS` ONLY!)."""
    for spec in DEV_WORLD_SPECS:
        fam = spec.family
        src_task, stream_tasks, manifest = build_world_tasks(spec)
        tasks_by_ev = {m["causal_event"]: tk for tk, m in zip(stream_tasks, manifest)}

        # 1. Verify Stage S reference solver passes private verifier
        c0_ep = src_task.declared_callable_entrypoints[0]
        c0_code = build_reference_dev_solver_code(fam, c0_ep, "C0")
        assert grade_m94_artifact_map(src_task, {src_task.declared_artifacts[0]: c0_code})["strict_pass"] is True

        uni_keys = expected_public_output_keys(src_task)
        prior_entries = [
            {
                "short_name": src_task.short_name,
                "task_id": src_task.task_id,
                "entrypoint": c0_ep,
                "submitted_code": c0_code,
                "output_schema_keys": uni_keys,
            }
        ]
        flat_snap = (
            {"stress_gains": [1.0]*4, "stress_biases": [0.0]*4}
            if fam == "alpha"
            else {"bank_gains": [1.0]*4, "bank_offsets": [0.0]*4}
        )
        graph, planner, reg = initialize_m95_wing_scoped_mechanism_graph(family=fam, forest=PhilosophicalForest())
        comp_id = "alpha_facility_composite_calibration" if fam == "alpha" else "beta_facility_composite_calibration"
        c0_guard = (
            m7a.ObservablePreconditionGuard("g_a_c0", observable_extractor="overstress_excess_max_history_and_hold", op="<=", value=0.0, applies_to_component_id="linear_viscoelastic_memory")
            if fam == "alpha"
            else m7a.ObservablePreconditionGuard("g_b_c0", observable_extractor="nonseparable_2d_additive_residual_ratio", op="<=", value=1e-6, applies_to_component_id="separable_kronecker_transport")
        )
        reg.register_binding(
            MechanismExecutableBinding(
                binding_id=f"bind_{fam}_c0",
                chain_id=fam,
                branch_tag="C0_base",
                mechanism_id=f"mech_{fam}_c0",
                mechanism_version=1,
                forest_node_id=f"node_{fam}_c0",
                component_id=c0_guard.applies_to_component_id,
                preserved_component_ids=[],
                violated_component_id=None,
                module_stem="dev_c0",
                module_digest=hashlib.sha256(c0_code.encode()).hexdigest(),
                function_name=c0_ep,
                qualification_id="hegel_public_scientific_gate_v4",
                parameter_mechanism_id=comp_id,
                parameter_version=1,
                precondition_guard=c0_guard.to_dict(),
                output_schema_keys=uni_keys,
                parameterized_closure_code=c0_code,
                parameterized_entrypoint=c0_ep,
            )
        )

        # 2. Negative Control `G1` (full calibration sweep in batch): ALL 3 ARMS PASS IN 0 TURNS!
        t_g1 = tasks_by_ev["G1"]
        pub_g1 = _mat_pub(t_g1)
        flat_snap = update_flat_parameter_snapshot_from_public_item(family=fam, public_item=pub_g1, prior_snapshot=flat_snap)
        upd_g1 = run_m95_wing_scoped_calibration_update(family=fam, episode_id="g1", public_item=pub_g1, graph=graph, planner=planner, registry=reg)
        assert upd_g1["any_wing_reweighted"] is True and upd_g1["sentinel_max_abs_prediction_drift"] == 0.0

        for arm_nm in ("Archive_AutoReplay", "Flat_ParameterizedCache", "Full_Mechanism_Graph"):
            tr_g1 = run_equal_tool_pre_llm_trial(
                arm=arm_nm, family=fam, task=t_g1, public_item=pub_g1, prior_archive_entries=prior_entries,
                flat_param_snapshot=flat_snap, graph=graph, registry=reg,
            )
            assert tr_g1["accepted_pre_llm"] is True, f"{fam} G1 failed on {arm_nm}: {tr_g1}"
            assert grade_m94_artifact_map(t_g1, {t_g1.declared_artifacts[0]: tr_g1["candidate_code"]})["strict_pass"] is True

        # 3. Counterexample `C`: ALL 3 ARMS REJECT `C0` pre-LLM! Then register `C1` solver.
        t_c = tasks_by_ev["C"]
        pub_c = _mat_pub(t_c)
        for arm_nm in ("Archive_AutoReplay", "Flat_ParameterizedCache", "Full_Mechanism_Graph"):
            tr_c = run_equal_tool_pre_llm_trial(
                arm=arm_nm, family=fam, task=t_c, public_item=pub_c, prior_archive_entries=prior_entries,
                flat_param_snapshot=flat_snap, graph=graph, registry=reg,
            )
            assert tr_c["accepted_pre_llm"] is False

        c1_ep = t_c.declared_callable_entrypoints[0]
        c1_code = build_reference_dev_solver_code(fam, c1_ep, "C1")
        assert grade_m94_artifact_map(t_c, {t_c.declared_artifacts[0]: c1_code})["strict_pass"] is True
        prior_entries.append({"short_name": t_c.short_name, "task_id": t_c.task_id, "entrypoint": c1_ep, "submitted_code": c1_code, "output_schema_keys": uni_keys})
        c1_guard = (
            m7a.ObservablePreconditionGuard("g_a_c1", observable_extractor="overstress_excess_max_history_and_hold", op=">", value=0.0, applies_to_component_id="linear_viscoelastic_memory")
            if fam == "alpha"
            else m7a.ObservablePreconditionGuard("g_b_c1", observable_extractor="nonseparable_2d_additive_residual_ratio", op=">", value=1e-6, applies_to_component_id="separable_kronecker_transport")
        )
        reg.register_binding(
            MechanismExecutableBinding(
                binding_id=f"bind_{fam}_c1",
                chain_id=fam,
                branch_tag="C1_extension",
                mechanism_id=f"mech_{fam}_c1",
                mechanism_version=1,
                forest_node_id=f"node_{fam}_c1",
                component_id="extension_c1",
                preserved_component_ids=[],
                violated_component_id=c0_guard.applies_to_component_id,
                module_stem="dev_c1",
                module_digest=hashlib.sha256(c1_code.encode()).hexdigest(),
                function_name=c1_ep,
                qualification_id="hegel_public_scientific_gate_v4",
                parameter_mechanism_id=comp_id,
                parameter_version=int(graph.mechanisms[comp_id].version),
                precondition_guard=c1_guard.to_dict(),
                output_schema_keys=uni_keys,
                parameterized_closure_code=c1_code,
                parameterized_entrypoint=c1_ep,
            )
        )

        # 4. Recovery `R` (sparse 2-row check): `Archive_AutoReplay` FAILS; on Beta `Flat_ParameterizedCache` FAILS while `Full_Mechanism_Graph` PASSES!
        t_r = tasks_by_ev["R"]
        pub_r = _mat_pub(t_r)
        flat_snap = update_flat_parameter_snapshot_from_public_item(family=fam, public_item=pub_r, prior_snapshot=flat_snap)
        run_m95_wing_scoped_calibration_update(family=fam, episode_id="r", public_item=pub_r, graph=graph, planner=planner, registry=reg)
        tr_r_arch = run_equal_tool_pre_llm_trial(arm="Archive_AutoReplay", family=fam, task=t_r, public_item=pub_r, prior_archive_entries=prior_entries)
        tr_r_flat = run_equal_tool_pre_llm_trial(arm="Flat_ParameterizedCache", family=fam, task=t_r, public_item=pub_r, prior_archive_entries=prior_entries, flat_param_snapshot=flat_snap)
        tr_r_full = run_equal_tool_pre_llm_trial(arm="Full_Mechanism_Graph", family=fam, task=t_r, public_item=pub_r, prior_archive_entries=prior_entries, graph=graph, registry=reg)
        assert tr_r_arch["accepted_pre_llm"] is False
        if fam == "beta":
            assert tr_r_flat["accepted_pre_llm"] is False
        assert tr_r_full["accepted_pre_llm"] is True and tr_r_full["selected_branch_tag"] == "C0_base"
        assert grade_m94_artifact_map(t_r, {t_r.declared_artifacts[0]: tr_r_full["candidate_code"]})["strict_pass"] is True

        # 5. Second Counterexample `C_prime` (`E03` positive `C1` guard selection!): `Archive` FAILS; `Flat` & `Full` (`C1_extension`) PASS!
        t_cp = tasks_by_ev["C_prime"]
        pub_cp = _mat_pub(t_cp)
        flat_snap = update_flat_parameter_snapshot_from_public_item(family=fam, public_item=pub_cp, prior_snapshot=flat_snap)
        run_m95_wing_scoped_calibration_update(family=fam, episode_id="cp", public_item=pub_cp, graph=graph, planner=planner, registry=reg)
        tr_cp_arch = run_equal_tool_pre_llm_trial(arm="Archive_AutoReplay", family=fam, task=t_cp, public_item=pub_cp, prior_archive_entries=prior_entries)
        tr_cp_flat = run_equal_tool_pre_llm_trial(arm="Flat_ParameterizedCache", family=fam, task=t_cp, public_item=pub_cp, prior_archive_entries=prior_entries, flat_param_snapshot=flat_snap)
        tr_cp_full = run_equal_tool_pre_llm_trial(arm="Full_Mechanism_Graph", family=fam, task=t_cp, public_item=pub_cp, prior_archive_entries=prior_entries, graph=graph, registry=reg)
        assert tr_cp_arch["accepted_pre_llm"] is False
        assert tr_cp_flat["accepted_pre_llm"] is True
        assert tr_cp_full["accepted_pre_llm"] is True and tr_cp_full["selected_branch_tag"] == "C1_extension"
        assert grade_m94_artifact_map(t_cp, {t_cp.declared_artifacts[0]: tr_cp_full["candidate_code"]})["strict_pass"] is True

        # 6. Scoped Recalibration `G2` (`A06` Fanout=2 + Sibling Wing-B Zero Drift!) & Post-G2 Mixed Batch `M`!
        t_g2 = tasks_by_ev["G2"]
        pub_g2 = _mat_pub(t_g2)
        flat_snap_g2 = update_flat_parameter_snapshot_from_public_item(family=fam, public_item=pub_g2, prior_snapshot=flat_snap)
        upd_g2 = run_m95_wing_scoped_calibration_update(family=fam, episode_id="g2", public_item=pub_g2, graph=graph, planner=planner, registry=reg)
        assert upd_g2["wing_receipts"]["wing_a"]["selected_action"] == "AncestorReweight"
        assert upd_g2["wing_receipts"]["wing_b"]["selected_action"] == "NoChange"
        assert upd_g2["wing_receipts"]["wing_b"]["max_weight_drift"] == 0.0
        assert upd_g2["sentinel_max_abs_prediction_drift"] == 0.0
        assert upd_g2["fanout_count"] == 2  # Propagated simultaneously to BOTH `C0` and `C1`!

        tr_g2_arch = run_equal_tool_pre_llm_trial(arm="Archive_AutoReplay", family=fam, task=t_g2, public_item=pub_g2, prior_archive_entries=prior_entries)
        tr_g2_flat = run_equal_tool_pre_llm_trial(arm="Flat_ParameterizedCache", family=fam, task=t_g2, public_item=pub_g2, prior_archive_entries=prior_entries, flat_param_snapshot=flat_snap_g2)
        tr_g2_full = run_equal_tool_pre_llm_trial(arm="Full_Mechanism_Graph", family=fam, task=t_g2, public_item=pub_g2, prior_archive_entries=prior_entries, graph=graph, registry=reg)
        assert tr_g2_arch["accepted_pre_llm"] is False
        assert tr_g2_flat["accepted_pre_llm"] is False
        assert tr_g2_full["accepted_pre_llm"] is True and tr_g2_full["selected_branch_tag"] == "C0_base"
        assert grade_m94_artifact_map(t_g2, {t_g2.declared_artifacts[0]: tr_g2_full["candidate_code"]})["strict_pass"] is True

        t_m = tasks_by_ev["M"]
        pub_m = _mat_pub(t_m)
        flat_snap_m = update_flat_parameter_snapshot_from_public_item(family=fam, public_item=pub_m, prior_snapshot=flat_snap_g2)
        run_m95_wing_scoped_calibration_update(family=fam, episode_id="m", public_item=pub_m, graph=graph, planner=planner, registry=reg)
        tr_m_arch = run_equal_tool_pre_llm_trial(arm="Archive_AutoReplay", family=fam, task=t_m, public_item=pub_m, prior_archive_entries=prior_entries)
        tr_m_flat = run_equal_tool_pre_llm_trial(arm="Flat_ParameterizedCache", family=fam, task=t_m, public_item=pub_m, prior_archive_entries=prior_entries, flat_param_snapshot=flat_snap_m)
        tr_m_full = run_equal_tool_pre_llm_trial(arm="Full_Mechanism_Graph", family=fam, task=t_m, public_item=pub_m, prior_archive_entries=prior_entries, graph=graph, registry=reg)
        assert tr_m_arch["accepted_pre_llm"] is False
        assert tr_m_flat["accepted_pre_llm"] is False
        assert tr_m_full["accepted_pre_llm"] is True and tr_m_full["selected_branch_tag"] == "C1_extension"
        assert grade_m94_artifact_map(t_m, {t_m.declared_artifacts[0]: tr_m_full["candidate_code"]})["strict_pass"] is True
