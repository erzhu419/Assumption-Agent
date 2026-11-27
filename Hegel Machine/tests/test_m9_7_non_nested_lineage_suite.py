"""Pre-Freeze Deterministic (`0` LLM Calls) Verification Suite for `M9.7` (`DEV_M97_WORLD_SPECS` ONLY — Zero Confirmation Seed Touch!)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_6_autonomous_structure_continual_world as m96,
    m9_7_non_nested_counterfactual_lineage_world as m97,
)


def _materialize_pub(task: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="t_m97_") as td:
        ns: dict[str, Any] = {}
        exec(compile(task.public_setup_py.replace("/holdout_task", td), "<pub>", "exec"), ns)
        return pickle.loads((Path(td) / f"public_{task.short_name}.pkl").read_bytes())[0]


def test_m97_strict_present_matched_R0_vs_U_iso_zero_present_shortcut() -> None:
    """Prove `N06 / §7.4` mathematically across BOTH `dev_alpha_m97_00` and `dev_beta_m97_00`:
    `R0` (`C0`) and `U_iso` (`C1`) have `0.0` difference across ALL present inputs, thresholds, hold vectors,
    calibration tables, feature panels, and terminal snapshots `t = 0-`, so ANY current-only policy is 100% blind!
    """
    for spec in m97.DEV_M97_WORLD_SPECS:
        assert spec.split == "dev"
        _, stream_tasks, manifest = m97.build_m97_world_tasks(spec)
        by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}
        pub_r0 = _materialize_pub(by_ev["R0"])
        pub_u = _materialize_pub(by_ev["U_iso"])

        v2_pay = (
            {"stress_gains": list(spec.v2_gains), "stress_biases": list(spec.v2_offsets)}
            if spec.family == "alpha"
            else {"bank_gains": list(spec.v2_gains), "bank_offsets": list(spec.v2_offsets)}
        )
        obs_r0 = m97.compute_m97_primitive_observables(spec.family, pub_r0, v2_pay)
        obs_u = m97.compute_m97_primitive_observables(spec.family, pub_u, v2_pay)

        # ALL present/metadata/terminal/Frobenius/aux observables (`obs_02..obs_08`) MUST match to `0.0`!
        for k_pres in (
            "obs_02_present_snapshot_exceedance_max",
            "obs_03_boundary_or_banding_control_scalar",
            "obs_04_terminal_step_level",
            "obs_05_spatial_or_panel_anisotropy_ratio",
            "obs_06_aux_telemetry_rms",
            "obs_07_input_frobenius_scale",
            "obs_08_declared_threshold_metadata",
        ):
            assert obs_r0[k_pres] == obs_u[k_pres], f"{spec.world_id} {k_pres}: {obs_r0[k_pres]} != {obs_u[k_pres]}"

        # ONLY past trajectory exceedance (`obs_01`) separates `R0` (`0.0`) from `U_iso` (`> 0.0`)!
        assert obs_r0["obs_01_past_series_exceedance_integral"] == 0.0
        assert float(obs_u["obs_01_past_series_exceedance_integral"] or 0.0) > 0.5

        if spec.family == "alpha":
            ds_r0, ds_u = pub_r0["dataset"], pub_u["dataset"]
            assert ds_r0["crit_stress"] == ds_u["crit_stress"] == ds_r0["yield_stress"] == ds_u["yield_stress"]
            assert np.array_equal(ds_r0["test"]["stress_hold_cmd"], ds_u["test"]["stress_hold_cmd"])
            assert np.array_equal(ds_r0["train"]["stress_hold_cmd"], ds_u["train"]["stress_hold_cmd"])
            assert np.allclose(ds_r0["test"]["past_stress_cmd"][:, -1], ds_u["test"]["past_stress_cmd"][:, -1])
            assert not np.allclose(ds_r0["test"]["past_stress_cmd"][:, :-1], ds_u["test"]["past_stress_cmd"][:, :-1])
        else:
            st_r0, st_u = pub_r0["study"], pub_u["study"]
            assert st_r0["critical_vortex_circulation"] == st_u["critical_vortex_circulation"]
            assert st_r0["past_vortex_circulation_trace"][-1] == st_u["past_vortex_circulation_trace"][-1]
            assert not np.allclose(st_r0["past_vortex_circulation_trace"][:-1], st_u["past_vortex_circulation_trace"][:-1])


def test_m97_3x3_mutual_non_subsumption_and_full_lineage_restoration_in_containers() -> None:
    """Prove `§7.3 + N05 + N07` in isolated Docker containers across BOTH `dev_alpha_m97_00` and `dev_beta_m97_00`:
    1. `3x3` cross-regime matrix: `C0`, `C1`, `C2` each pass ONLY their own regime (`3/3` diagonal PASS, `6/6` off-diagonal FAIL)!
    2. Permuted anonymized parents (`cal_node_p0`, `cal_node_p1`) + genuinely wired `aux_telemetry` nuisance parent (`cal_node_p2` -> `0.0` diff)!
    3. Full `C0 -> C1 -> R0 -> C2 -> R1 -> U_iso -> G_perm -> M_couple` lineage restoration!
    """
    for spec in m97.DEV_M97_WORLD_SPECS:
        fam = spec.family
        s_task, stream_tasks, manifest = m97.build_m97_world_tasks(spec)
        by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}
        v2_pay = (
            {"stress_gains": list(spec.v2_gains), "stress_biases": list(spec.v2_offsets)}
            if fam == "alpha"
            else {"bank_gains": list(spec.v2_gains), "bank_offsets": list(spec.v2_offsets)}
        )

        solvers = {rk: m97.build_m97_reference_dev_solver_code(fam, f"solve_ref_{rk}", rk) for rk in ("C0", "C1", "C2")}
        regime_tasks = {"C0": by_ev["R0"], "C1": by_ev["U_iso"], "C2": by_ev["C2"]}

        # 1. `3x3` Mutual Non-Subsumption Matrix in clean Docker containers (`diagonal == True`, `off-diagonal == False`!)
        for solver_rk, s_code in solvers.items():
            for task_rk, tk in regime_tasks.items():
                adapted = m95.render_parameterized_input_adapter(
                    family=fam,
                    source_code=s_code,
                    source_entrypoint=f"solve_ref_{solver_rk}",
                    target_entrypoint=tk.declared_callable_entrypoints[0],
                    param_payload=v2_pay,
                )
                pr = m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk, candidate_code=adapted)
                if solver_rk == task_rk:
                    assert pr["accepted"] is True, f"{spec.world_id}: {solver_rk} failed own regime {task_rk}: {pr}"
                    ver = m94.grade_m94_artifact_map(tk, {tk.declared_artifacts[0]: adapted})
                    assert ver["strict_pass"] is True, f"{spec.world_id}: {solver_rk} failed private verifier on {task_rk}: {ver}"
                else:
                    assert pr["accepted"] is False, f"{spec.world_id}: {solver_rk} unexpectedly subsumed {task_rk}!"

        # 2. Containerized permuted-parent perturbation sensitivity rejects genuinely wired `cal_node_p2` (`aux_telemetry`) & `cal_node_p3`!
        pub_c1 = _materialize_pub(by_ev["C1"])
        sens = m96.verify_upstream_parent_perturbation_sensitivity(
            family=fam,
            candidate_upstream_ids=["cal_node_p0", "cal_node_p1", "cal_node_p2", "cal_node_p3"],
            proposed_upstream_ids=["cal_node_p0", "cal_node_p1"],
            c1_canonical_code=solvers["C1"],
            c1_entrypoint="solve_ref_C1",
            public_item=pub_c1,
            base_param_payload=v2_pay,
            intervention_fn_by_upstream_id=m97.build_m97_intervention_functions(spec),
            run_in_container=True,
        )
        assert sens["executed_in_isolated_container"] is True
        assert sens["proposal_matches_empirical_sensitivity"] is True
        assert sens["sensitivity_max_diff_by_upstream_id"]["cal_node_p2"] == 0.0
        assert sens["sensitivity_max_diff_by_upstream_id"]["cal_node_p3"] == 0.0

        # 3. Verify `G_perm` + `M_couple` selective permuted-parent `v3` update (`cal_node_p0 -> v3`, `cal_node_p1` rejects EMI -> stays `v2`, `cal_node_p2` updates `aux_gain` without invalidating `(p0, p1)` bindings!)
        forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
        nid = next(iter(forest.nodes))
        graph, planner, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)
        uni_keys = m97.expected_m97_public_output_keys(fam)
        c0_sha = hashlib.sha256(solvers["C0"].encode()).hexdigest()
        q0 = reg.issue_qualification_receipt(
            qualification_protocol="hegel_public_scientific_gate_v4",
            verified_code_sha256=c0_sha,
            forest_node_id=nid,
            environment_digest="sha256:dev",
            evidence_digest="sha256:ev0",
        )
        reg.register_binding(
            m94.MechanismExecutableBinding(
                binding_id=f"bind_{spec.world_id}_b00",
                chain_id=fam,
                branch_tag="C0_base",
                mechanism_id=f"mech_{spec.world_id}_b00",
                mechanism_version=1,
                forest_node_id=nid,
                component_id="base",
                preserved_component_ids=[],
                violated_component_id=None,
                module_stem="b00",
                module_digest=c0_sha,
                function_name="solve_ref_C0",
                qualification_id="hegel_public_scientific_gate_v4",
                qualification_receipt_id=q0.receipt_id,
                parameter_mechanism_id="cal_node_p0",
                parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
                parameter_version=1,
                environment_digest="sha256:dev",
                evidence_receipt_id="sha256:ev0",
                output_schema_keys=uni_keys,
                canonical_algorithm_code=solvers["C0"],
                canonical_entrypoint="solve_ref_C0",
            ),
            strict_receipt_check=True,
        )
        # First update at `C1` (`v1 -> v2` on both `p0` and `p1`)
        m97.update_m97_permuted_calibration_graph(spec=spec, episode_id="dev_c1", public_item=pub_c1, graph=graph, planner=planner, registry=reg)
        # Then update at `G_perm` (`p0 -> v3`, `p1` rejects EMI -> `NoChange`, `p2` shifts `aux_gain` to ~3.6!)
        pub_g = _materialize_pub(by_ev["G_perm"])
        g_rec = m97.update_m97_permuted_calibration_graph(spec=spec, episode_id="dev_g", public_item=pub_g, graph=graph, planner=planner, registry=reg)
        assert g_rec["parent_receipts"]["cal_node_p0"]["selected_action"] == "AncestorReweight"
        assert g_rec["parent_receipts"]["cal_node_p1"]["selected_action"] == "NoChange"
        assert g_rec["cal_node_p2_version"] == 2
        assert g_rec["sentinel_p3_drift"] == 0.0

        # Verify assembled `v3` payload passes `M_couple` private verifier with `C1`!
        v3_pay = m97.assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=["cal_node_p0", "cal_node_p1"])
        m_tk = by_ev["M_couple"]
        m_cand = m95.render_parameterized_input_adapter(
            family=fam,
            source_code=solvers["C1"],
            source_entrypoint="solve_ref_C1",
            target_entrypoint=m_tk.declared_callable_entrypoints[0],
            param_payload=v3_pay,
        )
        assert m94.grade_m94_artifact_map(m_tk, {m_tk.declared_artifacts[0]: m_cand})["strict_pass"] is True


def test_m97_mocked_end_to_end_world_stream_and_ledger_conservation(monkeypatch: Any) -> None:
    """Smoke-test `_run_one_m97_world_arm_stream` + `assert_hierarchical_ledger_conservation` across BOTH arms on `dev_alpha_m97_00` (`0` real LLM calls!)."""
    import importlib.util
    import sys

    runner_p = Path(__file__).resolve().parents[1] / "scripts/run_m9_7_non_nested_lineage_experiment.py"
    sp_mod = importlib.util.spec_from_file_location("_m97_runner_smoke", runner_p)
    assert sp_mod is not None and sp_mod.loader is not None
    runner = importlib.util.module_from_spec(sp_mod)
    sys.modules["_m97_runner_smoke"] = runner
    sp_mod.loader.exec_module(runner)

    spec = m97.DEV_M97_WORLD_SPECS[0]
    s_task, stream_tasks, manifest = m97.build_m97_world_tasks(spec)
    ev_by_sn = {m["short_name"]: m for m in manifest}
    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    src_nid = next(iter(forest.nodes))

    init_g, _, init_reg, wiring_meta = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)
    c0_code = m97.build_m97_reference_dev_solver_code("alpha", s_task.declared_callable_entrypoints[0], "C0")
    c0_sha = hashlib.sha256(c0_code.encode()).hexdigest()
    uni_keys = m97.expected_m97_public_output_keys("alpha")
    q_s = init_reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=c0_sha,
        forest_node_id=src_nid,
        environment_digest="sha256:smoke",
        evidence_digest="sha256:ev_s",
    )
    init_reg.register_binding(
        m94.MechanismExecutableBinding(
            binding_id=f"bind_{spec.world_id}_b00",
            chain_id="alpha",
            branch_tag="C0_base",
            mechanism_id=f"mech_{spec.world_id}_b00",
            mechanism_version=1,
            forest_node_id=src_nid,
            component_id="base",
            preserved_component_ids=[],
            violated_component_id=None,
            module_stem=s_task.short_name,
            module_digest=c0_sha,
            function_name=s_task.declared_callable_entrypoints[0],
            qualification_id="hegel_public_scientific_gate_v4",
            qualification_protocol="hegel_public_scientific_gate_v4",
            qualification_receipt_id=q_s.receipt_id,
            qualification_status="verified",
            parameter_mechanism_id="cal_node_p0",
            parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
            parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
            parameter_version=1,
            structure_origin="unconstrained_initial_source",
            environment_digest="sha256:smoke",
            evidence_receipt_id="sha256:ev_s",
            precondition_guard={},
            applicability_guard={},
            selection_priority=5,
            output_schema_keys=uni_keys,
            canonical_algorithm_code=c0_code,
            canonical_entrypoint=s_task.declared_callable_entrypoints[0],
            canonical_algorithm_sha256=c0_sha,
            parameterized_closure_code=c0_code,
            parameterized_entrypoint=s_task.declared_callable_entrypoints[0],
            parameterized_closure_sha256=c0_sha,
        ),
        strict_receipt_check=True,
    )

    pub_s, pub_s_sha = runner._materialize_public_item_and_sha(s_task)
    src_obs = m97.compute_m97_primitive_observables("alpha", pub_s, {"stress_gains": [1.0] * 4, "stress_biases": [0.0] * 4})
    src_ent = m95.make_continual_archive_entry(
        short_name=s_task.short_name,
        task_id=s_task.task_id,
        canonical_entrypoint=s_task.declared_callable_entrypoints[0],
        canonical_algorithm_code=c0_code,
        output_schema_keys=uni_keys,
        public_qualification_status="public_gate_verified",
    )
    src_cal = runner._summarize_public_calibration_sweep("alpha", pub_s, s_task.short_name)

    call_seq = {"n": 0}
    recorded_calls: list[dict[str, Any]] = []

    def _mock_run_frontier_episode(ft: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ep_dir: Path = kwargs["episode_out_dir"]
        sdir = ep_dir / "04_submitted_artifacts_snapshot"
        sdir.mkdir(parents=True, exist_ok=True)
        sn = ft.short_name
        rk = ev_by_sn[sn]["true_regime"]
        bname = Path(ft.declared_artifacts[0]).name
        code_txt = m97.build_m97_reference_dev_solver_code("alpha", ft.declared_callable_entrypoints[0], rk)
        (sdir / f"slot_00__{bname}").write_text(code_txt, encoding="utf-8")
        (ep_dir / "01_decision_provenance_dag.json").write_text(
            json.dumps({"provenance_verification": {"valid": True}, "records": []}), encoding="utf-8"
        )
        call_seq["n"] += 1
        recorded_calls.append(
            {
                "call_uuid": f"mock-solve-{call_seq['n']}",
                "arm": kwargs.get("run_id", "").split("__")[2],
                "world_id": spec.world_id,
                "episode_short_name": sn,
                "call_role": "stream_solve",
                "prompt_tokens": 2000,
                "cached_tokens": 500,
                "completion_tokens": 400,
                "wall_ms": 100.0,
            }
        )
        return {"task_id": ft.task_id, "short_name": sn, "turns_used": 2, "ended_by_finish": True, "skill_activations": []}

    def _mock_call_structured(self_cli: Any, **kw: Any) -> dict[str, Any]:
        call_seq["n"] += 1
        meta = getattr(self_cli, "_hegel_call_meta", {})
        recorded_calls.append(
            {
                "call_uuid": f"mock-ind-{call_seq['n']}",
                "arm": meta.get("arm", "Hegel_DemandGated_Graph"),
                "world_id": spec.world_id,
                "episode_short_name": meta.get("episode_short_name", ""),
                "call_role": "autonomous_guard_induction",
                "prompt_tokens": 1200,
                "cached_tokens": 0,
                "completion_tokens": 250,
                "wall_ms": 80.0,
            }
        )
        props = kw["output_schema"]["properties"]["existing_branch_guards"]["properties"]
        ex_ids = sorted(props.keys())
        if len(ex_ids) == 1:
            # First split (`C0` vs `C1`)
            return {
                "ok": True,
                "parsed": {
                    "mechanistic_diagnosis": "C1 past crystallization work separates from C0 melt.",
                    "violated_component_id": "linear_melt_flow",
                    "preserved_component_ids": ["prony_memory"],
                    "dependent_upstream_parent_ids": ["cal_node_p0", "cal_node_p1"],
                    "existing_branch_guards": {
                        ex_ids[0]: {"all_of": [{"feature_name": "obs_01_past_series_exceedance_integral", "op": "<=", "threshold": 0.05}]}
                    },
                    "new_branch_applicability_guard": {
                        "all_of": [{"feature_name": "obs_01_past_series_exceedance_integral", "op": ">", "threshold": 0.05}]
                    },
                },
            }
        # Second split (`C0`, `C1` vs `C2` shear banding)
        return {
            "ok": True,
            "parsed": {
                "mechanistic_diagnosis": "C2 shear banding separates from C0 and C1.",
                "violated_component_id": "homogeneous_shear_envelope",
                "preserved_component_ids": ["prony_memory"],
                "dependent_upstream_parent_ids": ["cal_node_p0", "cal_node_p1"],
                "existing_branch_guards": {
                    ex_ids[0]: {
                        "all_of": [
                            {"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50},
                            {"feature_name": "obs_01_past_series_exceedance_integral", "op": "<=", "threshold": 0.05},
                        ]
                    },
                    ex_ids[1]: {
                        "all_of": [
                            {"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50},
                            {"feature_name": "obs_01_past_series_exceedance_integral", "op": ">", "threshold": 0.05},
                        ]
                    },
                },
                "new_branch_applicability_guard": {
                    "all_of": [{"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": ">", "threshold": 0.50}]
                },
            },
        }

    monkeypatch.setattr(runner.m6f, "run_frontier_episode", _mock_run_frontier_episode)
    monkeypatch.setattr(runner.m6f.FrontierCodexClient, "call_structured", _mock_call_structured)

    with tempfile.TemporaryDirectory(prefix="m97_smoke_") as td:
        td_p = Path(td)
        ckpt_p = td_p / "init_ckpt.json"
        init_g.save_checkpoint(ckpt_p)
        mock_codex = {
            "available": True,
            "auth_mode": "chatgpt",
            "provider_backend": "chatgpt_codex_cli",
            "executable": "/bin/true",
            "is_windows_exe_under_wsl": False,
            "stage_root_wsl": str(td_p / "stage"),
            "stage_root_cli_str": str(td_p / "stage"),
        }
        res_h = runner._run_one_m97_world_arm_stream(
            spec=spec,
            arm="Hegel_DemandGated_Graph",
            s_task=s_task,
            stream_tasks=stream_tasks,
            schedule_manifest=manifest,
            base_v7_forest=forest,
            base_v7_archive=archive,
            prior_v7_l1_stems=set(),
            base_graph_ckpt_path=ckpt_p,
            parent_wiring_meta=wiring_meta,
            source_archive_entry=src_ent,
            source_cal_summary=src_cal,
            source_forest_nid=src_nid,
            source_pub_sha256=pub_s_sha,
            source_observables=src_obs,
            env_image_digest="sha256:smoke",
            model="gpt-6.2-sol",
            codex_info=mock_codex,
            stream_out_dir=td_p / "hegel",
            run_id="smoke_run",
            max_turns=6,
            reasoning_effort="medium",
        )
        res_f = runner._run_one_m97_world_arm_stream(
            spec=spec,
            arm="Flat_EqualInfo_Multiversion",
            s_task=s_task,
            stream_tasks=stream_tasks,
            schedule_manifest=manifest,
            base_v7_forest=forest,
            base_v7_archive=archive,
            prior_v7_l1_stems=set(),
            base_graph_ckpt_path=ckpt_p,
            parent_wiring_meta=wiring_meta,
            source_archive_entry=src_ent,
            source_cal_summary=src_cal,
            source_forest_nid=src_nid,
            source_pub_sha256=pub_s_sha,
            source_observables=src_obs,
            env_image_digest="sha256:smoke",
            model="gpt-6.2-sol",
            codex_info=mock_codex,
            stream_out_dir=td_p / "flat",
            run_id="smoke_run",
            max_turns=6,
            reasoning_effort="medium",
        )
        assert res_h["strict_passes"] == 7
        # Raw in-batch `build_m97_reference_dev_solver_code` passes `5/7` on `Flat` (`F14` skips unverified `step 6` and reuses `step 5` `0`-turn on `step 7`!),
        # whereas `Hegel_DemandGated_Graph` passes `7/7` and closes `5/7` (`R0, R1, U_iso, G_perm, M_couple`) `0`-turn!
        assert res_f["strict_passes"] == 5
        assert res_f["zero_turn_passed_closures"] == 1
        # `C1` and `C2` synthesize (`2` episodes), while `R0, R1, U_iso, G_perm, M_couple` (`5` episodes!) close `0`-turn!
        assert res_h["zero_turn_passed_closures"] == 5
        assert sorted(res_h["final_bindings"].keys()) == [
            f"bind_{spec.world_id}_b00",
            f"bind_{spec.world_id}_b01",
            f"bind_{spec.world_id}_b02",
        ]
        assert all(not g["should_compile_synchronously"] for g in res_h["consolidation_gate_decisions"])
        assert all(ep["provenance_dag_valid"] for ep in res_h["episodes_meta"])

        from hegel_machine.cognition.cost_policy import aggregate_call_usages, assert_hierarchical_ledger_conservation

        h_calls = [c for c in recorded_calls if c["arm"] == "Hegel_DemandGated_Graph"]
        ref_led = aggregate_call_usages(h_calls, deduplicate_uuid=True)
        ep_leds = [
            aggregate_call_usages([c for c in h_calls if c["episode_short_name"] == ep["short_name"]], deduplicate_uuid=True)
            for ep in res_h["episodes_meta"]
        ]
        cons = assert_hierarchical_ledger_conservation(
            reference_arm_ledger=ref_led,
            named_partition_ledgers={"episodes": ep_leds, "clusters": [ref_led]},
            context_label="smoke_hegel",
        )
        assert cons["conserved"] is True


def test_m97_beta_self_guard_exception_and_covering_branch_lineage_regression() -> None:
    """Reproduce and prove the exact `run2` Beta edge case (`0` LLM calls):
    1. A verified `Stage S` Beta solver (`b00`) that explicitly raises `ValueError('Only open Robin')` on `C1`
       is recorded as `failure_kind == 'scientific_or_numerical'` (while a broken `0`-witness solver is still `'schema_or_interface'`).
    2. When `b01` empirically covers `b00` (`EmpiricalCoverageReceipt == 'supported_on_witnesses'`), `mint_immutable_conflict_branch`
       succeeds AND subsequent `verify_m97_multibranch_witness_partition` on a specialized `b02` (`C2`) permits `(b01, b00)` overlap!
    """
    spec = m97.DEV_M97_WORLD_SPECS[1]
    s_task, stream_tasks, manifest = m97.build_m97_world_tasks(spec)
    by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, planner, reg, _ = m97.initialize_m97_permuted_parent_graph(spec=spec, forest=forest)

    # `b00` with explicit self-guard `ValueError` (exactly like `gpt-6.2-sol` wrote on `beta_m97_c01_s00`!)
    raw_c0 = m97.build_m97_reference_dev_solver_code("beta", "solve_s00", "C0")
    guarded_c0 = raw_c0.replace(
        "def solve_s00(study: dict) -> dict:\n",
        "def solve_s00(study: dict) -> dict:\n"
        "    if study['recirculation_loop_mode'] != 0 or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']:\n"
        "        raise ValueError('Only open-outflow Robin regime is supported')\n",
    )
    c0_sha = hashlib.sha256(guarded_c0.encode()).hexdigest()
    uni_keys = m97.expected_m97_public_output_keys("beta")
    q0 = reg.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=c0_sha,
        forest_node_id=nid,
        environment_digest="sha256:test",
        evidence_digest="sha256:ev0",
    )
    reg.register_binding(
        m94.MechanismExecutableBinding(
            binding_id=f"bind_{spec.world_id}_b00",
            chain_id="beta",
            branch_tag="C0_base",
            mechanism_id=f"mech_{spec.world_id}_b00",
            mechanism_version=1,
            forest_node_id=nid,
            component_id="base",
            preserved_component_ids=[],
            violated_component_id=None,
            module_stem="s00",
            module_digest=c0_sha,
            function_name="solve_s00",
            qualification_id="hegel_public_scientific_gate_v4",
            qualification_protocol="hegel_public_scientific_gate_v4",
            qualification_receipt_id=q0.receipt_id,
            qualification_status="verified",
            parameter_mechanism_id="cal_node_p0",
            parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
            parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
            parameter_version=1,
            structure_origin="unconstrained_initial_source",
            environment_digest="sha256:test",
            evidence_receipt_id="sha256:ev0",
            output_schema_keys=uni_keys,
            canonical_algorithm_code=guarded_c0,
            canonical_entrypoint="solve_s00",
            canonical_algorithm_sha256=c0_sha,
            parameterized_closure_code=guarded_c0,
            parameterized_entrypoint="solve_s00",
            parameterized_closure_sha256=c0_sha,
        ),
        strict_receipt_check=True,
    )
    lineage = m96.ImmutableBranchLineageController(world_id=spec.world_id, family="beta", graph=graph, registry=reg)
    b0 = reg.bindings[f"bind_{spec.world_id}_b00"]
    pub_s = _materialize_pub(s_task)
    obs_s = m97.compute_m97_primitive_observables("beta", pub_s, {"bank_gains": [1.0] * 4, "bank_offsets": [0.0] * 4})
    lineage.record_tested_witness(binding=b0, public_input_sha256="a" * 64, passed=True, failure_kind="none", features=obs_s)

    # Probe `b0` on `C1` (`Step 1`): `b0` raises `ValueError` -> recorded as `'scientific_or_numerical'`!
    pub_c1 = _materialize_pub(by_ev["C1"])
    m97.update_m97_permuted_calibration_graph(spec=spec, episode_id="c1", public_item=pub_c1, graph=graph, planner=planner, registry=reg)
    tr_c1 = m97.run_m97_equal_capability_pre_llm_trial(
        arm="Hegel_DemandGated_Graph",
        spec=spec,
        task=by_ev["C1"],
        public_item=pub_c1,
        public_input_sha256="b" * 64,
        prior_archive_entries=[],
        lineage_ctrl=lineage,
        graph=graph,
        registry=reg,
    )
    assert tr_c1["accepted_pre_llm"] is False
    assert tr_c1["tested_witness"]["failure_kind"] == "domain_precondition"
    assert len(lineage.scientific_negative_witnesses_for(b0.binding_id)) == 1

    # Verify covering branch `(b01, b00)` overlap check in `verify_m97_multibranch_witness_partition`
    part = m97.verify_m97_multibranch_witness_partition(
        branch_guards={
            "b00": {
                "all_of": [
                    {"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50},
                    {"feature_name": "obs_01_past_series_exceedance_integral", "op": "<=", "threshold": 0.01},
                ]
            },
            "b01": {"all_of": [{"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": "<=", "threshold": 0.50}]},
            "b02": {"all_of": [{"feature_name": "obs_03_boundary_or_banding_control_scalar", "op": ">", "threshold": 0.50}]},
        },
        positive_witness_features_by_branch={
            "b00": [{"obs_01_past_series_exceedance_integral": 0.0, "obs_03_boundary_or_banding_control_scalar": 0.0}],
            "b01": [
                {"obs_01_past_series_exceedance_integral": 0.85, "obs_03_boundary_or_banding_control_scalar": 0.0},
                {"obs_01_past_series_exceedance_integral": 0.0, "obs_03_boundary_or_banding_control_scalar": 0.0},
            ],
            "b02": [{"obs_01_past_series_exceedance_integral": 0.0, "obs_03_boundary_or_banding_control_scalar": 0.85}],
        },
        covered_branch_pairs={("b01", "b00")},
    )
    assert part["valid"] is True


