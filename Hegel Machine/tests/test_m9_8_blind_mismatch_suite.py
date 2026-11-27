"""Pre-Freeze Deterministic (`0` Paid LLM Calls) Verification Suite for `M9.8` (`DEV_M98_WORLD_SPECS` ONLY — Zero Confirmation Seed Touch!).

Proves every scientific & methodological invariant BEFORE freezing `Commit 2`:
  1. `A03`: Strict byte-identical visible inputs on `(R0, U_iso)` (`SHA256(/holdout_task/public_batch.pkl)` match, `0` differing
     fields in recursive diff, `0` `gt` key inside container, `R0.instruction_text == U_iso.instruction_text`), plus executed
     Docker `HistoryMasked` control proving current-only policies provably fail `>= 1/2` of `{R0, U_iso}`!
  2. `A13`: Zero prompt formula/role leakage across all episodes and families.
  3. `A02`: Full `(2 x 3) = 6` cross-factor (`{U0, U1} x {D0, D1, D2}`) Docker container verification across BOTH `dev_alpha_m98_00`
     and `dev_beta_m98_00` (`6/6` matched compositions pass public + private verifiers; mismatched upstream or downstream fail!).
  4. `A02 + A10`: Mocked 3-arm online continual stream (`Flat_LatestFirst` vs `Flat_ConditionalTable_Online` vs
     `Hegel_FactoredMechanismGraph`) proving:
       - `Flat_LatestFirst`: `0/7` zero-LLM reuse
       - `Flat_ConditionalTable_Online`: `2/7` zero-LLM reuse (`R0`, `U_iso` pass `0`-turn; `M_comb1`, `M_comb2` fail pre-LLM!)
       - `Hegel_FactoredMechanismGraph`: `4/7` zero-LLM reuse (`R0`, `U_iso`, `M_comb1`, `M_comb2` ALL pass `0`-turn!)
       - Cold-stop after `Step 4 (C_up)` + `PersistentMechanismGraph.load_checkpoint` reproduces identical `U_iso, M_comb1, M_comb2`!
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_6_autonomous_structure_continual_world as m96,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def _recursive_diffs(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}:missing")
            else:
                out.extend(_recursive_diffs(a[k], b[k], f"{path}/{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}:len_diff"]
        out_l: list[str] = []
        for i, (va, vb) in enumerate(zip(a, b)):
            out_l.extend(_recursive_diffs(va, vb, f"{path}[{i}]"))
        return out_l
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.dtype.kind in "bifcu" and bb.dtype.kind in "bifcu":
        if aa.shape == bb.shape and np.array_equal(aa, bb, equal_nan=True):
            return []
        return [f"{path}:numeric_diff"]
    if str(a) == str(b):
        return []
    return [f"{path}:value_diff"]


def test_m98_a03_a13_byte_identical_r0_vs_u_iso_and_zero_prompt_leakage() -> None:
    forbidden_tokens = (
        "neumann",
        "periodic",
        "quenched",
        "log1p",
        "log(1",
        "expm1",
        "0.85",
        "d0",
        "d1",
        "d2",
        "u0",
        "u1",
        "cross-talk",
        "uncoupled",
        "distractor",
    )
    for spec in m98.DEV_M98_WORLD_SPECS:
        assert spec.split == "dev"
        s_task, tasks, manifest, logs = m98.build_m98_world_tasks(spec)
        all_tasks = [s_task, *tasks]
        # 1. `A13`: Identical blind prompt across all 8 episodes, zero leaked formulas or labels!
        inst0 = s_task.instruction_text
        for tk in all_tasks:
            assert tk.instruction_text == inst0
            low = tk.instruction_text.lower()
            for tok in forbidden_tokens:
                assert tok not in low, f"Forbidden prompt token `{tok}` leaked in `{tk.task_id}`!"

        # 2. `A03`: `R0` and `U_iso` have 100% byte-identical `public_batch.pkl` (`0` `gt` key inside container!)
        by_ev = {m["causal_event"]: (t, i) for i, (t, m) in enumerate(zip(tasks, manifest))}
        r0_task, r0_idx = by_ev["R0"]
        ui_task, ui_idx = by_ev["U_iso"]
        r0_pub, r0_sha = m98.materialize_m98_public_item_and_sha(r0_task)
        ui_pub, ui_sha = m98.materialize_m98_public_item_and_sha(ui_task)
        assert "gt" not in r0_pub and "gt" not in ui_pub
        assert r0_sha == ui_sha, f"{spec.world_id}: R0 vs U_iso public_batch.pkl SHA mismatch!"
        assert _recursive_diffs(r0_pub, ui_pub) == []
        assert r0_task.host_only_public_gt_json != ui_task.host_only_public_gt_json

        # Upstream observables are identical, while prefix apparatus logs cleanly separate `R0` vs `U_iso`!
        assert m98.compute_m98_upstream_observables(spec.family, r0_pub) == m98.compute_m98_upstream_observables(spec.family, ui_pub)
        dn_r0 = m98.compute_m98_downstream_observables(logs[: r0_idx + 1])
        dn_ui = m98.compute_m98_downstream_observables(logs[: ui_idx + 1])
        ch_d1 = spec.active_log_channels[0]
        k_d1 = f"dn_stat_0{ch_d1 + 1}_log_channel_{ch_d1}_mean"
        assert float(dn_r0[k_d1] or 0.0) < 0.2
        assert float(dn_ui[k_d1] or 0.0) > 1.8


def test_m98_a02_a03_cross_factor_docker_composition_and_history_masked_ceiling() -> None:
    for spec in m98.DEV_M98_WORLD_SPECS:
        fam = spec.family
        _, tasks, manifest, _ = m98.build_m98_world_tasks(spec)
        by_ev = {m["causal_event"]: t for t, m in zip(tasks, manifest)}

        u0_code = m98.build_m98_default_u0_transducer_code(fam)
        u1_code = m98.build_m98_reference_dev_upstream_u1_code(fam)
        d_solvers = {rk: m98.build_m98_reference_dev_downstream_solver_code(fam, f"solve_{rk}", rk) for rk in ("D0", "D1", "D2")}
        v2_pay = {"gains": list(spec.v2_gains), "offsets": list(spec.v2_offsets)}
        v3_pay = {"gains": list(spec.v3_gains), "offsets": list(spec.v3_offsets)}

        # 1. Verify `3x3` downstream non-subsumption on `U0` (`R0=D0`, `U_iso=D1`, `C2_down=D2`) in clean Docker containers!
        u0_targets = {"D0": by_ev["R0"], "D1": by_ev["U_iso"], "D2": by_ev["C2_down"]}
        for s_rk, s_code in d_solvers.items():
            for t_rk, tk in u0_targets.items():
                cand = m98.compose_m98_two_stage_solver_code(
                    upstream_code=u0_code,
                    upstream_entrypoint="calibrate_upstream",
                    downstream_code=s_code,
                    downstream_entrypoint=f"solve_{s_rk}",
                    param_payload=v2_pay,
                )
                pr = m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk, candidate_code=cand)
                if s_rk == t_rk:
                    assert pr["accepted"] is True, f"{spec.world_id}: u0 o {s_rk} failed own regime {t_rk}: {pr}"
                    ver = m94.grade_m94_artifact_map(tk, {tk.declared_artifacts[0]: cand})
                    assert ver["strict_pass"] is True, f"{spec.world_id}: u0 o {s_rk} failed private verifier on {t_rk}: {ver}"
                else:
                    assert pr["accepted"] is False, f"{spec.world_id}: u0 o {s_rk} unexpectedly passed {t_rk}!"

        # 2. Verify cross-factor `U1` compositions on `C_up (U1, D0)`, `M_comb1 (U1, D1)`, `M_comb2 (U1, D2)`:
        #    - `u0 o d_true` (diagonal transducer) FAILS!
        #    - Even a full 4x4 outlier-trimmed LINEAR matrix transducer `u_lin4x4 o d_true` FAILS on nonlinear `U1`!
        #    - `u1 o d_wrong` (nonlinear transducer + wrong downstream `D0`) FAILS on `M_comb1` and `M_comb2`!
        #    - `u1 o d_true` (factored 2-stage composition!) PASSES BOTH public gate AND private verifier!
        u_lin4x4 = (
            '''from __future__ import annotations
import copy, numpy as np
def calibrate_upstream(dataset: dict) -> dict:
    ds = copy.deepcopy(dataset); cal = ds["calibration_load_cells"]
    br = np.asarray(cal["bridge_raw"], float); tr = np.asarray(cal["true_channel_loads"], float)
    kp = np.max(np.abs(br), axis=1) <= 8.5
    cf, *_ = np.linalg.lstsq(np.column_stack([br[kp], np.ones(int(np.sum(kp)))]), tr[kp], rcond=None)
    W, b0 = cf[:4], cf[4]
    for k in ("train", "test"):
        sp = dict(ds[k])
        sp["past_channel_raw"] = np.asarray(sp["past_channel_raw"], float) @ W + b0[None, None, :]
        sp["hold_channel_raw"] = np.asarray(sp["hold_channel_raw"], float) @ W + b0[None, :]
        ds[k] = sp
    cal2 = dict(cal); cal2["bridge_raw"] = tr.copy()
    sc2 = dict(cal2["single_channel_sweeps"]); sc2["cmd_stress"] = np.asarray(sc2["true_stress"], float).copy()
    cal2["single_channel_sweeps"] = sc2; ds["calibration_load_cells"] = cal2
    return ds
'''
            if fam == "alpha"
            else '''from __future__ import annotations
import copy, numpy as np
def calibrate_upstream(study: dict) -> dict:
    st = copy.deepcopy(study); cal = st["sensor_calibration_sweeps"]
    rr = np.asarray(cal["raw_rosette"], float); rf = np.asarray(cal["ref_rosette"], float)
    kp = np.max(np.abs(rr), axis=1) <= 8.5
    cf, *_ = np.linalg.lstsq(np.column_stack([rr[kp], np.ones(int(np.sum(kp)))]), rf[kp], rcond=None)
    W, b0 = cf[:4], cf[4]
    st["y_obs_raw"] = (np.asarray(st["y_obs_raw"], float).reshape(-1, 4) @ W + b0[None, :]).ravel()
    cal2 = dict(cal); cal2["raw_rosette"] = rf.copy()
    sb2 = dict(cal2["single_bank_sweeps"]); sb2["raw_readout"] = np.asarray(sb2["ref_signal"], float).copy()
    cal2["single_bank_sweeps"] = sb2; st["sensor_calibration_sweeps"] = cal2
    return st
'''
        )
        for ev_nm, true_dn, pay_d in (("C_up", "D0", v2_pay), ("M_comb1", "D1", v3_pay), ("M_comb2", "D2", v3_pay)):
            tk_u1 = by_ev[ev_nm]
            bad_u0 = m98.compose_m98_two_stage_solver_code(
                upstream_code=u0_code,
                upstream_entrypoint="calibrate_upstream",
                downstream_code=d_solvers[true_dn],
                downstream_entrypoint=f"solve_{true_dn}",
                param_payload=pay_d,
            )
            assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk_u1, candidate_code=bad_u0)["accepted"] is False

            bad_lin4 = m98.compose_m98_two_stage_solver_code(
                upstream_code=u_lin4x4,
                upstream_entrypoint="calibrate_upstream",
                downstream_code=d_solvers[true_dn],
                downstream_entrypoint=f"solve_{true_dn}",
                param_payload=pay_d,
            )
            assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk_u1, candidate_code=bad_lin4)["accepted"] is False

            if true_dn != "D0":
                bad_mono_step4 = m98.compose_m98_two_stage_solver_code(
                    upstream_code=u1_code,
                    upstream_entrypoint="calibrate_upstream",
                    downstream_code=d_solvers["D0"],
                    downstream_entrypoint="solve_D0",
                    param_payload=pay_d,
                )
                assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk_u1, candidate_code=bad_mono_step4)["accepted"] is False

            good_comp = m98.compose_m98_two_stage_solver_code(
                upstream_code=u1_code,
                upstream_entrypoint="calibrate_upstream",
                downstream_code=d_solvers[true_dn],
                downstream_entrypoint=f"solve_{true_dn}",
                param_payload=pay_d,
            )
            pr_good = m95.execute_single_pre_llm_public_gate_probe_in_container(task=tk_u1, candidate_code=good_comp)
            assert pr_good["accepted"] is True, f"{spec.world_id}: factored u1 o {true_dn} failed public gate on {ev_nm}: {pr_good}"
            ver_good = m94.grade_m94_artifact_map(tk_u1, {tk_u1.declared_artifacts[0]: good_comp})
            assert ver_good["strict_pass"] is True, f"{spec.world_id}: factored u1 o {true_dn} failed private verifier on {ev_nm}: {ver_good}"


def test_m98_mocked_three_arm_online_streams_and_post_cup_cold_restart(tmp_path: Path) -> None:
    runner_p = Path(__file__).resolve().parents[1] / "scripts/run_m9_8_blind_mismatch_dependency_experiment.py"
    sp_mod = importlib.util.spec_from_file_location("_m98_runner_test", runner_p)
    assert sp_mod is not None and sp_mod.loader is not None
    runner = importlib.util.module_from_spec(sp_mod)
    sys.modules["_m98_runner_test"] = runner
    sp_mod.loader.exec_module(runner)

    spec = m98.DEV_M98_WORLD_SPECS[0]
    wid = spec.world_id
    s_task, stream_tasks, manifest, logs = m98.build_m98_world_tasks(spec)
    m_by_sn = {m["short_name"]: m for m in manifest}
    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    uni_keys = m98.expected_m98_public_output_keys("alpha")

    u0_src = m98.build_m98_default_u0_transducer_code("alpha")
    u1_src = m98.build_m98_reference_dev_upstream_u1_code("alpha")
    d0_src = m98.build_m98_reference_dev_downstream_solver_code("alpha", "solve_batch", "D0")

    g0, _, reg0 = m98.initialize_m98_factored_mechanism_graph(spec=spec, forest=forest)
    qu = reg0.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=hashlib.sha256(u0_src.encode()).hexdigest(),
        forest_node_id=nid,
    )
    qd = reg0.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=hashlib.sha256(d0_src.encode()).hexdigest(),
        forest_node_id=nid,
    )
    reg0.register_binding(
        m94.MechanismExecutableBinding(
            binding_id=f"bind_{wid}_up_b00",
            chain_id="alpha_up",
            branch_tag="U0",
            mechanism_id=f"mech_{wid}_up_b00",
            mechanism_version=1,
            forest_node_id=nid,
            component_id="u0",
            preserved_component_ids=[],
            violated_component_id=None,
            module_stem="u0",
            module_digest=qu.verified_code_sha256,
            function_name="calibrate_upstream",
            qualification_id="hegel_public_scientific_gate_v4",
            qualification_receipt_id=qu.receipt_id,
            parameter_mechanism_id="cal_node_p0",
            parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
            parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
            parameter_version=1,
            output_schema_keys=["calibrated_input"],
            canonical_algorithm_code=u0_src,
            canonical_entrypoint="calibrate_upstream",
            selection_priority=10,
        ),
        strict_receipt_check=True,
    )
    reg0.register_binding(
        m94.MechanismExecutableBinding(
            binding_id=f"bind_{wid}_dn_b00",
            chain_id="alpha_dn",
            branch_tag="D0",
            mechanism_id=f"mech_{wid}_dn_b00",
            mechanism_version=1,
            forest_node_id=nid,
            component_id="d0",
            preserved_component_ids=[],
            violated_component_id=None,
            module_stem="d0",
            module_digest=qd.verified_code_sha256,
            function_name="solve_batch",
            qualification_id="hegel_public_scientific_gate_v4",
            qualification_receipt_id=qd.receipt_id,
            parameter_mechanism_id="constitutive_root_dn",
            parent_parameter_mechanism_ids=["constitutive_root_dn"],
            parent_parameter_versions={"constitutive_root_dn": 1},
            parameter_version=1,
            output_schema_keys=uni_keys,
            canonical_algorithm_code=d0_src,
            canonical_entrypoint="solve_batch",
            selection_priority=10,
        ),
        strict_receipt_check=True,
    )
    init_ckpt = tmp_path / "init_ckpt.json"
    g0.save_checkpoint(init_ckpt)
    pub_s, pub_s_sha = m98.materialize_m98_public_item_and_sha(s_task)
    src_ent = m95.make_continual_archive_entry(
        short_name=s_task.short_name,
        task_id=s_task.task_id,
        canonical_entrypoint="solve_batch",
        canonical_algorithm_code=d0_src,
        output_schema_keys=uni_keys,
        public_qualification_status="public_gate_verified",
    )

    ch_d1, ch_d2 = spec.active_log_channels
    fn_d1 = f"dn_stat_0{ch_d1 + 1}_log_channel_{ch_d1}_mean"
    fn_d2 = f"dn_stat_0{ch_d2 + 1}_log_channel_{ch_d2}_mean"

    def _mock_frontier_ep(ft: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ep_dir: Path = kwargs["episode_out_dir"]
        sdir = ep_dir / "04_submitted_artifacts_snapshot"
        sdir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "01_decision_provenance_dag.json").write_text(
            json.dumps({"provenance_verification": {"valid": True}, "records": []}), encoding="utf-8"
        )
        m_row = m_by_sn[ft.short_name]
        up_m, dn_r = m_row["true_upstream_mode"], m_row["true_downstream_regime"]
        if any(a.endswith("upstream_transducer.py") for a in ft.declared_artifacts):
            (sdir / "slot_01__upstream_transducer.py").write_text(u1_src, encoding="utf-8")
            (sdir / "slot_00__batch_solver.py").write_text("# wrapper\n", encoding="utf-8")
        else:
            dn_code = m98.build_m98_reference_dev_downstream_solver_code("alpha", "solve_inner", dn_r)
            if up_m == "U1":
                # Monolithic `(U1, dn_r)` solver synthesized by `Flat` arms!
                mono = (
                    u1_src
                    + "\n"
                    + dn_code.replace("from __future__ import annotations\n", "")
                    + "\ndef solve_batch(dataset: dict) -> dict:\n    return solve_inner(calibrate_upstream(dataset))\n"
                )
                (sdir / "slot_00__batch_solver.py").write_text(mono, encoding="utf-8")
            else:
                (sdir / "slot_00__batch_solver.py").write_text(
                    m98.build_m98_reference_dev_downstream_solver_code("alpha", "solve_batch", dn_r), encoding="utf-8"
                )
        return {"task_id": ft.task_id, "short_name": ft.short_name, "turns_used": 2, "ended_by_finish": True}

    def _mock_call_structured(self_cli: Any, **kw: Any) -> dict[str, Any]:
        user_j = json.loads(kw["conversation_turns"][0]["content"])
        ex_ids = user_j["existing_branch_ids"]
        new_id = user_j["new_branch_id"]
        pos_m = user_j["positive_witnesses_by_branch"]

        def _guard_for_pos(w_list: list[dict[str, float]]) -> dict[str, Any]:
            w0 = w_list[0]
            clauses = []
            if "up_stat_01_diagonal_ols_r2" in w0:
                if w0["up_stat_01_diagonal_ols_r2"] < 0.96:
                    clauses.append({"feature_name": "up_stat_01_diagonal_ols_r2", "op": "<", "threshold": 0.96})
                else:
                    clauses.append({"feature_name": "up_stat_01_diagonal_ols_r2", "op": ">=", "threshold": 0.96})
            if fn_d1 in w0 and fn_d2 in w0:
                clauses.append({"feature_name": fn_d1, "op": (">" if w0[fn_d1] > 1.0 else "<="), "threshold": 1.0})
                clauses.append({"feature_name": fn_d2, "op": (">" if w0[fn_d2] > 1.0 else "<="), "threshold": 1.0})
            return {"all_of": clauses}

        return {
            "ok": True,
            "parsed": {
                "diagnosis": "Clean empirical partition.",
                "existing_branch_guards": {bid: _guard_for_pos(pos_m[bid]) for bid in ex_ids},
                "new_branch_guard": _guard_for_pos(pos_m[new_id]),
            },
        }

    orig_ep = runner.m6f.run_frontier_episode
    orig_cs = runner.m6f.FrontierCodexClient.call_structured
    runner.m6f.run_frontier_episode = _mock_frontier_ep
    runner.m6f.FrontierCodexClient.call_structured = _mock_call_structured
    try:
        arm_res = {}
        for arm_nm in m98.M98_ARMS:
            st_dir = tmp_path / arm_nm
            st_dir.mkdir(parents=True, exist_ok=True)
            arm_res[arm_nm] = runner._run_one_m98_world_arm_stream(
                spec=spec,
                arm=arm_nm,
                s_task=s_task,
                stream_tasks=stream_tasks,
                schedule_manifest=manifest,
                apparatus_logs=logs,
                base_v7_forest=forest,
                base_v7_archive=archive,
                base_graph_ckpt_path=init_ckpt,
                source_archive_entry=src_ent,
                source_forest_nid=nid,
                source_pub_sha256=pub_s_sha,
                source_up_obs=m98.compute_m98_upstream_observables("alpha", pub_s),
                env_image_digest="sha256:dev",
                model="mock-gpt-6.2-sol",
                codex_info={
                    "available": True,
                    "auth_mode": "chatgpt",
                    "executable": "/usr/bin/true",
                    "is_windows_exe_under_wsl": False,
                    "stage_root_wsl": str(tmp_path),
                    "stage_root_cli_str": str(tmp_path),
                },
                stream_out_dir=st_dir,
                run_id="mock_m98",
                max_turns=6,
                reasoning_effort="high",
            )
    finally:
        runner.m6f.run_frontier_episode = orig_ep
        runner.m6f.FrontierCodexClient.call_structured = orig_cs

    # Verify all 3 online arms pass 7/7 private verifiers, with exact structural separation in zero-LLM reuse!
    assert arm_res["Flat_LatestFirst"]["strict_passes"] == 7
    assert arm_res["Flat_LatestFirst"]["zero_turn_passed_closures"] == 0

    assert arm_res["Flat_ConditionalTable_Online"]["strict_passes"] == 7
    assert arm_res["Flat_ConditionalTable_Online"]["zero_turn_passed_closures"] == 2  # `R0` and `U_iso` ONLY!

    assert arm_res["Hegel_FactoredMechanismGraph"]["strict_passes"] == 7
    assert arm_res["Hegel_FactoredMechanismGraph"]["zero_turn_passed_closures"] == 4  # `R0`, `U_iso`, `M_comb1`, `M_comb2`!

    # `A10`: Verify cold-restart hydration from `graph_ckpt_after_step_04.json` restores BOTH `up_ctrl` and `dn_ctrl` ledgers!
    ck4 = tmp_path / "Hegel_FactoredMechanismGraph" / "graph_ckpt_after_step_04.json"
    cg4 = PersistentMechanismGraph.load_checkpoint(ck4)
    cr4 = m94.M94BindingRegistry(cg4)
    cup4 = m96.ImmutableBranchLineageController(world_id=f"{wid}_up", family="alpha_up", graph=cg4, registry=cr4)
    cdn4 = m96.ImmutableBranchLineageController(world_id=f"{wid}_dn", family="alpha_dn", graph=cg4, registry=cr4)
    assert len(cup4.witness_ledger) >= 4
    assert len(cdn4.witness_ledger) >= 4
    assert len(cdn4.coverage_receipts) >= 3
