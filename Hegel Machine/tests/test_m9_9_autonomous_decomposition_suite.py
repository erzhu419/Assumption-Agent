"""Pre-Freeze Deterministic (`0` Paid LLM Calls) Verification Suite for `M9.9` (`DEV_M99_WORLD_SPECS` ONLY — Zero Confirmation Seed Touch!).

Proves every scientific, mathematical, & audit-closure invariant BEFORE freezing `M9.9` confirmation:
  1. `R02/R03/A13`: Zero confirmation seed touch, identical base blind prompts (`0` leaked shift/interaction formulas or labels),
     exact `dn_obs(E2_down) == dn_obs(E5_coupled_break)` and `dn_obs(E6_separable_revisit) == dn_obs(E7_coupled_revisit)`.
  2. `SeparabilityWithdrawalReceipt`: Content-addressed `swdr_` integrity check, healthy-upstream `rmse <= 0.03` requirement,
     scientific-failure requirement, and checkpoint cold-hydration foreign-key & missing-digest fail-closed verification.
  3. Containerized `4`-Regime Non-Subsumption & `(U1 x D2_coupled)` Falsification (`DEV_M99_WORLD_SPECS` in `hegel-clean-scipy:m7v1`):
     - Autonomous decomposition verifier (`verify_m99_autonomous_decomposition_in_containers`) accepts genuine `(u0, d0, d1)` and
       rejects trivial identity no-op `upstream_code` (`return item`)!
     - `u1 o d1` passes `E4_valid_comb (U1, D1)` (public + private)!
     - `u1 o d2_sep` FAILS `E5_coupled_break (U1, D2_coupled)` (`0` false zero-LLM accepts!), while `u1` passes its direct
       `56/24` calibration holdout probe (`rmse <= 0.03`)!
     - `u0 o d2_sep` passes `E6_separable_revisit` and FAILS `E7_coupled_revisit`; `u1 o d2_coupled` passes `E7_coupled_revisit`
       and `u0 o d2_coupled` FAILS `E6_separable_revisit` (`<= 50%` factorized two-table ceiling!).
  4. Mocked 3-Arm Equal-Tool Continual Stream + Post-`E5` Cold-Restart Hydration:
     - `Flat_MonolithicConditionalTable_Online`: `7/7` pass, `2/7` zero-LLM (`E6, E7`; misses `E4_valid_comb`).
     - `Flat_PlainModuleRegistry_Online`: `7/7` pass, `1/7` zero-LLM (`E4_valid_comb`; thrashes on `E6` and `E7` due to `dn_obs` collision!).
     - `Hegel_AutonomousInteractionGraph_Online`: `7/7` pass, **`3/7` zero-LLM (`E4_valid_comb`, `E6_separable_revisit`, `E7_coupled_revisit`)**
       + verified `SeparabilityWithdrawalReceipt` + identical post-`E5` cold-restart trajectory!
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
    m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def test_m99_dev_only_prompt_hygiene_and_identical_d2_precursor_observables() -> None:
    forbidden = (
        "neumann",
        "periodic",
        "quenched",
        "log1p",
        "sqrt(1",
        "d2_coupled",
        "d0",
        "d1",
        "d2",
        "u0",
        "u1",
    )
    for spec in m99.DEV_M99_WORLD_SPECS:
        assert spec.split == "dev"
        s_task, tasks, manifest, logs = m99.build_m99_world_tasks(spec)
        assert len(logs) == 8
        inst0 = s_task.instruction_text
        for tk in [s_task, *tasks]:
            assert tk.instruction_text == inst0
            low = tk.instruction_text.lower()
            for tok in forbidden:
                assert tok not in low, f"Forbidden token `{tok}` leaked in `{tk.task_id}`!"

        # Verify `dn_obs(E2_down) == dn_obs(E5_coupled_break)` and `dn_obs(E6) == dn_obs(E7)`
        dn_e2 = m98.compute_m98_downstream_observables(logs[: 2 + 1])
        dn_e5 = m98.compute_m98_downstream_observables(logs[: 5 + 1])
        dn_e6 = m98.compute_m98_downstream_observables(logs[: 6 + 1])
        dn_e7 = m98.compute_m98_downstream_observables(logs[: 7 + 1])
        assert dn_e2 == dn_e5
        assert dn_e6 == dn_e7
        assert m99.check_witness_feature_collision({"e2": [dn_e2], "e5": [dn_e5]})["has_collision"] is True
        assert m99.check_witness_feature_collision({"e6": [dn_e6], "e7": [dn_e7]})["has_collision"] is True


def test_m99_separability_withdrawal_receipt_tamper_and_cold_hydration_contracts() -> None:
    spec = m99.DEV_M99_WORLD_SPECS[0]
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    graph, _, reg = m98.initialize_m98_factored_mechanism_graph(spec=spec.to_m98_compat_spec(), forest=forest)

    for bid, cid, parents, keys in (
        ("up_b01", "alpha_up", ["cal_node_p0", "cal_node_p1"], ["calibrated_input"]),
        ("dn_b02", "alpha_dn", ["constitutive_root_dn"], m98.expected_m98_public_output_keys("alpha")),
    ):
        code = "def f(x):\n    return x\n"
        sha = hashlib.sha256(code.encode()).hexdigest()
        q = reg.issue_qualification_receipt(
            qualification_protocol="hegel_public_scientific_gate_v4", verified_code_sha256=sha, forest_node_id=nid
        )
        reg.register_binding(
            m94.MechanismExecutableBinding(
                binding_id=bid,
                chain_id=cid,
                branch_tag="t",
                mechanism_id=f"m_{bid}",
                mechanism_version=1,
                forest_node_id=nid,
                component_id="c",
                preserved_component_ids=[],
                violated_component_id=None,
                module_stem="s",
                module_digest=sha,
                function_name="f",
                qualification_id="hegel_public_scientific_gate_v4",
                qualification_receipt_id=q.receipt_id,
                parameter_mechanism_id=parents[0],
                parameter_version=1,
                parent_parameter_mechanism_ids=parents,
                parent_parameter_versions={p: 1 for p in parents},
                canonical_algorithm_code=code,
                canonical_entrypoint="f",
                output_schema_keys=keys,
            ),
            strict_receipt_check=True,
        )

    rec = m99.SeparabilityWithdrawalReceipt(
        receipt_id="",
        world_id=spec.world_id,
        family="alpha",
        upstream_binding_id="up_b01",
        upstream_code_sha256=reg.bindings["up_b01"].authoritative_executed_code_sha256(),
        upstream_component_probe_id="uprobe_1234",
        upstream_holdout_rmse=0.00012,
        downstream_binding_id="dn_b02",
        downstream_code_sha256=reg.bindings["dn_b02"].authoritative_executed_code_sha256(),
        composed_candidate_sha256="c" * 64,
        counterexample_public_input_sha256="a" * 64,
        counterexample_canonical_numeric_sha256="b" * 64,
        joint_failure_kind="scientific_or_numerical",
        verifier_version="hegel_shared_math_handoff_kernel_v3_m961_three_layer",
        expanded_feature_names=list(m99.M99_ALL_OBSERVABLE_NAMES),
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    rec.verify_integrity()

    # Rejects malformed ID or timeout_or_infra joint failure!
    bad = rec.to_dict()
    bad["receipt_id"] = "swdr_invalid"
    with pytest.raises(ValueError, match="Tampered or malformed SeparabilityWithdrawalReceipt"):
        m99.SeparabilityWithdrawalReceipt.from_dict(bad)

    bad_infra = rec.to_dict()
    bad_infra["joint_failure_kind"] = "timeout_or_infra"
    with pytest.raises(ValueError, match="genuine scientific/domain"):
        m99.SeparabilityWithdrawalReceipt.from_dict(bad_infra)

    # Persist & hydrate cleanly, then verify missing `state_sha256` fails closed!
    m99.persist_separability_withdrawals_to_graph(graph, spec.world_id, [rec])
    hyd = m99.hydrate_separability_withdrawals_from_graph(graph, reg, spec.world_id)
    assert len(hyd) == 1 and hyd[0].receipt_id == rec.receipt_id

    graph.forest.questions[m99.M99_WITHDRAWAL_FOREST_KEY]["worlds"][spec.world_id].pop("state_sha256")
    with pytest.raises(ValueError, match="Missing state_sha256"):
        m99.hydrate_separability_withdrawals_from_graph(graph, reg, spec.world_id)


def test_m99_docker_autonomous_decomposition_verifier_and_interaction_falsification() -> None:
    for spec in m99.DEV_M99_WORLD_SPECS:
        fam = spec.family
        s_task, tasks, manifest, logs = m99.build_m99_world_tasks(spec)
        by_ev = {m["causal_event"]: (t, i + 1) for i, (t, m) in enumerate(zip(tasks, manifest))}

        u0_code = m98.build_m98_default_u0_transducer_code(fam)
        u1_code = m98.build_m98_reference_dev_upstream_u1_code(fam)
        d0_code = m99.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D0")
        d1_code = m99.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D1")
        d2_sep_code = m99.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D2")
        d2_cpl_code = m99.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D2_coupled")

        s0_pub, _ = m98.materialize_m98_public_item_and_sha(s_task)
        e1_tk, e1_st = by_ev["E1_down"]
        e1_pub, _ = m98.materialize_m98_public_item_and_sha(e1_tk)
        e1_ro = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[: e1_st + 1], indent=2)}
        e1_sha = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in e1_ro.items()}
        v2_pay = {"gains": list(spec.v2_gains), "offsets": list(spec.v2_offsets)}
        v3_pay = {"gains": list(spec.v3_gains), "offsets": list(spec.v3_offsets)}

        # 1. Autonomous decomposition verifier rejects no-op identity upstream and accepts valid `(u0, d0, d1)`!
        noop_up = "def calibrate_upstream(item, param_payload=None):\n    return item\n"
        bad_dec = m99.verify_m99_autonomous_decomposition_in_containers(
            family=fam,
            s0_task=s_task,
            s0_public_item=s0_pub,
            s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
            e1_task=e1_tk,
            e1_public_item=e1_pub,
            e1_pay=v2_pay,
            e1_ro_deps=e1_ro,
            e1_ro_shas=e1_sha,
            upstream_code=noop_up,
            d0_code=d0_code,
            d1_code=d1_code,
        )
        assert bad_dec["valid"] is False

        ok_dec = m99.verify_m99_autonomous_decomposition_in_containers(
            family=fam,
            s0_task=s_task,
            s0_public_item=s0_pub,
            s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
            e1_task=e1_tk,
            e1_public_item=e1_pub,
            e1_pay=v2_pay,
            e1_ro_deps=e1_ro,
            e1_ro_shas=e1_sha,
            upstream_code=u0_code,
            d0_code=d0_code,
            d1_code=d1_code,
        )
        assert ok_dec["valid"] is True, f"{spec.world_id} valid decomposition rejected: {ok_dec}"

        # 2. `E4_valid_comb (U1, D1)`: `u1 o d1` passes public + private!
        e4_tk, _ = by_ev["E4_valid_comb"]
        c_e4 = m98.compose_m98_two_stage_solver_code(
            upstream_code=u1_code, upstream_entrypoint="calibrate_upstream", downstream_code=d1_code, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e4_tk, candidate_code=c_e4)["accepted"] is True
        assert m94.grade_m94_artifact_map(e4_tk, {e4_tk.declared_artifacts[0]: c_e4})["strict_pass"] is True

        # 3. `E5_coupled_break (U1, D2_coupled)`: naive `u1 o d2_sep` FAILS public gate (`0` false accepts!), while `u1` passes direct probe
        #    and `u1 o d2_coupled` PASSES public + private!
        e5_tk, _ = by_ev["E5_coupled_break"]
        e5_pub, _ = m98.materialize_m98_public_item_and_sha(e5_tk)
        up_chk = m98.diagnose_m98_factored_conflict_locus(fam, e5_pub, chosen_up_code=u1_code, force_direct_component_probe=True)
        assert up_chk["execution_status"] == "completed" and float(up_chk["chosen_up_holdout_rmse"]) <= 0.03

        c_naive_e5 = m98.compose_m98_two_stage_solver_code(
            upstream_code=u1_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_sep_code, downstream_entrypoint="solve_downstream", param_payload=v2_pay
        )
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e5_tk, candidate_code=c_naive_e5)["accepted"] is False

        c_good_e5 = m98.compose_m98_two_stage_solver_code(
            upstream_code=u1_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_cpl_code, downstream_entrypoint="solve_downstream", param_payload=v2_pay
        )
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e5_tk, candidate_code=c_good_e5)["accepted"] is True
        assert m94.grade_m94_artifact_map(e5_tk, {e5_tk.declared_artifacts[0]: c_good_e5})["strict_pass"] is True

        # 4. Disjointness across `E6_separable_revisit (U0, D2_sep)` and `E7_coupled_revisit (U1, D2_coupled)`!
        e6_tk, _ = by_ev["E6_separable_revisit"]
        e7_tk, _ = by_ev["E7_coupled_revisit"]
        c_e6_sep = m98.compose_m98_two_stage_solver_code(
            upstream_code=u0_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_sep_code, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        c_e6_cpl = m98.compose_m98_two_stage_solver_code(
            upstream_code=u0_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_cpl_code, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        c_e7_sep = m98.compose_m98_two_stage_solver_code(
            upstream_code=u1_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_sep_code, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        c_e7_cpl = m98.compose_m98_two_stage_solver_code(
            upstream_code=u1_code, upstream_entrypoint="calibrate_upstream", downstream_code=d2_cpl_code, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e6_tk, candidate_code=c_e6_sep)["accepted"] is True
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e6_tk, candidate_code=c_e6_cpl)["accepted"] is False
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e7_tk, candidate_code=c_e7_sep)["accepted"] is False
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=e7_tk, candidate_code=c_e7_cpl)["accepted"] is True
        assert m94.grade_m94_artifact_map(e6_tk, {e6_tk.declared_artifacts[0]: c_e6_sep})["strict_pass"] is True
        assert m94.grade_m94_artifact_map(e7_tk, {e7_tk.declared_artifacts[0]: c_e7_cpl})["strict_pass"] is True


def test_m99_mocked_three_arm_equal_tool_streams_and_post_e5_cold_restart(tmp_path: Path) -> None:
    runner_p = Path(__file__).resolve().parents[1] / "scripts/run_m9_9_autonomous_decomposition_experiment.py"
    sp_mod = importlib.util.spec_from_file_location("_m99_runner_test", runner_p)
    assert sp_mod is not None and sp_mod.loader is not None
    runner = importlib.util.module_from_spec(sp_mod)
    sys.modules["_m99_runner_test"] = runner
    sp_mod.loader.exec_module(runner)

    spec = m99.DEV_M99_WORLD_SPECS[0]
    wid = spec.world_id
    s_task, stream_tasks, manifest, logs = m99.build_m99_world_tasks(spec)
    m_by_sn = {m["short_name"]: m for m in manifest}
    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    uni_keys = m98.expected_m98_public_output_keys("alpha")

    u0_src = m98.build_m98_default_u0_transducer_code("alpha")
    u1_src = m98.build_m98_reference_dev_upstream_u1_code("alpha")
    d0_dn = m99.build_m99_reference_dev_downstream_solver_code("alpha", "solve_downstream", "D0")
    d1_dn = m99.build_m99_reference_dev_downstream_solver_code("alpha", "solve_downstream", "D1")

    def _mono_solver(up_m: str, eff_r: str) -> str:
        u_c = u1_src if up_m == "U1" else u0_src
        d_c = m99.build_m99_reference_dev_downstream_solver_code("alpha", "solve_inner", eff_r)
        return (
            u_c
            + "\n"
            + d_c.replace("from __future__ import annotations\n", "")
            + "\ndef solve_batch(dataset: dict) -> dict:\n    return solve_inner(calibrate_upstream(dataset))\n"
        )

    s0_mono = _mono_solver("U0", "D0")
    s0_pub, s0_sha = m98.materialize_m98_public_item_and_sha(s_task)
    s0_ent = m95.make_continual_archive_entry(
        short_name=s_task.short_name,
        task_id=s_task.task_id,
        canonical_entrypoint="solve_batch",
        canonical_algorithm_code=s0_mono,
        output_schema_keys=uni_keys,
        public_qualification_status="public_gate_verified",
    )

    ch_d1, ch_d2 = spec.active_log_channels
    fn_d1 = f"dn_stat_0{ch_d1 + 1}_log_channel_{ch_d1}_mean"
    fn_d2 = f"dn_stat_0{ch_d2 + 1}_log_channel_{ch_d2}_mean"

    def _mock_frontier_ep(ft: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Verify `R03` equal tool flags on EVERY episode call!
        assert kwargs.get("enable_auto_skill_packaging") is False
        assert kwargs.get("allow_l1_primitives") is False
        ep_dir: Path = kwargs["episode_out_dir"]
        sdir = ep_dir / "04_submitted_artifacts_snapshot"
        sdir.mkdir(parents=True, exist_ok=True)
        m_row = m_by_sn[ft.short_name]
        up_m, eff_r = m_row["true_upstream_mode"], m_row["true_effective_regime"]
        if any(a.endswith("upstream_transducer.py") for a in ft.declared_artifacts):
            (sdir / "slot_01__upstream_transducer.py").write_text(u1_src, encoding="utf-8")
            (sdir / "slot_00__batch_solver.py").write_text("# wrapper\n", encoding="utf-8")
        elif any(a.endswith("downstream_solver.py") for a in ft.declared_artifacts):
            (sdir / "slot_01__downstream_solver.py").write_text(
                m99.build_m99_reference_dev_downstream_solver_code("alpha", "solve_downstream", eff_r), encoding="utf-8"
            )
            (sdir / "slot_00__batch_solver.py").write_text("# wrapper\n", encoding="utf-8")
        else:
            (sdir / "slot_00__batch_solver.py").write_text(_mono_solver(up_m, eff_r), encoding="utf-8")
        return {"task_id": ft.task_id, "short_name": ft.short_name, "turns_used": 2, "ended_by_finish": True}

    def _mock_call_structured(self_cli: Any, **kw: Any) -> dict[str, Any]:
        req_props = set((kw.get("output_schema") or {}).get("properties", {}).keys())
        if "upstream_code" in req_props:
            return {
                "ok": True,
                "parsed": {
                    "diagnosis": "Factored shared sensor calibration from constitutive creep.",
                    "upstream_code": u0_src,
                    "downstream_d0_code": d0_dn,
                    "downstream_d1_code": d1_dn,
                },
            }
        user_j = json.loads(kw["conversation_turns"][0]["content"])
        ex_ids = user_j["existing_branch_ids"]
        new_id = user_j["new_branch_id"]
        pos_m = user_j["positive_witnesses_by_branch"]

        spreads = user_j.get("empirical_feature_spreads") or {}
        all_b_ids = list(ex_ids) + [new_id]

        def _dn_cell(w: dict[str, float]) -> tuple[bool, bool] | None:
            if fn_d1 in w and fn_d2 in w:
                return (w[fn_d1] > 1.0, w[fn_d2] > 1.0)
            return None

        def _guard_for_pos(bid_cur: str, w_list: list[dict[str, float]]) -> dict[str, Any]:
            w0 = w_list[0]
            clauses = []
            my_cell = _dn_cell(w0)
            collides_on_dn = (
                my_cell is None
                or any(
                    other_id != bid_cur and _dn_cell(pos_m[other_id][0]) == my_cell
                    for other_id in all_b_ids
                )
            )
            if (
                "up_stat_01_diagonal_ols_r2" in w0
                and spreads.get("up_stat_01_diagonal_ols_r2", 1.0) >= 0.07
                and collides_on_dn
                and len({w["up_stat_01_diagonal_ols_r2"] < 0.93 for w in w_list}) == 1
            ):
                clauses.append(
                    {
                        "feature_name": "up_stat_01_diagonal_ols_r2",
                        "op": ("<" if w0["up_stat_01_diagonal_ols_r2"] < 0.93 else ">="),
                        "threshold": 0.93,
                    }
                )
            if fn_d1 in w0 and spreads.get(fn_d1, 1.0) >= 0.07:
                clauses.append({"feature_name": fn_d1, "op": (">" if w0[fn_d1] > 1.0 else "<="), "threshold": 1.0})
            if fn_d2 in w0 and spreads.get(fn_d2, 1.0) >= 0.07:
                clauses.append({"feature_name": fn_d2, "op": (">" if w0[fn_d2] > 1.0 else "<="), "threshold": 1.0})
            return {"all_of": clauses}

        return {
            "ok": True,
            "parsed": {
                "diagnosis": "Empirical witness partition.",
                "existing_branch_guards": {bid: _guard_for_pos(bid, pos_m[bid]) for bid in ex_ids},
                "new_branch_guard": _guard_for_pos(new_id, pos_m[new_id]),
            },
        }

    fake_codex = {
        "available": True,
        "auth_mode": "chatgpt",
        "executable": "/bin/true",
        "is_windows_exe_under_wsl": False,
        "stage_root_wsl": str(tmp_path / "stage"),
        "stage_root_cli_str": str(tmp_path / "stage"),
    }

    orig_ep = runner.m6f.run_frontier_episode
    orig_cs = runner.m6f.FrontierCodexClient.call_structured
    runner.m6f.run_frontier_episode = _mock_frontier_ep
    runner.m6f.FrontierCodexClient.call_structured = _mock_call_structured
    try:
        arm_res = {}
        for arm_nm in m99.M99_ARMS:
            st_dir = tmp_path / "continual_streams" / wid / arm_nm
            st_dir.mkdir(parents=True, exist_ok=True)
            arm_res[arm_nm] = runner._run_one_m99_world_arm_stream(
                spec=spec,
                arm=arm_nm,
                s_task=s_task,
                stream_tasks=stream_tasks,
                schedule_manifest=manifest,
                apparatus_logs=logs,
                base_v7_forest=forest,
                base_v7_archive=archive,
                source_monolithic_code=s0_mono,
                source_archive_entry=s0_ent,
                source_forest_nid=nid,
                source_pub_item=s0_pub,
                source_pub_sha256=s0_sha,
                env_image_digest="test_img",
                model="gpt-6.2-sol",
                codex_info=fake_codex,
                stream_out_dir=st_dir,
                run_id="mock_m99",
                max_turns=6,
                reasoning_effort="medium",
            )
    finally:
        runner.m6f.run_frontier_episode = orig_ep
        runner.m6f.FrontierCodexClient.call_structured = orig_cs

    assert arm_res["Flat_MonolithicConditionalTable_Online"]["strict_passes"] == 7
    assert arm_res["Flat_PlainModuleRegistry_Online"]["strict_passes"] == 7
    assert arm_res["Hegel_AutonomousInteractionGraph_Online"]["strict_passes"] == 7

    # Exact 3-way zero-LLM structural separation!
    assert arm_res["Flat_MonolithicConditionalTable_Online"]["zero_turn_passed_closures"] == 2  # E6, E7 (misses E4)
    assert arm_res["Flat_PlainModuleRegistry_Online"]["zero_turn_passed_closures"] == 1  # E4 only (thrashes on E6 & E7!)
    assert arm_res["Hegel_AutonomousInteractionGraph_Online"]["zero_turn_passed_closures"] == 3  # E4, E6, AND E7!
    assert len(arm_res["Hegel_AutonomousInteractionGraph_Online"]["separability_withdrawal_receipts"]) == 1

    # Verify cold hydration after `Step 5 (E5_coupled_break)` via `run_m99_factorized_ceiling_ablation_on_sealed_streams`!
    abl = runner.run_m99_factorized_ceiling_ablation_on_sealed_streams(
        world_specs=(spec,),
        built_worlds={wid: (s_task, stream_tasks, manifest, logs)},
        out_dir=tmp_path,
    )
    assert abl["summary_by_policy"]["factorized_fixed_D2_separable"]["strict_passes"] == 1
    assert abl["summary_by_policy"]["factorized_fixed_D2_coupled"]["strict_passes"] == 1
    assert abl["summary_by_policy"]["hegel_selective_cross_factor_graph"]["strict_passes"] == 2


def test_m99_large_margin_guard_partition_rejects_ambient_noise_splits() -> None:
    for spec in m99.DEV_M99_WORLD_SPECS:
        _, _, _, logs = m99.build_m99_world_tasks(spec)
        s0_dn = m98.compute_m98_downstream_observables(logs[:1])
        e1_dn = m98.compute_m98_downstream_observables(logs[:2])
        e2_dn = m98.compute_m98_downstream_observables(logs[:3])
        e3_dn = m98.compute_m98_downstream_observables(logs[:4])
        ch_d1, ch_d2 = spec.active_log_channels
        fn_d1 = f"dn_stat_0{ch_d1 + 1}_log_channel_{ch_d1}_mean"
        fn_d2 = f"dn_stat_0{ch_d2 + 1}_log_channel_{ch_d2}_mean"
        amb_fn = next(
            f"dn_stat_0{k + 1}_log_channel_{k}_mean"
            for k in range(8)
            if k not in (ch_d1, ch_d2)
        )
        pos_map = {
            "d0": [{**s0_dn, amb_fn: -0.015}],
            "d1": [{**e1_dn, amb_fn: 0.010}],
            "d2": [{**e2_dn, amb_fn: 0.012}],
        }
        # Spurious 1-clause ambient split on `d0` passes raw boolean check yet MUST fail large-margin check!
        bad_guards = {
            "d0": {"all_of": [{"feature_name": amb_fn, "op": "<=", "threshold": 0.0}]},
            "d1": {"all_of": [{"feature_name": fn_d1, "op": ">", "threshold": 1.0}]},
            "d2": {"all_of": [{"feature_name": fn_d2, "op": ">", "threshold": 1.0}]},
        }
        chk_bad = m99.verify_m99_large_margin_witness_partition(branch_guards=bad_guards, positive_witness_features_by_branch=pos_map)
        assert not chk_bad["valid"]
        assert any("low_variance_feature_split" in r for r in chk_bad["reasons"])

        # True 2-clause large-margin `d0` guard MUST pass AND match `e3_dn` (`D0`)!
        good_guards = {
            "d0": {
                "all_of": [
                    {"feature_name": fn_d1, "op": "<=", "threshold": 1.0},
                    {"feature_name": fn_d2, "op": "<=", "threshold": 1.0},
                ]
            },
            "d1": {"all_of": [{"feature_name": fn_d1, "op": ">", "threshold": 1.0}]},
            "d2": {"all_of": [{"feature_name": fn_d2, "op": ">", "threshold": 1.0}]},
        }
        chk_good = m99.verify_m99_large_margin_witness_partition(branch_guards=good_guards, positive_witness_features_by_branch=pos_map)
        assert chk_good["valid"], chk_good
        assert m99.evaluate_m96_autonomous_guard(good_guards["d0"], e3_dn, allow_open_feature_keys=True)["state"] == "TRUE"

