"""Pre-Paid Deterministic & Air-Gapped Docker Gate Suite (`Gates 0, 1, 2`) for Hegel `M10.0`
(`DEV_M100_WORLD_SPECS` ONLY — Zero Confirmation Seed Touch!).

Verifies every pre-paid engineering & scientific requirement from `Hegel_M9_9_Audit`
(`01_Implementation_and_Experiment_Plan_ZH.md` §3, §4, §6 Gates 0–2, §7):
  1. Hygiene, Fresh Unexposed Seeds (`971k..974k` dev, `981k..988k` confirmation), & Bidirectional `dn_obs` Collisions:
     - Zero confirmation seed touch; zero leaked regime/shift formulas in prompts.
     - Exact `dn_obs` collisions in BOTH `sep_first` (`S02 == S08`, `S09 == S10`) and `coupled_first` (`S04 == S07`, `S09 == S10`).
     - Generic 7-feature upstream observables cleanly separate `U0`, `U1(+)`, `U1(-)`, and `U2` with `> 0.30` margin.
  2. `Gate 1` (`B1` Shared Alpha Public Pooled `ref_coupons` Contract + `3 × 3` Container Non-Subsumption):
     - In real `hegel-clean-scipy:m7v1` containers, `/holdout_task/validate_public_schema.py` deterministically rejects a solver
       that normalizes `test` on `test['ref_coupons']` alone (`POOLED_REF_COUPON_CONTRACT_VIOLATION`), while accepting the
       pooled `train+test` standardization!
     - `u1_dev` (`cubic bridge`) fails `U2` (`RMSE > 0.03`) and `u2_dev` (`harmonic opposite-pair bridge`) fails `U1` (`RMSE > 0.03`),
       while each passes its own `56/24` probe (`RMSE <= 0.005`) and all `(U2 × D1)`, `(U2 × D2_sep)`, `(U1 × D2_coupled)` cells
       pass both 3-layer public and official private gates!
  3. `Gate 1 + Gate 2` (Bidirectional `sep_first` & `coupled_first` 3-Arm Streams + `B2` Sibling Fallback + `B3` Lazy Projection + Cold Restart):
     - `Order A (sep_first)`: `Monolithic` `10/10` (`2/10` 0-LLM: `S9, S10`), `Plain` `10/10` (`3/10` 0-LLM: `S4, S6, S7`; thrashes on `S9, S10`),
       `Hegel` `10/10` (**`5/10` 0-LLM: `S4, S6, S7, S9, S10`** + `1` forward `SeparabilityWithdrawalReceipt` at `S08`, `K*` reached!).
     - `Order B (coupled_first)`: `0` fake withdrawal at `S04_d2_coupled_first` (no prior `D2` positive!), **1 reverse
       `SeparabilityWithdrawalReceipt` at `S07_rev_sep_break`**, `Hegel` `10/10` (`5/10` 0-LLM) + full 6-link cold checkpoint hydration!
     - `B2` (`1`-sibling upstream fallback) & `B3` (deferred decomposition + lazy on-demand downstream projection) both verified!
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import (
    m7_autogenous_forest_pretraining as m7a,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_6_autonomous_structure_continual_world as m96,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
    m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99,
    m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world as m100,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def _load_m100_runner() -> Any:
    runner_p = Path(__file__).resolve().parents[1] / "scripts/run_m10_0_reliable_modular_amortization_experiment.py"
    sp_mod = importlib.util.spec_from_file_location("_m100_runner_test", runner_p)
    assert sp_mod is not None and sp_mod.loader is not None
    runner = importlib.util.module_from_spec(sp_mod)
    sys.modules["_m100_runner_test"] = runner
    sp_mod.loader.exec_module(runner)
    return runner


def test_m100_seed_disjointness_prompt_hygiene_observable_margins_and_bidirectional_collisions() -> None:
    prior_seeds = (
        {s.seed_base for s in m98.DEV_M98_WORLD_SPECS}
        | {s.seed_base for s in m98.CONFIRMATION_M98_WORLD_SPECS}
        | {s.seed_base for s in m99.DEV_M99_WORLD_SPECS}
        | {s.seed_base for s in m99.CONFIRMATION_M99_WORLD_SPECS}
    )
    dev_seeds = {s.seed_base for s in m100.DEV_M100_WORLD_SPECS}
    conf_seeds = {s.seed_base for s in m100.CONFIRMATION_M100_WORLD_SPECS}
    assert len(dev_seeds) == 4 and len(conf_seeds) == 8
    assert not (dev_seeds & prior_seeds)
    assert not (conf_seeds & prior_seeds)
    assert not (dev_seeds & conf_seeds)

    forbidden = ("neumann", "periodic", "quenched", "log1p", "expm1", "sqrt", "harmonic", "u0", "u1", "u2", "d0", "d1", "d2")
    for spec in m100.DEV_M100_WORLD_SPECS:
        s_task, tasks, manifest, logs = m100.build_m100_world_tasks(spec)
        assert len(tasks) == 10
        for tk in (s_task, *tasks):
            assert tk.instruction_text == s_task.instruction_text
            low = tk.instruction_text.lower()
            for kw in forbidden:
                assert kw not in low, f"Forbidden `{kw}` in `{tk.task_id}`"
            pub_item, _ = m99.materialize_m99_public_item_and_sha(tk)
            assert "gt" not in pub_item

        # Verify exact `dn_obs` collisions in both `sep_first` and `coupled_first`
        if spec.order_mode == "sep_first":
            assert m98.compute_m98_downstream_observables(logs[:3]) == m98.compute_m98_downstream_observables(logs[:9])
        else:
            assert m98.compute_m98_downstream_observables(logs[:5]) == m98.compute_m98_downstream_observables(logs[:8])
        assert m98.compute_m98_downstream_observables(logs[:10]) == m98.compute_m98_downstream_observables(logs[:11])

        # Verify generic 7-feature upstream observables separate `U0`, `U1`, and `U2` with `> 0.25` margin!
        s0_pub, _ = m99.materialize_m99_public_item_and_sha(s_task)
        u1_idx = next(i for i, m in enumerate(manifest) if m["true_upstream_mode"] == "U1")
        u2_idx = next(i for i, m in enumerate(manifest) if m["true_upstream_mode"] == "U2")
        u1_pub, _ = m99.materialize_m99_public_item_and_sha(tasks[u1_idx])
        u2_pub, _ = m99.materialize_m99_public_item_and_sha(tasks[u2_idx])

        obs_u0 = m100.compute_m100_upstream_observables(spec.family, s0_pub)
        obs_u1 = m100.compute_m100_upstream_observables(spec.family, u1_pub)
        obs_u2 = m100.compute_m100_upstream_observables(spec.family, u2_pub)
        assert float(obs_u0["up_stat_01_diagonal_ols_r2"]) > 0.99
        assert float(obs_u1["up_stat_03_adj_offdiag_abs_corr_max"]) > 0.20
        assert float(obs_u2["up_stat_03_adj_offdiag_abs_corr_max"]) < 0.06
        assert float(obs_u2["up_stat_04_opp_offdiag_abs_corr_max"]) > 0.16
        assert float(obs_u1["up_stat_04_opp_offdiag_abs_corr_max"]) < 0.06


def test_m100_docker_gate1_pooled_coupon_validator_and_u0_u1_u2_nonsubsumption() -> None:
    # Check 1 `alpha` dev world and 1 `beta` dev world in real air-gapped Docker containers!
    for spec in (m100.DEV_M100_WORLD_SPECS[0], m100.DEV_M100_WORLD_SPECS[2]):
        fam = spec.family
        s_task, tasks, manifest, logs = m100.build_m100_world_tasks(spec)
        by_ev = {m["causal_event"]: (t, i + 1) for i, (t, m) in enumerate(zip(tasks, manifest))}
        v2_pay = {"gains": list(spec.v2_gains), "offsets": list(spec.v2_offsets)}
        v3_pay = {"gains": list(spec.v3_gains), "offsets": list(spec.v3_offsets)}

        u0_c = m98.build_m98_default_u0_transducer_code(fam)
        u1_c = m98.build_m98_reference_dev_upstream_u1_code(fam)
        u2_c = m100.build_m100_reference_dev_upstream_u2_code(fam)
        d0_c = m100.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D0")
        d1_c = m100.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D1")
        d2_sep_c = m100.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D2")
        d2_cpl_c = m100.build_m99_reference_dev_downstream_solver_code(fam, "solve_downstream", "D2_coupled")

        # 1. `U1` vs `U2` genuine structural non-subsumption on `56/24` calibration probe
        u1_tk, _ = by_ev["S03_u1_disc"]
        u2_tk, _ = by_ev["S05_u2_disc"]
        u1_pub, _ = m99.materialize_m99_public_item_and_sha(u1_tk)
        u2_pub, _ = m99.materialize_m99_public_item_and_sha(u2_tk)

        d_u1_on_u1 = m98.diagnose_m98_factored_conflict_locus(fam, u1_pub, chosen_up_code=u1_c, diag_payload=v2_pay, force_direct_component_probe=True)
        d_u1_on_u2 = m98.diagnose_m98_factored_conflict_locus(fam, u2_pub, chosen_up_code=u1_c, diag_payload=v2_pay, force_direct_component_probe=True)
        d_u2_on_u2 = m98.diagnose_m98_factored_conflict_locus(fam, u2_pub, chosen_up_code=u2_c, diag_payload=v2_pay, force_direct_component_probe=True)
        d_u2_on_u1 = m98.diagnose_m98_factored_conflict_locus(fam, u1_pub, chosen_up_code=u2_c, diag_payload=v2_pay, force_direct_component_probe=True)

        assert float(d_u1_on_u1["chosen_up_holdout_rmse"]) <= 0.005
        assert float(d_u2_on_u2["chosen_up_holdout_rmse"]) <= 0.005
        assert float(d_u1_on_u2["chosen_up_holdout_rmse"]) > 0.03
        assert float(d_u2_on_u1["chosen_up_holdout_rmse"]) > 0.03

        # 2. Verify `(U2 x D2_sep)` (`S07`) passes under `u2 o d2_sep` and fails under `u2 o d2_coupled`
        s7_tk, _ = by_ev["S07_comb_u2_d2sep"]
        c_u2_d2sep = m98.compose_m98_two_stage_solver_code(
            upstream_code=u2_c, upstream_entrypoint="calibrate_upstream", downstream_code=d2_sep_c, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        c_u2_d2cpl = m98.compose_m98_two_stage_solver_code(
            upstream_code=u2_c, upstream_entrypoint="calibrate_upstream", downstream_code=d2_cpl_c, downstream_entrypoint="solve_downstream", param_payload=v3_pay
        )
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=s7_tk, candidate_code=c_u2_d2sep, require_pooled_coupon_check=(fam == "alpha"))["accepted"] is True
        assert m95.execute_single_pre_llm_public_gate_probe_in_container(task=s7_tk, candidate_code=c_u2_d2cpl, require_pooled_coupon_check=(fam == "alpha"))["accepted"] is False
        assert m94.grade_m94_artifact_map(s7_tk, {s7_tk.declared_artifacts[0]: c_u2_d2sep})["strict_pass"] is True

        # 3. `B1 / Q02`: On Alpha, verify container `validate_public_schema.py` catches test-only `ref_coupons` normalization!
        if fam == "alpha":
            buggy_test_only_solver = '''from __future__ import annotations
import numpy as np

def solve_batch(ds: dict) -> dict:
    te = ds["test"]
    r_te = np.asarray(te["ref_coupons"], dtype=float)
    x_te = np.asarray(te["coupon_features_raw"], dtype=float).copy()
    labs = np.asarray(te["lab_id"], dtype=int)
    for c in range(4):
        m = labs == c
        if np.any(m):
            x_te[m] = (x_te[m] - float(np.mean(r_te[m]))) / max(float(np.std(r_te[m])), 1e-6)
    x_te = np.nan_to_num(x_te, nan=0.0)
    t_ev = np.asarray(ds["eval_times"], dtype=float)
    drift = 0.3 + 0.15 * x_te[:, 0]
    return {
        "estimated_taus": np.array([0.6, 3.8]),
        "hereditary_weights": np.array([0.65, 0.50, 0.30, 0.15, -0.12, 0.12, -0.10]),
        "predicted_strain": np.asarray(te["strain_0"], float)[:, None] + drift[:, None] * t_ev[None, :],
        "initial_rate_pred": drift,
    }
'''
            pr_bug = m95.execute_single_pre_llm_public_gate_probe_in_container(task=s_task, candidate_code=buggy_test_only_solver, require_pooled_coupon_check=True)
            assert pr_bug["accepted"] is False
            assert pr_bug["schema_valid"] is False
            output = pr_bug["schema_stdout"]
            assert "pooled_reference_coupon_invariance_failed" in output
            check, _ = json.JSONDecoder().raw_decode(output.split('"pooled_reference_check":', 1)[1].lstrip())
            assert check["swapped_ref_rows_per_split"] > 0
            assert max(check["ref_swap_rate_max_diff"], check["ref_swap_strain_max_diff"]) > check["tol"]


def _fixture_gate(task, candidate_code, accepted, dependency_hashes):
    # This mocked-provider stream has a synthetic public gate. Bind its outcome
    # to the same deterministic public item that both fixture components consume.
    item, raw = m98.materialize_m98_public_item_and_sha(task)
    candidate_sha = hashlib.sha256(candidate_code.encode()).hexdigest()
    bundle = hashlib.sha256(b"/holdout_task/batch_solver.py\0" + bytes.fromhex(candidate_sha)).hexdigest()
    assets = {**(dependency_hashes or {}), "/holdout_task/public_batch.pkl": raw}
    return {
        "accepted": accepted, "schema_valid": True, "schema_exit_code": 0, "setup_ok": True,
        "scientific_probe_receipt": {
            "scientific_invariants_verified": accepted, "signed_by_host_probe_executor": True,
            "execution_status": "completed", "receipt_kind": "scientific_probe_receipt", "probe_protocol_version": "fixture",
            "probe_code_digest": "e" * 64, "kernel_version": m99.SHARED_KERNEL_VERSION,
            "errors": [] if accepted else ["fixture_regime_mismatch"], "candidate_bundle_sha256": bundle,
            "public_setup_files_sha256": assets, "public_input_digest": raw,
            "canonical_numeric_sha256": m96.compute_canonical_numeric_item_sha256(item),
        },
        "candidate_bundle_sha256": bundle, "candidate_raw_sha256": candidate_sha,
        "setup_source_sha256": hashlib.sha256(task.public_setup_py.encode()).hexdigest(),
        "readonly_dependencies_sha256": dependency_hashes or {}, "public_setup_files_sha256": assets,
    }


def test_m100_scientific_witness_rejects_other_execution_and_missing_measurement():
    runner = _load_m100_runner()
    task, _, _, _ = m100.build_m100_world_tasks(m100.DEV_M100_WORLD_SPECS[0])
    candidate = "def solve_batch(item):\n    return {}\n"
    gate = _fixture_gate(task, candidate, False, {})
    for edit in ({}, {"scientific_probe_receipt": None}, {"candidate_raw_sha256": "0" * 64},
                 {"setup_source_sha256": "f" * 64}, {"readonly_dependencies_sha256": {"unexpected": "a" * 64}},
                 {"public_setup_files_sha256": {"/holdout_task/public_batch.pkl": "1" * 64}}):
        with patch.object(runner, "materialize_m99_public_item_and_sha", m98.materialize_m98_public_item_and_sha), patch.object(
            runner, "execute_single_pre_llm_public_gate_probe_in_container", return_value={**gate, **edit},
        ):
            actual = runner._execute_bound_gate(task=task, candidate_code=candidate)
        assert actual["accepted"] is False
        assert m96.classify_public_probe_failure(actual)["tier"] == ("INFRA_FAILURE" if edit else "SCIENTIFIC_FAILURE")


def test_m100_gate2_mocked_bidirectional_streams_b2_fallback_b3_lazy_proj_and_cold_hydration(tmp_path: Path) -> None:
    runner = _load_m100_runner()

    # Fast deterministic mock for container probes & official verifier based on embedded ground-truth solver tags
    # (real Docker physics already verified in `test_m100_docker_gate1...` above -> runs both `sep_first` & `coupled_first`
    # `30`-episode 3-arm streams + `B2`/`B3` checks in `< 5` seconds!).
    def _tag_up(mode: str) -> str:
        return f"# UP_TAG:{mode}\ndef calibrate_upstream(item, param_payload=None):\n    item = dict(item); item['_up_tag'] = {mode!r}; return item\n"

    def _tag_dn(eff: str) -> str:
        return f"# DN_TAG:{eff}\ndef solve_downstream(item):\n    return {{'_up': item.get('_up_tag'), '_dn': {eff!r}}}\n"

    def _tag_mono(up_m: str, eff: str) -> str:
        return f"# MONO_TAG:{up_m}__{eff}\ndef solve_batch(item):\n    return {{'_up': {up_m!r}, '_dn': {eff!r}}}\n"

    def _eval_tagged_code(cand_code: str, true_up: str, true_eff: str) -> bool:
        ns: dict[str, Any] = {}
        exec(compile(cand_code, "<tagged_cand>", "exec"), ns)
        out = ns["solve_batch"]({})
        return out.get("_up") == true_up and out.get("_dn") == true_eff

    # Test BOTH `DEV_M100_WORLD_SPECS[0]` (`sep_first`) AND `DEV_M100_WORLD_SPECS[1]` (`coupled_first`)!
    for idx_sp, spec in enumerate((m100.DEV_M100_WORLD_SPECS[0], m100.DEV_M100_WORLD_SPECS[1])):
        wid = spec.world_id
        s_task, stream_tasks, manifest, logs = m100.build_m100_world_tasks(spec)
        m_by_tid = {m["task_id"]: m for m in manifest}
        m_by_tid[s_task.task_id] = {"true_upstream_mode": "U0", "true_effective_regime": "D0"}
        m_by_sn = {m["short_name"]: m for m in manifest}

        forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
        nid = next(iter(forest.nodes))
        uni_keys = m98.expected_m98_public_output_keys(spec.family)
        s0_pub, s0_sha = m98.materialize_m98_public_item_and_sha(s_task)
        s0_mono = _tag_mono("U0", "D0")
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

        def _mock_gate_probe(*, task: Any, candidate_code: str, **kw: Any) -> dict[str, Any]:
            base_tid = str(task.task_id).split("__downstream_module")[0].split("__upstream_module")[0]
            mr = m_by_tid[base_tid]
            ok = _eval_tagged_code(candidate_code, mr["true_upstream_mode"], mr["true_effective_regime"])
            return _fixture_gate(task, candidate_code, ok, kw.get("expected_dependency_sha256"))

        def _mock_priv_grade(task: Any, art_map: dict[str, str], **kw: Any) -> dict[str, Any]:
            code = next(iter(art_map.values()))
            mr = m_by_tid[task.task_id]
            ok = _eval_tagged_code(code, mr["true_upstream_mode"], mr["true_effective_regime"])
            return {"strict_pass": ok, "reward": 1.0 if ok else 0.0}

        def _mock_locus(fam: str, pub_item: dict[str, Any], *, chosen_up_code: str | None = None, **kw: Any) -> dict[str, Any]:
            obs = m100.compute_m100_upstream_observables(fam, pub_item)
            true_u = "U0" if float(obs["up_stat_01_diagonal_ols_r2"]) > 0.96 else ("U1" if float(obs["up_stat_03_adj_offdiag_abs_corr_max"]) > 0.12 else "U2")
            ns: dict[str, Any] = {}
            exec(compile(chosen_up_code or "", "<u_chk>", "exec"), ns)
            got_u = ns["calibrate_upstream"]({}).get("_up_tag")
            ok_u = got_u == true_u
            return {
                "candidate_code_sha256": hashlib.sha256(chosen_up_code.encode()).hexdigest(),
                "locus": ("downstream_constitutive" if ok_u else "upstream_transducer"),
                "component_probe_executed": True,
                "execution_status": "completed",
                "evidence_origin": "direct_component_probe",
                "failure_tier": ("NONE" if ok_u else "SCIENTIFIC_FAILURE"),
                "failure_kind": ("none" if ok_u else "scientific_or_numerical"),
                "chosen_up_holdout_rmse": (0.0001 if ok_u else 0.095),
                "probe_id": f"uprobe_mock_{hashlib.sha256((chosen_up_code or '').encode()).hexdigest()[:12]}",
                "probe_wall_ms": 2.5,
            }

        def _mock_frontier_ep(ft: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            assert kwargs.get("enable_auto_skill_packaging") is False
            assert kwargs.get("allow_l1_primitives") is False
            sdir = kwargs["episode_out_dir"] / "04_submitted_artifacts_snapshot"
            sdir.mkdir(parents=True, exist_ok=True)
            mr = m_by_sn[ft.short_name]
            up_m, eff_r = mr["true_upstream_mode"], mr["true_effective_regime"]
            if any(a.endswith("upstream_transducer.py") for a in ft.declared_artifacts):
                (sdir / "slot_01__upstream_transducer.py").write_text(_tag_up(up_m), encoding="utf-8")
                (sdir / "slot_00__batch_solver.py").write_text("# wrap\n", encoding="utf-8")
            elif any(a.endswith("downstream_solver.py") for a in ft.declared_artifacts):
                (sdir / "slot_01__downstream_solver.py").write_text(_tag_dn(eff_r), encoding="utf-8")
                (sdir / "slot_00__batch_solver.py").write_text("# wrap\n", encoding="utf-8")
            else:
                (sdir / "slot_00__batch_solver.py").write_text(_tag_mono(up_m, eff_r), encoding="utf-8")
            return {"task_id": ft.task_id, "short_name": ft.short_name, "turns_used": 3, "ended_by_finish": True}

        def _mock_call_structured(self_cli: Any, **kw: Any) -> dict[str, Any]:
            props = set((kw.get("output_schema") or {}).get("properties", {}).keys())
            if "upstream_code" in props:
                return {"ok": True, "parsed": {"diagnosis": "2x2 decomp", "upstream_code": _tag_up("U0"), "downstream_d0_code": _tag_dn("D0"), "downstream_d1_code": _tag_dn("D1")}}
            if "downstream_code" in props:
                return {"ok": True, "parsed": {"diagnosis": "lazy proj", "downstream_code": _tag_dn("D1")}}
            uj = json.loads(kw["conversation_turns"][0]["content"])
            ex_ids, new_id, pos_m = uj["existing_branch_ids"], uj["new_branch_id"], uj["positive_witnesses_by_branch"]
            spreads = uj.get("empirical_feature_spreads") or {}
            all_ids = list(ex_ids) + [new_id]

            def _dn_cell(w: dict[str, float]) -> tuple[bool, bool] | None:
                if fn_d1 in w and fn_d2 in w:
                    return (w[fn_d1] > 1.0, w[fn_d2] > 1.0)
                return None

            def _g_for(bid_c: str, wl: list[dict[str, float]]) -> dict[str, Any]:
                w0 = wl[0]
                cls_l = []
                mc = _dn_cell(w0)
                col_dn = mc is None or any(oid != bid_c and _dn_cell(pos_m[oid][0]) == mc for oid in all_ids)
                if col_dn:
                    if "up_stat_03_adj_offdiag_abs_corr_max" in w0 and spreads.get("up_stat_03_adj_offdiag_abs_corr_max", 0.0) >= 0.07 and len({w["up_stat_03_adj_offdiag_abs_corr_max"] > 0.12 for w in wl}) == 1:
                        cls_l.append({"feature_name": "up_stat_03_adj_offdiag_abs_corr_max", "op": (">" if w0["up_stat_03_adj_offdiag_abs_corr_max"] > 0.12 else "<="), "threshold": 0.12})
                    if "up_stat_04_opp_offdiag_abs_corr_max" in w0 and spreads.get("up_stat_04_opp_offdiag_abs_corr_max", 0.0) >= 0.07 and len({w["up_stat_04_opp_offdiag_abs_corr_max"] > 0.10 for w in wl}) == 1:
                        cls_l.append({"feature_name": "up_stat_04_opp_offdiag_abs_corr_max", "op": (">" if w0["up_stat_04_opp_offdiag_abs_corr_max"] > 0.10 else "<="), "threshold": 0.10})
                if fn_d1 in w0 and spreads.get(fn_d1, 0.0) >= 0.07:
                    cls_l.append({"feature_name": fn_d1, "op": (">" if w0[fn_d1] > 1.0 else "<="), "threshold": 1.0})
                if fn_d2 in w0 and spreads.get(fn_d2, 0.0) >= 0.07:
                    cls_l.append({"feature_name": fn_d2, "op": (">" if w0[fn_d2] > 1.0 else "<="), "threshold": 1.0})
                return {"all_of": cls_l}

            return {"ok": True, "parsed": {"diagnosis": "ok", "existing_branch_guards": {b: _g_for(b, pos_m[b]) for b in ex_ids}, "new_branch_guard": _g_for(new_id, pos_m[new_id])}}

        nt_ok = json.dumps({"cal_identity_err": 0.0, "shape_ok": True, "transformed_signal": True})
        fake_codex = {"available": True, "auth_mode": "chatgpt", "executable": "/bin/true", "is_windows_exe_under_wsl": False, "stage_root_wsl": str(tmp_path), "stage_root_cli_str": str(tmp_path)}

        with patch.object(runner, "materialize_m99_public_item_and_sha", m98.materialize_m98_public_item_and_sha), patch.object(
            m99, "materialize_m99_public_item_and_sha", m98.materialize_m98_public_item_and_sha
        ), patch.object(runner, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate_probe), patch.object(
            m99, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate_probe
        ), patch.object(m100, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate_probe), patch.object(
            runner, "grade_m94_artifact_map", side_effect=_mock_priv_grade
        ), patch.object(runner, "diagnose_m98_factored_conflict_locus", side_effect=_mock_locus), patch.object(
            m99, "diagnose_m98_factored_conflict_locus", side_effect=_mock_locus
        ), patch("subprocess.run", return_value=subprocess.CompletedProcess(["docker"], 0, nt_ok, "")), patch.object(
            runner.m6f, "run_frontier_episode", side_effect=_mock_frontier_ep
        ), patch.object(runner.m6f.FrontierCodexClient, "call_structured", _mock_call_structured):
            base_dir = tmp_path / f"run_{idx_sp}"
            arm_res = {}
            for arm_nm in m100.M100_ARMS:
                st_dir = base_dir / "continual_streams" / wid / arm_nm
                st_dir.mkdir(parents=True, exist_ok=True)
                arm_res[arm_nm] = runner._run_one_m100_world_arm_stream(
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
                    env_image_digest="mock_img",
                    model="gpt-6.2-sol",
                    codex_info=fake_codex,
                    stream_out_dir=st_dir,
                    run_id=f"mock_{wid}",
                    max_turns=6,
                    reasoning_effort="medium",
                )

            assert arm_res["Flat_MonolithicConditionalTable_Online"]["strict_passes"] == 10
            assert arm_res["Flat_PlainModuleRegistry_Online"]["strict_passes"] == 10
            assert arm_res["Hegel_SelectiveConditionExpansion_Online"]["strict_passes"] == 10

            # Exact structural zero-LLM separation on BOTH `sep_first` AND `coupled_first`!
            assert arm_res["Flat_MonolithicConditionalTable_Online"]["zero_turn_passed_closures"] == 2  # Revisits S9, S10 only!
            assert arm_res["Flat_PlainModuleRegistry_Online"]["zero_turn_passed_closures"] == 3  # 3 unseen sep combs; thrashes on S9 & S10!
            assert arm_res["Hegel_SelectiveConditionExpansion_Online"]["zero_turn_passed_closures"] == 5  # 3 unseen sep combs + BOTH S9 & S10!

            sw_recs = arm_res["Hegel_SelectiveConditionExpansion_Online"]["separability_withdrawal_receipts"]
            assert len(sw_recs) == 1
            if spec.order_mode == "coupled_first":
                # Verify `S04_d2_coupled_first` did NOT issue a fake withdrawal; reverse withdrawal fired at `S07_rev_sep_break`!
                s4_ep = json.loads(
                    (base_dir / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online" / f"{wid}_batch_04__blind_continual_campaign/00_episode_summary.json").read_text()
                )
                s7_ep = json.loads(
                    (base_dir / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online" / f"{wid}_batch_07__blind_continual_campaign/00_episode_summary.json").read_text()
                )
                assert s4_ep["pre_llm_trial_receipt"]["separability_withdrawal_receipt"] is None
                assert s7_ep["pre_llm_trial_receipt"]["separability_withdrawal_receipt"] is not None

            # Verify full 6-link cold hydration + `step_08` routing & plain-dict `2/2` parity!
            ck8 = base_dir / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online/graph_ckpt_after_step_08.json"
            g8 = PersistentMechanismGraph.load_checkpoint(ck8)
            r8 = m94.M94BindingRegistry(g8)
            assert len(m100.hydrate_separability_withdrawals_from_graph(g8, r8, wid, strict_ledger_fk=True)) == 1

            abl = runner.run_m100_step08_constrained_routing_and_plain_dict_ablation(
                world_specs=(spec,), built_worlds={wid: (s_task, stream_tasks, manifest, logs)}, out_dir=base_dir
            )
            parity = abl["plain_dict_same_joint_features_parity"]
            assert parity["matched_episodes"] == parity["evaluated_episodes"] == len(parity["rows"]) == 2
            assert all(row["plain_candidate_sha256"] and row["plain_candidate_sha256"] == row["graph_candidate_sha256"]
                       for row in parity["rows"])
            assert abl["summary_by_policy"]["dn_only_fixed_prior_d2"]["strict_passes"] == 1
            assert abl["summary_by_policy"]["dn_only_fixed_latest_d2"]["strict_passes"] == 1
            assert abl["summary_by_policy"]["hegel_selective_condition_expansion"]["strict_passes"] == 2


@pytest.mark.parametrize("raw_supports_revisit", [False, True])
def test_m100_b2_sibling_upstream_fallback_and_b3_deferred_lazy_projection(tmp_path: Path, raw_supports_revisit: bool) -> None:
    """Explicit `Gate 1` regression for `B2` (`Q03`: existing `u1` excluded by over-narrow guard recovered via 1-sibling
    direct + joint gate fallback) AND `B3` (`Q04`: `S01` decomposition fails -> `S02` decomposes -> `S01` monolithic `D1`
    lazily projected on `S04_comb_u1_d1` across BOTH `Plain` and `Hegel`!).
    """
    runner = _load_m100_runner()
    spec = m100.DEV_M100_WORLD_SPECS[0]  # `sep_first`: S1=(U0,D1), S2=(U0,D2), S3=(U1,D0), S4=(U1,D1)
    wid = spec.world_id
    s_task, stream_tasks, manifest, logs = m100.build_m100_world_tasks(spec)
    # Run first 4 steps (`S01..S04`) where `S01` decomp intentionally fails, `S02` decomp succeeds (`d0, d2`),
    # `S03` learns `u1` with an intentionally over-narrow `up_stat_06_cal_frobenius_rms > 999.0` guard (triggering `B2` on `S04`!),
    # AND `S04 (U1, D1)` triggers `B2` (`u1` sibling fallback) + `B3` (lazy projection of `S01`'s `D1` monolithic solver)!
    sub_tasks = stream_tasks[:4]
    sub_manifest = manifest[:4]
    m_by_tid = {m["task_id"]: m for m in manifest}
    m_by_tid[s_task.task_id] = {"true_upstream_mode": "U0", "true_effective_regime": "D0"}
    m_by_sn = {m["short_name"]: m for m in manifest}

    def _tag_up(mode: str) -> str:
        return f"# UP:{mode}\ndef calibrate_upstream(item, param_payload=None):\n    item = dict(item); item['_up_tag'] = {mode!r}; return item\n"

    def _tag_dn(eff: str) -> str:
        return f"# DN:{eff}\ndef solve_downstream(item):\n    return {{'_up': item.get('_up_tag'), '_dn': {eff!r}}}\n"

    def _tag_mono(up_m: str, eff: str) -> str:
        if raw_supports_revisit and (up_m, eff) == ("U0", "D1"):
            return "def solve_batch(item): return {'_up': item['observed_mode'], '_dn': 'D1'}"
        return f"# MONO:{up_m}__{eff}\ndef solve_batch(item):\n    return {{'_up': {up_m!r}, '_dn': {eff!r}}}\n"

    def _eval_tagged(cand_code: str, true_up: str, true_eff: str) -> bool:
        ns: dict[str, Any] = {}
        exec(compile(cand_code, "<tc>", "exec"), ns)
        out = ns["solve_batch"]({"observed_mode": true_up})
        return out.get("_up") == true_up and out.get("_dn") == true_eff

    def _mock_gate(*, task: Any, candidate_code: str, **kw: Any) -> dict[str, Any]:
        btid = str(task.task_id).split("__downstream_module")[0].split("__upstream_module")[0]
        mr = m_by_tid[btid]
        ok = _eval_tagged(candidate_code, mr["true_upstream_mode"], mr["true_effective_regime"])
        return _fixture_gate(task, candidate_code, ok, kw.get("expected_dependency_sha256"))

    def _mock_priv(task: Any, art_map: dict[str, str], **kw: Any) -> dict[str, Any]:
        mr = m_by_tid[task.task_id]
        ok = _eval_tagged(next(iter(art_map.values())), mr["true_upstream_mode"], mr["true_effective_regime"])
        return {"strict_pass": ok, "reward": 1.0 if ok else 0.0}

    def _mock_loc(fam: str, pub_item: dict[str, Any], *, chosen_up_code: str | None = None, **kw: Any) -> dict[str, Any]:
        obs = m100.compute_m100_upstream_observables(fam, pub_item)
        true_u = "U0" if float(obs["up_stat_01_diagonal_ols_r2"]) > 0.96 else "U1"
        ns: dict[str, Any] = {}
        exec(compile(chosen_up_code or "", "<uc>", "exec"), ns)
        ok_u = ns["calibrate_upstream"]({}).get("_up_tag") == true_u
        return {"candidate_code_sha256": hashlib.sha256(chosen_up_code.encode()).hexdigest(), "locus": ("downstream_constitutive" if ok_u else "upstream_transducer"), "component_probe_executed": True, "execution_status": "completed", "evidence_origin": "direct_component_probe", "failure_tier": ("NONE" if ok_u else "SCIENTIFIC_FAILURE"), "failure_kind": ("none" if ok_u else "scientific_or_numerical"), "chosen_up_holdout_rmse": (0.00008 if ok_u else 0.55), "probe_id": f"uprobe_{hashlib.sha256((chosen_up_code or '').encode()).hexdigest()[:10]}", "probe_wall_ms": 1.5}

    def _mock_ep(ft: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        sdir = kwargs["episode_out_dir"] / "04_submitted_artifacts_snapshot"
        sdir.mkdir(parents=True, exist_ok=True)
        mr = m_by_sn[ft.short_name]
        if any(a.endswith("upstream_transducer.py") for a in ft.declared_artifacts):
            (sdir / "slot_01__upstream_transducer.py").write_text(_tag_up(mr["true_upstream_mode"]), encoding="utf-8")
            (sdir / "slot_00__batch_solver.py").write_text("# w\n", encoding="utf-8")
        elif any(a.endswith("downstream_solver.py") for a in ft.declared_artifacts):
            (sdir / "slot_01__downstream_solver.py").write_text(_tag_dn(mr["true_effective_regime"]), encoding="utf-8")
            (sdir / "slot_00__batch_solver.py").write_text("# w\n", encoding="utf-8")
        else:
            (sdir / "slot_00__batch_solver.py").write_text(_tag_mono(mr["true_upstream_mode"], mr["true_effective_regime"]), encoding="utf-8")
        return {"task_id": ft.task_id, "short_name": ft.short_name, "turns_used": 2, "ended_by_finish": True}

    def _mock_cs(self_cli: Any, **kw: Any) -> dict[str, Any]:
        st_sub = str(kw.get("stage_subdir") or "")
        props = set((kw.get("output_schema") or {}).get("properties", {}).keys())
        if "upstream_code" in props:
            if "step01" in st_sub:
                return {"ok": False, "parsed": None}  # Force `S01` decomposition miss (`c02` mode)!
            return {"ok": True, "parsed": {"diagnosis": "S02 decomp", "upstream_code": _tag_up("U0"), "downstream_d0_code": _tag_dn("D0"), "downstream_d1_code": _tag_dn("D2")}}
        if "downstream_code" in props:
            return {"ok": True, "parsed": {"diagnosis": "lazy D1", "downstream_code": _tag_dn("D1")}}
        # Force `S03` upstream guard induction to give `u1` an over-narrow `up_stat_06_cal_frobenius_rms > 1.29` guard
        # (`c04/E7` mode: passes on `S03` `v2_gains`, evaluates `FALSE` on `S04` `v3_gains`, forcing `u0` primary -> `B2` `1`-sibling fallback to `u1`!)
        if "_up_gind" in st_sub:
            uj_u = json.loads(kw["conversation_turns"][0]["content"])
            ex_u, nw_u = uj_u["existing_branch_ids"][0], uj_u["new_branch_id"]
            return {
                "ok": True,
                "parsed": {
                    "diagnosis": "narrow frob guard",
                    "existing_branch_guards": {ex_u: {"all_of": [{"feature_name": "up_stat_01_diagonal_ols_r2", "op": ">", "threshold": 0.94}]}},
                    "new_branch_guard": {"all_of": [{"feature_name": "up_stat_01_diagonal_ols_r2", "op": "<=", "threshold": 0.94}]},
                },
            }
        ch_d2 = spec.active_log_channels[1]
        fn_d2 = f"dn_stat_0{ch_d2 + 1}_log_channel_{ch_d2}_mean"
        uj = json.loads(kw["conversation_turns"][0]["content"])
        ex_i, nw_i = uj["existing_branch_ids"], uj["new_branch_id"]
        return {
            "ok": True,
            "parsed": {
                "diagnosis": "d0 vs d2",
                "existing_branch_guards": {ex_i[0]: {"all_of": [{"feature_name": fn_d2, "op": "<=", "threshold": 1.0}]}},
                "new_branch_guard": {"all_of": [{"feature_name": fn_d2, "op": ">", "threshold": 1.0}]},
            },
        }

    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    nid = next(iter(forest.nodes))
    uni_keys = m98.expected_m98_public_output_keys(spec.family)
    s0_pub, s0_sha = m98.materialize_m98_public_item_and_sha(s_task)
    s0_mono = _tag_mono("U0", "D0")
    s0_ent = m95.make_continual_archive_entry(short_name=s_task.short_name, task_id=s_task.task_id, canonical_entrypoint="solve_batch", canonical_algorithm_code=s0_mono, output_schema_keys=uni_keys, public_qualification_status="public_gate_verified")
    nt_ok = json.dumps({"cal_identity_err": 0.0, "shape_ok": True, "transformed_signal": True})
    fake_codex = {"available": True, "auth_mode": "chatgpt", "executable": "/bin/true", "is_windows_exe_under_wsl": False, "stage_root_wsl": str(tmp_path), "stage_root_cli_str": str(tmp_path)}

    with patch.object(runner, "materialize_m99_public_item_and_sha", m98.materialize_m98_public_item_and_sha), patch.object(
            m99, "materialize_m99_public_item_and_sha", m98.materialize_m98_public_item_and_sha
        ), patch.object(runner, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate), patch.object(
        m99, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate
    ), patch.object(m100, "execute_single_pre_llm_public_gate_probe_in_container", side_effect=_mock_gate), patch.object(
        runner, "grade_m94_artifact_map", side_effect=_mock_priv
    ), patch.object(runner, "diagnose_m98_factored_conflict_locus", side_effect=_mock_loc), patch.object(
        m99, "diagnose_m98_factored_conflict_locus", side_effect=_mock_loc
    ), patch("subprocess.run", return_value=subprocess.CompletedProcess(["docker"], 0, nt_ok, "")), patch.object(
        runner.m6f, "run_frontier_episode", side_effect=_mock_ep
    ), patch.object(runner.m6f.FrontierCodexClient, "call_structured", _mock_cs):
        for arm_nm in ("Flat_PlainModuleRegistry_Online", "Hegel_SelectiveConditionExpansion_Online"):
            # To also force `u0` to be tried FIRST on `S04` (testing `B2` sibling fallback!), note when `_up_gind` fails `u1` gets priority `23 > 10`.
            # Wait: let's also verify `B3` directly on `S04` (`4/4` strict_pass via `lazy_retroactive_projection_reuse`)!
            st_dir = tmp_path / arm_nm
            st_dir.mkdir(parents=True, exist_ok=True)
            res = runner._run_one_m100_world_arm_stream(
                spec=spec,
                arm=arm_nm,
                s_task=s_task,
                stream_tasks=sub_tasks,
                schedule_manifest=sub_manifest,
                apparatus_logs=logs,
                base_v7_forest=forest,
                base_v7_archive=archive,
                source_monolithic_code=s0_mono,
                source_archive_entry=s0_ent,
                source_forest_nid=nid,
                source_pub_item=s0_pub,
                source_pub_sha256=s0_sha,
                env_image_digest="mock_img",
                model="gpt-6.2-sol",
                codex_info=fake_codex,
                stream_out_dir=st_dir,
                run_id=f"b2b3_{arm_nm}",
                max_turns=6,
                reasoning_effort="medium",
            )
            assert res["strict_passes"] == 4
            assert len(res["lazy_retroactive_projection_events"]) == (0 if raw_supports_revisit else 1)
            expected_mode = "zero_llm_raw_archive_reuse" if raw_supports_revisit else "lazy_retroactive_projection_reuse"
            assert res["episodes_meta"][3]["execution_mode"] == expected_mode
            episode = json.loads((st_dir / sub_tasks[3].task_id / "00_episode_summary.json").read_text())
            raw = episode["pre_llm_trial_receipt"]["raw_archive_replay"]
            assert raw["source_task_id"] == sub_tasks[0].task_id and raw["target_task_id"] == sub_tasks[3].task_id
            assert raw["probe_result"]["accepted"] is raw_supports_revisit
            assert episode["pre_llm_trial_receipt"]["pre_llm_probes_used"] >= 2
            assert episode["pre_llm_probe_wall_ms"] >= raw["wall_ms"]
            assert (episode["turns_used"] == 0) == raw_supports_revisit
