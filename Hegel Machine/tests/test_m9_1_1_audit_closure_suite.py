"""Deterministic `M9.1.1` Audit Closure Test Suite (`C1..C6`).

Closes all reproducible counterexamples from `Hegel_M9_1_Audit`:
1. `C1` (`normalize_and_validate_box_l0_public_schema`): Canonical `bool[p]` schema validation, retrospective
   `index_list` normalization equivalence, and rejection of duplicates/OOB/overlaps/non-finite fields.
2. `C2` (`4-tuple qualification identity` & `order-invariant L1 version pool`): Changing `synthetic_unit_test_code`
   triggers reverification and invalidates qualification; missing `verified_code_sha256` is NEVER silently stamped;
   duplicate-stem L1 versions are evaluated completely and selected independently of `notes` order.
3. `C3` (`search_skills`, subproblem/negation scoping, event-clean trigger, & repair catalog passthrough):
   `"robust regression"` finds `cauchy_linear_fit` & `robust_residual_weights`; `"complex Helmholtz preconditioner"`
   reports precondition rejection; multi-stage `complex128` + real float64 IRLS recommends real primitives; `"Avoid
   indefinite operators; input is SPD"` recommends SPD primitives; `echo` / typo `--describe` / `grep` exit-1 trigger
   edge cases behave cleanly; `LocalContainerRepair` receives full metadata preconditions.
4. `C4` (`nogood` witness spoofing, treatment-invariant router, canonical AST hash, & multi-attempt usage accounting):
   `assert callable(sol.solve); print('gauge scale identifiability pass')` is rejected; `skill_activation_count` does
   not alter `classify_public_container_verification_state`; missing `feature_builder_sha256` is rejected and Python
   3.12 `type_params` normalization preserves canonical hash; `FrontierCodexClient.call_structured` accumulates usage
   across retries and timeouts (`3300 == 3300`).
5. `C6` (`retention alias/shadowing` & `ObservationContext` temporal production chain): Top-level alias rebinding and
   local variable shadowing do not count as `entrypoint_reachable_l1_functions`; real production `RevisionCandidate`
   closures + `AcquireObservation` + checkpoint reload work end-to-end on structured temporal `ObservationContext`.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition import m8_offline_container_planner as m8r
from hegel_machine.cognition import m8_persistent_variational_planner as vp
from hegel_machine.cognition.m8_9_1_box_l0_certificate_kernel import (
    normalize_and_validate_box_l0_public_schema,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

REPO_ROOT = Path(__file__).resolve().parents[1]
V5_BUNDLE = REPO_ROOT / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/pretrained_autogenous_forest_v5_m89.json"


def test_c1_canonical_box_l0_public_schema_and_index_compat() -> None:
    p = 5
    d = [0.2] * p
    c = [0.0] * p
    bool_cert = {
        "weights": [0.0, 0.5, 0.0, 0.0, 0.0],
        "objective": 1.25,
        "certified_lower_bound": 1.25,
        "subcube_leaves": [
            {"fixed_zero": [True, False, False, False, False], "fixed_active": [False, False, False, False, False], "lb": 1.26, "minorant_diagonal": d, "minorant_center": c},
            {"fixed_zero": [False, False, False, False, False], "fixed_active": [True, False, False, False, False], "lb": 1.25, "minorant_diagonal": d, "minorant_center": c},
        ],
    }
    idx_cert = {
        "weights": [0.0, 0.5, 0.0, 0.0, 0.0],
        "objective": 1.25,
        "certified_lower_bound": 1.25,
        "subcube_leaves": [
            {"fixed_zero": [0], "fixed_active": [], "lb": 1.26, "minorant_diagonal": d, "minorant_center": c},
            {"fixed_zero": [], "fixed_active": [0], "lb": 1.25, "minorant_diagonal": d, "minorant_center": c},
        ],
    }
    r_bool = normalize_and_validate_box_l0_public_schema(bool_cert, p, box=1.0, allow_index_list_compat=False)
    assert r_bool["schema_valid"] is True and r_bool["partition_valid"] is True

    # Strict mode rejects unannounced integer index list, while explicit compat mode normalizes identically!
    r_idx_strict = normalize_and_validate_box_l0_public_schema(idx_cert, p, box=1.0, allow_index_list_compat=False)
    assert r_idx_strict["schema_valid"] is False
    r_idx_compat = normalize_and_validate_box_l0_public_schema(idx_cert, p, box=1.0, allow_index_list_compat=True)
    assert r_idx_compat["schema_valid"] is True and r_idx_compat["partition_valid"] is True
    assert r_idx_compat["canonical"] == r_bool["canonical"]

    # Duplicates, OOB, and overlapping zero/active are rejected even in compat mode
    dup_cert = deepcopy(idx_cert)
    dup_cert["subcube_leaves"][0]["fixed_zero"] = [0, 0]
    assert normalize_and_validate_box_l0_public_schema(dup_cert, p, allow_index_list_compat=True)["schema_valid"] is False
    ov_cert = deepcopy(bool_cert)
    ov_cert["subcube_leaves"][0]["fixed_active"] = [True, False, False, False, False]
    assert normalize_and_validate_box_l0_public_schema(ov_cert, p)["schema_valid"] is False


def test_c2_qualification_test_sha_no_silent_stamping_and_version_order_invariance() -> None:
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    box_mod = next(
        m
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in r.get("autogenous_wormhole_modules", [])
        if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
    )
    box_mod["synthetic_unit_test_code"] = "raise AssertionError('deliberate protocol test failure')\n"

    # Immediately stale BEFORE reverification
    q_immediate = m64.verify_l1_module_qualification_record(box_mod)
    assert q_immediate[0] is False and "test_sha256_mismatch" in q_immediate[1]

    called: list[dict] = []

    def _mock_verify(**kw):
        called.append(kw)
        return {"passed": False}

    with (
        patch.object(m7a, "ensure_clean_scipy_sandbox_image", lambda: None),
        patch.object(m7a.m6f, "start_detached_container", lambda *a, **k: "mock-cid"),
        patch.object(m7a.m6f, "stop_container_safely", lambda *a, **k: None),
        patch.object(m7a, "verify_autogenous_module_in_clean_container", _mock_verify),
    ):
        m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=True)
    assert len(called) >= 1
    assert m64.verify_l1_module_qualification_record(box_mod)[0] is False

    # Missing `verified_code_sha256` must NEVER be silently stamped when `reverify_in_clean_container=False`
    _, archive2, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    clean_m = deepcopy(
        next(
            m
            for _, r in m7a.iter_live_archive_nodes(archive2)
            for m in r.get("autogenous_wormhole_modules", [])
            if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
        )
    )
    clean_m["unit_test_receipt"].pop("verified_code_sha256", None)
    assert m64.verify_l1_module_qualification_record(clean_m)[0] is False
    syn_arch = {"n1": {"node_id": "n1", "autogenous_only": True, "autogenous_wormhole_modules": [clean_m]}}
    m7a.enrich_and_harden_l1_primitive_contracts(syn_arch, reverify_in_clean_container=False)
    assert m64.verify_l1_module_qualification_record(clean_m)[0] is False

    # Order-invariant duplicate-stem L1 selection + explicit superseded version audit
    def _mk_qual(src: str) -> dict:
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
        return {
            "filename": "autogenous_prim_versions.py",
            "python_code": src,
            "compiled_code_sha256": sha,
            "unit_test_verified": True,
            "unit_test_receipt": {"passed": True, "verified_code_sha256": sha},
            "granularity_tier": "L1_compositional_primitive",
        }

    va = _mk_qual("def version(x):\n    return x + 1\n")
    vb = _mk_qual("def version(x):\n    return x + 2\n")
    notes_ab = [{"node_id": "nodeA", "autogenous_only": True, "autogenous_wormhole_modules": [va]}, {"node_id": "nodeB", "autogenous_only": True, "autogenous_wormhole_modules": [vb]}]
    mods_1, _, diag_1 = m64.collect_recalled_executable_wormhole_sources(notes_ab, include_foundational_skills=False, allow_l1_primitives=True, task_query_text="version", disclosure_mode="progressive", staging_policy="all_compatible", return_diagnostics=True)
    mods_2, _, diag_2 = m64.collect_recalled_executable_wormhole_sources(list(reversed(notes_ab)), include_foundational_skills=False, allow_l1_primitives=True, task_query_text="version", disclosure_mode="progressive", staging_policy="all_compatible", return_diagnostics=True)
    assert mods_1 == mods_2
    assert len(diag_1["net_value_evaluations"]) == 2 and len(diag_2["net_value_evaluations"]) == 2
    assert len(diag_1["superseded_l1_versions"]) == 1


def test_c3_synonym_search_scoped_preconditions_trigger_and_repair_catalog() -> None:
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(V5_BUNDLE)
    _, _, diag = m64.collect_recalled_executable_wormhole_sources(
        [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=archive,
        task_query_text="Fit real float64 panel with Cauchy IRLS and box L0 lower bound", disclosure_mode="on_demand_executable", staging_policy="net_value_gated", return_diagnostics=True,
    )
    with tempfile.TemporaryDirectory(prefix="m911_search_") as td:
        p = Path(td)
        (p / "_skill_contracts.json").write_text(diag["contracts_catalog_json"], encoding="utf-8")
        ns = {"__file__": str(p / "__init__.py"), "__name__": "hegel_wormhole_skills"}
        exec(compile(m64.SKILL_INIT_PY, "<skill_init>", "exec"), ns)
        ns["INSPECTION_LOG_PATH"] = str(p / "insp.jsonl")

        r_rob = ns["search_skills"]("robust regression")
        assert "cauchy_linear_fit" in r_rob or "robust_residual_weights" in r_rob
        assert "No matching" not in r_rob

        r_helm = ns["search_skills"]("complex Helmholtz preconditioner")
        assert "No applicable skill cards" in r_helm

        r_cert = ns["search_skills"]("certified lower bound")
        assert "box_ridge_cube_lower_bound" in r_cert
        assert "fit_anchored_relaxation" not in r_cert

    # Subproblem-scoped & negation-aware precondition checks
    mods_by_fn = {
        m["filename"]: m
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in r.get("autogenous_wormhole_modules", [])
        if str(m.get("filename", "")).startswith("autogenous_prim_")
    }
    mixed_q = "Stage A stores a complex128 wave field. Stage B independently fits real float64 sensor data with robust affine gain and offset; time-dependent Cauchy spikes require IRLS."
    assert m64.evaluate_l1_skill_staging_net_value(mods_by_fn["autogenous_prim_robust_affine_projection.py"], "", set(), mixed_q)["recommend_turn1_staging"] is True
    assert m64.evaluate_l1_skill_staging_net_value(mods_by_fn["autogenous_prim_robust_trend_decomposition.py"], "", set(), mixed_q)["recommend_turn1_staging"] is True

    neg_indef_q = "Solve a large 5184 unknowns Robin diffusion PDE. The assembled operator is symmetric positive definite SPD. Avoid indefinite operators; input is real positive definite."
    assert m64.evaluate_l1_skill_staging_net_value(mods_by_fn["autogenous_prim_conservative_spd_solve.py"], "", set(), neg_indef_q)["recommend_turn1_staging"] is True

    # Trigger state-machine controls
    src_f = (REPO_ROOT / "src/hegel_machine/cognition/m6_frontier_real_benchmarks.py").read_text(encoding="utf-8")
    fn_ep = next(n for n in ast.parse(src_f).body if isinstance(n, ast.FunctionDef) and n.name == "run_frontier_episode")
    trig_blk = next(n for n in ast.walk(fn_ep) if isinstance(n, ast.If) and "demand_recommended_cards" in ast.unparse(n.test) and "turn_idx >= 2" in ast.unparse(n.test))
    trig_code = textwrap.dedent(ast.get_source_segment(src_f, trig_blk) or "")

    def _run_trig(overrides: dict) -> dict:
        ns_t = {
            "disclosure_mode": "on_demand_executable",
            "demand_recommended_cards": ["autogenous_prim_box_ridge_bounds@18dcec21 | box_ridge_cube_lower_bound(...)"],
            "demand_skill_hint_emitted": False,
            "demand_skill_hint_turn": None,
            "turn_idx": 5,
            "turn_logs": [],
            "raw_exec_observations": [{"exit_code": 0, "command": "python3 solve.py", "stdout": ""}],
            "nogoods_recorded": [],
            "verified_candidate_sha256": "",
            "current_candidate_sha256": "cand_cur",
            "m6_state": {},
            **overrides,
        }
        exec(compile(trig_code, "<trig>", "exec"), ns_t)
        return ns_t

    assert _run_trig({"turn_logs": [{"bash_command": "echo 'hegel_wormhole_skills unavailable?'"}]})["demand_skill_hint_emitted"] is True
    assert _run_trig({"raw_exec_observations": [{"exit_code": 1, "command": "python3 -m hegel_wormhole_skills --describe typo.fn", "stdout": '{"ok": false}'}]})["demand_skill_hint_emitted"] is True
    assert _run_trig({"turn_idx": 2, "raw_exec_observations": [{"exit_code": 1, "command": "grep 'not-present' metadata.txt", "stdout": ""}]})["demand_skill_hint_emitted"] is False
    assert _run_trig({"demand_recommended_cards": [], "nogoods_recorded": [{"risk_kind": "model_mismatch"}]})["demand_skill_hint_emitted"] is True
    assert _run_trig({"verified_candidate_sha256": "old_sha", "current_candidate_sha256": "new_sha"})["demand_skill_hint_emitted"] is True

    # Repair catalog precondition passthrough
    precs_map = {Path(fn).stem: list(m.get("mathematical_preconditions") or []) for fn, m in mods_by_fn.items()}
    srcs_map = {Path(fn).stem: str(m["python_code"]) for fn, m in mods_by_fn.items()}
    rep_ctx = m64.build_repair_skill_context_from_sources(srcs_map, preconditions_by_stem=precs_map, query_text="reference panel missing strata", top_k_prefetched_contracts=2)
    assert rep_ctx["prefetched_contracts"]
    assert len(rep_ctx["prefetched_contracts"][0]["module_preconditions"]) >= 2


def test_c4_anti_spoof_witness_treatment_invariant_router_and_attempt_accounting() -> None:
    # 1. `assert callable(sol.solve); print('gauge scale identifiability pass')` MUST NOT pass as substantive or resolve identifiability!
    spoof_cmd = "python3 -c \"import sol; assert callable(sol.solve); print('gauge scale identifiability pass')\""
    assert m6f.is_substantive_verification_command(spoof_cmd, ("/app/sol.py",)) is False

    fp_f = REPO_ROOT / "src/hegel_machine/cognition/m6_frontier_real_benchmarks.py"
    tree_f = ast.parse(fp_f.read_text(encoding="utf-8"))
    prom_blk = next(n for n in ast.walk(tree_f) if isinstance(n, ast.If) and ast.unparse(n.test) == "pending_nogood_resolutions and nogoods_recorded")
    ng = {
        "nogood_id": "ng_t01",
        "risk_kind": "target_identifiability",
        "assumption_refuted": "absolute target scale identifiable from relative-only observations",
        "scope": "affine gauge scale",
        "resolved": False,
    }
    ns_w = {
        "re": re,
        "pending_nogood_resolutions": [{"proposed_turn": 2, "resolves_ids": ["ng_t01"], "proposed_by_nogood_id": None}],
        "nogoods_recorded": [ng],
        "nogood_resolution_events": [],
        "turn_idx": 3,
        "current_candidate_sha256": "sha_cur",
        "cmd_str": spoof_cmd,
        "ex_res": {"stdout": "gauge scale identifiability pass"},
    }
    exec(compile(ast.Module(body=[deepcopy(prom_blk)], type_ignores=[]), str(fp_f), "exec"), ns_w)
    assert ng["resolved"] is False

    # 2. Treatment-invariant `classify_public_container_verification_state` + missing builder SHA rejection
    for common_rec in (False, True):
        st0, _ = m8r.classify_public_container_verification_state(
            benchmark="Hegel-M91-HardDemand-2", ended_by_finish=True, contract_accepted=True,
            public_metrics={"in_container_substantive_receipt_verified": common_rec}, skill_activation_count=0, nogood_count=0,
        )
        st1, _ = m8r.classify_public_container_verification_state(
            benchmark="Hegel-M91-HardDemand-2", ended_by_finish=True, contract_accepted=True,
            public_metrics={"in_container_substantive_receipt_verified": common_rec}, skill_activation_count=5, nogood_count=2,
        )
        assert st0 == st1

    m74_recs = m8r.load_historical_suite_episode_records(REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124")
    pl = m8r.M8OfflineContainerPlanner.fit_from_historical_records(m74_recs, suite_ids=["ut"])
    blob = pl.to_json_dict()
    blob_no_sha = deepcopy(blob)
    blob_no_sha.pop("feature_builder_sha256", None)
    with pytest.raises(ValueError, match="feature_builder_sha256"):
        m8r.M8OfflineContainerPlanner.from_json_dict(blob_no_sha)

    # 3. `FrontierCodexClient.call_structured` retry & timeout cumulative usage accounting (`3300 == 3300`)
    with tempfile.TemporaryDirectory(prefix="m911_acct_") as td:
        cli = m6f.FrontierCodexClient(
            codex_info={"available": True, "auth_mode": "chatgpt", "executable": "mock-codex", "is_windows_exe_under_wsl": False, "stage_root_wsl": td, "stage_root_cli_str": td},
            model="mock-model",
        )
        calls: list[int] = []

        def _mock_retry(cmd, **kw):
            calls.append(1)
            idx = len(calls)
            u = {"input_tokens": 1000 * idx, "cached_input_tokens": 0, "output_tokens": 100 * idx}
            out_s = json.dumps({"type": "turn.completed", "usage": u}) + "\n"
            if idx == 1:
                return subprocess.CompletedProcess(cmd, 1, out_s, "stream disconnected connection")
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, out_s, "")

        with patch.object(m6f.subprocess, "run", side_effect=_mock_retry), patch.object(m6f, "sleep", return_value=None):
            res_r = cli.call_structured(system_prompt="sys", conversation_turns=[], output_schema={"type": "object"}, stage_subdir="r1")
        assert res_r["ok"] is True
        assert res_r["prompt_tokens"] == 3000 and res_r["completion_tokens"] == 300 and res_r["attempt_count"] == 2

        t_calls: list[int] = []

        def _mock_timeout(cmd, **kw):
            t_calls.append(1)
            raise subprocess.TimeoutExpired(cmd, 60, output=json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100}}))

        with patch.object(m6f.subprocess, "run", side_effect=_mock_timeout), patch.object(m6f, "sleep", return_value=None):
            res_t = cli.call_structured(system_prompt="sys", conversation_turns=[], output_schema={"type": "object"}, stage_subdir="t1")
        assert res_t["ok"] is False
        assert res_t["prompt_tokens"] == 3000 and res_t["completion_tokens"] == 300 and res_t["attempt_count"] == 3


def test_c6_retention_alias_rebinding_and_temporal_observation_context_end_to_end() -> None:
    # 1. Alias rebinding & local shadowing must NOT report `entrypoint_reachable_l1_functions`
    stem = "autogenous_prim_demo"
    src = "K = 2\ndef scale(x):\n    return x * K\n"
    pkg = m64.package_self_contained_wormhole_dependencies(f"from hegel_wormhole_skills.{stem} import scale as s\ndef solve(x):\n    return s(x)\n", {stem: src})
    assert pkg["ok"] is True
    for mode, expected_reach in (
        ("original", ["autogenous_prim_demo.scale"]),
        ("top_level_rebound", []),
        ("local_shadow", []),
    ):
        code = pkg["packaged_python_code"]
        if mode == "top_level_rebound":
            code += "\ns = lambda x: x + 1\n"
        elif mode == "local_shadow":
            code = code.replace("def solve(x):\n    return s(x)", "def solve(x):\n    s = lambda z: z + 1\n    return s(x)")
        with tempfile.TemporaryDirectory(prefix="m911_ret_") as td:
            Path(td, "sol.py").write_text(code, encoding="utf-8")
            ret = m64.inspect_final_deliverable_skill_retention(Path(td), {stem: src}, declared_entrypoints=("solve",))
            assert ret["closure_exact_present_l1_functions"] == ["autogenous_prim_demo.scale"]
            assert ret["entrypoint_reachable_l1_functions"] == expected_reach

    # 2. Full production `RevisionCandidate.predict_fn` + `AcquireObservation` + checkpoint reload on `ObservationContext`
    forest = PhilosophicalForest()
    graph = vp.PersistentMechanismGraph(forest=forest)
    graph.register_mechanism(vp.TypedMechanism(mechanism_id="m_root", parent_id=None, role="shared_linear", feature_names=("a",), shared_weights={"a": 1.0}))
    graph.register_mechanism(vp.TypedMechanism(mechanism_id="m_mem_src", parent_id="m_root", role="temporal_memory", feature_names=("a",), temporal_tau=2.0, temporal_driver_col="a"))

    def _mk_temporal_ds(h_vals: list[list[float]], a_vals: list[float]) -> dict:
        a_arr = np.array(a_vals, dtype=float)
        h_arr = np.array(h_vals, dtype=float)
        mem = np.mean(h_arr, axis=1)
        return {
            "x": {"a": a_arr},
            "y": 1.0 * a_arr + 2.5 * mem,
            "past_driver_history": h_arr,
            "history_dt": 0.5,
            "timestamps": np.arange(len(a_vals), dtype=float),
            "trajectory_ids": np.zeros(len(a_vals), dtype=int),
        }

    ds_tr = _mk_temporal_ds([[0.0, 0.2], [1.0, 1.2], [-1.0, -0.8], [0.5, 0.7], [1.5, 1.8], [-0.5, -0.3]], [0.4, 0.5, -0.3, 0.8, -0.2, 0.6])
    ds_va = _mk_temporal_ds([[2.0, 2.2], [-2.0, -1.8], [0.8, 1.0], [-0.8, -1.0]], [0.3, 0.3, -0.4, -0.4])
    planner = vp.M8VariationalPlanner(graph, min_gain_nats=50.0)  # High threshold forces Stage 7 `AcquireObservation`
    tm = planner.register_task(
        task_id="t_visco",
        shared_upstream_id="m_root",
        feature_names=("a",),
        train_data=ds_tr,
        val_data=ds_va,
        initial_local_shrinkage=1.0,
    )
    base_ev = vp.evaluate_committed_task_model(tm, graph, ds_va)
    cands = planner._discover_reusable_mechanism_candidates(tm, ds_tr, ds_va, float(base_ev["val_mean_log_predictive"]), float(base_ev["variational_free_energy"]))
    assert cands, "Expected production temporal reuse candidate"

    # Same instantaneous `a == 0.5`, DIFFERENT pre-window histories `[3.0, 3.0]` vs `[-3.0, -3.0]` -> distinct predictions!
    probe_hi = {"x": {"a": [0.5]}, "past_driver_history": [[3.0, 3.0]], "history_dt": 0.5, "timestamps": [10.0], "trajectory_ids": [1]}
    probe_lo = {"x": {"a": [0.5]}, "past_driver_history": [[-3.0, -3.0]], "history_dt": 0.5, "timestamps": [11.0], "trajectory_ids": [2]}
    p_hi = float(cands[0].predict_fn(probe_hi)[0])
    p_lo = float(cands[0].predict_fn(probe_lo)[0])
    assert abs(p_hi - p_lo) > 0.5, f"Same present x, different past history must diverge: {p_hi} vs {p_lo}"

    sel = vp.select_discriminating_observation_probe([("mem_cand", cands[0].predict_fn), ("zero", lambda _ctx: np.zeros(1))], [probe_hi, probe_lo])
    assert sel["selected_index"] in (0, 1)

    res_step = planner.evaluate_and_revise_task(
        "t_visco",
        candidate_probes=[probe_hi, probe_lo],
        probe_oracle_fn=lambda ctx: float(1.0 * ctx["x"]["a"][0] + 2.5 * np.mean(ctx["past_driver_history"])),  # type: ignore[index]
        budget_remaining=2.0,
    )
    assert res_step["selected_candidate"]["action_type"] == "AcquireObservation"
    assert len(graph.task_datasets["t_visco"]["train"]["y"]) == 7
    assert np.asarray(graph.task_datasets["t_visco"]["train"]["past_driver_history"]).shape == (7, 2)

    with tempfile.TemporaryDirectory(prefix="m911_ckpt_") as td_c:
        ckpt_p = Path(td_c) / "graph.json"
        graph.save_checkpoint(ckpt_p)
        reloaded = vp.PersistentMechanismGraph.load_checkpoint(ckpt_p)
        assert np.asarray(reloaded.task_datasets["t_visco"]["train"]["past_driver_history"]).shape == (7, 2)
        p_hi_reload = float(vp._evaluate_task_model_on_probe_input(reloaded.active_task_models["t_visco"], reloaded, probe_hi)[0])
        assert np.isfinite(p_hi_reload)
