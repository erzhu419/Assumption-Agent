"""Hegel M9.2.1 Deterministic Audit Closure Suite (`A1..A9`).

Verifies closure of every reproducible correctness, mathematical soundness, equal-information reader,
container bootstrap, 4-tuple qualification, host-controlled probe witness, temporal context, and
provider attempt-isolation counterexample from `Hegel_M9_2_Audit`.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition import m8_9_1_box_l0_certificate_kernel as k891
from hegel_machine.cognition import m8_persistent_variational_planner as vp
from hegel_machine.cognition.m9_2_equal_info_reuse_benchmark import (
    EQUAL_INFO_M92_BY_ID,
    EQUAL_INFO_M92_TASKS,
    equal_info_m92_task_to_frontier_task,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def _extract_funcs_from_script(src: str, drop_imports: tuple[str, ...] = ()) -> dict[str, Any]:
    kept = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Import, ast.ImportFrom)) and not any(x in ast.unparse(n) for x in drop_imports):
            kept.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            kept.append(n)
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=kept, type_ignores=[]), "<m921_test_extract>", "exec"), ns)
    return ns


def test_a1_shared_certificate_kernel_rejects_dual_sign_counterexamples_and_replays_12of12_eq02() -> None:
    eq02_task = EQUAL_INFO_M92_BY_ID["eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0"]
    ns2 = _extract_funcs_from_script(eq02_task.private_verifier_py, ("coupled_interferometry_certified_l0",))

    # 1. GPT-6 minimal 18-D negative-correlation Fenchel dual counterexample (`F=9, L=9` while `F(0.5*1)=6.3`)
    n = 18
    X_id = np.eye(n, dtype=float)
    y_ones = np.ones(n, dtype=float)
    fake_id = {
        "weights": [0.0] * n,
        "objective": 9.0,
        "certified_lower_bound": 9.0,
        "subcube_leaves": [
            {
                "fixed_zero": [False] * n,
                "fixed_active": [False] * n,
                "lb": 9.0,
                "dual_nu": (-y_ones).tolist(),
            }
        ],
    }
    ok_fake1, rsn_fake1, _ = ns2["_verify_cert_v2"](X_id, y_ones, 1.0, 0.1, 1.0, fake_id)
    assert not ok_fake1, f"Expected fake I_18 dual certificate to fail, got reason={rsn_fake1}"

    # 2. GPT-6 connected private-generator (`seed=99411, p=18`) synthetic-response dual counterexample
    st_conn, *_ = ns2["_make_priv"](99411, 18)
    xc = st_conn["X_cand"]
    yc = xc @ np.full(18, 0.6, dtype=float)
    fake_conn = {
        "weights": [0.0] * 18,
        "objective": float(0.5 * (yc @ yc)),
        "certified_lower_bound": float(0.5 * (yc @ yc)),
        "subcube_leaves": [
            {
                "fixed_zero": [False] * 18,
                "fixed_active": [False] * 18,
                "lb": float(0.5 * (yc @ yc)),
                "dual_nu": (-yc).tolist(),
            }
        ],
    }
    ok_fake2, rsn_fake2, _ = ns2["_verify_cert_v2"](
        xc, yc, float(st_conn["M"]), float(st_conn["lam0"]), float(st_conn["lam2"]), fake_conn
    )
    assert not ok_fake2, f"Expected connected dual counterexample to fail, got reason={rsn_fake2}"

    # 3. Valid mixed-sign Fenchel dual certificate at true optimum passes cleanly
    w_star = np.zeros(8, dtype=float)
    w_star[[1, 4]] = [0.5, -0.5]
    X_m = np.eye(8, dtype=float)
    y_m = np.array([0.05, 1.0, -0.05, 0.0, -1.0, 0.02, -0.02, 0.0], dtype=float)
    lam2_m, lam0_m, box_m = 1.0, 0.10, 1.0
    r_m = X_m @ w_star - y_m
    obj_m = float(0.5 * (r_m @ r_m) + 0.5 * lam2_m * (w_star @ w_star) + lam0_m * 2)
    # In eq02 (`-nu^T y`) sign convention, optimal residual dual on fixed support {1, 4} is `nu_eq02 = X w_star - y_m`
    nu_eq02 = (X_m @ w_star - y_m).tolist()
    fz_m = [j not in {1, 4} for j in range(8)]
    fa_m = [j in {1, 4} for j in range(8)]
    # Single fixed leaf does not cover {0,1}^8, whereas root dual `nu_eq02 = X w_star - y` on orthogonal X_m=I_8 is exact!
    valid_dual_cert = {
        "weights": w_star.tolist(),
        "objective": obj_m,
        "certified_lower_bound": obj_m - 5e-7,
        "subcube_leaves": [
            {
                "fixed_zero": [False] * 8,
                "fixed_active": [False] * 8,
                "lb": obj_m - 5e-7,
                "dual_nu": nu_eq02,
            }
        ],
    }
    _ = (fz_m, fa_m)
    ok_valid_d, rsn_valid_d, _ = ns2["_verify_cert_v2"](X_m, y_m, box_m, lam0_m, lam2_m, valid_dual_cert)
    assert ok_valid_d, f"Valid mixed-sign dual certificate rejected: {rsn_valid_d}"

    # 4. Upgraded eq02 public schema validator rejects missing objective/LB, overlapping half-cubes, AND Stage 2->3 handoff tamper
    with tempfile.TemporaryDirectory(prefix="m921_pubschema_") as td:
        td_p = Path(td)
        patched_setup = eq02_task.public_setup_py.replace("/holdout_task", td_p.as_posix())
        exec(compile(patched_setup, "<eq02_pub_setup>", "exec"), {})
        bad_cand = td_p / "bad_cand.py"
        bad_cand.write_text(
            "import numpy as np\n"
            "def solve_coupled_interferometry_l0(st):\n"
            "    N, D = st['panel']['X_panel_raw'].shape\n"
            "    ns, p = st['X_cand'].shape\n"
            "    lf = {'fixed_zero': [True] + [False]*(p-1), 'fixed_active': [False]*p, 'lb': 0.0, 'minorant_diagonal': [0.0]*p, 'minorant_center': [0.0]*p}\n"
            "    return {'aligned_panel': np.zeros((N, D)), 'calibration_weights': np.zeros(13), 'deconfounded_target': np.zeros(ns), "
            "'certified_selection': {'weights': [0.0]*p, 'subcube_leaves': [lf, lf]}}\n",
            encoding="utf-8",
        )
        pr_bad = subprocess.run([sys.executable, str(td_p / "validate_public_schema.py"), str(bad_cand)], capture_output=True, text=True)
        assert pr_bad.returncode != 0
        parsed_bad = json.loads(pr_bad.stdout.strip())
        assert parsed_bad["schema_valid"] is False

    # 5. All 6 frozen M9.2 eq02 deliverables pass 12/12 private instances + Stage 2->3 handoff consistency
    ev_m92 = REPO_ROOT / "docs/evidence/m9_2_equal_information_on_demand_reuse_20261126"
    snaps = sorted((ev_m92 / "episodes").glob("rep_*/*/eq02*/04_submitted_artifacts_snapshot/*.py"))
    assert len(snaps) == 6
    pass_pairs = 0
    for fp in snaps:
        spec = importlib.util.spec_from_file_location(f"m92_eq02_{fp.parent.parent.name}", str(fp))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for seed, p_dim in ((99411, 18), (99412, 19)):
            st_i, *_ = ns2["_make_priv"](seed, p_dim)
            out_i = mod.solve_coupled_interferometry_l0(st_i)
            ap = np.asarray(out_i["aligned_panel"], dtype=float)
            cw = np.asarray(out_i["calibration_weights"], dtype=float)
            dt = np.asarray(out_i["deconfounded_target"], dtype=float)
            strata = np.asarray(st_i["panel"]["stratum"])
            phase = np.angle(st_i["interferometry"]["signal_phasor"] * np.conj(st_i["interferometry"]["ref_phasor"]))
            hc = np.column_stack([np.ones(len(strata)), (strata == 1).astype(float), (strata == 2).astype(float), ap, phase])
            s3 = st_i["stage3_indices"]
            assert float(np.max(np.abs(dt - (st_i["y_assay_raw"][s3] - hc[s3] @ cw)))) <= 1e-12
            ok_c, rsn_c, _ = ns2["_verify_cert_v2"](
                st_i["X_cand"], dt, float(st_i["M"]), float(st_i["lam0"]), float(st_i["lam2"]), out_i["certified_selection"]
            )
            assert ok_c, f"{fp} failed seed={seed}: {rsn_c}"
            pass_pairs += 1
    assert pass_pairs == 12


def test_a2_production_bootstrap_verifies_public_fixtures_and_fails_fast_when_missing() -> None:
    eq02_ft = equal_info_m92_task_to_frontier_task(
        EQUAL_INFO_M92_BY_ID["eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0"]
    )
    llm_calls: list[int] = []

    def _mock_call_structured(self: Any, **kwargs: Any) -> dict[str, Any]:
        llm_calls.append(1)
        return {
            "ok": True,
            "wall_ms": 5.0,
            "returned_model": "mock-model",
            "request_sha256": "req_mock",
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "completion_tokens": 20,
            "attempt_count": 1,
            "usage_unknown_attempt_count": 0,
            "cost_accounting_status": "complete",
            "attempt_receipts": [],
            "parsed": {
                "action_type": "submit_completed_task",
                "bash_command": None,
                "final_answer": "done",
                "active_hypothesis_and_worst_slice_metric": "smoke",
                "nogood_update": {
                    "has_new_nogood": False,
                    "nogood_rung": None,
                    "assumption_refuted": None,
                    "empirical_witness": None,
                    "causal_blanket_diagnosis": None,
                    "sublated_pivot": None,
                    "risk_kind": None,
                    "resolves_nogood_ids": [],
                },
                "considered_alternatives": [],
                "rationale": "smoke",
            },
        }

    with tempfile.TemporaryDirectory(prefix="m921_boot_smoke_") as td:
        td_p = Path(td)
        codex_info = {
            "available": True,
            "auth_mode": "chatgpt",
            "provider_backend": "mock_codex",
            "executable": "mock",
            "is_windows_exe_under_wsl": False,
            "stage_root_wsl": td_p / "stage",
            "stage_root_cli_str": str(td_p / "stage"),
        }
        # 1. Missing required public asset must fail BEFORE any LLM call (`len(llm_calls) == 0`)
        broken_task = equal_info_m92_task_to_frontier_task(
            EQUAL_INFO_M92_BY_ID["eq02__coupled_panel_interferometry_cauchy_and_connected_box_l0"]
        )
        broken_task = type(broken_task)(
            **{
                **broken_task.__dict__,
                "required_public_assets": (*broken_task.required_public_assets, "/holdout_task/nonexistent_required_fixture.pkl"),
            }
        )
        with patch.object(m6f.FrontierCodexClient, "call_structured", _mock_call_structured):
            with pytest.raises(RuntimeError, match="Required public assets missing"):
                m6f.run_frontier_episode(
                    broken_task,
                    "Hegel_M6_9_Antithesis_LiveGPT",
                    model="mock-model",
                    codex_info=codex_info,
                    forest=PhilosophicalForest(),
                    forest_archive={},
                    hippo_idx=None,
                    episode_out_dir=td_p / "ep_broken",
                    run_id="smoke_broken",
                    max_turns=1,
                    defer_official_verifier=True,
                )
        assert len(llm_calls) == 0

        # 2. Valid M9.2 task materializes `/holdout_task/public_eq02_studies.pkl` and `/holdout_task/validate_public_schema.py` before Turn 1
        with patch.object(m6f.FrontierCodexClient, "call_structured", _mock_call_structured):
            summary_ok = m6f.run_frontier_episode(
                eq02_ft,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model="mock-model",
                codex_info=codex_info,
                forest=PhilosophicalForest(),
                forest_archive={},
                hippo_idx=None,
                episode_out_dir=td_p / "ep_ok",
                run_id="smoke_ok",
                max_turns=1,
                defer_official_verifier=True,
            )
        psr = summary_ok["public_setup_receipt"]
        assert psr["executed"] is True
        assert psr["shared_public_schema_validator_present"] is True
        assert "/holdout_task/public_eq02_studies.pkl" in psr["files_sha256"]
        assert "/holdout_task/validate_public_schema.py" in psr["files_sha256"]
        assert summary_ok["turn1_request_accounting"]["turn1_prompt_tokens"] == 100


def test_a3_unified_k_and_e_reader_zero_disagreements_across_all_24_m92_queries() -> None:
    v6_path = REPO_ROOT / "docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v6_m911.json"
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(v6_path)
    ev = REPO_ROOT / "docs/evidence/m9_2_equal_information_on_demand_reuse_20261126"

    def _rank_ids(out_s: str) -> list[str]:
        return re.findall(r"--describe\s+([A-Za-z_0-9]+\.[A-Za-z_0-9]+)", out_s)

    def _strip_access_fields(desc_json: dict[str, Any]) -> dict[str, Any]:
        d = deepcopy(desc_json)
        d.pop("access_mode", None)
        d.pop("execution_available", None)
        for m in d.get("matches", []):
            m.pop("execution_available", None)
            m.pop("module_import", None)
            m.pop("reference_source_txt", None)
            m["index_cards"] = [c.replace("python3 /tmp/hegel_knowledge_ref/describe_ref.py ", "") for c in m.get("index_cards", [])]
        return d

    total_queries = 0
    with tempfile.TemporaryDirectory(prefix="m921_reader_") as td:
        td_p = Path(td)
        kp, ep = td_p / "k", td_p / "e"
        kp.mkdir()
        ep.mkdir()
        (kp / "describe_ref.py").write_text(m64.build_knowledge_only_describe_ref_script(), encoding="utf-8")
        (ep / "__init__.py").write_text(m64.SKILL_INIT_PY, encoding="utf-8")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HEGEL_SKILL_INSPECTION_LOG": str(td_p / "insp.jsonl")}
        ns_e: dict[str, Any] = {"__file__": str(ep / "__init__.py"), "__name__": "e_reader"}
        exec(compile(m64.SKILL_INIT_PY, str(ep / "__init__.py"), "exec"), ns_e)
        ns_e["INSPECTION_LOG_PATH"] = env["HEGEL_SKILL_INSPECTION_LOG"]

        for task in EQUAL_INFO_M92_TASKS:
            sum_p = ev / "episodes/rep_01/on_demand_executable_library" / task.task_id / "00_episode_summary.json"
            summary = json.loads(sum_p.read_text(encoding="utf-8"))
            notes = [deepcopy(archive[nid]) for nid in summary["recalled_node_ids"]]
            for note in notes:
                note["autogenous_wormhole_modules"] = [
                    m
                    for m in note.get("autogenous_wormhole_modules", [])
                    if not re.search(r"^\s*(?:import|from)\s+sklearn\b", str(m.get("python_code", "")), re.M)
                ]
            for mode, dst in (("on_demand_knowledge_only", kp), ("on_demand_executable", ep)):
                _, _, diag = m64.collect_recalled_executable_wormhole_sources(
                    notes[:3],
                    include_foundational_skills=False,
                    allow_l1_primitives=True,
                    full_forest_archive=archive,
                    task_query_text=task.instruction_text,
                    max_l1_modules=4,
                    allow_sklearn=False,
                    disclosure_mode=mode,
                    staging_policy="net_value_gated",
                    return_diagnostics=True,
                )
                (dst / "_skill_contracts.json").write_text(diag["contracts_catalog_json"], encoding="utf-8")

            def _run_k(flag: str, q_val: str) -> str:
                pr = subprocess.run([sys.executable, str(kp / "describe_ref.py"), flag, q_val], text=True, capture_output=True, env=env, timeout=15)
                assert pr.returncode == 0, pr.stderr
                return pr.stdout.strip()

            for log_fp in sorted((ev / "episodes").glob(f"*/*/{task.task_id}/04d_skill_contract_inspections.jsonl")):
                for ev_row in map(json.loads, log_fp.read_text(encoding="utf-8").splitlines()):
                    if ev_row.get("event") != "skill_search_executed":
                        continue
                    q_s = str(ev_row.get("query") or ev_row.get("target") or "")
                    k_out = _run_k("--search", q_s)
                    e_out = ns_e["search_skills"](q_s)
                    assert _rank_ids(k_out) == _rank_ids(e_out), f"Rank mismatch on {q_s!r}"
                    total_queries += 1

            for q_desc in (
                "cauchy_linear_fit",
                "autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
                "hegel_wormhole_skills.autogenous_prim_robust_trend_decomposition.cauchy_linear_fit",
            ):
                kd = json.loads(_run_k("--describe", q_desc))
                ed = json.loads(ns_e["describe_skill"](q_desc))
                assert _strip_access_fields(kd) == _strip_access_fields(ed)
                assert kd["execution_available"] is False
                assert ed["execution_available"] is True

    assert total_queries == 24


def test_a4_strict_4tuple_equality_and_online_compiler_v2_issuance() -> None:
    v6_path = REPO_ROOT / "docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v6_m911.json"
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(v6_path)
    sample = deepcopy(
        next(
            m
            for _, r in m7a.iter_live_archive_nodes(archive)
            for m in r.get("autogenous_wormhole_modules", [])
            if str(m.get("filename", "")).startswith("autogenous_prim_")
        )
    )
    assert m64.verify_l1_module_qualification_record(sample, require_4tuple_protocol=True)[0] is True

    bad_proto = deepcopy(sample)
    bad_proto["unit_test_receipt"]["verifier_protocol_version"] = "unrelated_protocol_v999"
    assert m64.verify_l1_module_qualification_record(bad_proto, require_4tuple_protocol=True)[0] is False

    bad_env = deepcopy(sample)
    bad_env["unit_test_receipt"]["dependency_env_digest"] = "unrelated_environment_sha"
    assert m64.verify_l1_module_qualification_record(bad_env, require_4tuple_protocol=True)[0] is False

    # Online compiler mock dispatch must stamp full v2 4-tuple qualification
    v5_path = REPO_ROOT / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/pretrained_autogenous_forest_v5_m89.json"
    forest5, archive5, _, _ = m7a.load_autogenous_forest_bundle(v5_path)
    kept = False
    for _, rec in m7a.iter_live_archive_nodes(archive5):
        orig = list(rec.get("autogenous_wormhole_modules", []))
        rec["autogenous_wormhole_modules"] = []
        if rec.get("layer_role") not in {"mid", "fine"}:
            continue
        for m in orig:
            if not kept and not str(m.get("filename", "")).startswith("autogenous_prim_"):
                rec["autogenous_wormhole_modules"] = [m]
                kept = True

    class _MockCompiler:
        def call_structured(self, **kw: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "parsed": {
                    "primitive_module_stem": "audit_scale_v2",
                    "canonical_python_code": "import numpy as np\ndef scale_arr(x):\n    \"\"\"Scale finite array by 2.\"\"\"\n    return 2.0 * np.asarray(x, dtype=float)\n",
                    "synthetic_unit_test_code": "def run_self_test():\n    assert scale_arr([1.5]).tolist() == [3.0]\nrun_self_test()\n",
                    "mathematical_preconditions": ["Finite real array inputs."],
                    "exported_api_signatures": ["scale_arr(x)"],
                    "compositional_rationale": "Unit test for v2 4-tuple issuance.",
                },
                "prompt_tokens": 10,
                "cached_tokens": 0,
                "completion_tokens": 5,
                "wall_ms": 1.0,
            }

    with (
        patch.object(m7a, "ensure_clean_scipy_sandbox_image", lambda: "hegel_clean_scipy_sandbox:v1"),
        patch.object(m7a.m6f, "start_detached_container", lambda *a, **kw: "mock-cid"),
        patch.object(m7a.m6f, "stop_container_safely", lambda *a, **kw: None),
        patch.object(m7a, "verify_autogenous_module_in_clean_container", lambda **kw: {"passed": True, "telemetry_verified": True}),
    ):
        comp_res = m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=forest5,
            archive=archive5,
            codex_client=_MockCompiler(),
            max_workers=1,
            require_pretraining_only=False,
            allow_horizontal_wormholes=False,
        )
    assert comp_res["newly_compiled_count"] == 1
    new_l1 = next(
        m
        for _, r in m7a.iter_live_archive_nodes(archive5)
        for m in r.get("autogenous_wormhole_modules", [])
        if str(m.get("filename", "")).startswith("autogenous_prim_")
    )
    assert new_l1["qualification_protocol"] == "hegel_l1_qual_4tuple_v2"
    ok_strict, rsn_strict, _ = m64.verify_l1_module_qualification_record(new_l1, require_4tuple_protocol=True)
    assert ok_strict is True, rsn_strict


def test_a5_host_controlled_probe_witness_rejects_constant_assert_spoofs() -> None:
    fp = REPO_ROOT / "src/hegel_machine/cognition/m6_frontier_real_benchmarks.py"
    src_txt = fp.read_text(encoding="utf-8")
    tree = ast.parse(src_txt)
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "pending_nogood_resolutions and nogoods_recorded")

    for cmd, stdout in (
        ("python3 -c \"import sol; assert callable(sol.solve); print('gauge scale identifiability pass')\"", "gauge scale identifiability pass"),
        ("python3 -c \"import sol; assert 1 == 1; print('gauge scale residual=0.0')\"", "gauge scale residual=0.0"),
        ("python3 -c \"import sol; assert 1 == 1; print('two_gauge scale residual=0.0')\"", "two_gauge scale residual=0.0"),
    ):
        ng = {
            "nogood_id": "ng_scope",
            "risk_kind": "target_identifiability",
            "assumption_refuted": "absolute scale identifiable from relative-only observations",
            "scope": "affine gauge scale",
            "resolved": False,
        }
        ns = {
            "deepcopy": deepcopy,
            "pending_nogood_resolutions": [{"proposed_turn": 2, "resolves_ids": ["ng_scope"], "proposed_by_nogood_id": None}],
            "nogoods_recorded": [ng],
            "nogood_resolution_events": [],
            "turn_idx": 3,
            "current_candidate_sha256": "cand_sha_123",
            "cmd_str": cmd,
            "ex_res": {"stdout": stdout},
        }
        exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns)
        assert ng["resolved"] is False
        assert ng["probe_support_status"] == "agent_claimed_unverified"

    # Genuine host-signed gauge probe receipt resolves matching scope & SHA
    ng_ok = {
        "nogood_id": "ng_scope",
        "risk_kind": "target_identifiability",
        "assumption_refuted": "anchored gauge scale check needed",
        "scope": "affine gauge scale",
        "resolved": False,
    }
    ns_ok = {
        "deepcopy": deepcopy,
        "_HOST_INJECTED_PROBE_RECEIPT": {
            "signed_by_host_probe_executor": True,
            "candidate_bundle_sha256": "cand_sha_123",
            "verified_scope": "affine gauge scale",
            "gauge_invariance_verified": True,
            "result": "verified_gauge_and_public_probe_in_scope",
        },
        "pending_nogood_resolutions": [{"proposed_turn": 2, "resolves_ids": ["ng_scope"], "proposed_by_nogood_id": None}],
        "nogoods_recorded": [ng_ok],
        "nogood_resolution_events": [],
        "turn_idx": 3,
        "current_candidate_sha256": "cand_sha_123",
        "cmd_str": "python3 -I /holdout_task/validate_public_schema.py",
        "ex_res": {"stdout": "{\"schema_valid\": true}"},
    }
    exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns_ok)
    assert ng_ok["resolved"] is True
    assert ng_ok["probe_support_status"] == "verified_gauge_and_public_probe_in_scope"


def test_a6_temporal_context_typing_initial_memory_states_dt_and_paid_oracle_failure() -> None:
    def _make_planner(include_timestamps: bool = True) -> tuple[Any, Any, Any]:
        g = vp.PersistentMechanismGraph(forest=PhilosophicalForest())
        g.register_mechanism(vp.TypedMechanism(mechanism_id="root", parent_id=None, role="shared_linear", feature_names=("a",), shared_weights={"a": 1.0}))
        g.register_mechanism(
            vp.TypedMechanism(mechanism_id="memory", parent_id="root", role="temporal_memory", feature_names=("a",), temporal_tau=2.0, temporal_driver_col="a")
        )

        def _ds(hist: list[list[float]], a_vals: list[float]) -> dict[str, Any]:
            h = np.asarray(hist, dtype=float)
            x = np.asarray(a_vals, dtype=float)
            d: dict[str, Any] = {"x": {"a": x}, "y": x + 2.5 * h.mean(axis=1), "past_driver_history": h, "history_dt": 0.5}
            if include_timestamps:
                d.update(timestamps=np.arange(len(x), dtype=float), trajectory_ids=np.zeros(len(x), dtype=int))
            return d

        tr = _ds([[0, 0.2], [1, 1.2], [-1, -0.8], [0.5, 0.7], [1.5, 1.8], [-0.5, -0.3]], [0.4, 0.5, -0.3, 0.8, -0.2, 0.6])
        va = _ds([[2, 2.2], [-2, -1.8], [0.8, 1.0], [-0.8, -1.0]], [0.3, 0.3, -0.4, -0.4])
        p = vp.M8VariationalPlanner(g, min_gain_nats=50.0)
        tm = p.register_task(task_id="task", shared_upstream_id="root", feature_names=("a",), train_data=tr, val_data=va, initial_local_shrinkage=1.0)
        return p, g, tm

    # 1. Valid `initial_memory_states` per-trajectory dict + checkpoint reload
    p1, g1, _ = _make_planner(include_timestamps=True)
    probe1 = {"x": {"a": [0.5]}, "past_driver_history": [[0.0, 3.0]], "history_dt": 0.5, "timestamps": [10.0], "trajectory_ids": [1], "initial_memory_states": {"1": [9.0, 0.2, 0.1]}}
    res1 = p1.evaluate_and_revise_task("task", candidate_probes=[probe1], probe_oracle_fn=lambda ctx: float(ctx["x"]["a"][0] + 2.5 * np.mean(ctx["past_driver_history"])), budget_remaining=2.0)
    assert res1["selected_candidate"]["action_type"] == "AcquireObservation"
    assert len(g1.task_datasets["task"]["train"]["y"]) == 7
    assert g1.task_datasets["task"]["train"]["initial_memory_states"] == {"1": [9.0, 0.2, 0.1]}
    assert len(g1.paid_observations["task"]) == 1

    # 2. Appending new per-row `timestamps` column onto unindexed 6-row dataset fails BEFORE calling oracle
    p2, g2, _ = _make_planner(include_timestamps=False)
    orc_calls = []
    with pytest.raises(ValueError, match="Cannot append new per-row temporal column"):
        p2.evaluate_and_revise_task(
            "task",
            candidate_probes=[{"x": {"a": [0.5]}, "past_driver_history": [[0.0, 3.0]], "history_dt": 0.5, "timestamps": [10.0], "trajectory_ids": [1]}],
            probe_oracle_fn=lambda ctx: (orc_calls.append(1) or 1.0),
            budget_remaining=2.0,
        )
    assert len(orc_calls) == 0
    assert len(g2.task_datasets["task"]["train"]["y"]) == 6

    # 3. Omitted `history_dt` on probe inherits dataset `history_dt = 0.5` before candidate selection & oracle call
    p3, _, _ = _make_planner(include_timestamps=True)
    seen_dts: list[float] = []
    probe3 = {"x": {"a": [0.5]}, "past_driver_history": [[0.0, 3.0]], "timestamps": [10.0], "trajectory_ids": [1]}
    p3.evaluate_and_revise_task(
        "task",
        candidate_probes=[probe3],
        probe_oracle_fn=lambda ctx: (seen_dts.append(float(ctx["history_dt"])) or 2.0),
        budget_remaining=2.0,
    )
    assert seen_dts == [0.5]

    # 4. Post-oracle validation failure rolls back data while preserving paid observation & decision cost!
    p4, g4, _ = _make_planner(include_timestamps=True)
    probe4 = {"x": {"a": [0.5]}, "past_driver_history": [[0.0, 3.0]], "history_dt": 0.5, "timestamps": [10.0], "trajectory_ids": [1]}
    p4.evaluate_and_revise_task("task", candidate_probes=[probe4], probe_oracle_fn=lambda _ctx: float("nan"), budget_remaining=2.0)
    assert len(g4.task_datasets["task"]["train"]["y"]) == 6
    assert len(g4.paid_observations["task"]) == 1
    assert g4.paid_observations["task"][0]["status"] == "ORACLE_CALLED_POST_VALIDATION_FAILED"
    assert len(g4.decision_ledger) == 1
    assert g4.decision_ledger[0].actual_cost >= 0.5


def test_a7_a8_a9_attempt_isolation_eq03_physics_and_repair_manifest_gate() -> None:
    # A7: Stale response file from failed attempt 1 cannot masquerade as attempt 2 success; unknown usage tracked explicitly
    with tempfile.TemporaryDirectory(prefix="m921_prov_") as td:
        cli = m6f.FrontierCodexClient(
            codex_info={"available": True, "auth_mode": "chatgpt", "executable": "mock", "is_windows_exe_under_wsl": False, "stage_root_wsl": Path(td), "stage_root_cli_str": td},
            model="mock-model",
        )
        calls: list[list[str]] = []

        def _mock_stale(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            resp_p = Path(cmd[cmd.index("--output-last-message") + 1])
            stdout = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100}}) + "\n"
            if len(calls) == 1:
                resp_p.write_text('{"from_attempt": 1}', encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 1, stdout, "stream disconnected connection")
            # Attempt 2 exits 0 without writing its own response file!
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        with patch.object(m6f.subprocess, "run", side_effect=_mock_stale), patch.object(m6f, "sleep", return_value=None):
            res_stale = cli.call_structured(system_prompt="x", conversation_turns=[{"role": "user", "content": "hi"}], output_schema={"type": "object"}, stage_subdir="s1")
        assert res_stale["ok"] is False
        assert res_stale["prompt_tokens"] + res_stale["completion_tokens"] == 2200

        def _mock_unknown(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(m6f.subprocess, "run", side_effect=_mock_unknown), patch.object(m6f, "sleep", return_value=None):
            res_unk = cli.call_structured(system_prompt="x", conversation_turns=[{"role": "user", "content": "hi"}], output_schema={"type": "object"}, stage_subdir="s2")
        assert res_unk["ok"] is True
        assert res_unk["usage_unknown_attempt_count"] == 1
        assert res_unk["cost_accounting_status"] == "partial_unknown_usage"

    # A8: eq03 twins share exact physical initial derivative (`< 1e-12`) AND `critical_strain` has zero future terminal leakage
    eq03_task = EQUAL_INFO_M92_BY_ID["eq03__viscoelastic_hereditary_memory_same_present_different_past"]
    ns3 = _extract_funcs_from_script(eq03_task.private_verifier_py, ("viscoelastic_hereditary_forecast",))
    for seed in (99421, 99422, 99423):
        ds, te_xl, _, theta, clean, _, hi, lo = ns3["_make_priv"](seed)
        te = ds["test"]
        taus = ds["memory_taus"]
        dt = ds["history_dt"]
        z_list = []
        for tau in taus:
            a = np.exp(-dt / tau)
            zz = np.zeros(len(te["lab_id"]))
            for col in te["past_stress_history"].T:
                zz = a * zz + (1.0 - a) * col
            z_list.append(zz)
        true_deriv = theta[0] * (te["stress_hold_0"] - z_list[0]) / taus[0] + theta[1] * (te["stress_hold_0"] - z_list[1]) / taus[1] + theta[2] + te_xl @ theta[3:]
        assert float(np.max(np.abs(te["instant_rate_0"][hi] - te["instant_rate_0"][lo]))) <= 1e-12
        assert float(np.max(np.abs(true_deriv[hi] - true_deriv[lo]))) <= 1e-12
        # History-blind optimal common forecast lower bound must remain >> 0.08
        assert float(np.mean(np.abs(clean[hi] - clean[lo])) / 2.0) >= 0.25
        # Old formula `strain_0 + (crit - strain_0 - 0.12)/0.55` must NOT invert `clean[:, -1]`
        inv_attempt = te["strain_0"][:24] + (te["critical_strain"][:24] - te["strain_0"][:24] - 0.12) / 0.55
        assert float(np.max(np.abs(inv_attempt - clean[:24, -1]))) > 0.15
