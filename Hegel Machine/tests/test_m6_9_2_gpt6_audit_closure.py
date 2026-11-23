"""Regression & architectural closure tests for Hegel M6.9.2 (`Hegel_M691_Audit` response).

Verifies every concrete bug fix (`E01`-`E04`, `K01`-`K05`, `A01`-`A03`, `L01`-`L02`, `D01`) AND
the two high-leverage architectural upgrades (real Python scientific AST -> `Program` subtree anti-unification
and uncertainty-triggered selective second-branch routing) without any premature defensive exploration narrowing.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np

from hegel_machine.cognition import (
    m6_3_sovereign_causal_wormhole_engine as tracka,
    m6_4_executable_wormhole_library as lib,
    m6_8_b_prime_capability_bridge as bridge,
    m6_9_a_dialectical_synthesis_engine as synth,
    m6_frontier_real_benchmarks as m6f,
)


def _json_sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def test_e01_qualify_staged_wormhole_modules_direct_argv_without_bash_syntax_error() -> None:
    """`E01`: `qualify_staged_wormhole_modules_in_container` must pass direct `python3 -c` argv (zero `bash -lc` `SyntaxError`)."""
    orig_run = subprocess.run

    with tempfile.TemporaryDirectory(prefix="m692_e01_") as td:
        td_p = Path(td)
        pkg_dir = td_p / "hegel_wormhole_skills"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(lib.SKILL_INIT_PY, encoding="utf-8")
        (pkg_dir / "multiscale_and_reconciliation_wormhole.py").write_text(
            bridge.SKILL_07_MULTISCALE_RECONCILIATION, encoding="utf-8"
        )

        def local_docker_exec(cmd, **kwargs):
            assert "bash" not in cmd and "-lc" not in cmd, f"Must not use bash -lc: {cmd}"
            py_idx = cmd.index("python3")
            env = {**kwargs.get("env", {}), "PYTHONPATH": str(td_p), "HEGEL_SKILL_CALLER_ROLE": "smoke_test"}
            return orig_run([sys.executable, *cmd[py_idx + 1 :]], env=env, **{k: v for k, v in kwargs.items() if k != "env"})

        with patch.object(bridge.subprocess, "run", local_docker_exec):
            res = bridge.qualify_staged_wormhole_modules_in_container(
                "cid_local",
                [{"module_import": "hegel_wormhole_skills.multiscale_and_reconciliation_wormhole"}],
            )
        assert len(res) == 1
        assert res[0]["lifecycle_state"] == "bound_and_import_qualified"
        assert res[0]["import_qualified"] is True
        assert res[0]["contract_validated"] is False
        assert res[0]["callable_count"] >= 2


def test_e02_symbolic_public_benchmark_supports_3arg_predict_abi_and_sentinel_coverage() -> None:
    """`E02`: `PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY` must invoke `regressor.predict(X_tr, y_tr, X_va)` and reject unsupported ABIs."""
    bench_tree = ast.parse(synth.PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY)
    sr_assign = next(
        n
        for n in ast.walk(bench_tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sr_code" for t in n.targets)
    )
    sr_code = ast.literal_eval(sr_assign.value)

    with tempfile.TemporaryDirectory(prefix="m692_e02_") as td:
        td_p = Path(td)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 4))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        np.savez(td_p / "training_data.npz", X=X, y=y)

        # 1. Real task ABI `predict(X_train, y_train, X_test)` must be called and score ~1.0!
        (td_p / "regressor.py").write_text(
            "import numpy as np\n"
            "def predict(X_train, y_train, X_test):\n"
            "    return (X_test[:, 0] * X_test[:, 1] > 0).astype(float)\n",
            encoding="utf-8",
        )
        p1 = subprocess.run([sys.executable, "-c", sr_code.replace("/app", str(td_p))], capture_output=True, text=True, timeout=20)
        assert p1.returncode == 0
        j1 = json.loads(p1.stdout.strip().splitlines()[-1])
        assert j1["status"] == "ok:predict_3arg"
        assert j1["oof_f1"] > 0.98

        # 2. Unsupported interface must return `unsupported_interface` and `oof_f1=None` (NEVER uninitialized memory!)
        (td_p / "regressor.py").write_text("def unrelated_helper(z):\n    return z\n", encoding="utf-8")
        p2 = subprocess.run([sys.executable, "-c", sr_code.replace("/app", str(td_p))], capture_output=True, text=True, timeout=20)
        assert p2.returncode == 0
        j2 = json.loads(p2.stdout.strip().splitlines()[-1])
        assert j2["status"] == "unsupported_interface"
        assert j2["oof_f1"] is None


def test_e03_symbolic_host_probe_univariate_term_and_sparse_cert_real_schema() -> None:
    """`E03`: `_probe_symbolic_predictor` must handle `h['term'] = ('linear', col)` without `KeyError: 'col'`,
    and `certified-sparse` public evaluator must validate real `{'support', 'coefficients', 'nodes'}` certificates!
    """
    with tempfile.TemporaryDirectory(prefix="m692_e03_sym_") as td:
        td_p = Path(td)
        model_fp = td_p / "regressor.py"
        model_fp.write_text("def predict(X_tr, y_tr, X_te):\n    return (X_te[:, 0] > 0).astype(float)\n", encoding="utf-8")
        rng = np.random.default_rng(31415)
        X = rng.normal(size=(120, 3))
        y = (X[:, 0] > 0).astype(int)
        data_fp = td_p / "training_data.npz"
        np.savez(data_fp, X=X, y=y)

        scns = {"track_wormhole_skill": lambda _: lambda fn: fn}
        exec(lib.SKILL_01_SYMBOLIC_DYNAMICAL.replace("from . import track_wormhole_skill", ""), scns)
        modname = "hegel_wormhole_skills.symbolic_and_dynamical_wormhole"
        fake_mod = types.ModuleType(modname)
        fake_mod.null_calibrated_symbolic_scanner = scns["null_calibrated_symbolic_scanner"]  # type: ignore[attr-defined]
        runner_ns = {"__name__": "m692_test"}
        exec(lib.HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY, runner_ns)
        runner_ns["_find_public_files"] = lambda _pat: [str(data_fp)]
        reports: list[str] = []
        with patch.dict(sys.modules, {modname: fake_mod}):
            runner_ns["_probe_symbolic_predictor"]([str(model_fp)], reports)
        assert len(reports) == 1
        assert "SYMBOLIC_NESTED_CV_AUDIT" in reports[0]
        assert "KeyError" not in reports[0]

    # Test real `{support, coefficients, nodes}` certified-sparse evaluation in `PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY`
    with tempfile.TemporaryDirectory(prefix="m692_e03_sparse_") as td_sp:
        sp_p = Path(td_sp)
        (sp_p / "data").mkdir()
        X_s = np.eye(3, dtype=float)
        y_s = np.array([2.0, 0.0, 0.0], dtype=float)
        np.save(sp_p / "data" / "X.npy", X_s)
        np.save(sp_p / "data" / "y.npy", y_s)
        (sp_p / "data" / "problem.json").write_text(
            json.dumps({"lambda_0": 0.1, "lambda_2": 0.05, "M": 5.0, "epsilon": 1e-4, "max_nodes": 100}),
            encoding="utf-8",
        )
        # Complete 2-leaf partition of `{0,1}^3` splitting on coordinate 0 (`B0=[0]` and `B1=[0]`)
        (sp_p / "results.json").write_text(
            json.dumps({"support": [0], "coefficients": [1.8], "nodes": [{"B0": [0], "B1": []}, {"B0": [], "B1": [0]}]}),
            encoding="utf-8",
        )
        bench_patched = synth.PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY.replace("/app", str(sp_p))
        pr = subprocess.run([sys.executable, "-c", bench_patched], capture_output=True, text=True, timeout=20)
        assert pr.returncode == 0
        out = json.loads(pr.stdout.strip().splitlines()[-1])
        assert out["sparse_cert_valid"] is True
        assert out["sparse_cert_nodes"] == 2
        assert out["sparse_cert_status"] == "verified_primal_and_dyadic_partition"


def test_e04_contract_validator_digest_recomputation_import_entrypoint_nan_and_verify_cmd() -> None:
    """`E04`: Contract validator must recompute container bytes SHA-256, run isolated import + entrypoint check,
    reject `NaN` in CSV/JSON, and `is_substantive_verification_command` must reject `import math; print('solver.py')`!
    """
    assert m6f.is_substantive_verification_command("python3 -c \"import math; print('solver.py')\"", ("/app/solver.py",)) is False
    assert (
        m6f.is_substantive_verification_command(
            "python3 -c \"from pathlib import Path; Path('/app/solver.py').write_text('raise RuntimeError(1)')\"",
            ("/app/solver.py",),
        )
        is False
    )
    assert m6f.is_substantive_verification_command("python3 -c \"import solver; print(solver.predict([1]))\"", ("/app/solver.py",)) is True

    orig_run = subprocess.run

    def local_validator(cmd, **kwargs):
        assert cmd[:4] == ["docker", "exec", "-w", "/"]
        assert cmd[5:7] == ["python3", "-c"]
        return orig_run([sys.executable, "-c", cmd[7], cmd[8]], **kwargs)

    bad_cases = [
        ("import_crashes", "solver.py", "def predict(x):\n    return x\nraise RuntimeError('boom')\n"),
        ("missing_dep", "solver.py", "import nonexistent_hegel_audit_dep_xyz\ndef predict(x):\n    return x\n"),
        ("no_entrypoint", "solver.py", "answer = 42\n"),
        ("nan_csv", "results.csv", "sample,value\ns1,nan\n"),
        ("nan_json", "results.json", '{"value": NaN}\n'),
    ]
    for label, fname, body in bad_cases:
        with tempfile.TemporaryDirectory(prefix=f"m692_e04_{label}_") as td:
            fp = Path(td) / fname
            fp.write_text(body, encoding="utf-8")
            b = fp.read_bytes()
            real_sha = hashlib.sha256(str(fp).encode("utf-8") + b"\0" + hashlib.sha256(b).digest()).hexdigest()
            with patch.object(bridge.subprocess, "run", local_validator):
                rec = bridge.verify_candidate_artifact_contract_in_container("local", (str(fp),), real_sha)
            assert rec["accepted"] is False, f"{label} should have been rejected: {rec}"

    # Valid Python candidate passes with matching SHA-256 and FAILS with forged SHA-256!
    with tempfile.TemporaryDirectory(prefix="m692_e04_valid_") as td:
        fp = Path(td) / "solver.py"
        fp.write_text("def predict(x):\n    return float(x) * 2.0\n", encoding="utf-8")
        b = fp.read_bytes()
        real_sha = hashlib.sha256(str(fp).encode("utf-8") + b"\0" + hashlib.sha256(b).digest()).hexdigest()
        with patch.object(bridge.subprocess, "run", local_validator):
            ok_rec = bridge.verify_candidate_artifact_contract_in_container("local", (str(fp),), real_sha)
            bad_sha_rec = bridge.verify_candidate_artifact_contract_in_container("local", (str(fp),), "0" * 64)
        assert ok_rec["accepted"] is True and ok_rec["digest_verified"] is True
        assert bad_sha_rec["accepted"] is False and any("candidate_sha256_mismatch" in r for r in bad_sha_rec["reasons"])


def test_k02_real_ast_to_program_lifting_distinguishes_linear_vs_sinusoidal_laws() -> None:
    """`K02` Architectural Upgrade: `SharedBaseSubtree` must lift actual Python mathematical ASTs so changing
    `a*x + 1` to `np.sin(a*x*x)` produces a genuinely distinct anti-unified operator skeleton!
    """
    subtree_hashes = []
    subtree_ops = []
    for variant in (0, 1):
        forest, archive, _ = m6f.build_m6_frontier_general_forest()
        with tempfile.TemporaryDirectory(prefix="m692_k02_") as td:
            pairs = []
            for i, fam in enumerate(["fam_symbolic_and_dynamical_laws", "fam_robust_control_and_calibration"]):
                ep = Path(td) / f"ep_{i}"
                (ep / "04b_producer_skill_manifest").mkdir(parents=True)
                code = (
                    f"def learned_law(x):\n    return {i + 2}.0 * x + 1.0\n"
                    if variant == 0
                    else f"def learned_law(x):\n    return np.sin({i + 2}.0 * x * x)\n"
                )
                man = {
                    "temporal": {"temporal_readout_mode": "mori_zwanzig_or_stage_history"},
                    "structure": {"full_producer_count": 1, "reusable_ast_function_closures": [{"name": "learned_law", "code": code}]},
                }
                (ep / "04b_producer_skill_manifest" / "executable_skill_manifest.json").write_text(json.dumps(man), encoding="utf-8")
                pairs.append(({"task_id": f"t_{i}", "short_name": f"t_{i}", "wormhole_family": fam, "strict_pass": True, "benchmark": "TB"}, ep))
            res = bridge.consolidate_wave1_skill_manifests_with_reconciliation(wave1_episode_dirs=pairs, forest=forest, archive=archive)
            assert len(res["registered_shared_base_subtrees"]) == 1
            sub_dict = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in forest.shared_base_subtrees.items()}
            subtree_hashes.append(_json_sha256(sub_dict))
            rec_first = next(iter(sub_dict.values()))
            subtree_ops.append(rec_first["shared_nodes"][0]["root_op"])

    assert subtree_hashes[0] != subtree_hashes[1]
    assert subtree_ops == ["add", "sin"]


def test_k04_l01_l02_pi_fold_cutoff_missing_slice_np_load_and_shadow_bins() -> None:
    """`K04`, `L01`, `L02`: Explicit `cutoff_t0` never leaks future points, missing slices block challenger,
    `np.load` is rejected from pure ASTs, and `y = -z_1` vs `y = +z_1` fails passive shadow consistency!
    """
    ns = {"track_wormhole_skill": lambda _: lambda fn: fn}
    exec(bridge.SKILL_07_MULTISCALE_RECONCILIATION.replace("from . import track_wormhole_skill", ""), ns)

    # K04: Explicit `cutoff_t0=0.0` with only 1 prefix point returns `insufficient_causal_history`
    t = np.arange(10, dtype=float)
    r_bad = ns["pi_fold_temporal_regime_selector"](t, np.sin(t), np.cos(t), cutoff_t0=0.0)
    assert r_bad["status"] == "insufficient_causal_history"
    assert r_bad["prefix_count"] == 1

    # Valid cutoff returns fitted parameters for downstream consumption
    r_ok = ns["pi_fold_temporal_regime_selector"](t, np.sin(t), np.cos(t), cutoff_t0=6.0)
    assert r_ok["status"] == "ok" and "fitted_parameters" in r_ok

    # L01: Missing minority slice is rejected
    adj = ns["adjudicate_incumbent_vs_challenger_slices"]({"majority": 1.0, "minority": 0.1}, {"majority": 0.4})
    assert adj["decision"] == "keep_incumbent"
    assert adj["reason"] == "missing_required_slices"

    # L02a: `np.load(path)` is rejected from pure function extraction
    assert lib.extract_pure_reusable_functions_from_python("def reader(p):\n    return np.load(p)\n") == []

    # L02b: Symmetric `y = -z_1` vs `y = +z_1` on `[-2,-1,0,1,2]` is detected by >=2 bins + pointwise RMSE!
    forest, archive = tracka.build_m6_1_clean_k40_forest()
    for v in archive.values():
        v["verified_mechanism_note"] = "old law y = -z_1"
        v.pop("historical_L1_shadow", None)
    rows = [{"z_1": float(x), "y": float(x)} for x in [-2, -1, 0, 1, 2]]
    rec = {
        "status": "confirmed_ok",
        "strict_pass": True,
        "task_id": "sym_test",
        "winning_python_code": "def predict(row):\n    return row['z_1']\n",
        "confirm_rmse": 0.0,
        "recalled_node_ids": ["node_C01_TreeA_prime_electro"],
        "observed_signatures": ["sig:c01_treeAprime_electro"],
        "violated_root_ids": [],
        "pilot_rows_snapshot": rows,
    }
    cres = tracka.consolidate_wave1_sovereign_m6_3(
        wave1_receipts=[rec], arm="Hegel_M6_3_Sovereign_LiveGPT", forest=forest, node_archive=archive
    )
    proj = cres["events"][0]["empirical_retroactive_projection"]
    assert proj["passive_shadow_consistent"] is False
    assert proj["max_bin_projection_defect"] > 1.5


def test_a02_a03_auditor_catches_all_4_mutations_and_digest_always_starts_at_turn() -> None:
    """`A02`, `A03`, `Q01`: Long-turn digest never slices mid-turn, selective router triggers only on uncertainty,
    and hardened `audit_m6_4_executable_wormhole_bundle.py` rejects `AUD01`-`AUD04`!
    """
    with tempfile.TemporaryDirectory(prefix="m692_dig_") as td:
        ep = Path(td)
        (ep / "02_human_readable_trajectory.md").write_text(
            "# Head\n\n### Turn 1 — huge output\n" + ("X" * 19000) + "\n### Turn 2 — done\nOK\n## 4. Official Verifier\nSECRET\n",
            encoding="utf-8",
        )
        d = json.loads(synth._extract_trajectory_evidence_digest(ep, {"arm": "Hegel_M6_9_Antithesis_LiveGPT"}, max_chars=14000))
        assert "arm" not in d
        assert d["trajectory_tail_public_evidence"].startswith("### Turn ")
        assert "SECRET" not in d["trajectory_tail_public_evidence"]

    # Selective second-branch router
    tb_task = m6f.FrontierBenchmarkTask(
        task_id="tb__virtual-baseline-localization",
        short_name="virtual-baseline-localization",
        benchmark="Terminal-Bench-Science-0.1",
        wave="wave_2",
        scientific_domain="SHM",
        wormhole_family="fam_shm",
        instruction_text="Localize within 20 mm.",
        agent_image_tag="mock",
        verifier_image_tag="mock",
        declared_artifacts=("/app/solution.py",),
    )
    clean_route = synth.should_spawn_dialectical_second_branch(
        task=tb_task,
        primary_summary={"ended_by_finish": True, "pullback_gate_triggered": False},
        contract_check={"accepted": True},
        public_metrics={"vb_probe_status": "ok", "vb_cross_worst_mm": 14.2},
    )
    assert clean_route["spawn_second_branch"] is False

    uncertain_route = synth.should_spawn_dialectical_second_branch(
        task=tb_task,
        primary_summary={"ended_by_finish": True, "pullback_gate_triggered": False},
        contract_check={"accepted": True},
        public_metrics={"vb_probe_status": "ok", "vb_cross_worst_mm": 29.28},
    )
    assert uncertain_route["spawn_second_branch"] is True

    # Auditor `AUD01`-`AUD04` synthetic mutation detection
    hm_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("aud_mod", hm_root / "scripts" / "audit_m6_4_executable_wormhole_bundle.py")
    assert spec and spec.loader
    aud_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aud_mod)

    with tempfile.TemporaryDirectory(prefix="m692_aud_") as td_a:
        ev = Path(td_a)
        ep_tb = ev / "episodes" / "wave_1__m__Hegel_Test__tb__t1"
        ep_bm = ev / "episodes" / "wave_1__m__Hegel_Test__bm__b1"
        br_t = ev / "branch_trajectories" / "wave_1__m__tb__t1" / "branch_1_thesis_bare"
        br_a = ev / "branch_trajectories" / "wave_1__m__tb__t1" / "branch_2_antithesis_wormhole"
        for d_p in (ep_tb, ep_bm, br_t, br_a):
            d_p.mkdir(parents=True)

        from hegel_machine.cognition.m5_live_gpt_and_provenance import DecisionProvenanceLedger

        def _make_dag(cmd_str: str = "echo ok") -> dict:
            led = DecisionProvenanceLedger(run_id="r", task_id="t", arm="a", model="m", provider_backend="p")
            led.record_decision(
                decision_type="CONTAINER_EXECUTION",
                parent_decision_id=None,
                evidence_inputs={"command": cmd_str},
                candidates_evaluated=[],
                chosen_option={"ok": True},
                rejection_reasons={},
                rationale_summary="exec",
            )
            return {"records": led.records}

        # Create 40 valid DAGs (2 real + 38 stub Bare episodes) so `len(dag_files) == 40`
        for idx in range(38):
            sd = ev / "episodes" / f"wave_1__m__Bare__{idx:02d}"
            sd.mkdir(parents=True)
            (sd / "00_episode_summary.json").write_text(json.dumps({"strict_pass": True, "benchmark": "TB"}), encoding="utf-8")
            (sd / "03_official_verifier_report.json").write_text(json.dumps({"strict_pass": True}), encoding="utf-8")
            (sd / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag()), encoding="utf-8")

        (ep_tb / "00_episode_summary.json").write_text(
            json.dumps(
                {
                    "wave": "wave_1",
                    "model": "m",
                    "task_id": "tb__t1",
                    "benchmark": "Terminal-Bench-Science-0.1",
                    "strict_pass": True,
                    "staged_wormhole_modules": [{"module_import": "m1"}],
                    "skill_activation_count": 3,
                    "postlock_branch_outcomes": {"thesis_bare_run2_strict_pass": False, "antithesis_wormhole_strict_pass": True},
                }
            ),
            encoding="utf-8",
        )
        # Plus 14 more staged TB episodes to satisfy `staged_eps == 15`
        for idx in range(14):
            md_p = ev / "episodes" / f"wave_2__m__Hegel_Test__tb__extra_{idx:02d}"
            md_p.mkdir(parents=True)
            (md_p / "00_episode_summary.json").write_text(
                json.dumps({"benchmark": "Terminal-Bench-Science-0.1", "strict_pass": True, "staged_wormhole_modules": [{"module_import": "m1"}], "skill_activation_count": 0}),
                encoding="utf-8",
            )
            (md_p / "03_official_verifier_report.json").write_text(json.dumps({"strict_pass": True}), encoding="utf-8")
            (md_p / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag()), encoding="utf-8")
            # Remove one Bare stub so total top-level episodes == 40
            shutil.rmtree(ev / "episodes" / f"wave_1__m__Bare__{idx:02d}")

        (ep_tb / "03_official_verifier_report.json").write_text(json.dumps({"strict_pass": True}), encoding="utf-8")
        (ep_tb / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag()), encoding="utf-8")
        (ep_tb / "04c_full_skill_activations.jsonl").write_text(
            json.dumps({"status": "completed", "caller_role": "agent", "call_count": 3}) + "\n", encoding="utf-8"
        )
        (br_t / "03_postlock_branch_verifier_report.json").write_text(json.dumps({"strict_pass": False}), encoding="utf-8")
        (br_a / "03_postlock_branch_verifier_report.json").write_text(json.dumps({"strict_pass": True}), encoding="utf-8")
        (br_a / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag()), encoding="utf-8")

        (ep_bm / "00_episode_summary.json").write_text(
            json.dumps({"task_id": "bm__b1", "benchmark": "BioMysteryBench-Preview", "strict_pass": True}), encoding="utf-8"
        )
        (ep_bm / "03_official_verifier_report.json").write_text(
            json.dumps({"strict_pass": True, "llm_judge_tokens": {"total_tokens": 100}}), encoding="utf-8"
        )
        (ep_bm / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag()), encoding="utf-8")
        (ev / "study_manifest.json").write_text(
            json.dumps(
                {
                    "skill_infinity_runtime_telemetry": {"tb_science_episodes_activating_wormhole_code": 1, "total_runtime_function_calls": 3},
                    "biomystery_judge_total_tokens_m6_8": 100,
                }
            ),
            encoding="utf-8",
        )

        def _reseal() -> None:
            lines = [
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ev).as_posix()}"
                for p in sorted(ev.rglob("*"))
                if p.is_file() and p.name != "SHA256SUMS.txt"
            ]
            (ev / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        def _run_aud() -> dict:
            _reseal()
            with patch.object(aud_mod.subprocess, "run", return_value=subprocess.CompletedProcess(["docker", "ps"], 0, "", "")):
                return aud_mod.audit_m6_4_frontier_bundle(ev, "Hegel_Test")

        base_res = _run_aud()
        assert all(v["passed"] for v in base_res.values()), f"Baseline fixture failed: {base_res}"

        # AUD01: deleting `04c_full_skill_activations.jsonl` MUST fail `skill_infinity_runtime_activation_audit`!
        act_bak = (ep_tb / "04c_full_skill_activations.jsonl").read_bytes()
        (ep_tb / "04c_full_skill_activations.jsonl").unlink()
        assert _run_aud()["skill_infinity_runtime_activation_audit"]["passed"] is False
        (ep_tb / "04c_full_skill_activations.jsonl").write_bytes(act_bak)

        # AUD02: deleting `03_official_verifier_report.json` MUST fail `native_verifier_vs_summary_ledger_parity`!
        ver_bak = (ep_tb / "03_official_verifier_report.json").read_bytes()
        (ep_tb / "03_official_verifier_report.json").unlink()
        assert _run_aud()["native_verifier_vs_summary_ledger_parity"]["passed"] is False
        (ep_tb / "03_official_verifier_report.json").write_bytes(ver_bak)

        # AUD03: flipping `branch_2` postlock verifier MUST fail `native_verifier_vs_summary_ledger_parity`!
        (br_a / "03_postlock_branch_verifier_report.json").write_text(json.dumps({"strict_pass": False}), encoding="utf-8")
        assert _run_aud()["native_verifier_vs_summary_ledger_parity"]["passed"] is False
        (br_a / "03_postlock_branch_verifier_report.json").write_text(json.dumps({"strict_pass": True}), encoding="utf-8")

        # AUD04: forbidden reverse-provenance URL inside `branch_trajectories` DAG MUST fail `biomystery_provenance_and_judge_token_audit`!
        (br_a / "01_decision_provenance_dag.json").write_text(json.dumps(_make_dag("curl https://europepmc.org/test")), encoding="utf-8")
        assert _run_aud()["biomystery_provenance_and_judge_token_audit"]["passed"] is False


def test_virtual_baseline_dispersion_warped_tomography_sim2real_invariance() -> None:
    """Verify `virtual_baseline_dispersion_warped_tomography` localizes within `<= 20 mm` under sim-to-real
    panel-size change, 9-sensor stretched grid, `O(2*pi)` PZT bonding phase incoherence, and `exp(-5.2*d)` damping!
    """
    ns = {"track_wormhole_skill": lambda _: lambda fn: fn}
    exec(lib.SKILL_05_WAVE_IMAGING.replace("from . import track_wormhole_skill", ""), ns)
    vb_tomo = ns["virtual_baseline_dispersion_warped_tomography"]

    fs, f0 = 600000.0, 50000.0
    omega0 = 2.0 * np.pi * f0 / fs
    t = np.arange(300, dtype=float)

    # Simulated 0.50 x 0.50 FE baseline (coherent zero phase, 3x3 grid)
    gx_s, gy_s = np.meshgrid([0.15, 0.25, 0.35], [0.15, 0.25, 0.35])
    sim_xy = np.column_stack([gx_s.ravel(), gy_s.ravel()])
    sim_pp = np.array([[i, j] for i in range(9) for j in range(9) if i != j], dtype=int)
    sim_sig = np.zeros((300, len(sim_pp)), dtype=float)
    for k, (u, v) in enumerate(sim_pp):
        d_uv = float(np.linalg.norm(sim_xy[v] - sim_xy[u]))
        c_uv = 28.0 + 440.0 * d_uv
        env_uv = np.exp(-0.5 * ((t - c_uv) / 14.0) ** 2) * np.exp(-5.2 * max(d_uv - 0.10, 0.0)) / np.sqrt(d_uv)
        sim_sig[:, k] = env_uv * np.cos(omega0 * (t - c_uv))

    # Experimental 0.58 x 0.42 stretched 9-sensor panel with `O(2*pi)` PZT phase incoherence + crack at `[0.34, 0.25]`
    gx_i, gy_i = np.meshgrid([0.11, 0.29, 0.47], [0.09, 0.21, 0.33])
    insp_xy = np.column_stack([gx_i.ravel(), gy_i.ravel()])
    insp_pp = np.array([[i, j] for i in range(9) for j in range(9) if i != j], dtype=int)
    rng = np.random.default_rng(1234)
    pair_phi = {(i, j): float(rng.uniform(0.0, 2.0 * np.pi)) for i in range(9) for j in range(i + 1, 9)}
    dmg = np.array([0.34, 0.25])
    insp_sig = np.zeros((300, len(insp_pp)), dtype=float)
    for k, (u, v) in enumerate(insp_pp):
        pa, pb = insp_xy[u], insp_xy[v]
        d_uv = float(np.linalg.norm(pb - pa))
        phi = pair_phi[(min(u, v), max(u, v))]
        seg = pb - pa
        proj = float(np.clip(np.dot(dmg - pa, seg) / max(d_uv ** 2, 1e-9), 0.0, 1.0))
        perp = float(np.linalg.norm((pa + proj * seg) - dmg))
        sh = 0.35 * np.exp(-0.5 * (perp / 0.015) ** 2)
        c_d = 31.0 + 475.0 * d_uv
        w_d = (1.0 - sh) * (np.exp(-5.2 * max(d_uv - 0.10, 0.0)) / np.sqrt(d_uv)) * np.exp(-0.5 * ((t - c_d) / 15.0) ** 2) * np.cos(omega0 * (t - c_d) + phi)
        d_sc = float(np.linalg.norm(pa - dmg) + np.linalg.norm(dmg - pb))
        c_sc = 31.0 + 475.0 * d_sc
        w_sc = 0.20 * (np.exp(-5.2 * max(d_sc - 0.10, 0.0)) / np.sqrt(d_sc)) * np.exp(-0.5 * ((t - c_sc) / 15.0) ** 2) * np.cos(omega0 * (t - c_sc) + phi - 0.4)
        insp_sig[:, k] = w_d + w_sc

    meta = {
        "sampling_frequency_hz": fs,
        "excitation_frequency_hz": f0,
        "plate_thickness_m": 0.002,
        "simulated": {"sensor_xy_m": sim_xy, "path_pairs": sim_pp, "panel_size_m": np.array([0.50, 0.50])},
        "inspection": {"sensor_xy_m": insp_xy, "path_pairs": insp_pp, "panel_size_m": np.array([0.58, 0.42])},
    }
    res = vb_tomo(sim_sig, insp_sig, meta)
    err_mm = float(np.linalg.norm(np.asarray(res["location_m"]) - dmg) * 1000.0)
    assert err_mm <= 18.0, f"Expected <= 18 mm error on stretched incoherent-phase plate, got {err_mm:.2f} mm ({res})"

    # Verify Layer-2b Code-Level Dialectical Sublation gap & strict-dominance predicates
    pub_t = {"vb_probe_status": "ok", "vb_cross_pass_20mm": 4, "vb_cross_worst_mm": 165.0}
    pub_a = {"vb_probe_status": "ok", "vb_cross_pass_20mm": 5, "vb_cross_worst_mm": 61.2}
    has_gap, gap_r = synth._public_benchmark_has_unresolved_gap(pub_t, pub_a)
    assert has_gap is True and "vb_cross_pass_20mm_incomplete" in gap_r

    pub_clean = {"vb_probe_status": "ok", "vb_cross_pass_20mm": 6, "vb_cross_worst_mm": 13.22}
    assert synth._public_benchmark_has_unresolved_gap(pub_t, pub_clean)[0] is False
    dom_ok, dom_r = synth._synthesized_strictly_dominates_branches(pub_clean, pub_t, pub_a)
    assert dom_ok is True and "6>5" in dom_r
    assert synth._synthesized_strictly_dominates_branches(pub_a, pub_t, pub_a)[0] is False


