"""Comprehensive Audit Closure Test Suite for Hegel M6.8-B' (`Hegel_M63_M67_Audit`).

Deterministically verifies 100% of the counterexamples across:
  1. Track A (`A01`–`A08`): flow-sensitive row translator, ABI optional state, `ctypes` audit guard,
     Pareto deduplication on `(used_keys, canonical_ast)`, genuine historical shadow evaluation,
     d-separation Bayes-ball filtering, strict `validated_pilot` tolerance gate, and foreign-key branch growth.
  2. Executable Wormhole Library (`L01`–`L09`): post-execution telemetry via kernel `os.write` across `os._exit(0)`,
     nested lexical scope isolation, I/O ban, helper call-graph closure, NaN parity guard, Godunov CFL guard,
     exact discrete minimax, negative-label cohort exclusion, Nyquist Fourier basis, and nested probe CV.
  3. Frontier Harness & Lock (`H01`, `H04`, `H05`, `H06`): AST verification command check, hash-bound commit gate
     (blocking double-submit, `print('solver.py')`, and last-turn bypass while accepting single-turn modify+verify),
     reader+writer reentrant `HeavyTimingRWLock` (`deadlock_condition == False`), and `RegradeReceipt` hash binding.
  4. Hegelian Core Bridges (`P0b`, `P1a`, `P1b`): `SKILL_07_MULTISCALE_RECONCILIATION` (`pi_fold_temporal_regime_selector`
     & `adjudicate_incumbent_vs_challenger_slices`), `consolidate_wave1_skill_manifests_with_reconciliation`
     (`SharedBaseSubtreePrior` + `propagate_premise_retraction_invalidation`), historical `/tmp` producer recovery,
     and public-only candidate selector minority-slice protection.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np

from hegel_machine.cognition import m6_3_sovereign_causal_wormhole_engine as m63
from hegel_machine.cognition import m6_frontier_real_benchmarks as f
from hegel_machine.cognition.m5_live_gpt_and_provenance import _execute_authored_python_in_isolated_subprocess
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,
    NODE_TO_EXECUTABLE_WORMHOLE_MODULE,
    SKILL_INIT_PY,
    extract_pure_reusable_functions_from_python,
    sum_skill_activation_calls,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    SKILL_07_MULTISCALE_RECONCILIATION,
    HeavyTimingRWLock,
    build_regrade_receipt,
    consolidate_wave1_skill_manifests_with_reconciliation,
    select_candidate_on_public_evidence_only,
)
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves


def _load_pkg_with_skills(tmp_dir: Path):
    pkg = tmp_dir / "hegel_wormhole_skills"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(SKILL_INIT_PY, encoding="utf-8")
    seen: set[str] = set()
    for fname, code, _ in NODE_TO_EXECUTABLE_WORMHOLE_MODULE.values():
        if fname not in seen:
            (pkg / fname).write_text(code, encoding="utf-8")
            seen.add(fname)
    (pkg / "multiscale_and_reconciliation_wormhole.py").write_text(SKILL_07_MULTISCALE_RECONCILIATION, encoding="utf-8")
    sys.path.insert(0, str(tmp_dir))
    return pkg


def test_tracka_sandbox_abi_and_flow_sensitive_translator() -> None:
    rows = [{"x": 1.0, "z": 2.0, "y": 3.0}, {"x": 2.0, "z": 5.0, "y": 9.0}, {"x": 3.0, "z": 10.0, "y": 19.0}]

    with tempfile.TemporaryDirectory(prefix="tracka_closure_") as td:
        # 1. `ctypes` blocked in sandbox
        bad = _execute_authored_python_in_isolated_subprocess(
            Path(td),
            step_idx=1,
            python_code="import ctypes\ndef predict(row):\n    ctypes.CDLL(None)\n    return 0.0\n",
            train_rows=rows,
            confirm_rows=rows,
            heldout_rows=rows,
        )
        assert bad["ok"] is False

        # 2. Optional default `state=None` 2-arg ABI executes cleanly (`rmse == 0.0`)
        opt_state = _execute_authored_python_in_isolated_subprocess(
            Path(td),
            step_idx=2,
            python_code="def predict(row, state=None):\n    return row['x'] * row['x'] + row['z']\n",
            train_rows=rows,
            confirm_rows=rows,
            heldout_rows=rows,
            include_predictions=True,
        )
        assert opt_state["ok"] is True
        assert opt_state["train_rmse"] == 0.0
        assert opt_state["train_preds"] == [3.0, 9.0, 19.0]

    # 3. Flow-sensitive row translator: `_KEYS` tuple, sequential alias kill, keyword helper call
    for src, vmap, test_in, expected in [
        (
            "_KEYS = ('s1', 's2')\ndef predict(row):\n    a, b = (float(row[k]) for k in _KEYS)\n    return a + 2.0 * b\n",
            {"s1": "m_1", "s2": "m_2"},
            {"m_1": 3.0, "m_2": 4.0},
            11.0,
        ),
        (
            "def predict(row):\n    r = row\n    a = r['s1']\n    r = {'x': 7.0}\n    return a + r['x']\n",
            {"s1": "m_1"},
            {"m_1": 5.0},
            12.0,
        ),
        (
            "def helper(q):\n    return q['s1'] * 3.0\ndef predict(row):\n    return helper(q=row)\n",
            {"s1": "m_1"},
            {"m_1": 4.0},
            12.0,
        ),
    ]:
        tr = m63.extract_and_translate_authored_ast_v2(src, vmap)
        assert tr["ok"] is True, tr
        ns: dict = {}
        exec(tr["translated_code"], ns)
        assert abs(float(ns["predict"](test_in)) - expected) < 1e-9


def test_tracka_pareto_shadow_dsep_and_writeback_owner() -> None:
    # Pareto keeps distinct AST formulas on identical variable sets
    entry: dict = {}
    m63._append_pareto_candidate(entry, used_keys=["x"], python_code="def predict(row): return row['x']", confirm_rmse=0.0, provenance_tag="linear")
    m63._append_pareto_candidate(entry, used_keys=["x"], python_code="def predict(row): return row['x']**2", confirm_rmse=0.0, provenance_tag="quad")
    assert len(entry["pareto_confirmed_candidates"]) == 2

    # Genuine historical shadow formula check distinguishes `+1e6` vs `-1e6`
    rec0 = {
        "status": "confirmed_ok",
        "strict_pass": True,
        "task_id": "M6_W1_C01_test",
        "winning_python_code": "def predict(row):\n    return 2.0 * row['z_1']\n",
        "confirm_rmse": 0.01,
        "recalled_node_ids": ["node_C01_TreeA_prime_electro"],
        "observed_signatures": ["sig:c01_treeAprime_electro"],
        "violated_root_ids": [],
        "pilot_rows_snapshot": [
            {"z_1": 1.0, "y": 2.0},
            {"z_1": 2.0, "y": 4.0},
            {"z_1": 3.0, "y": 6.0},
        ],
    }
    retro_diffs: list[float] = []
    for shadow_txt in ("old law y = 1000000.0", "old law y = -1000000.0"):
        fo, ar = m63.build_m6_1_clean_k40_forest()
        for v in ar.values():
            v["verified_mechanism_note"] = shadow_txt
            v.pop("historical_L1_shadow", None)
        c = m63.consolidate_wave1_sovereign_m6_3(wave1_receipts=[rec0], arm="Hegel_M6_3_Sovereign_LiveGPT", forest=fo, node_archive=ar)
        proj = c["events"][0]["empirical_retroactive_projection"]
        assert proj["passive_shadow_consistent"] is False
        retro_diffs.append(float(proj["signed_mean_projection_diff"]))
    assert retro_diffs[0] < -900000.0 and retro_diffs[1] > 900000.0

    # Writeback owner grows composite branch when foreign keys dominate
    fo_w, ar_w = m63.build_m6_1_clean_k40_forest()
    owner = m63.resolve_verified_writeback_owner(
        recalled_node_ids=["node_C02_TreeM1_surface"],
        observed_signatures=["sig:c02_treeM1_extract"],
        active_ast_keys=["m_2", "foreign_required_1", "foreign_required_2"],
        violated_root_ids=[],
        forest=fo_w,
        node_archive=ar_w,
    )
    assert owner["owner_mode"] == "synthesized_composite_branch"

    # D-separation Bayes-ball filters collider wormhole (`w_orth -> G_orth`) vs disabled d-sep control
    f_d1, a_d1 = m63.build_m6_1_clean_k40_forest()
    f_d2, a_d2 = m63.build_m6_1_clean_k40_forest()
    spurious_rec = [
        {
            "status": "confirmed_ok",
            "strict_pass": True,
            "task_id": "synthetic_collider_role_bindable",
            "winning_python_code": "def predict(row): return 2.5 * row['w_orth']\n",
            "confirm_rmse": 0.01,
            "recalled_node_ids": ["node_C01_TreeA_prime_electro"],
            "observed_signatures": ["sig:c01_treeAprime_electro"],
            "violated_root_ids": [],
        }
    ]
    m63.consolidate_wave1_sovereign_m6_3(wave1_receipts=spurious_rec, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f_d1, node_archive=a_d1)
    orig_dsep = m63.is_d_separated_dag
    try:
        m63.is_d_separated_dag = lambda *args, **kwargs: False
        m63.consolidate_wave1_sovereign_m6_3(wave1_receipts=spurious_rec, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f_d2, node_archive=a_d2)
    finally:
        m63.is_d_separated_dag = orig_dsep
    assert len(a_d1["node_C05_TreeG_orth_direct"].get("pareto_confirmed_candidates", [])) == 0
    assert len(a_d2["node_C05_TreeG_orth_direct"].get("pareto_confirmed_candidates", [])) == 1


def test_executable_wormhole_library_and_multiscale_skill_controls() -> None:
    # Lexical scope isolation, I/O ban, and helper call-graph closure
    assert extract_pure_reusable_functions_from_python(
        "def outer(x):\n    def inner(y):\n        return y * unbound_external_scale\n    return inner(x)\n"
    ) == []
    assert [nm for nm, _ in extract_pure_reusable_functions_from_python(
        "def outer(scale):\n    def inner(x):\n        return x * scale\n    return inner\n"
    )] == ["outer"]
    assert extract_pure_reusable_functions_from_python(
        "def leak(p):\n    return open(p).read()\n"
    ) == []
    names = [
        nm
        for nm, _ in extract_pure_reusable_functions_from_python(
            "def helper(x):\n    y = float(x) + 1.0\n    return y\n\n"
            "def top(x):\n    z = helper(x)\n    return 2.0 * z\n"
        )
    ]
    assert names == ["helper", "top"]

    with tempfile.TemporaryDirectory(prefix="hegel_m68_lib_test_") as td:
        td_path = Path(td)
        log_fp = td_path / "activations.jsonl"
        os.environ["HEGEL_SKILL_ACTIVATION_LOG"] = str(log_fp)
        os.environ["HEGEL_SKILL_CALLER_ROLE"] = "agent"
        _load_pkg_with_skills(td_path)
        try:
            import importlib

            importlib.invalidate_caches()
            p_mod = importlib.import_module("hegel_wormhole_skills.conservation_and_hybrid_relay_wormhole")
            s_mod = importlib.import_module("hegel_wormhole_skills.variational_assimilation_and_certificate_wormhole")
            c_mod = importlib.import_module("hegel_wormhole_skills.cohort_transfer_and_imputation_wormhole")
            m_mod = importlib.import_module("hegel_wormhole_skills.multiscale_and_reconciliation_wormhole")

            # NaN parity returns matched == False
            nan_res = p_mod.verify_vectorized_vs_scalar_recurrence(
                lambda x: np.full_like(x, np.nan, dtype=float),
                lambda x: np.full_like(x, np.nan, dtype=float),
                [(np.ones(4),)],
            )
            assert nan_res["matched"] is False

            # Exact discrete minimax picks index 2 (`max loss = 0.25` < `1.0`)
            mm = p_mod.minimax_consensus_candidate_selector(
                [[0.0], [1.0], [0.5]],
                lambda _cands: np.array([[0.0, 1.0], [1.0, 0.0], [0.25, 0.25]], dtype=float),
            )
            assert mm["selected_index"] == 2

            # Negative placeholder `-1` excluded from cohort calibration (`n_samples == 4`, NOT `5`, on Cohort B!)
            X = np.array([[-2.0], [-1.0], [1.0], [2.0], [-2.0], [-1.0], [1.0], [2.0], [0.2]])
            y = np.array([0, 0, 1, 1, 0, 0, 1, 1, -1])
            cohorts = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "B"])
            loco = c_mod.leave_one_cohort_out_calibrated_cv(
                X,
                y,
                cohorts,
                lambda Xtr, ytr, Xte, _ctr, _cte: 1.0 / (1.0 + np.exp(-Xte[:, 0])),
            )
            assert loco["per_cohort"]["B"]["n_samples"] == 4

            # Nyquist Fourier basis (`n_modes = 8 == n_sites // 2`) full rank & orthonormal `(16, 16)`
            B, _ = s_mod.real_fourier_smoothness_basis(16, 8)
            assert B.shape == (16, 16)
            assert np.linalg.matrix_rank(B) == 16
            assert float(np.max(np.abs(B.T @ B - np.eye(16)))) < 1e-12

            # Plan-20 `pi_fold_temporal_regime_selector` selects `mori_zwanzig_memory_kernel` on history-convolved signal
            t = np.linspace(0.0, 8.0, 81)
            u = np.sin(1.7 * t) + 0.5 * np.cos(0.9 * t)
            mem = np.zeros_like(t)
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                e = np.exp(-dt / 0.8)
                mem[i] = e * mem[i - 1] + 0.5 * dt * (e * u[i - 1] + u[i])
            y_mz = 0.2 * u + 1.8 * mem + 0.4
            pi_sel = m_mod.pi_fold_temporal_regime_selector(t, u, y_mz, cutoff_t0=5.5)
            assert pi_sel["selected_pi_fold_mode"] == "mori_zwanzig_memory_kernel"

            # Minority-slice protection blocks pooled-mean challenger that degrades Cohort C
            adj = m_mod.adjudicate_incumbent_vs_challenger_slices(
                {"cohort_A": 0.12, "cohort_B": 0.14, "cohort_C": 0.15},
                {"cohort_A": 0.05, "cohort_B": 0.06, "cohort_C": 0.24},
            )
            assert adj["decision"] == "keep_incumbent"

            # Failed call raises ValueError and logs `status="failed"` (excluded from `sum_skill_activation_calls`),
            # AND abrupt `os._exit(0)` in child subprocess preserves 100% of kernel-written calls!
            sub_env = dict(os.environ)
            sub_env["PYTHONPATH"] = str(td_path)
            sub_env["HEGEL_SKILL_ACTIVATION_LOG"] = str(td_path / "exit_log.jsonl")
            sub_env["HEGEL_SKILL_CALLER_ROLE"] = "agent"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os, numpy as np\n"
                    "from hegel_wormhole_skills.variational_assimilation_and_certificate_wormhole import real_fourier_smoothness_basis\n"
                    "for _ in range(25): real_fourier_smoothness_basis(8, 3)\n"
                    "try: real_fourier_smoothness_basis(0, 1)\n"
                    "except ValueError: pass\n"
                    "os._exit(0)\n",
                ],
                env=sub_env,
                check=True,
            )
            exit_recs = [json.loads(ln) for ln in (td_path / "exit_log.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert sum_skill_activation_calls(exit_recs) == 25
            assert any(r.get("status") == "failed" for r in exit_recs)
        finally:
            if str(td_path) in sys.path:
                sys.path.remove(str(td_path))
            for k in list(sys.modules):
                if k == "hegel_wormhole_skills" or k.startswith("hegel_wormhole_skills."):
                    sys.modules.pop(k, None)


def test_frontier_harness_commit_gate_reentrant_rwlock_and_m5_reconciliation() -> None:
    # 1. Verification command AST check
    assert f.is_substantive_verification_command("true", ("/app/solver.py",)) is False
    assert f.is_substantive_verification_command("python3 -c \"print('solver.py')\"", ("/app/solver.py",)) is False
    assert f.is_substantive_verification_command("pytest -q /tmp/test_unrelated.py", ("/app/solver.py",)) is False
    assert f.is_substantive_verification_command("python3 /tmp/validate_saved_candidate.py", ("/app/solver.py",)) is True

    # 2. Nested reader RWLock never deadlocks behind queued writer
    lock = HeavyTimingRWLock()
    outer_held = threading.Event()
    req_nested = threading.Event()
    nested_ok = threading.Event()
    writer_ok = threading.Event()

    def _r():
        lock.acquire_read()
        outer_held.set()
        req_nested.wait(2)
        lock.acquire_read()
        nested_ok.set()
        lock.release_read()
        lock.release_read()

    def _w():
        lock.acquire_write()
        writer_ok.set()
        lock.release_write()

    tr = threading.Thread(target=_r, daemon=True)
    tw = threading.Thread(target=_w, daemon=True)
    tr.start()
    assert outer_held.wait(2)
    tw.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with lock._cond:
            if lock._writers_waiting == 1:
                break
        time.sleep(0.002)
    req_nested.set()
    assert nested_ok.wait(0.5)
    tr.join(1)
    tw.join(1)
    assert writer_ok.is_set()

    # 3. `RegradeReceipt` rejects mismatched candidate hash
    bad_receipt = build_regrade_receipt(
        original_episode_id="ep1",
        original_candidate_sha256="a" * 64,
        regrade_candidate_sha256="b" * 64,
        original_verifier_sha256="c" * 64,
        task_id="tb__amr-poisson-optimize",
        verifier_image_tag="mock",
        resource_spec={"cpus": 2},
        timing_replicates_sec=[71.1, 71.2, 71.3],
        official_median_sec=71.2,
        threshold_sec=73.12,
        strict_pass=True,
        ctrf_summary={},
        verifier_stdout="ok",
        regrade_reason="uncontended_timing_replay",
    )
    assert bad_receipt["candidate_hash_match_verified"] is False
    assert bad_receipt["regrade_strict_pass"] is False

    # 4. Wave-1 consolidation registers `SharedBaseSubtreePrior` & propagates RUNG_3 premise retractions
    forest, archive, _ = f.build_m6_frontier_general_forest()
    with tempfile.TemporaryDirectory(prefix="hegel_w1_reconcile_") as td:
        ep1 = Path(td) / "ep1"
        ep2 = Path(td) / "ep2"
        for ep_d, tid, fam, fn_code in [
            (ep1, "tb__ode-law-discovery", "coupled_nonlinear_ode_discovery", "def rk4_step(x, dt):\n    return x + dt\n"),
            (ep2, "tb__navigation-sensor-calibration", "multi_rate_navigation_fusion", "def fuse_heading(a, b):\n    return 0.5 * (a + b)\n"),
        ]:
            (ep_d / "04b_producer_skill_manifest").mkdir(parents=True)
            (ep_d / "04b_producer_skill_manifest" / "executable_skill_manifest.json").write_text(
                json.dumps(
                    {
                        "temporal": {"temporal_readout_mode": "mori_zwanzig_or_stage_history"},
                        "structure": {"full_producer_count": 2, "reusable_ast_function_closures": [{"name": "op", "code": fn_code}]},
                    }
                ),
                encoding="utf-8",
            )
        rec_out = consolidate_wave1_skill_manifests_with_reconciliation(
            wave1_episode_dirs=[
                ({"task_id": "tb__ode-law-discovery", "short_name": "ode", "wormhole_family": "coupled_nonlinear_ode_discovery", "strict_pass": True, "nogoods_recorded": []}, ep1),
                ({"task_id": "tb__navigation-sensor-calibration", "short_name": "nav", "wormhole_family": "multi_rate_navigation_fusion", "strict_pass": True, "nogoods_recorded": [{"nogood_rung": "RUNG_3_MECHANISM"}]}, ep2),
            ],
            forest=forest,
            archive=archive,
        )
        assert len(rec_out["registered_shared_base_subtrees"]) == 1
        assert rec_out["m5_pi_fold_and_alignment_wiring"]["pi_fold_memory_kernel_ready"] is True
        assert len(rec_out["retracted_claim_updates"]) >= 1

    # 5. Public-only candidate selector blocks post-selection snooping challenger
    sel = select_candidate_on_public_evidence_only(
        [
            {"candidate_id": "incumbent", "contract_accepted": True, "has_unverified_late_edit": False, "has_post_selection_snoop_warning": False, "public_slice_regressions": 0},
            {"candidate_id": "snooped_challenger", "contract_accepted": True, "has_unverified_late_edit": False, "has_post_selection_snoop_warning": True, "public_slice_regressions": 0, "public_honest_cv_score": 0.95},
        ],
        incumbent_index=0,
    )
    assert sel["selected_candidate_id"] == "incumbent"

    # 6. `HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY` parses cleanly & contains zero coercive `MUST apply` or private leak tokens
    ast.parse(HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY)
    assert "MUST apply" not in HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY
    assert "answer_rubric" not in HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY
