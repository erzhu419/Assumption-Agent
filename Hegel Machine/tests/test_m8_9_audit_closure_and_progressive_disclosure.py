"""Audit-closure & progressive-disclosure regression tests for `M8.9` (`Hegel_M8_8_Audit` Closure).

Verifies:
1. `P0-1` (`C1`): `box_ridge_cube_lower_bound` full PSD minorant guard (`G - diag(d) >= 0` across ALL `p`,
   independent of `ridge`, plus conservative box slack compensation) against all 4 GPT-6 counterexamples,
   plus `withdrawn_qualifications` and clean-container re-verification (`v4 -> v5`).
2. `P0-2`: `uf01..uf04` strict `NaN`/`Inf`/boolean/gap rejection (`allow_nan=False`, `uf04` `gap <= 1e-5`),
   plus passing re-verification of sealed `M8.8` `uf04` deliverables (`l2_only` & `l1_l2`).
3. `P1-1` (`C2`): `PersistentVariationalPlanner` exact IEEE-754 cache identity over `past_driver_history`,
   `timestamps`, `history_dt`, `trajectory_ids`, `initial_memory_states`, and full incumbent mechanism spec.
4. `P1-2` & `P1-3` (`C3`, `C4`): Stale `compiled_code_sha256` rejection in `L1` compiler (`no` blind retention)
   and canonical multi-artifact `declared-artifact-bundle-v1` digest matching in `Wave-1` compiler.
5. `P1-4`, `P1-5`, `P1-6` (`D1..D4`): All 4 `auto_package_wormhole_imports_in_container` write/rollback
   fault-injection modes (`nonzero_plain`, `timeout`, `partial_failed_current`, `rollback_failed`),
   module attribute mutation & bare-module escape rejection (`m.K = 5`), fail-closed scanner payload validation,
   and closure-exact + alias-resolved + entrypoint-scoped `inspect_final_deliverable_skill_retention`.
6. `P1-7` (`C5`) & `6.2`/`6.3`: Witness-bound `nogood` resolution (`resolves_nogood_ids` deferred until a clean
   `VerificationReceipt`, zero dangling `superseded_by`), `[HOST ARTIFACT IDENTITY]` notice, and
   `M8OfflineContainerPlanner` `m8_9_prefix_clean_v2` (`f_grounded` without artificial `27/28` cliff).
7. `M8.9A`, `M8.9B` & `D5`: Lossless precondition deduplication, progressive 1-line index cards, pure-JSON
   `python3 -m hegel_wormhole_skills --list | --describe` without arbitrary code import, expected-net-value
   staging gate (`uf03`/`hs02` boundary-mismatch veto vs `hs01`/`hs04` positive recommendation), and
   `LocalContainerRepair` compact skill card visibility.
8. `Hegel-M89-HardStress-4`: Reference solvers pass (`strict_pass == True`) and naive/non-finite/mismatched
   baselines fail (`strict_pass == False`) across all 4 tasks (`hs01..hs04`).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import (
    run_unseen_postfreeze_private_verifier_by_id,
    unseen_postfreeze_tasks_by_id,
)
from hegel_machine.cognition.m8_9_hard_stress_benchmark import (
    HARD_STRESS_M89_BY_ID,
    hard_stress_task_to_frontier_task,
    run_hard_stress_private_verifier_by_id,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import execute_in_container_local_repair
from hegel_machine.cognition.m8_offline_container_planner import (
    ROUTER_FEATURE_SCHEMA_VERSION,
    M8OfflineContainerPlanner,
    build_container_pre_action_state,
    build_pre_action_feature_vector,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

REPO_ROOT = Path(__file__).resolve().parents[1]
M88_EV_DIR = REPO_ROOT / "docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125"
V4_BUNDLE_PATH = M88_EV_DIR / "pretrained_autogenous_forest_v4_m88.json"


def _make_tar_from_files(files_by_arcpath: dict[str, str], out_tar: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="m89_tar_") as td:
        td_p = Path(td)
        with tarfile.open(out_tar, "w") as tf:
            for arc, content in files_by_arcpath.items():
                rel = arc.lstrip("/")
                fp = td_p / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                tf.add(str(fp), arcname=rel)
    return out_tar


# ============================================================================
# 1. P0-1 (`C1`): PSD Minorant Guard Counterexamples & v4 -> v5 Re-Verification
# ============================================================================


def test_m8_9_p0_1_psd_minorant_guard_all_four_audit_counterexamples_and_v5_reverification() -> None:
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(V4_BUNDLE_PATH)

    # First run WITHOUT clean-container re-verification: stale receipt MUST be withdrawn (`qualified == False`)!
    enr_unverified = m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=False)
    assert len(enr_unverified["minorant_hardened_nodes"]) >= 1
    box_mod_unv = next(
        m
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in (r.get("autogenous_wormhole_modules") or [])
        if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
    )
    assert box_mod_unv["unit_test_verified"] is False
    assert len(box_mod_unv.get("withdrawn_qualifications", [])) >= 1
    nv_unv = m64.evaluate_l1_skill_staging_net_value(box_mod_unv, "", set(), "box l0 certified_lower_bound")
    assert nv_unv["qualified"] is False

    # Now re-verify in clean container: fresh `unit_test_receipt` binds to exact mutated SHA!
    enr_ver = m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=True)
    assert "autogenous_prim_box_ridge_bounds" in enr_ver["reverified_modules"]
    box_mod = next(
        m
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in (r.get("autogenous_wormhole_modules") or [])
        if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
    )
    actual_sha = hashlib.sha256(box_mod["python_code"].encode("utf-8")).hexdigest()
    assert box_mod["unit_test_verified"] is True
    assert box_mod["compiled_code_sha256"] == actual_sha
    assert box_mod["unit_test_receipt"]["verified_code_sha256"] == actual_sha

    pkg = m64.package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import gram_diagonal_minorant, box_ridge_cube_lower_bound\n",
        {"autogenous_prim_box_ridge_bounds": box_mod["python_code"]},
    )
    assert pkg["ok"] is True
    ns: dict[str, Any] = {}
    exec(pkg["packaged_python_code"], ns)
    fn_lb = ns["box_ridge_cube_lower_bound"]
    fn_diag = ns["gram_diagonal_minorant"]

    # Exact 4 counterexamples from `Hegel_M8_8_Audit/compiler/reproduce_m88_compiler_cache_counterexamples.py`:
    # True objective at w=0 with y=0, l0=0 is F(0) = 0.0, so ANY valid lower bound MUST satisfy lb <= 0.0 (or raise ValueError)!
    for p_dim, ridge_val, d_val, must_raise in (
        (1, 0.0, 2.0, True),
        (1, 1.0, 2.0, True),       # `ridge=1.0` MUST NOT mask non-PSD `G - diag(d) = 1 - 2 = -1`!
        (513, 0.0, 2.0, True),     # `p=513 > 512` MUST NOT skip PSD check!
        (1, 0.0, 1.00000005, False),  # Within numerical PSD tolerance (`1e-9..1e-7`): must either raise ValueError OR subtract conservative slack so lb <= 0.0!
    ):
        X = np.eye(p_dim, dtype=float)
        y = np.zeros(p_dim, dtype=float)
        c = np.ones(p_dim, dtype=float)
        d = np.full(p_dim, d_val, dtype=float)
        if must_raise:
            with pytest.raises(ValueError, match="PSD minorant"):
                fn_lb(X, y, c, 1.0, 0.0, ridge=ridge_val, diagonal=d)
        else:
            try:
                lb = float(fn_lb(X, y, c, 1.0, 0.0, ridge=ridge_val, diagonal=d))
                assert lb <= 1e-15, f"Unsound lower bound {lb} > F(0)=0.0 on near-boundary minorant!"
            except ValueError:
                pass

    # Also test correlated 2x2 matrix where diagonal entries obey d_j <= ||X_:,j||^2=1.0, yet G - diag(d) is indefinite!
    X_corr = np.array([[1.0, 0.9], [0.0, np.sqrt(1.0 - 0.81)]], dtype=float)  # diag(G) = [1, 1], lambda_min(G) = 0.1
    with pytest.raises(ValueError, match="PSD minorant"):
        fn_lb(X_corr, np.zeros(2), np.ones(2), 1.0, 0.0, ridge=2.0, diagonal=np.array([0.8, 0.8]))
    d_safe = fn_diag(X_corr)
    lb_safe = float(fn_lb(X_corr, np.zeros(2), np.ones(2), 1.0, 0.0, ridge=0.5, diagonal=d_safe))
    assert lb_safe <= 1e-12


# ============================================================================
# 2. P0-2: Non-Finite Verifier Hardening (`uf01..uf04`) & Sealed M8.8 Recheck
# ============================================================================


def test_m8_9_p0_2_uf01_to_uf04_nonfinite_rejection_and_sealed_m8_8_uf04_reverification() -> None:
    ensure_clean_scipy_sandbox_image()
    with tempfile.TemporaryDirectory(prefix="m89_uf_ver_") as td:
        td_p = Path(td)
        # 1. Exact GPT-6 `uf04` NaN control + Inf control + bool-scalar control + loose-gap (`5e-5 > 1e-5`) control MUST ALL FAIL!
        nan_uf04 = (
            "def solve_certified_box_l0(instance):\n"
            "    p = len(instance['X'][0])\n"
            "    return {'weights': [float('nan')]*p, 'objective': float('nan'), 'certified_lower_bound': float('nan'), 'nodes_explored': 1}\n"
        )
        bool_uf04 = (
            "def solve_certified_box_l0(instance):\n"
            "    p = len(instance['X'][0])\n"
            "    return {'weights': [0.0]*p, 'objective': True, 'certified_lower_bound': False, 'nodes_explored': True}\n"
        )
        loose_gap_uf04 = (
            "import numpy as np\n"
            "def solve_certified_box_l0(instance):\n"
            "    y = np.asarray(instance['y'], float); p = np.asarray(instance['X'], float).shape[1]\n"
            "    j0 = 0.5 * float(y @ y)\n"
            "    return {'weights': [0.0]*p, 'objective': j0, 'certified_lower_bound': j0 - 5e-5, 'nodes_explored': 1}\n"
        )
        for idx, bad_src in enumerate((nan_uf04, bool_uf04, loose_gap_uf04)):
            tar_p = _make_tar_from_files({"/holdout_task/certified_l0_solver.py": bad_src}, td_p / f"bad_uf04_{idx}.tar")
            rep = run_unseen_postfreeze_private_verifier_by_id("uf04__certified_box_l0_sensor_selection", tar_p)
            assert rep["strict_pass"] is False, f"Expected uf04 failure on control {idx}, got {rep}"

        # 2. NaN payloads on `uf01`, `uf02`, `uf03` MUST ALSO FAIL CLOSED (`strict_pass == False`, valid JSON)!
        uf_nan_stubs = {
            "uf01__electrical_resistivity_electrode_contact_tomography": (
                "/holdout_task/ert_solver.py",
                "def invert_ert_anomaly(survey):\n"
                "    n = len(survey['z_rx_m'])\n"
                "    return {'anomaly_xz_m': [float('nan'), float('nan')], 'rx_contact_offset_v': [float('nan')]*n}\n",
            ),
            "uf02__batch_drifted_substrate_inhibition_kinetics": (
                "/holdout_task/enzyme_plate_solver.py",
                "def estimate_variant_haldane_parameters(plates):\n"
                "    vids = sorted({w['variant_id'] for p in plates for w in p['wells'] if w.get('variant_id')})\n"
                "    return {v: {'V_max': float('nan'), 'K_m': float('nan'), 'K_i': float('nan')} for v in vids}\n",
            ),
            "uf03__orthotropic_robin_heat_sink_inversion": (
                "/holdout_task/thermal_sink_solver.py",
                "def localize_thermal_source(measurement):\n"
                "    return {'source_xy': [float('nan'), float('nan')], 'amplitude': float('nan'), 'junction_offset': float('nan')}\n",
            ),
        }
        for tid, (art_path, stub_code) in uf_nan_stubs.items():
            tar_p = _make_tar_from_files({art_path: stub_code}, td_p / f"nan_{tid[:4]}.tar")
            rep = run_unseen_postfreeze_private_verifier_by_id(tid, tar_p)
            assert rep["strict_pass"] is False, f"Expected {tid} NaN stub to fail, got {rep}"

        # 3. Sealed `M8.8` `uf04` (`l2_only` AND `l1_l2`) MUST PASS (`strict_pass == True`, `max_gap <= 1e-5`)!
        uf04_snaps = [
            p
            for p in sorted(M88_EV_DIR.glob("**/uf04__certified_box_l0_sensor_selection/04_submitted_artifacts_snapshot/certified_l0_solver.py"))
            if "_branches_work" not in p.as_posix()
        ]
        assert len(uf04_snaps) == 2, f"Expected 2 sealed M8.8 uf04 snapshots, found {uf04_snaps}"
        for idx_s, snap_py in enumerate(uf04_snaps):
            tar_p = _make_tar_from_files(
                {"/holdout_task/certified_l0_solver.py": snap_py.read_text(encoding="utf-8")},
                td_p / f"m88_uf04_{idx_s}.tar",
            )
            rep_ok = run_unseen_postfreeze_private_verifier_by_id("uf04__certified_box_l0_sensor_selection", tar_p)
            assert rep_ok["strict_pass"] is True, f"Sealed M8.8 uf04 ({snap_py}) failed hardened verifier: {rep_ok}"
            assert rep_ok["verifier_metrics"]["max_gap"] <= 1e-10


# ============================================================================
# 3. P1-1 (`C2`): Exact IEEE-754 Temporal & Incumbent Planner Cache Identity
# ============================================================================


def test_m8_9_p1_1_planner_temporal_history_and_incumbent_mechanism_exact_cache_identity() -> None:
    # Exact GPT-6 `temporal_cache_control` from `reproduce_m88_compiler_cache_counterexamples.py`!
    graph = PersistentMechanismGraph()
    graph.register_mechanism(TypedMechanism("root", None, "shared_upstream_premise", ("a",), shared_weights={"a": 0.0}))
    mech = TypedMechanism("mem", "root", "temporal_memory", ("a",), temporal_tau=1.5, temporal_driver_col="a")
    graph.register_mechanism(mech)

    a = np.linspace(-2.0, 2.0, 81)
    train = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": (a * a)[:, None]}
    bad = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": np.full((81, 1), 10.0)}
    good = {"x": {"a": a}, "y": 2.5 * a * a, "past_driver_history": (a * a)[:, None]}

    planner = M8VariationalPlanner(graph)
    tm = planner.register_task(
        task_id="temporal_view",
        shared_upstream_id="root",
        feature_names=("a",),
        train_data=train,
        val_data=bad,
    )

    def _run_discover(val_d: dict[str, Any]) -> list[float]:
        base = evaluate_committed_task_model(tm, graph, val_d)
        cands = planner._discover_reusable_mechanism_candidates(
            tm, train, val_d, float(base["val_mean_log_predictive"]), float(base["variational_free_energy"])
        )
        return [float(c.val_log_predictive_gain) for c in cands]

    first_bad = _run_discover(bad)
    assert len(first_bad) == 1 and first_bad[0] < -100.0
    # Previously in M8.8, `good` collided in `mech.inapplicable_task_ids` and returned `[]`; in M8.9 it MUST return `[> 2.5]`!
    second_good = _run_discover(good)
    assert len(second_good) == 1 and second_good[0] > 2.5

    # Also verify exact IEEE-754 sensitivity (`nextafter` sub-`1e-8` change -> distinct negative-cache key in `mech.inapplicable_task_ids`)
    n_keys_0 = len(mech.inapplicable_task_ids)
    assert n_keys_0 >= 1
    bad_eps = {"x": {"a": a.copy()}, "y": (2.5 * a * a).copy(), "past_driver_history": np.full((81, 1), 10.0)}
    bad_eps["x"]["a"][0] = np.nextafter(bad_eps["x"]["a"][0], np.inf)
    _run_discover(bad_eps)
    n_keys_1 = len(mech.inapplicable_task_ids)
    assert n_keys_1 > n_keys_0

    # Also verify incumbent mechanism weight/parameter mutation changes negative reuse key even with same `task_id` & `model_version`!
    graph.mechanisms["root"].shared_weights["a"] = 1.25
    _run_discover(bad)
    assert len(mech.inapplicable_task_ids) > n_keys_1


# ============================================================================
# 4. P1-2 & P1-3 (`C3`, `C4`): Stale Source SHA Rejection & Runtime Bundle Digest
# ============================================================================


def test_m8_9_p1_2_and_p1_3_stale_declared_source_rejection_and_canonical_wave1_bundle_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    # --- Part A (`P1-2` / `C3`): Exact GPT-6 `stale_declared_source_control`! ---
    forest, archive, _, _ = m7a.load_autogenous_forest_bundle(V4_BUNDLE_PATH)
    m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=True)

    target = "autogenous_dispersive_wave_tomography.py"
    for _, rec in m7a.iter_live_archive_nodes(archive):
        for mod in rec.get("autogenous_wormhole_modules", []):
            if mod.get("filename") == target:
                old_sha = str(mod["compiled_code_sha256"])
                mod["python_code"] = str(mod["python_code"]) + "\ndef added_changed_source_semantics(x):\n    return x + 42\n"
                assert hashlib.sha256(mod["python_code"].encode("utf-8")).hexdigest() != old_sha

    class _NoModelCalls:
        def call_structured(self, **_: Any) -> dict[str, Any]:
            raise AssertionError("Model call must NOT occur when stale declared source SHA is rejected fail-closed!")

    res_stale = m7a.sublate_autogenous_forest_into_compositional_primitives(
        forest=forest,
        archive=archive,
        codex_client=_NoModelCalls(),
        allow_horizontal_wormholes=False,
    )
    aff_ev = next(e for e in res_stale["events"] if e.get("source_l2_module") == target)
    assert aff_ev["status"] == "rejected_stale_declared_source_sha_mismatch"
    assert res_stale["retained_existing_count"] == 5

    # --- Part B (`P1-3` / `C4`): Canonical runtime `declared-artifact-bundle-v1` digest matching! ---
    called_snaps: list[str] = []

    def _fake_compile(**kw: Any) -> dict[str, Any]:
        called_snaps.append(str(kw.get("artifact_snapshot_dir") or kw.get("snapshot_dir") or "ok"))
        return {
            "compiled_ok": False,
            "compilation_mode": "mock_skip",
            "filename": "wave1_mock.py",
            "function_names": [],
            "llm_tokens": {"prompt": 0, "cached": 0, "completion": 0, "wall_ms": 0.0},
        }

    monkeypatch.setattr(m7a, "compile_episode_dialectical_sublation", _fake_compile)
    monkeypatch.setattr(m7a, "consolidate_autogenous_episode_into_forest", lambda **kw: {"consolidated": False})
    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="512m", cpus="1", network_mode="none")
    try:
        m6f.exec_container_bash(
            cid,
            "mkdir -p /app/sub1 /app/sub2 && printf 'alpha = 11\\n' > /app/sub1/mod.py && printf 'beta = 22\\n' > /app/sub2/other.py",
            timeout_sec=10,
        )
        declared = ("/app/sub1/mod.py", "/app/sub2/other.py")
        real_runtime_sha = m6f.compute_declared_artifacts_digest(cid, declared)
        assert len(real_runtime_sha) == 64

        with tempfile.TemporaryDirectory(prefix="m89_w1_snap_") as td:
            ep_dir = Path(td) / "ep_real"
            m6f.save_submitted_artifacts_snapshot(cid, declared, ep_dir / "04a_repaired_antithesis_snapshot")
            dummy_task = m6f.FrontierBenchmarkTask(
                task_id="t_real_bundle",
                short_name="t_real_bundle",
                benchmark="Terminal-Bench-Science",
                wave="wave_1",
                scientific_domain="Physics",
                wormhole_family="fam_test",
                instruction_text="Test multi-file bundle identity.",
                agent_image_tag="mock",
                verifier_image_tag="mock",
                declared_artifacts=declared,
            )
            # Legacy M8.8 basename digest MUST FAIL, while real runtime `compute_declared_artifacts_digest` MUST MATCH!
            legacy_hb = hashlib.sha256(b"mod.py\0alpha = 11\n\0other.py\0beta = 22\n\0").hexdigest()
            s_legacy = {"ended_by_finish": True, "local_container_repair_summary": {"repaired_verified": True, "repaired_candidate_sha256": legacy_hb}}
            s_real = {"ended_by_finish": True, "local_container_repair_summary": {"repaired_verified": True, "repaired_candidate_sha256": real_runtime_sha}}

            monkeypatch.setattr(m7a, "ensure_clean_scipy_sandbox_image", lambda: None)
            monkeypatch.setattr(m7a.m6f, "start_detached_container", lambda *a, **kw: "mock_cid")
            monkeypatch.setattr(m7a.m6f, "stop_container_safely", lambda *a, **kw: None)
            monkeypatch.setattr(m7a.m6f, "FrontierCodexClient", lambda **kw: object())

            w1_out = m7a.consolidate_wave1_via_aufhebung_compiler(
                wave1_antithesis_episodes=[(dummy_task, s_legacy, ep_dir), (dummy_task, s_real, ep_dir)],
                forest=forest,
                archive=archive,
                codex_info={"available": True, "auth_mode": "chatgpt", "executable": "codex", "provider_backend": "chatgpt"},
                model="gpt-6.2-sol",
                run_id="test_m89_bundle",
            )
            assert w1_out["events"][0]["status"] == "skipped_missing_or_mismatched_repaired_snapshot"
            assert w1_out["events"][1]["resolved_source_kind"] == "repaired_active_snapshot"
            assert len(called_snaps) == 1
    finally:
        m6f.stop_container_safely(cid)


# ============================================================================
# 5. P1-4, P1-5, P1-6 (`D1..D4`): Delivery Transactions, Immutability & Retention
# ============================================================================


def test_m8_9_p1_4_p1_5_p1_6_delivery_transactions_module_immutability_scanner_and_retention_controls() -> None:
    skill_src = "K = 2\ndef scale(x):\n    return x * K\n"
    originals = {
        "/task/a.py": "from hegel_wormhole_skills.demo import scale\ndef run_a(x):\n    return scale(x)\n",
        "/task/b.py": "from hegel_wormhole_skills.demo import scale\ndef run_b(x):\n    return scale(x)\n",
    }

    # 1. Exact 4 `txn_case` write-back controls from `reproduce_delivery_controls.py`!
    def _run_txn_case(mode: str) -> dict[str, Any]:
        state = dict(originals)

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            if args[:2] == ["docker", "cp"]:
                src, dst = args[2:4]
                if src.startswith("mock:"):
                    Path(dst).write_text(state[src[len("mock:") :]], encoding="utf-8")
                    return subprocess.CompletedProcess(args, 0, b"", b"")
                local = Path(src)
                target = dst[len("mock:") :]
                if local.name == "out_1.py":
                    if mode == "timeout":
                        state[target] = "# partial_on_timeout\n"
                        raise subprocess.TimeoutExpired(args, timeout=20)
                    if mode == "partial_failed_current":
                        state[target] = "# partly copied bytes\n"
                    return subprocess.CompletedProcess(args, 1, b"", b"write failed")
                if mode == "rollback_failed" and local.name.startswith("rb_"):
                    return subprocess.CompletedProcess(args, 1, b"", b"rollback failed")
                state[target] = local.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, b"", b"")

        with (
            patch.object(m64, "scan_wormhole_imports_in_container", return_value={"scan_ok": True, "hits": list(originals), "error": ""}),
            patch.object(m64.subprocess, "run", side_effect=fake_run),
        ):
            res = m64.auto_package_wormhole_imports_in_container("mock", tuple(originals), {"demo": skill_src})
        return {"res": res, "state": state}

    for ok_rb_mode in ("nonzero_plain", "timeout", "partial_failed_current"):
        out_m = _run_txn_case(ok_rb_mode)
        assert out_m["res"]["ok"] is False
        assert out_m["res"]["rolled_back"] is True
        assert out_m["res"]["rollback_verified"] is True
        assert out_m["state"] == originals, f"Container state corrupted under {ok_rb_mode}: {out_m}"

    out_rb_fail = _run_txn_case("rollback_failed")
    assert out_rb_fail["res"]["ok"] is False
    assert out_rb_fail["res"]["rolled_back"] is False
    assert out_rb_fail["res"]["rollback_verified"] is False

    # 2. Module attribute mutation (`m.K = 5`) & bare module object escape (`return m`) MUST fail closed!
    mut_pkg = m64.package_self_contained_wormhole_dependencies(
        "import hegel_wormhole_skills.demo as m\nm.K = 5\ndef solve(x):\n    return m.scale(x)\n",
        {"demo": skill_src},
    )
    assert mut_pkg["ok"] is False and "skill_module_attribute_mutation_unsupported" in mut_pkg["error"]

    esc_pkg = m64.package_self_contained_wormhole_dependencies(
        "import hegel_wormhole_skills.demo as m\ndef leak():\n    return m\n",
        {"demo": skill_src},
    )
    assert esc_pkg["ok"] is False and "bare_skill_module_object_escape_unsupported" in esc_pkg["error"]

    # 3. Scanner payload validation (`''`, `'{}'`, `'null'`, `'not json'` -> `scan_ok == False`, zero `TypeError`)!
    for bad_stdout in ("", "{}", "null", "not json", "[123]"):
        with patch.object(m64.subprocess, "run", return_value=subprocess.CompletedProcess(["mock"], 0, bad_stdout, "")):
            sc = m64.scan_wormhole_imports_in_container("mock", ("/task/a.py",))
            assert sc["scan_ok"] is False
    with patch.object(m64.subprocess, "run", return_value=subprocess.CompletedProcess(["mock"], 0, "[]", "")):
        sc_ok = m64.scan_wormhole_imports_in_container("mock", ("/task/a.py",))
        assert sc_ok["scan_ok"] is True and sc_ok["hits"] == []

    # 4. Exact 6 `retention` controls (`valid_control`, `alias_name`, `changed_function`, `changed_constant`, `unused_public_function`, `missing_ref`)!
    mod_nm = "autogenous_prim_demo"
    unaliased = "from hegel_wormhole_skills.autogenous_prim_demo import scale\ndef solve(x):\n    return scale(x)\n"
    aliased = "from hegel_wormhole_skills.autogenous_prim_demo import scale as s\ndef solve(x):\n    return s(x)\n"
    dead = "from hegel_wormhole_skills.autogenous_prim_demo import scale\ndef unused_diagnostic(x):\n    return scale(x)\ndef solve(x):\n    return x\n"

    def _eval_ret(cand_src: str, *, mutate: str | None = None, ref_map: dict[str, str] | None = None, eps: tuple[str, ...] = ()) -> dict[str, Any]:
        p_res = m64.package_self_contained_wormhole_dependencies(cand_src, {mod_nm: skill_src})
        assert p_res["ok"] is True
        code = p_res["packaged_python_code"]
        if mutate == "body":
            code = code.replace("return x * K", "return x + K")
        elif mutate == "constant":
            code = code.replace("K = 2", "K = 999")
        with tempfile.TemporaryDirectory(prefix="m89_ret_") as td:
            Path(td, "solution.py").write_text(code, encoding="utf-8")
            return m64.inspect_final_deliverable_skill_retention(
                Path(td),
                ({mod_nm: skill_src} if ref_map is None else ref_map),
                declared_entrypoints=eps,
            )

    r_valid = _eval_ret(unaliased)
    assert r_valid["final_retained_l1_modules"] == [mod_nm]
    assert r_valid["entrypoint_reachable_l1_functions"] == [f"{mod_nm}.scale"]
    assert r_valid["all_retained_ast_match_source"] is True

    r_alias = _eval_ret(aliased)
    assert r_alias["entrypoint_reachable_l1_functions"] == [f"{mod_nm}.scale"]

    r_chg_fn = _eval_ret(unaliased, mutate="body")
    assert r_chg_fn["final_retained_l1_modules"] == []
    assert r_chg_fn["entrypoint_reachable_l1_functions"] == []
    assert r_chg_fn["all_retained_ast_match_source"] is False

    r_chg_const = _eval_ret(unaliased, mutate="constant")
    assert r_chg_const["final_retained_l1_modules"] == []
    assert r_chg_const["entrypoint_reachable_l1_functions"] == []
    assert r_chg_const["all_retained_ast_match_source"] is False

    r_dead_default = _eval_ret(dead)
    assert r_dead_default["entrypoint_reachable_l1_functions"] == []
    assert r_dead_default["any_public_function_reachable_functions"] == [f"{mod_nm}.scale"]
    r_dead_explicit = _eval_ret(dead, eps=("solve",))
    assert r_dead_explicit["entrypoint_reachable_l1_functions"] == []

    r_missing_ref = _eval_ret(unaliased, ref_map={})
    assert r_missing_ref["final_retained_l1_modules"] == []
    assert r_missing_ref["all_retained_ast_match_source"] is False


# ============================================================================
# 6. P1-7 (`C5`) & 6.2/6.3: Witness-Bound Nogood Resolution & Router Schema v2
# ============================================================================


def test_m8_9_p1_7_witness_bound_nogood_resolution_and_router_schema_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    # --- Part A: Router Feature Schema `m8_9_prefix_clean_v2` removes artificial `27/28` (`0.96`) cliff! ---
    assert ROUTER_FEATURE_SCHEMA_VERSION == "m8_9_prefix_clean_v2"
    st_28 = build_container_pre_action_state(
        task_id="t_full_budget",
        benchmark="Hegel-OOD-Holdout-12",
        wave="wave_1",
        primary_summary={
            "ended_by_finish": True,
            "turns_used": 28,
            "max_turns": 28,
            "skill_activation_count": 1,
            "nogoods_recorded": [],
        },
        contract_check={"accepted": True},
        public_metrics={
            "public_output_contract_ok": True,
            "id_renaming_invariant_ok": True,
            "public_calibration_ok": True,
        },
    )
    feat_vec = build_pre_action_feature_vector(st_28)
    assert float(feat_vec[5]) == 1.0
    dummy_nig = {
        "posterior_mean": np.zeros(6),
        "posterior_precision": np.eye(6),
        "prior_precision": np.eye(6),
        "alpha_n": 3.0,
        "beta_n": 1.0,
        "log_marginal_likelihood": 0.0,
    }
    planner_dump = M8OfflineContainerPlanner(nig_fit=dummy_nig, fail_nig_fit=dummy_nig).to_json_dict()
    assert planner_dump["feature_schema_version"] == "m8_9_prefix_clean_v2"

    # --- Part B: End-to-end 4-turn `run_frontier_episode` witness-bound nogood resolution test! ---
    ensure_clean_scipy_sandbox_image()
    toy_ho = HoldoutOODTask(
        task_id="toy_ng_witness",
        short_name="toy_ng_witness",
        wave="m89_test",
        scientific_domain="Physics",
        wormhole_family="fam_test",
        declared_artifacts=("/holdout_task/sol.py",),
        instruction_text="Implement `def solve_val(x):` inside `/holdout_task/sol.py` returning `2.0 * x`.",
        public_setup_py=(
            "open('/holdout_task/readme.txt', 'w').write('ok\\n')\n"
            "open('/holdout_task/validate_public_schema.py', 'w').write("
            "'import json, sys\\nsys.path.insert(0, \"/holdout_task\")\\nimport sol\\n"
            "ok_pt = (sol.solve_val(3.0) == 6.0)\\nok_gauge = (sol.solve_val(0.0) == 0.0 and ok_pt)\\n"
            "print(json.dumps({\"schema_valid\": bool(ok_pt), \"gauge_invariance_verified\": bool(ok_gauge)}))\\n"
            "sys.exit(0 if ok_pt else 1)\\n')\n"
        ),
        private_verifier_py="import json; print(json.dumps({'strict_pass': True}))\n",
    )
    f_task = hard_stress_task_to_frontier_task(toy_ho)

    scripted_turns = [
        # Turn 1: Write broken candidate & record `ng_t01` (`target_identifiability`)
        {
            "action_type": "execute_in_remote_docker",
            "bash_command": "printf 'def solve_val(x):\\n    return x\\n' > /holdout_task/sol.py",
            "final_answer": None,
            "rationale": "Initial linear ansatz refuted by calibration gain.",
            "considered_alternatives": [],
            "active_hypothesis_and_worst_slice_metric": "err=1.0",
            "nogood_update": {
                "has_new_nogood": True,
                "nogood_rung": "RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM",
                "risk_kind": "target_identifiability",
                "resolves_nogood_ids": [],
                "assumption_refuted": "Unit slope fails calibration.",
                "empirical_witness": "err=1.0",
                "causal_blanket_diagnosis": "Missing factor 2.",
                "sublated_pivot": "Scale by 2.0.",
            },
        },
        # Turn 2: Claim `resolves_nogood_ids=['ng_t01']` (`has_new_nogood=False`) on a FAILING command (`exit 1`)!
        # Must NOT resolve `ng_t01` and MUST NOT attach dangling `superseded_by='ng_t02'`!
        {
            "action_type": "execute_in_remote_docker",
            "bash_command": "python3 -c 'import sys; sys.exit(1)'",
            "final_answer": None,
            "rationale": "Attempted verification failed.",
            "considered_alternatives": [],
            "active_hypothesis_and_worst_slice_metric": "failed check",
            "nogood_update": {
                "has_new_nogood": False,
                "nogood_rung": "NONE",
                "risk_kind": "none",
                "resolves_nogood_ids": ["ng_t01"],
                "assumption_refuted": None,
                "empirical_witness": None,
                "causal_blanket_diagnosis": None,
                "sublated_pivot": None,
            },
        },
        # Turn 3: Write 1-point-matching but unanchored candidate (`x + 3.0`, so `solve_val(3.0) == 6.0` passes schema,
        # while `solve_val(0.0) == 3.0 != 0.0` fails host gauge check!) -> `ng_t01` MUST STILL stay pending!
        {
            "action_type": "execute_in_remote_docker",
            "bash_command": "printf 'def solve_val(x):\\n    return x + 3.0\\n' > /holdout_task/sol.py && python3 -c \"import sys; sys.path.insert(0, '/holdout_task'); import sol; assert sol.solve_val(3.0) == 6.0\"",
            "final_answer": None,
            "rationale": "Matched single point without gauge anchor.",
            "considered_alternatives": [],
            "active_hypothesis_and_worst_slice_metric": "single_pt_err=0.0",
            "nogood_update": {
                "has_new_nogood": False,
                "nogood_rung": "NONE",
                "risk_kind": "none",
                "resolves_nogood_ids": [],
                "assumption_refuted": None,
                "empirical_witness": None,
                "causal_blanket_diagnosis": None,
                "sublated_pivot": None,
            },
        },
        # Turn 4: Fix anchor gauge (`2.0 * x`) and verify (`exit 0`) -> host probe signs `gauge_invariance_verified=True` and `ng_t01` resolves!
        {
            "action_type": "execute_in_remote_docker",
            "bash_command": "printf 'def solve_val(x):\\n    return 2.0 * x\\n' > /holdout_task/sol.py && python3 -c \"import sys; sys.path.insert(0, '/holdout_task'); import sol; assert sol.solve_val(0.0) == 0.0 and sol.solve_val(3.0) == 6.0; print('anchor_gauge_identifiability_verified')\"",
            "final_answer": None,
            "rationale": "Verified anchor gauge identifiability and exact slope.",
            "considered_alternatives": [],
            "active_hypothesis_and_worst_slice_metric": "exact=0.0, u == l",
            "nogood_update": {
                "has_new_nogood": False,
                "nogood_rung": "NONE",
                "risk_kind": "none",
                "resolves_nogood_ids": [],
                "assumption_refuted": None,
                "empirical_witness": None,
                "causal_blanket_diagnosis": None,
                "sublated_pivot": None,
            },
        },
        # Turn 5: Submit!
        {
            "action_type": "submit_completed_task",
            "bash_command": None,
            "final_answer": "done",
            "rationale": "Exact certificate u == l verified.",
            "considered_alternatives": [],
            "active_hypothesis_and_worst_slice_metric": "u == l",
            "nogood_update": {
                "has_new_nogood": False,
                "nogood_rung": "NONE",
                "risk_kind": "none",
                "resolves_nogood_ids": [],
                "assumption_refuted": None,
                "empirical_witness": None,
                "causal_blanket_diagnosis": None,
                "sublated_pivot": None,
            },
        },
    ]
    observed_ng_state_at_turn3_prompt: list[dict[str, Any]] = []
    observed_ng_state_at_turn4_prompt: list[dict[str, Any]] = []

    class _FakeSeqClient:
        def __init__(self, **_: Any) -> None:
            self.idx = 0

        def call_structured(self, *, conversation_turns: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
            if self.idx == 2:
                # Inspect `hegel_m6_live_state` at the start of Turn 3 (right AFTER Turn 2's `exit 1` command!)
                footer = json.loads(conversation_turns[-1]["content"])
                observed_ng_state_at_turn3_prompt.extend(footer["hegel_m6_live_state"]["active_structural_nogoods"])
            elif self.idx == 3:
                # Inspect `hegel_m6_live_state` at the start of Turn 4 (right AFTER Turn 3's non-identifiability check!)
                footer = json.loads(conversation_turns[-1]["content"])
                observed_ng_state_at_turn4_prompt.extend(footer["hegel_m6_live_state"]["active_structural_nogoods"])
            out_p = scripted_turns[self.idx]
            self.idx += 1
            return {
                "ok": True,
                "parsed": out_p,
                "prompt_tokens": 100,
                "cached_tokens": 0,
                "completion_tokens": 30,
                "wall_ms": 2.0,
                "returned_model": "gpt-6.2-sol",
            }

    monkeypatch.setattr(m6f, "FrontierCodexClient", _FakeSeqClient)
    forest = PhilosophicalForest()
    hippo = HippoRAGWormholeIndex(forest)
    with tempfile.TemporaryDirectory(prefix="m89_ng_ep_") as td:
        ep_sum = m6f.run_frontier_episode(
            f_task,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model="gpt-6.2-sol",
            codex_info={"available": True, "auth_mode": "chatgpt", "executable": "codex", "provider_backend": "chatgpt"},
            forest=forest,
            forest_archive={},
            hippo_idx=hippo,
            episode_out_dir=Path(td) / "ep",
            max_turns=6,
            defer_official_verifier=True,
        )
        # Right after Turn 2's failed command (`exit 1`), `ng_t01` MUST STILL BE UNRESOLVED!
        assert observed_ng_state_at_turn3_prompt[0]["nogood_id"] == "ng_t01"
        assert observed_ng_state_at_turn3_prompt[0]["resolved"] is False
        assert observed_ng_state_at_turn3_prompt[0]["superseded_by"] is None
        # Right after Turn 3's generic 1-point check (`exit 0` without identifiability witness), `ng_t01` MUST STILL BE UNRESOLVED!
        assert observed_ng_state_at_turn4_prompt[0]["nogood_id"] == "ng_t01"
        assert observed_ng_state_at_turn4_prompt[0]["resolved"] is False

        # After Turn 4's explicit anchor/gauge identifiability witness (`exit 0`), `ng_t01` resolves with verified SHA!
        final_ng = ep_sum["nogoods_recorded"][0]
        assert final_ng["resolved"] is True
        assert final_ng["resolved_at_turn"] == 4
        assert final_ng["superseded_by"] is None
        assert final_ng["resolved_candidate_sha256"] == ep_sum["final_candidate_sha256"]
        assert len(ep_sum["nogood_resolution_events"]) == 1

        # Verify `[HOST ARTIFACT IDENTITY]` was emitted in trajectory!
        traj_txt = (Path(td) / "ep/02_human_readable_trajectory.md").read_text(encoding="utf-8")
        assert "[HOST ARTIFACT IDENTITY]" in traj_txt


# ============================================================================
# 7. M8.9A, M8.9B & D5: Progressive Disclosure, Pure-JSON CLI & Net-Value Gate
# ============================================================================


def test_m8_9a_and_m8_9b_progressive_contract_disclosure_pure_json_cli_and_net_value_gate() -> None:
    _, archive, _, _ = m7a.load_autogenous_forest_bundle(V4_BUNDLE_PATH)
    m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=True)
    recalled = [
        deepcopy(r)
        for _, r in m7a.iter_live_archive_nodes(archive)
        if r.get("autogenous_wormhole_modules")
    ][:2]

    uf03_q = unseen_postfreeze_tasks_by_id()["uf03__orthotropic_robin_heat_sink_inversion"].instruction_text
    _, man_dedup = m64.collect_recalled_executable_wormhole_sources(
        recalled,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text=uf03_q,
        disclosure_mode="dedup_full",
        staging_policy="all_compatible",
    )
    _, man_prog_all = m64.collect_recalled_executable_wormhole_sources(
        recalled,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text=uf03_q,
        disclosure_mode="progressive",
        staging_policy="all_compatible",
    )
    _, man_prog_nv, diag_nv = m64.collect_recalled_executable_wormhole_sources(
        recalled,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text=uf03_q,
        disclosure_mode="progressive",
        staging_policy="net_value_gated",
        return_diagnostics=True,
    )

    # 1. Lossless precondition deduplication: per-function `api_quick_reference` has NO duplicate `mathematical_preconditions`,
    #    while module-level `mathematical_preconditions` preserves 100% of clauses!
    for entry in man_dedup:
        for fn_ref in entry["api_quick_reference"]:
            assert "mathematical_preconditions" not in fn_ref
        if entry["granularity_tier"] == "L1_compositional_primitive":
            assert len(entry["mathematical_preconditions"]) >= 2

    # 2. Arm 1 (`dedup_full` + `all_compatible`) and Arm 2 (`progressive` + `all_compatible`) stage the EXACT SAME modules,
    #    while `progressive` cuts Turn-1 JSON size by >= 45%!
    assert [m["module_import"] for m in man_dedup] == [m["module_import"] for m in man_prog_all]
    len_dedup = len(json.dumps(man_dedup))
    len_prog = len(json.dumps(man_prog_all))
    assert len_prog <= int(0.55 * len_dedup), f"Expected >=45% size reduction, got {len_prog} vs {len_dedup}"

    # 3. Arm 3 (`net_value_gated`): on `uf03` (small 2D Robin Darcy), `autogenous_prim_conservative_spd_solve` is vetoed from Turn-1 manifest!
    staged_nv_mods = {m["module_import"] for m in man_prog_nv}
    assert "hegel_wormhole_skills.autogenous_prim_conservative_spd_solve" not in staged_nv_mods
    spd_eval = next(e for e in diag_nv["net_value_evaluations"] if e["module_stem"] == "autogenous_prim_conservative_spd_solve")
    assert spd_eval["recommend_turn1_staging"] is False
    assert any("robin" in v for v in spd_eval["veto_reasons"])

    # Whereas on `hs01` (box-L0 certificate) and `hs04` (multi-center reference controls), `box_ridge_bounds` & `reference_panel_balancing` ARE recommended!
    _, man_hs01 = m64.collect_recalled_executable_wormhole_sources(
        recalled,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text=HARD_STRESS_M89_BY_ID["hs01__correlated_box_l0_subcube_certificate"].instruction_text,
        disclosure_mode="progressive",
        staging_policy="net_value_gated",
    )
    assert any("autogenous_prim_box_ridge_bounds" in m["module_import"] for m in man_hs01)

    _, man_hs04 = m64.collect_recalled_executable_wormhole_sources(
        recalled,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text=HARD_STRESS_M89_BY_ID["hs04__missing_panel_multicenter_stratum_calibration"].instruction_text,
        disclosure_mode="progressive",
        staging_policy="net_value_gated",
    )
    assert any("autogenous_prim_reference_panel_balancing" in m["module_import"] for m in man_hs04)

    # 4. Container `--list` / `--describe` pure-JSON CLI check: ZERO arbitrary skill module import side effect!
    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="512m", cpus="1", network_mode="none")
    try:
        m64.stage_recalled_executable_wormholes_in_container(
            cid,
            recalled,
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=archive,
            task_query_text=uf03_q,
            disclosure_mode="progressive",
            staging_policy="net_value_gated",
        )
        # Inject a canary side-effect trap at top of `autogenous_prim_conservative_spd_solve.py` on disk:
        # `--list` and `--describe` MUST read `_skill_contracts.json` ONLY and NEVER trigger `/tmp/canary_imported`!
        m6f.exec_container_bash(
            cid,
            "sed -i '1i import pathlib; pathlib.Path(\"/tmp/canary_imported\").write_text(\"1\")' /tmp/hegel_wormhole_skills/autogenous_prim_conservative_spd_solve.py",
            timeout_sec=10,
        )
        out_list = m6f.exec_container_bash(cid, "PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --list", timeout_sec=10)
        assert out_list["exit_code"] == 0
        list_cards = [ln for ln in out_list["stdout"].splitlines() if "--describe " in ln]
        assert len(list_cards) >= 6
        assert any("autogenous_prim_conservative_spd_solve" in ln for ln in list_cards)

        out_desc = m6f.exec_container_bash(
            cid,
            "PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_conservative_spd_solve.conservative_grid_operator",
            timeout_sec=10,
        )
        assert out_desc["exit_code"] == 0
        parsed_desc = json.loads(out_desc["stdout"])
        assert parsed_desc["ok"] is True
        assert parsed_desc["matches"][0]["mathematical_preconditions"]

        canary_chk = m6f.exec_container_bash(cid, "test ! -f /tmp/canary_imported && echo NO_IMPORT", timeout_sec=10)
        assert "NO_IMPORT" in canary_chk["stdout"]

        insp_events = m64.harvest_container_skill_inspections(cid)
        delivered = [ev for ev in insp_events if ev.get("event") == "skill_contract_delivered"]
        assert len(delivered) == 2
        assert delivered[1]["target"] == "autogenous_prim_conservative_spd_solve.conservative_grid_operator"
    finally:
        m6f.stop_container_safely(cid)


# ============================================================================
# 8. D5 (`P2.3`): LocalContainerRepair Exposes Compact Skill Index Cards
# ============================================================================


def test_m8_9_local_repair_exposes_compact_skill_index_cards() -> None:
    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="512m", cpus="1", network_mode="none")
    try:
        m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=10)
        m6f.exec_container_bash(
            cid,
            "python3 -c \"import pickle; pickle.dump({'plates': [{'plate_id':'P1','wells':[{'row':0,'col':0,'well_type':'POS_KILL','fluorescence':10.0},{'row':0,'col':1,'well_type':'DMSO','fluorescence':110.0},{'row':1,'col':1,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':-2.0,'fluorescence':105.0},{'row':1,'col':2,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':0.0,'fluorescence':60.0},{'row':1,'col':3,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':2.0,'fluorescence':15.0}]}], 'gt': {'C1': {'log10_ic50_uM': 0.0, 'hill_slope': 1.0}}}, open('/holdout_task/public_screen.pkl','wb'))\"",
            timeout_sec=10,
        )
        m6f.exec_container_bash(cid, "printf 'def estimate_compound_hill_parameters(plates):\\n    return {}\\n' > /holdout_task/assay_ic50.py", timeout_sec=10)

        captured_calls: list[dict[str, Any]] = []

        class _SpyRepairClient:
            def call_structured(self, **kw: Any) -> dict[str, Any]:
                captured_calls.append(kw)
                return {
                    "ok": True,
                    "prompt_tokens": 40,
                    "cached_tokens": 0,
                    "completion_tokens": 25,
                    "wall_ms": 3.0,
                    "parsed": {
                        "defect_diagnosis": "Empty dict.",
                        "invariance_guard_name": "norm_guard",
                        "invariance_principle": "Normalize by plate controls.",
                        "repaired_python_code": (
                            "from hegel_wormhole_skills.autogenous_prim_reference_panel_balancing import inverse_group_frequency_weights\n"
                            "def estimate_compound_hill_parameters(plates):\n"
                            "    _ = inverse_group_frequency_weights([0, 0, 1])\n"
                            "    out = {}\n"
                            "    for p in plates:\n"
                            "        for w in p['wells']:\n"
                            "            if w['well_type'] == 'COMPOUND':\n"
                            "                out[w['compound_id']] = {'log10_ic50_uM': 0.0, 'hill_slope': 1.0}\n"
                            "    return out\n"
                        ),
                    },
                }

        toy_task = HoldoutOODTask(
            task_id="ho06__plate_evaporation_deconfounded_ic50",
            short_name="toy_rep_cards",
            wave="wave_1",
            scientific_domain="Assay",
            wormhole_family="fam_test",
            declared_artifacts=("/holdout_task/assay_ic50.py",),
            instruction_text="Estimate compound Hill parameters across plates and controls.",
            public_setup_py="",
            private_verifier_py="",
        )
        rep = execute_in_container_local_repair(
            probe_cid=cid,
            task=toy_task,
            public_probe_before={"symmetry_probe_family": "compound_renaming_and_well_permutation_equivariance"},
            codex_client=_SpyRepairClient(),
            staged_skill_sources_by_stem={
                "autogenous_prim_reference_panel_balancing": (
                    "import numpy as np\n"
                    "def inverse_group_frequency_weights(groups):\n"
                    "    g = np.asarray(groups)\n"
                    "    return np.ones(len(g), dtype=float) / max(len(g), 1)\n"
                )
            },
        )
        assert rep["repaired_verified"] is True
        assert "static dependency packager automatically bundles" in captured_calls[0]["system_prompt"]
        u_payload = json.loads(captured_calls[0]["conversation_turns"][0]["content"])
        assert len(u_payload["available_wormhole_skill_cards"]) >= 1
        assert "autogenous_prim_reference_panel_balancing" in u_payload["available_wormhole_skill_cards"][0]
    finally:
        m6f.stop_container_safely(cid)


# ============================================================================
# 9. Hegel-M89-HardStress-4: Reference Solvers Pass & Naive Controls Fail
# ============================================================================


def test_m8_9_hard_stress_4_reference_solvers_pass_and_naive_controls_fail() -> None:
    ensure_clean_scipy_sandbox_image()
    with tempfile.TemporaryDirectory(prefix="m89_hs_ver_") as td:
        td_p = Path(td)

        # --- HS01: Reference exact B&B subcube certificate solver vs unbranched 1-leaf naive control ---
        hs01_naive = (
            "import numpy as np\n"
            "def solve_certified_box_l0(problem):\n"
            "    X, y = np.asarray(problem['X'], float), np.asarray(problem['y'], float)\n"
            "    j0 = 0.5 * float(y @ y)\n"
            "    return {'weights': [0.0]*X.shape[1], 'objective': j0, 'certified_lower_bound': 0.0,\n"
            "            'certificate_leaves': [{'fixed_zero': [], 'fixed_nonzero': [], 'leaf_lower_bound': 0.0}]}\n"
        )
        r_hs01_bad = run_hard_stress_private_verifier_by_id(
            "hs01__correlated_box_l0_subcube_certificate",
            _make_tar_from_files({"/holdout_task/certified_box_l0.py": hs01_naive}, td_p / "hs01_bad.tar"),
        )
        assert r_hs01_bad["strict_pass"] is False

        hs01_ref = r'''
import heapq
import numpy as np
from scipy.optimize import minimize

def _solve_fixed_support(X, y, ridge, l0_pen, box, supp):
    p = X.shape[1]
    w = np.zeros(p, dtype=float)
    if supp:
        Xs = X[:, supp]
        G = Xs.T @ Xs + ridge * np.eye(len(supp))
        u0 = np.clip(np.linalg.solve(G + 1e-12 * np.eye(len(supp)), Xs.T @ y), -box, box)
        res = minimize(
            lambda u: 0.5 * float(np.sum((Xs @ u - y) ** 2)) + 0.5 * ridge * float(u @ u),
            u0,
            jac=lambda u: Xs.T @ (Xs @ u - y) + ridge * u,
            bounds=[(-box, box)] * len(supp),
            method="L-BFGS-B",
            options={"ftol": 1e-13, "gtol": 1e-11},
        )
        w[supp] = np.clip(res.x, -box, box)
    r = X @ w - y
    k0 = int(np.sum(np.abs(w) > 1e-10))
    return w, float(0.5 * float(r @ r) + 0.5 * ridge * float(w @ w) + l0_pen * k0)

def _perspective_leaf_lb(X, y, ridge, l0_pen, box, b0, b1):
    """Exact convex perspective relaxation lower bound on subcube (b0, b1) via L-BFGS-B + exact Fenchel dual certificate!"""
    p = X.shape[1]
    act = [j for j in range(p) if j not in b0]
    b1_set = set(b1)
    if not act:
        return 0.5 * float(y @ y), np.zeros(p, dtype=float)
    Xa = X[:, act]
    # Dualize residual r = y - Xa w_a: for nu in R^n,
    # L(nu) = nu^T y - 0.5 ||nu||^2 - sum_{j in b1} max_{|w|<=M} [c_j w - 0.5*ridge*w^2] + l0*|b1|
    #         - sum_{j in free} max(0, max_{|w|<=M} [c_j w - 0.5*ridge*w^2] - l0), where c = Xa^T nu!
    # Every nu in R^n yields a 100% mathematically sound lower bound! Optimize primal perspective smooth surrogate to get near-optimal nu = y - Xa w*!
    thresh = np.sqrt(2.0 * l0_pen / max(ridge, 1e-12)) if ridge > 0 else box + 1.0
    def _h_free(u_val):
        au = abs(u_val)
        if thresh <= box:
            if au <= thresh:
                return np.sqrt(2.0 * ridge * l0_pen) * au, np.sqrt(2.0 * ridge * l0_pen) * np.sign(u_val)
            return 0.5 * ridge * u_val * u_val + l0_pen, ridge * u_val
        slope = (0.5 * ridge * box * box + l0_pen) / box
        return slope * au, slope * np.sign(u_val)

    # Coordinate descent on exact convex perspective relaxation!
    w_a = np.zeros(len(act), dtype=float)
    r = y.copy()
    col_n2 = np.sum(Xa * Xa, axis=0)
    is_b1 = np.array([j in b1_set for j in act], dtype=bool)
    for _ in range(45):
        max_chg = 0.0
        for k in range(len(act)):
            old_w = w_a[k]
            rk = r + Xa[:, k] * old_w
            ck = float(Xa[:, k] @ rk)
            ak = col_n2[k]
            if is_b1[k]:
                nw = float(np.clip(ck / max(ak + ridge, 1e-12), -box, box))
            else:
                if thresh <= box:
                    tau = np.sqrt(2.0 * ridge * l0_pen)
                    if abs(ck) <= tau:
                        nw = 0.0
                    else:
                        cand_lin = (abs(ck) - tau) / max(ak, 1e-12) * np.sign(ck)
                        if abs(cand_lin) <= thresh:
                            nw = float(cand_lin)
                        else:
                            nw = float(np.clip(ck / max(ak + ridge, 1e-12), -box, box))
                else:
                    slope = (0.5 * ridge * box * box + l0_pen) / box
                    nw = 0.0 if abs(ck) <= slope else float(np.clip((abs(ck) - slope) / max(ak, 1e-12) * np.sign(ck), -box, box))
            if nw != old_w:
                r = rk - Xa[:, k] * nw
                w_a[k] = nw
                max_chg = max(max_chg, abs(nw - old_w))
        if max_chg < 1e-11:
            break

    # Exact Fenchel dual evaluation at nu = r (100% guaranteed sound lower bound for ANY r!)
    c_vec = Xa.T @ r
    dual_pen = 0.0
    for k in range(len(act)):
        ck = abs(float(c_vec[k]))
        if ridge > 1e-12:
            w_star = min(ck / ridge, box)
            conj_k = ck * w_star - 0.5 * ridge * w_star * w_star
        else:
            conj_k = ck * box
        if is_b1[k]:
            dual_pen += conj_k
        else:
            dual_pen += max(0.0, conj_k - l0_pen)
    lb = float(r @ y - 0.5 * float(r @ r) + l0_pen * len(b1) - dual_pen)
    w_full = np.zeros(p, dtype=float)
    w_full[act] = w_a
    return lb, w_full

def solve_certified_box_l0(problem: dict) -> dict:
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    ridge = float(problem["ridge"])
    l0_pen = float(problem["l0_penalty"])
    box = float(problem["box"])
    gap_tol = float(problem.get("gap_tol", 1e-6))
    p = X.shape[1]

    best_w, best_ub = _solve_fixed_support(X, y, ridge, l0_pen, box, [])
    lb0, w_rel0 = _perspective_leaf_lb(X, y, ridge, l0_pen, box, frozenset(), frozenset())
    supp0 = [j for j in range(p) if abs(w_rel0[j]) > 0.08 * box]
    w_try, ub_try = _solve_fixed_support(X, y, ridge, l0_pen, box, supp0)
    if ub_try < best_ub:
        best_w, best_ub = w_try, ub_try

    pq = [(lb0, 0, frozenset(), frozenset(), w_rel0)]
    uid = 1
    certified_leaves = []
    while pq:
        lb, _, b0, b1, w_rel = heapq.heappop(pq)
        if lb >= best_ub - gap_tol + 1e-10:
            certified_leaves.append({"fixed_zero": sorted(b0), "fixed_nonzero": sorted(b1), "leaf_lower_bound": float(lb)})
            continue
        free = [j for j in range(p) if (j not in b0) and (j not in b1)]
        if not free:
            w_leaf, ub_leaf = _solve_fixed_support(X, y, ridge, l0_pen, box, sorted(b1))
            if ub_leaf < best_ub:
                best_w, best_ub = w_leaf, ub_leaf
            certified_leaves.append({"fixed_zero": sorted(b0), "fixed_nonzero": sorted(b1), "leaf_lower_bound": float(min(lb, ub_leaf))})
            continue
        # Fractional score: branch on free variable with largest |w_rel[j]| (or highest correlation with residual)
        j_br = max(free, key=lambda j: (abs(w_rel[j]), abs(float(X[:, j] @ (y - X @ w_rel)))))
        for next_b0, next_b1 in ((b0 | {j_br}, b1), (b0, b1 | {j_br})):
            c_lb, c_w = _perspective_leaf_lb(X, y, ridge, l0_pen, box, next_b0, next_b1)
            supp_cand = sorted(set(next_b1) | {j for j in range(p) if (j not in next_b0) and abs(c_w[j]) > 0.15 * box})
            w_p, ub_p = _solve_fixed_support(X, y, ridge, l0_pen, box, supp_cand)
            if ub_p < best_ub:
                best_w, best_ub = w_p, ub_p
            if c_lb >= best_ub - gap_tol + 1e-10:
                certified_leaves.append({"fixed_zero": sorted(next_b0), "fixed_nonzero": sorted(next_b1), "leaf_lower_bound": float(c_lb)})
            else:
                heapq.heappush(pq, (c_lb, uid, next_b0, next_b1, c_w))
                uid += 1

    # Sibling compaction pass to keep certificate compact (`<= max_leaves`)!
    leaf_map = {(frozenset(lf["fixed_zero"]), frozenset(lf["fixed_nonzero"])): lf["leaf_lower_bound"] for lf in certified_leaves}
    changed = True
    while changed:
        changed = False
        items = list(leaf_map.items())
        used = set()
        next_map = {}
        for i, ((b0_a, b1_a), lb_a) in enumerate(items):
            if i in used:
                continue
            merged = False
            for v in b0_a:
                peer_key = (b0_a - {v}, b1_a | {v})
                if peer_key in leaf_map and peer_key not in used:
                    p_b0, p_b1 = b0_a - {v}, b1_a
                    p_lb, _ = _perspective_leaf_lb(X, y, ridge, l0_pen, box, p_b0, p_b1)
                    if p_lb >= best_ub - gap_tol:
                        next_map[(p_b0, p_b1)] = float(p_lb)
                        used.add((b0_a, b1_a)); used.add(peer_key)
                        changed = True; merged = True
                        break
            if not merged and (b0_a, b1_a) not in used:
                next_map[(b0_a, b1_a)] = lb_a
                used.add((b0_a, b1_a))
        leaf_map = next_map

    final_leaves = [
        {"fixed_zero": sorted(b0), "fixed_nonzero": sorted(b1), "leaf_lower_bound": float(lb)}
        for (b0, b1), lb in leaf_map.items()
    ]
    cert_lb = min(float(lf["leaf_lower_bound"]) for lf in final_leaves)
    return {
        "weights": best_w.tolist(),
        "objective": float(best_ub),
        "certified_lower_bound": float(min(cert_lb, best_ub)),
        "certificate_leaves": final_leaves,
    }
'''
        r_hs01_ok = run_hard_stress_private_verifier_by_id(
            "hs01__correlated_box_l0_subcube_certificate",
            _make_tar_from_files({"/holdout_task/certified_box_l0.py": hs01_ref}, td_p / "hs01_ok.tar"),
        )
        assert r_hs01_ok["strict_pass"] is True, f"HS01 ref failed: {r_hs01_ok}"

        # --- HS02: Reference rotated-tensor Robin + Huber/Cauchy IRLS solver vs naive OLS/scalar control ---
        hs02_bad = (
            "import numpy as np\n"
            "def infer_tensor_darcy_sources(problem):\n"
            "    M, nx, ny = problem['basis'].shape\n"
            "    return {'source_weights': np.zeros(M), 'sensor_offset': 0.0, 'predicted_field': np.zeros((nx, ny))}\n"
        )
        r_hs02_bad = run_hard_stress_private_verifier_by_id(
            "hs02__rotated_tensor_robin_darcy_inversion",
            _make_tar_from_files({"/holdout_task/tensor_darcy_robin.py": hs02_bad}, td_p / "hs02_bad.tar"),
        )
        assert r_hs02_bad["strict_pass"] is False

        hs02_ref = r'''
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def _assemble(problem):
    Lx, Ly = float(problem["Lx"]), float(problem["Ly"])
    k_xx, k_xy, k_yy = problem["k_xx"], problem["k_xy"], problem["k_yy"]
    sigma = problem["sigma"]
    gamma_r, u_amb = float(problem["gamma_robin"]), float(problem["u_ambient"])
    nx, ny = k_xx.shape
    hx, hy = Lx / nx, Ly / ny
    A = lil_matrix((nx * ny, nx * ny), dtype=float)
    b_amb = np.zeros(nx * ny, dtype=float)
    def idx(i, j):
        return int(np.clip(i, 0, nx - 1)) * ny + int(np.clip(j, 0, ny - 1))
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            A[row, row] += hx * hy * float(sigma[i, j])
            if i > 0:
                c = (hy / hx) * (2.0 * k_xx[i, j] * k_xx[i - 1, j] / (k_xx[i, j] + k_xx[i - 1, j]))
                A[row, row] += c; A[row, idx(i - 1, j)] -= c
                kxy = 0.5 * (k_xy[i - 1, j] + k_xy[i, j])
                for p in (i - 1, i):
                    A[row, idx(p, j + 1)] += 0.25 * kxy; A[row, idx(p, j - 1)] -= 0.25 * kxy
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if i < nx - 1:
                c = (hy / hx) * (2.0 * k_xx[i, j] * k_xx[i + 1, j] / (k_xx[i, j] + k_xx[i + 1, j]))
                A[row, row] += c; A[row, idx(i + 1, j)] -= c
                kxy = 0.5 * (k_xy[i, j] + k_xy[i + 1, j])
                for p in (i, i + 1):
                    A[row, idx(p, j + 1)] -= 0.25 * kxy; A[row, idx(p, j - 1)] += 0.25 * kxy
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if j > 0:
                c = (hx / hy) * (2.0 * k_yy[i, j] * k_yy[i, j - 1] / (k_yy[i, j] + k_yy[i, j - 1]))
                A[row, row] += c; A[row, idx(i, j - 1)] -= c
                kxy = 0.5 * (k_xy[i, j - 1] + k_xy[i, j])
                for q in (j - 1, j):
                    A[row, idx(i + 1, q)] += 0.25 * kxy; A[row, idx(i - 1, q)] -= 0.25 * kxy
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if j < ny - 1:
                c = (hx / hy) * (2.0 * k_yy[i, j] * k_yy[i, j + 1] / (k_yy[i, j] + k_yy[i, j + 1]))
                A[row, row] += c; A[row, idx(i, j + 1)] -= c
                kxy = 0.5 * (k_xy[i, j] + k_xy[i, j + 1])
                for q in (j, j + 1):
                    A[row, idx(i + 1, q)] -= 0.25 * kxy; A[row, idx(i - 1, q)] += 0.25 * kxy
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
    return A.tocsc(), b_amb, hx * hy

def infer_tensor_darcy_sources(problem: dict) -> dict:
    A, b_amb, vol = _assemble(problem)
    basis = np.asarray(problem["basis"], dtype=float)
    M, nx, ny = basis.shape
    s_ij = np.asarray(problem["sensor_ij"], dtype=int)
    y_obs = np.asarray(problem["y_obs"], dtype=float)
    alpha = float(problem.get("tikhonov_alpha", 1e-4))
    u0 = spsolve(A, b_amb).reshape(nx, ny)
    G_cols = np.column_stack([
        spsolve(A, vol * basis[m].ravel()).reshape(nx, ny)[s_ij[:, 0], s_ij[:, 1]]
        for m in range(M)
    ])
    Phi = np.column_stack([G_cols, np.ones(len(y_obs), dtype=float)])
    target = y_obs - u0[s_ij[:, 0], s_ij[:, 1]]
    reg = np.diag([alpha] * M + [0.0])
    # Robust Huber/L1 warm-start so iteration 0 is never pulled by +-4.0 Cauchy spikes!
    off0 = float(np.median(target))
    w_ir = 1.0 / np.maximum(np.abs(target - off0), 0.05)
    theta = np.zeros(M + 1, dtype=float)
    for _ in range(45):
        W = Phi * w_ir[:, None]
        theta = np.linalg.solve(Phi.T @ W + reg, W.T @ target)
        res = target - Phi @ theta
        scale = max(float(np.median(np.abs(res))) * 1.4826, 0.002)
        w_ir = 1.0 / (1.0 + (res / (1.5 * scale)) ** 2)
    w_src = theta[:M]
    off = float(theta[M])
    u_pred = spsolve(A, b_amb + vol * np.tensordot(w_src, basis, axes=(0, 0)).ravel()).reshape(nx, ny)
    return {"source_weights": w_src, "sensor_offset": off, "predicted_field": u_pred}
'''
        r_hs02_ok = run_hard_stress_private_verifier_by_id(
            "hs02__rotated_tensor_robin_darcy_inversion",
            _make_tar_from_files({"/holdout_task/tensor_darcy_robin.py": hs02_ref}, td_p / "hs02_ok.tar"),
        )
        assert r_hs02_ok["strict_pass"] is True, f"HS02 ref failed: {r_hs02_ok}"

        # --- HS03: Reference hereditary ZOH memory solver vs zero-past-history naive control ---
        hs03_zero_past = r'''
import math, numpy as np
def deconvolve_hereditary_driver(problem):
    # Deliberately ignores `past_driver_history` (sets m_r = 0 at start)!
    t_b = np.asarray(problem["blind_run"]["timestamps"], float)
    return {"moduli": [1.0, 1.0, 1.0], "recovered_driver": np.zeros_like(t_b), "predicted_blind_response": np.zeros_like(t_b)}
'''
        r_hs03_bad = run_hard_stress_private_verifier_by_id(
            "hs03__irregular_dt_hereditary_memory_deconvolution",
            _make_tar_from_files({"/holdout_task/hereditary_deconv.py": hs03_zero_past}, td_p / "hs03_bad.tar"),
        )
        assert r_hs03_bad["strict_pass"] is False

        hs03_ref = r'''
import math
import numpy as np

def _init_past_mem(taus, past_t, past_u):
    m0 = np.zeros(len(taus), dtype=float)
    for r, tau in enumerate(taus):
        mr = float(past_u[0])
        for k in range(1, len(past_t)):
            a = math.exp(-(float(past_t[k] - past_t[k - 1])) / float(tau))
            mr = a * mr + (1.0 - a) * float(past_u[k])
        m0[r] = mr
    return m0

def _memory_affine_op(taus, past_t, past_u, t):
    """Build m_r = f_r + L_r @ u on main window `t`!"""
    T = len(t)
    m_init = _init_past_mem(taus, past_t, past_u)
    f_list, L_list = [], []
    for r, tau in enumerate(taus):
        f = np.zeros(T, dtype=float)
        L = np.zeros((T, T), dtype=float)
        a0 = math.exp(-float(t[0] - past_t[-1]) / float(tau))
        f[0] = a0 * m_init[r]
        L[0, 0] = 1.0 - a0
        for k in range(1, T):
            ak = math.exp(-float(t[k] - t[k - 1]) / float(tau))
            f[k] = ak * f[k - 1]
            L[k, :k] = ak * L[k - 1, :k]
            L[k, k] = 1.0 - ak
        f_list.append(f)
        L_list.append(L)
    return f_list, L_list

def deconvolve_hereditary_driver(problem: dict) -> dict:
    taus = [float(x) for x in problem["taus"]]
    cal = problem["calibration_run"]
    t_c = np.asarray(cal["timestamps"], dtype=float)
    u_c = np.asarray(cal["driver_u"], dtype=float)
    y_c = np.asarray(cal["y_obs"], dtype=float)
    fc, Lc = _memory_affine_op(taus, cal["past_timestamps"], cal["past_driver_history"], t_c)
    m1_c = fc[0] + Lc[0] @ u_c
    m2_c = fc[1] + Lc[1] @ u_c
    Phi_c = np.column_stack([u_c, m1_c, m2_c, t_c - t_c[0], np.ones_like(t_c)])
    w_ir = np.ones_like(y_c)
    theta = np.zeros(5, dtype=float)
    for _ in range(40):
        W = Phi_c * w_ir[:, None]
        theta = np.linalg.lstsq(W, w_ir * y_c, rcond=None)[0]
        res = y_c - Phi_c @ theta
        sc = max(float(np.median(np.abs(res))) * 1.4826, 0.008)
        w_ir = 1.0 / (1.0 + (res / (1.5 * sc)) ** 2)
    E_hat = theta[:3]

    bl = problem["blind_run"]
    t_b = np.asarray(bl["timestamps"], dtype=float)
    y_b = np.asarray(bl["y_obs"], dtype=float)
    lam = float(bl["smoothness_lambda"])
    fb, Lb = _memory_affine_op(taus, bl["past_timestamps"], bl["past_driver_history"], t_b)
    T = len(t_b)
    H = E_hat[0] * np.eye(T) + E_hat[1] * Lb[0] + E_hat[2] * Lb[1]
    drift_b = float(bl["drift_slope"]) * (t_b - t_b[0]) + float(bl["drift_offset"])
    rhs_y = y_b - drift_b - E_hat[1] * fb[0] - E_hat[2] * fb[1]
    D = np.zeros((T - 1, T), dtype=float)
    dts = np.sqrt(np.maximum(np.diff(t_b), 1e-9))
    for k in range(T - 1):
        D[k, k] = -1.0 / dts[k]
        D[k, k + 1] = 1.0 / dts[k]
    u_rec = np.linalg.solve(H.T @ H + lam * (D.T @ D), H.T @ rhs_y)
    y_pred = H @ u_rec + E_hat[1] * fb[0] + E_hat[2] * fb[1] + drift_b
    return {"moduli": E_hat.tolist(), "recovered_driver": u_rec, "predicted_blind_response": y_pred}
'''
        r_hs03_ok = run_hard_stress_private_verifier_by_id(
            "hs03__irregular_dt_hereditary_memory_deconvolution",
            _make_tar_from_files({"/holdout_task/hereditary_deconv.py": hs03_ref}, td_p / "hs03_ok.tar"),
        )
        assert r_hs03_ok["strict_pass"] is True, f"HS03 ref failed: {r_hs03_ok}"

        # --- HS04: Reference missing-control-tolerant aligner + stratum-calibrated logistic vs unaligned pooled control ---
        hs04_unaligned = r'''
import numpy as np
from scipy.optimize import minimize
def calibrate_multicenter_assay(dataset):
    X_tr = np.nan_to_num(np.asarray(dataset["train"]["X"], float), nan=0.0)
    y_tr = np.asarray(dataset["train"]["y"], float)
    X_q = np.nan_to_num(np.asarray(dataset["query"]["X"], float), nan=0.0)
    Phi = np.column_stack([np.ones(len(X_tr)), X_tr])
    res = minimize(lambda w: float(np.sum(np.logaddexp(0.0, Phi @ w) - y_tr * (Phi @ w))) + 0.1*float(w @ w), np.zeros(9))
    w = res.x
    p_q = 1.0 / (1.0 + np.exp(-(w[0] + X_q @ w[1:])))
    return {"aligned_query_features": X_q, "query_probabilities": p_q, "feature_weights": w[1:]}
'''
        r_hs04_bad = run_hard_stress_private_verifier_by_id(
            "hs04__missing_panel_multicenter_stratum_calibration",
            _make_tar_from_files({"/holdout_task/multicenter_calibration.py": hs04_unaligned}, td_p / "hs04_bad.tar"),
        )
        assert r_hs04_bad["strict_pass"] is False

        hs04_ref = r'''
import numpy as np
from scipy.optimize import minimize

def _align_centers_with_sparse_controls(ref_cid, X_ref, target_cid, X_target):
    D = X_ref.shape[1]
    centers = sorted(set(np.unique(ref_cid).tolist()) | set(np.unique(target_cid).tolist()))
    global_med = np.nanmedian(X_ref, axis=0)
    global_mad = np.nanmedian(np.abs(X_ref - global_med[None, :]), axis=0) * 1.4826
    global_mad = np.where(np.isfinite(global_mad) & (global_mad > 1e-6), global_mad, 1.0)

    shift_map, scale_map = {}, {}
    for c in centers:
        Xc = X_ref[ref_cid == c]
        med_c = np.full(D, np.nan, dtype=float)
        sc_c = np.full(D, np.nan, dtype=float)
        valid_cols = []
        for d in range(D):
            col = Xc[:, d]
            fin = col[np.isfinite(col)]
            if len(fin) >= 2:
                m_d = float(np.median(fin))
                s_d = float(np.median(np.abs(fin - m_d)) * 1.4826)
                if s_d < 1e-6:
                    s_d = float(np.std(fin))
                if s_d > 1e-6:
                    med_c[d] = m_d
                    sc_c[d] = s_d
                    valid_cols.append(d)
        # Fallback for sparse/missing control columns (`< 2` finite obs): use center's median shift/scale across valid columns!
        fallback_shift = float(np.median(med_c[valid_cols])) if valid_cols else 0.0
        fallback_scale = float(np.median(sc_c[valid_cols])) if valid_cols else 1.0
        for d in range(D):
            if not np.isfinite(med_c[d]):
                med_c[d] = fallback_shift if valid_cols else float(global_med[d])
            if not np.isfinite(sc_c[d]) or sc_c[d] <= 1e-6:
                sc_c[d] = fallback_scale if valid_cols else float(global_mad[d])
        shift_map[int(c)] = med_c
        scale_map[int(c)] = sc_c

    Z_out = np.zeros_like(X_target, dtype=float)
    for i in range(len(X_target)):
        c = int(target_cid[i])
        z_row = (X_target[i] - shift_map[c]) / scale_map[c]
        Z_out[i] = np.where(np.isfinite(z_row), z_row, 0.0)
    return Z_out

def calibrate_multicenter_assay(dataset: dict) -> dict:
    ref_cid = np.asarray(dataset["reference_controls"]["center_id"], dtype=int)
    X_ref = np.asarray(dataset["reference_controls"]["X"], dtype=float)
    tr_cid = np.asarray(dataset["train"]["center_id"], dtype=int)
    tr_s = np.asarray(dataset["train"]["stratum"], dtype=int)
    X_tr = np.asarray(dataset["train"]["X"], dtype=float)
    y_tr = np.asarray(dataset["train"]["y"], dtype=float)
    q_cid = np.asarray(dataset["query"]["center_id"], dtype=int)
    q_s = np.asarray(dataset["query"]["stratum"], dtype=int)
    X_q = np.asarray(dataset["query"]["X"], dtype=float)

    Z_tr = _align_centers_with_sparse_controls(ref_cid, X_ref, tr_cid, X_tr)
    Z_q = _align_centers_with_sparse_controls(ref_cid, X_ref, q_cid, X_q)

    def _design(s_vec, Z):
        s_oh = np.column_stack([(s_vec == 0).astype(float), (s_vec == 1).astype(float), (s_vec == 2).astype(float)])
        return np.column_stack([s_oh, Z])

    Phi_tr = _design(tr_s, Z_tr)
    # Inverse center-stratum frequency balancing so confounded centers cannot bias intercept/stratum calibration
    keys = [f"{int(c)}_{int(s)}" for c, s in zip(tr_cid, tr_s)]
    cnts = {}
    for k in keys:
        cnts[k] = cnts.get(k, 0) + 1
    w_samp = np.array([1.0 / cnts[k] for k in keys], dtype=float)
    w_samp *= len(w_samp) / np.sum(w_samp)

    def _loss_and_grad(theta):
        logits = Phi_tr @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        nll = float(np.sum(w_samp * (np.logaddexp(0.0, logits) - y_tr * logits))) + 0.25 * float(theta[3:] @ theta[3:])
        grad = Phi_tr.T @ (w_samp * (p - y_tr))
        grad[3:] += 0.50 * theta[3:]
        return nll, grad

    opt = minimize(_loss_and_grad, np.zeros(Phi_tr.shape[1]), jac=True, method="L-BFGS-B")
    theta_hat = opt.x
    logits_q = _design(q_s, Z_q) @ theta_hat
    probs_q = 1.0 / (1.0 + np.exp(-np.clip(logits_q, -35.0, 35.0)))
    return {
        "aligned_query_features": Z_q,
        "query_probabilities": probs_q,
        "feature_weights": theta_hat[3:],
    }
'''
        r_hs04_ok = run_hard_stress_private_verifier_by_id(
            "hs04__missing_panel_multicenter_stratum_calibration",
            _make_tar_from_files({"/holdout_task/multicenter_calibration.py": hs04_ref}, td_p / "hs04_ok.tar"),
        )
        assert r_hs04_ok["strict_pass"] is True, f"HS04 ref failed: {r_hs04_ok}"
