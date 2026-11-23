"""Unit & Container Integration Tests for `Hegel M7.0` Autogenous Philosophical Forest & Blank-Seed Pretraining."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    NODE_TO_EXECUTABLE_WORMHOLE_MODULE,
    extract_pure_reusable_functions_from_python,
    run_host_mechanical_wormhole_gate_probes_v2,
    stage_recalled_executable_wormholes_in_container,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    build_blank_axiomatic_philosophical_forest,
    build_disjoint_pretraining_suite,
    consolidate_autogenous_episode_into_forest,
    ensure_clean_scipy_sandbox_image,
    iter_live_archive_nodes,
    lift_operator_pipeline_to_program,
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
    validate_and_decorate_autogenous_python_code,
    verify_autogenous_module_in_clean_container,
)


def test_composite_ast_extractor_defeats_leaf_glue_slicing_trap() -> None:
    """Verify bottom-of-file physics operators using top-level `scipy` & constants are extracted in topological order!"""
    sample_battle_script = '''
import math
import numpy as np
from scipy.signal import hilbert

REG_GAIN = 1.85

def _prep_a(x):
    return np.asarray(x, dtype=float)

def _prep_b(x):
    return _prep_a(x) - np.mean(x)

def _prep_c(x):
    return np.clip(x, -10.0, 10.0)

def _prep_d(x):
    return x * 2.0

def show(msg):
    return f"Result: {msg}"

def dispersive_shadow_tomography(traces, dists, reg_gain=1.85):
    """High-level composite physical operator sitting at the bottom of solution.py."""
    arr = _prep_b(traces)
    env = np.abs(hilbert(arr, axis=0))
    pk = np.max(env, axis=0) * np.sqrt(np.maximum(dists, 0.05))
    log_y = np.log(np.maximum(pk, 1e-12))
    A = np.column_stack([np.ones_like(dists), dists])
    coef = np.linalg.solve(A.T @ A + reg_gain * np.eye(2), A.T @ log_y)
    return log_y - (A @ coef)
'''
    extracted = extract_pure_reusable_functions_from_python(sample_battle_script, max_funcs=4)
    names = [nm for nm, _ in extracted]
    assert "dispersive_shadow_tomography" in names
    assert "show" not in names
    # Topological order: `_prep_a` -> `_prep_b` -> `dispersive_shadow_tomography`!
    assert names.index("_prep_a") < names.index("_prep_b") < names.index("dispersive_shadow_tomography")

    # Verify against real sealed Wave-1 `baseline-free-localization` snapshot if present in workspace
    repo_root = Path(__file__).resolve().parents[1]
    real_bf = (
        repo_root
        / "docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/runs/gpt-6.2-sol__Hegel_M6_9_A_TwoBranch_LiveGPT/tb__baseline-free-localization/04_submitted_artifacts_snapshot/solution.py"
    )
    if real_bf.is_file():
        bf_fns = [nm for nm, _ in extract_pure_reusable_functions_from_python(real_bf.read_text(encoding="utf-8"), max_funcs=8)]
        # Previously ONLY exported `['_prepare', '_sample', '_ridge', '_path_design']`!
        assert any(k in bf_fns for k in ("localize", "_dispersive_scatter", "_shadow_map", "_scatter_image", "_fit_dispersion")), bf_fns


def test_blank_seed_forest_zero_handcrafted_skills_and_dynamic_bifurcation() -> None:
    """Verify Blank-Seed forest starts with 4 axioms, mounts 0 `SKILL_01..07`, splits nodes on preconditions, and anti-unifies across trees."""
    forest, archive, meta_dag = build_blank_axiomatic_philosophical_forest()
    assert len(forest.nodes) == 4
    assert len(forest.horizontal_wormhole_edges) == 0

    # Confirm host gate & staging refuse ALL foundational SKILL_01..07 on autogenous nodes
    root_notes = [rec for _, rec in iter_live_archive_nodes(archive)]
    gate_res = run_host_mechanical_wormhole_gate_probes_v2("mock", ("/app/solution.py",), root_notes)
    assert gate_res["overall_status"] == "not_applicable"
    staged_empty = stage_recalled_executable_wormholes_in_container("mock", root_notes)
    assert staged_empty == []

    # Consolidate Episode 1 (Tree A02: Dispersive Wave Tomography)
    ok1, _, pub1, pkg1 = validate_and_decorate_autogenous_python_code(
        '''
from scipy.signal import hilbert

def ridge_nuisance_shadow(traces: np.ndarray, dists: np.ndarray, reg: float = 1.8) -> np.ndarray:
    """Project out global amplitude decay with strong ridge penalty."""
    env = np.max(np.abs(hilbert(traces, axis=0)), axis=0) * np.sqrt(np.maximum(dists, 1e-2))
    y = np.log(np.maximum(env, 1e-12))
    X = np.column_stack([np.ones_like(dists), dists])
    beta = np.linalg.solve(X.T @ X + reg * np.eye(2), X.T @ y)
    return X @ beta - y
''',
        "autogenous_wave_shadow",
    )
    assert ok1 and pub1 == ["ridge_nuisance_shadow"]

    comp1 = {
        "compiled_ok": True,
        "compilation_mode": "llm_aufhebung_container_verified",
        "episode_id": "pt01_wave_a",
        "module_stem": "autogenous_wave_shadow",
        "filename": "autogenous_wave_shadow.py",
        "function_names": pub1,
        "exported_apis": "ridge_nuisance_shadow(traces, dists, reg=1.8)",
        "packaged_python_code": pkg1,
        "compiled_code_sha256": "a" * 64,
        "sublated_methodology_principle": "Align group slowness and use ridge >= 1.5 so near-sensor shadows are not absorbed.",
        "routed_signatures": ["sig:guided_wave_dispersion_and_geometry_transfer"],
        "scope_preconditions": [{"field": "reg_sensor", "op": ">=", "value": 1.5}],
        "regime_bifurcation": {
            "should_split_node": False,
            "regime_axis": "none",
            "branch_guard_op": ">=",
            "branch_guard_value": 0.0,
            "counter_regime_warning": "",
        },
        "operator_pipeline_stages": [
            {"stage_kind": "frontend", "numeric_param": 1.0},
            {"stage_kind": "nuisance_projection", "numeric_param": 1.8},
            {"stage_kind": "robust_inversion", "numeric_param": 0.15},
            {"stage_kind": "cross_slice_gate", "numeric_param": 4.0},
        ],
    }
    res1 = consolidate_autogenous_episode_into_forest(
        forest=forest,
        archive=archive,
        compilation=comp1,
        nogoods_recorded=[{"nogood_rung": "RUNG_3", "assumption_refuted": "weak gain ridge", "empirical_witness": "75mm err", "sublated_pivot": "reg >= 1.5"}],
    )
    assert res1["growth_action"] == "spawned_specialized_mid_node"
    assert len(forest.nodes) == 5

    # Consolidate Episode 1b (Conflicting matched-twin sub-regime -> Precondition Bifurcation!)
    comp1b = dict(comp1)
    comp1b["episode_id"] = "pt01_wave_matched_split"
    comp1b["module_stem"] = "autogenous_matched_coda_gate"
    comp1b["filename"] = "autogenous_matched_coda_gate.py"
    comp1b["regime_bifurcation"] = {
        "should_split_node": True,
        "regime_axis": "late_coda_rho",
        "branch_guard_op": ">",
        "branch_guard_value": 0.75,
        "counter_regime_warning": "When late_coda_rho <= 0.75, full-wave subtraction creates wall-echo ghosts!",
    }
    res1b = consolidate_autogenous_episode_into_forest(forest=forest, archive=archive, compilation=comp1b, nogoods_recorded=[])
    assert res1b["growth_action"] == "bifurcated_precondition_child_node"
    assert len(forest.nodes) == 6
    split_cc = forest.nodes[res1b["target_node_id"]]["claim_contract"]
    assert any(p["field"] == "late_coda_rho" and p["value"] == 0.75 for p in split_cc["preconditions"])

    # Consolidate Episode 2 under a DIFFERENT tree (`tree_A03_regularization`) with matching pipeline skeleton!
    ok2, _, pub2, pkg2 = validate_and_decorate_autogenous_python_code(
        '''
def huber_creep_stiffness(s_app: float, creep_rate: float, unload_rate: float) -> float:
    """Feng-Ngan viscoelastic hold-creep compliance correction."""
    inv_s = (1.0 / max(s_app, 1e-9)) - (creep_rate / max(abs(unload_rate), 1e-9))
    return 1.0 / max(inv_s, 1e-9)
''',
        "autogenous_creep_calibration",
    )
    assert ok2
    comp2 = {
        "compiled_ok": True,
        "compilation_mode": "llm_aufhebung_container_verified",
        "episode_id": "pt02_creep",
        "module_stem": "autogenous_creep_calibration",
        "filename": "autogenous_creep_calibration.py",
        "function_names": pub2,
        "exported_apis": "huber_creep_stiffness(s_app, creep_rate, unload_rate)",
        "packaged_python_code": pkg2,
        "compiled_code_sha256": "b" * 64,
        "sublated_methodology_principle": "Subtract frame compliance and hold-creep drift before fitting unloading power law.",
        "routed_signatures": ["sig:robust_physical_calibration_and_safety_margins"],
        "scope_preconditions": [],
        "regime_bifurcation": {
            "should_split_node": False,
            "regime_axis": "none",
            "branch_guard_op": ">=",
            "branch_guard_value": 0.0,
            "counter_regime_warning": "",
        },
        "operator_pipeline_stages": [
            {"stage_kind": "frontend", "numeric_param": 2.0},
            {"stage_kind": "nuisance_projection", "numeric_param": 0.08},
            {"stage_kind": "robust_inversion", "numeric_param": 1.35},
            {"stage_kind": "cross_slice_gate", "numeric_param": 3.0},
        ],
    }
    res2 = consolidate_autogenous_episode_into_forest(forest=forest, archive=archive, compilation=comp2, nogoods_recorded=[])
    assert res2["growth_action"] == "spawned_specialized_mid_node"
    # Cross-tree operator anti-unification MUST automatically discover SharedBaseSubtree & HorizontalWormholeEdge!
    assert len(forest.shared_base_subtrees) >= 1
    assert len(forest.horizontal_wormhole_edges) >= 1

    # Verify serialization & reload + HippoRAG retrieval
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "bundle.json"
        meta_s = serialize_autogenous_forest_bundle(forest=forest, archive=archive, meta_dag=meta_dag, out_path=bp)
        f_reloaded, a_reloaded, dag_reloaded, _ = load_autogenous_forest_bundle(bp)
        assert len(f_reloaded.nodes) == len(forest.nodes)
        assert len(f_reloaded.horizontal_wormhole_edges) == len(forest.horizontal_wormhole_edges)
        assert meta_s["bundle_sha256"] == a_reloaded["__autogenous_blank_seed__"]["loaded_bundle_sha256"]
        idx = HippoRAGWormholeIndex(f_reloaded)
        r_wave = idx.retrieve(["sig:guided_wave_dispersion_and_geometry_transfer"], task_view={"goal_kind": "scientific_terminal_discovery", "tolerance": 1.0}, top_k=3)
        rec_ids = [c["node_id"] for c in r_wave["candidates"]]
        assert any(nid.startswith("node_auto_") for nid in rec_ids)
        # Ensure ZERO foundational SKILL_01..07 node IDs exist anywhere!
        assert set(f_reloaded.nodes.keys()).isdisjoint(set(NODE_TO_EXECUTABLE_WORMHOLE_MODULE.keys()))


def test_clean_scipy_sandbox_and_pretraining_suite_zero_leakage() -> None:
    """Verify `hegel_clean_scipy_sandbox:v1` has ZERO benchmark dirs and runs unit-test verification + all 6 PT setups cleanly."""
    img = ensure_clean_scipy_sandbox_image()
    assert img == CLEAN_SCIPY_SANDBOX_IMAGE

    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2")
    try:
        chk = m6f.exec_container_bash(
            cid,
            "test ! -d /app && test ! -d /data && test ! -d /workspace && test ! -d /tests && echo CLEAN_ROOTFS_OK",
            timeout_sec=15,
        )
        assert "CLEAN_ROOTFS_OK" in chk["stdout"]

        # Verify container unit-test gate (`python3 -I` + `@track_wormhole_skill` telemetry!)
        ok, _, pub_fns, pkg = validate_and_decorate_autogenous_python_code(
            "from scipy.signal import hilbert\ndef env_peak(x: np.ndarray) -> float:\n    return float(np.max(np.abs(hilbert(x))))\n",
            "autogenous_env_test",
        )
        assert ok
        ut = verify_autogenous_module_in_clean_container(
            sandbox_cid=cid,
            mod_stem="autogenous_env_test",
            packaged_module_code=pkg,
            synthetic_unit_test_code="import numpy as np\nt = np.linspace(0, 1, 200)\nv = env_peak(2.5 * np.sin(20 * np.pi * t))\nassert 2.3 < v < 2.7\n",
            public_fn_names=pub_fns,
        )
        assert ut["passed"] is True
        assert ut["telemetry_verified"] is True

        # Smoke-check all 6 disjoint pretraining workspace setups inside the clean sandbox
        for pt in build_disjoint_pretraining_suite():
            m6f.exec_container_bash(cid, "rm -rf /pretrain_task/* && mkdir -p /pretrain_task", timeout_sec=10)
            ex = m6f.exec_container_bash(
                cid,
                f"python3 - <<'PY'\n{pt.setup_public_workspace_py}\nprint('PT_SETUP_OK:{pt.short_name}')\nPY",
                timeout_sec=35,
            )
            assert ex["exit_code"] == 0 and f"PT_SETUP_OK:{pt.short_name}" in ex["stdout"], (pt.task_id, ex["stderr"])
    finally:
        m6f.stop_container_safely(cid)
