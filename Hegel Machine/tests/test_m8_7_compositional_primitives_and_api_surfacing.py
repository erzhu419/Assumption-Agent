"""Hegel M8.7 Two-Tier Compositional Primitive Sublation & API Quick-Reference Verification Suite.

Covers:
  1. Strict `require_pretraining_only=True` contamination guard: rejects any forest archive tainted with `ho*`, `tb__*`,
     `wave1_`, or non-`autogenous_only` nodes while accepting clean `pt01..pt06` pretraining archives.
  2. Two-attempt `sublate_autogenous_forest_into_compositional_primitives` gate inside real air-gapped
     `hegel_clean_scipy_sandbox:v1` (`network_mode='none'`): mechanically rejects `sklearn` on attempt 1, verifies pure
     `numpy`/`scipy` `autogenous_prim_*` + `@track_wormhole_skill` telemetry on attempt 2, retains `synthetic_unit_test_code`
     and `source_candidate_sha256`, and prepends `L1_compositional_primitive` modules ahead of `L2` operators.
  3. `extract_skill_api_quick_reference` & `collect_recalled_executable_wormhole_sources`: verifies `L1`-first staging order
     across recalled notes, AST signature + 1-sentence docstring extraction, and end-to-end `M8.6` clean-room static
     closure packaging of an `autogenous_prim_*` import after deleting `/tmp/hegel_wormhole_skills`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import pytest

from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    auto_package_wormhole_imports_in_container,
    collect_recalled_executable_wormhole_sources,
    extract_skill_api_quick_reference,
    find_uninlined_wormhole_imports_in_container,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    verify_candidate_artifact_contract_in_container,
)
from hegel_machine.cognition.m6_frontier_real_benchmarks import (
    compute_declared_artifacts_digest,
    exec_container_bash,
    start_detached_container,
    stop_container_safely,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_7_pretraining_purity_guard_rejects_holdout_or_frontier_contamination() -> None:
    """Ensure `sublate_autogenous_forest_into_compositional_primitives(require_pretraining_only=True)` fails closed on Holdout/TB leakage."""
    bundle_fp = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, _, _ = m7a.load_autogenous_forest_bundle(bundle_fp)

    # Corrupt one node with a Holdout episode tag -> MUST raise ValueError!
    tainted_arch = deepcopy(archive)
    mid_nid = next(nid for nid, r in m7a.iter_live_archive_nodes(tainted_arch) if r.get("layer_role") == "mid")
    tainted_arch[mid_nid]["supported_by"].append("ho06__plate_evaporation_deconfounded_ic50")
    with pytest.raises(ValueError, match="Benchmark/Holdout provenance"):
        m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=forest,
            archive=tainted_arch,
            codex_client=None,
            require_pretraining_only=True,
        )

    # Corrupt `autogenous_only = False` -> MUST raise ValueError!
    non_auto_arch = deepcopy(archive)
    non_auto_arch[mid_nid]["autogenous_only"] = False
    with pytest.raises(ValueError, match="autogenous_only"):
        m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=forest,
            archive=non_auto_arch,
            codex_client=None,
            require_pretraining_only=True,
        )


def test_m8_7_sklearn_rejection_container_verified_l1_primitive_and_m86_static_packaging() -> None:
    """Verify `L1` primitive `sklearn` rejection, container unit-test verification, L1-first staging, and M8.6 clean-room packaging."""
    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    # Seed a single synthetic `pt` L2 node (simulating `pt03` with `sklearn` in the coarse L2 module)
    l2_demo_code = (
        '"""Demo L2 coarse module."""\n'
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from . import track_wormhole_skill\n\n"
        '@track_wormhole_skill("autogenous_coarse_panel_demo")\n'
        "def coarse_panel_fn(X: np.ndarray, groups: np.ndarray) -> np.ndarray:\n"
        "    return np.asarray(X, float) - np.mean(X, axis=0, keepdims=True)\n"
    )
    import hashlib as _hl
    l2_true_sha = _hl.sha256(l2_demo_code.encode("utf-8")).hexdigest()
    l2_comp = {
        "compiled_ok": True,
        "compilation_mode": "llm_aufhebung_container_verified",
        "episode_id": "pt03_control_panel_demo",
        "module_stem": "autogenous_coarse_panel_demo",
        "filename": "autogenous_coarse_panel_demo.py",
        "function_names": ["coarse_panel_fn"],
        "exported_apis": "coarse_panel_fn(X, groups)",
        "packaged_python_code": l2_demo_code,
        "compiled_code_sha256": l2_true_sha,
        "source_candidate_sha256": "b" * 64,
        "synthetic_unit_test_code": "assert True\n",
        "synthetic_unit_test_sha256": "c" * 64,
        "unit_test_receipt": {"passed": True, "mode": "container_verified"},
        "sublated_methodology_principle": "Estimate nuisance offset/surface strictly on control coordinates before target inversion.",
        "routed_signatures": ["sig:cohort_transfer_and_metadata_imputation"],
        "scope_preconditions": [],
        "regime_bifurcation": {"should_split_node": False, "regime_axis": "none", "branch_guard_op": ">=", "branch_guard_value": 0.0, "counter_regime_warning": ""},
        "operator_pipeline_stages": [{"stage_kind": "nuisance_projection", "numeric_param": 1.0}, {"stage_kind": "robust_inversion", "numeric_param": 1.0}],
    }
    cons = m7a.consolidate_autogenous_episode_into_forest(
        forest=forest, archive=archive, compilation=l2_comp, nogoods_recorded=[], allow_horizontal_wormholes=False
    )
    target_nid = cons["target_node_id"]

    class _TwoAttemptMockPrimitiveClient:
        def __init__(self) -> None:
            self.calls = 0

        def call_structured(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                # Attempt 1 tries to import `sklearn` -> MUST be rejected by `extra_forbidden_tokens=("sklearn",)`!
                return {
                    "ok": True,
                    "prompt_tokens": 80,
                    "cached_tokens": 0,
                    "completion_tokens": 40,
                    "wall_ms": 10.0,
                    "parsed": {
                        "primitive_module_stem": "autogenous_prim_bad_sklearn",
                        "exported_api_signatures": ["bad_fn(X)"],
                        "compositional_rationale": "Bad sklearn import.",
                        "canonical_python_code": "from sklearn.linear_model import Ridge\ndef bad_fn(X):\n    return X\n",
                        "synthetic_unit_test_code": "def run_self_test(): pass\nrun_self_test()\n",
                    },
                }
            return {
                "ok": True,
                "prompt_tokens": 110,
                "cached_tokens": 0,
                "completion_tokens": 75,
                "wall_ms": 15.0,
                "parsed": {
                    "primitive_module_stem": "autogenous_prim_nuisance_projection",
                    "exported_api_signatures": [
                        "project_control_polynomial_surface(coords: np.ndarray, values: np.ndarray, control_mask: np.ndarray, *, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]"
                    ],
                    "compositional_rationale": "Decouple 1D/2D/ND control-subset quadratic surface detrending from tabular group IDs.",
                    "canonical_python_code": (
                        "import numpy as np\n\n"
                        "def _quad_design(xy: np.ndarray) -> np.ndarray:\n"
                        "    arr = np.asarray(xy, dtype=float)\n"
                        "    if arr.ndim == 1:\n"
                        "        arr = arr[:, None]\n"
                        "    cols = [np.ones(arr.shape[0], dtype=float), arr, arr ** 2]\n"
                        "    return np.column_stack(cols)\n\n"
                        "def project_control_polynomial_surface(\n"
                        "    coords: np.ndarray,\n"
                        "    values: np.ndarray,\n"
                        "    control_mask: np.ndarray,\n"
                        "    *,\n"
                        "    ridge: float = 1e-4,\n"
                        ") -> tuple[np.ndarray, np.ndarray]:\n"
                        '    """Fit a ridge-regularized quadratic nuisance surface on `control_mask` rows of `(N, D)` `coords` and return `(detrended_values, fitted_surface)`."""\n'
                        "    X = _quad_design(coords)\n"
                        "    y = np.asarray(values, dtype=float).ravel()\n"
                        "    m = np.asarray(control_mask, dtype=bool).ravel()\n"
                        "    Xm, ym = X[m], y[m]\n"
                        "    gram = Xm.T @ Xm + float(ridge) * np.eye(X.shape[1])\n"
                        "    gram[0, 0] -= float(ridge)\n"
                        "    beta = np.linalg.solve(gram, Xm.T @ ym)\n"
                        "    surf = X @ beta\n"
                        "    return y - surf, surf\n"
                    ),
                    "synthetic_unit_test_code": (
                        "import numpy as np\n"
                        "def run_self_test():\n"
                        "    rng = np.random.default_rng(42)\n"
                        "    xy = rng.uniform(-1.0, 1.0, size=(80, 2))\n"
                        "    ctrl = np.arange(80) < 35\n"
                        "    bg = 2.0 - 0.5 * xy[:, 0] + 0.8 * (xy[:, 1] ** 2)\n"
                        "    sig = bg + np.where(ctrl, 0.0, -1.25)\n"
                        "    det, surf = project_control_polynomial_surface(xy, sig, ctrl, ridge=1e-9)\n"
                        "    assert np.max(np.abs(det[~ctrl] - (-1.25))) < 1e-6\n"
                        "    assert np.all(np.isfinite(surf))\n"
                        "run_self_test()\n"
                    ),
                },
            }

    mock_cli = _TwoAttemptMockPrimitiveClient()
    res = m7a.sublate_autogenous_forest_into_compositional_primitives(
        forest=forest,
        archive=archive,
        codex_client=mock_cli,
        run_id="test_m87",
        require_pretraining_only=True,
        allow_horizontal_wormholes=False,
    )
    assert mock_cli.calls == 2
    assert res["compiled_primitive_count"] == 1
    mods_after = archive[target_nid]["autogenous_wormhole_modules"]
    assert mods_after[0]["granularity_tier"] == "L1_compositional_primitive"
    assert mods_after[0]["filename"] == "autogenous_prim_nuisance_projection.py"
    assert mods_after[0]["source_candidate_sha256"] == l2_true_sha
    assert "def run_self_test" in mods_after[0]["synthetic_unit_test_code"]

    # Verify `collect_recalled_executable_wormhole_sources` stages `L1` FIRST and extracts `api_quick_reference`
    mods_map, manifest = collect_recalled_executable_wormhole_sources([archive[target_nid]], include_foundational_skills=False)
    assert manifest[0]["granularity_tier"] == "L1_compositional_primitive"
    assert manifest[0]["module_import"] == "hegel_wormhole_skills.autogenous_prim_nuisance_projection"
    qref = manifest[0]["api_quick_reference"]
    assert len(qref) == 1
    assert "project_control_polynomial_surface" in qref[0]["signature"]
    assert "quadratic nuisance surface" in qref[0]["doc_summary"]
    assert len(extract_skill_api_quick_reference(mods_after[0]["python_code"])) == 1

    # End-to-end M8.6 clean-room static closure packaging of the M8.7 `autogenous_prim_*` import (including private `_quad_design`!)
    staged_by_stem = {Path(fn).stem: src for fn, src in mods_map.items()}
    ensure_clean_scipy_sandbox_image()
    cid = start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    try:
        exec_container_bash(cid, "mkdir -p /holdout_task /tmp/hegel_wormhole_skills", timeout_sec=10)
        cand_py = (
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_nuisance_projection import project_control_polynomial_surface\n\n"
            "def solve_plate(xy, y, ctrl):\n"
            "    det, _ = project_control_polynomial_surface(xy, y, ctrl)\n"
            "    return det\n"
        )
        with tempfile.TemporaryDirectory(prefix="m87_pkg_") as td:
            fp = Path(td) / "solver.py"
            fp.write_text(cand_py, encoding="utf-8")
            subprocess.run(["docker", "cp", str(fp), f"{cid}:/holdout_task/solver.py"], check=True, timeout=20)

        pkg_res = auto_package_wormhole_imports_in_container(cid, ("/holdout_task/solver.py",), staged_by_stem)
        assert pkg_res["ok"] is True
        assert "_quad_design" in pkg_res["packaged_files"][0]["included_definitions"]["autogenous_prim_nuisance_projection"]

        exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
        assert find_uninlined_wormhole_imports_in_container(cid, ("/holdout_task/solver.py",)) == []
        post_sha = compute_declared_artifacts_digest(cid, ("/holdout_task/solver.py",))
        chk = verify_candidate_artifact_contract_in_container(
            cid, ("/holdout_task/solver.py",), post_sha, required_entrypoints=("solve_plate",)
        )
        assert chk["accepted"] is True
    finally:
        stop_container_safely(cid)
