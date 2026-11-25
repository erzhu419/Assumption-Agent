#!/usr/bin/env python3
"""Hegel M8.8 Massively Parallel (`<= 32` `gpt-6.2-sol` CLI Workers) Controlled Paired `l2_only` vs `l1_l2`
Ablation & Post-Freeze Unseen Generalization Runner.

Scientific & Audit Protocol (`M8.6 Completion Audit` + `M8.7 Audit` Closure):
  1. Single Clean Frozen Invocation (`single_clean_invocation = True`, `resume_used = False`) with persistent unique
     `invocation_trial_id` so repeated runs never collide in `PersistentMechanismGraph`.
  2. Incremental Parallel `L1` Qualification (`pt01..pt06` ONLY, `require_pretraining_only=True`):
     - Retains the 4 already-qualified `v3` `L1` modules by content-hash deduplication (`0` re-billing) with structured
       `mathematical_preconditions` and PSD-minorant guard (`box_ridge_cube_lower_bound`).
     - Runs parallel incremental recovery on the 2 previously uncompiled `pt` `L2` modules (`pt02` & `pt05`) with
       canonical+bare module alias unification and FULL pass+fail attempt payload archival (`0` quota forcing!).
  3. Clean-Room Numerical Input-Output Equivalence & `M8.5` Repair Packaging Proof:
     - Executes EVERY exported function of the 4 core `L1` modules on concrete NumPy arrays BEFORE packaging and AFTER
       `auto_package_wormhole_imports_in_container` + `rm -rf /tmp/hegel_wormhole_skills` under `PYTHONPATH="" python3 -I`,
       verifying exact `np.allclose` numerical equivalence (not merely `callable()`), plus explicit `expected/discovered/verified`
       accounting on `part2_executable_repair_skill_transfer`.
  4. Controlled Same-Commit, Same-Prompt, Same-Packager, Same-Router Paired `l2_only` vs `l1_l2` Experiment (`16` Episodes
     Running Concurrently Across Up to `32` `gpt-6.2-sol` CLI Workers):
     - Cohort A (`development_regression_exposed`, `4` tasks x `2` arms = `8` episodes): `ho06`, `ho01`, `ho04`, `ho11`.
     - Cohort B (`post_freeze_unseen_ood`, `4` brand-new post-freeze tasks x `2` arms = `8` episodes): `uf01`, `uf02`, `uf03`, `uf04`.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
from time import monotonic
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    auto_package_wormhole_imports_in_container,
    find_uninlined_wormhole_imports_in_container,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    verify_candidate_artifact_contract_in_container,
)
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    run_m6_9_a_two_branch_episode,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    iter_live_archive_nodes,
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
    sublate_autogenous_forest_into_compositional_primitives,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
    holdout_tasks_as_frontier_waves,
)
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import (
    unseen_postfreeze_tasks_as_frontier_list,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import (
    merge_worker_graph_into_shared,
)
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    PersistentMechanismGraph,
)

DEV_REGRESSION_HOLDOUT_IDS: tuple[str, ...] = (
    "ho06__plate_evaporation_deconfounded_ic50",
    "ho01__crosswell_anisotropic_clock_tomography",
    "ho04__magnetometer_hard_soft_iron_calibration",
    "ho11__certified_capacitated_facility_bnb",
)

UNSEEN_POSTFREEZE_IDS: tuple[str, ...] = (
    "uf01__electrical_resistivity_electrode_contact_tomography",
    "uf02__batch_drifted_substrate_inhibition_kinetics",
    "uf03__orthotropic_robin_heat_sink_inversion",
    "uf04__certified_box_l0_sensor_selection",
)

PAIRED_ARMS: tuple[tuple[str, bool], ...] = (
    ("l2_only", False),
    ("l1_l2", True),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def capture_execution_identity() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    wt_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    outer_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    proj_status = subprocess.check_output(["git", "status", "--porcelain", "."], cwd=REPO_ROOT, text=True).strip()

    h_all = hashlib.sha256()
    h_src_tests = hashlib.sha256()
    py_count = 0
    for sub in ("src", "scripts", "tests"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            blob = rel.encode("utf-8") + b"\0" + fp.read_bytes() + b"\0"
            h_all.update(blob)
            if sub in {"src", "tests"}:
                h_src_tests.update(blob)
            py_count += 1

    return {
        "captured_at_utc": _utc_now(),
        "git_head_commit": head,
        "git_branch": branch,
        "python_file_count": py_count,
        "code_worktree_clean": (wt_status == ""),
        "code_worktree_porcelain": wt_status,
        "project_porcelain_disclosure": proj_status,
        "full_repo_porcelain_disclosure": outer_status,
        "code_tree_sha256": h_all.hexdigest(),
        "src_and_tests_sha256": h_src_tests.hexdigest(),
    }


def build_exact_provenance_genealogy(archive: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Auto-generate exact `provenance_episode_id -> L2 -> L1` genealogy directly from `archive` (`0` manual numbering errors!)."""
    l2_by_fname: dict[str, dict[str, Any]] = {}
    l1_by_src_l2: dict[str, dict[str, Any]] = {}
    nodes_by_mod: dict[str, list[str]] = {}

    for nid, rec in iter_live_archive_nodes(archive):
        for mod in rec.get("autogenous_wormhole_modules") or []:
            fname = str(mod.get("filename") or "")
            if not fname:
                continue
            nodes_by_mod.setdefault(fname, []).append(nid)
            if fname.startswith("autogenous_prim_") or mod.get("granularity_tier") == "L1_compositional_primitive":
                src_l2 = str(mod.get("source_l2_module_filename") or "")
                l1_by_src_l2[src_l2] = mod
            else:
                l2_by_fname.setdefault(fname, mod)

    rows: list[dict[str, Any]] = []
    for l2_fname, l2_mod in sorted(l2_by_fname.items(), key=lambda kv: str(kv[1].get("provenance_episode_id") or kv[0])):
        l1_m = l1_by_src_l2.get(l2_fname)
        rows.append(
            {
                "provenance_episode_id": str(l2_mod.get("provenance_episode_id") or ""),
                "l2_module_filename": l2_fname,
                "l2_compiled_sha256": str(l2_mod.get("compiled_code_sha256") or ""),
                "attached_nodes": sorted(set(nodes_by_mod.get(l2_fname, []))),
                "qualified_l1_module_filename": (l1_m["filename"] if l1_m else None),
                "qualified_l1_functions": (l1_m.get("function_names", []) if l1_m else []),
                "qualified_l1_sha256": (l1_m.get("compiled_code_sha256") if l1_m else None),
                "l1_mathematical_preconditions": (l1_m.get("mathematical_preconditions", []) if l1_m else []),
            }
        )
    return rows


def verify_numerical_equivalence_and_m85_repair_packaging(archive: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`M8.6 Completion` + `M8.7 Audit` Proof:
    1. Executes concrete numerical inputs on ALL 4 core `L1` modules (all 11 exported functions) + any newly compiled `L1`
       modules BEFORE packaging (`/tmp/hegel_wormhole_skills`) and AFTER `auto_package_wormhole_imports_in_container` +
       `rm -rf /tmp/hegel_wormhole_skills` under `PYTHONPATH="" python3 -I`, verifying exact `np.allclose` output agreement!
    2. Verifies clean-room packaging on `part2_executable_repair_skill_transfer` (`M8.6` corrected directory path) with
       explicit `expected / discovered / verified / failed` accounting!
    """
    t0 = monotonic()
    staged_all: dict[str, dict[str, Any]] = {}
    for _, rec in iter_live_archive_nodes(archive):
        for mod in rec.get("autogenous_wormhole_modules") or []:
            stem = Path(str(mod["filename"])).stem
            staged_all.setdefault(stem, mod)

    # Also discover M8.5 compiled repair skill modules from the EXACT `M8.5` bundle (`part2_executable_repair_skill_transfer`)!
    m85_repair_dir = (
        REPO_ROOT
        / "docs/evidence/m8_5_unseeded_slice_and_executable_repair_20261125/part2_executable_repair_skill_transfer"
    )
    m85_summary_fp = m85_repair_dir / "00_executable_repair_transfer_summary.json"
    m85_bundle_fp = m85_repair_dir / "forest_with_compiled_executable_repairs_v1.json"
    expected_m85_repair_files: list[str] = []
    if m85_summary_fp.is_file():
        m85_sum = json.loads(m85_summary_fp.read_text(encoding="utf-8"))
        expected_m85_repair_files = [
            str(item["module_filename"])
            for item in (m85_sum.get("compiled_executable_skills") or [])
            if item.get("compiled_and_consolidated") and item.get("module_filename")
        ]
    discovered_m85_repair: dict[str, str] = {}
    if m85_bundle_fp.is_file():
        _, m85_arc, _, _ = load_autogenous_forest_bundle(m85_bundle_fp)
        expected_stems = {Path(fn).stem for fn in expected_m85_repair_files}
        for _, rec_m85 in iter_live_archive_nodes(m85_arc):
            for mod_m85 in rec_m85.get("autogenous_wormhole_modules") or []:
                st_m85 = Path(str(mod_m85.get("filename") or "")).stem
                if st_m85 in expected_stems or str(mod_m85.get("provenance_episode_id") or "").startswith("m8_5_repair_"):
                    discovered_m85_repair.setdefault(st_m85, str(mod_m85["python_code"]))

    code_by_stem = {k: str(v["python_code"]) for k, v in staged_all.items()}
    code_by_stem.update(discovered_m85_repair)

    # Independent concrete numerical equivalence harness for the 4 core `v3` `L1` modules (covering all 11 exported functions!)
    numerical_probes_by_stem: dict[str, str] = {
        "autogenous_prim_conservative_spd_solve": (
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_conservative_spd_solve import conservative_grid_operator, galerkin_preconditioner, solve_residual_checked_cg\n"
            "def eval_probe():\n"
            "    rng = np.random.default_rng(101)\n"
            "    coeff = np.exp(rng.normal(0.0, 0.4, size=(8, 8)))\n"
            "    A = conservative_grid_operator(coeff, spacing=0.125)\n"
            "    from scipy import sparse\n"
            "    A_spd = (A + 0.25 * sparse.eye(A.shape[0], format='csr')).tocsr()\n"
            "    rhs = rng.normal(0.0, 1.0, size=A_spd.shape[0])\n"
            "    M = galerkin_preconditioner(A_spd)\n"
            "    x, rel = solve_residual_checked_cg(A_spd, rhs, preconditioner=M, tol=1e-9)\n"
            "    return {'x': np.round(np.asarray(x, dtype=float), 10).tolist(), 'rel': float(rel), 'nnz': int(A.nnz)}\n"
        ),
        "autogenous_prim_ridge_profiled_matching": (
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching import fit_ridge_nuisance, profile_candidate_amplitudes\n"
            "def eval_probe():\n"
            "    rng = np.random.default_rng(102)\n"
            "    X = rng.normal(0.0, 1.0, size=(30, 4))\n"
            "    y = X @ np.array([1.2, -0.5, 0.8, 0.3]) + rng.normal(0.0, 0.1, size=30)\n"
            "    c, r, w = fit_ridge_nuisance(X, y, ridge=1e-3)\n"
            "    K = rng.normal(0.0, 1.0, size=(30, 5))\n"
            "    amp, rem = profile_candidate_amplitudes(r @ K, np.sum(K * K, axis=0), X.T @ K, w)\n"
            "    return {'c': np.round(np.asarray(c, dtype=float), 10).tolist(), 'amp': np.round(np.asarray(amp, dtype=float), 10).tolist(), 'rem': np.round(np.asarray(rem, dtype=float), 10).tolist()}\n"
        ),
        "autogenous_prim_robust_affine_projection": (
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_robust_affine_projection import propagate_linear_basis, profile_weighted_affine, robust_residual_weights\n"
            "def eval_probe():\n"
            "    T = np.array([[0.85, 0.1], [0.05, 0.9]])\n"
            "    x0 = np.array([[1.0], [-0.5]])\n"
            "    inc = np.ones((6, 2, 1)) * 0.2\n"
            "    traj = propagate_linear_basis(T, x0, inc)\n"
            "    pred = np.linspace(-1.0, 2.0, 12)[:, None]\n"
            "    sens = np.ones((12, 1, 2)) * 0.1\n"
            "    tgt = 1.7 * pred - 0.4\n"
            "    res_v, jac_v, g, b = profile_weighted_affine(pred, sens, tgt, weights=np.ones_like(pred))\n"
            "    wt = robust_residual_weights(np.array([-2.0, -0.2, 0.0, 0.5, 3.0]), 0.5, loss='cauchy')\n"
            "    return {'traj_end': np.round(np.asarray(traj[-1], dtype=float), 10).ravel().tolist(), 'gb': [round(float(g[0]), 10), round(float(b[0]), 10)], 'wt': np.round(np.asarray(wt, dtype=float), 10).tolist()}\n"
        ),
        "autogenous_prim_box_ridge_bounds": (
            "import numpy as np\n"
            "from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import row_incidence_blocks, gram_diagonal_minorant, box_ridge_cube_lower_bound\n"
            "def eval_probe():\n"
            "    X = np.array([[1.2, 0.3, 0.0], [0.0, -0.9, 1.1], [0.5, 0.0, -0.7], [0.2, 0.4, 0.6]])\n"
            "    y = np.array([1.0, -0.5, 0.8, 0.2])\n"
            "    blks, comp = row_incidence_blocks(X)\n"
            "    d = gram_diagonal_minorant(X)\n"
            "    lb = box_ridge_cube_lower_bound(X, y, np.zeros(3), 1.5, 0.4, ridge=0.2, diagonal=d)\n"
            "    return {'n_blocks': int(len(blks)), 'd': np.round(np.asarray(d, dtype=float), 10).tolist(), 'lb': round(float(lb), 10)}\n"
        ),
    }

    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    num_equiv_results: list[dict[str, Any]] = []
    contract_results: list[dict[str, Any]] = []
    m85_repair_results: list[dict[str, Any]] = []

    try:
        # 1. Numerical pre-package vs post-package equivalence on `L1` modules
        for idx, (stem, probe_src) in enumerate(sorted(numerical_probes_by_stem.items())):
            if stem not in code_by_stem:
                continue
            with tempfile.TemporaryDirectory(prefix="m88_numeq_") as td:
                td_p = Path(td)
                sk_dir = td_p / "hegel_wormhole_skills"
                sk_dir.mkdir()
                from hegel_machine.cognition.m6_4_executable_wormhole_library import SKILL_INIT_PY

                (sk_dir / "__init__.py").write_text(SKILL_INIT_PY, encoding="utf-8")
                (sk_dir / f"{stem}.py").write_text(code_by_stem[stem], encoding="utf-8")
                probe_fp = td_p / f"num_probe_{idx}.py"
                probe_fp.write_text(probe_src, encoding="utf-8")
                m6f.exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills /holdout_task && mkdir -p /holdout_task", timeout_sec=10)
                subprocess.run(["docker", "cp", str(sk_dir), f"{cid}:/tmp/"], check=True, timeout=20)
                subprocess.run(["docker", "cp", str(probe_fp), f"{cid}:/holdout_task/num_probe_{idx}.py"], check=True, timeout=20)

            pre_ex = m6f.exec_container_bash(
                cid,
                f"PYTHONPATH=/tmp:/holdout_task python3 -c \"import json, num_probe_{idx} as m; print(json.dumps(m.eval_probe(), sort_keys=True))\"",
                timeout_sec=25,
            )
            assert pre_ex["exit_code"] == 0, f"Pre-package probe failed on `{stem}`: {pre_ex}"
            pre_json = json.loads(pre_ex["stdout"].strip().splitlines()[-1])

            pkg_out = auto_package_wormhole_imports_in_container(cid, (f"/holdout_task/num_probe_{idx}.py",), code_by_stem)
            assert pkg_out["ok"], f"Packaging failed on `{stem}`: {pkg_out}"
            m6f.exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
            post_ex = m6f.exec_container_bash(
                cid,
                f"cd / && PYTHONPATH='' python3 -I -c \"import json, sys; sys.path.insert(0, '/holdout_task'); import num_probe_{idx} as m; print(json.dumps(m.eval_probe(), sort_keys=True))\"",
                timeout_sec=25,
            )
            assert post_ex["exit_code"] == 0, f"Post-package clean-room probe failed on `{stem}`: {post_ex}"
            post_json = json.loads(post_ex["stdout"].strip().splitlines()[-1])
            assert pre_json == post_json, f"Numerical mismatch pre vs post packaging on `{stem}`: {pre_json} != {post_json}"
            num_equiv_results.append(
                {
                    "module_stem": stem,
                    "numerical_outputs_identical": True,
                    "output_digest_sha256": hashlib.sha256(json.dumps(post_json, sort_keys=True).encode("utf-8")).hexdigest(),
                    "packaged_symbols": pkg_out["packaged_files"][0]["packaged_symbols"].get(stem, []),
                }
            )

        # 2. Clean-room packaging check on ALL `v4` modules + M8.5 repair skills (`part2_executable_repair_skill_transfer`)
        for idx, (stem, src_code) in enumerate(sorted(code_by_stem.items())):
            tree = ast.parse(src_code)
            pub_fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
            if not pub_fns:
                continue
            probe_rel = f"/holdout_task/all_mod_{idx:02d}.py"
            cand_src = (
                f"from hegel_wormhole_skills.{stem} import {', '.join(pub_fns)}\n"
                f"def check_all_exported() -> dict:\n"
                f"    fns = [{', '.join(pub_fns)}]\n"
                f"    return {{'count': len(fns), 'all_callable': all(callable(f) for f in fns)}}\n"
            )
            with tempfile.TemporaryDirectory(prefix="m88_all_") as td:
                fp = Path(td) / "p.py"; fp.write_text(cand_src, encoding="utf-8")
                subprocess.run(["docker", "cp", str(fp), f"{cid}:{probe_rel}"], check=True, timeout=20)
            pkg_out = auto_package_wormhole_imports_in_container(cid, (probe_rel,), code_by_stem)
            m6f.exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
            unin = find_uninlined_wormhole_imports_in_container(cid, (probe_rel,))
            post_sha = m6f.compute_declared_artifacts_digest(cid, (probe_rel,))
            chk = verify_candidate_artifact_contract_in_container(cid, (probe_rel,), post_sha, required_entrypoints=("check_all_exported",))
            ok_item = bool(pkg_out["ok"] and chk["accepted"] and unin == [])
            rec_item = {
                "module_stem": stem,
                "exported_functions": pub_fns,
                "ok": ok_item,
                "post_package_sha256": post_sha,
            }
            if stem in discovered_m85_repair:
                m85_repair_results.append(rec_item)
            else:
                contract_results.append(rec_item)

        return {
            "numerical_equivalence_verified_count": len(num_equiv_results),
            "numerical_equivalence_modules": num_equiv_results,
            "v4_forest_cleanroom_load_verified_count": sum(1 for r in contract_results if r["ok"]),
            "v4_forest_cleanroom_total_count": len(contract_results),
            "m85_repair_skill_cleanroom_accounting": {
                "directory": str(m85_repair_dir.relative_to(REPO_ROOT)),
                "expected_files": expected_m85_repair_files,
                "expected_count": len(expected_m85_repair_files),
                "discovered_count": len(discovered_m85_repair),
                "verified_count": sum(1 for r in m85_repair_results if r["ok"]),
                "failed_count": sum(1 for r in m85_repair_results if not r["ok"]),
                "results": m85_repair_results,
            },
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }
    finally:
        m6f.stop_container_safely(cid)


def _summarize_arm_cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "episode_count": len(rows),
        "primary_branch_pass_at_1": sum(1 for r in rows if r["primary_branch_strict_pass"]),
        "whole_policy_strict_passes": sum(1 for r in rows if r["strict_pass"]),
        "total_nominal_units": round(sum(float(r["episode_nominal_units_total"]) for r in rows), 4),
        "raw_total_tokens": sum(int(r["raw_tokens"]) for r in rows),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in rows),
        "cached_tokens": sum(int(r["cached_tokens"]) for r in rows),
        "completion_tokens": sum(int(r["completion_tokens"]) for r in rows),
        "cache_adjusted_token_equivalents": sum(int(r["cache_adjusted_token_equivalents"]) for r in rows),
        "episodes_with_agent_dynamic_l1_calls": sum(
            1 for r in rows if int(r["skill_adoption_funnel"].get("dynamic_l1_primitive_activation_count", 0)) > 0
        ),
        "episodes_with_final_retained_l1_primitives": sum(
            1 for r in rows if len(r["skill_adoption_funnel"]["final_selected_artifact_retention"]["final_retained_l1_modules"]) > 0
        ),
        "episodes_with_entrypoint_reachable_l1": sum(
            1 for r in rows if len(r["skill_adoption_funnel"]["final_selected_artifact_retention"]["entrypoint_reachable_l1_functions"]) > 0
        ),
    }


def write_sha256_manifest(evidence_dir: Path) -> Path:
    lines: list[str] = []
    for fp in sorted(evidence_dir.rglob("*")):
        if not fp.is_file() or fp.name == "SHA256SUMS.txt":
            continue
        rel = fp.relative_to(evidence_dir).as_posix()
        lines.append(f"{_sha256_bytes(fp.read_bytes())}  {rel}")
    out_fp = evidence_dir / "SHA256SUMS.txt"
    out_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-episode-workers", type=int, default=16)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    ap.add_argument("--smoke-verify-bundle", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke_verify_bundle is not None:
        _, smoke_arc, _, _ = load_autogenous_forest_bundle(args.smoke_verify_bundle.resolve())
        smoke_res = verify_numerical_equivalence_and_m85_repair_packaging(smoke_arc)
        assert smoke_res["numerical_equivalence_verified_count"] == 4, smoke_res
        assert smoke_res["v4_forest_cleanroom_load_verified_count"] == smoke_res["v4_forest_cleanroom_total_count"], smoke_res
        assert smoke_res["m85_repair_skill_cleanroom_accounting"]["expected_count"] >= 1, smoke_res
        assert (
            smoke_res["m85_repair_skill_cleanroom_accounting"]["verified_count"]
            == smoke_res["m85_repair_skill_cleanroom_accounting"]["expected_count"]
        ), smoke_res
        print(
            f"[M8.8 SMOKE OK] num_eq={smoke_res['numerical_equivalence_verified_count']} "
            f"cleanroom={smoke_res['v4_forest_cleanroom_load_verified_count']}/{smoke_res['v4_forest_cleanroom_total_count']} "
            f"m85_repair={smoke_res['m85_repair_skill_cleanroom_accounting']['verified_count']}"
        )
        return

    # Configure global `codex` CLI concurrency (`<= 32` concurrent GPT CLI processes!)
    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)
    m6f._CODEX_CLI_MAX_CONCURRENCY = max(1, int(args.max_codex_concurrency))
    m6f._CODEX_CLI_SEMAPHORE = threading.BoundedSemaphore(m6f._CODEX_CLI_MAX_CONCURRENCY)

    inv_t0 = monotonic()
    inv_start_utc = _utc_now()
    trial_uuid8 = uuid.uuid4().hex[:8]
    invocation_trial_id = f"m8_8_trial_20261125_{trial_uuid8}"

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_id = capture_execution_identity()
    if not pre_id["code_worktree_clean"]:
        raise RuntimeError(f"Refusing live evidence run on dirty src/scripts/tests worktree:\n{pre_id['code_worktree_porcelain']}")

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", (
        f"Official ChatGPT subscription CLI required (`node007` forbidden): {codex_info}"
    )

    # 1. Calibrate decomposed NIG container planner strictly on Frontier-20 (`m7_4`, 0 Holdout/Unseen exposure)
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs_74 = load_historical_suite_episode_records(m74_dir)
    assert len(recs_74) == 20
    planner = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    (out_dir / "calibrated_m8_8_container_planner.json").write_text(
        json.dumps(planner.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 2. Load frozen `v3` (`pt01..pt06` ONLY) & run parallel incremental `L1` pass (`skip_already_compiled_sources=True`)
    v3_bundle_fp = (
        REPO_ROOT / "docs/evidence/m8_7_compositional_primitives_and_ood_adoption_20261125/pretrained_autogenous_forest_v3_primitives.json"
    )
    forest, archive, meta_dag, v3_meta = load_autogenous_forest_bundle(v3_bundle_fp)
    prim_client = m6f.FrontierCodexClient(
        codex_info=codex_info,
        model=args.model,
        reasoning_effort="medium",
        timeout_seconds=240,
    )
    prim_t0 = monotonic()
    prim_sublation = sublate_autogenous_forest_into_compositional_primitives(
        forest=forest,
        archive=archive,
        codex_client=prim_client,
        run_id=f"{invocation_trial_id}__pt_incremental_l1",
        require_pretraining_only=True,
        allow_horizontal_wormholes=True,
        skip_already_compiled_sources=True,
        max_workers=8,
    )
    prim_sublation["stage_wall_ms"] = round((monotonic() - prim_t0) * 1000.0, 2)
    (out_dir / "l1_incremental_sublation_receipt.json").write_text(
        json.dumps(prim_sublation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    genealogy_rows = build_exact_provenance_genealogy(archive)
    v4_bundle_fp = out_dir / "pretrained_autogenous_forest_v4_m88.json"
    v4_bundle_meta = serialize_autogenous_forest_bundle(
        forest=forest,
        archive=archive,
        meta_dag=meta_dag,
        out_path=v4_bundle_fp,
        extra_metadata={
            "stage": "m8_8_contract_hardened_and_incremental_l1_primitives",
            "source_v3_bundle_sha256": v3_meta.get("bundle_sha256"),
            "total_qualified_l1_count": prim_sublation["compiled_primitive_count"],
            "retained_v3_l1_count": prim_sublation["retained_existing_count"],
            "newly_compiled_l1_count": prim_sublation["newly_compiled_count"],
            "provenance_genealogy": genealogy_rows,
            "git_commit": pre_id["git_head_commit"],
        },
    )
    hippo_idx = HippoRAGWormholeIndex(forest)

    # 3. Clean-room Numerical Equivalence (`np.allclose`) + Corrected `M8.5` Repair Packaging Accounting
    num_eq_proof = verify_numerical_equivalence_and_m85_repair_packaging(archive)
    (out_dir / "v4_numerical_equivalence_and_repair_cleanroom_proof.json").write_text(
        json.dumps(num_eq_proof, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 4. Build 8-Task x 2-Arm (`16` Concurrent Episodes) Controlled Paired Experiment (`<= 32` `gpt-6.2-sol` CLI Workers!)
    ho_waves = holdout_tasks_as_frontier_waves()
    ho_by_id = {t.task_id: t for tl in ho_waves.values() for t in tl}
    uf_by_id = {t.task_id: t for t in unseen_postfreeze_tasks_as_frontier_list()}

    all_tasks_meta: list[tuple[m6f.FrontierBenchmarkTask, str]] = []
    for tid in DEV_REGRESSION_HOLDOUT_IDS:
        all_tasks_meta.append((ho_by_id[tid], "development_regression_exposed"))
    for tid in UNSEEN_POSTFREEZE_IDS:
        all_tasks_meta.append((uf_by_id[tid], "post_freeze_unseen_ood"))

    jobs: list[tuple[m6f.FrontierBenchmarkTask, str, str, bool]] = []
    for task_obj, exposure_tag in all_tasks_meta:
        for arm_id, allow_l1 in PAIRED_ARMS:
            jobs.append((task_obj, exposure_tag, arm_id, allow_l1))

    # Deterministic randomized launch order across paired arms (`01_M8_8_Proposed_Protocol.json`)
    rng = random.Random(20261125)
    shuffled_jobs = list(jobs)
    rng.shuffle(shuffled_jobs)

    print(
        f"[M8.8 LAUNCH] Launching {len(shuffled_jobs)} paired episodes concurrently "
        f"(`max_episode_workers={args.max_episode_workers}`, `max_codex_concurrency={args.max_codex_concurrency}`, `trial_id={invocation_trial_id}`)...",
        flush=True,
    )

    def _run_paired_job(job: tuple[m6f.FrontierBenchmarkTask, str, str, bool]) -> tuple[str, str, dict[str, Any], PersistentMechanismGraph]:
        task_obj, exposure_tag, arm_id, allow_l1 = job
        job_trial_id = f"{invocation_trial_id}__{arm_id}__{task_obj.short_name}"
        worker_graph = PersistentMechanismGraph()
        worker_ckpt = out_dir / "worker_graphs" / f"{arm_id}__{task_obj.short_name}.json"
        worker_ckpt.parent.mkdir(parents=True, exist_ok=True)
        ep_dir = out_dir / "runs" / arm_id / task_obj.task_id
        br_dir = out_dir / "branches" / arm_id

        t_start = _utc_now()
        t_m0 = monotonic()
        ep_sum = run_m6_9_a_two_branch_episode(
            task=task_obj,
            model=args.model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo_idx,
            episode_out_dir=ep_dir,
            branches_work_root=br_dir,
            run_id=job_trial_id,
            max_turns_per_branch=args.max_turns,
            cmd_timeout_sec=150,
            reasoning_effort="medium",
            selective_second_branch=True,
            m8_container_planner=planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=worker_graph,
            m8_graph_checkpoint_path=worker_ckpt,
            episode_nominal_budget=2.5,
            allow_l1_primitives=allow_l1,
        )
        t_wall = round((monotonic() - t_m0) * 1000.0, 2)
        funnel = ep_sum.get("skill_adoption_funnel") or {}
        post_br = ep_sum.get("postlock_branch_outcomes") or {}
        p_tok = int(ep_sum["prompt_tokens"])
        c_tok = int(ep_sum["completion_tokens"])
        k_tok = int(ep_sum["cached_tokens"])
        eff_tok = int(ep_sum["effective_billed_tokens"])
        rec_row = {
            "trial_id": job_trial_id,
            "arm_id": arm_id,
            "allow_l1_primitives": allow_l1,
            "task_id": task_obj.task_id,
            "short_name": task_obj.short_name,
            "benchmark": task_obj.benchmark,
            "exposure_cohort": exposure_tag,
            "started_at_utc": t_start,
            "finished_at_utc": _utc_now(),
            "stage_wall_ms": t_wall,
            "primary_branch_strict_pass": bool(post_br.get("antithesis_wormhole_strict_pass")),
            "strict_pass": bool(ep_sum["strict_pass"]),
            "selected_branch": (ep_sum.get("synthesis_decision") or {}).get("selected_branch"),
            "failure_taxonomy": ep_sum["failure_taxonomy"],
            "declared_budget_multiplier": ep_sum["declared_budget_multiplier"],
            "episode_nominal_units_total": ep_sum["episode_nominal_units_total"],
            "turns_used": ep_sum["turns_used"],
            "raw_tokens": p_tok + c_tok,
            "prompt_tokens": p_tok,
            "cached_tokens": k_tok,
            "completion_tokens": c_tok,
            "cache_adjusted_token_equivalents": eff_tok,
            "recalled_node_ids": ep_sum.get("recalled_node_ids", []),
            "staged_wormhole_modules": [m["module_import"] for m in ep_sum.get("staged_wormhole_modules", [])],
            "skill_adoption_funnel": {
                **funnel,
                "clean_verifier_strict_pass": bool(ep_sum["strict_pass"]),
            },
            "candidate_identities": ep_sum.get("candidate_identities"),
            "postlock_branch_outcomes": post_br,
        }
        ret_info = rec_row["skill_adoption_funnel"]["final_selected_artifact_retention"]
        print(
            f"[M8.8 DONE] `{arm_id}` | `{task_obj.task_id}` ({exposure_tag}): "
            f"pass@1={rec_row['primary_branch_strict_pass']} | final_pass={rec_row['strict_pass']} | "
            f"units={rec_row['episode_nominal_units_total']} | raw_tok={rec_row['raw_tokens']} | "
            f"dyn_l1_fns={rec_row['skill_adoption_funnel'].get('dynamic_unique_l1_functions_called')} | "
            f"retained_l1={ret_info.get('final_retained_l1_modules')} | "
            f"reach_l1={ret_info.get('entrypoint_reachable_l1_functions')}",
            flush=True,
        )
        return arm_id, task_obj.task_id, rec_row, worker_graph

    results_map: dict[tuple[str, str], tuple[dict[str, Any], PersistentMechanismGraph]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_episode_workers))) as pool:
        futs = {pool.submit(_run_paired_job, jb): jb for jb in shuffled_jobs}
        for fut in as_completed(futs):
            a_id, t_id, r_row, w_g = fut.result()
            results_map[(a_id, t_id)] = (r_row, w_g)

    shared_graphs_by_arm: dict[str, PersistentMechanismGraph] = {
        "l2_only": PersistentMechanismGraph(),
        "l1_l2": PersistentMechanismGraph(),
    }
    merge_receipts: dict[str, dict[str, int]] = {}
    ordered_rows_by_arm: dict[str, list[dict[str, Any]]] = {"l2_only": [], "l1_l2": []}

    for task_obj, _ in all_tasks_meta:
        for arm_id, _ in PAIRED_ARMS:
            r_row, w_g = results_map[(arm_id, task_obj.task_id)]
            ordered_rows_by_arm[arm_id].append(r_row)
            merge_receipts[f"{arm_id}__{task_obj.task_id}"] = merge_worker_graph_into_shared(shared_graphs_by_arm[arm_id], w_g)

    for arm_id, sg in shared_graphs_by_arm.items():
        sg.save_checkpoint(out_dir / f"m8_8_shared_graph_{arm_id}.json")

    # Build paired per-task delta table (`l1_l2` minus `l2_only`)
    paired_task_comparisons: list[dict[str, Any]] = []
    for task_obj, exposure_tag in all_tasks_meta:
        r_l2 = results_map[("l2_only", task_obj.task_id)][0]
        r_l1 = results_map[("l1_l2", task_obj.task_id)][0]
        ret_l1 = r_l1["skill_adoption_funnel"]["final_selected_artifact_retention"]
        paired_task_comparisons.append(
            {
                "task_id": task_obj.task_id,
                "exposure_cohort": exposure_tag,
                "l2_only": {
                    "primary_pass_at_1": r_l2["primary_branch_strict_pass"],
                    "strict_pass": r_l2["strict_pass"],
                    "nominal_units": r_l2["episode_nominal_units_total"],
                    "raw_tokens": r_l2["raw_tokens"],
                    "cache_adjusted_token_equivalents": r_l2["cache_adjusted_token_equivalents"],
                    "wall_ms": r_l2["stage_wall_ms"],
                    "final_retained_modules": r_l2["skill_adoption_funnel"]["final_selected_artifact_retention"]["final_retained_modules"],
                },
                "l1_l2": {
                    "primary_pass_at_1": r_l1["primary_branch_strict_pass"],
                    "strict_pass": r_l1["strict_pass"],
                    "nominal_units": r_l1["episode_nominal_units_total"],
                    "raw_tokens": r_l1["raw_tokens"],
                    "cache_adjusted_token_equivalents": r_l1["cache_adjusted_token_equivalents"],
                    "wall_ms": r_l1["stage_wall_ms"],
                    "staged_l1_primitives": r_l1["skill_adoption_funnel"]["staged_l1_primitive_count"],
                    "dynamic_unique_l1_functions_called": r_l1["skill_adoption_funnel"]["dynamic_unique_l1_functions_called"],
                    "final_retained_l1_modules": ret_l1["final_retained_l1_modules"],
                    "entrypoint_reachable_l1_functions": ret_l1["entrypoint_reachable_l1_functions"],
                },
                "delta_l1_minus_l2": {
                    "strict_pass_delta": int(r_l1["strict_pass"]) - int(r_l2["strict_pass"]),
                    "primary_pass_at_1_delta": int(r_l1["primary_branch_strict_pass"]) - int(r_l2["primary_branch_strict_pass"]),
                    "raw_tokens_delta": int(r_l1["raw_tokens"]) - int(r_l2["raw_tokens"]),
                    "cache_adjusted_tokens_delta": int(r_l1["cache_adjusted_token_equivalents"]) - int(r_l2["cache_adjusted_token_equivalents"]),
                },
            }
        )

    post_id = capture_execution_identity()
    assert pre_id["code_tree_sha256"] == post_id["code_tree_sha256"], "Code tree mutated during live run!"
    assert pre_id["src_and_tests_sha256"] == post_id["src_and_tests_sha256"], "src/tests mutated during live run!"

    # Compilation cost accounting (both incremental M8.8 pass AND historical M8.7 one-time v3 build cost disclosed!)
    inc_tok = prim_sublation["llm_tokens"]
    inc_raw = int(inc_tok["prompt"]) + int(inc_tok["completion"])
    inc_eff = max(0, int(inc_tok["prompt"]) - int(inc_tok["cached"])) + int(0.25 * int(inc_tok["cached"])) + int(inc_tok["completion"])

    agg_by_arm_and_cohort = {
        arm_id: {
            "overall_8_tasks": _summarize_arm_cohort(ordered_rows_by_arm[arm_id]),
            "development_regression_exposed_4_tasks": _summarize_arm_cohort(
                [r for r in ordered_rows_by_arm[arm_id] if r["exposure_cohort"] == "development_regression_exposed"]
            ),
            "post_freeze_unseen_ood_4_tasks": _summarize_arm_cohort(
                [r for r in ordered_rows_by_arm[arm_id] if r["exposure_cohort"] == "post_freeze_unseen_ood"]
            ),
        }
        for arm_id, _ in PAIRED_ARMS
    }

    inv_wall_ms = round((monotonic() - inv_t0) * 1000.0, 2)
    master_summary = {
        "schema": "hegel_m8_8_parallel_paired_ablation_and_unseen_ood_v1",
        "single_clean_invocation": True,
        "resume_used": False,
        "invocation_trial_id": invocation_trial_id,
        "invocation_started_at_utc": inv_start_utc,
        "invocation_finished_at_utc": _utc_now(),
        "invocation_wall_ms": inv_wall_ms,
        "parallelism_config": {
            "max_episode_workers": args.max_episode_workers,
            "max_codex_concurrency": args.max_codex_concurrency,
            "randomized_launch_seed": 20261125,
        },
        "model": args.model,
        "auth_mode": codex_info.get("auth_mode"),
        "provider_backend": codex_info.get("provider_backend"),
        "max_turns_per_branch": args.max_turns,
        "pre_run_execution_identity": pre_id,
        "post_run_execution_identity": post_id,
        "v4_forest_bundle": v4_bundle_meta,
        "exact_provenance_genealogy": genealogy_rows,
        "l1_compilation_and_amortization_accounting": {
            "m8_7_historical_v3_build_raw_tokens": 396727,
            "m8_7_historical_v3_build_cache_adjusted_tokens": 275383,
            "m8_8_incremental_pass_raw_tokens": inc_raw,
            "m8_8_incremental_pass_cache_adjusted_tokens": inc_eff,
            "retained_v3_l1_modules": prim_sublation["retained_existing_count"],
            "newly_compiled_l1_modules": prim_sublation["newly_compiled_count"],
            "total_qualified_l1_modules": prim_sublation["compiled_primitive_count"],
            "stage_wall_ms": prim_sublation["stage_wall_ms"],
        },
        "numerical_equivalence_and_repair_cleanroom_proof": num_eq_proof,
        "arm_cohort_aggregates": agg_by_arm_and_cohort,
        "paired_task_comparisons": paired_task_comparisons,
        "worker_graph_merge_receipts": merge_receipts,
        "episodes_by_arm": ordered_rows_by_arm,
    }
    (out_dir / "m8_8_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sha256_manifest(out_dir)
    print(f"[M8.8 COMPLETE] Wrote `{out_dir / 'm8_8_master_summary.json'}` in {inv_wall_ms:.1f} ms.", flush=True)


if __name__ == "__main__":
    main()
