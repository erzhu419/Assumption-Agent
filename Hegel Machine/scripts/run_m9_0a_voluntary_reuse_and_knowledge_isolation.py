#!/usr/bin/env python3
"""Hegel M9.0a Controlled Voluntary Executable Reuse & Knowledge-Isolation Runner (`3 Arms x 4 Tasks x 2 Replicates = 24 Episodes`).

Experimental Design (executed on a SINGLE frozen clean commit with `<= 32` concurrent `gpt-6.2-sol` Codex CLI workers):
- **Arm 1 (`bare_no_library`)**: Identical sovereign `Hegel_M6_9_2` loop + `M8` container repair + `M8OfflineContainerPlanner`
  with an empty `PhilosophicalForest` (`0` reference notes, `0` executable skills).
- **Arm 2 (`knowledge_only_reference`)**: Frozen `v5` (`6` qualified `L1` modules, `17` functions) mounted as read-only
  `progressive_knowledge_only` reference cards + `/tmp/hegel_knowledge_ref/describe_ref.py` + `.reference.txt` notes,
  with **ZERO** importable `hegel_wormhole_skills` Python package (`isolates Pure Knowledge Value = Arm 2 - Arm 1`).
- **Arm 3 (`progressive_executable_library`)**: Frozen `v5` mounted as importable `/tmp/hegel_wormhole_skills` Python package
  (`disclosure_mode="progressive"`, `staging_policy="net_value_gated"`, `6`-module `--describe` catalog + two-phase atomic
  static dependency-closure packager; `isolates Pure Executable Reuse Value = Arm 3 - Arm 2`).
- **4 Stratified Task Families (`Hegel-M90a-VoluntaryReuse-4`, zero skill naming hints)** x **2 Independent Replicates (`rep_01`, `rep_02`)**
  submitted in interleaved `(replicate, task, arm)` order (`24` episodes total).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
from time import monotonic
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    evaluate_planner_on_records,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_0_voluntary_reuse_benchmark import (
    get_voluntary_reuse_m90a_frontier_tasks,
    make_vr01_instance,
    verify_symmetric_box_l0_proof_certificate,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

ARMS_CONFIG: tuple[tuple[str, str, str], ...] = (
    ("bare_no_library", "none", "none"),
    ("knowledge_only_reference", "progressive_knowledge_only", "net_value_gated"),
    ("progressive_executable_library", "progressive", "net_value_gated"),
)

VR04_INCOMPATIBLE_REAL_OR_SPD_L1_MODULES: frozenset[str] = frozenset(
    {
        "autogenous_prim_conservative_spd_solve",
        "autogenous_prim_box_ridge_bounds",
        "autogenous_prim_reference_panel_balancing",
        "autogenous_prim_robust_trend_decomposition",
        "autogenous_prim_robust_affine_projection",
    }
)


def _install_codex_concurrency_limiter(max_codex_concurrency: int) -> dict[str, Any]:
    sem = threading.BoundedSemaphore(max(1, int(max_codex_concurrency)))
    lock = threading.Lock()
    stats = {"active": 0, "peak_concurrent_codex_calls": 0, "total_codex_calls": 0}
    orig_call = m6f.FrontierCodexClient.call_structured

    def _throttled_call(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with sem:
            with lock:
                stats["active"] += 1
                stats["total_codex_calls"] += 1
                if stats["active"] > stats["peak_concurrent_codex_calls"]:
                    stats["peak_concurrent_codex_calls"] = stats["active"]
            try:
                return orig_call(self, *args, **kwargs)
            finally:
                with lock:
                    stats["active"] -= 1

    m6f.FrontierCodexClient.call_structured = _throttled_call  # type: ignore[method-assign]
    return stats


def _git_and_source_tree_state(ev_rel_prefix: str) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    status_lines = [
        ln
        for ln in subprocess.check_output(["git", "status", "--porcelain", "."], cwd=str(REPO_ROOT), text=True).splitlines()
        if ln.strip() and ev_rel_prefix not in ln
    ]
    h = hashlib.sha256()
    for sub in ("src", "scripts", "tests"):
        root_s = REPO_ROOT / sub
        for p in sorted(root_s.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            h.update(f"{rel}\0".encode("utf-8"))
            h.update(p.read_bytes())
            h.update(b"\0")
    return {
        "git_head": head,
        "dirty_lines": status_lines,
        "source_tree_py_sha256": h.hexdigest(),
    }


def _compute_pre_run_hardness_and_applicability_audit(
    eval_tasks: list[m6f.FrontierBenchmarkTask],
    v5_archive: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # 1. VR01 root relaxation gap audit across private seeds (99201, 99202, 99203)
    vr01_audit: list[dict[str, Any]] = []
    for sd, bsz in ((99201, (10, 10)), (99202, (10, 11)), (99203, (11, 11))):
        inst = make_vr01_instance(sd, bsz)
        prob = inst["problem"]
        X, y = prob["X"], prob["y"]
        n, p = X.shape
        exact_opt = inst["gt"]["exact_objective"]
        exact_w = inst["gt"]["exact_weights"]
        G = X.T @ X
        off = np.sum(np.abs(G), axis=1) - np.abs(np.diag(G))
        d_gersh = np.maximum(np.diag(G) - 1.0001 * off, 0.0)
        res_c = lsq_linear(
            np.vstack([X, math.sqrt(prob["lam2"]) * np.eye(p)]),
            np.concatenate([y, np.zeros(p)]),
            bounds=(-prob["M"], prob["M"]),
        )
        c_root = res_c.x
        nu_root = y - X @ c_root
        min_chk = verify_symmetric_box_l0_proof_certificate(
            X=X, y=y, lam2=prob["lam2"], lam0=prob["lam0"], M_box=prob["M"],
            weights=exact_w, reported_objective=exact_opt, certified_lower_bound=exact_opt,
            subcube_leaves=[{
                "fixed_zero": [False] * p, "fixed_active": [False] * p,
                "lb": exact_opt, "minorant_diagonal": d_gersh.tolist(), "minorant_center": c_root.tolist(),
            }],
        )
        fen_chk = verify_symmetric_box_l0_proof_certificate(
            X=X, y=y, lam2=prob["lam2"], lam0=prob["lam0"], M_box=prob["M"],
            weights=exact_w, reported_objective=exact_opt, certified_lower_bound=exact_opt,
            subcube_leaves=[{
                "fixed_zero": [False] * p, "fixed_active": [False] * p,
                "lb": exact_opt, "dual_nu": nu_root.tolist(),
            }],
        )
        v_min = float(min_chk["leaves_preview"][0]["verified_lb"])
        v_fen = float(fen_chk["leaves_preview"][0]["verified_lb"])
        vr01_audit.append(
            {
                "seed": sd,
                "p": p,
                "support_space_size": int(1 << p),
                "exact_optimum": round(exact_opt, 6),
                "root_minorant_lb": round(v_min, 6),
                "root_minorant_gap": round(exact_opt - v_min, 6),
                "root_fenchel_lb": round(v_fen, 6),
                "root_fenchel_gap": round(exact_opt - v_fen, 6),
                "min_gershgorin_diagonal": round(float(np.min(d_gersh)), 6),
            }
        )

    # 2. VR04 Helmholtz indefiniteness spectrum check on 32x32 grid
    nx = ny = 32
    Lx, Ly = 1.05, 0.95
    hx, hy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    k0 = 11.8
    n_sq = 1.05 + 0.90 * np.exp(-((XX - 0.52 * Lx) ** 2 + (YY - 0.47 * Ly) ** 2) / (2.0 * (0.17 ** 2))) - 0.18 * np.sin(2.0 * np.pi * YY / Ly)
    N = nx * ny
    H_re = np.diag(-(hx * hy) * (k0 ** 2) * n_sq.ravel())
    ids = np.arange(N).reshape(nx, ny)
    for i0, i1 in zip(ids[:-1, :].ravel(), ids[1:, :].ravel()):
        w = hy / hx
        H_re[i0, i0] += w
        H_re[i1, i1] += w
        H_re[i0, i1] -= w
        H_re[i1, i0] -= w
    for j0, j1 in zip(ids[:, :-1].ravel(), ids[:, 1:].ravel()):
        w = hx / hy
        H_re[j0, j0] += w
        H_re[j1, j1] += w
        H_re[j0, j1] -= w
        H_re[j1, j0] -= w
    eigs_re = np.linalg.eigvalsh(H_re)

    # 3. Net-value / applicability matrix across all 4 tasks
    gate_matrix: dict[str, Any] = {}
    for t_obj in eval_tasks:
        _, _, diag_t = m64.collect_recalled_executable_wormhole_sources(
            [],
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=v5_archive,
            task_query_text=t_obj.instruction_text,
            disclosure_mode="progressive",
            staging_policy="net_value_gated",
            return_diagnostics=True,
        )
        gate_matrix[t_obj.task_id] = {
            "recommended_turn1_l1_count": diag_t["recommended_turn1_l1_count"],
            "mounted_on_disk_l1_count": diag_t["mounted_on_disk_l1_count"],
            "queryable_catalog_l1_count": diag_t["queryable_catalog_l1_count"],
            "card_char_stats": diag_t["card_char_stats"],
            "selected_l1_modules": diag_t["selected_l1_modules"],
            "excluded_l1_candidates": diag_t["excluded_l1_candidates"],
            "net_value_evaluations": diag_t["net_value_evaluations"],
        }

    return {
        "schema": "hegel_m9_0a_hardness_and_applicability_audit_v1",
        "zero_skill_naming_hints_in_prompts": all(
            ("hegel_wormhole_skills" not in t.instruction_text and "autogenous_prim_" not in t.instruction_text)
            for t in eval_tasks
        ),
        "vr01_private_root_gap_audit": vr01_audit,
        "vr02_scale_and_partial_fit": {
            "grid_shape": [72, 72],
            "unknowns_count": 5184,
            "rhs_candidate_modes": 10,
            "conductivity_contrast_ratio": round(80.0 / 0.02, 1),
            "partial_fit_design": "conservative_grid_operator mismatches Robin BCs; galerkin_preconditioner + solve_residual_checked_cg + fit_ridge_nuisance + profile_candidate_amplitudes match inner subproblems",
        },
        "vr03_prehistory_and_panel_composition": {
            "centers": 6,
            "strata": 3,
            "control_panel_columns": 5,
            "missing_control_rate": 0.12,
            "telemetry_cauchy_spike_rate": 0.111,
        },
        "vr04_helmholtz_indefinite_spectrum": {
            "grid_shape": [32, 32],
            "unknowns_count": 1024,
            "min_hermitian_part_eigenvalue": round(float(eigs_re[0]), 6),
            "max_hermitian_part_eigenvalue": round(float(eigs_re[-1]), 6),
            "negative_eigenvalue_count": int(np.sum(eigs_re < -1e-6)),
            "strictly_indefinite": bool(eigs_re[0] < -0.05 and eigs_re[-1] > 1.0),
        },
        "net_value_gate_matrix": gate_matrix,
    }


def _extract_primary_branch_metrics(ep_dir: Path, ep_sum: dict[str, Any]) -> dict[str, Any]:
    work_root = ep_dir / "_branches_work"
    anti_sums = sorted(work_root.glob("**/branch_2_antithesis_wormhole/00_episode_summary.json")) if work_root.is_dir() else []
    if anti_sums:
        a_s = json.loads(anti_sums[0].read_text(encoding="utf-8"))
        a_f = a_s.get("skill_adoption_funnel") or {}
        a_ret = a_f.get("final_selected_artifact_retention") or {}
        return {
            "primary_max_turns": int(a_s.get("max_turns", 0)),
            "primary_turns_used": int(a_s.get("turns_used", 0)),
            "primary_ended_by_finish": bool(a_s.get("ended_by_finish", False)),
            "primary_strict_pass_postlock": bool((ep_sum.get("postlock_branch_outcomes") or {}).get("antithesis_wormhole_strict_pass", False)),
            "primary_prompt_tokens": int(a_s.get("prompt_tokens", 0)),
            "primary_cached_tokens": int(a_s.get("cached_tokens", 0)),
            "primary_completion_tokens": int(a_s.get("completion_tokens", 0)),
            "primary_total_tokens": int(a_s.get("total_tokens", 0)),
            "primary_effective_billed_tokens": float(a_s.get("effective_billed_tokens", 0.0)),
            "primary_wall_ms": float(a_s.get("wall_ms", 0.0)),
            "turn1_staged_manifest_json_chars": len(json.dumps(a_s.get("staged_wormhole_modules", []), ensure_ascii=False)),
            "recommended_turn1_l1_count": int(a_f.get("recommended_turn1_l1_count", 0)),
            "mounted_on_disk_l1_count": int(a_f.get("mounted_on_disk_l1_count", 0)),
            "knowledge_reference_l1_count": int(a_f.get("knowledge_reference_l1_count", 0)),
            "queryable_catalog_l1_count": int(a_f.get("queryable_catalog_l1_count", 0)),
            "card_char_stats": dict(a_f.get("card_char_stats") or {}),
            "describe_skill_inspection_count": int(a_f.get("describe_skill_inspection_count", 0)),
            "describe_skill_queries": list(a_f.get("describe_skill_queries", [])),
            "inspected_knowledge_ref_in_bash": bool(a_f.get("inspected_knowledge_ref_in_bash", False)),
            "inspected_source_in_bash": bool(a_f.get("inspected_source_in_bash", False)),
            "primary_dynamic_l1_calls": int(a_f.get("dynamic_l1_primitive_activation_count", 0)),
            "primary_dynamic_unique_l1_modules": list(a_f.get("dynamic_unique_l1_modules_called", [])),
            "primary_dynamic_unique_l1_functions": list(a_f.get("dynamic_unique_l1_functions_called", [])),
            "primary_retained_l1_modules": list(a_ret.get("final_retained_l1_modules", [])),
            "primary_entrypoint_reachable_l1_functions": list(a_ret.get("entrypoint_reachable_l1_functions", [])),
        }
    f0 = ep_sum.get("skill_adoption_funnel") or {}
    r0 = f0.get("final_selected_artifact_retention") or {}
    return {
        "primary_max_turns": int(ep_sum.get("max_turns", 0)),
        "primary_turns_used": int(ep_sum.get("turns_used", 0)),
        "primary_ended_by_finish": bool(ep_sum.get("ended_by_finish", False)),
        "primary_strict_pass_postlock": bool(ep_sum.get("strict_pass", False)),
        "primary_prompt_tokens": int(ep_sum.get("prompt_tokens", 0)),
        "primary_cached_tokens": int(ep_sum.get("cached_tokens", 0)),
        "primary_completion_tokens": int(ep_sum.get("completion_tokens", 0)),
        "primary_total_tokens": int(ep_sum.get("total_tokens", 0)),
        "primary_effective_billed_tokens": float(ep_sum.get("effective_billed_tokens", 0.0)),
        "primary_wall_ms": float(ep_sum.get("wall_ms", 0.0)),
        "turn1_staged_manifest_json_chars": len(json.dumps(ep_sum.get("staged_wormhole_modules", []), ensure_ascii=False)),
        "recommended_turn1_l1_count": int(f0.get("recommended_turn1_l1_count", 0)),
        "mounted_on_disk_l1_count": int(f0.get("mounted_on_disk_l1_count", 0)),
        "knowledge_reference_l1_count": int(f0.get("knowledge_reference_l1_count", 0)),
        "queryable_catalog_l1_count": int(f0.get("queryable_catalog_l1_count", 0)),
        "card_char_stats": dict(f0.get("card_char_stats") or {}),
        "describe_skill_inspection_count": int(f0.get("describe_skill_inspection_count", 0)),
        "describe_skill_queries": list(f0.get("describe_skill_queries", [])),
        "inspected_knowledge_ref_in_bash": bool(f0.get("inspected_knowledge_ref_in_bash", False)),
        "inspected_source_in_bash": bool(f0.get("inspected_source_in_bash", False)),
        "primary_dynamic_l1_calls": int(f0.get("dynamic_l1_primitive_activation_count", 0)),
        "primary_dynamic_unique_l1_modules": list(f0.get("dynamic_unique_l1_modules_called", [])),
        "primary_dynamic_unique_l1_functions": list(f0.get("dynamic_unique_l1_functions_called", [])),
        "primary_retained_l1_modules": list(r0.get("final_retained_l1_modules", [])),
        "primary_entrypoint_reachable_l1_functions": list(r0.get("entrypoint_reachable_l1_functions", [])),
    }


def _write_sha256sums(evidence_dir: Path) -> int:
    sha_fp = evidence_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for p in sorted(evidence_dir.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS.txt":
            continue
        rel = p.relative_to(evidence_dir).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    sha_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel M9.0a 3-Arm x 4-Task x 2-Replicate Voluntary Reuse Experiment")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m9_0a_voluntary_reuse_and_knowledge_isolation_20261126",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-turns", type=int, default=28)
    parser.add_argument("--num-replicates", type=int, default=2)
    parser.add_argument("--max-episode-workers", type=int, default=24)
    parser.add_argument("--max-codex-concurrency", type=int, default=32)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    t_global0 = monotonic()
    ev_dir: Path = args.evidence_dir.resolve()
    ev_rel = ev_dir.relative_to(REPO_ROOT).as_posix() if ev_dir.is_relative_to(REPO_ROOT) else ev_dir.name
    start_state = _git_and_source_tree_state(ev_rel)
    if start_state["dirty_lines"] and not args.allow_dirty:
        raise RuntimeError(
            f"Refusing to start M9.0a on dirty working tree (commit first for clean provenance!): {start_state['dirty_lines']}"
        )

    ev_dir.mkdir(parents=True, exist_ok=True)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT CLI required: {codex_info}"
    ensure_clean_scipy_sandbox_image()
    codex_stats = _install_codex_concurrency_limiter(args.max_codex_concurrency)

    print(
        f"[M9.0a] Frozen Commit: {start_state['git_head']} | TreeSHA={start_state['source_tree_py_sha256'][:12]} | "
        f"Model={args.model} | MaxTurns={args.max_turns} | Replicates={args.num_replicates} | MaxCodex={args.max_codex_concurrency}",
        flush=True,
    )

    # 1. Load frozen `v5` bundle WITHOUT modifying a single byte (`0` online skill compilation!)
    v5_src_fp = (
        REPO_ROOT
        / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/pretrained_autogenous_forest_v5_m89.json"
    )
    v5_sha = hashlib.sha256(v5_src_fp.read_bytes()).hexdigest()
    v5_forest, v5_archive, _, _ = m7a.load_autogenous_forest_bundle(v5_src_fp)
    # Verify all 6 L1 modules pass strict fail-closed qualification check!
    for _, rec in m7a.iter_live_archive_nodes(v5_archive):
        for mod in rec.get("autogenous_wormhole_modules") or []:
            if str(mod.get("filename") or "").startswith("autogenous_prim_"):
                q_ok, q_rsn, _ = m64.verify_l1_module_qualification_record(mod)
                assert q_ok, f"Frozen v5 L1 module failed strict qualification: {mod.get('filename')} -> {q_rsn}"

    # 2. Fit `M8OfflineContainerPlanner` (`m8_9_prefix_clean_v2`) on sealed `m7_4` (`Frontier-20`)
    m74_suite_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m74_records = load_historical_suite_episode_records(m74_suite_dir)
    cal_planner = M8OfflineContainerPlanner.fit_from_historical_records(m74_records, suite_ids=[m74_suite_dir.name])
    cal_resub = evaluate_planner_on_records(cal_planner, m74_records)
    (ev_dir / "m9_0a_router_fit_receipt.json").write_text(
        json.dumps(
            {
                "evaluation_kind_note": "in_sample_resubstitution_on_m7_4_see_m8_9_1_loto_for_cross_validated",
                "resubstitution_summary": cal_resub,
                "planner": cal_planner.to_json_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 3. Assemble `Hegel-M90a-VoluntaryReuse-4` tasks and seal pre-run hardness & applicability audit
    eval_tasks = get_voluntary_reuse_m90a_frontier_tasks()
    assert len(eval_tasks) == 4
    hardness_audit = _compute_pre_run_hardness_and_applicability_audit(eval_tasks, v5_archive)
    (ev_dir / "m9_0a_pre_run_hardness_and_applicability_audit.json").write_text(
        json.dumps(hardness_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 4. Build interleaved job list: `(rep_id, task_obj, arm_id, disc_mode, stg_pol)`
    replicates = [f"rep_{r_idx:02d}" for r_idx in range(1, int(args.num_replicates) + 1)]
    jobs: list[tuple[str, m6f.FrontierBenchmarkTask, str, str, str]] = []
    for rep_id in replicates:
        for t_obj in eval_tasks:
            for arm_id, disc_mode, stg_pol in ARMS_CONFIG:
                jobs.append((rep_id, t_obj, arm_id, disc_mode, stg_pol))

    episodes_root = ev_dir / "episodes"
    resumed_lock = threading.Lock()
    run_counters = {"resumed_episode_count": 0, "fresh_executed_episode_count": 0}

    print(f"[M9.0a] Launching {len(jobs)} interleaved episodes across {args.max_episode_workers} workers...", flush=True)

    def _run_single_job(
        job: tuple[str, m6f.FrontierBenchmarkTask, str, str, str],
    ) -> tuple[str, str, str, dict[str, Any]]:
        rep_id, task_obj, arm_id, disc_mode, stg_pol = job
        ep_dir = episodes_root / rep_id / arm_id / task_obj.task_id
        sum_fp = ep_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            with resumed_lock:
                run_counters["resumed_episode_count"] += 1
            return rep_id, arm_id, task_obj.task_id, json.loads(sum_fp.read_text(encoding="utf-8"))

        if arm_id == "bare_no_library":
            ep_forest = PhilosophicalForest()
            ep_archive: dict[str, dict[str, Any]] = {}
            allow_l1 = False
            eff_disc = "dedup_full"
            eff_stg = "all_compatible"
        else:
            ep_forest = deepcopy(v5_forest)
            ep_archive = deepcopy(v5_archive)
            ep_archive["__autogenous_blank_seed__"] = {"enabled": True}
            allow_l1 = True
            eff_disc = disc_mode
            eff_stg = stg_pol

        ep_hippo = HippoRAGWormholeIndex(ep_forest)
        ep_graph = PersistentMechanismGraph(forest=ep_forest)
        ep_planner = M8OfflineContainerPlanner.from_json_dict(cal_planner.to_json_dict())

        res = run_m6_9_a_two_branch_episode(
            task_obj,
            model=args.model,
            codex_info=codex_info,
            forest=ep_forest,
            forest_archive=ep_archive,
            hippo_idx=ep_hippo,
            episode_out_dir=ep_dir,
            branches_work_root=ep_dir / "_branches_work",
            run_id=f"m9_0a_{rep_id}_{arm_id}",
            max_turns_per_branch=args.max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=args.reasoning_effort,
            selective_second_branch=True,
            m8_container_planner=ep_planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=ep_graph,
            m8_graph_checkpoint_path=ep_dir / "m8_graph_checkpoint.json",
            episode_nominal_budget=2.5,
            allow_l1_primitives=allow_l1,
            disclosure_mode=eff_disc,
            staging_policy=eff_stg,
        )
        with resumed_lock:
            run_counters["fresh_executed_episode_count"] += 1
        return rep_id, arm_id, task_obj.task_id, res

    results_grid: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        rep_id: {a[0]: {} for a in ARMS_CONFIG} for rep_id in replicates
    }
    job_errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_episode_workers))) as pool:
        fut_to_job = {pool.submit(_run_single_job, jb): jb for jb in jobs}
        done_cnt = 0
        for fut in as_completed(fut_to_job):
            done_cnt += 1
            rep_id, t_obj, arm_id, _, _ = fut_to_job[fut]
            try:
                r_rep, r_arm, r_tid, r_sum = fut.result()
                results_grid[r_rep][r_arm][r_tid] = r_sum
                prim = _extract_primary_branch_metrics(episodes_root / r_rep / r_arm / r_tid, r_sum)
                print(
                    f"[{done_cnt:02d}/{len(jobs):02d}] {r_rep} | {r_arm:30s} | {r_tid[:28]:28s} -> "
                    f"prim_1x={prim['primary_strict_pass_postlock']} | policy={r_sum['strict_pass']} | "
                    f"prim_turns={prim['primary_turns_used']:2d} | dyn_l1={prim['primary_dynamic_l1_calls']} | "
                    f"ret_l1={len(prim['primary_retained_l1_modules'])} | prim_tok={prim['primary_total_tokens']:,}",
                    flush=True,
                )
            except Exception as exc:
                err_rec = {"replicate": rep_id, "arm": arm_id, "task_id": t_obj.task_id, "error": f"{type(exc).__name__}:{exc}"}
                job_errors.append(err_rec)
                print(f"[ERROR] {err_rec}", flush=True)

    if job_errors:
        (ev_dir / "m9_0a_job_errors.json").write_text(json.dumps(job_errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"M9.0a encountered {len(job_errors)} episode error(s): {job_errors}")

    end_state = _git_and_source_tree_state(ev_rel)

    # 5. Extract flat episode rows + per-task/replicate comparisons + arm aggregates
    episode_rows: list[dict[str, Any]] = []
    for rep_id in replicates:
        for t_obj in eval_tasks:
            tid = t_obj.task_id
            is_neg_ctrl = tid.startswith("vr04__")
            for arm_id, disc_mode, stg_pol in ARMS_CONFIG:
                s = results_grid[rep_id][arm_id][tid]
                ep_dir = episodes_root / rep_id / arm_id / tid
                prim = _extract_primary_branch_metrics(ep_dir, s)
                funnel = s.get("skill_adoption_funnel") or {}
                ret = funnel.get("final_selected_artifact_retention") or {}
                dyn_l1_mods = list(funnel.get("dynamic_unique_l1_modules_called", []))
                ret_l1_mods = list(ret.get("final_retained_l1_modules", []))
                invalid_vr04_mods = (
                    sorted(set(dyn_l1_mods) & VR04_INCOMPATIBLE_REAL_OR_SPD_L1_MODULES) if is_neg_ctrl else []
                )
                episode_rows.append(
                    {
                        "replicate": rep_id,
                        "arm": arm_id,
                        "task_id": tid,
                        "is_negative_control_task": is_neg_ctrl,
                        "disclosure_mode": disc_mode,
                        "staging_policy": stg_pol,
                        "primary_max_turns": prim["primary_max_turns"],
                        "primary_1x_strict_pass": prim["primary_strict_pass_postlock"],
                        "policy_strict_pass": bool(s["strict_pass"]),
                        "primary_1x_turns_used": prim["primary_turns_used"],
                        "policy_turns_used": int(s["turns_used"]),
                        "policy_nominal_units": float(s.get("episode_nominal_units_total", 1.0)),
                        "second_branch_spawned": bool((s.get("postlock_branch_outcomes") or {}).get("thesis_spawned", False)),
                        "local_repair_attempted": bool((s.get("local_container_repair_summary") or {}).get("attempted", False)),
                        "local_repair_verified": bool((s.get("local_container_repair_summary") or {}).get("repaired_verified", False)),
                        "selected_branch": (s.get("synthesis_decision") or {}).get("selected_branch"),
                        "turn1_staged_manifest_json_chars": prim["turn1_staged_manifest_json_chars"],
                        "recommended_turn1_l1_count": prim["recommended_turn1_l1_count"],
                        "mounted_on_disk_l1_count": prim["mounted_on_disk_l1_count"],
                        "knowledge_reference_l1_count": prim["knowledge_reference_l1_count"],
                        "queryable_catalog_l1_count": prim["queryable_catalog_l1_count"],
                        "card_char_stats": prim["card_char_stats"],
                        "describe_skill_inspection_count": prim["describe_skill_inspection_count"],
                        "describe_skill_queries": prim["describe_skill_queries"],
                        "inspected_knowledge_ref_in_bash": prim["inspected_knowledge_ref_in_bash"],
                        "inspected_source_in_bash": prim["inspected_source_in_bash"],
                        "dynamic_l1_calls": int(funnel.get("dynamic_l1_primitive_activation_count", 0)),
                        "dynamic_unique_l1_modules": dyn_l1_mods,
                        "dynamic_unique_l1_functions": list(funnel.get("dynamic_unique_l1_functions_called", [])),
                        "final_retained_l1_modules": ret_l1_mods,
                        "entrypoint_reachable_l1_functions": list(ret.get("entrypoint_reachable_l1_functions", [])),
                        "all_retained_ast_match_source": bool(ret.get("all_retained_ast_match_source", True)),
                        "invalid_precondition_l1_modules_called": invalid_vr04_mods,
                        "primary_1x_prompt_tokens": prim["primary_prompt_tokens"],
                        "primary_1x_cached_tokens": prim["primary_cached_tokens"],
                        "primary_1x_completion_tokens": prim["primary_completion_tokens"],
                        "primary_1x_total_tokens": prim["primary_total_tokens"],
                        "primary_1x_effective_billed_tokens": round(prim["primary_effective_billed_tokens"], 2),
                        "primary_1x_wall_sec": round(prim["primary_wall_ms"] / 1000.0, 2),
                        "policy_total_tokens": int(s["total_tokens"]),
                        "policy_effective_billed_tokens": round(float(s["effective_billed_tokens"]), 2),
                        "policy_wall_sec": round(float(s["wall_ms"]) / 1000.0, 2),
                        "verifier_metrics": (s.get("verifier_report") or {}).get("verifier_metrics"),
                    }
                )

    def _agg_rows(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n_s = max(len(sub), 1)
        rep_prim_rates = [
            sum(1 for x in sub if x["replicate"] == r_id and x["primary_1x_strict_pass"])
            / max(1, sum(1 for x in sub if x["replicate"] == r_id))
            for r_id in replicates
            if any(x["replicate"] == r_id for x in sub)
        ]
        rep_pol_rates = [
            sum(1 for x in sub if x["replicate"] == r_id and x["policy_strict_pass"])
            / max(1, sum(1 for x in sub if x["replicate"] == r_id))
            for r_id in replicates
            if any(x["replicate"] == r_id for x in sub)
        ]
        return {
            "episode_count": len(sub),
            "primary_1x_strict_pass_count": sum(1 for x in sub if x["primary_1x_strict_pass"]),
            "primary_1x_strict_pass_rate": round(sum(1 for x in sub if x["primary_1x_strict_pass"]) / n_s, 4),
            "primary_1x_replicate_rates": [round(v, 4) for v in rep_prim_rates],
            "primary_1x_replicate_sd": round(statistics.stdev(rep_prim_rates), 4) if len(rep_prim_rates) >= 2 else 0.0,
            "policy_strict_pass_count": sum(1 for x in sub if x["policy_strict_pass"]),
            "policy_strict_pass_rate": round(sum(1 for x in sub if x["policy_strict_pass"]) / n_s, 4),
            "policy_replicate_rates": [round(v, 4) for v in rep_pol_rates],
            "policy_replicate_sd": round(statistics.stdev(rep_pol_rates), 4) if len(rep_pol_rates) >= 2 else 0.0,
            "mean_primary_1x_turns": round(sum(x["primary_1x_turns_used"] for x in sub) / n_s, 2),
            "mean_policy_turns": round(sum(x["policy_turns_used"] for x in sub) / n_s, 2),
            "mean_policy_nominal_units": round(sum(x["policy_nominal_units"] for x in sub) / n_s, 3),
            "second_branch_spawn_count": sum(1 for x in sub if x["second_branch_spawned"]),
            "local_repair_attempt_count": sum(1 for x in sub if x["local_repair_attempted"]),
            "local_repair_verified_count": sum(1 for x in sub if x["local_repair_verified"]),
            "mean_turn1_manifest_json_chars": round(sum(x["turn1_staged_manifest_json_chars"] for x in sub) / n_s, 1),
            "mean_recommended_turn1_l1_count": round(sum(x["recommended_turn1_l1_count"] for x in sub) / n_s, 2),
            "mean_mounted_on_disk_l1_count": round(sum(x["mounted_on_disk_l1_count"] for x in sub) / n_s, 2),
            "mean_knowledge_reference_l1_count": round(sum(x["knowledge_reference_l1_count"] for x in sub) / n_s, 2),
            "mean_queryable_catalog_l1_count": round(sum(x["queryable_catalog_l1_count"] for x in sub) / n_s, 2),
            "episodes_with_describe_cli": sum(1 for x in sub if x["describe_skill_inspection_count"] > 0),
            "total_describe_cli_inspections": sum(x["describe_skill_inspection_count"] for x in sub),
            "episodes_with_knowledge_ref_read": sum(1 for x in sub if x["inspected_knowledge_ref_in_bash"]),
            "episodes_with_dynamic_l1_calls": sum(1 for x in sub if x["dynamic_l1_calls"] > 0),
            "total_dynamic_l1_calls": sum(x["dynamic_l1_calls"] for x in sub),
            "episodes_with_retained_l1_deliverable": sum(1 for x in sub if len(x["final_retained_l1_modules"]) > 0),
            "episodes_with_entrypoint_reachable_l1": sum(1 for x in sub if len(x["entrypoint_reachable_l1_functions"]) > 0),
            "episodes_with_invalid_precondition_l1_calls": sum(
                1 for x in sub if len(x["invalid_precondition_l1_modules_called"]) > 0
            ),
            "primary_1x_total_tokens_sum": sum(x["primary_1x_total_tokens"] for x in sub),
            "primary_1x_effective_billed_tokens_sum": round(sum(x["primary_1x_effective_billed_tokens"] for x in sub), 2),
            "mean_primary_1x_wall_sec": round(sum(x["primary_1x_wall_sec"] for x in sub) / n_s, 2),
            "policy_total_tokens_sum": sum(x["policy_total_tokens"] for x in sub),
            "policy_effective_billed_tokens_sum": round(sum(x["policy_effective_billed_tokens"] for x in sub), 2),
            "mean_policy_wall_sec": round(sum(x["policy_wall_sec"] for x in sub) / n_s, 2),
        }

    arm_aggregates: dict[str, Any] = {}
    for arm_id, disc_mode, stg_pol in ARMS_CONFIG:
        all_a = [x for x in episode_rows if x["arm"] == arm_id]
        pos_a = [x for x in all_a if not x["is_negative_control_task"]]
        neg_a = [x for x in all_a if x["is_negative_control_task"]]
        by_task_a = {t.task_id: _agg_rows([x for x in all_a if x["task_id"] == t.task_id]) for t in eval_tasks}
        arm_aggregates[arm_id] = {
            "disclosure_mode": disc_mode,
            "staging_policy": stg_pol,
            "all_8_episodes": _agg_rows(all_a),
            "positive_fit_6_episodes_vr01_vr03": _agg_rows(pos_a),
            "negative_control_2_episodes_vr04": _agg_rows(neg_a),
            "by_task_family": by_task_a,
        }

    def _contrast(hi_arm: str, lo_arm: str, label_holds: str) -> dict[str, Any]:
        hi = arm_aggregates[hi_arm]["all_8_episodes"]
        lo = arm_aggregates[lo_arm]["all_8_episodes"]
        hi_pos = arm_aggregates[hi_arm]["positive_fit_6_episodes_vr01_vr03"]
        lo_pos = arm_aggregates[lo_arm]["positive_fit_6_episodes_vr01_vr03"]
        return {
            "holds_constant": label_holds,
            "all_8_primary_1x_pass_delta": hi["primary_1x_strict_pass_count"] - lo["primary_1x_strict_pass_count"],
            "all_8_policy_pass_delta": hi["policy_strict_pass_count"] - lo["policy_strict_pass_count"],
            "positive_6_primary_1x_pass_delta": hi_pos["primary_1x_strict_pass_count"] - lo_pos["primary_1x_strict_pass_count"],
            "positive_6_policy_pass_delta": hi_pos["policy_strict_pass_count"] - lo_pos["policy_strict_pass_count"],
            "mean_primary_1x_turns_delta": round(hi["mean_primary_1x_turns"] - lo["mean_primary_1x_turns"], 2),
            "positive_6_mean_primary_1x_turns_delta": round(hi_pos["mean_primary_1x_turns"] - lo_pos["mean_primary_1x_turns"], 2),
            "primary_1x_total_tokens_pct_change": round(
                100.0 * (hi["primary_1x_total_tokens_sum"] - lo["primary_1x_total_tokens_sum"]) / max(1, lo["primary_1x_total_tokens_sum"]),
                2,
            ),
            "policy_total_tokens_pct_change": round(
                100.0 * (hi["policy_total_tokens_sum"] - lo["policy_total_tokens_sum"]) / max(1, lo["policy_total_tokens_sum"]),
                2,
            ),
            "mean_primary_1x_wall_sec_delta": round(hi["mean_primary_1x_wall_sec"] - lo["mean_primary_1x_wall_sec"], 2),
        }

    contrasts = {
        "contrast_1_pure_knowledge_effect_arm2_minus_arm1": _contrast(
            "knowledge_only_reference",
            "bare_no_library",
            "zero executable skill imports in both arms; isolates mathematical reference cards + contracts + readable .reference.txt notes",
        ),
        "contrast_2_pure_executable_reuse_effect_arm3_minus_arm2": _contrast(
            "progressive_executable_library",
            "knowledge_only_reference",
            "identical v5 mathematical contracts & preconditions; isolates importable executable Python skills + atomic static closure packaging",
        ),
        "contrast_3_end_to_end_library_effect_arm3_minus_arm1": _contrast(
            "progressive_executable_library",
            "bare_no_library",
            "identical sovereign Hegel M6.9.2 + M8 repair + selective router; measures total progressive executable library net value over bare",
        ),
    }

    master_summary = {
        "experiment_id": "hegel_m9_0a_voluntary_reuse_and_knowledge_isolation_20261126",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "code_commit_at_start": start_state["git_head"],
            "code_commit_at_end": end_state["git_head"],
            "same_frozen_commit_throughout": start_state["git_head"] == end_state["git_head"],
            "source_tree_py_sha256_at_start": start_state["source_tree_py_sha256"],
            "source_tree_py_sha256_at_end": end_state["source_tree_py_sha256"],
            "zero_source_mutation_during_run": start_state["source_tree_py_sha256"] == end_state["source_tree_py_sha256"],
            "git_dirty_at_start": len(start_state["dirty_lines"]) > 0,
            "git_dirty_files_at_start": start_state["dirty_lines"],
            "git_dirty_at_end": len(end_state["dirty_lines"]) > 0,
            "resumed_episode_count": run_counters["resumed_episode_count"],
            "fresh_executed_episode_count": run_counters["fresh_executed_episode_count"],
            "single_clean_unresumed_invocation": (
                len(start_state["dirty_lines"]) == 0
                and len(end_state["dirty_lines"]) == 0
                and start_state["source_tree_py_sha256"] == end_state["source_tree_py_sha256"]
                and run_counters["resumed_episode_count"] == 0
            ),
            "frozen_v5_bundle_path": str(v5_src_fp.relative_to(REPO_ROOT).as_posix()),
            "frozen_v5_bundle_sha256": v5_sha,
            "online_skill_compilation_enabled": False,
        },
        "protocol": {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "auth_mode": codex_info["auth_mode"],
            "provider_backend": codex_info["provider_backend"],
            "primary_max_turns": args.max_turns,
            "observed_primary_max_turns": sorted({x["primary_max_turns"] for x in episode_rows}),
            "num_replicates": args.num_replicates,
            "interleaved_submission_order": True,
            "zero_skill_naming_hints": hardness_audit["zero_skill_naming_hints_in_prompts"],
        },
        "concurrency_telemetry": {
            "configured_max_episode_workers": args.max_episode_workers,
            "configured_max_codex_concurrency": args.max_codex_concurrency,
            "peak_concurrent_codex_calls": codex_stats["peak_concurrent_codex_calls"],
            "total_codex_calls": codex_stats["total_codex_calls"],
            "total_wall_clock_sec": round(monotonic() - t_global0, 2),
        },
        "arm_aggregates": arm_aggregates,
        "factor_isolated_contrasts": contrasts,
        "episode_rows": episode_rows,
    }
    (ev_dir / "m9_0a_voluntary_reuse_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_hashed = _write_sha256sums(ev_dir)
    print(f"[M9.0a] COMPLETE in {round(monotonic() - t_global0, 2)}s | Hashed {n_hashed} files into SHA256SUMS.txt", flush=True)


if __name__ == "__main__":
    main()
