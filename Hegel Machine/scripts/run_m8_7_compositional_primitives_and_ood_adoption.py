#!/usr/bin/env python3
"""Hegel M8.7 Single-Invocation Pretraining-Only `L1` Primitive Sublation & OOD Holdout-12 Adoption Runner.

Scientific & Audit Protocol:
  1. Zero Holdout-12 / Frontier-20 Leakage (`require_pretraining_only=True`):
     Loads `pretrained_autogenous_forest_v2.json` (`pt01..pt06` ONLY) and runs live `gpt-6.2-sol`
     (`auth_mode == "chatgpt"`) `sublate_autogenous_forest_into_compositional_primitives` inside air-gapped
     `hegel_clean_scipy_sandbox:v1` (`network_mode="none"`) to distill pure `numpy`/`scipy` (`ZERO sklearn`)
     `L1_compositional_primitive` modules (`autogenous_prim_*.py`), serializing `pretrained_autogenous_forest_v3_primitives.json`.
  2. Clean-Room Static Packaging Proof (`M8.6 + M8.7`):
     Verifies that EVERY `L1` primitive module and `L2` operator in `v3` passes `auto_package_wormhole_imports_in_container`,
     physical `rm -rf /tmp/hegel_wormhole_skills` deletion, `verify_candidate_artifact_contract_in_container`, and
     isolated `PYTHONPATH="" python3 -I` execution.
  3. Downstream Blind OOD Holdout-12 Benign Numerical Slice (`ho06`, `ho01`, `ho08` in a single unified invocation):
     Measures the 5-layer skill adoption funnel (`recalled -> staged_l1 -> inspected -> dynamic_calls -> static_packaged -> strict_pass`)
     and compares directly against the sealed `M8.6` (`v2` coarse-only) baseline on `ho06` and `ho01`!
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic
from typing import Any

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
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
    sublate_autogenous_forest_into_compositional_primitives,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
    holdout_tasks_as_frontier_waves,
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

M87_SLICE_TASK_IDS: tuple[str, ...] = (
    "ho06__plate_evaporation_deconfounded_ic50",
    "ho01__crosswell_anisotropic_clock_tomography",
    "ho08__prony_viscoelastic_memory_inversion",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def capture_execution_identity() -> dict[str, Any]:
    """Compute exact Python source tree SHA-256 (`rel\\0bytes\\0`) matching M8.5/M8.6 audit verification formula."""
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


def verify_all_v3_modules_cleanroom_packaging(archive: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify M8.6 static closure packaging across ALL `L1` primitives & `L2` modules in `v3` inside `hegel_clean_scipy_sandbox:v1`."""
    t0 = monotonic()
    staged_all: dict[str, tuple[str, str]] = {}
    for nid, rec in sorted(archive.items()):
        if str(nid).startswith("__") or not isinstance(rec, dict):
            continue
        for mod in rec.get("autogenous_wormhole_modules") or []:
            stem = Path(str(mod["filename"])).stem
            tier = str(mod.get("granularity_tier") or ("L1_compositional_primitive" if stem.startswith("autogenous_prim_") else "L2_domain_operator"))
            staged_all.setdefault(stem, (tier, str(mod["python_code"])))

    code_by_stem = {k: v[1] for k, v in staged_all.items()}
    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    verified: list[dict[str, Any]] = []
    try:
        m6f.exec_container_bash(cid, "mkdir -p /holdout_task /tmp/hegel_wormhole_skills", timeout_sec=10)
        for idx, (stem, (tier, src_code)) in enumerate(sorted(staged_all.items())):
            tree = ast.parse(src_code)
            pub_fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
            if not pub_fns:
                continue
            imp_clause = ", ".join(pub_fns)
            probe_rel = f"/holdout_task/v3_probe_{idx:02d}.py"
            probe_mod = f"v3_probe_{idx:02d}"
            cand_src = (
                f"import inspect\n"
                f"import numpy as np\n"
                f"from hegel_wormhole_skills.{stem} import {imp_clause}\n\n"
                f"def check_all_exported() -> dict:\n"
                f"    fns = [{', '.join(pub_fns)}]\n"
                f"    return {{'count': len(fns), 'all_callable': all(callable(f) for f in fns), 'names': [f.__name__ for f in fns]}}\n"
            )
            with tempfile.TemporaryDirectory(prefix="m87_cleanroom_") as td:
                fp = Path(td) / f"{probe_mod}.py"
                fp.write_text(cand_src, encoding="utf-8")
                subprocess.run(["docker", "cp", str(fp), f"{cid}:{probe_rel}"], check=True, timeout=20)

            pkg_out = auto_package_wormhole_imports_in_container(cid, (probe_rel,), code_by_stem)
            assert pkg_out["ok"], f"Packaging failed for `{stem}`: {pkg_out}"
            m6f.exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
            uninlined = find_uninlined_wormhole_imports_in_container(cid, (probe_rel,))
            post_sha = m6f.compute_declared_artifacts_digest(cid, (probe_rel,))
            chk = verify_candidate_artifact_contract_in_container(
                cid, (probe_rel,), post_sha, required_entrypoints=("check_all_exported",)
            )
            ex = m6f.exec_container_bash(
                cid,
                f"cd / && PYTHONPATH='' python3 -I -c \""
                f"import json, sys; sys.path.insert(0, '/holdout_task'); import {probe_mod} as pm; "
                f"print(json.dumps(pm.check_all_exported()))\"",
                timeout_sec=20,
            )
            assert ex["exit_code"] == 0 and chk["accepted"] and uninlined == [], (stem, chk, ex)
            verified.append(
                {
                    "module_stem": stem,
                    "granularity_tier": tier,
                    "exported_functions": pub_fns,
                    "post_package_sha256": post_sha,
                    "contract_accepted": chk["accepted"],
                    "included_definitions": pkg_out["packaged_files"][0]["included_definitions"].get(stem, []),
                }
            )
        return {
            "verified": len(verified) > 0,
            "verified_module_count": len(verified),
            "l1_primitive_module_count": sum(1 for v in verified if v["granularity_tier"] == "L1_compositional_primitive"),
            "l2_operator_module_count": sum(1 for v in verified if v["granularity_tier"] != "L1_compositional_primitive"),
            "verified_modules": verified,
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }
    finally:
        m6f.stop_container_safely(cid)


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
        default=REPO_ROOT / "docs/evidence/m8_7_compositional_primitives_and_ood_adoption_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=18)
    args = ap.parse_args()

    inv_t0 = monotonic()
    inv_start_utc = _utc_now()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_id = capture_execution_identity()
    if not pre_id["code_worktree_clean"]:
        raise RuntimeError(f"Refusing live evidence run on dirty src/scripts/tests worktree:\n{pre_id['code_worktree_porcelain']}")

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", (
        f"Official ChatGPT subscription CLI required (`node007` forbidden): {codex_info}"
    )

    # 1. Calibrate decomposed NIG container planner strictly on Frontier-20 (`m7_4`, 0 Holdout-12 exposure)
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs_74 = load_historical_suite_episode_records(m74_dir)
    assert len(recs_74) == 20
    planner = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    (out_dir / "calibrated_m8_7_container_planner.json").write_text(
        json.dumps(planner.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 2. Load strictly unseeded `pt01..pt06` `v2` forest (`0` Holdout-12 exposure) & compile `L1` schema-free primitives!
    v2_bundle_fp = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, meta_dag, v2_meta = load_autogenous_forest_bundle(v2_bundle_fp)
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
        run_id="m8_7_pt_only_primitives",
        require_pretraining_only=True,
        allow_horizontal_wormholes=True,
    )
    prim_sublation["stage_wall_ms"] = round((monotonic() - prim_t0) * 1000.0, 2)
    (out_dir / "l1_primitive_sublation_receipt.json").write_text(
        json.dumps(prim_sublation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert prim_sublation["compiled_primitive_count"] >= 4, f"Too few L1 primitives compiled: {prim_sublation}"

    v3_bundle_fp = out_dir / "pretrained_autogenous_forest_v3_primitives.json"
    v3_bundle_meta = serialize_autogenous_forest_bundle(
        forest=forest,
        archive=archive,
        meta_dag=meta_dag,
        out_path=v3_bundle_fp,
        extra_metadata={
            "stage": "m8_7_pt_only_compositional_l1_primitives",
            "source_v2_bundle_sha256": v2_meta.get("bundle_sha256"),
            "compiled_l1_primitive_count": prim_sublation["compiled_primitive_count"],
            "zero_holdout_and_frontier_exposure_verified": True,
            "git_commit": pre_id["git_head_commit"],
        },
    )
    hippo_idx = HippoRAGWormholeIndex(forest)

    # 3. Clean-room packaging proof across ALL `L1` + `L2` modules in `v3`
    cleanroom_v3 = verify_all_v3_modules_cleanroom_packaging(archive)
    (out_dir / "v3_cleanroom_packaging_proof.json").write_text(
        json.dumps(cleanroom_v3, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 4. Run 3 benign OOD Holdout-12 numerical tasks (`ho06`, `ho01`, `ho08`) mounting `v3` (`max_workers=3`)
    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho_by_id = {t.task_id: t for tl in ho_waves.values() for t in tl}
    shared_graph = PersistentMechanismGraph()
    graph_ckpt_fp = out_dir / "m8_7_shared_mechanism_graph_checkpoint.json"
    runs_dir = out_dir / "runs"
    branches_dir = out_dir / "branches"
    run_id = "m8_7_ood_primitive_adoption_20261125"

    def _run_one_task(tid: str) -> tuple[str, dict[str, Any], PersistentMechanismGraph]:
        task_obj = all_ho_by_id[tid]
        worker_graph = PersistentMechanismGraph()
        worker_ckpt = out_dir / f"worker_graph_{task_obj.short_name}.json"
        t_start = _utc_now()
        t_m0 = monotonic()
        ep_dir = runs_dir / tid
        ep_sum = run_m6_9_a_two_branch_episode(
            task=task_obj,
            model=args.model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo_idx,
            episode_out_dir=ep_dir,
            branches_work_root=branches_dir,
            run_id=f"{run_id}__{task_obj.short_name}",
            max_turns_per_branch=args.max_turns,
            cmd_timeout_sec=150,
            reasoning_effort="medium",
            selective_second_branch=True,
            m8_container_planner=planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=worker_graph,
            m8_graph_checkpoint_path=worker_ckpt,
            episode_nominal_budget=2.5,
        )
        t_wall = round((monotonic() - t_m0) * 1000.0, 2)
        funnel = ep_sum.get("skill_adoption_funnel") or {}
        rec_row = {
            "task_id": tid,
            "short_name": task_obj.short_name,
            "started_at_utc": t_start,
            "finished_at_utc": _utc_now(),
            "stage_wall_ms": t_wall,
            "strict_pass": bool(ep_sum["strict_pass"]),
            "failure_taxonomy": ep_sum["failure_taxonomy"],
            "declared_budget_multiplier": ep_sum["declared_budget_multiplier"],
            "episode_nominal_units_total": ep_sum["episode_nominal_units_total"],
            "turns_used": ep_sum["turns_used"],
            "prompt_tokens": ep_sum["prompt_tokens"],
            "cached_tokens": ep_sum["cached_tokens"],
            "completion_tokens": ep_sum["completion_tokens"],
            "effective_billed_tokens": ep_sum["effective_billed_tokens"],
            "recalled_node_ids": ep_sum.get("recalled_node_ids", []),
            "staged_wormhole_modules": [m["module_import"] for m in ep_sum.get("staged_wormhole_modules", [])],
            "skill_packaging_events": ep_sum.get("skill_packaging_events", []),
            "skill_adoption_funnel": {
                **funnel,
                "clean_verifier_strict_pass": bool(ep_sum["strict_pass"]),
            },
            "selective_router_decision": ep_sum.get("selective_router_decision"),
            "local_container_repair_summary": ep_sum.get("local_container_repair_summary"),
            "candidate_identities": ep_sum.get("candidate_identities"),
            "postlock_branch_outcomes": ep_sum.get("postlock_branch_outcomes"),
        }
        print(
            f"[M8.7 DONE] `{tid}`: strict_pass={rec_row['strict_pass']} | "
            f"units={rec_row['episode_nominal_units_total']} | "
            f"funnel={rec_row['skill_adoption_funnel']}",
            flush=True,
        )
        return tid, rec_row, worker_graph

    by_tid: dict[str, tuple[dict[str, Any], PersistentMechanismGraph]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_run_one_task, tid): tid for tid in M87_SLICE_TASK_IDS}
        for fut in as_completed(futs):
            tid_done, row_done, wg_done = fut.result()
            by_tid[tid_done] = (row_done, wg_done)

    task_records: list[dict[str, Any]] = []
    worker_merge_receipts: dict[str, dict[str, int]] = {}
    for tid in M87_SLICE_TASK_IDS:
        row_d, wg_d = by_tid[tid]
        task_records.append(row_d)
        worker_merge_receipts[tid] = merge_worker_graph_into_shared(shared_graph, wg_d)

    shared_graph.save_checkpoint(graph_ckpt_fp)
    post_id = capture_execution_identity()
    assert pre_id["code_tree_sha256"] == post_id["code_tree_sha256"], "Code tree mutated during live run!"
    assert pre_id["src_and_tests_sha256"] == post_id["src_and_tests_sha256"], "src/tests mutated during live run!"

    # Load M8.6 (`v2` coarse-only) summary for direct head-to-head adoption comparison on `ho06` & `ho01`
    m86_sum_fp = REPO_ROOT / "docs/evidence/m8_6_static_packager_and_benign_adoption_20261125/m8_6_master_summary.json"
    m86_comparison: dict[str, Any] = {}
    if m86_sum_fp.is_file():
        m86_raw = json.loads(m86_sum_fp.read_text(encoding="utf-8"))
        m86_by_tid = {r["task_id"]: r for r in m86_raw.get("tasks", [])}
        for r in task_records:
            tid = r["task_id"]
            if tid in m86_by_tid:
                m86_f = m86_by_tid[tid].get("skill_adoption_funnel") or {}
                m87_f = r["skill_adoption_funnel"]
                m86_comparison[tid] = {
                    "m8_6_v2_coarse": {
                        "staged_modules": m86_f.get("staged_module_count", 0),
                        "inspected_in_bash": m86_f.get("inspected_in_bash", False),
                        "dynamic_skill_calls": m86_f.get("dynamic_skill_activation_count", 0),
                        "static_packaged_deliverables": m86_f.get("static_packaged_deliverable_count", 0),
                        "strict_pass": m86_by_tid[tid].get("strict_pass"),
                    },
                    "m8_7_v3_l1_primitives": {
                        "staged_modules": m87_f.get("staged_module_count", 0),
                        "staged_l1_primitives": m87_f.get("staged_l1_primitive_count", 0),
                        "inspected_in_bash": m87_f.get("inspected_in_bash", False),
                        "dynamic_skill_calls": m87_f.get("dynamic_skill_activation_count", 0),
                        "dynamic_l1_primitive_calls": m87_f.get("dynamic_l1_primitive_activation_count", 0),
                        "static_packaged_deliverables": m87_f.get("static_packaged_deliverable_count", 0),
                        "static_packaged_l1_primitives": m87_f.get("static_packaged_l1_primitive_count", 0),
                        "strict_pass": r["strict_pass"],
                    },
                }

    inv_wall_ms = round((monotonic() - inv_t0) * 1000.0, 2)
    master_summary = {
        "schema": "hegel_m8_7_compositional_primitives_and_ood_adoption_v1",
        "single_clean_invocation": True,
        "resume_used": False,
        "invocation_started_at_utc": inv_start_utc,
        "invocation_finished_at_utc": _utc_now(),
        "invocation_wall_ms": inv_wall_ms,
        "model": args.model,
        "auth_mode": codex_info.get("auth_mode"),
        "provider_backend": codex_info.get("provider_backend"),
        "max_turns_per_branch": args.max_turns,
        "source_v2_forest_bundle_sha256": v2_meta.get("bundle_sha256"),
        "v3_primitives_forest_bundle": v3_bundle_meta,
        "pre_run_execution_identity": pre_id,
        "post_run_execution_identity": post_id,
        "l1_primitive_compilation_summary": {
            "compiled_primitive_count": prim_sublation["compiled_primitive_count"],
            "total_source_l2_modules": prim_sublation["total_source_l2_modules"],
            "require_pretraining_only": prim_sublation["require_pretraining_only"],
            "llm_tokens": prim_sublation["llm_tokens"],
            "stage_wall_ms": prim_sublation["stage_wall_ms"],
        },
        "v3_cleanroom_packaging_proof": {
            "verified": cleanroom_v3["verified"],
            "verified_module_count": cleanroom_v3["verified_module_count"],
            "l1_primitive_module_count": cleanroom_v3["l1_primitive_module_count"],
            "l2_operator_module_count": cleanroom_v3["l2_operator_module_count"],
        },
        "ood_benign_slice_aggregate": {
            "task_count": len(task_records),
            "strict_passes": sum(1 for r in task_records if r["strict_pass"]),
            "total_nominal_units": round(sum(float(r["episode_nominal_units_total"]) for r in task_records), 4),
            "total_prompt_tokens": sum(int(r["prompt_tokens"]) for r in task_records),
            "total_cached_tokens": sum(int(r["cached_tokens"]) for r in task_records),
            "total_completion_tokens": sum(int(r["completion_tokens"]) for r in task_records),
            "total_effective_billed_tokens": sum(int(r["effective_billed_tokens"]) for r in task_records),
            "tasks_inspected_skills_in_bash": sum(1 for r in task_records if r["skill_adoption_funnel"].get("inspected_in_bash")),
            "tasks_with_any_skill_executed_or_packaged": sum(
                1 for r in task_records if r["skill_adoption_funnel"].get("any_skill_executed_or_packaged")
            ),
            "total_dynamic_skill_activations": sum(int(r["skill_adoption_funnel"].get("dynamic_skill_activation_count", 0)) for r in task_records),
            "total_dynamic_l1_primitive_activations": sum(
                int(r["skill_adoption_funnel"].get("dynamic_l1_primitive_activation_count", 0)) for r in task_records
            ),
            "total_static_packaged_deliverables": sum(
                int(r["skill_adoption_funnel"].get("static_packaged_deliverable_count", 0)) for r in task_records
            ),
            "total_static_packaged_l1_primitives": sum(
                int(r["skill_adoption_funnel"].get("static_packaged_l1_primitive_count", 0)) for r in task_records
            ),
        },
        "head_to_head_vs_m8_6_coarse": m86_comparison,
        "worker_graph_merge_receipts": worker_merge_receipts,
        "tasks": task_records,
    }
    (out_dir / "m8_7_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sha256_manifest(out_dir)
    print(f"[M8.7 COMPLETE] Wrote `{out_dir / 'm8_7_master_summary.json'}` in {inv_wall_ms:.1f} ms.", flush=True)


if __name__ == "__main__":
    main()
