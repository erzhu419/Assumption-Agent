#!/usr/bin/env python3
"""Hegel M8.6 Single-Invocation Clean-Room Static Packager & Benign Numerical Adoption Runner.

Protocol & Integrity Guarantees (`Hegel_M8_5_Audit` Closure):
  1. Single unified process invocation (no `--resume` wall-time ambiguity): records `invocation_started_at_utc`,
     `invocation_finished_at_utc`, `invocation_wall_ms`, and per-stage wall durations.
  2. Strict pre-run / post-run code-tree identity (`pre["code_tree_sha256"] == post["code_tree_sha256"]`,
     `pre["src_and_tests_sha256"] == post["src_and_tests_sha256"]`), disclosing outer repository submodule status.
  3. Official ChatGPT CLI ONLY (`auth_mode == "chatgpt"`, `gpt-6.2-sol`; zero `node007` local endpoints).
  4. Evaluates 2 benign numerical `Holdout-12` tasks (`ho06__plate_evaporation_deconfounded_ic50` and
     `ho01__crosswell_anisotropic_clock_tomography`) using the unseeded autogenous forest bundle (`0` Holdout-12
     exposure) + `M8.5.1` decomposed NIG container planner (calibrated strictly on `m7_4` Frontier-20, `0` Holdout-12
     exposure) + `M8.6` static AST dependency closure packager (`enable_auto_skill_packaging=True`), recording the
     complete 5-layer skill adoption funnel and cryptographic `SHA256SUMS.txt`.
"""

from __future__ import annotations

import argparse
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    auto_package_wormhole_imports_in_container,
    find_uninlined_wormhole_imports_in_container,
    package_self_contained_wormhole_dependencies,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    verify_candidate_artifact_contract_in_container,
)
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    run_m6_9_a_two_branch_episode,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    load_autogenous_forest_bundle,
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


BENIGN_SLICE_TASK_IDS: tuple[str, ...] = (
    "ho06__plate_evaporation_deconfounded_ic50",
    "ho01__crosswell_anisotropic_clock_tomography",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _compute_tree_sha256(paths: list[Path]) -> str:
    entries: list[str] = []
    for root in paths:
        if root.is_file():
            rel = root.relative_to(REPO_ROOT).as_posix()
            entries.append(f"{rel}:{_sha256_bytes(root.read_bytes())}")
        elif root.is_dir():
            for fp in sorted(root.rglob("*")):
                if not fp.is_file():
                    continue
                if "__pycache__" in fp.parts or fp.suffix in {".pyc", ".pyo"}:
                    continue
                rel = fp.relative_to(REPO_ROOT).as_posix()
                entries.append(f"{rel}:{_sha256_bytes(fp.read_bytes())}")
    return _sha256_bytes("\n".join(sorted(entries)).encode("utf-8"))


def capture_execution_identity() -> dict[str, Any]:
    """Compute exact Python source tree SHA-256 (`rel\\0bytes\\0`) matching M8.5 audit verification formula."""
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


def run_mechanical_autogenous_packager_cleanroom_proof(archive: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Deterministically prove M8.6 static closure packaging across ALL real autogenous modules in `pretrained_autogenous_forest_v2.json`
    PLUS M8.5 compiled repair modules inside `hegel_clean_scipy_sandbox:v1` with `/tmp/hegel_wormhole_skills` physically deleted!
    """
    import ast as _ast

    t0 = monotonic()
    staged_real: dict[str, str] = {}
    for nid, rec in sorted(archive.items()):
        if str(nid).startswith("__") or not isinstance(rec, dict):
            continue
        for mod in rec.get("autogenous_wormhole_modules") or []:
            stem = Path(str(mod["filename"])).stem
            staged_real[stem] = str(mod["python_code"])

    m85_part2_dir = REPO_ROOT / "docs/evidence/m8_5_unseeded_slice_and_executable_repair_20261125/part2_executable_repair_transfer"
    for extra_bundle_fp in sorted(m85_part2_dir.glob("*forest*.json")) if m85_part2_dir.is_dir() else []:
        _, ex_arch, _, _ = load_autogenous_forest_bundle(extra_bundle_fp)
        for nid, rec in sorted(ex_arch.items()):
            if str(nid).startswith("__") or not isinstance(rec, dict):
                continue
            for mod in rec.get("autogenous_wormhole_modules") or []:
                stem = Path(str(mod["filename"])).stem
                staged_real.setdefault(stem, str(mod["python_code"]))

    assert staged_real, "Expected >= 1 autogenous module in pretrained forest archive"

    ensure_clean_scipy_sandbox_image()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    verified_modules: list[dict[str, Any]] = []
    try:
        m6f.exec_container_bash(cid, "mkdir -p /holdout_task /tmp/hegel_wormhole_skills", timeout_sec=10)
        for idx, (target_stem, real_mod_code) in enumerate(sorted(staged_real.items())):
            if "sklearn" in real_mod_code:
                # Skip sklearn-importing modules only if container seccomp restricts openblas/gomp, or test if available
                pass
            mod_tree = _ast.parse(real_mod_code)
            pub_fns = [
                n.name
                for n in mod_tree.body
                if isinstance(n, _ast.FunctionDef) and not n.name.startswith("_")
            ]
            if not pub_fns:
                continue
            fn0 = pub_fns[0]
            probe_rel = f"/holdout_task/packaged_probe_{idx:02d}.py"
            probe_mod_nm = f"packaged_probe_{idx:02d}"
            harness_cand = (
                f"import inspect\n"
                f"import numpy as np\n"
                f"from hegel_wormhole_skills.{target_stem} import {fn0} as _autogenous_op\n\n"
                f"def verify_packaged_callable() -> dict:\n"
                f"    sig = str(inspect.signature(_autogenous_op))\n"
                f"    return {{'callable': callable(_autogenous_op), 'fn': '{fn0}', 'signature': sig}}\n"
            )
            with tempfile.TemporaryDirectory(prefix="m86_real_proof_") as td:
                fp = Path(td) / f"{probe_mod_nm}.py"
                fp.write_text(harness_cand, encoding="utf-8")
                subprocess.run(["docker", "cp", str(fp), f"{cid}:{probe_rel}"], check=True, timeout=20)

            pre_sha = m6f.compute_declared_artifacts_digest(cid, (probe_rel,))
            pkg_out = auto_package_wormhole_imports_in_container(cid, (probe_rel,), staged_real)
            assert pkg_out["ok"], f"Auto-packager failed on `{target_stem}`: {pkg_out}"

            m6f.exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
            uninlined_after = find_uninlined_wormhole_imports_in_container(cid, (probe_rel,))
            post_sha = m6f.compute_declared_artifacts_digest(cid, (probe_rel,))
            chk = verify_candidate_artifact_contract_in_container(
                cid,
                (probe_rel,),
                post_sha,
                required_entrypoints=("verify_packaged_callable",),
            )
            ex = m6f.exec_container_bash(
                cid,
                f"cd / && PYTHONPATH='' python3 -I -c \""
                f"import json, sys; sys.path.insert(0, '/holdout_task'); import {probe_mod_nm} as pm; "
                f"print(json.dumps(pm.verify_packaged_callable()))\"",
                timeout_sec=20,
            )
            assert ex["exit_code"] == 0 and chk["accepted"] and uninlined_after == [], (target_stem, chk, ex)
            call_info = json.loads(ex["stdout"].strip().splitlines()[-1])
            verified_modules.append(
                {
                    "module_stem": target_stem,
                    "imported_function": fn0,
                    "pre_package_sha256": pre_sha,
                    "post_package_sha256": post_sha,
                    "contract_accepted": chk["accepted"],
                    "computed_candidate_sha256_matches_post_package": (chk["computed_candidate_sha256"] == post_sha),
                    "uninlined_imports_after_tmp_deleted": uninlined_after,
                    "cleanroom_execution_output": call_info,
                    "included_definitions": pkg_out["packaged_files"][0]["included_definitions"].get(target_stem, []),
                }
            )
        return {
            "verified": len(verified_modules) > 0,
            "verified_module_count": len(verified_modules),
            "verified_modules": verified_modules,
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
        default=REPO_ROOT / "docs/evidence/m8_6_static_packager_and_benign_adoption_20261125",
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

    # 1. Calibrate decomposed NIG container planner strictly on historical Frontier-20 (`m7_4`, 0 Holdout-12 exposure)
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs_74 = load_historical_suite_episode_records(m74_dir)
    assert len(recs_74) == 20, f"Expected 20 historical Frontier-20 records, got {len(recs_74)}"
    planner = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    (out_dir / "calibrated_m8_6_container_planner.json").write_text(
        json.dumps(planner.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 2. Load strictly unseeded OOD autogenous forest bundle (`m7_3` v2, `0` Holdout-12 exposure!)
    bundle_fp = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, meta_dag, bundle_meta = load_autogenous_forest_bundle(bundle_fp)
    _ = meta_dag
    hippo_idx = HippoRAGWormholeIndex(forest)

    # 3. Run deterministic clean-room packaging proof across all real autogenous & compiled-repair modules
    cleanroom_proof = run_mechanical_autogenous_packager_cleanroom_proof(archive)
    (out_dir / "mechanical_packager_cleanroom_proof.json").write_text(
        json.dumps(cleanroom_proof, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 4. Run 2 benign numerical Holdout-12 tasks in a single clean invocation (`max_workers=2`) with worker-graph merge
    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho_by_id = {t.task_id: t for tl in ho_waves.values() for t in tl}
    shared_graph = PersistentMechanismGraph()
    graph_ckpt_fp = out_dir / "m8_6_shared_mechanism_graph_checkpoint.json"
    runs_dir = out_dir / "runs"
    branches_dir = out_dir / "branches"
    run_id = "m8_6_benign_adoption_20261125"

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
            f"[M8.6 DONE] `{tid}`: strict_pass={rec_row['strict_pass']} | "
            f"units={rec_row['episode_nominal_units_total']} | "
            f"funnel={rec_row['skill_adoption_funnel']}",
            flush=True,
        )
        return tid, rec_row, worker_graph

    by_tid: dict[str, tuple[dict[str, Any], PersistentMechanismGraph]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one_task, tid): tid for tid in BENIGN_SLICE_TASK_IDS}
        for fut in as_completed(futs):
            tid_done, row_done, wg_done = fut.result()
            by_tid[tid_done] = (row_done, wg_done)

    task_records: list[dict[str, Any]] = []
    worker_merge_receipts: dict[str, dict[str, int]] = {}
    for tid in BENIGN_SLICE_TASK_IDS:
        row_d, wg_d = by_tid[tid]
        task_records.append(row_d)
        worker_merge_receipts[tid] = merge_worker_graph_into_shared(shared_graph, wg_d)

    shared_graph.save_checkpoint(graph_ckpt_fp)
    post_id = capture_execution_identity()
    assert pre_id["code_tree_sha256"] == post_id["code_tree_sha256"], "Code tree mutated during live run!"
    assert pre_id["src_and_tests_sha256"] == post_id["src_and_tests_sha256"], "src/tests mutated during live run!"

    inv_wall_ms = round((monotonic() - inv_t0) * 1000.0, 2)
    master_summary = {
        "schema": "hegel_m8_6_static_packager_and_benign_adoption_v1",
        "single_clean_invocation": True,
        "resume_used": False,
        "invocation_started_at_utc": inv_start_utc,
        "invocation_finished_at_utc": _utc_now(),
        "invocation_wall_ms": inv_wall_ms,
        "model": args.model,
        "auth_mode": codex_info.get("auth_mode"),
        "provider_backend": codex_info.get("provider_backend"),
        "max_turns_per_branch": args.max_turns,
        "pretrained_forest_bundle_sha256": bundle_meta.get("bundle_sha256"),
        "pre_run_execution_identity": pre_id,
        "post_run_execution_identity": post_id,
        "worker_graph_merge_receipts": worker_merge_receipts,
        "mechanical_packager_cleanroom_proof": {
            "verified": cleanroom_proof["verified"],
            "verified_module_count": cleanroom_proof["verified_module_count"],
            "verified_module_stems": [m["module_stem"] for m in cleanroom_proof["verified_modules"]],
        },
        "benign_numerical_slice_aggregate": {
            "task_count": len(task_records),
            "strict_passes": sum(1 for r in task_records if r["strict_pass"]),
            "total_nominal_units": round(sum(float(r["episode_nominal_units_total"]) for r in task_records), 4),
            "total_prompt_tokens": sum(int(r["prompt_tokens"]) for r in task_records),
            "total_cached_tokens": sum(int(r["cached_tokens"]) for r in task_records),
            "total_completion_tokens": sum(int(r["completion_tokens"]) for r in task_records),
            "total_effective_billed_tokens": sum(int(r["effective_billed_tokens"]) for r in task_records),
            "tasks_inspected_skills_in_bash": sum(1 for r in task_records if r["skill_adoption_funnel"].get("inspected_in_bash")),
            "total_dynamic_skill_activations": sum(int(r["skill_adoption_funnel"].get("dynamic_skill_activation_count", 0)) for r in task_records),
            "total_static_packaged_deliverables": sum(int(r["skill_adoption_funnel"].get("static_packaged_deliverable_count", 0)) for r in task_records),
        },
        "tasks": task_records,
    }
    (out_dir / "m8_6_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sha256_manifest(out_dir)
    print(f"[M8.6 COMPLETE] Wrote `{out_dir / 'm8_6_master_summary.json'}` in {inv_wall_ms:.1f} ms.", flush=True)


if __name__ == "__main__":
    main()
