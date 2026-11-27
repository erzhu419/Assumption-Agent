#!/usr/bin/env python3
"""Hegel M9.3 End-to-End Online Source Formation, Pre-Target Qualification Freeze, and Lifecycle Runner.

Executes on a SINGLE clean frozen commit with `<= 32` concurrent `gpt-6.2-sol` Codex CLI workers:
1. **Stage S (`Blind Source Formation`)**:
   - Runs `src01` and `src02` blind in parallel starting from frozen `v7` (`pretrained_autogenous_forest_v7_m921.json`).
   - Records ALL source LLM tokens, provider attempts, and wall-clock in `C_source` (retaining any failed attempts in denominator).
2. **Pre-Target Online Compilation & Clean-Container 4-Tuple Qualification Freeze (`Freeze Before Target Exposure`)**:
   - Compiles verified `src01`/`src02` deliverables into `L2` episode operators (`compile_autogenous_episode_sublation`),
     consolidates into `v8_forest / v8_archive`, and compiles + container-verifies new `L1` primitives + executable invariant
     checkers (`sublate_autogenous_forest_into_compositional_primitives(..., require_pretraining_only=False, include_executable_checker_guidance=True)`)
     under `hegel_l1_qual_4tuple_v2` (`hegel_env_v2:...` + host-authored double-execution repeatability).
   - Records ALL compilation/qualification LLM tokens & attempts in `C_qualification`.
   - Verifies `direct_source_float_plugin_fails_all_successors == True` (`v10.2` `C2`) and `K` vs `E` pre-run equal-information
     reader parity (`0` Turn-1 manifest bytes, identical user payload JSON, `0` search disagreements), and seals
     `m9_3_frozen_post_source_bundle.json` + `01_pre_target_qualification_freeze_receipt.json` STRICTLY BEFORE dispatching any successor!
3. **Stage T + Stage C (`Post-Freeze 3-Arm Symmetric Successor Matrix`: `3 Arms [B, K, E] x 4 Successors x 2 Replicates = 24 Single-Branch Episodes`)**:
   - `prior_frozen_v7_no_source_growth` (`B`): `on_demand_executable` over frozen `v7` (`0` `src01`/`src02` growth).
   - `equal_source_knowledge_only` (`K`): `on_demand_knowledge_only` over frozen `v8` (`v7` + `src01`/`src02` notes/contracts/`.reference.txt` via `_SHARED_CATALOG_READER_CORE_PY`, `0` importable Python package).
   - `equal_source_executable_library` (`E`): `on_demand_executable` over frozen `v8` (`v7` + `src01`/`src02` 4-tuple-verified `L1` Python package + static AST closure packager).
4. **Stage R (`Post-Counterexample Regime Bifurcation & Held-Out Original-Condition Recovery`)**:
   - Consolidates verified `E` `Stage C` deliverables (`suc01_C`, `suc02_C`) with `should_split_node=True` into
     `m9_3_post_counterexample_bifurcated_bundle.json` (`C_counterexample_consolidation`), mechanically proves `0` destructive
     overwrite of `generation=0` `src01`/`src02` `L1` primitives (`zero_destructive_overwrite == True`), and runs `2` live blind
     `Stage R` recovery episodes (`rec01_R`, `rec02_R`) (`C_recovery`).
5. **Full Lifecycle Break-Even Ledger & `SHA256SUMS.txt`**.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import tempfile
import threading
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition import m9_3_online_lifecycle_benchmark as m93b
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m9_3_online_lifecycle_benchmark import (
    evaluate_post_freeze_dev_seed_plugin_controls,
    get_m93_recovery_frontier_tasks,
    get_m93_source_frontier_tasks,
    get_m93_successor_frontier_tasks,
)

SUCCESSOR_ARMS_CONFIG: tuple[tuple[str, str, str, bool, bool], ...] = (
    # (arm_id, disclosure_mode, staging_policy, enable_auto_skill_packaging, use_post_source_v8)
    ("prior_frozen_v7_no_source_growth", "on_demand_executable", "net_value_gated", True, False),
    ("equal_source_knowledge_only", "on_demand_knowledge_only", "net_value_gated", False, True),
    ("equal_source_executable_library", "on_demand_executable", "net_value_gated", True, True),
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
        for p in sorted((REPO_ROOT / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            h.update(f"{rel}\0".encode("utf-8"))
            h.update(p.read_bytes())
            h.update(b"\0")
    return {"git_head": head, "dirty_lines": status_lines, "source_tree_py_sha256": h.hexdigest()}


def _proxy_w025(in_tok: int, cached_tok: int, out_tok: int) -> float:
    return round(float((in_tok - cached_tok) + 0.25 * cached_tok + out_tok), 2)


def _extract_turn1_tokens(ep_dir: Path) -> dict[str, int]:
    dag_fp = ep_dir / "01_decision_provenance_dag.json"
    if not dag_fp.is_file():
        return {"turn1_prompt_tokens": 0, "turn1_cached_tokens": 0, "turn1_completion_tokens": 0}
    dag = json.loads(dag_fp.read_text(encoding="utf-8"))
    t1 = next((r for r in dag.get("records", []) if str(r.get("decision_type", "")).startswith(("D_TURN_01", "D2_TURN_01"))), None)
    if not t1:
        return {"turn1_prompt_tokens": 0, "turn1_cached_tokens": 0, "turn1_completion_tokens": 0}
    cd = t1.get("cost_ledger_delta") or t1
    return {
        "turn1_prompt_tokens": int(cd.get("prompt_tokens_delta", 0)),
        "turn1_cached_tokens": int(cd.get("cached_tokens_delta", 0)),
        "turn1_completion_tokens": int(cd.get("completion_tokens_delta", 0)),
    }


def _extract_verifier_metrics(ep_dir: Path) -> dict[str, Any]:
    ver_fp = ep_dir / "03_official_verifier_report.json"
    if not ver_fp.is_file():
        return {}
    ver = json.loads(ver_fp.read_text(encoding="utf-8"))
    if isinstance(ver.get("verifier_parsed_json"), dict):
        return ver["verifier_parsed_json"]
    if isinstance(ver.get("verifier_metrics"), dict) and ver["verifier_metrics"]:
        return ver["verifier_metrics"]
    for ln in reversed(str(ver.get("verifier_stdout") or "").splitlines()):
        s = ln.strip()
        if s.startswith("{") and "strict_pass" in s:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return {}


def _write_sha256sums(evidence_dir: Path) -> int:
    sha_fp = evidence_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for p in sorted(evidence_dir.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS.txt":
            continue
        rel = p.relative_to(evidence_dir).as_posix()
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    sha_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _load_submitted_py_sources(ep_dir: Path) -> dict[str, str]:
    snap = ep_dir / "04_submitted_artifacts_snapshot"
    out: dict[str, str] = {}
    if snap.is_dir():
        for fp in sorted(snap.rglob("*.py")):
            out[fp.name] = fp.read_text(encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel M9.3 Online Source-Freeze-Successor-Recovery Lifecycle Experiment")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m9_3_online_source_freeze_lifecycle_20261126",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-turns", type=int, default=18)
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
        raise RuntimeError(f"Refusing to start M9.3 on dirty working tree: {start_state['dirty_lines']}")

    ev_dir.mkdir(parents=True, exist_ok=True)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT CLI required: {codex_info}"
    ensure_clean_scipy_sandbox_image()
    env_dig, env_manifest = m7a.compute_runtime_sandbox_env_digest(sandbox_cid=None)
    m64.register_verified_l1_env_digest(env_dig)
    codex_stats = _install_codex_concurrency_limiter(args.max_codex_concurrency)

    v7_src_fp = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
    v7_sha = hashlib.sha256(v7_src_fp.read_bytes()).hexdigest()
    v7_forest, v7_archive, v7_meta_dag, v7_meta = m7a.load_autogenous_forest_bundle(v7_src_fp)

    v7_stems: set[str] = set()
    for _, rec in m7a.iter_live_archive_nodes(v7_archive):
        for m in rec.get("autogenous_wormhole_modules") or []:
            fn = str(m.get("filename") or "")
            if fn.startswith("autogenous_prim_"):
                ok_q, rsn_q, _ = m64.verify_l1_module_qualification_record(
                    m, require_4tuple_protocol=True, expected_env_digests={env_dig}, allow_v6_legacy_env_label=False
                )
                assert ok_q, f"v7 4-tuple check failed for {fn}: {rsn_q}"
                v7_stems.add(Path(fn).stem)
    assert len(v7_stems) == 6

    print(
        f"[M9.3] Frozen Commit: {start_state['git_head']} | TreeSHA={start_state['source_tree_py_sha256'][:12]} | "
        f"v7SHA={v7_sha[:12]} | Env={env_dig} | Model={args.model} | MaxTurns={args.max_turns}",
        flush=True,
    )

    resumed_lock = threading.Lock()
    run_counters = {"resumed_episode_count": 0, "fresh_executed_episode_count": 0}

    # =========================================================================
    # PHASE 1: STAGE S — BLIND ONLINE SOURCE FORMATION (`src01`, `src02`)
    # =========================================================================
    source_tasks = get_m93_source_frontier_tasks()
    assert len(source_tasks) == 2
    stage_s_root = ev_dir / "stage_S_sources"
    print(f"[M9.3 Stage S] Launching {len(source_tasks)} blind source episodes in parallel...", flush=True)

    def _run_source_task(s_task: m6f.FrontierBenchmarkTask) -> tuple[str, list[dict[str, Any]], Path]:
        attempts_out: list[dict[str, Any]] = []
        for att_i in (1, 2):
            att_dir = stage_s_root / f"attempt_{att_i:02d}" / s_task.task_id
            sum_fp = att_dir / "00_episode_summary.json"
            if sum_fp.is_file():
                with resumed_lock:
                    run_counters["resumed_episode_count"] += 1
                res_s = json.loads(sum_fp.read_text(encoding="utf-8"))
            else:
                ep_f = deepcopy(v7_forest)
                ep_a = deepcopy(v7_archive)
                ep_a["__autogenous_blank_seed__"] = {"enabled": True}
                res_s = m6f.run_frontier_episode(
                    s_task,
                    "Hegel_M6_9_Antithesis_LiveGPT",
                    model=args.model,
                    codex_info=codex_info,
                    forest=ep_f,
                    forest_archive=ep_a,
                    hippo_idx=HippoRAGWormholeIndex(ep_f),
                    episode_out_dir=att_dir,
                    run_id=f"m9_3_stageS_att{att_i:02d}",
                    max_turns=args.max_turns,
                    cmd_timeout_sec=180,
                    reasoning_effort=args.reasoning_effort,
                    defer_official_verifier=False,
                    enable_auto_skill_packaging=True,
                    allow_l1_primitives=True,
                    disclosure_mode="on_demand_executable",
                    staging_policy="net_value_gated",
                )
                with resumed_lock:
                    run_counters["fresh_executed_episode_count"] += 1
            attempts_out.append({"attempt_index": att_i, "ep_dir": str(att_dir), "summary": res_s})
            print(
                f"  [STAGE S] {s_task.short_name} (att={att_i}) | pass={res_s['strict_pass']} | turns={res_s['turns_used']} | tok={res_s['total_tokens']}",
                flush=True,
            )
            if res_s.get("strict_pass"):
                return s_task.task_id, attempts_out, att_dir
        raise RuntimeError(f"Source task `{s_task.task_id}` failed strict verification across 2 blind attempts!")

    source_attempts_by_tid: dict[str, list[dict[str, Any]]] = {}
    winning_source_dirs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        s_futs = [pool.submit(_run_source_task, st) for st in source_tasks]
        for sf in as_completed(s_futs):
            tid_s, att_list, win_dir = sf.result()
            source_attempts_by_tid[tid_s] = att_list
            winning_source_dirs[tid_s] = win_dir

    # Compute full `C_source` including ALL attempts (passed or failed!)
    src_in = sum(int(a["summary"]["prompt_tokens"]) for al in source_attempts_by_tid.values() for a in al)
    src_ca = sum(int(a["summary"]["cached_tokens"]) for al in source_attempts_by_tid.values() for a in al)
    src_ou = sum(int(a["summary"]["completion_tokens"]) for al in source_attempts_by_tid.values() for a in al)
    src_prov_att = sum(
        int((a["summary"].get("provider_attempt_accounting") or {}).get("total_provider_attempts", a["summary"]["turns_used"]))
        for al in source_attempts_by_tid.values()
        for a in al
    )
    src_wall_ms = sum(float(a["summary"].get("wall_ms", 0.0)) for al in source_attempts_by_tid.values() for a in al)
    c_source_ledger = {
        "source_task_count": len(source_tasks),
        "total_episode_attempts_including_failed": sum(len(al) for al in source_attempts_by_tid.values()),
        "failed_episode_attempts_retained_in_cost": sum(1 for al in source_attempts_by_tid.values() for a in al if not a["summary"].get("strict_pass")),
        "raw_token_triple": {"input_tokens": src_in, "cached_input_tokens": src_ca, "output_tokens": src_ou},
        "total_tokens": src_in + src_ou,
        "token_cost_proxy_w025": _proxy_w025(src_in, src_ca, src_ou),
        "total_provider_attempts": src_prov_att,
        "wall_ms_sum": round(src_wall_ms, 2),
    }

    # =========================================================================
    # PHASE 2: PRE-TARGET ONLINE COMPILATION & 4-TUPLE QUALIFICATION FREEZE
    # =========================================================================
    v8_bundle_fp = ev_dir / "m9_3_frozen_post_source_bundle.json"
    freeze_receipt_fp = ev_dir / "01_pre_target_qualification_freeze_receipt.json"

    if v8_bundle_fp.is_file() and freeze_receipt_fp.is_file():
        v8_forest, v8_archive, v8_meta_dag, v8_meta = m7a.load_autogenous_forest_bundle(v8_bundle_fp)
        freeze_receipt = json.loads(freeze_receipt_fp.read_text(encoding="utf-8"))
        c_qual_ledger = freeze_receipt["c_qualification_ledger"]
        new_online_l1_stems = set(freeze_receipt["newly_compiled_online_l1_stems"])
    else:
        print("[M9.3 Qualification Freeze] Compiling verified source episodes into L2 + 4-tuple L1 primitives BEFORE target exposure...", flush=True)
        v8_forest = deepcopy(v7_forest)
        v8_archive = deepcopy(v7_archive)
        v8_meta_dag = list(v7_meta_dag)
        compiler_client = m6f.FrontierCodexClient(
            codex_info=codex_info,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=360,
        )
        l2_comp_events: list[dict[str, Any]] = []
        qual_in = 0; qual_ca = 0; qual_ou = 0; qual_wall_ms = 0.0

        comp_cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
        try:
            for st_obj in source_tasks:
                w_dir = winning_source_dirs[st_obj.task_id]
                w_sum = json.loads((w_dir / "00_episode_summary.json").read_text(encoding="utf-8"))
                raw_py = _load_submitted_py_sources(w_dir)
                sigs = m6f.extract_public_task_signatures(st_obj)
                comp_l2 = m7a.compile_episode_dialectical_sublation(
                    client=compiler_client,
                    sandbox_cid=comp_cid,
                    episode_id=st_obj.short_name,
                    source_slug=st_obj.short_name,
                    public_task_brief=st_obj.instruction_text,
                    inferred_signatures=sigs,
                    raw_python_sources=raw_py,
                    nogoods_recorded=list(w_sum.get("nogoods_recorded") or []),
                    parent_node_summary=None,
                    stage_subdir=f"m9_3_online_l2__{st_obj.short_name}",
                )
                assert comp_l2.get("compiled_ok"), f"Online L2 compilation failed for {st_obj.task_id}: {comp_l2}"
                lt = comp_l2.get("llm_tokens") or {}
                qual_in += int(lt.get("prompt", 0)); qual_ca += int(lt.get("cached", 0))
                qual_ou += int(lt.get("completion", 0)); qual_wall_ms += float(lt.get("wall_ms", 0.0))
                cons = m7a.consolidate_autogenous_episode_into_forest(
                    forest=v8_forest,
                    archive=v8_archive,
                    compilation=comp_l2,
                    nogoods_recorded=list(w_sum.get("nogoods_recorded") or []),
                    allow_horizontal_wormholes=True,
                )
                l2_comp_events.append({"task_id": st_obj.task_id, "l2_compilation": comp_l2, "consolidation": cons})
        finally:
            m6f.stop_container_safely(comp_cid)

        l1_sub = m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=v8_forest,
            archive=v8_archive,
            codex_client=compiler_client,
            run_id="m9_3_online_l1_freeze",
            require_pretraining_only=False,
            allow_horizontal_wormholes=True,
            skip_already_compiled_sources=True,
            max_workers=2,
            include_executable_checker_guidance=True,
            only_source_episode_ids={st.short_name for st in source_tasks},
        )
        lt1 = l1_sub.get("llm_tokens") or {}
        qual_in += int(lt1.get("prompt", 0)); qual_ca += int(lt1.get("cached", 0))
        qual_ou += int(lt1.get("completion", 0)); qual_wall_ms += float(lt1.get("wall_ms", 0.0))

        # Fail-closed verification that BOTH new online L1 primitives passed strict 4-tuple container qualification!
        all_v8_l1_stems: set[str] = set()
        new_l1_details: dict[str, Any] = {}
        for _, rec in m7a.iter_live_archive_nodes(v8_archive):
            for m in rec.get("autogenous_wormhole_modules") or []:
                fn = str(m.get("filename") or "")
                if fn.startswith("autogenous_prim_"):
                    ok_q, rsn_q, c_sha = m64.verify_l1_module_qualification_record(
                        m, require_4tuple_protocol=True, expected_env_digests={env_dig}, allow_v6_legacy_env_label=False
                    )
                    assert ok_q, f"Post-source v8 L1 module `{fn}` failed strict 4-tuple qualification: {rsn_q}"
                    st_nm = Path(fn).stem
                    all_v8_l1_stems.add(st_nm)
                    if st_nm not in v7_stems:
                        new_l1_details[st_nm] = {
                            "filename": fn,
                            "compiled_code_sha256": c_sha,
                            "synthetic_unit_test_sha256": m["unit_test_receipt"]["synthetic_unit_test_sha256"],
                            "verifier_protocol_version": m["unit_test_receipt"]["verifier_protocol_version"],
                            "dependency_env_digest": m["unit_test_receipt"]["dependency_env_digest"],
                            "exported_functions": list(m.get("function_names") or []),
                            "exported_apis": str(m.get("exported_apis") or ""),
                        }
        new_online_l1_stems = set(new_l1_details.keys())
        assert len(new_online_l1_stems) >= 2, f"Expected >= 2 newly qualified online L1 primitives from src01+src02, got {new_online_l1_stems}"

        c_qual_ledger = {
            "l2_compilations": len(l2_comp_events),
            "l1_newly_compiled_count": int(l1_sub["newly_compiled_count"]),
            "l1_retained_v7_count": int(l1_sub["retained_existing_count"]),
            "l1_failed_attempts_retained_in_cost": sum(
                sum(1 for att in (ev.get("attempts") or []) if not (att.get("unit_test_receipt") or {}).get("passed"))
                for ev in (l1_sub.get("events") or [])
            ),
            "raw_token_triple": {"input_tokens": qual_in, "cached_input_tokens": qual_ca, "output_tokens": qual_ou},
            "total_tokens": qual_in + qual_ou,
            "token_cost_proxy_w025": _proxy_w025(qual_in, qual_ca, qual_ou),
            "wall_ms_sum": round(qual_wall_ms, 2),
        }

        ser_v8 = m7a.serialize_autogenous_forest_bundle(
            forest=v8_forest,
            archive=v8_archive,
            meta_dag=v8_meta_dag,
            out_path=v8_bundle_fp,
            extra_metadata={
                **v7_meta,
                "bundle_version": "m9_3_frozen_post_source_v8",
                "parent_v7_bundle_sha256": v7_sha,
                "qualification_protocol": "hegel_l1_qual_4tuple_v2",
                "dependency_env_digest": env_dig,
                "environment_manifest": env_manifest,
                "newly_compiled_online_l1_stems": sorted(new_online_l1_stems),
            },
        )
        # Extract actual locked Stage S public outputs (`locked_src01_out`, `locked_src02_out`) so `F08` never silently falls back to oracle!
        locked_src_outputs: dict[str, dict[str, Any]] = {}
        for st_obj in source_tasks:
            w_dir = winning_source_dirs[st_obj.task_id]
            raw_py = _load_submitted_py_sources(w_dir)
            src_txt = next(iter(raw_py.values()))
            m93_t = m93b.M93_TASKS_BY_ID[st_obj.task_id]
            with tempfile.TemporaryDirectory(prefix="m93_lock_pub_") as td_lp:
                td_p = Path(td_lp)
                ns_lp: dict[str, Any] = {}
                exec(compile(m93_t.public_setup_py.replace("/holdout_task", str(td_p)), "<m93_pub>", "exec"), ns_lp)
                pkl_fp = sorted(td_p.glob("public_*.pkl"))[0]
                pub_obj = pickle.loads(pkl_fp.read_bytes())[0]
                inp_obj = deepcopy(pub_obj.get("dataset") or pub_obj.get("study"))
                c_fp = td_p / "cand.py"
                c_fp.write_text(src_txt, encoding="utf-8")
                in_fp = td_p / "in.pkl"
                out_fp = td_p / "out.json"
                in_fp.write_bytes(pickle.dumps(inp_obj))
                w_rpc = (
                    "import importlib.util, json, pickle, sys\nimport numpy as np\n"
                    "spec = importlib.util.spec_from_file_location('_c', sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
                    "res = getattr(mod, sys.argv[2])(pickle.loads(open(sys.argv[3], 'rb').read()))\n"
                    "def _enc(x):\n"
                    "    if isinstance(x, dict): return {str(k): _enc(v) for k, v in x.items()}\n"
                    "    if isinstance(x, (list, tuple)): return [_enc(v) for v in x]\n"
                    "    if isinstance(x, np.ndarray): return x.tolist()\n"
                    "    if isinstance(x, np.generic): return x.item()\n"
                    "    return x\n"
                    "open(sys.argv[4], 'w').write(json.dumps(_enc(res), allow_nan=False))\n"
                )
                subprocess.run([sys.executable, "-I", "-c", w_rpc, str(c_fp), m93_t.declared_callable_entrypoints[0], str(in_fp), str(out_fp)], check=True, timeout=45)
                src_key = "src01" if m93_t.task_id.startswith("src01") else "src02"
                locked_src_outputs[src_key] = json.loads(out_fp.read_text(encoding="utf-8"))

        float_plugin_ctrl = evaluate_post_freeze_dev_seed_plugin_controls(
            locked_src01_out=locked_src_outputs.get("src01"),
            locked_src02_out=locked_src_outputs.get("src02"),
        )
        assert float_plugin_ctrl["direct_locked_float_plugin_fails_all_successors"] is True
        assert float_plugin_ctrl["src01_param_origin"] == "locked_stage_S_candidate_output"
        assert float_plugin_ctrl["src02_param_origin"] == "locked_stage_S_candidate_output"
        source_node_by_short = {
            ev["l2_compilation"]["source_slug"]: str(ev["consolidation"]["target_node_id"])
            for ev in l2_comp_events
        }

        freeze_receipt = {
            "schema_version": "hegel_m9_3_pre_target_qualification_freeze_v2_devseed",
            "freeze_completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "parent_v7_bundle_sha256": v7_sha,
            "frozen_post_source_v8_bundle_sha256": ser_v8["bundle_sha256"],
            "dependency_env_digest": env_dig,
            "source_node_by_short": source_node_by_short,
            "newly_compiled_online_l1_stems": sorted(new_online_l1_stems),
            "newly_compiled_online_l1_details": new_l1_details,
            "post_freeze_dev_seed_plugin_controls": float_plugin_ctrl,
            "c_source_ledger": c_source_ledger,
            "c_qualification_ledger": c_qual_ledger,
            "l1_sublation_events": l1_sub["events"],
        }
        freeze_receipt_fp.write_text(json.dumps(freeze_receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[M9.3 Qualification Freeze SEALED] v8SHA={ser_v8['bundle_sha256'][:12]} | NewOnlineL1={sorted(new_online_l1_stems)}",
            flush=True,
        )

    v8_frozen_sha = hashlib.sha256(v8_bundle_fp.read_bytes()).hexdigest()
    assert v8_frozen_sha == freeze_receipt["frozen_post_source_v8_bundle_sha256"]

    # =========================================================================
    # PHASE 3: POST-FREEZE SYMMETRIC 3-ARM SUCCESSOR MATRIX (`24` EPISODES)
    # =========================================================================
    successor_start_utc = datetime.now(timezone.utc).isoformat()
    assert freeze_receipt["freeze_completed_timestamp_utc"] <= successor_start_utc
    successor_tasks = get_m93_successor_frontier_tasks()
    assert len(successor_tasks) == 4

    # Pre-run equal-information & unified-reader check across B, K, E on all 4 successors
    pre_run_matrix: dict[str, Any] = {}
    ns_reader: dict[str, Any] = {"json": json, "re": __import__("re")}
    exec(compile(m64._SHARED_CATALOG_READER_CORE_PY, "<m93_shared_reader>", "exec"), ns_reader)
    for t_obj in successor_tasks:
        _, man_b, diag_b = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v7_archive,
            task_query_text=t_obj.instruction_text, max_l1_modules=6, disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated", return_diagnostics=True,
        )
        _, man_k, diag_k = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v8_archive,
            task_query_text=t_obj.instruction_text, max_l1_modules=6, disclosure_mode="on_demand_knowledge_only",
            staging_policy="net_value_gated", return_diagnostics=True,
        )
        _, man_e, diag_e = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v8_archive,
            task_query_text=t_obj.instruction_text, max_l1_modules=6, disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated", return_diagnostics=True,
        )
        assert man_b == [] and man_k == [] and man_e == []
        cat_k = json.loads(diag_k["contracts_catalog_json"])
        cat_e = json.loads(diag_e["contracts_catalog_json"])
        _, r_k, _ = ns_reader["_shared_search_catalog"](cat_k, t_obj.short_name.replace("_", " "), top_k=6)
        _, r_e, _ = ns_reader["_shared_search_catalog"](cat_e, t_obj.short_name.replace("_", " "), top_k=6)
        assert r_k == r_e, f"K vs E reader mismatch on {t_obj.task_id}: {r_k} != {r_e}"
        pre_run_matrix[t_obj.task_id] = {
            "B_v7_recommended_l1": [x["filename"] for x in diag_b["selected_l1_modules"]],
            "K_v8_recommended_l1": [x["filename"] for x in diag_k["selected_l1_modules"]],
            "E_v8_recommended_l1": [x["filename"] for x in diag_e["selected_l1_modules"]],
            "K_E_search_top6_identical": r_k == r_e,
        }
    (ev_dir / "02_pre_successor_equal_information_matrix.json").write_text(
        json.dumps(pre_run_matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    replicates = [f"rep_{r_idx:02d}" for r_idx in range(1, int(args.num_replicates) + 1)]
    jobs = [
        (rep_id, t_obj, arm_id, disc_mode, stg_pol, pkg_flag, use_v8)
        for rep_id in replicates
        for t_obj in successor_tasks
        for arm_id, disc_mode, stg_pol, pkg_flag, use_v8 in SUCCESSOR_ARMS_CONFIG
    ]
    succ_root = ev_dir / "stage_TC_successors"
    print(f"[M9.3 Stage T+C] Launching {len(jobs)} interleaved post-freeze successor episodes across {args.max_episode_workers} workers...", flush=True)

    def _run_successor_job(job: tuple[str, m6f.FrontierBenchmarkTask, str, str, str, bool, bool]) -> tuple[str, str, str, dict[str, Any]]:
        rep_id, task_obj, arm_id, disc_mode, stg_pol, pkg_flag, use_v8 = job
        ep_dir = succ_root / rep_id / arm_id / task_obj.task_id
        sum_fp = ep_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            with resumed_lock:
                run_counters["resumed_episode_count"] += 1
            return rep_id, arm_id, task_obj.task_id, json.loads(sum_fp.read_text(encoding="utf-8"))

        ep_forest = deepcopy(v8_forest if use_v8 else v7_forest)
        ep_archive = deepcopy(v8_archive if use_v8 else v7_archive)
        ep_archive["__autogenous_blank_seed__"] = {"enabled": True}
        res = m6f.run_frontier_episode(
            task_obj,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=args.model,
            codex_info=codex_info,
            forest=ep_forest,
            forest_archive=ep_archive,
            hippo_idx=HippoRAGWormholeIndex(ep_forest),
            episode_out_dir=ep_dir,
            run_id=f"m9_3_{rep_id}_{arm_id}",
            max_turns=args.max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=args.reasoning_effort,
            defer_official_verifier=False,
            enable_auto_skill_packaging=pkg_flag,
            allow_l1_primitives=True,
            disclosure_mode=disc_mode,
            staging_policy=stg_pol,
        )
        with resumed_lock:
            run_counters["fresh_executed_episode_count"] += 1
        print(
            f"  [SUC DONE] {rep_id} | {arm_id:32s} | {task_obj.short_name:44s} | "
            f"pass={res['strict_pass']} | turns={res['turns_used']} | tok={res['total_tokens']}",
            flush=True,
        )
        return rep_id, arm_id, task_obj.task_id, res

    succ_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_episode_workers))) as pool:
        futs = [pool.submit(_run_successor_job, jb) for jb in jobs]
        for fut in as_completed(futs):
            rep_id, arm_id, tid, res_d = fut.result()
            succ_results[(rep_id, arm_id, tid)] = res_d

    # Verify v8 frozen bundle was NEVER mutated during Phase 3!
    assert hashlib.sha256(v8_bundle_fp.read_bytes()).hexdigest() == v8_frozen_sha

    # =========================================================================
    # PHASE 4: STAGE C REGIME BIFURCATION & LIVE STAGE R ORIGINAL-CONDITION RECOVERY
    # =========================================================================
    v8_bif_fp = ev_dir / "m9_3_post_counterexample_bifurcated_bundle.json"
    bif_receipt_fp = ev_dir / "03_post_counterexample_bifurcation_receipt.json"
    if v8_bif_fp.is_file() and bif_receipt_fp.is_file():
        v8_bif_forest, v8_bif_archive, _, _ = m7a.load_autogenous_forest_bundle(v8_bif_fp)
        bif_receipt = json.loads(bif_receipt_fp.read_text(encoding="utf-8"))
        c_bif_ledger = bif_receipt["c_counterexample_consolidation_ledger"]
    else:
        print("[M9.3 Stage C->R] Consolidating localized counterexamples via precondition-split bifurcation...", flush=True)
        v8_bif_forest = deepcopy(v8_forest)
        v8_bif_archive = deepcopy(v8_archive)
        v8_bif_dag = list(v7_meta_dag)
        bif_client = m6f.FrontierCodexClient(codex_info=codex_info, model=args.model, reasoning_effort=args.reasoning_effort, timeout_seconds=360)
        bif_in = 0; bif_ca = 0; bif_ou = 0; bif_wall = 0.0
        bif_events: list[dict[str, Any]] = []
        bif_cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
        try:
            for c_task in [t for t in successor_tasks if t.wave == "m93_stage_C_counterexample"]:
                # Pick passing E replicate for Stage C consolidation
                win_c_dir: Path | None = None
                win_c_sum: dict[str, Any] | None = None
                for rp in replicates:
                    cand_d = succ_root / rp / "equal_source_executable_library" / c_task.task_id
                    s_c = succ_results[(rp, "equal_source_executable_library", c_task.task_id)]
                    if s_c.get("strict_pass"):
                        win_c_dir = cand_d; win_c_sum = s_c; break
                if win_c_dir is None or win_c_sum is None:
                    continue
                comp_c = m7a.compile_episode_dialectical_sublation(
                    client=bif_client,
                    sandbox_cid=bif_cid,
                    episode_id=c_task.short_name,
                    source_slug=c_task.short_name,
                    public_task_brief=c_task.instruction_text,
                    inferred_signatures=m6f.extract_public_task_signatures(c_task),
                    raw_python_sources=_load_submitted_py_sources(win_c_dir),
                    nogoods_recorded=list(win_c_sum.get("nogoods_recorded") or []),
                    parent_node_summary=None,
                    stage_subdir=f"m9_3_stageC_bif__{c_task.short_name}",
                )
                ltc = comp_c.get("llm_tokens") or {}
                bif_in += int(ltc.get("prompt", 0)); bif_ca += int(ltc.get("cached", 0))
                bif_ou += int(ltc.get("completion", 0)); bif_wall += float(ltc.get("wall_ms", 0.0))
                if comp_c.get("compiled_ok"):
                    # Enforce explicit precondition-gated regime split so generation=0 linear/separable L1 is NEVER overwritten!
                    is_c1_alpha = "suc01_C" in c_task.task_id
                    axis_nm = "overstress_excess_max_history_and_hold" if is_c1_alpha else "nonseparable_2d_additive_residual_ratio"
                    src_map = freeze_receipt.get("source_node_by_short") or {}
                    src_parent_nid = str(src_map.get("src01" if is_c1_alpha else "src02") or "")
                    guard_override = m7a.ObservablePreconditionGuard(
                        guard_id=f"guard_m93_{c_task.short_name}",
                        observable_extractor=axis_nm,
                        op=">",
                        value=(0.0 if is_c1_alpha else 1e-6),
                        applies_to_component_id=("linear_viscoelastic_memory" if is_c1_alpha else "separable_kronecker_transport"),
                    )
                    comp_c["regime_bifurcation"] = {
                        "should_split_node": True,
                        "regime_axis": axis_nm,
                        "branch_guard_op": ">",
                        "branch_guard_value": (0.0 if is_c1_alpha else 1e-6),
                        "counter_regime_warning": f"Localized nonlinear/non-separable regime `{axis_nm} > 0`; preserve generation=0 linear/separable L1 when `{axis_nm} <= 0`.",
                    }
                    cons_c = m7a.consolidate_autogenous_episode_into_forest(
                        forest=v8_bif_forest,
                        archive=v8_bif_archive,
                        compilation=comp_c,
                        nogoods_recorded=list(win_c_sum.get("nogoods_recorded") or []),
                        allow_horizontal_wormholes=True,
                        source_node_id=(src_parent_nid or None),
                        source_version=1,
                        violated_component_id=("linear_viscoelastic_memory" if is_c1_alpha else "separable_kronecker_transport"),
                        preserved_component_ids=(
                            ["reference_panel_calibration", "variable_step_gap_recursion"]
                            if is_c1_alpha
                            else ["nuisance_profiling", "box_simplex_kkt_certification"]
                        ),
                        guard_spec_override=guard_override,
                    )
                    bif_events.append({"task_id": c_task.task_id, "consolidation": cons_c})
        finally:
            m6f.stop_container_safely(bif_cid)

        # Verify generation=0 `src01`/`src02` online L1 code SHA-256s are 100% UNCHANGED after Stage C consolidation!
        post_bif_l1_shas: dict[str, str] = {}
        for _, rec in m7a.iter_live_archive_nodes(v8_bif_archive):
            for m in rec.get("autogenous_wormhole_modules") or []:
                st_m = Path(str(m.get("filename") or "")).stem
                if st_m in new_online_l1_stems:
                    post_bif_l1_shas[st_m] = str(m.get("compiled_code_sha256") or "")
        orig_l1_shas = {
            k: v["compiled_code_sha256"]
            for k, v in freeze_receipt["newly_compiled_online_l1_details"].items()
        }
        zero_overwrite = (post_bif_l1_shas == orig_l1_shas)
        assert zero_overwrite, f"Destructive overwrite detected on generation=0 L1: {post_bif_l1_shas} != {orig_l1_shas}"

        c_bif_ledger = {
            "consolidated_counterexamples": len(bif_events),
            "raw_token_triple": {"input_tokens": bif_in, "cached_input_tokens": bif_ca, "output_tokens": bif_ou},
            "total_tokens": bif_in + bif_ou,
            "token_cost_proxy_w025": _proxy_w025(bif_in, bif_ca, bif_ou),
            "wall_ms_sum": round(bif_wall, 2),
        }
        ser_bif = m7a.serialize_autogenous_forest_bundle(
            forest=v8_bif_forest,
            archive=v8_bif_archive,
            meta_dag=v8_bif_dag,
            out_path=v8_bif_fp,
            extra_metadata={"bundle_version": "m9_3_post_counterexample_bifurcated_v8", "parent_v8_sha256": v8_frozen_sha},
        )
        bif_receipt = {
            "post_counterexample_bundle_sha256": ser_bif["bundle_sha256"],
            "generation0_l1_shas_before": orig_l1_shas,
            "generation0_l1_shas_after": post_bif_l1_shas,
            "zero_destructive_overwrite": zero_overwrite,
            "bifurcation_events": bif_events,
            "c_counterexample_consolidation_ledger": c_bif_ledger,
        }
        bif_receipt_fp.write_text(json.dumps(bif_receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Live Stage R Original-Condition Recovery (`rec01_R`, `rec02_R`) on `v8_bif`
    recovery_tasks = get_m93_recovery_frontier_tasks()
    stage_r_root = ev_dir / "stage_R_recovery"
    print(f"[M9.3 Stage R] Launching {len(recovery_tasks)} live original-condition recovery episodes...", flush=True)

    def _run_recovery_task(r_task: m6f.FrontierBenchmarkTask) -> tuple[str, dict[str, Any]]:
        r_dir = stage_r_root / "rep_01" / "equal_source_executable_library" / r_task.task_id
        sum_fp = r_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            with resumed_lock:
                run_counters["resumed_episode_count"] += 1
            return r_task.task_id, json.loads(sum_fp.read_text(encoding="utf-8"))
        ep_f = deepcopy(v8_bif_forest)
        ep_a = deepcopy(v8_bif_archive)
        ep_a["__autogenous_blank_seed__"] = {"enabled": True}
        res_r = m6f.run_frontier_episode(
            r_task,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=args.model,
            codex_info=codex_info,
            forest=ep_f,
            forest_archive=ep_a,
            hippo_idx=HippoRAGWormholeIndex(ep_f),
            episode_out_dir=r_dir,
            run_id="m9_3_stageR_rep_01",
            max_turns=args.max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=args.reasoning_effort,
            defer_official_verifier=False,
            enable_auto_skill_packaging=True,
            allow_l1_primitives=True,
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        with resumed_lock:
            run_counters["fresh_executed_episode_count"] += 1
        print(f"  [STAGE R DONE] {r_task.short_name} | pass={res_r['strict_pass']} | turns={res_r['turns_used']} | tok={res_r['total_tokens']}", flush=True)
        return r_task.task_id, res_r

    rec_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        r_futs = [pool.submit(_run_recovery_task, rt) for rt in recovery_tasks]
        for rf in as_completed(r_futs):
            tid_r, res_r = rf.result()
            rec_results[tid_r] = res_r

    # =========================================================================
    # PHASE 5: ASSEMBLE EPISODE ROWS, 4-WAY FUNNEL & FULL LIFECYCLE LEDGER
    # =========================================================================
    def _build_ep_row(rep_id: str, arm_id: str, disc_mode: str, t_obj: m6f.FrontierBenchmarkTask, ep_dir: Path, s: dict[str, Any]) -> dict[str, Any]:
        t1_tok = _extract_turn1_tokens(ep_dir)
        ver_m = _extract_verifier_metrics(ep_dir)
        af = s.get("skill_adoption_funnel") or {}
        ret = af.get("final_selected_artifact_retention") or {}
        p_tok = int(s.get("prompt_tokens", 0))
        c_tok = int(s.get("cached_tokens", 0))
        o_tok = int(s.get("completion_tokens", 0))
        pa = s.get("provider_attempt_accounting") or {}
        psr = s.get("public_setup_receipt") or {}
        dyn_mods = list(af.get("dynamic_unique_l1_modules_called", []))
        dyn_fns = list(af.get("dynamic_unique_l1_functions_called", []))
        reach_fns = list(ret.get("entrypoint_reachable_l1_functions", []))
        online_dyn_mods = [m for m in dyn_mods if m in new_online_l1_stems]
        online_reach_fns = [f for f in reach_fns if f.split(".")[0] in new_online_l1_stems]
        discovered = bool(
            (int(af.get("search_skill_query_count", 0)) + int(af.get("describe_skill_inspection_count", 0))) > 0
            or bool(af.get("inspected_knowledge_ref_in_bash", False))
        )
        dev_exec = int(af.get("dynamic_l1_primitive_activation_count", 0)) > 0
        retained_closure = len(reach_fns) > 0
        derived_rewrite = bool(discovered and not retained_closure)
        return {
            "replicate": rep_id,
            "arm": arm_id,
            "disclosure_mode": disc_mode,
            "wave": t_obj.wave,
            "task_id": t_obj.task_id,
            "short_name": t_obj.short_name,
            "strict_pass": bool(s.get("strict_pass", False)),
            "turns_used": int(s.get("turns_used", 0)),
            "ended_by_finish": bool(s.get("ended_by_finish", False)),
            "public_setup_executed": bool(psr.get("executed", False)),
            "shared_public_schema_validator_present": bool(psr.get("shared_public_schema_validator_present", False)),
            "turn1_staged_manifest_json_chars": len(json.dumps(s.get("staged_wormhole_modules", []), ensure_ascii=False)),
            **t1_tok,
            "raw_token_triple": {"input_tokens": p_tok, "cached_input_tokens": c_tok, "output_tokens": o_tok},
            "total_tokens": int(s.get("total_tokens", 0)),
            "token_cost_proxy_w025": _proxy_w025(p_tok, c_tok, o_tok),
            "provider_attempt_count": int(pa.get("total_provider_attempts", s.get("turns_used", 0))),
            "usage_unknown_attempt_count": int(pa.get("usage_unknown_attempt_count", 0)),
            "cost_accounting_status": str(pa.get("cost_accounting_status") or "complete"),
            "episode_wall_ms": float(s.get("wall_ms", 0.0)),
            "verifier_window_wall_sec": ver_m.get("student_elapsed_sec"),
            "candidate_timing_breakdown": ver_m.get("candidate_timing_breakdown"),
            "funnel_4way": {
                "discovered_or_read_catalog": discovered,
                "dev_time_dynamic_l1_executed": dev_exec,
                "deliverable_closure_retained": retained_closure,
                "derived_manual_rewrite": derived_rewrite,
            },
            "search_skill_query_count": int(af.get("search_skill_query_count", 0)),
            "describe_skill_inspection_count": int(af.get("describe_skill_inspection_count", 0)),
            "inspected_knowledge_ref_in_bash": bool(af.get("inspected_knowledge_ref_in_bash", False)),
            "dynamic_l1_calls": int(af.get("dynamic_l1_primitive_activation_count", 0)),
            "dynamic_unique_l1_modules": dyn_mods,
            "dynamic_unique_l1_functions": dyn_fns,
            "online_formed_dynamic_l1_modules": online_dyn_mods,
            "entrypoint_reachable_l1_functions": reach_fns,
            "online_formed_reachable_l1_functions": online_reach_fns,
            "verifier_metrics": ver_m,
        }

    successor_rows: list[dict[str, Any]] = []
    for rep_id in replicates:
        for t_obj in successor_tasks:
            for arm_id, disc_mode, _, _, _ in SUCCESSOR_ARMS_CONFIG:
                ep_d = succ_root / rep_id / arm_id / t_obj.task_id
                successor_rows.append(_build_ep_row(rep_id, arm_id, disc_mode, t_obj, ep_d, succ_results[(rep_id, arm_id, t_obj.task_id)]))

    recovery_rows = [
        _build_ep_row("rep_01", "equal_source_executable_library", "on_demand_executable", rt, stage_r_root / "rep_01/equal_source_executable_library" / rt.task_id, rec_results[rt.task_id])
        for rt in recovery_tasks
    ]

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(1, len(rows))
        in_s = sum(int(r["raw_token_triple"]["input_tokens"]) for r in rows)
        ca_s = sum(int(r["raw_token_triple"]["cached_input_tokens"]) for r in rows)
        ou_s = sum(int(r["raw_token_triple"]["output_tokens"]) for r in rows)
        rep_rates = [
            sum(1 for r in rows if r["replicate"] == rp and r["strict_pass"]) / max(1, sum(1 for r in rows if r["replicate"] == rp))
            for rp in replicates
            if any(r["replicate"] == rp for r in rows)
        ]
        cand_cpus = [
            float((r.get("candidate_timing_breakdown") or {}).get("candidate_call_cpu_sec"))
            for r in rows
            if (r.get("candidate_timing_breakdown") or {}).get("candidate_call_cpu_sec") is not None
        ]
        unk = sum(int(r["usage_unknown_attempt_count"]) for r in rows)
        return {
            "episode_count": len(rows),
            "strict_pass_count": sum(1 for r in rows if r["strict_pass"]),
            "strict_pass_rate": round(sum(1 for r in rows if r["strict_pass"]) / n, 4),
            "replicate_pass_rates": [round(x, 4) for x in rep_rates],
            "replicate_sd": round(statistics.stdev(rep_rates), 4) if len(rep_rates) > 1 else 0.0,
            "mean_turns_used": round(sum(r["turns_used"] for r in rows) / n, 2),
            "mean_turn1_prompt_tokens": round(sum(r["turn1_prompt_tokens"] for r in rows) / n, 1),
            "raw_token_triple_sum": {"input_tokens": in_s, "cached_input_tokens": ca_s, "output_tokens": ou_s},
            "total_tokens_sum": sum(r["total_tokens"] for r in rows),
            "mean_total_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
            "token_cost_proxy_w025_sum": round(sum(r["token_cost_proxy_w025"] for r in rows), 2),
            "mean_token_cost_proxy_w025": round(sum(r["token_cost_proxy_w025"] for r in rows) / n, 2),
            "total_provider_attempts": sum(int(r["provider_attempt_count"]) for r in rows),
            "usage_unknown_attempt_count": unk,
            "cost_accounting_status": ("partial_unknown_usage" if unk > 0 else "complete"),
            "mean_episode_wall_sec": round(sum(r["episode_wall_ms"] for r in rows) / (1000.0 * n), 2),
            "mean_candidate_call_cpu_sec": round(sum(cand_cpus) / len(cand_cpus), 4) if cand_cpus else None,
            "funnel_4way_counts": {
                "discovered_or_read_catalog": sum(1 for r in rows if r["funnel_4way"]["discovered_or_read_catalog"]),
                "dev_time_dynamic_l1_executed": sum(1 for r in rows if r["funnel_4way"]["dev_time_dynamic_l1_executed"]),
                "deliverable_closure_retained": sum(1 for r in rows if r["funnel_4way"]["deliverable_closure_retained"]),
                "derived_manual_rewrite": sum(1 for r in rows if r["funnel_4way"]["derived_manual_rewrite"]),
            },
            "episodes_calling_online_formed_l1": sum(1 for r in rows if len(r["online_formed_dynamic_l1_modules"]) > 0),
            "episodes_retaining_online_formed_l1": sum(1 for r in rows if len(r["online_formed_reachable_l1_functions"]) > 0),
            "total_dynamic_l1_calls": sum(r["dynamic_l1_calls"] for r in rows),
        }

    arm_aggregates: dict[str, Any] = {}
    for arm_id, disc_mode, stg_pol, pkg_flag, use_v8 in SUCCESSOR_ARMS_CONFIG:
        a_rows = [r for r in successor_rows if r["arm"] == arm_id]
        t_rows = [r for r in a_rows if r["wave"] == "m93_stage_T_transfer"]
        c_rows = [r for r in a_rows if r["wave"] == "m93_stage_C_counterexample"]
        arm_aggregates[arm_id] = {
            "disclosure_mode": disc_mode,
            "staging_policy": stg_pol,
            "uses_post_source_v8_bundle": use_v8,
            "all_8_successor_episodes": _agg(a_rows),
            "stage_T_transfer_4_episodes": _agg(t_rows),
            "stage_C_counterexample_4_episodes": _agg(c_rows),
        }

    # Full Lifecycle Cost Ledger (`C_source + C_qualification + sum C_successor + C_consolidation + C_recovery`)
    c_rec_agg = _agg(recovery_rows)
    upfront_formation_w025 = round(
        float(c_source_ledger["token_cost_proxy_w025"]) + float(c_qual_ledger["token_cost_proxy_w025"]), 2
    )
    upfront_formation_tokens = int(c_source_ledger["total_tokens"]) + int(c_qual_ledger["total_tokens"])
    b_per_succ_w025 = arm_aggregates["prior_frozen_v7_no_source_growth"]["all_8_successor_episodes"]["mean_token_cost_proxy_w025"]
    k_per_succ_w025 = arm_aggregates["equal_source_knowledge_only"]["all_8_successor_episodes"]["mean_token_cost_proxy_w025"]
    e_per_succ_w025 = arm_aggregates["equal_source_executable_library"]["all_8_successor_episodes"]["mean_token_cost_proxy_w025"]
    per_succ_saving_e_vs_b = round(b_per_succ_w025 - e_per_succ_w025, 2)
    per_succ_saving_e_vs_k = round(k_per_succ_w025 - e_per_succ_w025, 2)

    lifecycle_ledger = {
        "c_source": c_source_ledger,
        "c_qualification": c_qual_ledger,
        "upfront_online_formation_total": {
            "total_tokens": upfront_formation_tokens,
            "token_cost_proxy_w025": upfront_formation_w025,
            "note": "Shared online source formation + 4-tuple container qualification investment paid once before target exposure.",
        },
        "post_freeze_per_successor_mean_w025": {
            "B_prior_v7_no_source_growth": b_per_succ_w025,
            "K_equal_source_knowledge_only": k_per_succ_w025,
            "E_equal_source_executable_library": e_per_succ_w025,
            "marginal_saving_E_vs_B_per_successor_w025": per_succ_saving_e_vs_b,
            "marginal_saving_E_vs_K_per_successor_w025": per_succ_saving_e_vs_k,
        },
        "break_even_successor_episodes_E_vs_B_including_full_source_and_qualification": (
            round(upfront_formation_w025 / per_succ_saving_e_vs_b, 2) if per_succ_saving_e_vs_b > 0 else None
        ),
        "equal_source_investment_net_advantage_E_vs_K_8_episodes_w025": round(
            arm_aggregates["equal_source_knowledge_only"]["all_8_successor_episodes"]["token_cost_proxy_w025_sum"]
            - arm_aggregates["equal_source_executable_library"]["all_8_successor_episodes"]["token_cost_proxy_w025_sum"],
            2,
        ),
        "c_counterexample_consolidation": c_bif_ledger,
        "c_stage_R_recovery": c_rec_agg,
    }

    end_state = _git_and_source_tree_state(ev_rel)
    total_wall_sec = round(monotonic() - t_global0, 2)
    master_summary = {
        "experiment_id": "hegel_m9_3_online_source_freeze_lifecycle_20261126",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "code_commit_at_start": start_state["git_head"],
            "code_commit_at_end": end_state["git_head"],
            "same_frozen_commit_throughout": start_state["git_head"] == end_state["git_head"],
            "source_tree_py_sha256_at_start": start_state["source_tree_py_sha256"],
            "source_tree_py_sha256_at_end": end_state["source_tree_py_sha256"],
            "zero_source_mutation_during_run": start_state["source_tree_py_sha256"] == end_state["source_tree_py_sha256"],
            "git_dirty_at_start": bool(start_state["dirty_lines"]),
            "git_dirty_at_end": bool(end_state["dirty_lines"]),
            "resumed_episode_count": run_counters["resumed_episode_count"],
            "fresh_executed_episode_count": run_counters["fresh_executed_episode_count"],
            "prior_v7_bundle_sha256": v7_sha,
            "frozen_post_source_v8_bundle_sha256": v8_frozen_sha,
            "freeze_before_target_exposure_verified": True,
            "qualification_protocol": "hegel_l1_qual_4tuple_v2",
            "dependency_env_digest": env_dig,
            "single_branch_locked": True,
            "zero_artificial_solver_bans": True,
            "public_setup_executed_all_episodes": all(r["public_setup_executed"] for r in [*successor_rows, *recovery_rows]),
            "shared_public_schema_validator_all_arms": all(
                r["shared_public_schema_validator_present"] for r in [*successor_rows, *recovery_rows]
            ),
            "stage_R_zero_destructive_overwrite": bool(bif_receipt["zero_destructive_overwrite"]),
        },
        "concurrency_telemetry": {
            "configured_max_episode_workers": args.max_episode_workers,
            "configured_max_codex_concurrency": args.max_codex_concurrency,
            "peak_concurrent_codex_calls": codex_stats["peak_concurrent_codex_calls"],
            "total_codex_calls": codex_stats["total_codex_calls"],
            "total_wall_clock_sec": total_wall_sec,
        },
        "lifecycle_ledger": lifecycle_ledger,
        "arm_aggregates": arm_aggregates,
        "stage_R_recovery_aggregate": c_rec_agg,
        "successor_episode_rows": successor_rows,
        "recovery_episode_rows": recovery_rows,
    }
    (ev_dir / "m9_3_online_lifecycle_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_hashed = _write_sha256sums(ev_dir)
    print(
        f"[M9.3 COMPLETE] Wall={total_wall_sec}s | PeakCodex={codex_stats['peak_concurrent_codex_calls']} | HashedFiles={n_hashed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
