#!/usr/bin/env python3
"""Run the Sovereign M6.3 Causal AST + Partial-Subtree + Equal-Content Learned-Flat Live Study.

Executes 96 fresh live ChatGPT-subscription (`auth_mode == 'chatgpt'`, zero `ruoli.dev`) episodes:
  - `EqualContent_LearnedFlat_LiveGPT` (online Wave-1 Python AST writeback `max_hops=0` + BM25 re-index + pilot warmstart)
  - `Hegel_M6_3_Sovereign_LiveGPT` (verified instance-owner + Pareto multi-candidate partial-subtree archive +
    2-hop `d-sepset` wormhole AST role translation + composite node growth + slice precondition BFS + pilot warmstart)
across `gpt-6-luna` and `gpt-6.2-sol` (`24 tasks x 2 arms x 2 models = 96 fresh episodes`),
alongside sealed 192-episode M6.1 baselines (`Strong_Ordinary_LiveGPT`, `EqualAuth_Flat_LiveGPT`,
`Hegel_M5_Full_LiveGPT`, `Hegel_M6_1_ZeroLeakage_LiveGPT`) on identical `seed=20261121` tasks and `1.5/row` pricing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
from time import monotonic
from typing import Any

HEGEL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = HEGEL_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    CodexSubscriptionClient,
    SubscriptionQuotaPausedError,
    discover_chatgpt_subscription_codex_cli,
    verify_decision_provenance_chain,
)
from hegel_machine.cognition.m6_1_zero_leakage_ast_causal_benchmark import (
    EQUAL_UNIT_ROW_COST,
    build_m6_1_clean_k40_forest,
)
from hegel_machine.cognition.m6_3_sovereign_causal_wormhole_engine import (
    consolidate_wave1_sovereign_m6_3,
    run_single_m6_3_episode,
)
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves


ACTIVE_M6_3_ARMS: tuple[str, ...] = (
    "EqualContent_LearnedFlat_LiveGPT",
    "Hegel_M6_3_Sovereign_LiveGPT",
)


def _forest_state_sha16(node_archive: dict[str, dict[str, Any]]) -> str:
    canonical = {
        nid: {
            "rung": v.get("epistemic_rung"),
            "ref": v.get("confirmed_python_reference"),
            "pareto_n": len(v.get("pareto_confirmed_candidates") or []),
        }
        for nid, v in sorted(node_archive.items())
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _write_sha256sums(ev_dir: Path) -> int:
    lines: list[str] = []
    files = sorted(p for p in ev_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    for fp in files:
        lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.relative_to(ev_dir).as_posix()}")
    (ev_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_3_sovereign_ast_causal_96ep_20261121",
    )
    parser.add_argument("--models", nargs="+", default=["gpt-6-luna", "gpt-6.2-sol"])
    parser.add_argument("--arms", nargs="+", default=list(ACTIVE_M6_3_ARMS))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not codex_info.get("available") or codex_info.get("auth_mode") != "chatgpt":
        print(f"ERROR: ChatGPT subscription unavailable: {codex_info}", file=sys.stderr)
        return 2

    ev_dir: Path = args.evidence_dir
    ws_root = ev_dir / "workspaces_and_dags"
    ws_root.mkdir(parents=True, exist_ok=True)

    waves = build_m6_causal_24_waves(seed=20261121)
    models = list(args.models)
    arms = tuple(args.arms)

    state_by_ma: dict[tuple[str, str], tuple[Any, dict[str, dict[str, Any]], HippoRAGWormholeIndex, FlatEqualMetadataIndex]] = {}
    for m in models:
        for arm in arms:
            f_obj, f_arch = build_m6_1_clean_k40_forest()
            state_by_ma[(m, arm)] = (f_obj, f_arch, HippoRAGWormholeIndex(f_obj), FlatEqualMetadataIndex(f_obj))

    all_receipts: list[dict[str, Any]] = []
    quota_paused: dict[str, str] = {}
    pause_lock = threading.Lock()
    t0 = monotonic()

    def _exec_one(model: str, arm: str, ds: Any) -> dict[str, Any] | None:
        with pause_lock:
            if model in quota_paused:
                return None
        f_obj, f_arch, h_idx, fl_idx = state_by_ma[(model, arm)]
        f_sha = _forest_state_sha16(f_arch)
        ep_dir = ws_root / ds.wave / f"{model}__{arm}__{ds.task_id}"
        rec_p = ep_dir / "episode_receipt.json"
        dag_p = ep_dir / "decision_provenance_dag.json"

        # M6-12.1 Fix: Cache valid ONLY if exact `forest_snapshot_sha16` and `max_turns` match!
        if rec_p.is_file() and dag_p.is_file():
            try:
                cached = json.loads(rec_p.read_text(encoding="utf-8"))
                dag_obj = json.loads(dag_p.read_text(encoding="utf-8"))
                if (
                    dag_obj.get("provenance_verification", {}).get("valid")
                    and cached.get("forest_snapshot_sha16") == f_sha
                    and cached.get("max_turns") == args.max_turns
                ):
                    cached["pilot_rows_snapshot"] = list(ds.pilot_rows)
                    return cached
            except Exception:
                pass

        client = CodexSubscriptionClient(
            codex_info=codex_info,
            model=model,
            reasoning_effort="medium",
            timeout_seconds=180,
        )
        try:
            rec, records = run_single_m6_3_episode(
                ds,
                arm,
                client=client,
                workspace_dir=ep_dir,
                forest=f_obj,
                node_archive=f_arch,
                hippo_idx=h_idx,
                flat_idx=fl_idx,
                max_turns=args.max_turns,
            )
            rec["forest_snapshot_sha16"] = f_sha
            rec["max_turns"] = args.max_turns
            prov = verify_decision_provenance_chain(records)
            assert prov["valid"], f"Invalid DAG: {prov}"
            # Strip bulky pilot_rows_snapshot from disk JSON while keeping in-memory copy for Wave-1 consolidation
            rec_disk = {k: v for k, v in rec.items() if k != "pilot_rows_snapshot"}
            rec_p.write_text(json.dumps(rec_disk, indent=2, ensure_ascii=False), encoding="utf-8")
            dag_p.write_text(
                json.dumps({"provenance_verification": prov, "records": records}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"  [DONE] {ds.wave} | {model:11s} | {arm:32s} | {ds.task_id:42s} -> "
                f"pass={str(rec['strict_pass']):5s} abi={str(rec.get('abi_mode'))} warm={rec.get('warmstart_candidates_evaluated')}",
                flush=True,
            )
            return rec
        except SubscriptionQuotaPausedError as exc:
            with pause_lock:
                quota_paused[model] = str(exc)
            print(f"  [QUOTA PAUSED] {model}: {exc}", flush=True)
            return None

    # --- Wave 1 ---
    w1_jobs = [(m, arm, ds) for ds in waves["wave_1"] for m in models for arm in arms]
    print(f"=== Starting M6.3 Sovereign Causal AST WAVE 1 ({len(w1_jobs)} episodes, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_exec_one, m, a, d) for (m, a, d) in w1_jobs]):
            r = fut.result()
            if r is not None:
                all_receipts.append(r)

    # --- Order-Invariant Sovereign Consolidation + Re-Indexing (`LearnedFlat` & `M6.3`) ---
    writeback_ledger: dict[str, dict[str, Any]] = {}
    for m in models:
        writeback_ledger[m] = {}
        for arm in arms:
            w1_sub = [r for r in all_receipts if r["model"] == m and r["arm"] == arm and r["wave"] == "wave_1"]
            f_obj, f_arch, _, _ = state_by_ma[(m, arm)]
            wb_res = consolidate_wave1_sovereign_m6_3(
                wave1_receipts=w1_sub,
                arm=arm,
                forest=f_obj,
                node_archive=f_arch,
            )
            # Rebuild both indices so newly grown composite nodes & updated contracts are live in Wave 2!
            state_by_ma[(m, arm)] = (f_obj, f_arch, HippoRAGWormholeIndex(f_obj), FlatEqualMetadataIndex(f_obj))
            writeback_ledger[m][arm] = wb_res
        (ev_dir / f"sovereign_wave1_consolidation_{m}.json").write_text(
            json.dumps(writeback_ledger[m], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Wave 2 ---
    w2_jobs = [(m, arm, ds) for ds in waves["wave_2"] for m in models for arm in arms]
    print(f"=== Starting M6.3 Sovereign Causal AST WAVE 2 ({len(w2_jobs)} episodes, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_exec_one, m, a, d) for (m, a, d) in w2_jobs]):
            r = fut.result()
            if r is not None:
                all_receipts.append(r)

    # Save JSONL receipt index
    clean_recs = [{k: v for k, v in r.items() if k != "pilot_rows_snapshot"} for r in all_receipts]
    (ev_dir / "all_episode_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in clean_recs) + "\n",
        encoding="utf-8",
    )

    summary_by_model: dict[str, dict[str, Any]] = {}
    for m in models:
        summary_by_model[m] = {}
        for arm in arms:
            sub = [r for r in all_receipts if r["model"] == m and r["arm"] == arm]
            in_scope = [r for r in sub if not r["is_out_of_scope"]]
            oos = [r for r in sub if r["is_out_of_scope"]]
            w1_in = [r for r in in_scope if r["wave"] == "wave_1"]
            w2_in = [r for r in in_scope if r["wave"] == "wave_2"]
            wb_c = float(writeback_ledger[m][arm]["writeback_cost_C"])
            ep_cost_sum = sum(float(r["composite_cost"]) for r in sub)
            summary_by_model[m][arm] = {
                "episode_count": len(sub),
                "in_scope_pass": f"{sum(1 for r in in_scope if r['strict_pass'])}/{len(in_scope)}",
                "in_scope_pass_rate": round(sum(1 for r in in_scope if r["strict_pass"]) / max(1, len(in_scope)), 4),
                "wave_1_in_scope_pass": f"{sum(1 for r in w1_in if r['strict_pass'])}/{len(w1_in)}",
                "wave_2_in_scope_pass": f"{sum(1 for r in w2_in if r['strict_pass'])}/{len(w2_in)}",
                "oos_abstain_pass": f"{sum(1 for r in oos if r['strict_pass'])}/{len(oos)}",
                "false_confirmations": sum(1 for r in sub if r["false_confirmation"]),
                "provider_errors": sum(int(r.get("provider_errors", 0)) for r in sub),
                "equal_unit_row_cost": EQUAL_UNIT_ROW_COST,
                "episode_composite_cost_C": round(ep_cost_sum, 3),
                "writeback_cost_C": round(wb_c, 3),
                "total_billed_cost_including_writeback_C": round(ep_cost_sum + wb_c, 3),
                "total_tokens": sum(int(r["total_tokens"]) for r in sub),
            }

    manifest = {
        "study_id": "m6_3_sovereign_ast_causal_96ep_20261121",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "equal_unit_row_cost": EQUAL_UNIT_ROW_COST,
        "completed_episode_count": len(all_receipts),
        "expected_episode_count": len(w1_jobs) + len(w2_jobs),
        "quota_paused_models": quota_paused,
        "wall_sec": round(monotonic() - t0, 2),
        "summary_by_model": summary_by_model,
    }
    (ev_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_files = _write_sha256sums(ev_dir)
    print(f"=== Finished M6.3 Sovereign AST Study: {len(all_receipts)}/{len(w1_jobs)+len(w2_jobs)} eps, {n_files} SHA-256 files ===", flush=True)
    print(json.dumps(summary_by_model, indent=2, ensure_ascii=False), flush=True)
    return 0 if not quota_paused else 3


if __name__ == "__main__":
    raise SystemExit(main())
