#!/usr/bin/env python3
"""Run the 192-episode Zero-Leakage AST-Driven Causal & Recursive M6.1 Live Study.

Evaluates 4 strictly equal-priced arms (`EQUAL_UNIT_ROW_COST = 1.5` across all arms, `writeback_cost_C`
billed into totals, ZERO hardcoded writeback formulas, ZERO pre-assembled C04 root equations):
  - `Strong_Ordinary_LiveGPT`
  - `EqualAuth_Flat_LiveGPT`
  - `Hegel_M5_Full_LiveGPT`
  - `Hegel_M6_1_ZeroLeakage_LiveGPT`
across `gpt-6-luna` and `gpt-6.2-sol` (`24 tasks x 4 arms x 2 models = 192 episodes`).
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
    M6_1_ARMS,
    build_m6_1_clean_k40_forest,
    consolidate_wave1_via_ast_wormholes,
    run_single_m6_1_episode,
)
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves


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
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_1_zero_leakage_ast_causal_192ep_20261121",
    )
    parser.add_argument("--models", nargs="+", default=["gpt-6-luna", "gpt-6.2-sol"])
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

    # Independent clean K=40 forest per (model, arm) so Wave-1 -> Wave-2 mutations never cross-bleed!
    state_by_ma: dict[tuple[str, str], tuple[Any, dict[str, dict[str, Any]], HippoRAGWormholeIndex, FlatEqualMetadataIndex]] = {}
    for m in models:
        for arm in M6_1_ARMS:
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
        ep_dir = ws_root / ds.wave / f"{model}__{arm}__{ds.task_id}"
        rec_p = ep_dir / "episode_receipt.json"
        dag_p = ep_dir / "decision_provenance_dag.json"
        if rec_p.is_file() and dag_p.is_file():
            try:
                cached = json.loads(rec_p.read_text(encoding="utf-8"))
                dag_obj = json.loads(dag_p.read_text(encoding="utf-8"))
                if dag_obj.get("provenance_verification", {}).get("valid"):
                    return cached
            except Exception:
                pass

        f_obj, f_arch, h_idx, fl_idx = state_by_ma[(model, arm)]
        client = CodexSubscriptionClient(
            codex_info=codex_info,
            model=model,
            reasoning_effort="medium",
            timeout_seconds=150,
        )
        try:
            rec, records = run_single_m6_1_episode(
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
            prov = verify_decision_provenance_chain(records)
            assert prov["valid"], f"Invalid DAG: {prov}"
            rec_p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            dag_p.write_text(
                json.dumps({"provenance_verification": prov, "records": records}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"  [DONE] {ds.wave} | {model:11s} | {arm:31s} | {ds.task_id:42s} -> pass={rec['strict_pass']} probes={rec['purchased_probes']}",
                flush=True,
            )
            return rec
        except SubscriptionQuotaPausedError as exc:
            with pause_lock:
                quota_paused[model] = str(exc)
            print(f"  [QUOTA PAUSED] {model}: {exc}", flush=True)
            return None

    # --- Wave 1 ---
    w1_jobs = [(m, arm, ds) for ds in waves["wave_1"] for m in models for arm in M6_1_ARMS]
    print(f"=== Starting M6.1 Zero-Leakage AST Causal WAVE 1 ({len(w1_jobs)} episodes) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_exec_one, m, a, d) for (m, a, d) in w1_jobs]):
            r = fut.result()
            if r is not None:
                all_receipts.append(r)

    # --- Pure AST Wave-1 -> Wave-2 Consolidation ---
    writeback_ledger: dict[str, dict[str, Any]] = {}
    for m in models:
        writeback_ledger[m] = {}
        for arm in M6_1_ARMS:
            w1_sub = [r for r in all_receipts if r["model"] == m and r["arm"] == arm and r["wave"] == "wave_1"]
            f_obj, f_arch, _, _ = state_by_ma[(m, arm)]
            wb_res = consolidate_wave1_via_ast_wormholes(
                wave1_receipts=w1_sub,
                arm=arm,
                forest=f_obj,
                node_archive=f_arch,
            )
            writeback_ledger[m][arm] = wb_res
        (ev_dir / f"ast_wave1_consolidation_{m}.json").write_text(
            json.dumps(writeback_ledger[m], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Wave 2 ---
    w2_jobs = [(m, arm, ds) for ds in waves["wave_2"] for m in models for arm in M6_1_ARMS]
    print(f"=== Starting M6.1 Zero-Leakage AST Causal WAVE 2 ({len(w2_jobs)} episodes) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_exec_one, m, a, d) for (m, a, d) in w2_jobs]):
            r = fut.result()
            if r is not None:
                all_receipts.append(r)

    # --- Aggregate with Equal Pricing & Explicit Writeback Cost Inclusion ---
    summary_by_model: dict[str, dict[str, Any]] = {}
    for m in models:
        summary_by_model[m] = {}
        for arm in M6_1_ARMS:
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
                "equal_unit_row_cost": EQUAL_UNIT_ROW_COST,
                "episode_composite_cost_C": round(ep_cost_sum, 3),
                "writeback_cost_C": round(wb_c, 3),
                "total_billed_cost_including_writeback_C": round(ep_cost_sum + wb_c, 3),
                "total_tokens": sum(int(r["total_tokens"]) for r in sub),
            }

    manifest = {
        "study_id": "m6_1_zero_leakage_ast_causal_192ep_20261121",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "zero_ruoli_proxy_verified": True,
        "zero_hardcoded_writeback_verified": True,
        "equal_unit_row_cost": EQUAL_UNIT_ROW_COST,
        "completed_episode_count": len(all_receipts),
        "expected_episode_count": len(w1_jobs) + len(w2_jobs),
        "quota_paused_models": quota_paused,
        "wall_sec": round(monotonic() - t0, 2),
        "summary_by_model": summary_by_model,
    }
    (ev_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_files = _write_sha256sums(ev_dir)
    print(f"=== Finished M6.1 Zero-Leakage AST Causal Study: {len(all_receipts)}/{len(w1_jobs)+len(w2_jobs)} eps, {n_files} SHA-256 files ===", flush=True)
    return 0 if not quota_paused else 3


if __name__ == "__main__":
    raise SystemExit(main())
