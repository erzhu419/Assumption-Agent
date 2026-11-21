"""Forensic Inspector & Verifier for Hegel V18.1 Decision-Level Provenance Logs.

Allows human auditors or external GPT sandboxes to:
  1. Cryptographically verify every `record_sha256` hash chain, `parent_decision_id` causal link,
     and step-by-step cost conservation across all deterministic & Multi-Model Live-GPT episodes (`--verify-all`);
  2. Print the complete causal decision trace tree for any `(model, task_id, arm)` (`--model ... --task ... --arm ...`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m5_live_gpt_and_provenance import verify_decision_provenance_chain


def _load_grouped_ledger(jsonl_path: Path) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    if not jsonl_path.is_file():
        return grouped
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            model_key = str(rec.get("model") or rec.get("run_id") or "default")
            grouped[(model_key, rec["task_id"], rec["arm"])].append(rec)
    for k in grouped:
        grouped[k].sort(key=lambda r: int(r["decision_seq"]))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify & inspect M5 decision provenance DAGs")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "v18_1_m5_live_gpt_and_decision_provenance_20261120",
    )
    parser.add_argument("--verify-all", action="store_true", help="Verify SHA256SUMS and all decision hash chains")
    parser.add_argument("--model", type=str, default="", help="Filter by model slug substring (e.g. gpt-6-luna, gpt-6.1-terra)")
    parser.add_argument("--task", type=str, default="", help="Filter by task_id substring")
    parser.add_argument("--arm", type=str, default="", help="Filter by arm substring")
    args = parser.parse_args()

    ev = args.evidence_dir.resolve()
    det_groups = _load_grouped_ledger(ev / "deterministic_decision_provenance_ledger.jsonl")
    live_groups = _load_grouped_ledger(ev / "live_gpt_decision_provenance_ledger.jsonl")

    if args.verify_all or (not args.model and not args.task and not args.arm):
        sha_file = ev / "SHA256SUMS.txt"
        sha_ok = 0
        sha_total = 0
        if sha_file.is_file():
            for ln in sha_file.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                sha_total += 1
                exp, rel = ln.split(maxsplit=1)
                act = hashlib.sha256((ev / rel.lstrip("*")).read_bytes()).hexdigest()
                if act == exp:
                    sha_ok += 1
        det_valid = sum(1 for recs in det_groups.values() if verify_decision_provenance_chain(recs)["valid"])
        live_valid = sum(1 for recs in live_groups.values() if verify_decision_provenance_chain(recs)["valid"])
        live_backends = sorted({r.get("provider_backend", "unknown") for recs in live_groups.values() for r in recs})
        live_models = sorted({m for (m, _, _) in live_groups.keys()})
        print(json.dumps({
            "sha256_files_verified": f"{sha_ok}/{sha_total}",
            "deterministic_chains_verified": f"{det_valid}/{len(det_groups)}",
            "deterministic_total_decisions": sum(len(v) for v in det_groups.values()),
            "live_subscription_chains_verified": f"{live_valid}/{len(live_groups)}",
            "live_subscription_total_decisions": sum(len(v) for v in live_groups.values()),
            "live_subscription_models": live_models,
            "live_provider_backends": live_backends,
        }, indent=2, ensure_ascii=False))

    if args.model or args.task or args.arm:
        combined = {
            **{("det", *k): v for k, v in det_groups.items()},
            **{("live_sub", *k): v for k, v in live_groups.items()},
        }
        for (src_kind, m_name, tid, arm_n), recs in sorted(combined.items()):
            if args.model and args.model not in m_name:
                continue
            if args.task and args.task not in tid:
                continue
            if args.arm and args.arm not in arm_n:
                continue
            print(f"\n=== [{src_kind} | model={m_name}] {tid} | {arm_n} ({len(recs)} atomic decisions) ===")
            for r in recs:
                d = r["cost_ledger_delta"]
                print(
                    f"  [{r['decision_seq']:02d}] {r['decision_id']} <- {r['parent_decision_id']}\n"
                    f"       type:   {r['decision_type']} (backend={r.get('provider_backend')})\n"
                    f"       chosen: {json.dumps(r['chosen_option'], ensure_ascii=False)}\n"
                    f"       reject: {json.dumps(r['rejection_reasons'], ensure_ascii=False)}\n"
                    f"       why:    {r['rationale_summary']}\n"
                    f"       cost:   +M={d['measurement_delta']}, +K={d['compute_delta']}, "
                    f"tokens={d['prompt_tokens_delta']}(cached:{d.get('cached_tokens_delta', 0)})+{d['completion_tokens_delta']}, "
                    f"sha={r['record_sha256'][:12]}"
                )


if __name__ == "__main__":
    main()
