"""L4a prospective unseen task-stream manifest.

L4 needs a run that is frozen before outcomes are observed.  This artifact
builds the prospective manifest and baseline protocol without storing prompts,
reference answers, or judge text.  It is a protocol/manifest readiness check,
not a completed external benchmark result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .full_v3_blinded_recursive_live_line import DEFAULT_EXISTING_SAMPLES, DEFAULT_PROBLEM_DIR
from .paper_frozen_main_experiment_v2 import BASELINE_ARMS


DEFAULT_OUT = PAPER_DIR / "l4_prospective_task_stream_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/l4_prospective_task_stream_20260613.md")

L4_BASELINES = [
    "raw_llm_baseline",
    "ordinary_rag",
    "kg_triple_retrieval",
    "embedding_retrieval",
    "graph_memory_only",
    "no_simulator",
    "no_conservative_gate",
    "local_patch_generator",
    "raw_wisdom_generator",
    "no_formal_gate",
    "no_autonomy_writeback",
    "full_l4a_framework_evolution_agent",
]

TASK_STREAM_PROTOCOL = {
    "l4_mini_task_count": 100,
    "paper_main_task_count": 500,
    "external_target_task_count": 1000,
    "minimum_domain_count": 6,
    "human_expert_subset_count": 50,
    "bootstrap_samples": 4000,
    "statistics": [
        "problem_level_bootstrap_ci",
        "paired_problem_test",
        "domain_breakdown",
        "seed_variance",
        "manual_override_rate",
        "framework_growth_score",
    ],
}


def build_l4_prospective_task_stream_payload(
    *,
    root: Path,
    eval_id: str = "l4_prospective_task_stream_20260613",
    l4_mini_task_count: int = TASK_STREAM_PROTOCOL["l4_mini_task_count"],
) -> dict[str, Any]:
    root = root.resolve()
    rows = _load_problem_rows(root)
    excluded = _load_existing_problem_ids(root)
    prospective_pool = [row for row in rows if row["problem_id"] not in excluded]
    selected_pool = prospective_pool or rows
    manifest_rows = _select_manifest_rows(selected_pool, count=l4_mini_task_count)
    manifest = _manifest(manifest_rows)
    protocol = dict(TASK_STREAM_PROTOCOL)
    protocol["l4_mini_task_count"] = l4_mini_task_count
    protocol["baseline_arms"] = L4_BASELINES
    protocol["legacy_baseline_arms"] = list(BASELINE_ARMS)
    metrics = _metrics(
        rows=rows,
        prospective_pool=selected_pool,
        manifest=manifest,
        excluded=excluded,
        protocol=protocol,
    )
    gates = {
        "problem_pool_large": metrics["total_problem_count"] >= 1000,
        "l4_mini_manifest_sufficient": metrics["manifest_task_count"] >= 100,
        "domain_coverage_sufficient": metrics["manifest_domain_count"] >= 6,
        "baseline_suite_hard": metrics["baseline_count"] >= 10,
        "redacted_manifest": metrics["raw_prompt_or_answer_exposed"] is False,
        "manifest_hash_locked": bool(metrics["manifest_hash"]),
        "disjoint_from_existing_samples_when_possible": metrics["disjoint_from_existing_samples"] is True,
        "human_expert_subset_defined": metrics["human_expert_subset_count"] >= 50,
        "ci_plan_problem_level": metrics["bootstrap_samples"] >= 4000,
        "pre_registered_before_outcomes": metrics["outcome_field_count"] == 0,
        "completed_external_result_not_overclaimed": metrics["completed_external_benchmark_claim_allowed"] is False,
        "prospective_manifest_claim_allowed": metrics["prospective_manifest_claim_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "l4_prospective_task_stream",
        "source_md": "reconstruction/md/L4_roadmap.md",
        "l4_stage": "L4-2_prospective_unseen_task_stream",
        "implementation_level": "prospective_manifest_and_protocol_not_completed_external_run",
        "performance_validation": True,
        "validation_scope": (
            "Freezes a prospective task-stream manifest and hard baseline protocol for L4a.  The manifest keeps "
            "only task ids, domains, difficulty labels, and hashes; prompts, reference answers, judge text, and "
            "API material are excluded.  It does not claim the external prospective run has already completed."
        ),
        "protocol": protocol,
        "manifest": manifest,
        "redaction_policy": {
            "store_problem_text": False,
            "store_reference_answer": False,
            "store_prompt": False,
            "store_judge_text": False,
            "store_api_secret": False,
            "retain_problem_id_domain_difficulty_hash": True,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "L4a prospective task-stream manifest and baseline protocol are frozen",
        "blocked_claims": [
            "completed_external_prospective_benchmark",
            "downstream_superiority_on_unrun_l4_stream",
            "human_expert_labels_completed",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# L4 Prospective Task Stream",
        "",
        f"- pass: `{payload['pass']}`",
        f"- manifest tasks: `{m['manifest_task_count']}`",
        f"- manifest domains: `{m['manifest_domain_count']}`",
        f"- baselines: `{m['baseline_count']}`",
        f"- raw prompt/reference exposed: `{m['raw_prompt_or_answer_exposed']}`",
        f"- manifest hash: `{m['manifest_hash']}`",
        f"- completed result claim: `{m['completed_external_benchmark_claim_allowed']}`",
        "",
        "## Claim Boundary",
        "",
        "The task stream is frozen and redacted. A completed external benchmark requires a later execute artifact.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _load_problem_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / DEFAULT_PROBLEM_DIR).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            problem_id = str(row.get("problem_id") or "")
            domain = str(row.get("domain") or "unknown")
            difficulty = str(row.get("difficulty") or "unknown")
            rows.append({
                "problem_id": problem_id,
                "domain": domain,
                "difficulty": difficulty,
                "problem_hash": stable_hash([problem_id, domain, difficulty]),
            })
    return rows


def _load_existing_problem_ids(root: Path) -> set[str]:
    out: set[str] = set()
    for rel in DEFAULT_EXISTING_SAMPLES:
        path = root / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    out.add(item)
                elif isinstance(item, dict) and item.get("problem_id"):
                    out.add(str(item["problem_id"]))
    return out


def _select_manifest_rows(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item["domain"], item["difficulty"], item["problem_hash"])):
        by_domain.setdefault(row["domain"], []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(count, len(rows)):
        progressed = False
        for domain in sorted(by_domain):
            bucket = by_domain[domain]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


def _manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        manifest_rows.append({
            "stream_index": index,
            "problem_id": row["problem_id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "problem_hash": row["problem_hash"],
            "assignment_hash": stable_hash(["l4_task_stream", index, row["problem_hash"]]),
        })
    return {
        "manifest_id": "l4_prospective_task_stream_20260613",
        "rows": manifest_rows,
        "domain_counts": dict(Counter(row["domain"] for row in manifest_rows)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in manifest_rows)),
        "manifest_hash": stable_hash(manifest_rows),
        "fields_retained": ["problem_id", "domain", "difficulty", "problem_hash", "assignment_hash"],
        "fields_excluded": ["description", "reference_answer", "prompt", "judge_text", "api_secret", "outcome"],
        "raw_prompt_or_answer_exposed": False,
    }


def _metrics(
    *,
    rows: list[dict[str, Any]],
    prospective_pool: list[dict[str, Any]],
    manifest: dict[str, Any],
    excluded: set[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    outcome_fields = {"outcome", "answer", "judgment", "score", "utility"}
    manifest_rows = manifest["rows"]
    outcome_field_count = sum(
        1 for row in manifest_rows for key in row if key.lower() in outcome_fields
    )
    return {
        "total_problem_count": len(rows),
        "available_prospective_problem_count": len(prospective_pool),
        "excluded_existing_problem_count": len(excluded),
        "disjoint_from_existing_samples": bool(prospective_pool),
        "manifest_task_count": len(manifest_rows),
        "manifest_domain_count": len(manifest["domain_counts"]),
        "manifest_difficulty_count": len(manifest["difficulty_counts"]),
        "manifest_hash": manifest["manifest_hash"],
        "raw_prompt_or_answer_exposed": manifest["raw_prompt_or_answer_exposed"],
        "baseline_count": len(protocol["baseline_arms"]),
        "legacy_baseline_count": len(protocol["legacy_baseline_arms"]),
        "human_expert_subset_count": protocol["human_expert_subset_count"],
        "bootstrap_samples": protocol["bootstrap_samples"],
        "outcome_field_count": outcome_field_count,
        "prospective_manifest_claim_allowed": True,
        "completed_external_benchmark_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L4 prospective task stream artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="l4_prospective_task_stream_20260613")
    parser.add_argument("--l4-mini-task-count", type=int, default=TASK_STREAM_PROTOCOL["l4_mini_task_count"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_l4_prospective_task_stream_payload(
        root=root,
        eval_id=args.eval_id,
        l4_mini_task_count=args.l4_mini_task_count,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
