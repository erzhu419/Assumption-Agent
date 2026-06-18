"""Backprop-style selector diagnostics for HLE triad runs.

This module reads already-redacted HLE artifacts and uses only hashes,
correctness labels, variants, and module metadata.  It does not persist raw
questions, gold answers, rationales, canaries, or prediction text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_SOURCE = PAPER_DIR / "hle_fair_runs" / "hle_fair_same_model_n60_gpt54mini_nohardtimeout_20260618"
DEFAULT_OUT = PAPER_DIR / "hle_selector_backprop_20260618.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_selector_backprop_20260618.md")


def build_hle_selector_backprop_payload(
    *,
    root: Path,
    eval_id: str = "hle_selector_backprop_20260618",
    sources: list[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    source_paths = sources or [DEFAULT_SOURCE]
    rows: list[dict[str, Any]] = []
    loaded_sources: list[dict[str, Any]] = []
    for source in source_paths:
        source_path = source if source.is_absolute() else root / source
        loaded = _load_redacted_rows(source_path)
        rows.extend(loaded)
        loaded_sources.append({
            "source": str(source),
            "row_count": len(loaded),
        })
    triads = _complete_triads(rows)
    outcome_buckets = Counter()
    loss_buckets = Counter()
    module_flags = Counter()
    selection_methods = Counter()
    recommended_adjustments = Counter()
    regression_cases: list[dict[str, Any]] = []
    policy_scores = Counter()

    for pid, variants in sorted(triads.items()):
        raw = variants["raw"]
        hippo = variants["hipporag_baseline"]
        agent = variants["assumption_agent_recursive_verify"]
        raw_correct = bool(raw.get("correct"))
        hippo_correct = bool(hippo.get("correct"))
        agent_correct = bool(agent.get("correct"))
        outcome_buckets[f"raw{int(raw_correct)}_hippo{int(hippo_correct)}_agent{int(agent_correct)}"] += 1
        ce = agent.get("component_efficacy") if isinstance(agent.get("component_efficacy"), dict) else {}
        selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
        method = str(selection.get("selection_method") or "unknown")
        gate_status = _verified_gate_status(agent)
        selection_methods[f"{gate_status}::{method}"] += 1

        if agent_correct:
            policy_scores["agent_current"] += 1
        if raw_correct:
            policy_scores["always_raw"] += 1
        if hippo_correct:
            policy_scores["always_hipporag"] += 1
        if _agent_verified(agent):
            if agent_correct:
                policy_scores["verified_else_raw"] += 1
                policy_scores["verified_else_hipporag"] += 1
        else:
            if raw_correct:
                policy_scores["verified_else_raw"] += 1
            if hippo_correct:
                policy_scores["verified_else_hipporag"] += 1
        if _baseline_consensus(raw, hippo):
            if raw_correct:
                policy_scores["baseline_consensus_else_agent"] += 1
        elif agent_correct:
            policy_scores["baseline_consensus_else_agent"] += 1

        if agent_correct or not (raw_correct or hippo_correct):
            continue

        flags = ce.get("flags") if isinstance(ce.get("flags"), dict) else {}
        for key, value in flags.items():
            if value:
                module_flags[key] += 1
        adjustment = _recommended_adjustment(
            raw_correct=raw_correct,
            hippo_correct=hippo_correct,
            agent=agent,
        )
        recommended_adjustments[adjustment] += 1
        loss_buckets[_loss_bucket(raw_correct=raw_correct, hippo_correct=hippo_correct, agent=agent)] += 1
        if len(regression_cases) < 40:
            regression_cases.append({
                "problem_id_hash": pid,
                "category": agent.get("category"),
                "raw_subject": agent.get("raw_subject"),
                "answer_type": agent.get("answer_type"),
                "raw_correct": raw_correct,
                "hipporag_correct": hippo_correct,
                "agent_correct": agent_correct,
                "agent_selection_method": method,
                "verified_gate_status": gate_status,
                "fallback_prompt_kind": (
                    (selection.get("verified_or_abstain_gate") or {}).get("fallback_prompt_kind")
                    if isinstance(selection.get("verified_or_abstain_gate"), dict)
                    else None
                ),
                "module_flags": sorted(key for key, value in flags.items() if value),
                "recommended_adjustment": adjustment,
            })

    triad_count = len(triads)
    policy_table = {
        name: {
            "correct": int(correct),
            "accuracy": round(int(correct) / triad_count, 4) if triad_count else None,
            "delta_vs_agent_current": (
                round((int(correct) - int(policy_scores.get("agent_current", 0))) / triad_count, 4)
                if triad_count else None
            ),
        }
        for name, correct in sorted(policy_scores.items())
    }
    gates = {
        "redacted_rows_loaded": bool(rows),
        "complete_triad_rows_available": triad_count > 0,
        "hipporag_preserve_policy_beats_current": (
            int(policy_scores.get("verified_else_hipporag", 0))
            > int(policy_scores.get("agent_current", 0))
        ),
        "raw_content_not_persisted": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "hle_selector_backprop",
        "performance_validation": True,
        "validation_scope": (
            "Offline backprop over redacted HLE triad artifacts.  Uses only correctness flags, hashes, "
            "variants, and module metadata to infer selector adjustments."
        ),
        "loaded_sources": loaded_sources,
        "metrics": {
            "row_count": len(rows),
            "complete_triad_count": triad_count,
            "outcome_buckets": dict(sorted(outcome_buckets.items())),
            "agent_regression_count_when_any_control_correct": sum(recommended_adjustments.values()),
            "selection_methods": dict(sorted(selection_methods.items())),
            "loss_buckets": dict(sorted(loss_buckets.items())),
            "module_flags_in_regressions": dict(module_flags.most_common()),
            "recommended_adjustments": dict(recommended_adjustments.most_common()),
            "policy_simulation": policy_table,
        },
        "regression_cases": regression_cases,
        "gates": gates,
        "pass": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _load_redacted_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(path.glob("*_shard_*.json")):
            rows.extend(_rows_from_json(child))
        return rows
    payload_rows = _rows_from_json(path)
    if payload_rows:
        return payload_rows
    sibling_dir = path.with_suffix("")
    if sibling_dir.is_dir():
        return _load_redacted_rows(sibling_dir)
    return []


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("rows") or payload.get("run_rows") or []
    return rows if isinstance(rows, list) else []


def _complete_triads(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_problem: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pid = str(row.get("problem_id_hash") or "")
        variant = str(row.get("variant") or "")
        if pid and variant in {"raw", "hipporag_baseline", "assumption_agent_recursive_verify"}:
            by_problem[pid][variant] = row
    return {
        pid: variants
        for pid, variants in by_problem.items()
        if {"raw", "hipporag_baseline", "assumption_agent_recursive_verify"} <= set(variants)
    }


def _agent_verified(agent: dict[str, Any]) -> bool:
    return _verified_gate_status(agent) == "allowed"


def _verified_gate_status(agent: dict[str, Any]) -> str:
    ce = agent.get("component_efficacy") if isinstance(agent.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    gate = selection.get("verified_or_abstain_gate")
    if isinstance(gate, dict) and gate.get("status"):
        return str(gate.get("status"))
    return "unknown"


def _baseline_consensus(raw: dict[str, Any], hippo: dict[str, Any]) -> bool:
    return bool(raw.get("prediction_hash")) and raw.get("prediction_hash") == hippo.get("prediction_hash")


def _loss_bucket(*, raw_correct: bool, hippo_correct: bool, agent: dict[str, Any]) -> str:
    ce = agent.get("component_efficacy") if isinstance(agent.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    method = str(selection.get("selection_method") or "unknown")
    if hippo_correct and not raw_correct:
        control = "hipporag_only_correct"
    elif raw_correct and not hippo_correct:
        control = "raw_only_correct"
    else:
        control = "raw_and_hipporag_correct"
    return f"{control}::{_verified_gate_status(agent)}::{method}"


def _recommended_adjustment(*, raw_correct: bool, hippo_correct: bool, agent: dict[str, Any]) -> str:
    ce = agent.get("component_efficacy") if isinstance(agent.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    method = str(selection.get("selection_method") or "")
    gate_status = _verified_gate_status(agent)
    if gate_status == "abstained" and hippo_correct:
        return "prefer_hipporag_preserve_selector_for_unverified_mc"
    if gate_status == "abstained" and raw_correct:
        return "prefer_raw_preserve_selector_for_unverified_mc"
    if method == "candidate_claim_verifier_priority" and hippo_correct:
        return "tighten_candidate_claim_verifier_with_baseline_negative_control"
    return "inspect_selector_or_add_answer_bearing_verifier"


def format_hle_selector_backprop_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# HLE Selector Backprop",
        "",
        f"- pass: `{payload['pass']}`",
        f"- row count: `{metrics['row_count']}`",
        f"- complete triads: `{metrics['complete_triad_count']}`",
        f"- agent regressions where any control is correct: `{metrics['agent_regression_count_when_any_control_correct']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        "",
        "## Policy Simulation",
        "",
        "| policy | correct | accuracy | delta vs current |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in sorted(metrics["policy_simulation"].items()):
        lines.append(f"| `{name}` | `{row['correct']}` | `{row['accuracy']}` | `{row['delta_vs_agent_current']}` |")
    lines.extend([
        "",
        "## Recommended Adjustments",
        "",
        "| adjustment | count |",
        "| --- | ---: |",
    ])
    for key, value in metrics["recommended_adjustments"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Loss Buckets",
        "",
        "| bucket | count |",
        "| --- | ---: |",
    ])
    for key, value in metrics["loss_buckets"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend([
        "",
        "Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build redacted HLE selector backprop diagnostics.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_selector_backprop_20260618")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sources = [Path(value) for value in args.source] if args.source else [DEFAULT_SOURCE]
    payload = build_hle_selector_backprop_payload(root=root, eval_id=args.eval_id, sources=sources)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    md_out = Path(args.md_out)
    if not md_out.is_absolute():
        md_out = root / md_out
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(format_hle_selector_backprop_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "md_out": str(md_out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
