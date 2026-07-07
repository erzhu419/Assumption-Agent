"""Promotion report for HLE fast-policy candidates.

This report keeps fair controls and failure mining separate:

* triad rows compare agent against raw/HippoRAG on the same unseen problems;
* agent-only transition rows feed the fast-policy miner;
* selector simulations estimate whether a fallback policy would have helped.

It is redacted by construction: raw questions, raw answers, and option text are
not copied into the report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .autonomy_journal import stable_hash
from .hle_fast_policy_miner import mine_fast_policy_hypotheses
from .hle_transition_dataset import (
    TRANSITION_DATASET_VERSION,
    build_transition_dataset,
    load_hle_result_rows_from_path,
)


PROMOTION_REPORT_VERSION = "hle_fast_policy_promotion_report_v1"
AGENT_VARIANT = "assumption_agent_recursive_verify"
CONTROL_VARIANTS = ("raw", "hipporag_baseline")
TRIAD_VARIANTS = (*CONTROL_VARIANTS, AGENT_VARIANT)


def build_hle_fast_policy_promotion_report(
    *,
    paths: Sequence[str | Path],
    eval_id: str = "hle_fast_policy_promotion_report",
    min_unseen_triads: int = 24,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    source_audits: list[dict[str, Any]] = []
    for path in paths:
        source_path = Path(path).expanduser()
        rows.extend(load_hle_result_rows_from_path(source_path))
        source_hashes.append(stable_hash({"source_path": str(source_path)}))
        source_audits.append(_source_audit_from_path(source_path))
    return build_hle_fast_policy_promotion_report_from_rows(
        rows,
        eval_id=eval_id,
        source_hashes=source_hashes,
        source_audits=source_audits,
        min_unseen_triads=min_unseen_triads,
    )


def build_hle_fast_policy_promotion_report_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    eval_id: str = "hle_fast_policy_promotion_report",
    source_hashes: Sequence[str] | None = None,
    source_audits: Sequence[dict[str, Any]] | None = None,
    min_unseen_triads: int = 24,
) -> dict[str, Any]:
    redacted_rows = [dict(row) for row in rows if isinstance(row, dict)]
    triads = _complete_triads(redacted_rows)
    triad_metrics = _triad_metrics(triads)
    selector_simulation = _selector_policy_simulation(triads)
    agent_rows = [
        row
        for row in redacted_rows
        if str(row.get("variant") or "") == AGENT_VARIANT
    ]
    agent_transition_dataset = build_transition_dataset(agent_rows)
    agent_mining = mine_fast_policy_hypotheses(agent_transition_dataset)
    blockers = _promotion_blockers(
        triad_metrics=triad_metrics,
        selector_simulation=selector_simulation,
        agent_transition_summary=agent_transition_dataset["summary"],
        agent_mining=agent_mining,
        source_audits=list(source_audits or []),
        min_unseen_triads=min_unseen_triads,
    )
    recommendation = (
        "eligible_for_shadow_promotion_review"
        if not blockers
        else "do_not_promote_collect_more_unseen_or_fix_blockers"
    )
    return {
        "eval_id": eval_id,
        "report_version": PROMOTION_REPORT_VERSION,
        "transition_dataset_version": TRANSITION_DATASET_VERSION,
        "source_artifact_hashes": list(source_hashes or []),
        "source_audits": list(source_audits or []),
        "row_count": len(redacted_rows),
        "triad_metrics": triad_metrics,
        "selector_policy_simulation": selector_simulation,
        "agent_transition_summary": agent_transition_dataset["summary"],
        "agent_mined_policy_summary": {
            "hypothesis_count": len(agent_mining.get("hypotheses") or []),
            "hypothesis_actions": [row.get("action") for row in agent_mining.get("hypotheses") or []],
            "promotion_gate_blockers": list(agent_mining.get("promotion_gate_blockers") or []),
            "summary": agent_mining.get("summary") or {},
        },
        "promotion_gate": {
            "passed": not blockers,
            "blockers": blockers,
            "min_unseen_triads": min_unseen_triads,
            "policy": (
                "Promote only if agent is not below best same-model control, enough unseen triads exist, "
                "agent no_fallback count is zero, and selector/policy simulations show a nontrivial gain "
                "or stability improvement without raw-content leakage."
            ),
        },
        "recommendation": recommendation,
        "raw_content_persisted": False,
    }


def format_hle_fast_policy_promotion_markdown(payload: dict[str, Any]) -> str:
    triad = payload["triad_metrics"]
    gate = payload["promotion_gate"]
    selector = payload["selector_policy_simulation"]
    lines = [
        "# HLE Fast Policy Promotion Report",
        "",
        f"- pass: `{gate['passed']}`",
        f"- recommendation: `{payload['recommendation']}`",
        f"- row count: `{payload['row_count']}`",
        f"- complete triads: `{triad['complete_triad_count']}`",
        f"- promotion blockers: `{gate['blockers']}`",
        "",
        "## Source Audit",
        "",
        "| fresh exclusion | pollution pass | paper clean | planned shards | excluded old problems | duplicate sample hashes | top-level errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for audit in payload.get("source_audits") or []:
        lines.append(
            f"| `{audit.get('fresh_exclusion_verified')}` | `{audit.get('pollution_pass')}` | "
            f"`{audit.get('paper_clean_pass')}` | `{audit.get('planned_shard_count')}` | "
            f"`{audit.get('excluded_existing_problem_count')}` | "
            f"`{audit.get('duplicate_sample_problem_hash_count')}` | "
            f"`{audit.get('top_level_error_count')}` |"
        )
    lines.extend([
        "",
        "## Triad Accuracy",
        "",
        "| model | variant | n | correct | accuracy | errors |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for model, model_row in sorted(triad["by_model"].items()):
        for variant, row in sorted(model_row["by_variant"].items()):
            lines.append(
                f"| `{model}` | `{variant}` | `{row['n']}` | `{row['correct']}` | "
                f"`{row['accuracy']}` | `{row['error_count']}` |"
            )
    lines.extend([
        "",
        "## Agent Vs Controls",
        "",
        "| model | best control | agent acc | control acc | margin | passed |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for model, row in sorted(triad["agent_vs_best_control"].items()):
        lines.append(
            f"| `{model}` | `{row['best_control_variant']}` | `{row['agent_accuracy']}` | "
            f"`{row['best_control_accuracy']}` | `{row['agent_minus_best_control']}` | `{row['passed']}` |"
        )
    lines.extend([
        "",
        "## Selector Simulation",
        "",
        "| policy | correct | accuracy | delta vs agent |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, row in sorted(selector["policy_table"].items()):
        lines.append(
            f"| `{name}` | `{row['correct']}` | `{row['accuracy']}` | `{row['delta_vs_agent_current']}` |"
        )
    lines.extend([
        "",
        "## Agent Failure Mining",
        "",
        f"- hypotheses: `{payload['agent_mined_policy_summary']['hypothesis_actions']}`",
        f"- miner blockers: `{payload['agent_mined_policy_summary']['promotion_gate_blockers']}`",
        "",
        "Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.",
    ])
    return "\n".join(lines) + "\n"


def _complete_triads(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = str(row.get("model") or "")
        problem_id = str(row.get("problem_id_hash") or "")
        variant = str(row.get("variant") or "")
        if model and problem_id and variant in TRIAD_VARIANTS:
            grouped[(model, problem_id)][variant] = row
    return {
        key: variants
        for key, variants in grouped.items()
        if set(TRIAD_VARIANTS) <= set(variants)
    }


def _triad_metrics(triads: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_model_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    outcome_buckets: Counter[str] = Counter()
    for (model, _problem_id), variants in triads.items():
        for variant in TRIAD_VARIANTS:
            by_model_rows[model][variant].append(variants[variant])
        raw_correct = bool(variants["raw"].get("correct"))
        hippo_correct = bool(variants["hipporag_baseline"].get("correct"))
        agent_correct = bool(variants[AGENT_VARIANT].get("correct"))
        outcome_buckets[f"raw{int(raw_correct)}_hippo{int(hippo_correct)}_agent{int(agent_correct)}"] += 1

    by_model: dict[str, Any] = {}
    agent_vs_best: dict[str, Any] = {}
    for model, by_variant_rows in sorted(by_model_rows.items()):
        variant_metrics = {
            variant: _variant_metrics(rows)
            for variant, rows in sorted(by_variant_rows.items())
        }
        by_model[model] = {
            "complete_triad_count": len(next(iter(by_variant_rows.values()), [])),
            "by_variant": variant_metrics,
        }
        agent = variant_metrics.get(AGENT_VARIANT, {})
        controls = [
            {"variant": variant, **variant_metrics[variant]}
            for variant in CONTROL_VARIANTS
            if variant in variant_metrics
        ]
        if controls and agent:
            best_control = max(controls, key=lambda row: float(row.get("accuracy") or 0.0))
            margin = round(float(agent.get("accuracy") or 0.0) - float(best_control.get("accuracy") or 0.0), 4)
            agent_vs_best[model] = {
                "passed": margin >= 0.0,
                "agent_accuracy": agent.get("accuracy"),
                "best_control_variant": best_control.get("variant"),
                "best_control_accuracy": best_control.get("accuracy"),
                "agent_minus_best_control": margin,
                "controls": controls,
            }
    return {
        "complete_triad_count": len(triads),
        "outcome_buckets": dict(sorted(outcome_buckets.items())),
        "by_model": by_model,
        "agent_vs_best_control": agent_vs_best,
        "raw_content_persisted": False,
    }


def _variant_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct") is True)
    error_count = sum(1 for row in rows if row.get("error"))
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "error_count": error_count,
    }


def _selector_policy_simulation(
    triads: dict[tuple[str, str], dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    policy_scores: Counter[str] = Counter()
    regression_adjustments: Counter[str] = Counter()
    for (_model, _problem_id), variants in triads.items():
        raw = variants["raw"]
        hippo = variants["hipporag_baseline"]
        agent = variants[AGENT_VARIANT]
        raw_correct = bool(raw.get("correct"))
        hippo_correct = bool(hippo.get("correct"))
        agent_correct = bool(agent.get("correct"))
        gate_status = _verified_gate_status(agent)

        if agent_correct:
            policy_scores["agent_current"] += 1
        if raw_correct:
            policy_scores["always_raw"] += 1
        if hippo_correct:
            policy_scores["always_hipporag"] += 1

        if gate_status == "allowed":
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

        if not agent_correct and (raw_correct or hippo_correct):
            regression_adjustments[_recommended_adjustment(raw_correct, hippo_correct, agent)] += 1

    triad_count = len(triads)
    agent_current = int(policy_scores.get("agent_current", 0))
    policy_table = {
        name: {
            "correct": int(correct),
            "accuracy": round(int(correct) / triad_count, 4) if triad_count else None,
            "delta_vs_agent_current": round((int(correct) - agent_current) / triad_count, 4)
            if triad_count
            else None,
        }
        for name, correct in sorted(policy_scores.items())
    }
    best_policy = None
    if policy_table:
        best_policy = max(
            policy_table.items(),
            key=lambda item: (item[1]["correct"], item[1]["delta_vs_agent_current"] or 0.0, item[0]),
        )[0]
    best_delta = (
        policy_table.get(best_policy, {}).get("delta_vs_agent_current")
        if best_policy is not None
        else None
    )
    return {
        "complete_triad_count": triad_count,
        "policy_table": policy_table,
        "best_policy": best_policy,
        "best_delta_vs_agent_current": best_delta,
        "recommended_adjustments": dict(regression_adjustments.most_common()),
        "raw_content_persisted": False,
    }


def _promotion_blockers(
    *,
    triad_metrics: dict[str, Any],
    selector_simulation: dict[str, Any],
    agent_transition_summary: dict[str, Any],
    agent_mining: dict[str, Any],
    source_audits: list[dict[str, Any]],
    min_unseen_triads: int,
) -> list[str]:
    blockers: list[str] = []
    if triad_metrics.get("complete_triad_count", 0) < min_unseen_triads:
        blockers.append(f"insufficient_unseen_triads_min_{min_unseen_triads}")
    if not triad_metrics.get("agent_vs_best_control"):
        blockers.append("missing_complete_raw_hipporag_agent_triads")
    for model, row in sorted((triad_metrics.get("agent_vs_best_control") or {}).items()):
        if not row.get("passed"):
            blockers.append(f"agent_below_best_control:{model}")
    if int(agent_transition_summary.get("no_fallback_count") or 0) > 0:
        blockers.append("agent_no_fallback_present")
    if _control_error_count(triad_metrics) > 0:
        blockers.append("control_or_agent_error_rows_present")
    for audit in source_audits:
        if audit and audit.get("fresh_exclusion_verified") is False:
            blockers.append("fresh_exclusion_not_verified")
    best_delta = selector_simulation.get("best_delta_vs_agent_current")
    if best_delta is None or float(best_delta) <= 0.0:
        blockers.append("no_selector_policy_gain")
    for blocker in agent_mining.get("promotion_gate_blockers") or []:
        normalized = _normalize_agent_miner_blocker(str(blocker))
        if normalized and normalized not in blockers:
            blockers.append(normalized)
    return blockers


def _source_audit_from_path(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "source_artifact_hash": stable_hash({"source_path": str(path)}),
            "status": "unreadable",
            "raw_content_persisted": False,
        }
    if not isinstance(payload, dict):
        return {
            "source_artifact_hash": stable_hash({"source_path": str(path)}),
            "status": "not_object",
            "raw_content_persisted": False,
        }
    sampling = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {}
    pollution = payload.get("pollution_audit") if isinstance(payload.get("pollution_audit"), dict) else {}
    fresh = (
        pollution.get("fresh_problem_hash_exclusion")
        if isinstance(pollution.get("fresh_problem_hash_exclusion"), dict)
        else {}
    )
    cache_live = (
        pollution.get("cache_live_separation")
        if isinstance(pollution.get("cache_live_separation"), dict)
        else {}
    )
    exclude_enabled_count = int(fresh.get("exclude_existing_enabled_shard_count") or 0)
    planned_shards = int(sampling.get("planned_shard_count") or 0)
    duplicate_count = int(fresh.get("duplicate_sample_problem_hash_count") or 0)
    fresh_verified = (
        bool(payload.get("pollution_pass"))
        and exclude_enabled_count > 0
        and (planned_shards == 0 or exclude_enabled_count >= planned_shards)
        and duplicate_count == 0
    )
    return {
        "source_artifact_hash": stable_hash({"source_path": str(path)}),
        "eval_id_hash": stable_hash({"eval_id": payload.get("eval_id")}),
        "status": "loaded",
        "fresh_exclusion_verified": fresh_verified,
        "pollution_pass": bool(payload.get("pollution_pass")),
        "paper_clean_pass": bool(payload.get("paper_clean_pass")),
        "paper_clean_failed_gates": list(payload.get("paper_clean_failed_gates") or []),
        "planned_shard_count": planned_shards,
        "requested_total_sample_size": sampling.get("requested_total_sample_size"),
        "exclude_existing_enabled_shard_count": exclude_enabled_count,
        "excluded_existing_problem_count": int(fresh.get("excluded_existing_problem_count") or 0),
        "distinct_sample_problem_hash_count": int(fresh.get("distinct_sample_problem_hash_count") or 0),
        "duplicate_sample_problem_hash_count": duplicate_count,
        "top_level_error_count": int(cache_live.get("top_level_error_count") or 0),
        "raw_content_persisted": False,
    }


def _normalize_agent_miner_blocker(blocker: str) -> str:
    if not blocker:
        return ""
    if blocker.startswith("insufficient_unseen_transition_rows_min_"):
        return ""
    if blocker == "no_fallback_present":
        return ""
    if blocker == "missing_fair_control_or_split_metadata":
        return ""
    return f"agent_miner:{blocker}"


def _control_error_count(triad_metrics: dict[str, Any]) -> int:
    count = 0
    for model_row in (triad_metrics.get("by_model") or {}).values():
        for variant_row in (model_row.get("by_variant") or {}).values():
            count += int(variant_row.get("error_count") or 0)
    return count


def _verified_gate_status(agent: dict[str, Any]) -> str:
    ce = agent.get("component_efficacy") if isinstance(agent.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    gate = selection.get("verified_or_abstain_gate")
    if isinstance(gate, dict) and gate.get("status"):
        return str(gate.get("status"))
    return "unknown"


def _baseline_consensus(raw: dict[str, Any], hippo: dict[str, Any]) -> bool:
    return bool(raw.get("prediction_hash")) and raw.get("prediction_hash") == hippo.get("prediction_hash")


def _recommended_adjustment(raw_correct: bool, hippo_correct: bool, agent: dict[str, Any]) -> str:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="HLE fair-control aggregate/shard JSON artifacts")
    parser.add_argument("--eval-id", default="hle_fast_policy_promotion_report")
    parser.add_argument("--min-unseen-triads", type=int, default=24)
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--md-out", help="Optional output Markdown path")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    payload = build_hle_fast_policy_promotion_report(
        paths=args.paths,
        eval_id=args.eval_id,
        min_unseen_triads=args.min_unseen_triads,
    )
    text = json.dumps(payload, ensure_ascii=True, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.md_out:
        Path(args.md_out).write_text(format_hle_fast_policy_promotion_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
