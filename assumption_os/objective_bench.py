"""External objective-task benchmark for V5 verifier gates.

The fresh-ablation acceptance gate is scoped to trigger/control rows.  V5 needs
an additional objective-task layer so a candidate cannot be promoted only
because it won the rows that activated it.  This module summarizes independent
task results by proposal id and exposes a compact pass/fail signal to
``verifier_stack``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


INTERNAL_LABEL_SOURCES = {
    "internal_trigger_control",
    "candidate_acceptance",
    "trigger_control_acceptance",
}


def build_objective_benchmark_payload(
    *,
    proposal_payload: dict,
    acceptance_payload: dict | None,
    eval_id: str,
    task_results: Iterable[dict] | None = None,
    min_task_count: int = 2,
    min_family_count: int = 2,
    min_mean_delta: float = 0.15,
    max_loss_rate: float = 0.25,
) -> dict:
    """Summarize external objective-task results by proposal.

    ``task_results`` rows are intentionally simple so callers can feed model
    judgments, deterministic objective scores, or cached benchmark outcomes:

    ``proposal_id``, ``task_id``, ``task_family``, ``candidate_score``, and
    ``baseline_score`` are the core fields.  ``label_source`` should be external
    to trigger/control acceptance for a row to count toward V5.
    """

    rows = [_normalize_task(row) for row in task_results or []]
    rows_by_proposal: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_proposal.setdefault(row["proposal_id"], []).append(row)
    acceptance_by_id = {
        row.get("proposal_id"): row
        for row in (acceptance_payload or {}).get("summaries", [])
        if row.get("proposal_id")
    }
    proposal_ids = _proposal_ids(proposal_payload, acceptance_by_id, rows_by_proposal)
    summaries = [
        _summarize_proposal(
            proposal_id=proposal_id,
            acceptance=acceptance_by_id.get(proposal_id, {}),
            rows=rows_by_proposal.get(proposal_id, []),
            min_task_count=min_task_count,
            min_family_count=min_family_count,
            min_mean_delta=min_mean_delta,
            max_loss_rate=max_loss_rate,
        )
        for proposal_id in proposal_ids
    ]
    accepted = [row for row in summaries if row["acceptance_decision"] == "accept"]
    rejected = [row for row in summaries if row["acceptance_decision"] in {"reject_benefit", "reject_harm"}]
    external_accepted_pass = all(row["objective_gate_passed"] for row in accepted) if accepted else True
    external_rejected_covered = all(row["external_task_count"] >= 1 for row in rejected) if rejected else True
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_acceptance_eval_id": (acceptance_payload or {}).get("eval_id"),
        "thresholds": {
            "min_task_count": min_task_count,
            "min_family_count": min_family_count,
            "min_mean_delta": min_mean_delta,
            "max_loss_rate": max_loss_rate,
        },
        "proposal_count": len(summaries),
        "task_count": len(rows),
        "external_task_count": sum(row["external_task_count"] for row in summaries),
        "decision_counts": dict(Counter(row["objective_decision"] for row in summaries)),
        "accepted_external_pass_count": sum(
            1 for row in accepted if row["objective_gate_passed"]
        ),
        "accepted_count": len(accepted),
        "rejected_external_covered_count": sum(
            1 for row in rejected if row["external_task_count"] >= 1
        ),
        "rejected_count": len(rejected),
        "external_accepted_pass": external_accepted_pass,
        "external_rejected_covered": external_rejected_covered,
        "pass": external_accepted_pass and external_rejected_covered,
        "summaries": summaries,
    }


def _proposal_ids(
    proposal_payload: dict,
    acceptance_by_id: dict[str, dict],
    rows_by_proposal: dict[str, list[dict]],
) -> list[str]:
    ids = [
        proposal.get("proposal_id")
        for proposal in proposal_payload.get("proposals", [])
        if proposal.get("proposal_id")
    ]
    ids.extend(pid for pid in acceptance_by_id if pid)
    ids.extend(pid for pid in rows_by_proposal if pid)
    return _dedupe(ids)


def _summarize_proposal(
    *,
    proposal_id: str,
    acceptance: dict,
    rows: list[dict],
    min_task_count: int,
    min_family_count: int,
    min_mean_delta: float,
    max_loss_rate: float,
) -> dict:
    external_rows = [
        row for row in rows
        if row["label_source"] not in INTERNAL_LABEL_SOURCES
        and not row.get("uses_trigger_control_rows", False)
    ]
    outcomes = Counter(row["outcome"] for row in external_rows)
    deltas = [float(row["score_delta"]) for row in external_rows]
    mean_delta = round(sum(deltas) / len(deltas), 4) if deltas else None
    loss_rate = round(outcomes.get("loss", 0) / len(external_rows), 4) if external_rows else None
    family_count = len({row["task_family"] for row in external_rows if row["task_family"]})
    objective_gate_passed = (
        len(external_rows) >= min_task_count
        and family_count >= min_family_count
        and mean_delta is not None
        and mean_delta >= min_mean_delta
        and (loss_rate is not None and loss_rate <= max_loss_rate)
    )
    if not external_rows:
        decision = "missing"
        rationale = "No external objective-task rows were available."
    elif objective_gate_passed:
        decision = "pass"
        rationale = "External objective tasks passed minimum scope, benefit, and loss gates."
    else:
        decision = "fail"
        rationale = "External objective tasks did not satisfy the V5 benchmark gate."
    return {
        "proposal_id": proposal_id,
        "acceptance_decision": acceptance.get("decision"),
        "objective_decision": decision,
        "objective_gate_passed": objective_gate_passed,
        "task_count": len(rows),
        "external_task_count": len(external_rows),
        "family_count": family_count,
        "outcomes": dict(outcomes),
        "mean_score_delta": mean_delta,
        "loss_rate": loss_rate,
        "task_ids": [row["task_id"] for row in external_rows],
        "families": sorted({row["task_family"] for row in external_rows if row["task_family"]}),
        "label_sources": sorted({row["label_source"] for row in external_rows}),
        "rationale": rationale,
    }


def _normalize_task(row: dict) -> dict:
    candidate_score = float(row.get("candidate_score", 0.0) or 0.0)
    baseline_score = float(row.get("baseline_score", 0.0) or 0.0)
    delta = candidate_score - baseline_score
    outcome = row.get("outcome")
    if outcome not in {"win", "loss", "tie"}:
        if delta > 0:
            outcome = "win"
        elif delta < 0:
            outcome = "loss"
        else:
            outcome = "tie"
    return {
        "proposal_id": str(row.get("proposal_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "task_family": str(row.get("task_family") or "external_objective"),
        "label_source": str(row.get("label_source") or "external_objective_task"),
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "score_delta": round(delta, 4),
        "outcome": outcome,
        "uses_trigger_control_rows": bool(row.get("uses_trigger_control_rows", False)),
        "metadata": row.get("metadata", {}),
    }


def _dedupe(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_json(path: Path | None) -> dict | list:
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--acceptance", default=None)
    ap.add_argument("--task-results", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    task_payload = _load_json(_resolve(root, args.task_results))
    task_results = task_payload if isinstance(task_payload, list) else task_payload.get("tasks", [])
    payload = build_objective_benchmark_payload(
        proposal_payload=_load_json(_resolve(root, args.proposals)) or {},
        acceptance_payload=_load_json(_resolve(root, args.acceptance)) if args.acceptance else None,
        task_results=task_results,
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
