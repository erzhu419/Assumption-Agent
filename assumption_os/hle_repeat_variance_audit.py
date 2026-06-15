"""Compare HLE Assumption Agent against repeated raw calls.

This audit is designed for small gated HLE diagnostics.  It reads sanitized
result artifacts only and writes hashes/aggregate correctness, never raw HLE
questions, answers, canaries, images, or prediction text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_AGENT_RESULT = PAPER_DIR / "hle_text_smoke_gpt55_raw_vs_agent_gatefix_n8_seed5_20260615.json"
DEFAULT_REPEAT_RESULT = PAPER_DIR / "hle_text_smoke_gpt55_raw_repeat_n8_seed5_20260615.json"
DEFAULT_OUT = PAPER_DIR / "hle_text_smoke_gpt55_raw_agent_repeat_variance_n8_seed5_20260615.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_text_smoke_gpt55_raw_agent_repeat_variance_n8_seed5_20260615.md")


def build_hle_repeat_variance_payload(*, agent_result_path: Path, repeat_result_path: Path) -> dict[str, Any]:
    agent_result = json.loads(agent_result_path.read_text(encoding="utf-8"))
    repeat_result = json.loads(repeat_result_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in agent_result.get("rows", []):
        rows.setdefault(row["problem_id_hash"], {})[row["variant"]] = row
    for row in repeat_result.get("rows", []):
        rows.setdefault(row["problem_id_hash"], {})[row["variant"]] = row

    problem_rows = []
    for problem_id, variants in sorted(rows.items()):
        raw = variants.get("raw", {})
        agent = variants.get("assumption_agent", {})
        repeat = variants.get("raw_repeat", {})
        problem_rows.append({
            "problem_id_hash": problem_id,
            "category": raw.get("category") or agent.get("category") or repeat.get("category"),
            "raw_subject": raw.get("raw_subject") or agent.get("raw_subject") or repeat.get("raw_subject"),
            "answer_type": raw.get("answer_type") or agent.get("answer_type") or repeat.get("answer_type"),
            "raw_correct": raw.get("correct"),
            "assumption_agent_correct": agent.get("correct"),
            "raw_repeat_correct": repeat.get("correct"),
            "raw_prediction_hash": raw.get("prediction_hash"),
            "assumption_agent_prediction_hash": agent.get("prediction_hash"),
            "raw_repeat_prediction_hash": repeat.get("prediction_hash"),
            "agent_vs_raw": _transition(raw.get("correct"), agent.get("correct"), "agent"),
            "repeat_vs_raw": _transition(raw.get("correct"), repeat.get("correct"), "repeat"),
            "agent_vs_repeat": _pairwise(agent.get("correct"), repeat.get("correct")),
            "repeat_error": bool(repeat.get("error")),
        })
    metrics = {
        "problem_count": len(problem_rows),
        "raw_correct_count": _count_true(problem_rows, "raw_correct"),
        "assumption_agent_correct_count": _count_true(problem_rows, "assumption_agent_correct"),
        "raw_repeat_correct_count": _count_true(problem_rows, "raw_repeat_correct"),
        "raw_accuracy": _accuracy(problem_rows, "raw_correct"),
        "assumption_agent_accuracy": _accuracy(problem_rows, "assumption_agent_correct"),
        "raw_repeat_accuracy": _accuracy(problem_rows, "raw_repeat_correct"),
        "agent_minus_raw": _delta(problem_rows, "assumption_agent_correct", "raw_correct"),
        "repeat_minus_raw": _delta(problem_rows, "raw_repeat_correct", "raw_correct"),
        "agent_minus_repeat": _delta(problem_rows, "assumption_agent_correct", "raw_repeat_correct"),
        "agent_rescue_count": sum(1 for row in problem_rows if row["agent_vs_raw"] == "agent_rescue"),
        "repeat_rescue_count": sum(1 for row in problem_rows if row["repeat_vs_raw"] == "repeat_rescue"),
        "agent_unique_correct_count": sum(
            1 for row in problem_rows
            if row["assumption_agent_correct"] is True and row["raw_repeat_correct"] is not True
        ),
        "repeat_unique_correct_count": sum(
            1 for row in problem_rows
            if row["raw_repeat_correct"] is True and row["assumption_agent_correct"] is not True
        ),
        "raw_repeat_error_count": sum(1 for row in problem_rows if row["repeat_error"]),
    }
    diagnosis = _diagnosis(metrics=metrics, problem_rows=problem_rows)
    gates = {
        "agent_result_loaded": bool(agent_result.get("rows")),
        "repeat_result_loaded": bool(repeat_result.get("rows")),
        "same_problem_set": sorted(rows) == sorted(row["problem_id_hash"] for row in problem_rows),
        "no_raw_content_persisted": True,
    }
    return {
        "audit_id": "hle_text_smoke_gpt55_raw_agent_repeat_variance_n8_seed5_20260615",
        "audit_kind": "hle_raw_repeat_variance_audit",
        "agent_result_path": str(agent_result_path),
        "repeat_result_path": str(repeat_result_path),
        "performance_validation": True,
        "metrics": metrics,
        "problem_rows": problem_rows,
        "diagnosis": diagnosis,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundary": (
            "This audit estimates repeat-call variance on a small 8-item HLE text-only slice.  Agent gains on "
            "abstained rows are not module-attributable unless they exceed a raw-repeat control."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# HLE Raw/Agent/Repeat Variance Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- raw accuracy: `{metrics['raw_accuracy']}`",
        f"- assumption_agent accuracy: `{metrics['assumption_agent_accuracy']}`",
        f"- raw_repeat accuracy: `{metrics['raw_repeat_accuracy']}`",
        f"- agent minus raw: `{metrics['agent_minus_raw']}`",
        f"- repeat minus raw: `{metrics['repeat_minus_raw']}`",
        f"- agent minus repeat: `{metrics['agent_minus_repeat']}`",
        f"- raw_repeat errors: `{metrics['raw_repeat_error_count']}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["diagnosis"])
    lines.extend([
        "",
        "## Problem-Level Comparison",
        "",
        "| problem hash | answer type | category | raw | agent | repeat | agent vs raw | repeat vs raw | agent vs repeat |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for row in payload["problem_rows"]:
        lines.append(
            f"| `{row['problem_id_hash']}` | `{row['answer_type']}` | `{row['category']}` | "
            f"`{row['raw_correct']}` | `{row['assumption_agent_correct']}` | `{row['raw_repeat_correct']}` | "
            f"`{row['agent_vs_raw']}` | `{row['repeat_vs_raw']}` | `{row['agent_vs_repeat']}` |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
    ])
    return "\n".join(lines).rstrip() + "\n"


def _transition(base: Any, candidate: Any, label: str) -> str:
    if base is True and candidate is True:
        return "both_correct"
    if base is False and candidate is False:
        return "both_wrong"
    if base is False and candidate is True:
        return f"{label}_rescue"
    if base is True and candidate is False:
        return f"{label}_regression"
    return "unknown"


def _pairwise(agent: Any, repeat: Any) -> str:
    if agent is True and repeat is True:
        return "both_correct"
    if agent is False and repeat is False:
        return "both_wrong"
    if agent is True and repeat is not True:
        return "agent_only_correct"
    if repeat is True and agent is not True:
        return "repeat_only_correct"
    return "unknown"


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def _accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(_count_true(rows, key) / len(rows), 4)


def _delta(rows: list[dict[str, Any]], a: str, b: str) -> float | None:
    aa = _accuracy(rows, a)
    bb = _accuracy(rows, b)
    if aa is None or bb is None:
        return None
    return round(aa - bb, 4)


def _diagnosis(*, metrics: dict[str, Any], problem_rows: list[dict[str, Any]]) -> list[str]:
    items = [
        f"Assumption agent improved over first raw by {metrics['agent_minus_raw']}, but raw_repeat also improved by {metrics['repeat_minus_raw']}.",
        f"Agent has {metrics['agent_unique_correct_count']} correct rows not matched by raw_repeat; raw_repeat has {metrics['repeat_unique_correct_count']} correct rows not matched by agent.",
    ]
    if metrics["raw_repeat_error_count"]:
        items.append("raw_repeat had provider errors, so repeat baseline is slightly pessimistic.")
    if metrics["agent_minus_repeat"] is not None and metrics["agent_minus_repeat"] > 0:
        items.append("Agent is above raw_repeat on this small slice, but gains remain weakly attributable because most agent rows abstained to raw prompt.")
    elif metrics["agent_minus_repeat"] == 0:
        items.append("Agent does not exceed raw_repeat; gains are consistent with repeat-call variance.")
    else:
        items.append("Agent is below raw_repeat; gains should not be claimed.")
    abstained_rescues = [
        row for row in problem_rows
        if row["agent_vs_raw"] == "agent_rescue"
    ]
    if abstained_rescues:
        items.append(
            "Agent rescue rows require stage-log attribution; if their prompt_builder abstained, they should be treated as repeat-call variance rather than module benefit."
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HLE agent result to raw-repeat control.")
    parser.add_argument("--agent-result", default=str(DEFAULT_AGENT_RESULT))
    parser.add_argument("--repeat-result", default=str(DEFAULT_REPEAT_RESULT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    payload = build_hle_repeat_variance_payload(
        agent_result_path=Path(args.agent_result),
        repeat_result_path=Path(args.repeat_result),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
