"""Evaluate OperatorSpec application fidelity for cached Phase2 variants."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

from assumption_os.application_fidelity import audit_answer_application
from assumption_os.graph_memory import JsonlGraphStore
from assumption_os.operator_specs import build_operator_specs

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
DEFAULT_GRAPH = PROJECT.parent / "phase four" / "assumption_graph"
DEFAULT_OUT_DIR = DEFAULT_GRAPH
DEFAULT_MD_DIR = PROJECT.parent / "reconstruction" / "md"


LLM_FIDELITY_PROMPT = """# OperatorSpec Application Fidelity Judge

You judge whether the answer truly executed the supplied OperatorSpecs.

## Problem
{problem}

## OperatorSpecs
{operator_specs}

## Answer
{answer}

For each operator, decide whether required slots are substantively filled.
Do not give credit for merely mentioning the source claim or using generic words.

Output JSON only:
{{
  "used_assumption_ids": ["..."],
  "ignored_assumption_ids": ["..."],
  "misapplied_assumption_ids": ["..."],
  "decorative_use_count": 0,
  "slot_completion_rate": 0.0,
  "application_fidelity": 0.0,
  "operators": [
    {{
      "source_id": "...",
      "filled_slots": ["..."],
      "missing_slots": ["..."],
      "verdict": "used|ignored|decorative|misapplied",
      "reasoning": "short reason"
    }}
  ]
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sample", default="proposal_samples/codex_operator_ab_non_bypass_n12_20260621.json")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--out", default="")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--min-slot-completion", type=float, default=0.65)
    ap.add_argument("--llm-judge", action="store_true")
    args = ap.parse_args()

    sample_path = _resolve_sample(args.sample)
    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    if args.max_problems:
        rows = rows[: args.max_problems]
    answers = _load_json(ANSWERS_DIR / f"{args.variant}_answers.json")
    meta = _load_json(ANSWERS_DIR / f"{args.variant}_meta.json")
    graph_store = JsonlGraphStore(args.graph)

    llm_client = None
    llm_parse = None
    if args.llm_judge:
        from llm_client import create_client, parse_json_from_llm

        llm_client = create_client()
        llm_parse = parse_json_from_llm

    audits: dict[str, Any] = {}
    for row in rows:
        pid = row["problem_id"]
        answer = answers.get(pid, "")
        item_meta = meta.get(pid, {}) if isinstance(meta.get(pid), dict) else {}
        specs = _operator_specs_for_problem(item_meta, graph_store)
        audit = audit_answer_application(
            answer,
            specs,
            min_slot_completion=args.min_slot_completion,
        )
        audit.update({
            "problem_id": pid,
            "domain": row.get("domain", item_meta.get("domain", "?")),
            "difficulty": row.get("difficulty", item_meta.get("difficulty", "?")),
            "answer_present": bool(answer),
            "operator_source_ids": [spec.get("source_id") for spec in specs],
        })
        if args.llm_judge and specs and answer:
            audit["llm_audit"] = _llm_fidelity_judge(
                llm_client,
                llm_parse,
                problem=row.get("description", ""),
                answer=answer,
                specs=specs,
            )
        audits[pid] = audit

    summary = _summarize(audits)
    payload = {
        "eval_id": f"{args.variant}_operator_fidelity",
        "variant": args.variant,
        "sample": str(sample_path),
        "graph": str(Path(args.graph)),
        "min_slot_completion": args.min_slot_completion,
        "llm_judge": args.llm_judge,
        "summary": summary,
        "audits": audits,
    }
    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{args.variant}_operator_fidelity_20260621.json"
    md_path = Path(args.md_out) if args.md_out else DEFAULT_MD_DIR / f"{args.variant}_operator_fidelity_20260621.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out_path)
    print(md_path)


def _operator_specs_for_problem(meta: dict[str, Any], graph_store: JsonlGraphStore) -> list[dict[str, Any]]:
    specs = meta.get("assumption_operator_specs")
    if isinstance(specs, list) and specs:
        return [dict(spec) for spec in specs if isinstance(spec, dict)]
    ids = meta.get("assumption_operator_ids") or []
    nodes = [graph_store.nodes[node_id] for node_id in ids if node_id in graph_store.nodes]
    return [spec.to_dict() for spec in build_operator_specs(nodes, max_specs=max(1, len(nodes)))]


def _llm_fidelity_judge(client, parse_json_from_llm, *, problem: str, answer: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = LLM_FIDELITY_PROMPT.format(
        problem=problem,
        answer=answer,
        operator_specs=json.dumps(specs, ensure_ascii=False, indent=2),
    )
    for attempt in range(4):
        try:
            response = client.generate(prompt, max_tokens=700, temperature=0.1)
            return parse_json_from_llm(response["text"])
        except Exception as exc:
            if attempt == 3:
                return {"parse_or_call_error": str(exc)}
            time.sleep(3 * (attempt + 1))
    return {"parse_or_call_error": "unreachable"}


def _summarize(audits: dict[str, Any]) -> dict[str, Any]:
    rows = list(audits.values())
    operator_rows = [row for row in rows if row.get("operator_count", 0)]
    by_domain: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("domain", "?")].append(row)
    for domain, items in sorted(grouped.items()):
        op_items = [item for item in items if item.get("operator_count", 0)]
        by_domain[domain] = _summary_for_rows(op_items if op_items else items)
        by_domain[domain]["operator_problem_n"] = len(op_items)
        by_domain[domain]["problem_n"] = len(items)
    summary = _summary_for_rows(operator_rows)
    summary.update({
        "problem_n": len(rows),
        "operator_problem_n": len(operator_rows),
        "operator_count": sum(row.get("operator_count", 0) for row in rows),
        "gate_status_counts": dict(Counter(
            row.get("operator_count", 0) and "operator_present" or "no_operator"
            for row in rows
        )),
        "by_domain": by_domain,
    })
    return summary


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "pass_rate": 0.0,
            "application_fidelity_mean": 0.0,
            "slot_completion_mean": 0.0,
            "decorative_use_rate": 0.0,
            "used_operator_rate": 0.0,
            "ignored_operator_count": 0,
            "misapplied_operator_count": 0,
        }
    operator_count = sum(row.get("operator_count", 0) for row in rows)
    used_count = sum(len(row.get("used_assumption_ids", [])) for row in rows)
    ignored_count = sum(len(row.get("ignored_assumption_ids", [])) for row in rows)
    misapplied_count = sum(len(row.get("misapplied_assumption_ids", [])) for row in rows)
    return {
        "n": len(rows),
        "pass_rate": round(sum(1 for row in rows if row.get("pass")) / len(rows), 4),
        "application_fidelity_mean": round(statistics.mean(row.get("application_fidelity", 0.0) for row in rows), 4),
        "slot_completion_mean": round(statistics.mean(row.get("slot_completion_rate", 0.0) for row in rows), 4),
        "decorative_use_rate": round(sum(1 for row in rows if row.get("decorative_use_count", 0)) / len(rows), 4),
        "used_operator_rate": round(used_count / operator_count, 4) if operator_count else 0.0,
        "ignored_operator_count": ignored_count,
        "misapplied_operator_count": misapplied_count,
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Operator Fidelity: {payload['variant']}",
        "",
        "## Summary",
        "",
        f"- problems: `{summary['problem_n']}`",
        f"- operator problems: `{summary['operator_problem_n']}`",
        f"- operators: `{summary['operator_count']}`",
        f"- pass rate: `{summary['pass_rate']}`",
        f"- application fidelity mean: `{summary['application_fidelity_mean']}`",
        f"- slot completion mean: `{summary['slot_completion_mean']}`",
        f"- decorative use rate: `{summary['decorative_use_rate']}`",
        f"- used operator rate: `{summary['used_operator_rate']}`",
        "",
        "## By Domain",
        "",
    ]
    for domain, row in summary["by_domain"].items():
        lines.append(
            f"- `{domain}`: problem_n={row['problem_n']}, operator_problem_n={row['operator_problem_n']}, "
            f"pass_rate={row['pass_rate']}, fidelity={row['application_fidelity_mean']}, "
            f"slot_completion={row['slot_completion_mean']}"
        )
    return "\n".join(lines) + "\n"


def _resolve_sample(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = CACHE / raw
    return candidate if candidate.exists() else path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
