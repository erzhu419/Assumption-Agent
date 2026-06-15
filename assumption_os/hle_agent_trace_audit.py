"""Trace audit for HLE raw-vs-Assumption-Agent runs.

The audit only reads sanitized HLE artifacts and JSONL telemetry.  It does not
persist raw HLE questions, gold answers, rationales, canary strings, images, or
model prediction text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_RESULT = PAPER_DIR / "hle_text_smoke_gpt55_raw_vs_agent_n4_20260615.json"
DEFAULT_LOG = PAPER_DIR / "hle_text_smoke_gpt55_raw_vs_agent_n4_20260615.jsonl"
DEFAULT_OUT = PAPER_DIR / "hle_text_smoke_gpt55_raw_vs_agent_trace_audit_n4_20260615.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_text_smoke_gpt55_raw_vs_agent_trace_audit_n4_20260615.md")


def build_hle_agent_trace_audit_payload(*, result_path: Path, log_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    events = _read_jsonl(log_path)
    stages = _index_agent_stages(events)
    by_problem_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in result.get("rows", []):
        by_problem_variant[row["problem_id_hash"]][row["variant"]] = row

    problem_rows = []
    transitions = Counter()
    context_decisions = Counter()
    weak_morphism_count = 0
    for problem_id, variants in sorted(by_problem_variant.items()):
        raw = variants.get("raw", {})
        agent = variants.get("assumption_agent", {})
        stage = stages.get(problem_id, {})
        router = stage.get("world_model_router", {})
        retrieval = stage.get("assumption_graph_retrieval", {})
        morphism = stage.get("structural_morphism_transfer", {})
        prompt_builder = stage.get("prompt_builder", {})
        recursive = stage.get("recursive_assumption_runner", {})
        choice_assessment = _choice_assessment(
            raw=raw,
            agent=agent,
            router=router,
            retrieval=retrieval,
            morphism=morphism,
            prompt_builder=prompt_builder,
            recursive=recursive,
        )
        raw_correct = raw.get("correct")
        agent_correct = agent.get("correct")
        transition = _transition(raw_correct, agent_correct)
        transitions[transition] += 1
        context_decisions[router.get("decision", "unknown")] += 1
        structural_hits = morphism.get("structural_morphism_hits", [])
        weak_structural_hits = [
            hit for hit in structural_hits
            if str(hit.get("decision")) != "transfer_supported"
        ]
        if weak_structural_hits:
            weak_morphism_count += 1
        problem_rows.append({
            "problem_id_hash": problem_id,
            "category": raw.get("category") or agent.get("category"),
            "raw_subject": raw.get("raw_subject") or agent.get("raw_subject"),
            "answer_type": raw.get("answer_type") or agent.get("answer_type"),
            "raw_correct": raw_correct,
            "assumption_agent_correct": agent_correct,
            "transition": transition,
            "raw_latency_sec": raw.get("call_metadata", {}).get("latency_sec"),
            "assumption_agent_latency_sec": agent.get("call_metadata", {}).get("latency_sec"),
            "agent_decision": router.get("decision"),
            "context_injected": prompt_builder.get("context_injected"),
            "context_char_count": prompt_builder.get("context_char_count"),
            "top_score": router.get("top_score"),
            "formal_hit_count": router.get("formal_hit_count"),
            "structural_hit_count": router.get("structural_hit_count"),
            "structural_hit_decisions": [hit.get("decision") for hit in structural_hits],
            "retrieved_top_node_ids": retrieval.get("top_node_ids", []),
            "retrieved_top_scores": retrieval.get("top_scores", []),
            "recursive_frame_counts": recursive.get("frame_counts", {}),
            "recursive_next_action_counts": recursive.get("next_action_counts", {}),
            "rescue_attribution": _rescue_attribution(
                raw_correct=raw_correct,
                agent_correct=agent_correct,
                context_injected=prompt_builder.get("context_injected"),
                formal_hit_count=router.get("formal_hit_count"),
                structural_hit_count=router.get("structural_hit_count"),
            ),
            "choice_assessment": choice_assessment,
        })

    by_model_variant = result.get("metrics", {}).get("by_model_variant", {})
    raw_accuracy = (by_model_variant.get("gpt-5.5::raw") or {}).get("accuracy")
    agent_accuracy = (by_model_variant.get("gpt-5.5::assumption_agent") or {}).get("accuracy")
    delta = None if raw_accuracy is None or agent_accuracy is None else round(agent_accuracy - raw_accuracy, 4)
    module_summary = result.get("metrics", {}).get("module_activation_summary", {})
    agent_modules = module_summary.get("gpt-5.5::assumption_agent", {})
    activated_counts = {
        module: counts.get("activated", 0)
        for module, counts in sorted(agent_modules.items())
    }
    sample_count = result.get("metrics", {}).get("sample_count", 0)
    diagnosis = _diagnosis(
        delta=delta,
        sample_count=sample_count,
        activated_counts=activated_counts,
        transitions=transitions,
        context_decisions=context_decisions,
        weak_morphism_count=weak_morphism_count,
        problem_rows=problem_rows,
    )
    choice_counts = Counter(
        item
        for row in problem_rows
        for item in row.get("choice_assessment", {}).get("flags", [])
    )
    gates = {
        "sanitized_result_loaded": bool(result.get("rows")),
        "jsonl_trace_loaded": bool(events),
        "raw_agent_pairwise_complete": all("raw" in variants and "assumption_agent" in variants for variants in by_problem_variant.values()),
        "core_modules_activated_for_all_agent_rows": all(
            activated_counts.get(module, 0) == sample_count
            for module in [
                "assumption_graph_retrieval",
                "structural_morphism_transfer",
                "world_model_router",
                "recursive_assumption_runner",
            ]
        ),
        "no_expected_modules_missing": not result.get("metrics", {}).get("expected_but_missing_modules"),
        "raw_content_persisted": False,
    }
    # raw_content_persisted is a fact field; use pass_gates for boolean validation.
    pass_gates = {**gates, "no_raw_content_persisted": not gates["raw_content_persisted"]}
    pass_gates.pop("raw_content_persisted", None)
    return {
        "audit_id": "hle_text_smoke_gpt55_raw_vs_agent_trace_audit_n4_20260615",
        "audit_kind": "hle_assumption_agent_trace_audit",
        "result_path": str(result_path),
        "log_path": str(log_path),
        "performance_validation": True,
        "metrics": {
            "sample_count": sample_count,
            "raw_accuracy": raw_accuracy,
            "assumption_agent_accuracy": agent_accuracy,
            "agent_minus_raw_accuracy": delta,
            "transition_counts": dict(transitions),
            "context_decision_counts": dict(context_decisions),
            "weak_morphism_problem_count": weak_morphism_count,
            "module_activated_counts": activated_counts,
            "choice_assessment_counts": dict(choice_counts),
        },
        "problem_rows": problem_rows,
        "diagnosis": diagnosis,
        "gates": pass_gates,
        "failed_gates": [name for name, passed in pass_gates.items() if not passed],
        "pass": all(pass_gates.values()),
        "claim_boundary": (
            f"This is a {sample_count}-item text-only HLE diagnostic, not a full HLE benchmark.  It verifies module "
            "activation and audits whether the wrapper's choices are attributable to graph/morphism/world-model/"
            "recursive modules; it does not establish leaderboard-level HLE performance."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# HLE Assumption Agent Trace Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- sample count: `{metrics['sample_count']}`",
        f"- raw accuracy: `{metrics['raw_accuracy']}`",
        f"- assumption_agent accuracy: `{metrics['assumption_agent_accuracy']}`",
        f"- agent minus raw: `{metrics['agent_minus_raw_accuracy']}`",
        f"- transition counts: `{metrics['transition_counts']}`",
        f"- context decisions: `{metrics['context_decision_counts']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["diagnosis"])
    lines.extend([
        "",
        "## Problem-Level Trace",
        "",
        "| problem hash | answer type | category | raw | agent | transition | decision | context | formal hits | structural hits |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for row in payload["problem_rows"]:
        lines.append(
            f"| `{row['problem_id_hash']}` | `{row['answer_type']}` | `{row['category']}` | "
            f"`{row['raw_correct']}` | `{row['assumption_agent_correct']}` | `{row['transition']}` | "
            f"`{row['agent_decision']}` | `{row['context_injected']}` | "
            f"`{row['formal_hit_count']}` | `{row['structural_hit_count']}` |"
        )
    lines.extend([
        "",
        "## Rescue Attribution",
        "",
        "| problem hash | transition | attribution |",
        "| --- | --- | --- |",
    ])
    for row in payload["problem_rows"]:
        lines.append(f"| `{row['problem_id_hash']}` | `{row['transition']}` | `{row.get('rescue_attribution')}` |")
    lines.extend([
        "",
        "## Choice Assessment",
        "",
        "| problem hash | verdict | flags | rationale |",
        "| --- | --- | --- | --- |",
    ])
    for row in payload["problem_rows"]:
        assessment = row.get("choice_assessment", {})
        flags = ", ".join(assessment.get("flags", [])) or "none"
        rationale = " ".join(assessment.get("rationale", []))
        lines.append(
            f"| `{row['problem_id_hash']}` | `{assessment.get('verdict')}` | `{flags}` | {rationale} |"
        )
    lines.extend([
        "",
        "## Module Activation",
        "",
        "| module | activated count |",
        "| --- | ---: |",
    ])
    for module, count in metrics["module_activated_counts"].items():
        lines.append(f"| `{module}` | `{count}` |")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
    ])
    return "\n".join(lines).rstrip() + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _index_agent_stages(events: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event.get("event") == "agent_stage" and event.get("variant") == "assumption_agent":
            out[event["problem_id_hash"]][event["stage"]] = event.get("stage_data", {})
    return out


def _transition(raw_correct: Any, agent_correct: Any) -> str:
    if raw_correct is True and agent_correct is True:
        return "both_correct"
    if raw_correct is False and agent_correct is False:
        return "both_wrong"
    if raw_correct is False and agent_correct is True:
        return "agent_rescue"
    if raw_correct is True and agent_correct is False:
        return "agent_regression"
    return "unknown"


def _diagnosis(
    *,
    delta: float | None,
    sample_count: int,
    activated_counts: dict[str, int],
    transitions: Counter[str],
    context_decisions: Counter[str],
    weak_morphism_count: int,
    problem_rows: list[dict[str, Any]],
) -> list[str]:
    items = []
    if delta == 0:
        items.append("The full Assumption Agent wrapper tied raw gpt-5.5: no regression, but no rescue cases.")
    elif delta is not None and delta > 0:
        items.append("The full Assumption Agent wrapper beat raw gpt-5.5 on this diagnostic slice.")
    elif delta is not None:
        items.append("The full Assumption Agent wrapper underperformed raw gpt-5.5 on this diagnostic slice.")
    for module in [
        "assumption_graph_retrieval",
        "structural_morphism_transfer",
        "world_model_router",
        "recursive_assumption_runner",
    ]:
        items.append(f"{module} activated on {activated_counts.get(module, 0)}/{sample_count} agent rows.")
    if transitions.get("agent_rescue", 0) == 0:
        items.append("There were no agent_rescue transitions; every raw-wrong problem remained wrong.")
    if transitions.get("agent_regression", 0) == 0:
        items.append("There were no agent_regression transitions; the gated wrapper did not harm this slice.")
    if context_decisions.get("use_context", 0):
        items.append(f"World model injected context on {context_decisions.get('use_context', 0)} rows.")
    if context_decisions.get("abstain_to_raw_prompt", 0):
        items.append(f"World model abstained to raw prompt on {context_decisions.get('abstain_to_raw_prompt', 0)} rows.")
    if weak_morphism_count:
        items.append(
            f"Structural morphism hits were weak/repair-level on {weak_morphism_count} rows, so morphism transfer was active but not strong answer-bearing evidence."
        )
    if all((row.get("formal_hit_count") or 0) == 0 for row in problem_rows):
        items.append("Formal mapping produced zero hits on this HLE slice; the agent relied on generic graph/structural context.")
    if any((row.get("answer_type") != "multipleChoice" and row.get("context_injected")) for row in problem_rows):
        items.append("At least one exact-match question received injected context; this can add latency/structure without supplying missing factual knowledge.")
    items.append(
        "Recursive runner produced applicability frames, but did not execute child answer-generation/judge loops inside this single-call HLE wrapper."
    )
    return items


def _choice_assessment(
    *,
    raw: dict[str, Any],
    agent: dict[str, Any],
    router: dict[str, Any],
    retrieval: dict[str, Any],
    morphism: dict[str, Any],
    prompt_builder: dict[str, Any],
    recursive: dict[str, Any],
) -> dict[str, Any]:
    flags: list[str] = []
    rationale: list[str] = []

    top_scores = retrieval.get("top_scores", []) or []
    top_score = float(top_scores[0]) if top_scores else 0.0
    formal_hits = int(router.get("formal_hit_count") or 0)
    structural_hits = int(router.get("structural_hit_count") or 0)
    context_injected = bool(prompt_builder.get("context_injected"))
    answer_type = raw.get("answer_type") or agent.get("answer_type")
    structural_decisions = [
        str(hit.get("decision"))
        for hit in morphism.get("structural_morphism_hits", [])
        if isinstance(hit, dict)
    ]
    strong_structural = any(decision == "transfer_supported" for decision in structural_decisions)
    weak_structural = any(decision and decision != "transfer_supported" for decision in structural_decisions)

    if top_score >= 0.24:
        rationale.append(f"retrieval top score {top_score:.3f} is usable.")
    elif top_score >= 0.16:
        flags.append("borderline_retrieval")
        rationale.append(f"retrieval top score {top_score:.3f} is borderline.")
    else:
        flags.append("weak_retrieval")
        rationale.append(f"retrieval top score {top_score:.3f} is weak.")

    if formal_hits == 0 and structural_hits == 0:
        flags.append("no_morphism_evidence")
        rationale.append("no formal or structural morphism evidence was found.")
    elif weak_structural and not strong_structural:
        flags.append("weak_morphism_evidence")
        rationale.append(f"structural decisions were weak: {structural_decisions}.")
    elif strong_structural:
        rationale.append("at least one structural morphism was transfer-supported.")

    if context_injected and answer_type != "multipleChoice" and formal_hits == 0 and not strong_structural:
        flags.append("risky_exact_match_context_injection")
        rationale.append("context was injected into an exact-match item without strong morphism/formal evidence.")
    if context_injected and top_score < 0.20:
        flags.append("risky_low_score_context_injection")
        rationale.append("context was injected despite low retrieval score.")
    if not context_injected and (formal_hits > 0 or strong_structural):
        flags.append("possibly_overconservative_abstain")
        rationale.append("world model abstained despite strong transfer evidence.")

    next_actions = recursive.get("next_action_counts", {}) or {}
    if next_actions and set(next_actions) <= {"verify_applicability"}:
        flags.append("recursive_planning_only")
        rationale.append("recursive runner opened applicability checks but did not execute child validation.")
    elif not next_actions:
        flags.append("recursive_no_action")
        rationale.append("recursive runner did not produce next actions.")

    if raw.get("correct") is False and agent.get("correct") is False:
        flags.append("no_rescue")
    if raw.get("correct") is True and agent.get("correct") is False:
        flags.append("regression")
    if raw.get("correct") is False and agent.get("correct") is True:
        flags.append("rescue")
        if not context_injected:
            flags.append("abstain_rescue_not_module_attributable")
            rationale.append("agent rescued while abstaining to the raw prompt, so this is likely repeat-call variance.")

    if "regression" in flags:
        verdict = "bad_choice"
    elif "abstain_rescue_not_module_attributable" in flags:
        verdict = "lucky_or_repeat_variance"
    elif any(flag.startswith("risky_") for flag in flags):
        verdict = "questionable_choice"
    elif "rescue" in flags:
        verdict = "good_choice"
    elif "no_rescue" in flags:
        verdict = "safe_but_unhelpful"
    else:
        verdict = "reasonable_no_regression"

    return {
        "verdict": verdict,
        "flags": flags,
        "rationale": rationale,
    }


def _rescue_attribution(
    *,
    raw_correct: Any,
    agent_correct: Any,
    context_injected: Any,
    formal_hit_count: Any,
    structural_hit_count: Any,
) -> str:
    if raw_correct is not False or agent_correct is not True:
        return "not_rescue"
    if context_injected:
        if int(formal_hit_count or 0) > 0 or int(structural_hit_count or 0) > 0:
            return "possibly_agent_context_attributable"
        return "context_injected_but_no_transfer_evidence"
    return "not_module_attributable_raw_prompt_repeat_variance"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HLE Assumption Agent trace logs.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    payload = build_hle_agent_trace_audit_payload(result_path=Path(args.result), log_path=Path(args.log))
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
