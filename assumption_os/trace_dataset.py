"""Build trainable trace-to-outcome rows for the assumption loop.

Runtime traces and harness artifacts are useful only if they can be linked to
judged outcomes.  This module joins sample rows, answer metadata, runtime
events, and pairwise judgments into compact rows that a world model or residual
clusterer can consume without re-reading raw prompts or answers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .manifest_logger import redact_secrets
from .schema import ResidualType, stable_id


def build_trace_dataset_payload(
    *,
    root: Path,
    sample_path: Path,
    meta_path: Path,
    judgments_path: Path,
    intervention_variant: str,
    baseline_variant: str,
    eval_id: str,
    trace_events_path: Path | None = None,
    trace_summary_path: Path | None = None,
    allow_artifact_trace: bool = False,
) -> dict:
    """Join runtime traces with judged outcomes and residual labels."""

    sample_rows = _load_sample(sample_path)
    sample_by_pid = {str(row["problem_id"]): row for row in sample_rows if row.get("problem_id")}
    meta_by_pid = _load_json_dict(meta_path)
    judgments = _load_json_dict(judgments_path)
    trace_events = _load_trace_events(trace_events_path=trace_events_path, trace_summary_path=trace_summary_path)
    trace_by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in trace_events:
        pid = str(event.get("problem_id") or "")
        if pid:
            trace_by_pid[pid].append(event)
    judgment_variant_a, judgment_variant_b = _judgment_variant_order(
        judgments_path,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
    )

    rows: list[dict[str, Any]] = []
    for pid in sorted(sample_by_pid):
        judgment = judgments.get(pid)
        if not isinstance(judgment, dict):
            continue
        problem = sample_by_pid[pid]
        meta = meta_by_pid.get(pid, {}) if isinstance(meta_by_pid.get(pid), dict) else {}
        events = list(trace_by_pid.get(pid, []))
        trace_source = "first_party_runtime" if events else "missing"
        first_party_trace = bool(events)
        if not events and allow_artifact_trace:
            events = [_artifact_replay_event(problem=problem, meta=meta, meta_path=meta_path)]
            trace_source = "artifact_replay"

        rows.append(_build_row(
            eval_id=eval_id,
            problem=problem,
            meta=meta,
            judgment=judgment,
            events=events,
            trace_source=trace_source,
            first_party_trace=first_party_trace,
            intervention_variant=intervention_variant,
            baseline_variant=baseline_variant,
            judgment_variant_a=judgment_variant_a,
            judgment_variant_b=judgment_variant_b,
            judgments_path=judgments_path,
        ))

    payload = _summarize_rows(
        eval_id=eval_id,
        root=root,
        sample_path=sample_path,
        meta_path=meta_path,
        judgments_path=judgments_path,
        trace_events_path=trace_events_path,
        trace_summary_path=trace_summary_path,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
        rows=rows,
    )
    clean_payload = redact_secrets(payload)
    clean_payload["secret_leak_detected"] = _contains_secret(clean_payload)
    return clean_payload


def _build_row(
    *,
    eval_id: str,
    problem: dict[str, Any],
    meta: dict[str, Any],
    judgment: dict[str, Any],
    events: list[dict[str, Any]],
    trace_source: str,
    first_party_trace: bool,
    intervention_variant: str,
    baseline_variant: str,
    judgment_variant_a: str | None,
    judgment_variant_b: str | None,
    judgments_path: Path,
) -> dict:
    pid = str(problem["problem_id"])
    outcome = _outcome_from_judgment(
        judgment,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
    )
    score_intervention, score_baseline = _scores_from_judgment(
        judgment,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
        judgment_variant_a=judgment_variant_a,
        judgment_variant_b=judgment_variant_b,
    )
    event_counts = Counter(str(event.get("event_type") or event.get("action_type") or "unknown") for event in events)
    component_counts = Counter(str(event.get("component") or "unknown") for event in events)
    active_ids = _collect_assumption_ids(events)
    gold_ids = {f"strategy_{sid}" for sid in problem.get("coverage_tags", [])}
    gold_hit = bool(gold_ids & set(active_ids))
    residual_type, residual = _residual_for_row(
        outcome=outcome,
        active_ids=active_ids,
        gold_hit=gold_hit,
        meta=meta,
        component_counts=component_counts,
        judgment=judgment,
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
    )
    row = {
        "row_id": stable_id("trace_row", eval_id, pid, intervention_variant, baseline_variant),
        "eval_id": eval_id,
        "problem_id": pid,
        "domain": problem.get("domain"),
        "difficulty": problem.get("difficulty"),
        "frame": meta.get("frame"),
        "bypass_route": meta.get("bypass_route"),
        "trace_source": trace_source,
        "first_party_trace": first_party_trace,
        "trace_event_count": len(events),
        "event_counts": dict(event_counts),
        "component_counts": dict(component_counts),
        "components": sorted(component_counts),
        "prompt_kinds": sorted({
            str((event.get("artifacts") or {}).get("prompt_kind"))
            for event in events
            if (event.get("artifacts") or {}).get("prompt_kind")
        }),
        "activated_assumption_ids": active_ids,
        "gold_assumption_ids": sorted(gold_ids),
        "gold_hit": gold_hit,
        "outcome": outcome,
        "label": {"win": 1.0, "tie": 0.5, "loss": 0.0}.get(outcome, 0.5),
        "winner": judgment.get("winner"),
        "score_intervention": score_intervention,
        "score_baseline": score_baseline,
        "score_delta": _score_delta(score_intervention, score_baseline),
        "residual_type": residual_type.value,
        "residual": residual,
        "trainable": outcome in {"win", "loss"} and bool(events),
        "judgment_artifact": str(judgments_path),
        "judgment_reasoning_preview": _preview(judgment.get("reasoning")),
        "features": {
            "domain": problem.get("domain"),
            "difficulty": problem.get("difficulty"),
            "frame": meta.get("frame"),
            "bypass_route": meta.get("bypass_route"),
            "trace_event_count": len(events),
            "event_counts": dict(event_counts),
            "component_counts": dict(component_counts),
            "active_assumption_count": len(active_ids),
            "gold_hit": gold_hit,
        },
    }
    return row


def _summarize_rows(
    *,
    eval_id: str,
    root: Path,
    sample_path: Path,
    meta_path: Path,
    judgments_path: Path,
    trace_events_path: Path | None,
    trace_summary_path: Path | None,
    intervention_variant: str,
    baseline_variant: str,
    rows: list[dict[str, Any]],
) -> dict:
    outcome_counts = Counter(row["outcome"] for row in rows)
    residual_type_counts = Counter(row["residual_type"] for row in rows)
    event_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    for row in rows:
        event_counts.update(row["event_counts"])
        component_counts.update(row["component_counts"])
    judged_rows = [row for row in rows if row["outcome"] in {"win", "loss", "tie"}]
    trace_rows = [row for row in judged_rows if row["trace_event_count"] > 0]
    trainable_rows = [row for row in rows if row["trainable"]]
    score_deltas = [row["score_delta"] for row in rows if row["score_delta"] is not None]
    return {
        "eval_id": eval_id,
        "source": {
            "root": ".",
            "sample_path": _display_path(root, sample_path),
            "meta_path": _display_path(root, meta_path),
            "judgments_path": _display_path(root, judgments_path),
            "trace_events_path": _display_path(root, trace_events_path) if trace_events_path else None,
            "trace_summary_path": _display_path(root, trace_summary_path) if trace_summary_path else None,
            "intervention_variant": intervention_variant,
            "baseline_variant": baseline_variant,
        },
        "row_count": len(rows),
        "judged_row_count": len(judged_rows),
        "trainable_row_count": len(trainable_rows),
        "first_party_trace_count": sum(1 for row in rows if row["first_party_trace"]),
        "artifact_replay_count": sum(1 for row in rows if row["trace_source"] == "artifact_replay"),
        "missing_trace_count": sum(1 for row in rows if row["trace_event_count"] == 0),
        "outcome_counts": dict(outcome_counts),
        "residual_type_counts": dict(residual_type_counts),
        "event_counts": dict(event_counts),
        "component_counts": dict(component_counts),
        "traced_outcome_coverage": round(len(trace_rows) / len(judged_rows), 4) if judged_rows else 0.0,
        "assumption_id_coverage": round(
            sum(1 for row in trace_rows if row["activated_assumption_ids"]) / len(trace_rows),
            4,
        ) if trace_rows else 0.0,
        "mean_score_delta": round(sum(score_deltas) / len(score_deltas), 4) if score_deltas else None,
        "rows": rows,
    }


def _load_trace_events(*, trace_events_path: Path | None, trace_summary_path: Path | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if trace_events_path and trace_events_path.exists():
        for line in trace_events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(_normalize_event(event, source="events_jsonl"))
    if trace_summary_path and trace_summary_path.exists():
        summary = json.loads(trace_summary_path.read_text(encoding="utf-8"))
        for manifest in summary.get("manifests", []):
            if isinstance(manifest, dict):
                events.append(_event_from_manifest(manifest, source="summary_manifest"))
    return events


def _normalize_event(event: dict[str, Any], *, source: str) -> dict[str, Any]:
    clean = redact_secrets(dict(event))
    clean.setdefault("event_type", clean.get("action_type", "unknown"))
    clean.setdefault("artifacts", {})
    clean.setdefault("metadata", {})
    clean["metadata"] = {**clean.get("metadata", {}), "trace_event_source": source}
    return clean


def _event_from_manifest(manifest: dict[str, Any], *, source: str) -> dict[str, Any]:
    return _normalize_event({
        "event_type": manifest.get("action_type", "unknown"),
        "problem_id": manifest.get("problem_id"),
        "component": manifest.get("component"),
        "assumption": manifest.get("assumption"),
        "assumption_ids": manifest.get("assumption_ids", []),
        "artifacts": manifest.get("artifacts", {}),
        "metadata": manifest.get("metadata", {}),
        "observed_effect": manifest.get("observed_effect"),
    }, source=source)


def _artifact_replay_event(*, problem: dict[str, Any], meta: dict[str, Any], meta_path: Path) -> dict[str, Any]:
    return {
        "event_type": "tool_use",
        "problem_id": problem.get("problem_id"),
        "component": "artifact_replay_answer_meta",
        "assumption": "Cached answer metadata can be replayed as bounded trace input when first-party runtime trace is unavailable.",
        "artifacts": {
            "path": str(meta_path),
            "domain": problem.get("domain"),
            "difficulty": problem.get("difficulty"),
            "frame": meta.get("frame"),
            "bypass_route": meta.get("bypass_route"),
        },
        "metadata": {"trace_event_source": "artifact_replay"},
    }


def _collect_assumption_ids(events: Iterable[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for event in events:
        ids.extend(str(x) for x in event.get("assumption_ids", []) if x)
        artifacts = event.get("artifacts") or {}
        metadata = event.get("metadata") or {}
        for key in ("activated_assumption_ids", "active_assumption_ids", "assumption_ids"):
            ids.extend(str(x) for x in artifacts.get(key, []) if x)
            ids.extend(str(x) for x in metadata.get(key, []) if x)
    return sorted(dict.fromkeys(ids))


def _outcome_from_judgment(
    judgment: dict[str, Any],
    *,
    intervention_variant: str,
    baseline_variant: str,
) -> str:
    winner = str(judgment.get("winner", "tie"))
    if winner == intervention_variant:
        return "win"
    if winner == baseline_variant:
        return "loss"
    if winner.lower() in {"tie", "draw", "none"}:
        return "tie"
    a_was = str(judgment.get("a_was", "")).upper()
    if winner.upper() == "A":
        return "win" if a_was == "A" else "loss" if a_was == "B" else "tie"
    if winner.upper() == "B":
        return "win" if a_was == "B" else "loss" if a_was == "A" else "tie"
    return "tie"


def _scores_from_judgment(
    judgment: dict[str, Any],
    *,
    intervention_variant: str,
    baseline_variant: str,
    judgment_variant_a: str | None,
    judgment_variant_b: str | None,
) -> tuple[float | None, float | None]:
    score_a = _as_float(judgment.get("score_a"))
    score_b = _as_float(judgment.get("score_b"))
    if judgment_variant_a and judgment_variant_b:
        scores = {judgment_variant_a: score_a, judgment_variant_b: score_b}
        return scores.get(intervention_variant), scores.get(baseline_variant)
    a_was = str(judgment.get("a_was", "")).upper()
    if a_was == "A":
        return score_a, score_b
    if a_was == "B":
        return score_b, score_a
    return None, None


def _judgment_variant_order(
    path: Path,
    *,
    intervention_variant: str,
    baseline_variant: str,
) -> tuple[str | None, str | None]:
    stem = path.stem
    if stem == f"{intervention_variant}_vs_{baseline_variant}":
        return intervention_variant, baseline_variant
    if stem == f"{baseline_variant}_vs_{intervention_variant}":
        return baseline_variant, intervention_variant
    if stem.startswith(f"{intervention_variant}_vs_") and stem.endswith(baseline_variant):
        return intervention_variant, baseline_variant
    if stem.startswith(f"{baseline_variant}_vs_") and stem.endswith(intervention_variant):
        return baseline_variant, intervention_variant
    return None, None


def _residual_for_row(
    *,
    outcome: str,
    active_ids: list[str],
    gold_hit: bool,
    meta: dict[str, Any],
    component_counts: Counter[str],
    judgment: dict[str, Any],
    intervention_variant: str,
    baseline_variant: str,
) -> tuple[ResidualType, str | None]:
    if outcome == "win":
        return ResidualType.NO_RESIDUAL, None
    if outcome == "tie":
        return ResidualType.UNKNOWN, None
    reason = _preview(judgment.get("reasoning"), limit=220)
    if active_ids:
        if gold_hit:
            return (
                ResidualType.OPTIMIZATION,
                f"Active assumptions included a gold tag, but {baseline_variant} beat {intervention_variant}; optimize execution payload. Judge: {reason}",
            )
        return (
            ResidualType.MEMORY_DEFECT,
            f"Active assumptions missed the sample gold tags and {baseline_variant} won; update retrieval memory. Judge: {reason}",
        )
    if meta.get("bypass_route") or "phase2_math_science_bypass" in component_counts or "phase2_cache_hit" in component_counts:
        return (
            ResidualType.OPTIMIZATION,
            f"No graph ids fired, but bypass/cache route {meta.get('bypass_route')} lost; optimize the bypass bridge. Judge: {reason}",
        )
    return (
        ResidualType.UNKNOWN,
        f"No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: {reason}",
    )


def _load_sample(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [
            {**row, "problem_id": row.get("problem_id", pid)}
            for pid, row in payload.items()
            if isinstance(row, dict)
        ]
    return []


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {
            str(row["problem_id"]): row
            for row in payload
            if isinstance(row, dict) and row.get("problem_id")
        }
    return {}


def _contains_secret(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return redact_secrets(text) != text


def _preview(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _resolve(root: Path, path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--trace-events", default=None)
    ap.add_argument("--trace-summary", default=None)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--allow-artifact-trace", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_trace_dataset_payload(
        root=root,
        sample_path=_resolve(root, args.sample),
        meta_path=_resolve(root, args.meta),
        judgments_path=_resolve(root, args.judgments),
        trace_events_path=_resolve(root, args.trace_events),
        trace_summary_path=_resolve(root, args.trace_summary),
        intervention_variant=args.intervention,
        baseline_variant=args.baseline,
        eval_id=args.eval_id,
        allow_artifact_trace=args.allow_artifact_trace,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
