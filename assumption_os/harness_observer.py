"""Harness artifact observer for manifest coverage.

The generic manifest logger records events when callers already know what
happened.  This module audits existing harness artifacts and converts them into
TrialManifest-shaped events so cached judge/meta/log files stop being invisible
to the Assumption Graph.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .graph_memory import JsonlGraphStore
from .manifest_logger import build_component_manifest_payload, events_from_run_logs, redact_secrets


DEFAULT_ARTIFACT_GLOBS = {
    "judgment_json": ["phase two/analysis/cache/judgments/*.json"],
    "answer_meta_json": ["phase two/analysis/cache/answers/*_meta.json"],
    "run_log": [
        "phase four/assumption_graph/*.log",
        "phase six/autonomous/exp*_run.log",
    ],
}


def build_harness_observer_payload(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
    artifact_paths: Iterable[Path] | None = None,
    max_files_per_kind: int = 6,
    max_events_per_file: int = 12,
    include_covered: bool = False,
    writeback: bool = False,
) -> dict:
    """Discover/convert harness artifacts and optionally persist manifests."""

    store = JsonlGraphStore(graph_dir)
    discovered = (
        list(artifact_paths)
        if artifact_paths is not None
        else discover_harness_artifacts(root=root, max_files_per_kind=max_files_per_kind)
    )
    events = events_from_harness_artifacts(
        root=root,
        artifact_paths=discovered,
        max_events_per_file=max_events_per_file,
    )
    preexisting_paths = _trial_artifact_paths(store)
    event_paths = _event_artifact_paths(events)
    backfill_events = [
        event for event in events
        if include_covered or str(event.get("artifacts", {}).get("path")) not in preexisting_paths
    ]
    payload = build_component_manifest_payload(
        eval_id=eval_id,
        events=backfill_events,
        store=store,
        writeback=writeback,
    )
    post_store = JsonlGraphStore(graph_dir) if writeback else store
    post_paths = _trial_artifact_paths(post_store)
    payload["artifact_coverage"] = {
        "artifact_file_count": len(discovered),
        "event_artifact_file_count": len(event_paths),
        "preexisting_covered_file_count": len(event_paths & preexisting_paths),
        "post_covered_file_count": len(event_paths & post_paths),
        "uncovered_after_writeback": sorted(event_paths - post_paths),
        "full_coverage_after_writeback": bool(event_paths <= post_paths),
        "artifact_paths": [_display_path(root, path) for path in discovered],
    }
    payload["discovered_event_count"] = len(events)
    payload["backfilled_event_count"] = len(backfill_events)
    payload["skipped_covered_event_count"] = len(events) - len(backfill_events)
    payload["discovered_event_counts"] = dict(Counter(
        event.get("event_type", "unknown")
        for event in events
    ))
    payload["artifact_kind_counts"] = dict(Counter(
        event.get("metadata", {}).get("artifact_kind", "unknown")
        for event in events
    ))
    payload["secret_leak_detected"] = _has_secret_probe(payload)
    return payload


def discover_harness_artifacts(*, root: Path, max_files_per_kind: int = 6) -> list[Path]:
    """Return a bounded, deterministic list of harness artifacts to observe."""

    paths: list[Path] = []
    for globs in DEFAULT_ARTIFACT_GLOBS.values():
        bucket: list[Path] = []
        for pattern in globs:
            bucket.extend(path for path in root.glob(pattern) if path.is_file())
        paths.extend(sorted(bucket)[:max_files_per_kind])
    return _dedupe_paths(paths)


def events_from_harness_artifacts(
    *,
    root: Path,
    artifact_paths: Iterable[Path],
    max_events_per_file: int = 12,
) -> list[dict[str, Any]]:
    """Convert judge/meta/log artifacts into manifest event dicts."""

    events: list[dict[str, Any]] = []
    for path in _dedupe_paths(artifact_paths):
        if not path.exists():
            continue
        if path.suffix == ".log":
            log_events = events_from_run_logs(
                root=root,
                log_paths=[path],
                max_events_per_file=max_events_per_file,
            )
            for event in log_events:
                event.setdefault("metadata", {})["artifact_kind"] = "run_log"
            events.extend(log_events)
            continue
        if _is_judgment_json(path):
            events.extend(_events_from_judgment_json(root, path, max_events_per_file=max_events_per_file))
            continue
        if _is_answer_meta_json(path):
            events.append(_event_from_answer_meta_json(root, path))
            continue
        events.append(_event_from_generic_artifact(root, path))
    return events


def _events_from_judgment_json(root: Path, path: Path, *, max_events_per_file: int) -> list[dict[str, Any]]:
    rel = _display_path(root, path)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return [_event_from_generic_artifact(root, path)]
    events = []
    row_count = len(payload)
    for row_index, (problem_id, row) in enumerate(sorted(payload.items())):
        if row_index >= max_events_per_file:
            break
        row = row if isinstance(row, dict) else {"value": row}
        winner = row.get("winner")
        score_a = row.get("score_a")
        score_b = row.get("score_b")
        events.append({
            "event_type": "judge_call",
            "problem_id": f"artifact::{rel}::{problem_id}",
            "component": "harness_observer",
            "assumption": "Judgment artifacts are evidence-bearing evaluator calls.",
            "why_selected": "Existing pairwise judgment cache rows should be visible to manifest coverage.",
            "expected_effect": "Recover one judged row without importing full answer text.",
            "observed_effect": f"winner={winner}; score_a={score_a}; score_b={score_b}",
            "verifier": "artifact_judgment_parser",
            "artifacts": {
                "path": rel,
                "problem_id": problem_id,
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
                "a_was": row.get("a_was"),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
                "reasoning_preview": _preview(row.get("reasoning")),
                "row_index": row_index,
                "row_count": row_count,
            },
            "metadata": {
                "artifact_kind": "judgment_json",
                "row_index": row_index,
                "row_count": row_count,
            },
        })
    return events


def _event_from_answer_meta_json(root: Path, path: Path) -> dict[str, Any]:
    rel = _display_path(root, path)
    payload = _load_json(path)
    rows = payload if isinstance(payload, dict) else {}
    frame_counts = Counter()
    route_counts = Counter()
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        if row.get("frame"):
            frame_counts[str(row["frame"])] += 1
        if row.get("bypass_route"):
            route_counts[str(row["bypass_route"])] += 1
    return {
        "event_type": "llm_call",
        "problem_id": f"artifact::{rel}",
        "component": "harness_observer",
        "assumption": "Answer meta artifacts summarize model execution decisions.",
        "why_selected": "Answer meta caches expose framing, rewrite, and bypass decisions that should be auditable.",
        "expected_effect": "Record bounded metadata coverage without storing full prompts or answers.",
        "observed_effect": f"rows={len(rows)}; frames={dict(frame_counts)}",
        "verifier": "artifact_answer_meta_parser",
        "artifacts": {
            "path": rel,
            "row_count": len(rows),
            "sample_problem_ids": list(sorted(rows))[:5],
            "frame_counts": dict(frame_counts),
            "bypass_route_counts": dict(route_counts),
        },
        "metadata": {"artifact_kind": "answer_meta_json"},
    }


def _event_from_generic_artifact(root: Path, path: Path) -> dict[str, Any]:
    rel = _display_path(root, path)
    payload = None
    try:
        payload = _load_json(path)
    except Exception:
        payload = None
    return {
        "event_type": "tool_use",
        "problem_id": f"artifact::{rel}",
        "component": "harness_observer",
        "assumption": "Harness artifacts should be represented in manifest coverage.",
        "why_selected": "Unclassified artifact files still provide execution evidence.",
        "expected_effect": "Record path, size, and top-level shape for audit.",
        "observed_effect": f"bytes={path.stat().st_size}",
        "verifier": "artifact_shape_parser",
        "artifacts": {
            "path": rel,
            "bytes": path.stat().st_size,
            "top_level_type": type(payload).__name__ if payload is not None else None,
            "top_level_count": len(payload) if hasattr(payload, "__len__") else None,
        },
        "metadata": {"artifact_kind": "generic_artifact"},
    }


def _trial_artifact_paths(store: JsonlGraphStore) -> set[str]:
    paths = set()
    for trial in store.trials.values():
        artifacts = trial.artifacts or {}
        path = artifacts.get("path")
        if path:
            paths.add(str(path))
    return paths


def _event_artifact_paths(events: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(event.get("artifacts", {}).get("path"))
        for event in events
        if event.get("artifacts", {}).get("path")
    }


def _has_secret_probe(payload: dict) -> bool:
    text = json.dumps(redact_secrets(payload), ensure_ascii=False)
    probes = [
        "redaction" + "-probe-",
        "secret" + "_token=",
        "api" + "_key=",
    ]
    return any(probe in text for probe in probes)


def _is_judgment_json(path: Path) -> bool:
    return path.suffix == ".json" and path.parent.name == "judgments"


def _is_answer_meta_json(path: Path) -> bool:
    return path.suffix == ".json" and path.name.endswith("_meta.json") and path.parent.name == "answers"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--artifacts", nargs="*", default=None)
    ap.add_argument("--max-files-per-kind", type=int, default=6)
    ap.add_argument("--max-events-per-file", type=int, default=12)
    ap.add_argument("--include-covered", action="store_true",
                    help="also write events for artifact paths that already have graph manifest coverage")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_harness_observer_payload(
        root=root,
        graph_dir=_resolve(root, args.graph_dir),
        eval_id=args.eval_id,
        artifact_paths=[_resolve(root, p) for p in args.artifacts] if args.artifacts else None,
        max_files_per_kind=args.max_files_per_kind,
        max_events_per_file=args.max_events_per_file,
        include_covered=args.include_covered,
        writeback=args.writeback,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
