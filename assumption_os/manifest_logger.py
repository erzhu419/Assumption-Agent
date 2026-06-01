"""TrialManifest logging for agent component events.

The reconstruction notes require LLM calls, retrievals, judge calls, and
tool-use to be logged in the same falsifiable format as assumption updates.
This module keeps that contract small: callers provide event metadata, the
logger redacts secret-looking values, creates a TrialManifest, and optionally
writes it to the graph store.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .graph_memory import JsonlGraphStore
from .schema import ResidualType, TrialManifest, TrialStatus, stable_id


SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|(?i:api[_-]?key)\s*[:=]\s*[^,\s}\]]+|(?i:secret[_-]?token)\s*[:=]\s*[^,\s}\]]+)"
)

COMPONENT_EVENTS = {
    "llm_call",
    "retrieval",
    "judge_call",
    "tool_use",
    "simulator_rollout",
    "recursive_daemon_iteration",
}

JUDGE_HEADER_RE = re.compile(r"^=== JUDGE\s+(?P<label>\S+)\s+(?P<candidate>\S+)\s+vs\s+(?P<baseline>\S+)\s+rows=(?P<rows>\d+)\s*===")
DONE_RE = re.compile(r"^=== DONE\s+(?P<kind>\S+)\s+(?P<label>\S+).*?returncode=(?P<returncode>-?\d+).*?elapsed=(?P<elapsed>[0-9.]+)s\s*===")
MODEL_RE = re.compile(r"LLM provider:\s*(?P<provider>[^,]+),\s*model:\s*(?P<model>\S+)")
PLANNED_CALLS_RE = re.compile(r"Total LLM calls planned:\s*(?P<calls>\d+)")


def make_component_manifest(
    *,
    event_type: str,
    problem_id: str,
    component: str,
    assumption: str,
    why_selected: str,
    expected_effect: str,
    assumption_ids: Iterable[str] | None = None,
    verifier: str | None = None,
    verification_plan: str | None = None,
    rollback_condition: str | None = None,
    cost: float = 0.0,
    status: TrialStatus | str = TrialStatus.OBSERVED,
    observed_effect: str | None = None,
    residual: str | None = None,
    residual_type: ResidualType | str | None = None,
    artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    parent_trial_id: str | None = None,
    eval_id: str | None = None,
    trial_id: str | None = None,
) -> TrialManifest:
    """Create a redacted TrialManifest for one component event."""

    normalized_event = event_type if event_type in COMPONENT_EVENTS else f"component_{event_type}"
    clean_artifacts = redact_secrets(artifacts or {})
    clean_metadata = redact_secrets({
        **(metadata or {}),
        "event_type": event_type,
        "normalized_event_type": normalized_event,
    })
    if eval_id:
        clean_metadata["eval_id"] = eval_id
    manifest = TrialManifest(
        problem_id=problem_id,
        action_type=normalized_event,
        component=component,
        assumption=assumption,
        why_selected=why_selected,
        expected_effect=expected_effect,
        assumption_ids=list(assumption_ids or []),
        predicted_regressions=list((metadata or {}).get("predicted_regressions", [])),
        verifier=verifier,
        verification_plan=verification_plan,
        rollback_condition=rollback_condition,
        cost=cost,
        status=status,
        observed_effect=observed_effect,
        residual=residual,
        residual_type=residual_type,
        artifacts=clean_artifacts,
        metadata=clean_metadata,
        parent_trial_id=parent_trial_id,
        trial_id=trial_id or stable_id(
            "trial",
            eval_id or "",
            problem_id,
            normalized_event,
            component,
            assumption,
            json.dumps(clean_metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    return manifest


def log_component_manifest(
    store: JsonlGraphStore,
    *,
    persist: bool = True,
    **kwargs: Any,
) -> TrialManifest:
    """Create and append a component manifest to a graph store."""

    manifest = make_component_manifest(**kwargs)
    store.append_trial(manifest)
    if persist:
        store.flush()
    return manifest


def build_component_manifest_payload(
    *,
    eval_id: str,
    events: Iterable[dict[str, Any]],
    store: JsonlGraphStore | None = None,
    writeback: bool = False,
) -> dict:
    """Normalize many event dicts into TrialManifest records."""

    manifests = []
    for event in events:
        manifest = make_component_manifest(eval_id=eval_id, **event)
        manifests.append(manifest)
        if store is not None and writeback:
            store.append_trial(manifest)
    if store is not None and writeback:
        store.flush()
    return {
        "eval_id": eval_id,
        "writeback": writeback,
        "event_count": len(manifests),
        "event_counts": dict(Counter(m.action_type for m in manifests)),
        "component_counts": dict(Counter(m.component or "" for m in manifests)),
        "trial_ids": [m.trial_id for m in manifests],
        "manifests": [m.to_dict() for m in manifests],
    }


def events_from_run_logs(
    *,
    root: Path,
    log_paths: Iterable[Path],
    max_events_per_file: int = 25,
) -> list[dict[str, Any]]:
    """Convert existing run logs into component-manifest event dicts."""

    events = []
    for path in log_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = _display_path(root, path)
        parsed = _events_from_judge_log(root=root, path=path, text=text, max_events=max_events_per_file)
        if parsed:
            events.extend(parsed)
            continue
        events.append(_event_from_generic_log(rel, text))
    return events


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-looking strings before writing manifests."""

    if isinstance(value, str):
        return SECRET_RE.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, tuple):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        return {str(k): redact_secrets(v) for k, v in value.items()}
    return value


def _events_from_judge_log(*, root: Path, path: Path, text: str, max_events: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    rel = _display_path(root, path)
    provider = None
    model = None
    events = []
    current: dict[str, Any] | None = None
    for line in lines:
        model_match = MODEL_RE.search(line)
        if model_match:
            provider = model_match.group("provider").strip()
            model = model_match.group("model").strip()
            if current is not None:
                current["artifacts"]["provider"] = provider
                current["artifacts"]["model"] = model
            continue
        header = JUDGE_HEADER_RE.match(line)
        if header:
            if current is not None:
                events.append(current)
            data = header.groupdict()
            current = {
                "event_type": "judge_call",
                "problem_id": f"log::{rel}::{data['label']}",
                "component": "run_log_ingest",
                "assumption": "Existing judge run logs are evidence-bearing component calls.",
                "why_selected": "Historical judge logs should be available to the assumption manifest layer.",
                "expected_effect": "Recover proposal, candidate, baseline, row count, model, and return status from the run log.",
                "observed_effect": "judge run started; completion status parsed if present",
                "verifier": "run_log_parser",
                "artifacts": {
                    "path": rel,
                    "header": line,
                    "proposal_or_label": data["label"],
                    "candidate_variant": data["candidate"],
                    "baseline_variant": data["baseline"],
                    "rows": int(data["rows"]),
                    "provider": provider,
                    "model": model,
                },
                "metadata": {"log_kind": "judge_run"},
            }
            if len(events) >= max_events:
                break
            continue
        done = DONE_RE.match(line)
        if done and current is not None:
            data = done.groupdict()
            current["observed_effect"] = f"returncode={data['returncode']}; elapsed_sec={data['elapsed']}"
            current["artifacts"]["returncode"] = int(data["returncode"])
            current["artifacts"]["elapsed_sec"] = float(data["elapsed"])
    if current is not None and len(events) < max_events:
        events.append(current)
    return events[:max_events]


def _event_from_generic_log(path: str, text: str) -> dict[str, Any]:
    lines = text.splitlines()
    planned_calls = None
    for line in lines:
        match = PLANNED_CALLS_RE.search(line)
        if match:
            planned_calls = int(match.group("calls"))
            break
    event_type = "llm_call" if planned_calls else "tool_use"
    return {
        "event_type": event_type,
        "problem_id": f"log::{path}",
        "component": "run_log_ingest",
        "assumption": "Existing run logs should be summarized as manifest evidence.",
        "why_selected": "Historical execution logs contain model-call or tool-use evidence not yet represented in the graph.",
        "expected_effect": "Record log metadata and a redacted tail for audit without importing the full raw log.",
        "observed_effect": f"lines={len(lines)}; planned_calls={planned_calls}",
        "verifier": "run_log_parser",
        "artifacts": {
            "path": path,
            "line_count": len(lines),
            "planned_calls": planned_calls,
            "tail": "\n".join(lines[-12:]),
        },
        "metadata": {"log_kind": "generic_run"},
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    if isinstance(loaded, list):
        return loaded
    return loaded.get("events", [])


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
    ap.add_argument("--events", default=None, help="JSON/JSONL file of component event dicts")
    ap.add_argument("--logs", nargs="*", default=None, help="Optional run logs to convert into manifest events")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    store = JsonlGraphStore(_resolve(root, args.graph_dir))
    events = _load_events(_resolve(root, args.events)) if args.events else []
    if args.logs:
        events.extend(events_from_run_logs(
            root=root,
            log_paths=[_resolve(root, p) for p in args.logs],
        ))
    payload = build_component_manifest_payload(
        eval_id=args.eval_id,
        events=events,
        store=store,
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
