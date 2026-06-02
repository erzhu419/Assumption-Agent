"""Artifact adapter for preflight fresh-ablation queues.

Fresh-ablation queue rows carry shell command hints.  The daemon can execute
those commands, but the command usually writes answer caches rather than the
candidate-acceptance judgment JSON directly.  This module makes that artifact
layer explicit: parse the command, locate answer/judgment caches, build the
cached judge command, and expose existing judgments as ``JudgmentSet`` objects.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Iterable

from .recursive_executor import JudgmentSet


DEFAULT_CACHE_ROOT = Path("phase two/analysis/cache")
DEFAULT_JUDGE_SCRIPT = Path("phase one/scripts/validation/cached_framework.py")


def build_queue_artifact_eval_payload(
    *,
    root: Path,
    preflight_payload: dict,
    eval_id: str,
    baseline_variant: str,
    proposal_ids: Iterable[str] | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    judge_script: Path = DEFAULT_JUDGE_SCRIPT,
) -> dict:
    """Inspect queue commands and locate answer/judgment artifacts.

    The function does not call an LLM.  It builds the exact cached-judge command
    and discovers judgment files if a prior judge run has already written them.
    """

    selected = set(proposal_ids or [])
    plans = []
    for row in _ready_rows(preflight_payload):
        proposal_id = str(row.get("proposal_id") or "")
        if selected and proposal_id not in selected:
            continue
        plans.append(_build_plan(
            root=root,
            row=row,
            baseline_variant=baseline_variant,
            cache_root=cache_root,
            judge_script=judge_script,
        ))
    ready_for_acceptance = [plan for plan in plans if plan["existing_judgment_paths"]]
    return {
        "eval_id": eval_id,
        "source_preflight_eval_id": preflight_payload.get("eval_id"),
        "baseline_variant": baseline_variant,
        "cache_root": _display_path(root, _resolve(root, cache_root)),
        "judge_script": _display_path(root, _resolve(root, judge_script)),
        "plan_count": len(plans),
        "parsed_command_count": sum(1 for plan in plans if plan["command_parse_ok"]),
        "sample_found_count": sum(1 for plan in plans if plan["sample_exists"]),
        "candidate_answer_ready_count": sum(1 for plan in plans if plan["candidate_answer_ready"]),
        "baseline_answer_ready_count": sum(1 for plan in plans if plan["baseline_answer_ready"]),
        "judge_command_count": sum(1 for plan in plans if plan["judge_command"]),
        "existing_judgment_plan_count": len(ready_for_acceptance),
        "judgment_set_count": len(ready_for_acceptance),
        "trigger_judgment_count": sum(plan["judged_trigger_count"] for plan in plans),
        "control_judgment_count": sum(plan["judged_control_count"] for plan in plans),
        "missing_artifact_counts": _missing_counts(plans),
        "plans": plans,
    }


def judgment_sets_from_artifact_eval(payload: dict) -> list[JudgmentSet]:
    """Convert discovered judgment artifacts into recursive executor inputs."""

    out: list[JudgmentSet] = []
    baseline_variant = payload.get("baseline_variant", "")
    for plan in payload.get("plans", []):
        paths = [Path(p) for p in plan.get("existing_judgment_paths", [])]
        if not paths:
            continue
        proposal_id = plan.get("proposal_id")
        candidate_variant = plan.get("candidate_variant")
        if not proposal_id or not candidate_variant or not baseline_variant:
            continue
        out.append(JudgmentSet(
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            judgment_paths=paths,
            proposal_ids=[proposal_id],
        ))
    return out


def _build_plan(
    *,
    root: Path,
    row: dict,
    baseline_variant: str,
    cache_root: Path,
    judge_script: Path,
) -> dict:
    proposal_id = str(row.get("proposal_id") or "")
    command = str(row.get("command_hint") or "")
    tokens = _split_command(command)
    candidate_variant = _option(tokens, "--variant") or _variant_from_proposal_id(proposal_id)
    sample_path = _resolve(root, _option(tokens, "--sample")) if _option(tokens, "--sample") else None
    cache_abs = _resolve(root, cache_root)
    candidate_answer_path = cache_abs / "answers" / f"{candidate_variant}_answers.json"
    baseline_answer_path = cache_abs / "answers" / f"{baseline_variant}_answers.json"
    candidate_judgment_path = cache_abs / "judgments" / f"{candidate_variant}_vs_{baseline_variant}.json"
    reverse_judgment_path = cache_abs / "judgments" / f"{baseline_variant}_vs_{candidate_variant}.json"

    sample_problem_ids = _sample_problem_ids(sample_path)
    trigger_ids = list(row.get("trigger_problem_ids") or row.get("active_trigger_problem_ids") or [])
    control_ids = list(row.get("control_problem_ids") or [])
    probe_ids = _dedupe([*trigger_ids, *control_ids])
    if not probe_ids:
        probe_ids = sample_problem_ids

    candidate_answers = _load_mapping(candidate_answer_path)
    baseline_answers = _load_mapping(baseline_answer_path)
    existing_judgment_paths = [
        path for path in [candidate_judgment_path, reverse_judgment_path]
        if path.exists()
    ]
    judgments = _load_judgments(existing_judgment_paths)
    judged_trigger_count = sum(1 for pid in trigger_ids if pid in judgments)
    judged_control_count = sum(1 for pid in control_ids if pid in judgments)

    judge_command = ""
    if sample_path is not None and candidate_variant and baseline_variant:
        judge_command = _judge_command(
            root=root,
            judge_script=_resolve(root, judge_script),
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            sample_path=sample_path,
        )

    missing = []
    if not tokens:
        missing.append("command_parse")
    if sample_path is None or not sample_path.exists():
        missing.append("sample")
    if not _answers_cover(candidate_answers, probe_ids):
        missing.append("candidate_answers")
    if not _answers_cover(baseline_answers, probe_ids):
        missing.append("baseline_answers")
    if not existing_judgment_paths:
        missing.append("judgments")

    return {
        "proposal_id": proposal_id,
        "candidate_variant": candidate_variant,
        "baseline_variant": baseline_variant,
        "command_parse_ok": bool(tokens and candidate_variant),
        "command_hint": command,
        "sample_path": str(sample_path) if sample_path else "",
        "sample_exists": bool(sample_path and sample_path.exists()),
        "sample_problem_count": len(sample_problem_ids),
        "trigger_problem_count": len(trigger_ids),
        "control_problem_count": len(control_ids),
        "candidate_answer_path": str(candidate_answer_path),
        "baseline_answer_path": str(baseline_answer_path),
        "candidate_answer_coverage": _coverage(candidate_answers, probe_ids),
        "baseline_answer_coverage": _coverage(baseline_answers, probe_ids),
        "candidate_answer_ready": _answers_cover(candidate_answers, probe_ids),
        "baseline_answer_ready": _answers_cover(baseline_answers, probe_ids),
        "expected_judgment_path": str(candidate_judgment_path),
        "reverse_judgment_path": str(reverse_judgment_path),
        "existing_judgment_paths": [str(path) for path in existing_judgment_paths],
        "judged_trigger_count": judged_trigger_count,
        "judged_control_count": judged_control_count,
        "judge_command": judge_command,
        "missing_artifacts": missing,
    }


def _ready_rows(preflight_payload: dict) -> list[dict]:
    return [
        row
        for row in preflight_payload.get("summaries", [])
        if row.get("readiness") == "ready_for_fresh_ablation" and row.get("command_hint")
    ]


def _split_command(command: str) -> list[str]:
    if not command:
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _option(tokens: list[str], name: str) -> str:
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return ""


def _variant_from_proposal_id(proposal_id: str) -> str:
    if proposal_id.startswith("prop_"):
        return "proposal_" + proposal_id.removeprefix("prop_")
    return proposal_id


def _sample_problem_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    out = []
    for row in rows:
        if isinstance(row, dict) and row.get("problem_id"):
            out.append(str(row["problem_id"]))
    return _dedupe(out)


def _load_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_judgments(paths: Iterable[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in paths:
        for pid, row in _load_mapping(path).items():
            if isinstance(row, dict):
                out.setdefault(str(pid), []).append(row)
    return out


def _answers_cover(answers: dict, problem_ids: list[str]) -> bool:
    return bool(problem_ids) and all(bool(answers.get(pid)) for pid in problem_ids)


def _coverage(answers: dict, problem_ids: list[str]) -> dict:
    have = sum(1 for pid in problem_ids if bool(answers.get(pid)))
    total = len(problem_ids)
    return {
        "covered": have,
        "total": total,
        "rate": round(have / total, 4) if total else 0.0,
    }


def _judge_command(
    *,
    root: Path,
    judge_script: Path,
    candidate_variant: str,
    baseline_variant: str,
    sample_path: Path,
) -> str:
    return " ".join([
        "python3",
        shlex.quote(_display_path(root, judge_script)),
        "--judge",
        shlex.quote(candidate_variant),
        shlex.quote(baseline_variant),
        "--sample",
        shlex.quote(str(sample_path)),
    ])


def _missing_counts(plans: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        for name in plan.get("missing_artifacts", []):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _dedupe(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve(root: Path, path: str | Path | None) -> Path:
    if path is None:
        return root
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
    ap.add_argument("--preflight-payload", required=True)
    ap.add_argument("--baseline-variant", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--proposal-ids", nargs="*", default=None)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    preflight_path = _resolve(root, args.preflight_payload)
    payload = build_queue_artifact_eval_payload(
        root=root,
        preflight_payload=json.loads(preflight_path.read_text(encoding="utf-8")),
        baseline_variant=args.baseline_variant,
        eval_id=args.eval_id,
        proposal_ids=args.proposal_ids,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
