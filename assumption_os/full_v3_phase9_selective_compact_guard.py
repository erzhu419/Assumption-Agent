"""Phase9 heldout validation for selective compact-frame guard.

The compact frame guard beat V1 on the first 31 active cases but was slightly
below original V3.  This module tests a selective-retention profile:

- use compact frame guard only for route tags whose training residuals show
  V1-regression lift without catastrophic V3 regression;
- otherwise keep original V3.

It validates the profile on the remaining active cases from the same route plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .full_v3_phase9_frame_morphism_repair import FRAME_MORPHISM_REPAIR_PROMPT
from .full_v3_phase9_v1_live_regression import (
    DEFAULT_EVAL_ID as PHASE9_BASE_EVAL_ID,
    DEFAULT_RUN_DIR,
    _display,
    _judgment_valid,
    _load_dotenv_if_present,
    _load_json,
    _prompt_for_arm,
    _resolve,
)
from .structural_live_ablation import (
    PAIRWISE_JUDGE_PROMPT,
    _call_with_retry,
    _parse_judge_json,
    _requests_client_for_alias,
    _winner_to_arm,
    _write_jsonl,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase9_selective_compact_guard_heldout_20260611.json"
DEFAULT_EVAL_ID = "full_v3_phase9_selective_compact_guard_heldout_20260611"
COMPACT_ARM = "v3_selective_compact_guard"
V3_ARM = "v3_full"
V1_ARM = "v1_case_reflection_kernel"
SELECTED_TAGS = {"S14", "S19"}


def build_full_v3_phase9_selective_compact_guard_payload(
    *,
    root: Path,
    eval_id: str = DEFAULT_EVAL_ID,
    phase9_eval_id: str = PHASE9_BASE_EVAL_ID,
    execution_mode: str = "dry_run",
    solver_model: str = "gpt_mini",
    judge_model: str = "gpt_mini",
    solve_workers: int = 8,
    judge_workers: int = 4,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    _load_dotenv_if_present(root)
    run_dir = _resolve(root, run_dir or DEFAULT_RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_compact = _load_json(root / PAPER_DIR / "full_v3_phase9_compact_frame_guard_20260611.json")
    phase9 = _load_json(root / PAPER_DIR / f"{phase9_eval_id}.json")
    all_cases = _load_all_route_cases(root=root, phase9_eval_id=phase9_eval_id)
    train_ids = {case["problem_id"] for case in (phase9.get("route_plan") or {}).get("active_cases", [])}
    heldout_cases = [case for case in all_cases if case["problem_id"] not in train_ids]
    selected_cases = [case for case in heldout_cases if _use_compact(case)]
    paths = _paths(run_dir, eval_id)
    env_ready = _env_ready(solver_model=solver_model, judge_model=judge_model)

    if execution_mode == "execute" and not env_ready:
        run_status = "blocked_env_missing"
        answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
        judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
    elif execution_mode == "execute":
        answers, judgments = _execute(
            cases=heldout_cases,
            selected_cases=selected_cases,
            paths=paths,
            solver_model=solver_model,
            judge_model=judge_model,
            solve_workers=solve_workers,
            judge_workers=judge_workers,
        )
        run_status = "execute_complete"
    elif execution_mode == "summarize":
        answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
        judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
        run_status = "summarize_complete" if judgments else "summarize_missing_judgments"
    else:
        answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
        judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
        run_status = "dry_run_complete"

    pair_summaries = _pair_summaries(cases=heldout_cases, selected_cases=selected_cases, judgments=judgments)
    policy = _policy_summary(cases=heldout_cases, selected_cases=selected_cases, judgments=judgments)
    metrics = _metrics(
        cases=heldout_cases,
        selected_cases=selected_cases,
        answers=answers,
        pair_summaries=pair_summaries,
        policy=policy,
        phase9=phase9,
        train_compact=train_compact,
        env_ready=env_ready,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
    )
    gates = _gates(metrics=metrics, execution_mode=execution_mode, run_status=run_status)
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase9_selective_compact_guard_heldout",
        "performance_validation": True,
        "execution_mode": execution_mode,
        "run_status": run_status,
        "validation_scope": (
            "Heldout validation of a selective compact-frame guard profile.  The policy uses compact guard only "
            "on tags S14/S19 and keeps original V3 elsewhere."
        ),
        "selector": {
            "selected_tags": sorted(SELECTED_TAGS),
            "selection_rule": (
                "Activate compact guard on route tags whose first-slice residuals had V1-regression lift and "
                "acceptable V3 non-inferiority; otherwise use original V3."
            ),
            "train_source": "full_v3_phase9_compact_frame_guard_20260611",
        },
        "heldout_case_counts": {
            "all_route_cases": len(all_cases),
            "train_cases": len(train_ids),
            "heldout_cases": len(heldout_cases),
            "selected_compact_cases": len(selected_cases),
        },
        "pair_summaries": pair_summaries,
        "policy_summary": policy,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_run_paths": {
            "answers_path": _display(root, paths["answers_path"]),
            "judgments_path": _display(root, paths["judgments_path"]),
            "forensic_path": _display(root, paths["forensic_path"]),
            "compact_payload_contains_prompts_answers": False,
        },
        "pass": all(gates.values()),
        "interpretation": _interpretation(metrics),
    }


def _execute(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    paths: dict[str, Path],
    solver_model: str,
    judge_model: str,
    solve_workers: int,
    judge_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
    judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
    solver = _requests_client_for_alias(solver_model)
    judge = _requests_client_for_alias(judge_model)
    _solve_missing(
        cases=cases,
        selected_cases=selected_cases,
        answers=answers,
        answers_path=paths["answers_path"],
        forensic_path=paths["forensic_path"],
        solver=solver,
        solver_model=solver_model,
        max_workers=solve_workers,
    )
    _judge_missing(
        cases=cases,
        selected_cases=selected_cases,
        answers=answers,
        judgments=judgments,
        judgments_path=paths["judgments_path"],
        forensic_path=paths["forensic_path"],
        judge=judge,
        judge_model=judge_model,
        max_workers=judge_workers,
    )
    return answers, judgments


def _solve_missing(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    answers_path: Path,
    forensic_path: Path,
    solver: Any,
    solver_model: str,
    max_workers: int,
) -> None:
    selected_ids = {case["problem_id"] for case in selected_cases}
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        answers.setdefault(pid, {})
        for arm in [V1_ARM, V3_ARM]:
            if not answers[pid].get(arm):
                jobs.append((case, arm))
        if pid in selected_ids and not answers[pid].get(COMPACT_ARM):
            jobs.append((case, COMPACT_ARM))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_solve_one, case, arm, solver, solver_model, forensic_path) for case, arm in jobs]
        for fut in as_completed(futures):
            pid, arm, text = fut.result()
            answers.setdefault(pid, {})[arm] = text
            completed += 1
            if completed % 10 == 0:
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 selective solve] {completed}/{len(jobs)}", flush=True)
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_one(case: dict[str, Any], arm: str, solver: Any, solver_model: str, forensic_path: Path) -> tuple[str, str, str]:
    if arm == COMPACT_ARM:
        prompt = FRAME_MORPHISM_REPAIR_PROMPT.format(problem=case["description"], context=case["structural_context"])
    else:
        prompt = _prompt_for_arm(case, arm)
    t0 = time.time()
    response = _call_with_retry(solver, prompt, max_tokens=1100, temperature=0.3)
    text = response.get("text", "").strip()
    _write_jsonl(forensic_path, {
        "role": "solver",
        "eval_kind": "phase9_selective_compact_guard",
        "problem_id": case["problem_id"],
        "arm": arm,
        "model_alias": solver_model,
        "model": response.get("model", ""),
        "prompt_len": len(prompt),
        "answer_len": len(text),
        "elapsed": time.time() - t0,
        "prompt": prompt,
        "answer": text,
        "error": response.get("error", ""),
    })
    return case["problem_id"], arm, text


def _judge_missing(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    judgments: dict[str, Any],
    judgments_path: Path,
    forensic_path: Path,
    judge: Any,
    judge_model: str,
    max_workers: int,
) -> None:
    selected_ids = {case["problem_id"] for case in selected_cases}
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        judgments.setdefault(pid, {})
        if answers.get(pid, {}).get(V3_ARM) and answers.get(pid, {}).get(V1_ARM):
            pair = f"{V3_ARM}_vs_{V1_ARM}"
            if not _judgment_valid(judgments[pid].get(pair)):
                jobs.append((case, V3_ARM, V1_ARM, pair))
        if pid in selected_ids and answers.get(pid, {}).get(COMPACT_ARM):
            for reference_arm in [V1_ARM, V3_ARM]:
                if not answers.get(pid, {}).get(reference_arm):
                    continue
                pair = f"{COMPACT_ARM}_vs_{reference_arm}"
                if not _judgment_valid(judgments[pid].get(pair)):
                    jobs.append((case, COMPACT_ARM, reference_arm, pair))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_judge_one, case, left_arm, right_arm, pair, answers, judge, judge_model, forensic_path)
            for case, left_arm, right_arm, pair in jobs
        ]
        for fut in as_completed(futures):
            pid, pair, judgment = fut.result()
            judgments.setdefault(pid, {})[pair] = judgment
            completed += 1
            if completed % 10 == 0:
                judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 selective judge] {completed}/{len(jobs)}", flush=True)
    judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_one(
    case: dict[str, Any],
    left_arm: str,
    right_arm: str,
    pair: str,
    answers: dict[str, Any],
    judge: Any,
    judge_model: str,
    forensic_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    pid = case["problem_id"]
    swap = int(hashlib.sha1(f"{pid}:{pair}:selective".encode()).hexdigest(), 16) % 2 == 1
    a_arm, b_arm = (right_arm, left_arm) if swap else (left_arm, right_arm)
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        problem=case["description"][:3000],
        reference=json.dumps(case.get("reference_answer", {}), ensure_ascii=False)[:3000],
        answer_a=answers[pid][a_arm][:3500],
        answer_b=answers[pid][b_arm][:3500],
    )
    t0 = time.time()
    response = _call_with_retry(judge, prompt, max_tokens=260, temperature=0.0)
    raw = response.get("text", "").strip()
    parsed = _parse_judge_json(raw)
    winner = _winner_to_arm(parsed.get("winner", "tie"), a_arm=a_arm, b_arm=b_arm)
    valid = bool(raw) and not response.get("error") and not (winner == "tie" and parsed.get("reason") == "judge_json_parse_failed")
    judgment = {
        "pair": pair,
        "winner": winner,
        "raw_winner": parsed.get("winner", "tie"),
        "a_arm": a_arm,
        "b_arm": b_arm,
        "reason": parsed.get("reason", ""),
        "model_alias": judge_model,
        "model": response.get("model", ""),
        "valid": valid,
        "error": response.get("error", ""),
    }
    _write_jsonl(forensic_path, {
        "role": "judge",
        "eval_kind": "phase9_selective_compact_guard",
        "problem_id": pid,
        "pair": pair,
        "model_alias": judge_model,
        "model": response.get("model", ""),
        "prompt_len": len(prompt),
        "raw": raw,
        "judgment": judgment,
        "elapsed": time.time() - t0,
        "error": response.get("error", ""),
    })
    return pid, pair, judgment


def _pair_summaries(*, cases: list[dict[str, Any]], selected_cases: list[dict[str, Any]], judgments: dict[str, Any]) -> dict[str, Any]:
    pairs = [f"{V3_ARM}_vs_{V1_ARM}", f"{COMPACT_ARM}_vs_{V1_ARM}", f"{COMPACT_ARM}_vs_{V3_ARM}"]
    selected_ids = {case["problem_id"] for case in selected_cases}
    out = {}
    for pair in pairs:
        rows = []
        for case in cases:
            if pair.startswith(COMPACT_ARM) and case["problem_id"] not in selected_ids:
                continue
            judgment = judgments.get(case["problem_id"], {}).get(pair)
            if not _judgment_valid(judgment):
                continue
            positive_arm = pair.split("_vs_", 1)[0]
            winner = judgment.get("winner")
            outcome = "win" if winner == positive_arm else "tie" if winner == "tie" else "loss"
            rows.append(_row(case, outcome, winner, judgment.get("reason", "")))
        out[pair] = _stats(pair, rows)
    return out


def _policy_summary(*, cases: list[dict[str, Any]], selected_cases: list[dict[str, Any]], judgments: dict[str, Any]) -> dict[str, Any]:
    selected_ids = {case["problem_id"] for case in selected_cases}
    vs_v1_rows = []
    vs_v3_rows = []
    for case in cases:
        pid = case["problem_id"]
        if pid in selected_ids:
            v1_judgment = judgments.get(pid, {}).get(f"{COMPACT_ARM}_vs_{V1_ARM}")
            v3_judgment = judgments.get(pid, {}).get(f"{COMPACT_ARM}_vs_{V3_ARM}")
            if _judgment_valid(v1_judgment):
                outcome = _outcome(v1_judgment.get("winner"), positive_arm=COMPACT_ARM)
                vs_v1_rows.append(_row(case, outcome, v1_judgment.get("winner"), v1_judgment.get("reason", "")))
            if _judgment_valid(v3_judgment):
                outcome = _outcome(v3_judgment.get("winner"), positive_arm=COMPACT_ARM)
                vs_v3_rows.append(_row(case, outcome, v3_judgment.get("winner"), v3_judgment.get("reason", "")))
        else:
            v1_judgment = judgments.get(pid, {}).get(f"{V3_ARM}_vs_{V1_ARM}")
            if _judgment_valid(v1_judgment):
                outcome = _outcome(v1_judgment.get("winner"), positive_arm=V3_ARM)
                vs_v1_rows.append(_row(case, outcome, v1_judgment.get("winner"), v1_judgment.get("reason", "")))
            vs_v3_rows.append(_row(case, "tie", "tie", "policy kept original V3"))
    return {
        "policy_vs_v1": _stats("policy_vs_v1", vs_v1_rows),
        "policy_vs_original_v3": _stats("policy_vs_original_v3", vs_v3_rows),
    }


def _stats(pair: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [_value(row["outcome"]) for row in rows]
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[str(row.get("route_strategy_tag"))].append(row)
        by_pattern[str(row.get("pattern_id"))].append(row)
    return {
        "pair": pair,
        "n": len(rows),
        "outcomes": dict(counts),
        "utility": round(sum(values) / max(1, len(values)), 4) if rows else 0.0,
        "margin_over_tie": round((sum(values) / max(1, len(values))) - 0.5, 4) if rows else 0.0,
        "win_rate": round(counts["win"] / max(1, len(rows)), 4) if rows else 0.0,
        "loss_rate": round(counts["loss"] / max(1, len(rows)), 4) if rows else 0.0,
        "by_tag": {tag: _group(group) for tag, group in sorted(by_tag.items())},
        "by_pattern": {pattern: _group(group) for pattern, group in sorted(by_pattern.items())},
        "rows": rows,
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [_value(row["outcome"]) for row in rows]
    return {"n": len(rows), "outcomes": dict(counts), "utility": round(sum(values) / max(1, len(values)), 4)}


def _row(case: dict[str, Any], outcome: str, winner: str, reason: str) -> dict[str, Any]:
    return {
        "problem_id": case["problem_id"],
        "domain": case.get("domain"),
        "pattern_id": case.get("top_pattern_id"),
        "route_strategy_tag": case.get("route_strategy_tag"),
        "outcome": outcome,
        "winner": winner,
        "reason": reason,
    }


def _outcome(winner: str, *, positive_arm: str) -> str:
    if winner == positive_arm:
        return "win"
    if winner == "tie":
        return "tie"
    return "loss"


def _metrics(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    pair_summaries: dict[str, Any],
    policy: dict[str, Any],
    phase9: dict[str, Any],
    train_compact: dict[str, Any],
    env_ready: bool,
    solve_workers: int,
    judge_workers: int,
) -> dict[str, Any]:
    selected_ids = {case["problem_id"] for case in selected_cases}
    v3v1 = pair_summaries.get(f"{V3_ARM}_vs_{V1_ARM}", {})
    policy_v1 = policy["policy_vs_v1"]
    policy_v3 = policy["policy_vs_original_v3"]
    train_metrics = train_compact.get("metrics", {})
    return {
        "heldout_case_count": len(cases),
        "selected_compact_case_count": len(selected_cases),
        "selected_compact_rate": round(len(selected_cases) / max(1, len(cases)), 4),
        "answer_cell_count": sum(len(row) for row in answers.values() if isinstance(row, dict)),
        "live_env_ready": env_ready,
        "solve_workers": solve_workers,
        "judge_workers": judge_workers,
        "planned_answer_calls": len(cases) * 2 + len(selected_cases),
        "planned_judge_calls": len(cases) + len(selected_cases) * 2,
        "planned_total_model_calls": len(cases) * 3 + len(selected_cases) * 3,
        "v3_vs_v1_heldout_n": int(v3v1.get("n") or 0),
        "v3_vs_v1_heldout_utility": float(v3v1.get("utility") or 0.0),
        "policy_vs_v1_heldout_n": int(policy_v1.get("n") or 0),
        "policy_vs_v1_heldout_utility": float(policy_v1.get("utility") or 0.0),
        "policy_vs_v1_heldout_margin": float(policy_v1.get("margin_over_tie") or 0.0),
        "policy_vs_v3_heldout_n": int(policy_v3.get("n") or 0),
        "policy_vs_v3_heldout_utility": float(policy_v3.get("utility") or 0.0),
        "policy_lift_over_v3_vs_v1_heldout": round(float(policy_v1.get("utility") or 0.0) - float(v3v1.get("utility") or 0.0), 4),
        "train_compact_vs_v1_margin": float(train_metrics.get("repair_vs_v1_margin") or 0.0),
        "train_compact_vs_v3_utility": float(train_metrics.get("repair_vs_v3_utility") or 0.0),
        "phase9_base_v3_vs_v1_margin": float((phase9.get("metrics") or {}).get("same_batch_v3_vs_v1_margin") or 0.0),
        "compact_payload_contains_prompts_answers": False,
        "selected_tags_observed": sorted({case.get("route_strategy_tag") for case in selected_cases}),
        "unselected_answered_as_v3_count": len([case for case in cases if case["problem_id"] not in selected_ids]),
    }


def _gates(*, metrics: dict[str, Any], execution_mode: str, run_status: str) -> dict[str, bool]:
    gates = {
        "heldout_slice_large_enough": metrics["heldout_case_count"] >= 50,
        "selected_compact_slice_nonempty": metrics["selected_compact_case_count"] >= 10,
        "model_call_budget_reported": metrics["planned_total_model_calls"] == (
            metrics["heldout_case_count"] * 3 + metrics["selected_compact_case_count"] * 3
        ),
        "compact_payload_redacted": metrics["compact_payload_contains_prompts_answers"] is False,
    }
    if execution_mode in {"execute", "summarize"}:
        gates.update({
            "live_run_completed": run_status in {"execute_complete", "summarize_complete"},
            "policy_all_cases_judged": metrics["policy_vs_v1_heldout_n"] == metrics["heldout_case_count"],
            "policy_beats_v1_hard_gate": metrics["policy_vs_v1_heldout_margin"] >= 0.10,
            "policy_improves_over_v3_heldout": metrics["policy_lift_over_v3_vs_v1_heldout"] > 0.03,
            "policy_noninferior_to_original_v3": metrics["policy_vs_v3_heldout_utility"] >= 0.50,
        })
    else:
        gates["dry_run_ready"] = True
    return gates


def _load_all_route_cases(*, root: Path, phase9_eval_id: str) -> list[dict[str, Any]]:
    path = root / PAPER_DIR / "fresh_live_runs" / f"{phase9_eval_id}_route_plan_summary.json"
    payload = _load_json(path)
    return payload.get("cases") or payload.get("plan", {}).get("cases") or []


def _use_compact(case: dict[str, Any]) -> bool:
    return str(case.get("route_strategy_tag")) in SELECTED_TAGS


def _value(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "tie":
        return 0.5
    return 0.0


def _env_ready(*, solver_model: str, judge_model: str) -> bool:
    try:
        _requests_client_for_alias(solver_model)
        _requests_client_for_alias(judge_model)
        return True
    except Exception:
        return False


def _paths(run_dir: Path, eval_id: str) -> dict[str, Path]:
    return {
        "answers_path": run_dir / f"{eval_id}_answers.json",
        "judgments_path": run_dir / f"{eval_id}_judgments.json",
        "forensic_path": run_dir / f"{eval_id}_forensic.jsonl",
    }


def _interpretation(metrics: dict[str, Any]) -> str:
    if metrics["policy_vs_v1_heldout_n"] == 0:
        return "Dry-run ready; execute mode is needed for heldout performance validation."
    if (
        metrics["policy_vs_v1_heldout_margin"] >= 0.10
        and metrics["policy_lift_over_v3_vs_v1_heldout"] > 0.03
        and metrics["policy_vs_v3_heldout_utility"] >= 0.50
    ):
        return "Selective compact guard generalizes on heldout and can be retained as the V1-regression profile."
    return "Selective compact guard did not clear heldout gates; keep as exploration evidence only."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase9 selective compact guard heldout validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    parser.add_argument("--phase9-eval-id", default=PHASE9_BASE_EVAL_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execution-mode", choices=["dry_run", "execute", "summarize"], default="dry_run")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--solver-model", default="gpt_mini")
    parser.add_argument("--judge-model", default="gpt_mini")
    parser.add_argument("--solve-workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    execution_mode = "execute" if args.execute else "summarize" if args.summarize else args.execution_mode
    root = Path(args.root).resolve()
    payload = build_full_v3_phase9_selective_compact_guard_payload(
        root=root,
        eval_id=args.eval_id,
        phase9_eval_id=args.phase9_eval_id,
        execution_mode=execution_mode,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        solve_workers=args.solve_workers,
        judge_workers=args.judge_workers,
        run_dir=Path(args.run_dir),
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "execution_mode": payload["execution_mode"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
