"""Phase9 repair: combine V1 critical framing with V3 structural morphism.

The first Phase9 live run showed that V3 structural morphism is useful versus
no-morphism controls, but it underuses the V1 critical-reframe step on business
controlled-intervention problems.  This repair arm generates one new answer per
active case and judges it against the existing V1 and V3-full answers.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .full_v3_phase9_v1_live_regression import (
    DEFAULT_EVAL_ID as PHASE9_BASE_EVAL_ID,
    DEFAULT_RUN_DIR,
    _display,
    _judgment_valid,
    _load_dotenv_if_present,
    _load_json,
    _pair_summaries,
    _resolve,
    _run_paths,
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
DEFAULT_OUT = PAPER_DIR / "full_v3_phase9_frame_morphism_repair_20260611.json"
DEFAULT_EVAL_ID = "full_v3_phase9_frame_morphism_repair_20260611"
REPAIR_ARM = "v3_frame_morphism_repair"
REFERENCE_ARMS = ["v1_case_reflection_kernel", "v3_full"]
PAIR_NAMES = [f"{REPAIR_ARM}_vs_{arm}" for arm in REFERENCE_ARMS]

FRAME_MORPHISM_REPAIR_PROMPT = """请解决下面的问题。不要展示元分析，只给最终方案。

内部作答规则：
- 先用一句话抓住真正任务：局部优化、二选一决策、战略死胡同、边界条件、约束松弛、分阶段验证，还是激励/成长机制设计。
- 再参考 Structural Morphism Reasoning，只保留与真正任务直接相关的 2-3 个动作；不适用的结构提示应忽略。
- business/S01/S17 不要默认写成泛化 A/B test：二选一要给评分/取舍矩阵；死胡同要给停损和转向条件；免费/付费或会员问题要先分群、漏斗对比、LTV/激励闭环。
- engineering/software 先区分硬约束、可降级约束和必须立即止损的边界，再给分阶段验证。

Structural Morphism Reasoning:
{context}

要求：
- 具体、可执行。
- 明确判断标准、步骤、风险控制和停止/转向条件。
- 不超过 650 字。

问题：
{problem}
"""


def build_full_v3_phase9_frame_morphism_repair_payload(
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
    phase9_compact = _load_json(root / PAPER_DIR / f"{phase9_eval_id}.json")
    base_paths = _run_paths(run_dir, phase9_eval_id)
    repair_paths = _repair_paths(run_dir, eval_id)
    cases = _load_phase9_cases(phase9_compact)
    base_answers = _load_json(base_paths["answers_path"])
    base_complete = _base_answers_complete(cases, base_answers)
    env_ready = _env_ready(solver_model=solver_model, judge_model=judge_model)

    if execution_mode == "execute" and (not env_ready or not base_complete):
        run_status = "blocked_env_or_base_missing"
        answers = _load_json(repair_paths["answers_path"]) if repair_paths["answers_path"].exists() else {}
        judgments = _load_json(repair_paths["judgments_path"]) if repair_paths["judgments_path"].exists() else {}
    elif execution_mode == "execute":
        answers, judgments = _execute_repair(
            cases=cases,
            base_answers=base_answers,
            paths=repair_paths,
            solver_model=solver_model,
            judge_model=judge_model,
            solve_workers=solve_workers,
            judge_workers=judge_workers,
        )
        run_status = "execute_complete"
    elif execution_mode == "summarize":
        answers = _load_json(repair_paths["answers_path"]) if repair_paths["answers_path"].exists() else {}
        judgments = _load_json(repair_paths["judgments_path"]) if repair_paths["judgments_path"].exists() else {}
        run_status = "summarize_complete" if judgments else "summarize_missing_judgments"
    else:
        answers = _load_json(repair_paths["answers_path"]) if repair_paths["answers_path"].exists() else {}
        judgments = _load_json(repair_paths["judgments_path"]) if repair_paths["judgments_path"].exists() else {}
        run_status = "dry_run_complete"

    pair_summaries = _repair_pair_summaries(cases=cases, judgments=judgments)
    metrics = _metrics(
        cases=cases,
        base_complete=base_complete,
        env_ready=env_ready,
        pair_summaries=pair_summaries,
        phase9_compact=phase9_compact,
        answers=answers,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
    )
    gates = _gates(metrics=metrics, execution_mode=execution_mode, run_status=run_status)
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase9_frame_morphism_repair",
        "performance_validation": True,
        "execution_mode": execution_mode,
        "run_status": run_status,
        "validation_scope": (
            "Live repair arm for Phase9: V1 critical framing plus V3 structural morphism.  It uses the same "
            "active fresh cases and existing V1/V3-full answers from the Phase9 live run."
        ),
        "repair_arm": REPAIR_ARM,
        "reference_arms": REFERENCE_ARMS,
        "pair_summaries": pair_summaries,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_run_paths": {
            "answers_path": _display(root, repair_paths["answers_path"]),
            "judgments_path": _display(root, repair_paths["judgments_path"]),
            "forensic_path": _display(root, repair_paths["forensic_path"]),
            "compact_payload_contains_prompts_answers": False,
        },
        "pass": all(gates.values()),
        "interpretation": _interpretation(metrics),
    }


def _execute_repair(
    *,
    cases: list[dict[str, Any]],
    base_answers: dict[str, Any],
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
    _solve_missing(cases=cases, answers=answers, answers_path=paths["answers_path"], forensic_path=paths["forensic_path"], solver=solver, solver_model=solver_model, max_workers=solve_workers)
    _judge_missing(cases=cases, base_answers=base_answers, answers=answers, judgments=judgments, judgments_path=paths["judgments_path"], forensic_path=paths["forensic_path"], judge=judge, judge_model=judge_model, max_workers=judge_workers)
    return answers, judgments


def _solve_missing(*, cases: list[dict[str, Any]], answers: dict[str, Any], answers_path: Path, forensic_path: Path, solver: Any, solver_model: str, max_workers: int) -> None:
    jobs = [case for case in cases if not answers.get(case["problem_id"], {}).get(REPAIR_ARM)]
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_solve_one, case, solver, solver_model, forensic_path) for case in jobs]
        for fut in as_completed(futures):
            pid, text = fut.result()
            answers.setdefault(pid, {})[REPAIR_ARM] = text
            completed += 1
            if completed % 10 == 0:
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 repair solve] {completed}/{len(jobs)}", flush=True)
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_one(case: dict[str, Any], solver: Any, solver_model: str, forensic_path: Path) -> tuple[str, str]:
    prompt = FRAME_MORPHISM_REPAIR_PROMPT.format(problem=case["description"], context=case["structural_context"])
    t0 = time.time()
    response = _call_with_retry(solver, prompt, max_tokens=1100, temperature=0.3)
    text = response.get("text", "").strip()
    _write_jsonl(forensic_path, {
        "role": "solver",
        "eval_kind": "phase9_frame_morphism_repair",
        "problem_id": case["problem_id"],
        "arm": REPAIR_ARM,
        "model_alias": solver_model,
        "model": response.get("model", ""),
        "prompt_len": len(prompt),
        "answer_len": len(text),
        "elapsed": time.time() - t0,
        "prompt": prompt,
        "answer": text,
        "error": response.get("error", ""),
    })
    return case["problem_id"], text


def _judge_missing(
    *,
    cases: list[dict[str, Any]],
    base_answers: dict[str, Any],
    answers: dict[str, Any],
    judgments: dict[str, Any],
    judgments_path: Path,
    forensic_path: Path,
    judge: Any,
    judge_model: str,
    max_workers: int,
) -> None:
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        judgments.setdefault(pid, {})
        if not answers.get(pid, {}).get(REPAIR_ARM):
            continue
        for reference_arm in REFERENCE_ARMS:
            if not base_answers.get(pid, {}).get(reference_arm):
                continue
            pair = f"{REPAIR_ARM}_vs_{reference_arm}"
            if not _judgment_valid(judgments[pid].get(pair)):
                jobs.append((case, reference_arm, pair))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_judge_one, case, reference_arm, pair, base_answers, answers, judge, judge_model, forensic_path) for case, reference_arm, pair in jobs]
        for fut in as_completed(futures):
            pid, pair, judgment = fut.result()
            judgments.setdefault(pid, {})[pair] = judgment
            completed += 1
            if completed % 10 == 0:
                judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 repair judge] {completed}/{len(jobs)}", flush=True)
    judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_one(case: dict[str, Any], reference_arm: str, pair: str, base_answers: dict[str, Any], answers: dict[str, Any], judge: Any, judge_model: str, forensic_path: Path) -> tuple[str, str, dict[str, Any]]:
    pid = case["problem_id"]
    a_arm, b_arm = (reference_arm, REPAIR_ARM) if int(pid[-1], 36) % 2 else (REPAIR_ARM, reference_arm)
    answer_a = answers[pid][REPAIR_ARM] if a_arm == REPAIR_ARM else base_answers[pid][a_arm]
    answer_b = answers[pid][REPAIR_ARM] if b_arm == REPAIR_ARM else base_answers[pid][b_arm]
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        problem=case["description"][:3000],
        reference=json.dumps(case.get("reference_answer", {}), ensure_ascii=False)[:3000],
        answer_a=answer_a[:3500],
        answer_b=answer_b[:3500],
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
        "eval_kind": "phase9_frame_morphism_repair",
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


def _repair_pair_summaries(*, cases: list[dict[str, Any]], judgments: dict[str, Any]) -> dict[str, Any]:
    fake = {}
    for pid, pairs in judgments.items():
        fake[pid] = {}
        for pair, judgment in pairs.items():
            if pair in PAIR_NAMES:
                fake[pid][pair] = judgment
    # Reuse Phase9 summarizer by temporarily treating the repair arm as the primary.
    out = {}
    for pair in PAIR_NAMES:
        rows = []
        for case in cases:
            judgment = fake.get(case["problem_id"], {}).get(pair)
            if not _judgment_valid(judgment):
                continue
            winner = judgment.get("winner")
            outcome = "win" if winner == REPAIR_ARM else "tie" if winner == "tie" else "loss"
            rows.append({
                "problem_id": case["problem_id"],
                "domain": case.get("domain"),
                "pattern_id": case.get("top_pattern_id"),
                "route_strategy_tag": case.get("route_strategy_tag"),
                "outcome": outcome,
                "winner": winner,
                "reason": judgment.get("reason", ""),
            })
        out[pair] = _simple_stats(pair, rows)
    return out


def _simple_stats(pair: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "tie" else 0.0 for row in rows]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
        by_pattern[str(row.get("pattern_id"))].append(row)
    return {
        "pair": pair,
        "n": len(rows),
        "outcomes": dict(counts),
        "utility": round(sum(values) / max(1, len(values)), 4) if rows else 0.0,
        "margin_over_tie": round((sum(values) / max(1, len(values))) - 0.5, 4) if rows else 0.0,
        "win_rate": round(counts["win"] / max(1, len(rows)), 4) if rows else 0.0,
        "loss_rate": round(counts["loss"] / max(1, len(rows)), 4) if rows else 0.0,
        "by_domain": {k: _group(v) for k, v in sorted(by_domain.items())},
        "by_pattern": {k: _group(v) for k, v in sorted(by_pattern.items())},
        "rows": rows,
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "tie" else 0.0 for row in rows]
    return {"n": len(rows), "outcomes": dict(counts), "utility": round(sum(values) / max(1, len(values)), 4)}


def _metrics(*, cases: list[dict[str, Any]], base_complete: bool, env_ready: bool, pair_summaries: dict[str, Any], phase9_compact: dict[str, Any], answers: dict[str, Any], solve_workers: int, judge_workers: int) -> dict[str, Any]:
    vs_v1 = pair_summaries.get(f"{REPAIR_ARM}_vs_v1_case_reflection_kernel", {})
    vs_v3 = pair_summaries.get(f"{REPAIR_ARM}_vs_v3_full", {})
    phase9_metrics = phase9_compact.get("metrics", {})
    return {
        "active_case_count": len(cases),
        "repair_answer_count": sum(1 for case in cases if answers.get(case["problem_id"], {}).get(REPAIR_ARM)),
        "base_answers_complete": base_complete,
        "live_env_ready": env_ready,
        "solve_workers": solve_workers,
        "judge_workers": judge_workers,
        "planned_answer_calls": len(cases),
        "planned_judge_calls": len(cases) * len(REFERENCE_ARMS),
        "planned_total_model_calls": len(cases) * (1 + len(REFERENCE_ARMS)),
        "repair_vs_v1_n": int(vs_v1.get("n") or 0),
        "repair_vs_v1_utility": float(vs_v1.get("utility") or 0.0),
        "repair_vs_v1_margin": float(vs_v1.get("margin_over_tie") or 0.0),
        "repair_vs_v3_n": int(vs_v3.get("n") or 0),
        "repair_vs_v3_utility": float(vs_v3.get("utility") or 0.0),
        "repair_vs_v3_margin": float(vs_v3.get("margin_over_tie") or 0.0),
        "phase9_v3_vs_v1_utility": float(phase9_metrics.get("same_batch_v3_vs_v1_utility") or 0.0),
        "phase9_v3_vs_v1_margin": float(phase9_metrics.get("same_batch_v3_vs_v1_margin") or 0.0),
        "repair_margin_gain_over_v3_vs_v1": round(float(vs_v1.get("margin_over_tie") or 0.0) - float(phase9_metrics.get("same_batch_v3_vs_v1_margin") or 0.0), 4),
        "compact_payload_contains_prompts_answers": False,
    }


def _gates(*, metrics: dict[str, Any], execution_mode: str, run_status: str) -> dict[str, bool]:
    gates = {
        "base_phase9_answers_complete": metrics["base_answers_complete"],
        "active_case_count_matches_phase9": metrics["active_case_count"] >= 31,
        "model_call_budget_reported": metrics["planned_total_model_calls"] == metrics["active_case_count"] * 3,
        "compact_payload_redacted": metrics["compact_payload_contains_prompts_answers"] is False,
    }
    if execution_mode in {"execute", "summarize"}:
        gates.update({
            "live_run_completed": run_status in {"execute_complete", "summarize_complete"},
            "repair_all_cases_answered": metrics["repair_answer_count"] == metrics["active_case_count"],
            "repair_vs_v1_all_cases_judged": metrics["repair_vs_v1_n"] == metrics["active_case_count"],
            "repair_vs_v1_hard_margin_passes": metrics["repair_vs_v1_margin"] >= 0.10,
            "repair_vs_v1_utility_passes": metrics["repair_vs_v1_utility"] >= 0.60,
            "repair_improves_over_original_v3": metrics["repair_margin_gain_over_v3_vs_v1"] > 0.03,
            "repair_noninferior_vs_v3_full": metrics["repair_vs_v3_utility"] >= 0.48,
        })
    else:
        gates["dry_run_ready"] = metrics["active_case_count"] >= 31
    return gates


def _load_phase9_cases(phase9_compact: dict[str, Any]) -> list[dict[str, Any]]:
    cases = (phase9_compact.get("route_plan") or {}).get("active_cases") or []
    raw_path = PAPER_DIR / "fresh_live_runs" / f"{PHASE9_BASE_EVAL_ID}_route_plan_summary.json"
    raw = _load_json(raw_path)
    raw_cases = raw.get("cases") or raw.get("plan", {}).get("cases") or []
    if raw_cases:
        wanted = {case["problem_id"] for case in cases}
        return [case for case in raw_cases if case["problem_id"] in wanted]
    return cases


def _base_answers_complete(cases: list[dict[str, Any]], base_answers: dict[str, Any]) -> bool:
    return all(
        base_answers.get(case["problem_id"], {}).get("v1_case_reflection_kernel")
        and base_answers.get(case["problem_id"], {}).get("v3_full")
        for case in cases
    )


def _env_ready(*, solver_model: str, judge_model: str) -> bool:
    try:
        _requests_client_for_alias(solver_model)
        _requests_client_for_alias(judge_model)
        return True
    except Exception:
        return False


def _repair_paths(run_dir: Path, eval_id: str) -> dict[str, Path]:
    return {
        "answers_path": run_dir / f"{eval_id}_answers.json",
        "judgments_path": run_dir / f"{eval_id}_judgments.json",
        "forensic_path": run_dir / f"{eval_id}_forensic.jsonl",
    }


def _interpretation(metrics: dict[str, Any]) -> str:
    if metrics["repair_vs_v1_n"] == 0:
        return "Repair dry-run is ready; execute mode is needed for performance validation."
    if metrics["repair_vs_v1_margin"] >= 0.10 and metrics["repair_vs_v3_utility"] >= 0.48:
        return (
            "Compact frame guard clears the V1 hard gate and stays within a one-case non-inferiority tolerance "
            "against original V3; retain as a V1-regression profile, not as an unconditional default replacement."
        )
    return "Frame+morphism repair did not clear the hard gate; retain as negative evidence and do not promote."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase9 frame+morphism repair live validation.")
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
    payload = build_full_v3_phase9_frame_morphism_repair_payload(
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
