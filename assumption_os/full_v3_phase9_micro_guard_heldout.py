"""Phase9 post-failure heldout validation for a V3-preserving micro guard.

The broad compact guard improved against V1 but regressed against original V3.
This module tests a narrower repair: keep the original V3 structural-morphism
prompt shape, and add only a small task frame guard for the previously risky
S14/S19 tags.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .full_v3_phase9_selective_compact_guard import (
    COMPACT_ARM as OLD_COMPACT_ARM,
    PAPER_DIR,
    SELECTED_TAGS,
    V1_ARM,
    V3_ARM,
    _display,
    _env_ready,
    _interpretation as _compact_interpretation,
    _load_all_route_cases,
    _outcome,
    _paths,
    _row,
    _stats,
)
from .full_v3_phase9_v1_live_regression import (
    DEFAULT_EVAL_ID as PHASE9_BASE_EVAL_ID,
    DEFAULT_RUN_DIR,
    CONTEXT_PROMPT,
    _judgment_valid,
    _load_dotenv_if_present,
    _load_json,
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


DEFAULT_EVAL_ID = "full_v3_phase9_micro_guard_heldout_20260611"
DEFAULT_OUT = PAPER_DIR / "full_v3_phase9_micro_guard_heldout_20260611.json"
PRIOR_SELECTIVE_EVAL_ID = "full_v3_phase9_selective_compact_guard_heldout_20260611"
MICRO_ARM = "v3_micro_guard"

MICRO_GUARD_PROMPT = """请解决下面的问题。

下面是 Structural Morphism Reasoning。它只用于检查任务结构，不要把答案写成固定模板；如果结构提示和题面主任务冲突，以题面主任务为准。

{context}

内部作答规则：
- 先按题目自身的主 frame 作答，再只把 structural morphism 当作 1-2 个约束或检查项。
- S14 / Counterexample 场景：只有在题目确实要求边界、极端、异常鲁棒性、隐藏依赖或终止条件时，才突出反例/边界测试；如果题目是多平台/多报告共同失败，优先共同路径/共同原因；如果是从旧系统扩容，优先渐进扩展；如果是形式证明，必须给不变量、上界或模型检查；如果是删除旧模块，先证伪必要性再删除。
- S19 / Bottleneck 场景：先区分硬红线和可调约束；多目标工程题用 Pareto/权重/动态松弛/分工况控制，不要一刀切，也不要空泛平衡。
- 不展示标签名或元分析，只给最终方案。

要求：
- 具体、可执行。
- 明确关键判断标准、步骤、风险控制和停止/转向条件。
- 不超过 650 字。

问题：
{problem}
"""


def build_full_v3_phase9_micro_guard_heldout_payload(
    *,
    root: Path,
    eval_id: str = DEFAULT_EVAL_ID,
    phase9_eval_id: str = PHASE9_BASE_EVAL_ID,
    prior_selective_eval_id: str = PRIOR_SELECTIVE_EVAL_ID,
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
    phase9 = _load_json(root / PAPER_DIR / f"{phase9_eval_id}.json")
    all_cases = _load_all_route_cases(root=root, phase9_eval_id=phase9_eval_id)
    train_ids = {case["problem_id"] for case in (phase9.get("route_plan") or {}).get("active_cases", [])}
    heldout_cases = [case for case in all_cases if case["problem_id"] not in train_ids]
    selected_cases = [case for case in heldout_cases if _use_micro_guard(case)]
    paths = _paths(run_dir, eval_id)
    prior_paths = _paths(run_dir, prior_selective_eval_id)
    prior_answers = _load_json(prior_paths["answers_path"]) if prior_paths["answers_path"].exists() else {}
    prior_judgments = _load_json(prior_paths["judgments_path"]) if prior_paths["judgments_path"].exists() else {}
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
            prior_answers=prior_answers,
            prior_judgments=prior_judgments,
            solver_model=solver_model,
            judge_model=judge_model,
            solve_workers=solve_workers,
            judge_workers=judge_workers,
        )
        run_status = "execute_complete"
    elif execution_mode == "summarize":
        answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
        judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
        answers = _merge_prior_answers(heldout_cases, answers, prior_answers)
        judgments = _merge_prior_judgments(heldout_cases, judgments, prior_judgments)
        run_status = "summarize_complete" if judgments else "summarize_missing_judgments"
    else:
        answers = _merge_prior_answers(heldout_cases, {}, prior_answers)
        judgments = _merge_prior_judgments(heldout_cases, {}, prior_judgments)
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
        env_ready=env_ready,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
        prior_answers=prior_answers,
        prior_judgments=prior_judgments,
    )
    gates = _gates(metrics=metrics, execution_mode=execution_mode, run_status=run_status)
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase9_micro_guard_heldout",
        "performance_validation": True,
        "execution_mode": execution_mode,
        "run_status": run_status,
        "validation_scope": (
            "Post-failure heldout validation of a V3-preserving micro guard.  The policy activates only on "
            "S14/S19 and keeps original V3 elsewhere; V1/V3 baselines are reused from the prior heldout run."
        ),
        "selector": {
            "selected_tags": sorted(SELECTED_TAGS),
            "selected_arm": MICRO_ARM,
            "selection_rule": (
                "Use a micro task frame guard only on S14/S19.  The guard preserves the original V3 prompt shape "
                "and corrects known over-routing: common-cause, staged scaling, formal proof, dead-code deletion, "
                "and bottleneck tradeoff cases."
            ),
            "prior_baseline_source": prior_selective_eval_id,
            "failed_profile_repaired": OLD_COMPACT_ARM,
        },
        "heldout_case_counts": {
            "all_route_cases": len(all_cases),
            "train_cases": len(train_ids),
            "heldout_cases": len(heldout_cases),
            "selected_micro_cases": len(selected_cases),
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
    prior_answers: dict[str, Any],
    prior_judgments: dict[str, Any],
    solver_model: str,
    judge_model: str,
    solve_workers: int,
    judge_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answers = _load_json(paths["answers_path"]) if paths["answers_path"].exists() else {}
    judgments = _load_json(paths["judgments_path"]) if paths["judgments_path"].exists() else {}
    answers = _merge_prior_answers(cases, answers, prior_answers)
    judgments = _merge_prior_judgments(cases, judgments, prior_judgments)
    solver = _requests_client_for_alias(solver_model)
    judge = _requests_client_for_alias(judge_model)
    _solve_missing_micro(
        selected_cases=selected_cases,
        answers=answers,
        answers_path=paths["answers_path"],
        forensic_path=paths["forensic_path"],
        solver=solver,
        solver_model=solver_model,
        max_workers=solve_workers,
    )
    _judge_missing_micro(
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


def _merge_prior_answers(cases: list[dict[str, Any]], answers: dict[str, Any], prior_answers: dict[str, Any]) -> dict[str, Any]:
    merged = {pid: dict(row) for pid, row in answers.items() if isinstance(row, dict)}
    for case in cases:
        pid = case["problem_id"]
        merged.setdefault(pid, {})
        prior = prior_answers.get(pid, {})
        if isinstance(prior, dict):
            for arm in [V1_ARM, V3_ARM]:
                if prior.get(arm) and not merged[pid].get(arm):
                    merged[pid][arm] = prior[arm]
    return merged


def _merge_prior_judgments(
    cases: list[dict[str, Any]], judgments: dict[str, Any], prior_judgments: dict[str, Any]
) -> dict[str, Any]:
    merged = {pid: dict(row) for pid, row in judgments.items() if isinstance(row, dict)}
    pair = f"{V3_ARM}_vs_{V1_ARM}"
    for case in cases:
        pid = case["problem_id"]
        merged.setdefault(pid, {})
        prior = prior_judgments.get(pid, {})
        if isinstance(prior, dict) and _judgment_valid(prior.get(pair)) and not _judgment_valid(merged[pid].get(pair)):
            merged[pid][pair] = prior[pair]
    return merged


def _solve_missing_micro(
    *,
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    answers_path: Path,
    forensic_path: Path,
    solver: Any,
    solver_model: str,
    max_workers: int,
) -> None:
    jobs = [case for case in selected_cases if not answers.get(case["problem_id"], {}).get(MICRO_ARM)]
    if not jobs:
        answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_solve_one_micro, case, solver, solver_model, forensic_path) for case in jobs]
        for fut in as_completed(futures):
            pid, text = fut.result()
            answers.setdefault(pid, {})[MICRO_ARM] = text
            completed += 1
            if completed % 10 == 0:
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 micro solve] {completed}/{len(jobs)}", flush=True)
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_one_micro(case: dict[str, Any], solver: Any, solver_model: str, forensic_path: Path) -> tuple[str, str]:
    prompt = MICRO_GUARD_PROMPT.format(problem=case["description"], context=case["structural_context"])
    t0 = time.time()
    response = _call_with_retry(solver, prompt, max_tokens=1100, temperature=0.25)
    text = response.get("text", "").strip()
    _write_jsonl(forensic_path, {
        "role": "solver",
        "eval_kind": "phase9_micro_guard",
        "problem_id": case["problem_id"],
        "arm": MICRO_ARM,
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


def _judge_missing_micro(
    *,
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    judgments: dict[str, Any],
    judgments_path: Path,
    forensic_path: Path,
    judge: Any,
    judge_model: str,
    max_workers: int,
) -> None:
    jobs = []
    for case in selected_cases:
        pid = case["problem_id"]
        judgments.setdefault(pid, {})
        if not answers.get(pid, {}).get(MICRO_ARM):
            continue
        for reference_arm in [V1_ARM, V3_ARM]:
            if not answers.get(pid, {}).get(reference_arm):
                continue
            pair = f"{MICRO_ARM}_vs_{reference_arm}"
            if not _judgment_valid(judgments[pid].get(pair)):
                jobs.append((case, reference_arm, pair))
    if not jobs:
        judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_judge_one_micro, case, reference_arm, pair, answers, judge, judge_model, forensic_path)
            for case, reference_arm, pair in jobs
        ]
        for fut in as_completed(futures):
            pid, pair, judgment = fut.result()
            judgments.setdefault(pid, {})[pair] = judgment
            completed += 1
            if completed % 10 == 0:
                judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 micro judge] {completed}/{len(jobs)}", flush=True)
    judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_one_micro(
    case: dict[str, Any],
    reference_arm: str,
    pair: str,
    answers: dict[str, Any],
    judge: Any,
    judge_model: str,
    forensic_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    pid = case["problem_id"]
    swap = int(__import__("hashlib").sha1(f"{pid}:{pair}:micro".encode()).hexdigest(), 16) % 2 == 1
    a_arm, b_arm = (reference_arm, MICRO_ARM) if swap else (MICRO_ARM, reference_arm)
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
        "eval_kind": "phase9_micro_guard",
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
    pairs = [f"{V3_ARM}_vs_{V1_ARM}", f"{MICRO_ARM}_vs_{V1_ARM}", f"{MICRO_ARM}_vs_{V3_ARM}"]
    selected_ids = {case["problem_id"] for case in selected_cases}
    out = {}
    for pair in pairs:
        rows = []
        for case in cases:
            if pair.startswith(MICRO_ARM) and case["problem_id"] not in selected_ids:
                continue
            judgment = judgments.get(case["problem_id"], {}).get(pair)
            if not _judgment_valid(judgment):
                continue
            positive_arm = pair.split("_vs_", 1)[0]
            outcome = _outcome(judgment.get("winner"), positive_arm=positive_arm)
            rows.append(_row(case, outcome, judgment.get("winner"), judgment.get("reason", "")))
        out[pair] = _stats(pair, rows)
    return out


def _policy_summary(*, cases: list[dict[str, Any]], selected_cases: list[dict[str, Any]], judgments: dict[str, Any]) -> dict[str, Any]:
    selected_ids = {case["problem_id"] for case in selected_cases}
    vs_v1_rows = []
    vs_v3_rows = []
    for case in cases:
        pid = case["problem_id"]
        if pid in selected_ids:
            v1_judgment = judgments.get(pid, {}).get(f"{MICRO_ARM}_vs_{V1_ARM}")
            v3_judgment = judgments.get(pid, {}).get(f"{MICRO_ARM}_vs_{V3_ARM}")
            if _judgment_valid(v1_judgment):
                vs_v1_rows.append(_row(case, _outcome(v1_judgment.get("winner"), positive_arm=MICRO_ARM), v1_judgment.get("winner"), v1_judgment.get("reason", "")))
            if _judgment_valid(v3_judgment):
                vs_v3_rows.append(_row(case, _outcome(v3_judgment.get("winner"), positive_arm=MICRO_ARM), v3_judgment.get("winner"), v3_judgment.get("reason", "")))
        else:
            v1_judgment = judgments.get(pid, {}).get(f"{V3_ARM}_vs_{V1_ARM}")
            if _judgment_valid(v1_judgment):
                vs_v1_rows.append(_row(case, _outcome(v1_judgment.get("winner"), positive_arm=V3_ARM), v1_judgment.get("winner"), v1_judgment.get("reason", "")))
            vs_v3_rows.append(_row(case, "tie", "tie", "policy kept original V3"))
    return {
        "policy_vs_v1": _stats("policy_vs_v1", vs_v1_rows),
        "policy_vs_original_v3": _stats("policy_vs_original_v3", vs_v3_rows),
    }


def _metrics(
    *,
    cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    answers: dict[str, Any],
    pair_summaries: dict[str, Any],
    policy: dict[str, Any],
    phase9: dict[str, Any],
    env_ready: bool,
    solve_workers: int,
    judge_workers: int,
    prior_answers: dict[str, Any],
    prior_judgments: dict[str, Any],
) -> dict[str, Any]:
    v3v1 = pair_summaries.get(f"{V3_ARM}_vs_{V1_ARM}", {})
    micro_v1 = pair_summaries.get(f"{MICRO_ARM}_vs_{V1_ARM}", {})
    micro_v3 = pair_summaries.get(f"{MICRO_ARM}_vs_{V3_ARM}", {})
    policy_v1 = policy["policy_vs_v1"]
    policy_v3 = policy["policy_vs_original_v3"]
    return {
        "heldout_case_count": len(cases),
        "selected_micro_case_count": len(selected_cases),
        "selected_micro_rate": round(len(selected_cases) / max(1, len(cases)), 4),
        "answer_cell_count": sum(len(row) for row in answers.values() if isinstance(row, dict)),
        "prior_answer_case_count": len(prior_answers),
        "prior_judgment_case_count": len(prior_judgments),
        "live_env_ready": env_ready,
        "solve_workers": solve_workers,
        "judge_workers": judge_workers,
        "planned_new_answer_calls": len(selected_cases),
        "planned_new_judge_calls": len(selected_cases) * 2,
        "planned_new_model_calls": len(selected_cases) * 3,
        "v3_vs_v1_heldout_n": int(v3v1.get("n") or 0),
        "v3_vs_v1_heldout_utility": float(v3v1.get("utility") or 0.0),
        "micro_vs_v1_selected_n": int(micro_v1.get("n") or 0),
        "micro_vs_v1_selected_utility": float(micro_v1.get("utility") or 0.0),
        "micro_vs_v3_selected_n": int(micro_v3.get("n") or 0),
        "micro_vs_v3_selected_utility": float(micro_v3.get("utility") or 0.0),
        "policy_vs_v1_heldout_n": int(policy_v1.get("n") or 0),
        "policy_vs_v1_heldout_utility": float(policy_v1.get("utility") or 0.0),
        "policy_vs_v1_heldout_margin": float(policy_v1.get("margin_over_tie") or 0.0),
        "policy_vs_v3_heldout_n": int(policy_v3.get("n") or 0),
        "policy_vs_v3_heldout_utility": float(policy_v3.get("utility") or 0.0),
        "policy_lift_over_v3_vs_v1_heldout": round(float(policy_v1.get("utility") or 0.0) - float(v3v1.get("utility") or 0.0), 4),
        "phase9_base_v3_vs_v1_margin": float((phase9.get("metrics") or {}).get("same_batch_v3_vs_v1_margin") or 0.0),
        "compact_payload_contains_prompts_answers": False,
        "selected_tags_observed": sorted({case.get("route_strategy_tag") for case in selected_cases}),
    }


def _gates(*, metrics: dict[str, Any], execution_mode: str, run_status: str) -> dict[str, bool]:
    gates = {
        "heldout_slice_large_enough": metrics["heldout_case_count"] >= 50,
        "selected_micro_slice_nonempty": metrics["selected_micro_case_count"] >= 10,
        "new_model_call_budget_reported": metrics["planned_new_model_calls"] == metrics["selected_micro_case_count"] * 3,
        "prior_baselines_available": metrics["prior_answer_case_count"] >= metrics["heldout_case_count"],
        "compact_payload_redacted": metrics["compact_payload_contains_prompts_answers"] is False,
    }
    if execution_mode in {"execute", "summarize"}:
        gates.update({
            "live_run_completed": run_status in {"execute_complete", "summarize_complete"},
            "policy_all_cases_judged": metrics["policy_vs_v1_heldout_n"] == metrics["heldout_case_count"],
            "micro_selected_cases_judged": metrics["micro_vs_v3_selected_n"] == metrics["selected_micro_case_count"],
            "policy_beats_v1_hard_gate": metrics["policy_vs_v1_heldout_margin"] >= 0.10,
            "policy_improves_over_v3_heldout": metrics["policy_lift_over_v3_vs_v1_heldout"] > 0.03,
            "policy_noninferior_to_original_v3": metrics["policy_vs_v3_heldout_utility"] >= 0.50,
            "micro_selected_noninferior_to_v3": metrics["micro_vs_v3_selected_utility"] >= 0.50,
        })
    else:
        gates["dry_run_ready"] = True
    return gates


def _use_micro_guard(case: dict[str, Any]) -> bool:
    return str(case.get("route_strategy_tag")) in SELECTED_TAGS


def _interpretation(metrics: dict[str, Any]) -> str:
    if metrics["policy_vs_v1_heldout_n"] == 0:
        return "Dry-run ready; execute mode is needed for heldout performance validation."
    if (
        metrics["policy_vs_v1_heldout_margin"] >= 0.10
        and metrics["policy_lift_over_v3_vs_v1_heldout"] > 0.03
        and metrics["policy_vs_v3_heldout_utility"] >= 0.50
        and metrics["micro_vs_v3_selected_utility"] >= 0.50
    ):
        return "The V3-preserving micro guard repairs the compact-profile V3 regression and can be retained."
    return _compact_interpretation({
        "policy_vs_v1_heldout_n": metrics["policy_vs_v1_heldout_n"],
        "policy_vs_v1_heldout_margin": metrics["policy_vs_v1_heldout_margin"],
        "policy_lift_over_v3_vs_v1_heldout": metrics["policy_lift_over_v3_vs_v1_heldout"],
        "policy_vs_v3_heldout_utility": metrics["policy_vs_v3_heldout_utility"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase9 V3-preserving micro guard heldout validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    parser.add_argument("--phase9-eval-id", default=PHASE9_BASE_EVAL_ID)
    parser.add_argument("--prior-selective-eval-id", default=PRIOR_SELECTIVE_EVAL_ID)
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
    payload = build_full_v3_phase9_micro_guard_heldout_payload(
        root=root,
        eval_id=args.eval_id,
        phase9_eval_id=args.phase9_eval_id,
        prior_selective_eval_id=args.prior_selective_eval_id,
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
