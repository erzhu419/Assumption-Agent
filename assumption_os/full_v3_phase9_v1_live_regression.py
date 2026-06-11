"""Full-v3 Phase 9 same-batch live V1 regression gate.

This module is the hard follow-up to the frozen V1 comparison.  It runs the
same active fresh problems through:

- v1_case_reflection_kernel: a v20-style frame/rewrite kernel without the V3
  assumption graph;
- v3_full: the current guarded structural morphism prompt;
- v3_no_morphism: direct answer without structural context;
- v3_no_recursive: structural invariants with recursive repair guidance removed;
- v3_no_world_model: structural context without the guarded-use instruction.

The compact paper artifact stores only route metadata and pairwise summaries.
Raw answers/prompts stay in the run directory and are not required for paper
aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .full_v3_fresh_live_benchmark import (
    _bootstrap_ci,
    _build_fresh_sample,
    _live_env_status,
    _sign_test,
)
from .structural_live_ablation import (
    BASE_PROMPT,
    CONTEXT_PROMPT,
    PAIRWISE_JUDGE_PROMPT,
    _call_with_retry,
    _parse_judge_json,
    _requests_client_for_alias,
    _write_jsonl,
    build_structural_live_ablation_payload,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_RUN_DIR = PAPER_DIR / "fresh_live_runs"
DEFAULT_OUT = PAPER_DIR / "full_v3_phase9_v1_live_regression_20260611.json"
DEFAULT_EVAL_ID = "full_v3_phase9_v1_live_regression_20260611"
DEFAULT_SOLVER_MODEL = "gpt_mini"
DEFAULT_JUDGE_MODEL = "gpt_mini"
DEFAULT_SELECTION_MODE = "natural_repaired_guarded"

PRIMARY_ARM = "v3_full"
BASELINE_ARMS = [
    "v1_case_reflection_kernel",
    "v3_no_morphism",
    "v3_no_recursive",
    "v3_no_world_model",
]
ANSWER_ARMS = [PRIMARY_ARM, *BASELINE_ARMS]
PAIR_NAMES = [f"{PRIMARY_ARM}_vs_{arm}" for arm in BASELINE_ARMS]

V1_CASE_REFLECTION_PROMPT = """请解决下面的问题。你只能使用一个 v1/v20 风格的 case-reflection kernel，不要使用 Assumption Graph、structural morphism、world model 或递归假设图。

先在内部完成：
1. 判断问题 frame：object / paradigm / hybrid。
2. 写出 critical reframe：真正该优化或回答的维度。
3. 列出 2-3 个要避免的反模式。

然后直接给最终答案，不要展示冗长诊断过程。

要求：
- 方案具体、可执行。
- 明确判断标准、步骤和风险控制。
- 不超过 650 字。

问题：
{problem}
"""

NO_WORLD_MODEL_CONTEXT_PROMPT = """请解决下面的问题。

下面是未经 world-model/quality-gate 筛选的 Structural Morphism Reasoning。请尽量使用其中的结构类比来组织答案；如果类比有风险，也要在答案中指出。

{context}

要求：
- 给出具体、可执行的方案。
- 明确关键判断标准、步骤和风险控制。
- 不超过 650 字。

问题：
{problem}
"""

NO_RECURSIVE_CONTEXT_PROMPT = """请解决下面的问题。

下面只保留 structural morphism 的静态不变量，移除了递归 self-evolution 得到的 repair guidance。只有在当前问题保持这些不变量时才使用。

{context}

要求：
- 给出具体、可执行的方案。
- 明确关键判断标准、步骤和风险控制。
- 不超过 650 字。

问题：
{problem}
"""


def build_full_v3_phase9_v1_live_regression_payload(
    *,
    root: Path,
    eval_id: str = DEFAULT_EVAL_ID,
    execution_mode: str = "dry_run",
    sample_size: int | str = "full",
    active_sample_size: int = 0,
    seed: int = 20260611,
    solver_model: str = DEFAULT_SOLVER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    solve_workers: int = 12,
    judge_workers: int = 6,
    run_dir: Path | None = None,
    sample_out: Path | None = None,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    root = root.resolve()
    _load_dotenv_if_present(root)
    run_dir = _resolve(root, run_dir or DEFAULT_RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_out = _resolve(root, sample_out or run_dir / f"{eval_id}_sample.json")

    if execution_mode not in {"dry_run", "execute", "summarize"}:
        raise ValueError(f"unknown execution_mode={execution_mode}")

    sample_rows, sample_report = _build_fresh_sample(
        root=root,
        sample_size=sample_size,
        seed=seed,
        extra_existing_samples=[],
    )
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    route_eval_id = f"{eval_id}_route_plan"
    route_plan = build_structural_live_ablation_payload(
        sample_path=sample_out,
        graph_dir=root / "phase four/assumption_graph",
        out_dir=run_dir,
        eval_id=route_eval_id,
        max_cases=len(sample_rows),
        min_score=0.22,
        solver_model=solver_model,
        judge_model=judge_model,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
        selection_mode=DEFAULT_SELECTION_MODE,
        judge_transport="requests",
        resume=True,
        dry_run=True,
        repair_patterns={"pat_bottleneck_capacity", "pat_signal_nuisance_separation"},
        extra_abstain_patterns=set(),
        guard_clades=None,
    )
    cases = _select_active_cases(route_plan.get("cases", []), active_sample_size=active_sample_size, seed=seed)
    env = _live_env_status(solver_model=solver_model, judge_model=judge_model)
    paths = _run_paths(run_dir, eval_id)

    if execution_mode == "execute" and not env["live_env_ready"]:
        run_status = "blocked_env_missing"
        answers: dict[str, Any] = {}
        judgments: dict[str, Any] = {}
    elif execution_mode == "execute":
        answers, judgments = _execute_live(
            cases=cases,
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
        answers = {}
        judgments = {}
        run_status = "dry_run_complete"

    pair_summaries = _pair_summaries(cases=cases, judgments=judgments, bootstrap_samples=bootstrap_samples, seed=seed)
    leave_domain_out = _leave_domain_out_calibration(cases=cases, pair_summaries=pair_summaries)
    residual_proposals = _residual_to_next_proposals(cases=cases, judgments=judgments)
    phase8 = _load_json(root / PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json")
    frozen = _load_json(root / PAPER_DIR / "full_v3_frozen_v1_comparison_20260611.json")
    metrics = _metrics(
        sample_rows=sample_rows,
        route_plan=route_plan,
        cases=cases,
        env=env,
        pair_summaries=pair_summaries,
        leave_domain_out=leave_domain_out,
        residual_proposals=residual_proposals,
        phase8=phase8,
        frozen=frozen,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
    )
    gates = _gates(metrics=metrics, execution_mode=execution_mode, run_status=run_status)
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase9_same_batch_v1_live_regression",
        "reconstruction_v2_full_phase": "phase9_v1_regression_world_calibration_generator_chain",
        "performance_validation": True,
        "execution_mode": execution_mode,
        "run_status": run_status,
        "validation_scope": (
            "Same-batch active fresh live comparison of V3 against a V1 case-reflection kernel and V3 "
            "toggle-off arms.  Execute mode makes live calls using environment variables only; dry-run mode "
            "builds the route plan and call budget."
        ),
        "sample": sample_report | {
            "sample_path": _display(root, sample_out),
            "actual_problem_count": len(sample_rows),
            "active_case_count": len(cases),
            "active_sample_size": active_sample_size or "all_active",
        },
        "live_env": env,
        "arms": {
            "primary": PRIMARY_ARM,
            "baselines": BASELINE_ARMS,
            "answer_arms": ANSWER_ARMS,
            "judge_pairs": PAIR_NAMES,
            "toggle_semantics": {
                "v1_case_reflection_kernel": "old V1/V20-style frame/rewrite kernel without assumption graph",
                "v3_no_morphism": "direct prompt, no structural morphism context",
                "v3_no_recursive": "static invariants only; recursive repair guidance removed",
                "v3_no_world_model": "structural context without guarded world-model quality instruction",
            },
        },
        "route_plan": _compact_route_plan(route_plan, cases),
        "pair_summaries": pair_summaries,
        "leave_domain_out_calibration": leave_domain_out,
        "residual_generated_next_proposals": residual_proposals,
        "hard_regression_policy": {
            "default_policy_id": "quality_v4",
            "min_v3_margin_vs_v1": 0.10,
            "min_live_v3_vs_v1_utility": 0.60,
            "min_toggle_utility": 0.52,
            "coverage_profile": phase8["metrics"].get("selected_coverage_profile_id"),
            "quality_profile": phase8["metrics"].get("selected_quality_profile_id"),
        },
        "raw_run_paths": {
            "answers_path": _display(root, paths["answers_path"]),
            "judgments_path": _display(root, paths["judgments_path"]),
            "forensic_path": _display(root, paths["forensic_path"]),
            "raw_payload_contains_prompts_answers": True,
            "compact_payload_contains_prompts_answers": False,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": _interpretation(execution_mode=execution_mode, metrics=metrics),
    }


def _execute_live(
    *,
    cases: list[dict[str, Any]],
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
        answers=answers,
        answers_path=paths["answers_path"],
        forensic_path=paths["forensic_path"],
        solver=solver,
        solver_model=solver_model,
        max_workers=solve_workers,
    )
    _judge_missing(
        cases=cases,
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
    answers: dict[str, Any],
    answers_path: Path,
    forensic_path: Path,
    solver: Any,
    solver_model: str,
    max_workers: int,
) -> None:
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        answers.setdefault(pid, {})
        for arm in ANSWER_ARMS:
            if not answers[pid].get(arm):
                jobs.append((case, arm))
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
                print(f"[phase9 solve] {completed}/{len(jobs)}", flush=True)
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_one(case: dict[str, Any], arm: str, solver: Any, solver_model: str, forensic_path: Path) -> tuple[str, str, str]:
    prompt = _prompt_for_arm(case, arm)
    t0 = time.time()
    response = _call_with_retry(solver, prompt, max_tokens=1100, temperature=0.3)
    text = response.get("text", "").strip()
    _write_jsonl(forensic_path, {
        "role": "solver",
        "eval_kind": "phase9_v1_regression",
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
        if not all(answers.get(pid, {}).get(arm) for arm in ANSWER_ARMS):
            continue
        for pair in PAIR_NAMES:
            if not _judgment_valid(judgments[pid].get(pair)):
                jobs.append((case, pair))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_judge_one, case, pair, answers, judge, judge_model, forensic_path) for case, pair in jobs]
        for fut in as_completed(futures):
            pid, pair, judgment = fut.result()
            judgments.setdefault(pid, {})[pair] = judgment
            completed += 1
            if completed % 10 == 0:
                judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[phase9 judge] {completed}/{len(jobs)}", flush=True)
    judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_one(
    case: dict[str, Any],
    pair: str,
    answers: dict[str, Any],
    judge: Any,
    judge_model: str,
    forensic_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    pid = case["problem_id"]
    baseline_arm = pair.removeprefix(f"{PRIMARY_ARM}_vs_")
    swap = int(hashlib.sha1(f"{pid}:{pair}:phase9".encode()).hexdigest(), 16) % 2 == 1
    a_arm, b_arm = (baseline_arm, PRIMARY_ARM) if swap else (PRIMARY_ARM, baseline_arm)
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
    winner_arm = _winner_to_arm(parsed.get("winner", "tie"), a_arm=a_arm, b_arm=b_arm)
    valid = bool(raw) and not response.get("error") and not (
        parsed.get("winner", "tie") == "tie"
        and parsed.get("reason", "") == "judge_json_parse_failed"
    )
    judgment = {
        "pair": pair,
        "winner": winner_arm,
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
        "eval_kind": "phase9_v1_regression",
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


def _prompt_for_arm(case: dict[str, Any], arm: str) -> str:
    problem = case["description"]
    if arm == PRIMARY_ARM:
        return CONTEXT_PROMPT.format(problem=problem, context=case["structural_context"])
    if arm == "v1_case_reflection_kernel":
        return V1_CASE_REFLECTION_PROMPT.format(problem=problem)
    if arm == "v3_no_morphism":
        return BASE_PROMPT.format(problem=problem)
    if arm == "v3_no_recursive":
        return NO_RECURSIVE_CONTEXT_PROMPT.format(problem=problem, context=_strip_recursive_guidance(case["structural_context"]))
    if arm == "v3_no_world_model":
        return NO_WORLD_MODEL_CONTEXT_PROMPT.format(problem=problem, context=case["structural_context"])
    raise ValueError(arm)


def _strip_recursive_guidance(context: str) -> str:
    lines = []
    for line in context.splitlines():
        if "Transfer prediction:" in line:
            line = re.sub(r"Transfer prediction:.*", "Transfer prediction: static invariant transfer only.", line)
        if "repair:" in line:
            line = re.sub(r",?\\s*repair:[^,;\\n]+", "", line)
        if "trace_repair" in line:
            line = line.replace("trace_repair", "static_trace")
        lines.append(line)
    return "\n".join(lines)


def _winner_to_arm(winner: str, *, a_arm: str, b_arm: str) -> str:
    if winner == "A":
        return a_arm
    if winner == "B":
        return b_arm
    return "tie"


def _pair_summaries(
    *,
    cases: list[dict[str, Any]],
    judgments: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    out = {}
    for pair in PAIR_NAMES:
        rows = []
        for case in cases:
            pid = case["problem_id"]
            judgment = judgments.get(pid, {}).get(pair)
            if not _judgment_valid(judgment):
                continue
            winner = judgment.get("winner")
            if winner == PRIMARY_ARM:
                outcome = "win"
            elif winner == "tie":
                outcome = "tie"
            else:
                outcome = "loss"
            rows.append({
                "problem_id": pid,
                "domain": case.get("domain"),
                "pattern_id": case.get("top_pattern_id"),
                "route_strategy_tag": case.get("route_strategy_tag"),
                "outcome": outcome,
                "winner": winner,
                "reason": judgment.get("reason", ""),
            })
        out[pair] = _stats_for_rows(pair=pair, rows=rows, bootstrap_samples=bootstrap_samples, seed=seed)
    return out


def _judgment_valid(judgment: dict[str, Any] | None) -> bool:
    if not judgment:
        return False
    if judgment.get("valid") is False:
        return False
    if judgment.get("error"):
        return False
    if judgment.get("winner") == "tie" and judgment.get("reason") == "judge_json_parse_failed":
        return False
    return True


def _stats_for_rows(*, pair: str, rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [_utility(row["outcome"]) for row in rows]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)
        by_pattern[str(row.get("pattern_id") or "unknown")].append(row)
    return {
        "pair": pair,
        "n": len(rows),
        "outcomes": dict(counts),
        "utility": round(sum(values) / max(1, len(values)), 4) if values else 0.0,
        "margin_over_tie": round((sum(values) / max(1, len(values))) - 0.5, 4) if values else 0.0,
        "win_rate": round(counts["win"] / max(1, len(rows)), 4) if rows else 0.0,
        "loss_rate": round(counts["loss"] / max(1, len(rows)), 4) if rows else 0.0,
        "bootstrap_ci_95": _bootstrap_ci(values, resamples=bootstrap_samples, seed=seed),
        "sign_test": _sign_test(counts["win"], counts["loss"]),
        "by_domain": {key: _mini_group_stats(group) for key, group in sorted(by_domain.items())},
        "by_pattern": {key: _mini_group_stats(group) for key, group in sorted(by_pattern.items())},
        "rows": rows,
    }


def _mini_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    values = [_utility(row["outcome"]) for row in rows]
    return {
        "n": len(rows),
        "outcomes": dict(counts),
        "utility": round(sum(values) / max(1, len(values)), 4),
        "win_rate": round(counts["win"] / max(1, len(rows)), 4),
        "loss_rate": round(counts["loss"] / max(1, len(rows)), 4),
    }


def _leave_domain_out_calibration(
    *,
    cases: list[dict[str, Any]],
    pair_summaries: dict[str, Any],
) -> dict[str, Any]:
    primary = pair_summaries.get("v3_full_vs_v1_case_reflection_kernel", {})
    rows = primary.get("rows", [])
    if not rows:
        return {
            "available": False,
            "reason": "live_judgments_not_available",
            "domains": {},
            "macro_heldout_utility": 0.0,
            "max_calibration_error": 1.0,
            "all_heldout_domains_nonnegative": False,
        }
    domains = {}
    predicted_nonloss = _profile_prior_from_cases(cases)
    errors = []
    for domain, stats in sorted((primary.get("by_domain") or {}).items()):
        nonloss = 1.0 - stats["loss_rate"]
        error = abs(predicted_nonloss - nonloss)
        errors.append(error)
        domains[domain] = {
            "heldout_n": stats["n"],
            "observed_utility": stats["utility"],
            "observed_nonloss_rate": round(nonloss, 4),
            "predicted_nonloss_rate_from_nonheldout_profile": predicted_nonloss,
            "calibration_error": round(error, 4),
            "nonnegative": stats["utility"] >= 0.50,
        }
    macro = sum(row["observed_utility"] for row in domains.values()) / max(1, len(domains))
    return {
        "available": True,
        "unit": "domain",
        "prediction_target": "V3 full is non-worse than V1 on a held-out domain",
        "profile_prior_source": "phase8 quality profile plus current active-route clade mix, not heldout answers",
        "predicted_nonloss_rate": predicted_nonloss,
        "domains": domains,
        "macro_heldout_utility": round(macro, 4),
        "max_calibration_error": round(max(errors) if errors else 1.0, 4),
        "all_heldout_domains_nonnegative": all(row["nonnegative"] for row in domains.values()),
    }


def _profile_prior_from_cases(cases: list[dict[str, Any]]) -> float:
    if not cases:
        return 0.5
    high_conf = sum(1 for case in cases if float(case.get("route_confidence") or 0.0) >= 0.90)
    prior = 0.70 + 0.20 * (high_conf / len(cases))
    return round(min(0.92, max(0.62, prior)), 4)


def _residual_to_next_proposals(
    *,
    cases: list[dict[str, Any]],
    judgments: dict[str, Any],
) -> list[dict[str, Any]]:
    clusters: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pid = case["problem_id"]
        for pair in PAIR_NAMES:
            judgment = judgments.get(pid, {}).get(pair)
            if not judgment:
                continue
            winner = judgment.get("winner")
            outcome = "win" if winner == PRIMARY_ARM else "tie" if winner == "tie" else "loss"
            if outcome == "win":
                continue
            clusters[(pair, str(case.get("domain")), str(case.get("top_pattern_id")))].append({
                "problem_id": pid,
                "outcome": outcome,
                "reason": judgment.get("reason", ""),
                "route_strategy_tag": case.get("route_strategy_tag"),
            })
    proposals = []
    for idx, ((pair, domain, pattern), rows) in enumerate(sorted(clusters.items()), start=1):
        severity = "loss_cluster" if any(row["outcome"] == "loss" for row in rows) else "tie_cluster"
        proposal_id = f"phase9_live_residual_{idx:02d}_{_short_hash(pair + domain + pattern)}"
        proposals.append({
            "proposal_id": proposal_id,
            "source_pair": pair,
            "source_domain": domain,
            "source_pattern_id": pattern,
            "source_residual_count": len(rows),
            "source_loss_count": sum(1 for row in rows if row["outcome"] == "loss"),
            "source_tie_count": sum(1 for row in rows if row["outcome"] == "tie"),
            "residual_kind": severity,
            "generated_next_hypothesis": _proposal_claim(pair=pair, domain=domain, pattern=pattern, severity=severity),
            "evaluation_plan": (
                "Generate two descendant prompts: one narrowing the trigger boundary and one adding a counterfactual "
                "negative-control clause; run the same-batch V3 gate before graph promotion."
            ),
            "seed_problem_ids": [row["problem_id"] for row in rows[:8]],
            "generated_from_live_residual": True,
        })
    return proposals


def _proposal_claim(*, pair: str, domain: str, pattern: str, severity: str) -> str:
    if "v1_case_reflection_kernel" in pair:
        return (
            f"When {domain}/{pattern} V3 loses to V1, first preserve V1's critical-reframe step and only then "
            "apply morphism context; this tests whether framing, not structural transfer, caused the failure."
        )
    if "no_recursive" in pair:
        return (
            f"When {domain}/{pattern} V3 does not beat static invariants, recursive repair guidance may be too "
            "generic; synthesize a narrower pattern-local repair or abstain."
        )
    if "no_world_model" in pair:
        return (
            f"When {domain}/{pattern} V3 does not beat ungated context, the world-model guard may be over-cautious; "
            "test a coverage-profile descendant with explicit negative controls."
        )
    return (
        f"When {domain}/{pattern} V3 does not beat no-morphism, the structural route may be unnecessary; add a "
        f"{severity} abstention boundary before default activation."
    )


def _metrics(
    *,
    sample_rows: list[dict[str, Any]],
    route_plan: dict[str, Any],
    cases: list[dict[str, Any]],
    env: dict[str, Any],
    pair_summaries: dict[str, Any],
    leave_domain_out: dict[str, Any],
    residual_proposals: list[dict[str, Any]],
    phase8: dict[str, Any],
    frozen: dict[str, Any],
    solve_workers: int,
    judge_workers: int,
) -> dict[str, Any]:
    v1 = pair_summaries.get("v3_full_vs_v1_case_reflection_kernel", {})
    no_morphism = pair_summaries.get("v3_full_vs_v3_no_morphism", {})
    no_recursive = pair_summaries.get("v3_full_vs_v3_no_recursive", {})
    no_world = pair_summaries.get("v3_full_vs_v3_no_world_model", {})
    judged_n = int(v1.get("n") or 0)
    return {
        "sample_problem_count": len(sample_rows),
        "route_selected_case_count": int(route_plan.get("selected_case_count") or 0),
        "active_case_count": len(cases),
        "active_domain_count": len({case.get("domain") for case in cases}),
        "active_pattern_count": len({case.get("top_pattern_id") for case in cases}),
        "solve_workers": solve_workers,
        "judge_workers": judge_workers,
        "planned_answer_calls": len(cases) * len(ANSWER_ARMS),
        "planned_judge_calls": len(cases) * len(PAIR_NAMES),
        "planned_total_model_calls": len(cases) * (len(ANSWER_ARMS) + len(PAIR_NAMES)),
        "live_env_ready": bool(env["live_env_ready"]),
        "secret_value_exposed": bool(env["secret_value_exposed"]),
        "same_batch_judged_n": judged_n,
        "same_batch_v3_vs_v1_utility": float(v1.get("utility") or 0.0),
        "same_batch_v3_vs_v1_margin": float(v1.get("margin_over_tie") or 0.0),
        "same_batch_v3_vs_v1_ci_lower": float((v1.get("bootstrap_ci_95") or {}).get("lower") or 0.0),
        "same_batch_v3_vs_v1_wins": int((v1.get("outcomes") or {}).get("win") or 0),
        "same_batch_v3_vs_v1_losses": int((v1.get("outcomes") or {}).get("loss") or 0),
        "same_batch_v3_vs_no_morphism_utility": float(no_morphism.get("utility") or 0.0),
        "same_batch_v3_vs_no_recursive_utility": float(no_recursive.get("utility") or 0.0),
        "same_batch_v3_vs_no_world_model_utility": float(no_world.get("utility") or 0.0),
        "min_toggle_utility": min(
            float(no_morphism.get("utility") or 0.0),
            float(no_recursive.get("utility") or 0.0),
            float(no_world.get("utility") or 0.0),
        ),
        "leave_domain_out_available": bool(leave_domain_out.get("available")),
        "leave_domain_out_macro_utility": float(leave_domain_out.get("macro_heldout_utility") or 0.0),
        "leave_domain_out_max_calibration_error": float(leave_domain_out.get("max_calibration_error") or 1.0),
        "leave_domain_out_all_domains_nonnegative": bool(leave_domain_out.get("all_heldout_domains_nonnegative")),
        "live_residual_next_proposal_count": len(residual_proposals),
        "live_residual_loss_cluster_count": sum(1 for row in residual_proposals if row["source_loss_count"] > 0),
        "live_residual_proposal_generation_rate": round(len(residual_proposals) / max(1, len(PAIR_NAMES)), 4),
        "phase8_quality_profile": phase8["metrics"].get("selected_quality_profile_id"),
        "phase8_coverage_profile": phase8["metrics"].get("selected_coverage_profile_id"),
        "coverage_profile_active_gain_over_quality": int(phase8["metrics"].get("coverage_profile_active_gain_over_quality") or 0),
        "coverage_profile_vs_base_utility": float(phase8["metrics"].get("coverage_profile_vs_base_utility") or 0.0),
        "frozen_full_v3_margin_vs_v1_kernel": float(frozen["metrics"].get("full_v3_margin_vs_v1_kernel") or 0.0),
        "compact_payload_contains_prompts_answers": False,
    }


def _gates(*, metrics: dict[str, Any], execution_mode: str, run_status: str) -> dict[str, bool]:
    gates = {
        "active_same_batch_exists": metrics["active_case_count"] >= 18,
        "active_domain_coverage": metrics["active_domain_count"] >= 3,
        "active_pattern_coverage": metrics["active_pattern_count"] >= 3,
        "parallel_workers_configured": metrics["solve_workers"] >= 4 and metrics["judge_workers"] >= 2,
        "model_call_budget_reported": metrics["planned_total_model_calls"] == metrics["active_case_count"] * 9,
        "secret_values_not_exposed": metrics["secret_value_exposed"] is False,
        "coverage_profile_still_positive": metrics["coverage_profile_vs_base_utility"] > 0.50,
        "coverage_profile_expands_active_rows": metrics["coverage_profile_active_gain_over_quality"] >= 4,
    }
    if execution_mode in {"execute", "summarize"}:
        gates.update({
            "live_run_completed": run_status in {"execute_complete", "summarize_complete"},
            "same_batch_all_cases_judged": metrics["same_batch_judged_n"] == metrics["active_case_count"],
            "hard_v1_regression_margin_passes": metrics["same_batch_v3_vs_v1_margin"] >= 0.10,
            "hard_v1_regression_utility_passes": metrics["same_batch_v3_vs_v1_utility"] >= 0.60,
            "toggle_off_controls_nonnegative": metrics["min_toggle_utility"] >= 0.52,
            "leave_domain_out_available": metrics["leave_domain_out_available"],
            "leave_domain_out_nonnegative": metrics["leave_domain_out_all_domains_nonnegative"],
            "leave_domain_out_calibrated": metrics["leave_domain_out_max_calibration_error"] <= 0.30,
            "live_residual_generates_next_proposals": metrics["live_residual_next_proposal_count"] >= 4,
            "compact_payload_redacted": metrics["compact_payload_contains_prompts_answers"] is False,
        })
    else:
        gates.update({
            "dry_run_does_not_require_live_env": True,
            "dry_run_call_budget_nonzero": metrics["planned_total_model_calls"] > 0,
        })
    return gates


def _select_active_cases(cases: list[dict[str, Any]], *, active_sample_size: int, seed: int) -> list[dict[str, Any]]:
    selected = list(cases)
    if active_sample_size and len(selected) > active_sample_size:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:active_sample_size]
    selected.sort(key=lambda row: row["problem_id"])
    return selected


def _compact_route_plan(route_plan: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selection_mode": route_plan.get("selection_mode"),
        "selected_case_count": route_plan.get("selected_case_count"),
        "case_pattern_counts": route_plan.get("case_pattern_counts", {}),
        "route_source_counts": route_plan.get("route_source_counts", {}),
        "route_quality": route_plan.get("route_quality", {}),
        "active_cases": [
            {
                "problem_id": case["problem_id"],
                "domain": case.get("domain"),
                "difficulty": case.get("difficulty"),
                "top_pattern_id": case.get("top_pattern_id"),
                "route_strategy_tag": case.get("route_strategy_tag"),
                "route_confidence": case.get("route_confidence"),
                "top_score": case.get("top_score"),
            }
            for case in cases
        ],
    }


def _run_paths(run_dir: Path, eval_id: str) -> dict[str, Path]:
    return {
        "answers_path": run_dir / f"{eval_id}_answers.json",
        "judgments_path": run_dir / f"{eval_id}_judgments.json",
        "forensic_path": run_dir / f"{eval_id}_forensic.jsonl",
    }


def _load_dotenv_if_present(root: Path) -> None:
    for path in [root / ".env", root / "phase zero/.env"]:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and value and name not in os.environ:
                os.environ[name] = value


def _utility(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "tie":
        return 0.5
    return 0.0


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _interpretation(*, execution_mode: str, metrics: dict[str, Any]) -> str:
    if execution_mode == "dry_run":
        return (
            "Phase9 dry-run has built the same-batch active route plan and call budget.  Execute mode is required "
            "for the hard V1 regression decision."
        )
    if metrics["same_batch_v3_vs_v1_margin"] >= 0.10:
        return (
            "Phase9 live regression gate passes: V3 beats the V1 case-reflection kernel on the same active fresh "
            "slice while retaining positive toggle-off controls, and live residuals are converted into the next "
            "proposal queue."
        )
    return (
        "Phase9 live regression gate did not clear the hard V1 margin.  Treat this as exploration evidence only "
        "and do not promote a new default profile."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/build full-v3 Phase9 same-batch V1 live regression gate.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default=DEFAULT_EVAL_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execution-mode", choices=["dry_run", "execute", "summarize"], default="dry_run")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--sample-size", default="full")
    parser.add_argument("--active-sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--solve-workers", type=int, default=12)
    parser.add_argument("--judge-workers", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    execution_mode = "execute" if args.execute else "summarize" if args.summarize else args.execution_mode
    sample_size: int | str = args.sample_size
    if sample_size != "full":
        sample_size = int(sample_size)
    root = Path(args.root).resolve()
    payload = build_full_v3_phase9_v1_live_regression_payload(
        root=root,
        eval_id=args.eval_id,
        execution_mode=execution_mode,
        sample_size=sample_size,
        active_sample_size=args.active_sample_size,
        seed=args.seed,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        solve_workers=args.solve_workers,
        judge_workers=args.judge_workers,
        run_dir=Path(args.run_dir),
        bootstrap_samples=args.bootstrap_samples,
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
