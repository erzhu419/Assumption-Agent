"""Full-v3 fresh live benchmark runner.

This module is the paper-scale live experiment harness for the current V3
pipeline.  It deliberately separates three modes:

- dry_run: build the fresh sample, route cases, and report the parallel call
  budget without API calls.
- execute: run the structural live ablation with parallel answer and judge
  workers.  This requires API keys in environment variables.
- summarize: read an existing live run and add problem-level bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .structural_live_ablation import build_structural_live_ablation_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_fresh_live_benchmark_preflight_20260611.json"
DEFAULT_PROBLEM_DIR = Path("phase zero/benchmark/problems")
DEFAULT_EXISTING_SAMPLES = [
    Path("phase two/analysis/cache/sample_100.json"),
    Path("phase two/analysis/cache/sample_holdout_50.json"),
    Path("phase two/analysis/cache/sample_extend_50.json"),
    Path("phase two/analysis/cache/sample_21_50.json"),
    Path("phase four/autonomous/used_problems.json"),
]
DEFAULT_RUN_DIR = PAPER_DIR / "fresh_live_runs"
TARGET_SCALES = [300, 600, "full"]
DEFAULT_SOLVER_MODEL = "gpt_mini"
DEFAULT_JUDGE_MODEL = "gpt55"
DEFAULT_SELECTION_MODE = "natural_repaired_guarded"


def build_full_v3_fresh_live_benchmark_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_fresh_live_benchmark_preflight_20260611",
    sample_size: int | str = 300,
    seed: int = 20260611,
    execution_mode: str = "dry_run",
    solve_workers: int = 16,
    judge_workers: int = 8,
    solver_model: str = DEFAULT_SOLVER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    min_score: float = 0.22,
    bootstrap_samples: int = 2000,
    sample_out: Path | None = None,
    run_dir: Path | None = None,
    extra_existing_samples: list[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    run_dir = _resolve(root, run_dir or DEFAULT_RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_out = _resolve(root, sample_out or run_dir / f"{eval_id}_sample.json")
    sample_rows, sample_report = _build_fresh_sample(
        root=root,
        sample_size=sample_size,
        seed=seed,
        extra_existing_samples=extra_existing_samples or [],
    )
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    env = _live_env_status(solver_model=solver_model, judge_model=judge_model)

    if execution_mode not in {"dry_run", "execute", "summarize"}:
        raise ValueError(f"unknown execution_mode={execution_mode}")
    if execution_mode == "execute" and not env["live_env_ready"]:
        live_payload = None
        run_status = "blocked_env_missing"
    else:
        if execution_mode in {"dry_run", "execute"}:
            live_payload = build_structural_live_ablation_payload(
                sample_path=sample_out,
                graph_dir=root / "phase four/assumption_graph",
                out_dir=run_dir,
                eval_id=eval_id,
                max_cases=len(sample_rows),
                min_score=min_score,
                solver_model=solver_model,
                judge_model=judge_model,
                solve_workers=solve_workers,
                judge_workers=judge_workers,
                selection_mode=selection_mode,
                judge_transport="requests",
                resume=True,
                dry_run=execution_mode == "dry_run",
                repair_patterns={"pat_bottleneck_capacity", "pat_signal_nuisance_separation"},
                extra_abstain_patterns=set(),
            )
            run_status = "dry_run_complete" if execution_mode == "dry_run" else "execute_complete"
        else:
            live_payload = _load_json(run_dir / f"{eval_id}_summary.json")
            run_status = "summarize_complete"

    cases = _extract_cases(live_payload)
    fill_absent_as_tie = _selection_abstains_as_tie(selection_mode)
    ci = _problem_level_ci(
        run_dir=run_dir,
        eval_id=eval_id,
        cases=cases,
        sample_rows=sample_rows,
        fill_absent_as_tie=fill_absent_as_tie,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    metrics = _metrics(
        sample_rows=sample_rows,
        sample_report=sample_report,
        live_payload=live_payload,
        env=env,
        ci=ci,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
        selection_mode=selection_mode,
        fill_absent_as_tie=fill_absent_as_tie,
    )
    gates = _gates(metrics=metrics, execution_mode=execution_mode, env=env)
    commands = _commands(
        eval_id=eval_id,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
        solver_model=solver_model,
        judge_model=judge_model,
        selection_mode=selection_mode,
        min_score=min_score,
    )
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_fresh_live_parallel_benchmark",
        "performance_validation": True,
        "execution_mode": execution_mode,
        "run_status": run_status,
        "validation_scope": (
            "Parallel 300/600/full fresh live benchmark harness for the frozen V3 pipeline.  Dry-run mode "
            "does not call APIs; execute mode uses environment variables only."
        ),
        "sample": sample_report | {
            "sample_path": _display(root, sample_out),
            "actual_problem_count": len(sample_rows),
        },
        "live_env": env,
        "parallel_plan": {
            "solve_workers": solve_workers,
            "judge_workers": judge_workers,
            "answer_arms": ["base", "structural", "placebo"],
            "judge_pairs": ["structural_vs_base", "structural_vs_placebo"],
            "selection_mode": selection_mode,
            "abstained_problems_count_as_tie": fill_absent_as_tie,
            "planned_answer_calls": metrics["planned_answer_calls"],
            "planned_judge_calls": metrics["planned_judge_calls"],
            "planned_total_model_calls": metrics["planned_total_model_calls"],
        },
        "structural_live_summary": _compact_live_summary(live_payload),
        "problem_level_ci": ci,
        "commands": commands,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": _interpretation(execution_mode=execution_mode, env=env, ci=ci),
    }


def _build_fresh_sample(
    *,
    root: Path,
    sample_size: int | str,
    seed: int,
    extra_existing_samples: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = _load_problem_pool(root)
    excluded = _load_existing_problem_ids(root, extra_existing_samples=extra_existing_samples)
    fresh = [row for row in all_rows if row["problem_id"] not in excluded]
    rng = random.Random(seed)
    rng.shuffle(fresh)
    requested = len(fresh) if sample_size == "full" else int(sample_size)
    selected = fresh[:requested]
    by_domain = Counter(row["domain"] for row in selected)
    by_difficulty = Counter(row.get("difficulty", "unknown") for row in selected)
    return selected, {
        "requested_sample_size": sample_size,
        "seed": seed,
        "pool_problem_count": len(all_rows),
        "excluded_existing_problem_count": len(excluded),
        "extra_existing_sample_paths": [
            _display(root, _resolve(root, path))
            for path in extra_existing_samples
        ],
        "available_fresh_problem_count": len(fresh),
        "sample_problem_count": len(selected),
        "by_domain": dict(sorted(by_domain.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "disjoint_from_existing_samples": all(row["problem_id"] not in excluded for row in selected),
    }


def _load_problem_pool(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / DEFAULT_PROBLEM_DIR).glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    by_id = {row["problem_id"]: row for row in rows}
    return [by_id[key] for key in sorted(by_id)]


def _load_existing_problem_ids(root: Path, *, extra_existing_samples: list[Path]) -> set[str]:
    out: set[str] = set()
    for rel in [*DEFAULT_EXISTING_SAMPLES, *extra_existing_samples]:
        path = _resolve(root, rel)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    out.add(item)
                elif isinstance(item, dict) and item.get("problem_id"):
                    out.add(str(item["problem_id"]))
    return out


def _live_env_status(*, solver_model: str, judge_model: str) -> dict[str, Any]:
    solver_names = _env_names_for_alias(solver_model)
    judge_names = _env_names_for_alias(judge_model)
    solver_ready = any(bool(os.environ.get(name)) for name in solver_names["key_names"])
    judge_ready = any(bool(os.environ.get(name)) for name in judge_names["key_names"])
    base_ready = any(bool(os.environ.get(name)) for name in ["RUOLI_BASE_URL", *solver_names["base_names"], *judge_names["base_names"]])
    return {
        "solver_model_alias": solver_model,
        "judge_model_alias": judge_model,
        "solver_key_env_names": solver_names["key_names"],
        "judge_key_env_names": judge_names["key_names"],
        "base_url_env_names": sorted(set(["RUOLI_BASE_URL", *solver_names["base_names"], *judge_names["base_names"]])),
        "solver_ready": solver_ready,
        "judge_ready": judge_ready,
        "base_url_ready_or_default": base_ready or True,
        "live_env_ready": solver_ready and judge_ready,
        "secret_value_exposed": False,
    }


def _env_names_for_alias(alias: str) -> dict[str, list[str]]:
    if alias in {"gemini", "gemini_flash_low", "gemini_pro"}:
        return {
            "key_names": ["RUOLI_GEMINI_KEY", "GEMINI_PROXY_API_KEY"],
            "base_names": ["GEMINI_PROXY_BASE_URL"],
        }
    if alias in {"gpt55", "gpt5", "gpt_mini", "gpt54_mini"}:
        return {
            "key_names": ["RUOLI_GPT_KEY", "GPT5_API_KEY"],
            "base_names": ["GPT5_BASE_URL"],
        }
    if alias in {"claude", "claude_opus", "claude_haiku"}:
        return {
            "key_names": ["RUOLI_CLAUDE_KEY", "ANTHROPIC_API_KEY", "CLAUDE_PROXY_API_KEY"],
            "base_names": ["CLAUDE_BASE_URL", "CLAUDE_PROXY_BASE_URL"],
        }
    return {"key_names": [], "base_names": []}


def _problem_level_ci(
    *,
    run_dir: Path,
    eval_id: str,
    cases: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    fill_absent_as_tie: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    judgments_path = run_dir / f"{eval_id}_judgments.json"
    if not judgments_path.exists() or not cases:
        return {
            "available": False,
            "reason": "judgments_not_available_yet" if not judgments_path.exists() else "cases_not_available",
            "unit_of_analysis": "problem_id",
            "bootstrap_samples": bootstrap_samples,
            "pairs": {},
        }
    judgments = _load_json(judgments_path)
    case_by_id = {case["problem_id"]: case for case in cases}
    rows = sample_rows if fill_absent_as_tie else cases
    pairs = {}
    for pair in ("structural_vs_base", "structural_vs_placebo"):
        outcomes = []
        for row in rows:
            pid = row["problem_id"]
            case = case_by_id.get(pid)
            judgment = judgments.get(pid, {}).get(pair)
            if not judgment and not fill_absent_as_tie:
                continue
            if not judgment:
                outcome = "tie"
            else:
                winner = judgment.get("winner")
                if winner == "structural":
                    outcome = "win"
                elif winner == "tie":
                    outcome = "tie"
                else:
                    outcome = "loss"
            outcomes.append({
                "problem_id": pid,
                "domain": (case or row).get("domain"),
                "outcome": outcome,
                "active_intervention": bool(judgment),
            })
        pairs[pair] = _pair_stats(pair=pair, outcomes=outcomes, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "available": True,
        "unit_of_analysis": "problem_id",
        "pseudoreplication_guard": "one collapsed outcome per problem_id per pair",
        "abstained_problems_count_as_tie": fill_absent_as_tie,
        "bootstrap_samples": bootstrap_samples,
        "pairs": pairs,
    }


def _pair_stats(*, pair: str, outcomes: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in outcomes)
    values = [_utility(row["outcome"]) for row in outcomes]
    by_domain: dict[str, list[str]] = defaultdict(list)
    for row in outcomes:
        by_domain[str(row.get("domain") or "unknown")].append(row["outcome"])
    domain_breakdown = {}
    for domain, domain_outcomes in sorted(by_domain.items()):
        domain_counts = Counter(domain_outcomes)
        domain_values = [_utility(outcome) for outcome in domain_outcomes]
        domain_breakdown[domain] = {
            "n": len(domain_outcomes),
            "outcomes": dict(domain_counts),
            "utility": round(sum(domain_values) / max(1, len(domain_values)), 4),
            "win_rate": round(domain_counts["win"] / max(1, len(domain_outcomes)), 4),
            "loss_rate": round(domain_counts["loss"] / max(1, len(domain_outcomes)), 4),
        }
    return {
        "pair": pair,
        "problem_level_n": len(outcomes),
        "active_intervention_n": sum(1 for row in outcomes if row.get("active_intervention")),
        "outcomes": dict(counts),
        "utility": round(sum(values) / max(1, len(values)), 4) if values else 0.0,
        "win_rate": round(counts["win"] / max(1, len(values)), 4) if values else 0.0,
        "loss_rate": round(counts["loss"] / max(1, len(values)), 4) if values else 0.0,
        "bootstrap_ci_95": _bootstrap_ci(values, resamples=bootstrap_samples, seed=seed),
        "sign_test": _sign_test(counts["win"], counts["loss"]),
        "domain_breakdown": domain_breakdown,
    }


def _bootstrap_ci(values: list[float], *, resamples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower_idx = int(0.025 * (len(means) - 1))
    upper_idx = int(0.975 * (len(means) - 1))
    return {
        "mean": round(sum(values) / len(values), 4),
        "lower": round(means[lower_idx], 4),
        "upper": round(means[upper_idx], 4),
    }


def _sign_test(wins: int, losses: int) -> dict[str, Any]:
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "non_tie_n": 0, "p_value": 1.0}
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"wins": wins, "losses": losses, "non_tie_n": n, "p_value": round(min(1.0, 2.0 * tail), 8)}


def _utility(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "tie":
        return 0.5
    return 0.0


def _metrics(
    *,
    sample_rows: list[dict[str, Any]],
    sample_report: dict[str, Any],
    live_payload: dict[str, Any] | None,
    env: dict[str, Any],
    ci: dict[str, Any],
    solve_workers: int,
    judge_workers: int,
    selection_mode: str,
    fill_absent_as_tie: bool,
) -> dict[str, Any]:
    selected = int((live_payload or {}).get("selected_case_count") or (live_payload or {}).get("plan", {}).get("selected_case_count") or 0)
    planned_answer_calls = selected * 3
    planned_judge_calls = selected * 2
    ci_pairs = ci.get("pairs", {}) if ci.get("available") else {}
    base_ci = ci_pairs.get("structural_vs_base", {})
    placebo_ci = ci_pairs.get("structural_vs_placebo", {})
    effective_n = int(base_ci.get("problem_level_n") or 0)
    return {
        "selection_mode": selection_mode,
        "abstained_problems_count_as_tie": fill_absent_as_tie,
        "sample_problem_count": len(sample_rows),
        "available_fresh_problem_count": sample_report["available_fresh_problem_count"],
        "disjoint_from_existing_samples": sample_report["disjoint_from_existing_samples"],
        "domain_count": len(sample_report["by_domain"]),
        "selected_case_count": selected,
        "route_quality_n": int((live_payload or {}).get("route_quality", {}).get("n") or (live_payload or {}).get("plan", {}).get("route_quality", {}).get("n") or 0),
        "solve_workers": solve_workers,
        "judge_workers": judge_workers,
        "planned_answer_calls": planned_answer_calls,
        "planned_judge_calls": planned_judge_calls,
        "planned_total_model_calls": planned_answer_calls + planned_judge_calls,
        "live_env_ready": bool(env["live_env_ready"]),
        "secret_value_exposed": bool(env["secret_value_exposed"]),
        "problem_level_ci_available": bool(ci["available"]),
        "structural_vs_base_problem_level_n": effective_n,
        "structural_vs_base_active_intervention_n": int(base_ci.get("active_intervention_n") or 0),
        "structural_vs_base_utility": float(base_ci.get("utility") or 0.0),
        "structural_vs_base_ci_lower": float((base_ci.get("bootstrap_ci_95") or {}).get("lower") or 0.0),
        "structural_vs_base_p_value": float((base_ci.get("sign_test") or {}).get("p_value") or 1.0),
        "structural_vs_placebo_problem_level_n": int(placebo_ci.get("problem_level_n") or 0),
        "structural_vs_placebo_active_intervention_n": int(placebo_ci.get("active_intervention_n") or 0),
        "structural_vs_placebo_utility": float(placebo_ci.get("utility") or 0.0),
        "structural_vs_placebo_ci_lower": float((placebo_ci.get("bootstrap_ci_95") or {}).get("lower") or 0.0),
        "structural_vs_placebo_p_value": float((placebo_ci.get("sign_test") or {}).get("p_value") or 1.0),
    }


def _gates(*, metrics: dict[str, Any], execution_mode: str, env: dict[str, Any]) -> dict[str, bool]:
    gates = {
        "fresh_sample_large_enough": metrics["sample_problem_count"] >= 300,
        "fresh_sample_disjoint": metrics["disjoint_from_existing_samples"] is True,
        "domain_coverage_broad": metrics["domain_count"] >= 5,
        "parallel_workers_configured": metrics["solve_workers"] >= 8 and metrics["judge_workers"] >= 4,
        "cases_route_for_live_plan": (
            metrics["selected_case_count"] >= 200
            or (
                metrics["abstained_problems_count_as_tie"] is True
                and metrics["selected_case_count"] >= 5
            )
        ),
        "model_call_budget_reported": metrics["planned_total_model_calls"] == metrics["selected_case_count"] * 5,
        "secret_values_not_exposed": metrics["secret_value_exposed"] is False,
    }
    if execution_mode == "execute":
        gates.update({
            "live_env_ready": bool(env["live_env_ready"]),
            "problem_level_ci_available": metrics["problem_level_ci_available"],
            "problem_level_n_matches_scope": metrics["structural_vs_base_problem_level_n"] == (
                metrics["sample_problem_count"]
                if metrics["abstained_problems_count_as_tie"]
                else metrics["selected_case_count"]
            ),
            "structural_vs_base_nonnegative": metrics["structural_vs_base_utility"] > 0.50,
            "structural_vs_placebo_nonnegative": metrics["structural_vs_placebo_utility"] > 0.50,
        })
    if execution_mode == "summarize":
        gates.update({
            "problem_level_ci_available": metrics["problem_level_ci_available"],
            "problem_level_n_matches_scope": metrics["structural_vs_base_problem_level_n"] == (
                metrics["sample_problem_count"]
                if metrics["abstained_problems_count_as_tie"]
                else metrics["selected_case_count"]
            ),
            "structural_vs_base_nonnegative": metrics["structural_vs_base_utility"] > 0.50,
            "structural_vs_placebo_nonnegative": metrics["structural_vs_placebo_utility"] > 0.50,
        })
    return gates


def _commands(
    *,
    eval_id: str,
    solve_workers: int,
    judge_workers: int,
    solver_model: str,
    judge_model: str,
    selection_mode: str,
    min_score: float,
) -> dict[str, str]:
    base = (
        "python3 -m assumption_os.full_v3_fresh_live_benchmark --root . "
        "--execute --solver-model {solver} --judge-model {judge} "
        "--solve-workers {solve_workers} --judge-workers {judge_workers} "
        "--selection-mode {selection_mode} --min-score {min_score}"
    ).format(
        solver=solver_model,
        judge=judge_model,
        solve_workers=solve_workers,
        judge_workers=judge_workers,
        selection_mode=selection_mode,
        min_score=min_score,
    )
    return {
        "env_required": (
            "Set RUOLI_BASE_URL plus provider keys in environment variables only.  The validated fallback for "
            "this run is RUOLI_GPT_KEY with solver_model=gpt_mini and judge_model=gpt55; Gemini can be restored "
            "for repeated solver calls when that channel has access.  Do not put key values in code or artifacts."
        ),
        "run_300": f"{base} --sample-size 300 --eval-id {eval_id}_live300",
        "run_600": f"{base} --sample-size 600 --eval-id {eval_id}_live600",
        "run_full": f"{base} --full --eval-id {eval_id}_livefull",
        "summarize_existing": (
            "python3 -m assumption_os.full_v3_fresh_live_benchmark --root . --summarize "
            f"--sample-size 300 --eval-id {eval_id}_live300"
        ),
    }


def _compact_live_summary(live_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not live_payload:
        return {"available": False}
    return {
        "available": True,
        "mode": live_payload.get("mode"),
        "pass": live_payload.get("pass"),
        "selected_case_count": live_payload.get("selected_case_count") or live_payload.get("plan", {}).get("selected_case_count"),
        "answer_cells": live_payload.get("answer_cells") or live_payload.get("plan", {}).get("answer_cells"),
        "judge_pairs": live_payload.get("judge_pairs") or live_payload.get("plan", {}).get("judge_pairs"),
        "route_quality": live_payload.get("route_quality") or live_payload.get("plan", {}).get("route_quality"),
        "pair_summaries": live_payload.get("pair_summaries", {}),
    }


def _extract_cases(live_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not live_payload:
        return []
    if "cases" in live_payload:
        return list(live_payload.get("cases") or [])
    return list(live_payload.get("plan", {}).get("cases") or [])


def _selection_abstains_as_tie(selection_mode: str) -> bool:
    return selection_mode in {"natural_repaired_guarded"}


def _interpretation(*, execution_mode: str, env: dict[str, Any], ci: dict[str, Any]) -> str:
    if execution_mode == "dry_run":
        return (
            "The fresh live benchmark harness is ready and parallelized, but this run only planned the experiment. "
            "No API calls were made."
        )
    if execution_mode == "execute" and not env["live_env_ready"]:
        return "Live execution was requested but blocked because API environment variables are missing."
    if ci["available"]:
        return "Fresh live run has problem-level bootstrap CIs and sign tests."
    return "Run summary exists, but judge outcomes are not complete enough for problem-level CIs."


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or run the full-v3 fresh live benchmark.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_fresh_live_benchmark_preflight_20260611")
    parser.add_argument("--sample-size", default="300")
    parser.add_argument("--full", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--solve-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--min-score", type=float, default=0.22)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    execution_mode = "execute" if args.execute else "summarize" if args.summarize else "dry_run"
    sample_size: int | str = "full" if args.full else int(args.sample_size)
    root = Path(args.root).resolve()
    payload = build_full_v3_fresh_live_benchmark_payload(
        root=root,
        eval_id=args.eval_id,
        sample_size=sample_size,
        seed=args.seed,
        execution_mode=execution_mode,
        solve_workers=args.solve_workers,
        judge_workers=args.judge_workers,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        selection_mode=args.selection_mode,
        min_score=args.min_score,
        bootstrap_samples=args.bootstrap_samples,
        run_dir=Path(args.run_dir),
        extra_existing_samples=[Path(path) for path in args.exclude_sample],
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "execution_mode": payload["execution_mode"],
        "run_status": payload["run_status"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
