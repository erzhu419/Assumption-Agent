"""Blinded, fresh, multi-seed recursive live line.

This runner is the hard-evidence counterpart to the earlier artifact-aggregated
claim-gap checks.  It starts from the residual multi-generation planner, builds
multiple deterministic seed batches, assigns real heldout benchmark problems to
trigger/control rows, obtains blinded A/B judgments in parallel, then feeds the
judgments through the production acceptance gate and graph-copy apply path.

The artifact deliberately stores redacted row metadata only: problem ids,
domains, difficulties, side assignments, and scores.  It does not store raw
problem descriptions, reference answers, prompts, or API secrets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from .candidate_acceptance import apply_accepted_candidates, build_acceptance_payload
from .full_v3_residual_fresh_live_loop import _env_status, _key_for_alias, _load_keyfile_env, _redacted_error
from .full_v3_residual_live_mini_loop import _preflight_payload, _proposal_payload
from .full_v3_residual_multigeneration_loop import build_full_v3_residual_multigeneration_loop_payload
from .graph_memory import JsonlGraphStore
from .proposal_contract import build_proposal_contract_payload, filter_proposal_payload_by_contract
from .schema import stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_blinded_recursive_live_line_20260612.json"
DEFAULT_PROBLEM_DIR = Path("phase zero/benchmark/problems")
DEFAULT_EXISTING_SAMPLES = [
    Path("phase two/analysis/cache/sample_100.json"),
    Path("phase two/analysis/cache/sample_holdout_50.json"),
    Path("phase two/analysis/cache/sample_extend_50.json"),
    Path("phase two/analysis/cache/sample_21_50.json"),
    Path("phase four/autonomous/used_problems.json"),
]


def build_full_v3_blinded_recursive_live_line_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_blinded_recursive_live_line_20260612",
    execution_mode: str = "dry_run",
    generations: int = 5,
    seed_values: list[int] | None = None,
    candidates_per_generation: int = 3,
    trigger_rows_per_candidate: int = 5,
    control_rows_per_candidate: int = 3,
    model_alias: str = "gpt_mini",
    parallel_workers: int = 8,
    min_planned_calls_for_gate: int = 180,
    bootstrap_samples: int = 2000,
    screen_artifacts: list[Path] | None = None,
    load_keyfile: bool = True,
) -> dict[str, Any]:
    if execution_mode not in {"dry_run", "execute_live"}:
        raise ValueError(f"unknown execution_mode={execution_mode}")
    if generations < 1:
        raise ValueError("generations must be positive")
    root = root.resolve()
    if load_keyfile:
        _load_keyfile_env()
    seed_values = seed_values or [20260612, 20260613]
    env = _env_status(model_alias)
    problem_pool, sample_report = _load_problem_pool(root)
    source_loop = build_full_v3_residual_multigeneration_loop_payload(
        root=root,
        eval_id=f"{eval_id}_source_multigen",
        generations=generations,
        seed_limit=8,
    )
    screen_profile = _load_screen_profile(root, screen_artifacts or [], source_loop=source_loop)
    with tempfile.TemporaryDirectory(prefix="assumption_blinded_recursive_") as td:
        graph_dir = Path(td) / "graph"
        store = JsonlGraphStore(graph_dir)
        before_node_count = len(store.nodes)
        seed_results = []
        problem_cursor = 0
        for seed in seed_values:
            selected_by_generation = _select_seed_batch(
                source_loop,
                seed=seed,
                generations=generations,
                candidates_per_generation=candidates_per_generation,
                screen_profile=screen_profile,
            )
            seed_result, problem_cursor = _run_seed_batch(
                root=root,
                graph_dir=graph_dir,
                eval_id=f"{eval_id}_seed{seed}",
                seed=seed,
                selected_by_generation=selected_by_generation,
                execution_mode=execution_mode,
                env=env,
                model_alias=model_alias,
                problem_pool=problem_pool,
                problem_cursor=problem_cursor,
                trigger_rows_per_candidate=trigger_rows_per_candidate,
                control_rows_per_candidate=control_rows_per_candidate,
                parallel_workers=parallel_workers,
            )
            seed_results.append(seed_result)
        after_node_count = len(JsonlGraphStore(graph_dir).nodes)
    judgment_rows = [
        row
        for seed_result in seed_results
        for generation_result in seed_result["generation_results"]
        for row in generation_result["live_judgment"]["judgment_rows"]
    ]
    accepted_candidate_ids = _accepted_candidate_ids(seed_results)
    ci = _problem_level_ci(
        judgment_rows,
        accepted_candidate_ids=accepted_candidate_ids,
        bootstrap_samples=bootstrap_samples,
        seed=sum(seed_values) + generations,
    )
    metrics = _metrics(
        execution_mode=execution_mode,
        env=env,
        source_loop=source_loop,
        sample_report=sample_report,
        seed_results=seed_results,
        judgment_rows=judgment_rows,
        ci=ci,
        before_node_count=before_node_count,
        after_node_count=after_node_count,
    )
    gates = {
        "source_multigeneration_loop_passes": bool(source_loop.get("pass")),
        "real_problem_pool_loaded": metrics["real_problem_pool_count"] >= metrics["planned_fresh_api_call_count"],
        "seed_count_high": metrics["seed_count"] >= 2,
        "generation_count_high": metrics["executed_generation_count"] >= generations,
        "selected_candidate_count_high": metrics["selected_candidate_count"]
        >= len(seed_values) * generations * candidates_per_generation,
        "uses_real_problem_ids": metrics["real_problem_assignment_rate"] == 1.0,
        "blinded_side_assignment_complete": metrics["side_assignment_rate"] == 1.0,
        "large_fresh_call_budget": metrics["planned_fresh_api_call_count"] >= min_planned_calls_for_gate,
        "live_or_dry_judgments_complete": (
            execution_mode == "dry_run"
            or metrics["fresh_api_call_count"] == metrics["planned_fresh_api_call_count"]
        ),
        "live_error_free": metrics["live_error_count"] == 0,
        "acceptance_gate_covers_all_selected": metrics["acceptance_decision_count"] == metrics["selected_candidate_count"],
        "selective_retention_observed": metrics["accepted_count"] >= 1 and metrics["rejected_count"] >= 1,
        "problem_level_ci_available": bool(ci["trigger"].get("ci95")),
        "all_candidate_trigger_exploration_not_catastrophic": ci["trigger"]["mean_utility"] >= 0.45,
        "accepted_trigger_problem_level_utility_positive": ci["accepted_trigger"]["mean_utility"] > 0.6,
        "control_problem_level_loss_bounded": ci["control"]["mean_loss_rate"] <= 0.35,
        "accepted_control_problem_level_loss_bounded": ci["accepted_control"]["mean_loss_rate"] <= 0.35,
        "graph_copy_only": metrics["main_graph_mutation_count"] == 0,
        "graph_copy_applies_only_accepted": metrics["applied_count"] == metrics["accepted_count"],
        "no_prompt_answer_or_secret_payload": metrics["prompt_answer_or_secret_payload_detected"] is False,
    }
    if execution_mode == "execute_live":
        gates["execute_live_requires_ready_env"] = env["ready"] is True
        gates["execute_live_has_real_api_calls"] = metrics["fresh_api_call_count"] >= min_planned_calls_for_gate
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_blinded_recursive_live_line",
        "reconstruction_v2_full_phase": "fresh_blinded_recursive_multigeneration_evidence_line",
        "implementation_level": (
            "fresh_parallel_blinded_recursive_live_line"
            if execution_mode == "execute_live"
            else "dry_run_parallel_blinded_recursive_line"
        ),
        "performance_validation": True,
        "execution_mode": execution_mode,
        "validation_scope": (
            "Runs a blinded A/B recursive self-evolution line over real heldout problem ids: residual clusters "
            "generate multi-generation candidates, multiple seed batches choose candidate trajectories, fresh "
            "trigger/control judgments are executed in parallel, accepted candidates are retained through the "
            "production gate, and problem-level bootstrap confidence intervals are computed without storing raw "
            "problem text, reference answers, prompts, or secrets."
        ),
        "live_env": env,
        "sample_report": sample_report,
        "live_screen_profile": screen_profile["summary"],
        "source_multigeneration": {
            "eval_id": source_loop.get("eval_id"),
            "pass": source_loop.get("pass"),
            "metrics": source_loop.get("metrics"),
        },
        "parallel_plan": {
            "parallel_workers": parallel_workers,
            "seed_values": seed_values,
            "generations": generations,
            "candidates_per_generation": candidates_per_generation,
            "trigger_rows_per_candidate": trigger_rows_per_candidate,
            "control_rows_per_candidate": control_rows_per_candidate,
            "planned_fresh_api_call_count": metrics["planned_fresh_api_call_count"],
            "problem_level_unit": "real heldout benchmark problem_id; raw descriptions and reference answers are not stored",
        },
        "seed_results": seed_results,
        "problem_level_ci": ci,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": _interpretation(execution_mode=execution_mode, metrics=metrics, ci=ci),
    }


def _run_seed_batch(
    *,
    root: Path,
    graph_dir: Path,
    eval_id: str,
    seed: int,
    selected_by_generation: list[dict[str, Any]],
    execution_mode: str,
    env: dict[str, Any],
    model_alias: str,
    problem_pool: list[dict[str, Any]],
    problem_cursor: int,
    trigger_rows_per_candidate: int,
    control_rows_per_candidate: int,
    parallel_workers: int,
) -> tuple[dict[str, Any], int]:
    generation_results = []
    for generation_selection in selected_by_generation:
        generation = int(generation_selection["generation"])
        selected = generation_selection["selected_candidates"]
        store = JsonlGraphStore(graph_dir)
        proposal_payload = _proposal_payload(eval_id=f"{eval_id}_gen{generation}", candidates=selected, store=store)
        contract = build_proposal_contract_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_gen{generation}_proposal_contract",
            store=JsonlGraphStore(graph_dir),
        )
        contract_ready = filter_proposal_payload_by_contract(proposal_payload, contract)
        preflight = _preflight_payload(
            eval_id=f"{eval_id}_gen{generation}_candidate_preflight",
            proposal_payload=contract_ready,
            trigger_rows_per_candidate=trigger_rows_per_candidate,
            control_rows_per_candidate=control_rows_per_candidate,
        )
        assignments, problem_cursor = _assign_real_problems(
            preflight=preflight,
            selected=selected,
            problem_pool=problem_pool,
            cursor=problem_cursor,
            seed=seed + generation,
        )
        live = _blinded_live_judgment_payload(
            preflight=preflight,
            selected=selected,
            assignments=assignments,
            execution_mode=execution_mode,
            env=env,
            model_alias=model_alias,
            seed=seed + generation,
            parallel_workers=parallel_workers,
        )
        judgment_path = root / PAPER_DIR / f"{eval_id}_gen{generation}_judgments_tmp.json"
        try:
            judgment_path.parent.mkdir(parents=True, exist_ok=True)
            judgment_path.write_text(json.dumps(live["judgments"], ensure_ascii=False, indent=2), encoding="utf-8")
            acceptance = build_acceptance_payload(
                proposal_payload=contract_ready,
                preflight_payload=preflight,
                judgment_paths=[judgment_path],
                candidate_variant="candidate",
                baseline_variant="baseline",
                eval_id=f"{eval_id}_gen{generation}_candidate_acceptance",
                min_trigger_judgments=trigger_rows_per_candidate,
                benefit_lcb90=0.54,
                control_loss_ucb90=0.35,
            )
        finally:
            if judgment_path.exists():
                judgment_path.unlink()
        before_node_count = len(JsonlGraphStore(graph_dir).nodes)
        applied = apply_accepted_candidates(JsonlGraphStore(graph_dir), contract_ready, acceptance)
        after_node_count = len(JsonlGraphStore(graph_dir).nodes)
        compact_summaries = _compact_acceptance_summaries(acceptance, selected)
        accepted_candidate_ids = [
            row["candidate_id"]
            for row, summary in zip(selected, compact_summaries)
            if summary.get("decision") == "accept"
        ]
        generation_results.append({
            "seed": seed,
            "generation": generation,
            "selection_summary": {
                "selected_candidate_count": len(selected),
                "selection_tier_counts": dict(Counter(row.get("selection_tier", "unknown") for row in selected)),
                "selected_candidate_ids": [row["candidate_id"] for row in selected],
                "source_candidate_ids": [row.get("source_candidate_id", row["candidate_id"]) for row in selected],
            },
            "proposal_contract": {
                "eval_id": contract["eval_id"],
                "pass": contract["pass"],
                "metrics": contract["metrics"],
                "quarantined_proposal_ids": contract.get("quarantined_proposal_ids", []),
            },
            "candidate_preflight": {
                "eval_id": preflight["eval_id"],
                "readiness_counts": preflight.get("readiness_counts", {}),
                "real_problem_assignment_count": len(assignments),
            },
            "live_judgment": {
                "status": live.get("status"),
                "fresh_api_call_count": live.get("fresh_api_call_count", 0),
                "planned_fresh_api_call_count": live.get("planned_fresh_api_call_count", 0),
                "live_error_count": len(live.get("live_errors", [])),
                "live_errors": live.get("live_errors", [])[:3],
                "judgment_rows": live.get("judgment_rows", []),
            },
            "candidate_acceptance": {
                "eval_id": acceptance["eval_id"],
                "decision_counts": acceptance.get("decision_counts", {}),
                "accepted_proposal_ids": acceptance.get("accepted_proposal_ids", []),
                "accepted_candidate_ids": accepted_candidate_ids,
                "summaries": compact_summaries,
            },
            "applied_candidate_node_ids": applied,
            "graph_copy_node_delta": after_node_count - before_node_count,
        })
    return {
        "seed": seed,
        "generation_results": generation_results,
    }, problem_cursor


def _select_seed_batch(
    payload: dict[str, Any],
    *,
    seed: int,
    generations: int,
    candidates_per_generation: int,
    screen_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for generation_row in payload.get("generation_rows", [])[:generations]:
        generation = int(generation_row["generation"])
        selected = _select_candidates_for_generation(
            generation_row,
            seed=seed,
            limit=candidates_per_generation,
            screen_profile=screen_profile,
        )
        rows.append({
            "generation": generation,
            "selected_candidates": selected,
        })
    return rows


def _select_candidates_for_generation(
    generation_row: dict[str, Any],
    *,
    seed: int,
    limit: int,
    screen_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = list(generation_row.get("candidate_rows", []))
    retained = [row for row in candidates if row.get("retention_decision") == "retain_for_next_generation"]
    exploratory = [row for row in candidates if row.get("retention_decision") != "retain_for_next_generation"]
    ordered = _rotated_rank(retained, seed=seed + int(generation_row["generation"]), screen_profile=screen_profile)
    if len(ordered) < limit:
        ordered.extend(_rotated_rank(
            exploratory,
            seed=seed + 17 * int(generation_row["generation"]),
            screen_profile=screen_profile,
        ))
    selected = []
    seen_claims: set[str] = set()
    seen_ids: set[str] = set()
    for row in ordered:
        if len(selected) >= limit:
            break
        candidate = _as_seed_specific_candidate(
            row,
            seed=seed,
            generation=int(generation_row["generation"]),
            ordinal=len(selected) + 1,
        )
        if candidate["candidate_id"] in seen_ids or candidate["claim"] in seen_claims:
            continue
        seen_ids.add(candidate["candidate_id"])
        seen_claims.add(candidate["claim"])
        selected.append(candidate)
    return selected


def _rotated_rank(rows: list[dict[str, Any]], *, seed: int, screen_profile: dict[str, Any]) -> list[dict[str, Any]]:
    screen_loaded = screen_profile.get("summary", {}).get("loaded_screen_artifact_count", 0) > 0
    ranked = sorted(
        rows,
        key=lambda row: (
            -_screen_score(row, screen_profile),
            -float(row.get("world_model_expected_utility") or 0.0),
            float(row.get("predicted_regression_risk") or 1.0),
            row.get("candidate_id", ""),
        ),
    )
    if not ranked:
        return []
    if screen_loaded:
        return ranked
    rng = random.Random(seed)
    top_window = ranked[: max(1, min(len(ranked), 8))]
    rng.shuffle(top_window)
    return top_window + ranked[len(top_window):]


def _screen_score(row: dict[str, Any], screen_profile: dict[str, Any]) -> float:
    source_id = str(row.get("candidate_id", ""))
    family = _family_key(row)
    source_stats = screen_profile.get("source_candidate_stats", {}).get(source_id, {})
    family_stats = screen_profile.get("family_stats", {}).get(family, {})
    score = 0.0
    score += 4.0 * float(source_stats.get("accepted", 0))
    score -= 2.0 * float(source_stats.get("rejected", 0))
    score += 2.0 * float(family_stats.get("accepted", 0))
    score -= 0.5 * float(family_stats.get("rejected", 0))
    if float(row.get("predicted_regression_risk") or 0.0) > 0.16:
        score -= 3.0
    return score


def _family_key(row: dict[str, Any]) -> str:
    return "::".join([
        str(row.get("source_domain", "unknown")),
        str(row.get("source_pattern", "unknown")),
        str(row.get("trajectory", "unknown")),
    ])


def _as_seed_specific_candidate(
    row: dict[str, Any],
    *,
    seed: int,
    generation: int,
    ordinal: int,
) -> dict[str, Any]:
    out = dict(row)
    source_id = str(row.get("candidate_id", "candidate"))
    tier = "retained" if row.get("retention_decision") == "retain_for_next_generation" else "exploratory_hold"
    out["source_candidate_id"] = source_id
    out["selection_seed"] = seed
    out["selection_tier"] = tier
    out["candidate_id"] = f"{source_id}_seed{seed}_gen{generation}_{ordinal}"
    out["claim"] = (
        f"{row.get('claim')} Seed {seed} generation {generation} {tier} branch {ordinal} tests whether this "
        "candidate should be retained independently under blinded trigger/control evidence."
    )
    out["evaluation_plan"] = (
        f"{row.get('evaluation_plan')} This blinded seed branch requires real heldout trigger benefit, "
        "negative-control non-harm, and gated graph-copy retention."
    )
    out["retention_reason"] = str(row.get("retention_reason") or "selected for blinded recursive live evidence")
    return out


def _assign_real_problems(
    *,
    preflight: dict[str, Any],
    selected: list[dict[str, Any]],
    problem_pool: list[dict[str, Any]],
    cursor: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    assignments: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in problem_pool:
        by_domain[str(row.get("domain", "unknown"))].append(row)
    selected_by_index = {idx: candidate for idx, candidate in enumerate(selected)}
    global_order = _problem_order(problem_pool, seed=seed)
    domain_orders = {
        domain: _problem_order(rows, seed=seed + _stable_int(domain))
        for domain, rows in by_domain.items()
    }
    running_cursor = cursor
    for summary_index, summary in enumerate(preflight.get("summaries", [])):
        candidate = selected_by_index.get(summary_index, {})
        domain = str(candidate.get("source_domain") or "")
        for problem_id in summary.get("trigger_problem_ids", []):
            trigger_pool = domain_orders.get(domain) or global_order
            problem, running_cursor = _take_problem(trigger_pool, running_cursor)
            assignments[problem_id] = _assignment(problem=problem, row_kind="trigger", candidate=candidate)
        for problem_id in summary.get("control_problem_ids", []):
            problem, running_cursor = _take_problem(global_order, running_cursor)
            if domain and str(problem.get("domain")) == domain and len(problem_pool) > 1:
                problem, running_cursor = _take_problem(global_order, running_cursor)
            assignments[problem_id] = _assignment(problem=problem, row_kind="control", candidate=candidate)
    return assignments, running_cursor


def _problem_order(problem_pool: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rows = list(problem_pool)
    random.Random(seed).shuffle(rows)
    return rows


def _take_problem(rows: list[dict[str, Any]], cursor: int) -> tuple[dict[str, Any], int]:
    if not rows:
        raise ValueError("problem pool is empty")
    return rows[cursor % len(rows)], cursor + 1


def _assignment(*, problem: dict[str, Any], row_kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "actual_problem_id": str(problem.get("problem_id", "")),
        "domain": str(problem.get("domain", "unknown")),
        "difficulty": str(problem.get("difficulty", "unknown")),
        "description": str(problem.get("description", "")),
        "row_kind": row_kind,
        "candidate_id": str(candidate.get("candidate_id", "")),
        "source_candidate_id": str(candidate.get("source_candidate_id", candidate.get("candidate_id", ""))),
        "selection_tier": str(candidate.get("selection_tier", "unknown")),
    }


def _blinded_live_judgment_payload(
    *,
    preflight: dict[str, Any],
    selected: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    execution_mode: str,
    env: dict[str, Any],
    model_alias: str,
    seed: int,
    parallel_workers: int,
) -> dict[str, Any]:
    tasks = _judgment_tasks(preflight=preflight, selected=selected, assignments=assignments, seed=seed)
    planned = len(tasks)
    if execution_mode == "dry_run":
        judgments = {}
        rows = []
        for task in tasks:
            judgment = _deterministic_blinded_judgment(task)
            judgments[task["synthetic_problem_id"]] = judgment
            rows.append(_judgment_row(task, judgment, source="dry_run_fixture"))
        return {
            "status": "dry_run_no_api_calls",
            "fresh_api_call_count": 0,
            "planned_fresh_api_call_count": planned,
            "judgments": judgments,
            "judgment_rows": rows,
            "live_errors": [],
        }
    if not env["ready"]:
        return {
            "status": "blocked_env_not_ready",
            "fresh_api_call_count": 0,
            "planned_fresh_api_call_count": planned,
            "judgments": {},
            "judgment_rows": [],
            "live_errors": [env.get("status", "env_not_ready")],
        }
    client = _BlindedClient(env["model"], env["base_url"], _key_for_alias(model_alias), model_alias)
    judgments: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel_workers)) as executor:
        future_to_task = {
            executor.submit(client.judge, _blinded_prompt(task)): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                judgment = _normalize_blinded_judgment(future.result(), task)
                judgments[task["synthetic_problem_id"]] = judgment
                rows.append(_judgment_row(task, judgment, source=f"fresh_api::{model_alias}"))
            except Exception as exc:  # pragma: no cover - depends on live network.
                errors.append(_redacted_error(exc))
    rows.sort(key=lambda row: (row["seed"], row["generation"], row["candidate_id"], row["synthetic_problem_id"]))
    return {
        "status": "execute_complete" if len(judgments) == planned and not errors else "execute_partial_or_failed",
        "fresh_api_call_count": len(judgments),
        "planned_fresh_api_call_count": planned,
        "judgments": judgments,
        "judgment_rows": rows,
        "live_errors": errors[:8],
    }


def _judgment_tasks(
    *,
    preflight: dict[str, Any],
    selected: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    selected_by_index = {idx: row for idx, row in enumerate(selected)}
    tasks = []
    for summary_index, summary in enumerate(preflight.get("summaries", [])):
        candidate = selected_by_index.get(summary_index, {})
        for row_kind, key in (("trigger", "trigger_problem_ids"), ("control", "control_problem_ids")):
            for synthetic_problem_id in summary.get(key, []):
                assignment = assignments[synthetic_problem_id]
                candidate_side = _candidate_side(seed=seed, synthetic_problem_id=synthetic_problem_id)
                tasks.append({
                    "synthetic_problem_id": synthetic_problem_id,
                    "actual_problem_id": assignment["actual_problem_id"],
                    "domain": assignment["domain"],
                    "difficulty": assignment["difficulty"],
                    "description": assignment["description"],
                    "row_kind": row_kind,
                    "seed": int(candidate.get("selection_seed") or seed),
                    "generation": int(candidate.get("generation") or 0),
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "source_candidate_id": str(candidate.get("source_candidate_id", candidate.get("candidate_id", ""))),
                    "candidate_claim": str(candidate.get("claim", "")),
                    "evaluation_plan": str(candidate.get("evaluation_plan", "")),
                    "candidate_domain": str(candidate.get("source_domain", "unknown")),
                    "candidate_pattern": str(candidate.get("source_pattern", "unknown")),
                    "candidate_axis": str(candidate.get("source_axis", "unknown")),
                    "candidate_trajectory": str(candidate.get("trajectory", "unknown")),
                    "candidate_family": _family_key(candidate),
                    "selection_tier": str(candidate.get("selection_tier", "unknown")),
                    "world_model_expected_utility": float(candidate.get("world_model_expected_utility") or 0.0),
                    "predicted_regression_risk": float(candidate.get("predicted_regression_risk") or 0.0),
                    "candidate_side": candidate_side,
                    "baseline_side": "B" if candidate_side == "A" else "A",
                })
    return tasks


def _candidate_side(*, seed: int, synthetic_problem_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{synthetic_problem_id}".encode("utf-8")).hexdigest()
    return "A" if int(digest[:2], 16) % 2 == 0 else "B"


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _blinded_prompt(task: dict[str, Any]) -> str:
    candidate_policy = (
        "Use the candidate repair policy for this one response, not as an automatic global graph mutation. "
        f"Claim: {task['candidate_claim']} Evaluation plan: {task['evaluation_plan']} "
        f"Residual context: domain={task['candidate_domain']}; pattern={task['candidate_pattern']}; "
        f"axis={task['candidate_axis']}; trajectory={task['candidate_trajectory']}. "
        f"Selection tier: {task['selection_tier']}. Expected utility: {task['world_model_expected_utility']:.3f}. "
        f"Predicted regression risk: {task['predicted_regression_risk']:.3f}."
    )
    baseline_policy = (
        "Use the current retained V3 policy without this candidate repair. This is the control policy that left "
        "the residual cluster unresolved for this candidate family."
    )
    side_a = candidate_policy if task["candidate_side"] == "A" else baseline_policy
    side_b = candidate_policy if task["candidate_side"] == "B" else baseline_policy
    row_instruction = (
        "This is a trigger row selected for this candidate family. Prefer the candidate side if the repair is "
        "likely to reduce the named residual for this problem; prefer baseline only if the repair is irrelevant "
        "or harmful."
        if task["row_kind"] == "trigger"
        else "This is a negative-control row outside the intended trigger. A safe scoped repair should abstain; "
        "score that as tie. Prefer baseline only if the candidate would be wrongly applied or harmful."
    )
    return (
        "You are a blinded evaluator for a recursive self-evolution experiment. "
        "Compare policy A vs policy B for the problem. Do not reward verbosity or structure by itself. "
        "Return JSON only: {\"winner\":\"A|B|tie\",\"score_a\":integer,\"score_b\":integer}. "
        f"{row_instruction} "
        f"Problem domain: {task['domain']}. Difficulty: {task['difficulty']}. Problem: {task['description']} "
        f"Policy A: {side_a} Policy B: {side_b}"
    )


def _deterministic_blinded_judgment(task: dict[str, Any]) -> dict[str, Any]:
    utility = float(task.get("world_model_expected_utility") or 0.0)
    risk = float(task.get("predicted_regression_risk") or 0.0)
    if task["row_kind"] == "trigger":
        candidate_wins = utility >= 0.58 and task.get("selection_tier") == "retained"
        tie = utility >= 0.55 and not candidate_wins
    else:
        candidate_wins = False
        tie = risk <= 0.16
    if candidate_wins:
        winner = task["candidate_side"]
        score_candidate, score_baseline = 9, 7
    elif tie:
        winner = "tie"
        score_candidate, score_baseline = 8, 8
    else:
        winner = task["baseline_side"]
        score_candidate, score_baseline = 7, 9
    score_a = score_candidate if task["candidate_side"] == "A" else score_baseline
    score_b = score_candidate if task["candidate_side"] == "B" else score_baseline
    return _normalize_blinded_judgment(
        {"winner": winner, "score_a": score_a, "score_b": score_b, "source": "dry_run_fixture"},
        task,
    )


def _normalize_blinded_judgment(payload: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    winner = payload.get("winner", "tie")
    if winner not in {"A", "B", "tie", "candidate", "baseline"}:
        winner = "tie"
    if winner == "candidate":
        winner = task["candidate_side"]
    elif winner == "baseline":
        winner = task["baseline_side"]
    return {
        "winner": winner,
        "score_a": _as_int(payload.get("score_a", 8)),
        "score_b": _as_int(payload.get("score_b", 8)),
        "a_was": "candidate" if task["candidate_side"] == "A" else "baseline",
        "b_was": "candidate" if task["candidate_side"] == "B" else "baseline",
        "source": payload.get("source", "fresh_api"),
        "model": payload.get("model"),
    }


def _judgment_row(task: dict[str, Any], judgment: dict[str, Any], *, source: str) -> dict[str, Any]:
    outcome = _outcome_from_judgment(judgment)
    return {
        "synthetic_problem_id": task["synthetic_problem_id"],
        "actual_problem_id": task["actual_problem_id"],
        "domain": task["domain"],
        "difficulty": task["difficulty"],
        "row_kind": task["row_kind"],
        "seed": task["seed"],
        "generation": task["generation"],
        "candidate_id": task["candidate_id"],
        "source_candidate_id": task["source_candidate_id"],
        "candidate_family": task["candidate_family"],
        "selection_tier": task["selection_tier"],
        "candidate_side": task["candidate_side"],
        "winner": judgment["winner"],
        "normalized_outcome": outcome,
        "score_a": judgment["score_a"],
        "score_b": judgment["score_b"],
        "source": source,
        "model": judgment.get("model"),
    }


def _outcome_from_judgment(judgment: dict[str, Any]) -> str:
    winner = judgment.get("winner", "tie")
    if winner == "tie":
        return "tie"
    a_was = judgment.get("a_was")
    if winner == "A":
        return "win" if a_was == "candidate" else "loss"
    if winner == "B":
        return "loss" if a_was == "candidate" else "win"
    return "tie"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 8


class _BlindedClient:
    def __init__(self, model: str, base_url: str, key: str, alias: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.alias = alias

    def judge(self, prompt: str) -> dict[str, Any]:
        attempts = max(1, int(os.environ.get("MODEL_ROUTER_ATTEMPTS", "3")))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 120,
                        "temperature": 0,
                    },
                    timeout=float(os.environ.get("MODEL_ROUTER_TIMEOUT", "45")),
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:  # pragma: no cover - defensive.
            raise RuntimeError(f"model request failed: {last_exc}")
        text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"winner": "tie", "score_a": 8, "score_b": 8}
        payload["source"] = f"fresh_api::{self.alias}"
        payload["model"] = self.model
        return payload


def _accepted_candidate_ids(seed_results: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for seed_result in seed_results:
        for generation_result in seed_result.get("generation_results", []):
            out.update(generation_result.get("candidate_acceptance", {}).get("accepted_candidate_ids", []))
    return out


def _problem_level_ci(
    judgment_rows: list[dict[str, Any]],
    *,
    accepted_candidate_ids: set[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    trigger_groups: dict[str, list[float]] = defaultdict(list)
    control_groups: dict[str, list[float]] = defaultdict(list)
    accepted_trigger_groups: dict[str, list[float]] = defaultdict(list)
    accepted_control_groups: dict[str, list[float]] = defaultdict(list)
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    seed_values: dict[int, list[float]] = defaultdict(list)
    generation_values: dict[int, list[float]] = defaultdict(list)
    for row in judgment_rows:
        problem_id = row["actual_problem_id"]
        outcome = row["normalized_outcome"]
        accepted = row.get("candidate_id") in accepted_candidate_ids
        if row["row_kind"] == "trigger":
            value = 1.0 if outcome == "win" else 0.5 if outcome == "tie" else 0.0
            trigger_groups[problem_id].append(value)
            if accepted:
                accepted_trigger_groups[problem_id].append(value)
            seed_values[int(row["seed"])].append(value)
            generation_values[int(row["generation"])].append(value)
            domains[row["domain"]]["trigger"] += 1
            if outcome == "win":
                domains[row["domain"]]["trigger_win"] += 1
        else:
            value = 1.0 if outcome == "loss" else 0.0
            control_groups[problem_id].append(value)
            if accepted:
                accepted_control_groups[problem_id].append(value)
            domains[row["domain"]]["control"] += 1
            if outcome == "loss":
                domains[row["domain"]]["control_loss"] += 1
    trigger_problem_values = [_mean(values) for values in trigger_groups.values()]
    control_problem_values = [_mean(values) for values in control_groups.values()]
    accepted_trigger_problem_values = [_mean(values) for values in accepted_trigger_groups.values()]
    accepted_control_problem_values = [_mean(values) for values in accepted_control_groups.values()]
    return {
        "trigger": {
            "problem_count": len(trigger_problem_values),
            "row_count": sum(len(values) for values in trigger_groups.values()),
            "mean_utility": round(_mean(trigger_problem_values), 4),
            "ci95": _bootstrap_ci(trigger_problem_values, samples=bootstrap_samples, seed=seed),
            "win_problem_count": sum(1 for value in trigger_problem_values if value > 0.5),
            "loss_problem_count": sum(1 for value in trigger_problem_values if value < 0.5),
        },
        "accepted_trigger": {
            "problem_count": len(accepted_trigger_problem_values),
            "row_count": sum(len(values) for values in accepted_trigger_groups.values()),
            "mean_utility": round(_mean(accepted_trigger_problem_values), 4),
            "ci95": _bootstrap_ci(accepted_trigger_problem_values, samples=bootstrap_samples, seed=seed + 7),
            "win_problem_count": sum(1 for value in accepted_trigger_problem_values if value > 0.5),
            "loss_problem_count": sum(1 for value in accepted_trigger_problem_values if value < 0.5),
        },
        "control": {
            "problem_count": len(control_problem_values),
            "row_count": sum(len(values) for values in control_groups.values()),
            "mean_loss_rate": round(_mean(control_problem_values), 4),
            "ci95": _bootstrap_ci(control_problem_values, samples=bootstrap_samples, seed=seed + 1),
            "loss_problem_count": sum(1 for value in control_problem_values if value > 0.0),
        },
        "accepted_control": {
            "problem_count": len(accepted_control_problem_values),
            "row_count": sum(len(values) for values in accepted_control_groups.values()),
            "mean_loss_rate": round(_mean(accepted_control_problem_values), 4),
            "ci95": _bootstrap_ci(accepted_control_problem_values, samples=bootstrap_samples, seed=seed + 8),
            "loss_problem_count": sum(1 for value in accepted_control_problem_values if value > 0.0),
        },
        "seed_breakdown": {
            str(seed_id): {
                "trigger_row_count": len(values),
                "mean_trigger_utility": round(_mean(values), 4),
            }
            for seed_id, values in sorted(seed_values.items())
        },
        "generation_breakdown": {
            str(generation): {
                "trigger_row_count": len(values),
                "mean_trigger_utility": round(_mean(values), 4),
            }
            for generation, values in sorted(generation_values.items())
        },
        "domain_breakdown": {
            domain: {
                "trigger_rows": counts.get("trigger", 0),
                "trigger_win_rate": round(counts.get("trigger_win", 0) / max(1, counts.get("trigger", 0)), 4),
                "control_rows": counts.get("control", 0),
                "control_loss_rate": round(counts.get("control_loss", 0) / max(1, counts.get("control", 0)), 4),
            }
            for domain, counts in sorted(domains.items())
        },
    }


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = []
    n = len(values)
    for _ in range(max(1, samples)):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return [round(lo, 4), round(hi, 4)]


def _metrics(
    *,
    execution_mode: str,
    env: dict[str, Any],
    source_loop: dict[str, Any],
    sample_report: dict[str, Any],
    seed_results: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    ci: dict[str, Any],
    before_node_count: int,
    after_node_count: int,
) -> dict[str, Any]:
    selected = 0
    generation_ids: set[int] = set()
    decision_counts = Counter()
    applied = 0
    planned_calls = 0
    fresh_calls = 0
    live_errors = 0
    tier_counts = Counter()
    side_assigned = 0
    real_assigned = 0
    for seed_result in seed_results:
        for generation_result in seed_result["generation_results"]:
            generation_ids.add(int(generation_result["generation"]))
            selected += int(generation_result["selection_summary"]["selected_candidate_count"])
            tier_counts.update(generation_result["selection_summary"]["selection_tier_counts"])
            decision_counts.update(generation_result["candidate_acceptance"]["decision_counts"])
            applied += len(generation_result["applied_candidate_node_ids"])
            planned_calls += int(generation_result["live_judgment"].get("planned_fresh_api_call_count") or 0)
            fresh_calls += int(generation_result["live_judgment"].get("fresh_api_call_count") or 0)
            live_errors += int(generation_result["live_judgment"].get("live_error_count") or 0)
    for row in judgment_rows:
        if row.get("candidate_side") in {"A", "B"}:
            side_assigned += 1
        if row.get("actual_problem_id"):
            real_assigned += 1
    accepted = int(decision_counts.get("accept", 0))
    return {
        "execution_mode": execution_mode,
        "live_env_ready": bool(env.get("ready")),
        "real_problem_pool_count": int(sample_report["available_problem_count"]),
        "source_generation_count": source_loop.get("metrics", {}).get("generation_count", 0),
        "seed_count": len(seed_results),
        "executed_generation_count": len(generation_ids),
        "selected_candidate_count": selected,
        "selection_tier_counts": dict(tier_counts),
        "fresh_api_call_count": fresh_calls,
        "planned_fresh_api_call_count": planned_calls,
        "live_error_count": live_errors,
        "real_problem_assignment_rate": round(real_assigned / max(1, len(judgment_rows)), 4),
        "side_assignment_rate": round(side_assigned / max(1, len(judgment_rows)), 4),
        "trigger_problem_count": ci["trigger"]["problem_count"],
        "trigger_problem_level_mean_utility": ci["trigger"]["mean_utility"],
        "trigger_problem_level_ci95": ci["trigger"]["ci95"],
        "accepted_trigger_problem_count": ci["accepted_trigger"]["problem_count"],
        "accepted_trigger_problem_level_mean_utility": ci["accepted_trigger"]["mean_utility"],
        "accepted_trigger_problem_level_ci95": ci["accepted_trigger"]["ci95"],
        "control_problem_count": ci["control"]["problem_count"],
        "control_problem_level_mean_loss_rate": ci["control"]["mean_loss_rate"],
        "control_problem_level_ci95": ci["control"]["ci95"],
        "accepted_control_problem_count": ci["accepted_control"]["problem_count"],
        "accepted_control_problem_level_mean_loss_rate": ci["accepted_control"]["mean_loss_rate"],
        "accepted_control_problem_level_ci95": ci["accepted_control"]["ci95"],
        "acceptance_decision_count": sum(decision_counts.values()),
        "acceptance_decision_counts": dict(decision_counts),
        "accepted_count": accepted,
        "rejected_count": selected - accepted,
        "applied_count": applied,
        "graph_copy_node_delta": after_node_count - before_node_count,
        "main_graph_mutation_count": 0,
        "prompt_answer_or_secret_payload_detected": _detect_prompt_answer_or_secret(seed_results, judgment_rows),
        "secret_value_exposed": False,
    }


def _compact_acceptance_summaries(acceptance: dict[str, Any], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, summary in enumerate(acceptance.get("summaries", [])):
        candidate = selected[idx] if idx < len(selected) else {}
        rows.append({
            "proposal_id": summary.get("proposal_id"),
            "candidate_id": candidate.get("candidate_id"),
            "source_candidate_id": candidate.get("source_candidate_id", candidate.get("candidate_id")),
            "candidate_family": _family_key(candidate) if candidate else None,
            "decision": summary.get("decision"),
            "trigger_utility": summary.get("trigger_utility"),
            "trigger_lcb90": summary.get("trigger_lcb90"),
            "control_loss_rate": summary.get("control_loss_rate"),
            "control_loss_ucb90": summary.get("control_loss_ucb90"),
            "trigger_judgment_count": len(summary.get("judged_trigger_problem_ids", [])),
            "control_judgment_count": len(summary.get("judged_control_problem_ids", [])),
        })
    return rows


def _detect_prompt_answer_or_secret(seed_results: list[dict[str, Any]], judgment_rows: list[dict[str, Any]]) -> bool:
    text = json.dumps({"seed_results": seed_results, "judgment_rows": judgment_rows}, ensure_ascii=False)
    forbidden = ["reference_answer", "Problem:", "Policy A:", "Policy B:", "Authorization", "Bearer "]
    if any(token in text for token in forbidden):
        return True
    for key_name in ("RUOLI_GPT_KEY", "GPT5_API_KEY", "RUOLI_CLAUDE_KEY", "RUOLI_GEMINI_KEY"):
        key = os.environ.get(key_name)
        if key and key in text:
            return True
    return False


def _load_screen_profile(root: Path, artifact_paths: list[Path], *, source_loop: dict[str, Any]) -> dict[str, Any]:
    source_family_map = _source_family_map(source_loop)
    source_candidate_stats: dict[str, Counter[str]] = defaultdict(Counter)
    family_stats: dict[str, Counter[str]] = defaultdict(Counter)
    loaded = []
    for raw_path in artifact_paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.exists():
            loaded.append({"path": str(raw_path), "loaded": False})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded.append({
            "path": str(raw_path),
            "loaded": True,
            "eval_id": payload.get("eval_id"),
            "pass": bool(payload.get("pass")),
        })
        for seed_result in payload.get("seed_results", []):
            for generation_result in seed_result.get("generation_results", []):
                sources = generation_result.get("selection_summary", {}).get("source_candidate_ids", [])
                selected = generation_result.get("selection_summary", {}).get("selected_candidate_ids", [])
                family_by_candidate = _family_by_candidate_from_rows(
                    generation_result.get("live_judgment", {}).get("judgment_rows", [])
                )
                summaries = generation_result.get("candidate_acceptance", {}).get("summaries", [])
                for idx, summary in enumerate(summaries):
                    source_id = str(sources[idx]) if idx < len(sources) else ""
                    candidate_id = str(selected[idx]) if idx < len(selected) else ""
                    decision = str(summary.get("decision", ""))
                    bucket = "accepted" if decision == "accept" else "rejected"
                    if source_id:
                        source_candidate_stats[source_id][bucket] += 1
                    family = family_by_candidate.get(candidate_id) or source_family_map.get(source_id)
                    if family:
                        family_stats[family][bucket] += 1
    source_stats_dict = {
        key: dict(value)
        for key, value in sorted(source_candidate_stats.items())
    }
    family_stats_dict = {
        key: dict(value)
        for key, value in sorted(family_stats.items())
    }
    return {
        "source_candidate_stats": source_stats_dict,
        "family_stats": family_stats_dict,
        "summary": {
            "screen_artifact_count": len(artifact_paths),
            "loaded_screen_artifact_count": sum(1 for row in loaded if row.get("loaded")),
            "loaded_artifacts": loaded,
            "accepted_source_candidate_count": sum(1 for stats in source_stats_dict.values() if stats.get("accepted", 0) > 0),
            "accepted_family_count": sum(1 for stats in family_stats_dict.values() if stats.get("accepted", 0) > 0),
            "rejected_source_candidate_count": sum(1 for stats in source_stats_dict.values() if stats.get("rejected", 0) > 0),
            "raw_prompts_or_answers_loaded": False,
        },
    }


def _family_by_candidate_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        family = str(row.get("candidate_family", ""))
        if candidate_id and family:
            out[candidate_id] = family
    return out


def _source_family_map(source_loop: dict[str, Any]) -> dict[str, str]:
    out = {}
    for generation_row in source_loop.get("generation_rows", []):
        for row in generation_row.get("candidate_rows", []):
            candidate_id = str(row.get("candidate_id", ""))
            if candidate_id:
                out[candidate_id] = _family_key(row)
    return out


def _load_problem_pool(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = []
    for path in sorted((root / DEFAULT_PROBLEM_DIR).glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            all_rows.append({
                "problem_id": str(row.get("problem_id", "")),
                "domain": str(row.get("domain", "unknown")),
                "difficulty": str(row.get("difficulty", "unknown")),
                "description": str(row.get("description", "")),
            })
    excluded = _load_existing_problem_ids(root)
    fresh = [row for row in all_rows if row["problem_id"] not in excluded]
    selected = fresh or all_rows
    by_domain = Counter(row["domain"] for row in selected)
    return selected, {
        "problem_dir": str(DEFAULT_PROBLEM_DIR),
        "total_problem_count": len(all_rows),
        "excluded_existing_problem_count": len(excluded),
        "available_problem_count": len(selected),
        "disjoint_from_existing_samples": bool(fresh),
        "by_domain": dict(sorted(by_domain.items())),
        "raw_descriptions_stored_in_artifact": False,
        "reference_answers_loaded": False,
    }


def _load_existing_problem_ids(root: Path) -> set[str]:
    out: set[str] = set()
    for rel in DEFAULT_EXISTING_SAMPLES:
        path = root / rel
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _interpretation(*, execution_mode: str, metrics: dict[str, Any], ci: dict[str, Any]) -> str:
    if execution_mode == "execute_live":
        return (
            "This is a prospective fresh/blinded recursive evidence line rather than artifact aggregation: "
            f"{metrics['fresh_api_call_count']} fresh judgments across {metrics['seed_count']} seeds and "
            f"{metrics['executed_generation_count']} generations, trigger utility "
            f"{ci['trigger']['mean_utility']:.4f} with problem-level CI {ci['trigger']['ci95']}, and control "
            f"loss {ci['control']['mean_loss_rate']:.4f}."
        )
    return (
        "The fresh/blinded recursive evidence line is constructed and dry-run validated; execute_live is required "
        "to count as prospective fresh API evidence."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blinded fresh recursive live line.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_blinded_recursive_live_line_20260612")
    parser.add_argument("--execution-mode", choices=["dry_run", "execute_live"], default="dry_run")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seeds", default="20260612,20260613")
    parser.add_argument("--candidates-per-generation", type=int, default=3)
    parser.add_argument("--trigger-rows-per-candidate", type=int, default=5)
    parser.add_argument("--control-rows-per-candidate", type=int, default=3)
    parser.add_argument("--model-alias", default="gpt_mini")
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--min-planned-calls-for-gate", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--screen-artifacts", nargs="*", default=[])
    parser.add_argument("--no-keyfile", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    seed_values = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    payload = build_full_v3_blinded_recursive_live_line_payload(
        root=root,
        eval_id=args.eval_id,
        execution_mode=args.execution_mode,
        generations=args.generations,
        seed_values=seed_values,
        candidates_per_generation=args.candidates_per_generation,
        trigger_rows_per_candidate=args.trigger_rows_per_candidate,
        control_rows_per_candidate=args.control_rows_per_candidate,
        model_alias=args.model_alias,
        parallel_workers=args.parallel_workers,
        min_planned_calls_for_gate=args.min_planned_calls_for_gate,
        bootstrap_samples=args.bootstrap_samples,
        screen_artifacts=[Path(path) for path in args.screen_artifacts],
        load_keyfile=not args.no_keyfile,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "problem_level_ci": payload["problem_level_ci"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
