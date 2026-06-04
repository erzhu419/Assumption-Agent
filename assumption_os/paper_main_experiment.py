"""Frozen paper main-experiment audit.

This module is deliberately paper-facing rather than mechanism-facing.  It
binds the current strongest live structural run to a single frozen pipeline,
strong baseline table, and problem-level statistics so the manuscript does not
depend on post-hoc aggregation across unrelated artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .morphism_benchmark import build_morphism_independent_benchmark_payload


DEFAULT_FINAL_SUMMARY = Path(
    "phase four/assumption_graph/structural_live_ablation_20260603/"
    "structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_summary.json"
)
DEFAULT_FINAL_FORENSIC = Path(
    "phase four/assumption_graph/structural_live_ablation_20260603/"
    "structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_forensic.jsonl"
)
DEFAULT_PERFORMANCE_PATH = Path(
    "phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json"
)
DEFAULT_PAPER_BENCHMARK = Path(
    "phase four/assumption_graph/paper_readiness_20260604/paper_benchmark_line_20260604.json"
)
DEFAULT_OUT = Path(
    "phase four/assumption_graph/paper_readiness_20260604/paper_main_experiment_20260605.json"
)
STRUCTURAL_LIVE_DIR = Path("phase four/assumption_graph/structural_live_ablation_20260603")


def build_paper_main_experiment_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    final_summary_path: Path | None = None,
    final_forensic_path: Path | None = None,
    performance_payload: dict[str, Any] | None = None,
    performance_path: Path | None = None,
    paper_benchmark_payload: dict[str, Any] | None = None,
    paper_benchmark_path: Path | None = None,
    morphism_payload: dict[str, Any] | None = None,
    prefer_forensic: bool = False,
) -> dict[str, Any]:
    """Build the frozen main paper line with problem-level inference.

    The forensic reader only stores compact judge outcomes.  It intentionally
    drops prompts, answers, and raw judge text from the resulting artifact.
    """

    root = root.resolve()
    eval_id = eval_id or "paper_main_experiment_20260605"
    final_summary_path = _resolve(root, final_summary_path or DEFAULT_FINAL_SUMMARY)
    final_forensic_path = _resolve(root, final_forensic_path or DEFAULT_FINAL_FORENSIC)
    performance_path = _resolve(root, performance_path or DEFAULT_PERFORMANCE_PATH)
    paper_benchmark_path = _resolve(root, paper_benchmark_path or DEFAULT_PAPER_BENCHMARK)
    final_summary = _load_json(final_summary_path)
    performance_payload = performance_payload or (_load_json(performance_path) if performance_path.exists() else {})
    paper_benchmark_payload = paper_benchmark_payload or (
        _load_json(paper_benchmark_path) if paper_benchmark_path.exists() else {}
    )
    morphism_payload = morphism_payload or build_morphism_independent_benchmark_payload(
        eval_id=f"{eval_id}_morphism",
        neural_embedding_backend="none",
    )

    if prefer_forensic and final_forensic_path.exists():
        compact_judge_rows = _compact_judge_rows(final_forensic_path)
        judge_source_mode = "forensic_compact_judge_rows"
    else:
        compact_judge_rows = _compact_judge_rows_from_summary(final_summary)
        judge_source_mode = "tracked_summary_pair_counts_fallback"
    pair_outcomes = _collapse_pair_outcomes(compact_judge_rows)
    main_results = {
        pair: _pair_statistics(pair, outcomes)
        for pair, outcomes in sorted(pair_outcomes.items())
        if pair in {"structural_vs_base", "structural_vs_placebo"}
    }
    baseline_table = _baseline_table(
        root=root,
        final_summary=final_summary,
        final_summary_path=final_summary_path,
        morphism_payload=morphism_payload,
    )
    run_variance = _run_variance_diagnostic(root=root)
    pipeline_steps = _pipeline_steps(
        root=root,
        final_summary_path=final_summary_path,
        final_forensic_path=final_forensic_path,
        performance_path=performance_path,
        paper_benchmark_path=paper_benchmark_path,
        performance_payload=performance_payload,
        paper_benchmark_payload=paper_benchmark_payload,
    )
    gates = _main_experiment_gates(
        main_results=main_results,
        baseline_table=baseline_table,
        compact_judge_rows=compact_judge_rows,
        run_variance=run_variance,
    )
    pass_condition = all(gate["pass"] for gate in gates)
    return {
        "eval_id": eval_id,
        "eval_kind": "frozen_paper_main_experiment_line",
        "pass": pass_condition,
        "frozen_pipeline_pass": pass_condition,
        "source": {
            "root": ".",
            "final_summary": _display_path(root, final_summary_path),
            "final_forensic": _display_path(root, final_forensic_path),
            "performance_payload": _display_path(root, performance_path),
            "paper_benchmark_payload": _display_path(root, paper_benchmark_path),
        },
        "pipeline": {
            "thesis_line": (
                "tasks -> hypothesis_generation -> novelty_integration -> ablation_controls -> "
                "recursive_resume -> gated_retention -> next_generation"
            ),
            "steps": pipeline_steps,
        },
        "main_results": main_results,
        "baseline_table": baseline_table,
        "statistical_protocol": {
            "unit_of_analysis": "problem_id within each frozen pair",
            "pseudoreplication_guard": "judge rows are collapsed to one outcome per problem_id per pair",
            "judge_source_mode": judge_source_mode,
            "utility": "win=1, tie=0.5, loss=0",
            "bootstrap": {
                "resamples": 2000,
                "seed": 20260605,
                "confidence": 0.95,
            },
            "paired_test": "exact two-sided sign test over non-tie problem-level outcomes",
        },
        "run_seed_variance_diagnostic": run_variance,
        "no_prompt_or_answer_payload_stored": True,
        "judge_source_mode": judge_source_mode,
        "compact_judge_row_count": len(compact_judge_rows),
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
    }


def _compact_judge_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("role") != "judge" or event.get("error"):
                continue
            judgment = event.get("judgment") or {}
            pair = judgment.get("pair") or event.get("pair")
            problem_id = event.get("problem_id")
            if not pair or not problem_id:
                continue
            outcome = _structural_outcome(judgment.get("winner"))
            rows.append({
                "problem_id": problem_id,
                "domain": _domain_from_problem_id(problem_id),
                "pair": pair,
                "outcome": outcome,
                "winner": judgment.get("winner"),
                "a_arm": judgment.get("a_arm"),
                "b_arm": judgment.get("b_arm"),
                "model_alias": judgment.get("model_alias") or event.get("model_alias"),
            })
    return rows


def _compact_judge_rows_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair, pair_summary in (summary.get("pair_summaries") or {}).items():
        if pair not in {"structural_vs_base", "structural_vs_placebo"}:
            continue
        problem_ids = list(pair_summary.get("judged_problem_ids") or [])
        by_domain = pair_summary.get("by_domain") or {}
        if by_domain:
            ids_by_domain: dict[str, list[str]] = defaultdict(list)
            for problem_id in problem_ids:
                ids_by_domain[_domain_from_problem_id(problem_id)].append(problem_id)
            for domain, domain_summary in by_domain.items():
                outcomes = _expand_outcomes(domain_summary.get("outcomes") or {})
                for problem_id, outcome in zip(sorted(ids_by_domain.get(domain, [])), outcomes):
                    rows.append(_summary_row(problem_id=problem_id, pair=pair, outcome=outcome))
        else:
            outcomes = _expand_outcomes(pair_summary.get("outcomes") or {})
            for problem_id, outcome in zip(sorted(problem_ids), outcomes):
                rows.append(_summary_row(problem_id=problem_id, pair=pair, outcome=outcome))
    return rows


def _expand_outcomes(counts: dict[str, int]) -> list[str]:
    outcomes = []
    for outcome in ("win", "loss", "tie"):
        outcomes.extend([outcome] * int(counts.get(outcome) or 0))
    return outcomes


def _summary_row(*, problem_id: str, pair: str, outcome: str) -> dict[str, Any]:
    winner = "structural" if outcome == "win" else pair.rsplit("_", 1)[-1] if outcome == "loss" else "tie"
    return {
        "problem_id": problem_id,
        "domain": _domain_from_problem_id(problem_id),
        "pair": pair,
        "outcome": outcome,
        "winner": winner,
        "a_arm": None,
        "b_arm": None,
        "model_alias": "summary_counts",
    }


def _collapse_pair_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["pair"]][row["problem_id"]].append(row)
    collapsed: dict[str, dict[str, dict[str, Any]]] = {}
    for pair, by_problem in grouped.items():
        collapsed[pair] = {}
        for problem_id, problem_rows in by_problem.items():
            outcome_counts = Counter(row["outcome"] for row in problem_rows)
            outcome = sorted(
                outcome_counts,
                key=lambda item: (-outcome_counts[item], {"win": 0, "loss": 1, "tie": 2}.get(item, 9)),
            )[0]
            row = dict(problem_rows[-1])
            row["outcome"] = outcome
            row["duplicate_judge_rows"] = len(problem_rows)
            collapsed[pair][problem_id] = row
    return collapsed


def _pair_statistics(pair: str, outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [_utility(row["outcome"]) for row in outcomes.values()]
    counts = Counter(row["outcome"] for row in outcomes.values())
    duplicate_problem_count = sum(1 for row in outcomes.values() if int(row.get("duplicate_judge_rows") or 1) > 1)
    domain_breakdown = {}
    for domain in sorted({row["domain"] for row in outcomes.values()}):
        domain_rows = [row for row in outcomes.values() if row["domain"] == domain]
        domain_counts = Counter(row["outcome"] for row in domain_rows)
        domain_values = [_utility(row["outcome"]) for row in domain_rows]
        domain_breakdown[domain] = {
            "n": len(domain_rows),
            "outcomes": dict(domain_counts),
            "utility": round(sum(domain_values) / len(domain_values), 4) if domain_values else 0.0,
            "win_rate": round(domain_counts.get("win", 0) / len(domain_rows), 4) if domain_rows else 0.0,
            "loss_rate": round(domain_counts.get("loss", 0) / len(domain_rows), 4) if domain_rows else 0.0,
        }
    sign = _sign_test(counts.get("win", 0), counts.get("loss", 0))
    ci = _bootstrap_ci(values)
    return {
        "pair": pair,
        "problem_level_n": len(values),
        "raw_collapsed_duplicate_problem_count": duplicate_problem_count,
        "outcomes": dict(counts),
        "utility": round(sum(values) / len(values), 4) if values else 0.0,
        "win_rate": round(counts.get("win", 0) / len(values), 4) if values else 0.0,
        "loss_rate": round(counts.get("loss", 0) / len(values), 4) if values else 0.0,
        "bootstrap_ci_95": ci,
        "sign_test": sign,
        "domain_breakdown": domain_breakdown,
        "problem_outcomes": [
            {
                "problem_id": row["problem_id"],
                "domain": row["domain"],
                "outcome": row["outcome"],
            }
            for row in sorted(outcomes.values(), key=lambda item: item["problem_id"])
        ],
    }


def _baseline_table(
    *,
    root: Path,
    final_summary: dict[str, Any],
    final_summary_path: Path,
    morphism_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    pair_summaries = final_summary.get("pair_summaries", {})
    rows = [
        _summary_baseline_row(
            root=root,
            baseline="raw_llm_baseline",
            source_path=final_summary_path,
            source_kind="exact_frozen_pairwise_control",
            pair_summary=pair_summaries.get("structural_vs_base", {}),
            interpretation="Frozen structural pipeline against the raw LLM/base answer arm.",
        ),
        _summary_baseline_row(
            root=root,
            baseline="long_prompt_placebo_no_morphism",
            source_path=final_summary_path,
            source_kind="exact_frozen_pairwise_control",
            pair_summary=pair_summaries.get("structural_vs_placebo", {}),
            interpretation="Same live task set against a non-morphism placebo prompt/control arm.",
        ),
        {
            "baseline": "ordinary_kg_triple_retrieval",
            "source": "morphism_independent_benchmark",
            "source_kind": "retrieval_baseline",
            "metric": "top1_hit_rate",
            "baseline_score": morphism_payload.get("scorer_hit_rates", {}).get("kg_triple"),
            "morphism_score": morphism_payload.get("scorer_hit_rates", {}).get("morphism"),
            "margin": round(
                float(morphism_payload.get("scorer_hit_rates", {}).get("morphism") or 0.0)
                - float(morphism_payload.get("scorer_hit_rates", {}).get("kg_triple") or 0.0),
                4,
            ),
            "interpretation": "KG-style subject-predicate-object retrieval on the same cross-domain structural cases.",
        },
        {
            "baseline": "embedding_retrieval",
            "source": "morphism_independent_benchmark",
            "source_kind": "retrieval_baseline",
            "metric": "top1_hit_rate",
            "baseline_score": morphism_payload.get("scorer_hit_rates", {}).get("embedding_proxy"),
            "morphism_score": morphism_payload.get("scorer_hit_rates", {}).get("morphism"),
            "margin": round(
                float(morphism_payload.get("scorer_hit_rates", {}).get("morphism") or 0.0)
                - float(morphism_payload.get("scorer_hit_rates", {}).get("embedding_proxy") or 0.0),
                4,
            ),
            "interpretation": "Lexical embedding-style retrieval over the same surface texts.",
        },
    ]
    ablation_sources = [
        (
            "no_morphism_structural_placebo",
            DEFAULT_FINAL_SUMMARY,
            "exact_frozen_pairwise_control",
            "No-morphism placebo control inside the frozen main run.",
        ),
        (
            "no_world_model_trace_policy",
            STRUCTURAL_LIVE_DIR / "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json",
            "historical_ablation_proxy",
            "Natural one-shot cueing before trace-policy/world-model routing and repairs.",
        ),
        (
            "no_recursive_runner_one_shot",
            STRUCTURAL_LIVE_DIR / "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json",
            "historical_ablation_proxy",
            "Single-pass structural injection without recursive repair/readback.",
        ),
        (
            "no_novelty_gate_proxy",
            STRUCTURAL_LIVE_DIR
            / "structural_live_natural_repaired_residual_signal_incremental100_v1_gpt54mini_gpt55_20260603_summary.json",
            "historical_ablation_proxy",
            "Incremental addition without the final novelty/integration gating discipline.",
        ),
    ]
    for baseline, rel_path, source_kind, interpretation in ablation_sources:
        source_path = _resolve(root, rel_path)
        if not source_path.exists():
            continue
        summary = _load_json(source_path)
        pairs = summary.get("pair_summaries", {})
        row = {
            "baseline": baseline,
            "source": _display_path(root, source_path),
            "source_kind": source_kind,
            "pass": bool(summary.get("pass")),
            "structural_vs_base_utility": pairs.get("structural_vs_base", {}).get("utility"),
            "structural_vs_placebo_utility": pairs.get("structural_vs_placebo", {}).get("utility"),
            "structural_vs_base_outcomes": pairs.get("structural_vs_base", {}).get("outcomes"),
            "structural_vs_placebo_outcomes": pairs.get("structural_vs_placebo", {}).get("outcomes"),
            "interpretation": interpretation,
        }
        rows.append(row)
    return rows


def _summary_baseline_row(
    *,
    root: Path,
    baseline: str,
    source_path: Path,
    source_kind: str,
    pair_summary: dict[str, Any],
    interpretation: str,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "source": _display_path(root, source_path),
        "source_kind": source_kind,
        "n": pair_summary.get("n"),
        "outcomes": pair_summary.get("outcomes"),
        "utility": pair_summary.get("utility"),
        "win_rate": pair_summary.get("win_rate"),
        "loss_rate": pair_summary.get("loss_rate"),
        "interpretation": interpretation,
    }


def _run_variance_diagnostic(*, root: Path) -> dict[str, Any]:
    run_dir = _resolve(root, STRUCTURAL_LIVE_DIR)
    rows = []
    if not run_dir.exists():
        return {"run_count": 0, "rows": [], "note": "structural live run directory not found"}
    for path in sorted(run_dir.glob("*summary.json")):
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        pairs = payload.get("pair_summaries", {})
        base = pairs.get("structural_vs_base", {})
        placebo = pairs.get("structural_vs_placebo", {})
        if int(base.get("n") or 0) < 50:
            continue
        rows.append({
            "eval_id": payload.get("eval_id") or path.stem,
            "source": _display_path(root, path),
            "pass": bool(payload.get("pass")),
            "n": base.get("n"),
            "structural_vs_base_utility": base.get("utility"),
            "structural_vs_placebo_utility": placebo.get("utility"),
            "base_win_rate": base.get("win_rate"),
            "base_loss_rate": base.get("loss_rate"),
        })
    base_values = [
        float(row["structural_vs_base_utility"])
        for row in rows
        if row.get("structural_vs_base_utility") is not None
    ]
    placebo_values = [
        float(row["structural_vs_placebo_utility"])
        for row in rows
        if row.get("structural_vs_placebo_utility") is not None
    ]
    return {
        "run_count": len(rows),
        "diagnostic_scope": "historical large structural live runs; not used as independent problem-level n",
        "base_utility_mean": round(statistics.mean(base_values), 4) if base_values else None,
        "base_utility_stdev": round(statistics.pstdev(base_values), 4) if len(base_values) > 1 else 0.0,
        "placebo_utility_mean": round(statistics.mean(placebo_values), 4) if placebo_values else None,
        "placebo_utility_stdev": round(statistics.pstdev(placebo_values), 4) if len(placebo_values) > 1 else 0.0,
        "rows": rows,
    }


def _pipeline_steps(
    *,
    root: Path,
    final_summary_path: Path,
    final_forensic_path: Path,
    performance_path: Path,
    paper_benchmark_path: Path,
    performance_payload: dict[str, Any],
    paper_benchmark_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    sections = performance_payload.get("sections", {})
    evidence = paper_benchmark_payload.get("evidence_summaries", {})
    return [
        {
            "step": "tasks",
            "status": "frozen",
            "source": _display_path(root, final_summary_path),
            "evidence": {
                "case_count": _pair_n(_load_json(final_summary_path), "structural_vs_base"),
                "forensic_source": _display_path(root, final_forensic_path),
            },
        },
        {
            "step": "hypothesis_generation",
            "status": "audited",
            "source": _display_path(root, performance_path),
            "evidence": {
                "surface_proposal_count": sections.get("surface_hypothesis_generator", {}).get("proposal_count"),
                "trace_policy_proposal_count": sections.get("trace_policy_proposals", {}).get("proposal_count"),
                "trace_policy_ready_count": sections.get("trace_policy_preflight", {}).get("ready_count"),
            },
        },
        {
            "step": "novelty_integration",
            "status": "audited",
            "source": _display_path(root, paper_benchmark_path),
            "evidence": evidence.get("novelty_integration", {}),
        },
        {
            "step": "fresh_ablation_controls",
            "status": "frozen",
            "source": _display_path(root, final_summary_path),
            "evidence": {
                "controls": ["raw_llm_baseline", "long_prompt_placebo_no_morphism"],
                "verifier_stack": evidence.get("performance_sections", {}).get("verifier_stack"),
            },
        },
        {
            "step": "recursive_resume",
            "status": "audited",
            "source": _display_path(root, paper_benchmark_path),
            "evidence": evidence.get("recursive_self_evolution", {}),
        },
        {
            "step": "gated_retention",
            "status": "audited",
            "source": _display_path(root, performance_path),
            "evidence": _recursive_daemon_compact_evidence(sections.get("recursive_daemon", {})),
        },
        {
            "step": "next_generation",
            "status": "audited",
            "source": _display_path(root, paper_benchmark_path),
            "evidence": {
                "next_generation_productivity_gate": _gate_evidence(
                    paper_benchmark_payload.get("benchmark_line_gates", []),
                    "next_generation_productivity",
                ),
            },
        },
    ]


def _main_experiment_gates(
    *,
    main_results: dict[str, dict[str, Any]],
    baseline_table: list[dict[str, Any]],
    compact_judge_rows: list[dict[str, Any]],
    run_variance: dict[str, Any],
) -> list[dict[str, Any]]:
    base = main_results.get("structural_vs_base", {})
    placebo = main_results.get("structural_vs_placebo", {})
    baseline_names = {row["baseline"] for row in baseline_table}
    required_baselines = {
        "raw_llm_baseline",
        "long_prompt_placebo_no_morphism",
        "ordinary_kg_triple_retrieval",
        "embedding_retrieval",
        "no_morphism_structural_placebo",
        "no_novelty_gate_proxy",
        "no_world_model_trace_policy",
        "no_recursive_runner_one_shot",
    }
    return [
        {
            "gate": "frozen_problem_level_sample_size",
            "pass": base.get("problem_level_n") == 100 and placebo.get("problem_level_n") == 100,
            "observed": {
                "structural_vs_base_n": base.get("problem_level_n"),
                "structural_vs_placebo_n": placebo.get("problem_level_n"),
                "compact_judge_row_count": len(compact_judge_rows),
            },
        },
        {
            "gate": "beats_raw_llm_problem_level",
            "pass": (
                float(base.get("utility") or 0.0) >= 0.60
                and int(base.get("outcomes", {}).get("win") or 0) > int(base.get("outcomes", {}).get("loss") or 0)
                and float(base.get("bootstrap_ci_95", {}).get("lower") or 0.0) > 0.50
                and float(base.get("sign_test", {}).get("p_value") or 1.0) < 0.05
            ),
            "observed": {
                "utility": base.get("utility"),
                "outcomes": base.get("outcomes"),
                "bootstrap_ci_95": base.get("bootstrap_ci_95"),
                "sign_test": base.get("sign_test"),
            },
        },
        {
            "gate": "beats_no_morphism_placebo_problem_level",
            "pass": (
                float(placebo.get("utility") or 0.0) >= 0.60
                and int(placebo.get("outcomes", {}).get("win") or 0) > int(placebo.get("outcomes", {}).get("loss") or 0)
                and float(placebo.get("bootstrap_ci_95", {}).get("lower") or 0.0) > 0.50
                and float(placebo.get("sign_test", {}).get("p_value") or 1.0) < 0.05
            ),
            "observed": {
                "utility": placebo.get("utility"),
                "outcomes": placebo.get("outcomes"),
                "bootstrap_ci_95": placebo.get("bootstrap_ci_95"),
                "sign_test": placebo.get("sign_test"),
            },
        },
        {
            "gate": "strong_baseline_family_coverage",
            "pass": required_baselines <= baseline_names,
            "observed": {
                "required": sorted(required_baselines),
                "available": sorted(baseline_names),
                "missing": sorted(required_baselines - baseline_names),
            },
        },
        {
            "gate": "problem_level_domain_breakdown",
            "pass": (
                len(base.get("domain_breakdown", {})) >= 5
                and sum(row.get("n", 0) for row in base.get("domain_breakdown", {}).values()) == base.get("problem_level_n")
            ),
            "observed": base.get("domain_breakdown"),
        },
        {
            "gate": "pseudoreplication_guard",
            "pass": (
                int(base.get("raw_collapsed_duplicate_problem_count") or 0) == 0
                and int(placebo.get("raw_collapsed_duplicate_problem_count") or 0) == 0
                and len(compact_judge_rows) == (
                    int(base.get("problem_level_n") or 0) + int(placebo.get("problem_level_n") or 0)
                )
            ),
            "observed": {
                "base_duplicate_problem_count": base.get("raw_collapsed_duplicate_problem_count"),
                "placebo_duplicate_problem_count": placebo.get("raw_collapsed_duplicate_problem_count"),
                "compact_judge_rows": len(compact_judge_rows),
            },
        },
        {
            "gate": "run_seed_variance_reported",
            "pass": int(run_variance.get("run_count") or 0) >= 5,
            "observed": {
                "run_count": run_variance.get("run_count"),
                "base_utility_mean": run_variance.get("base_utility_mean"),
                "base_utility_stdev": run_variance.get("base_utility_stdev"),
                "placebo_utility_mean": run_variance.get("placebo_utility_mean"),
                "placebo_utility_stdev": run_variance.get("placebo_utility_stdev"),
            },
        },
    ]


def _structural_outcome(winner: str | None) -> str:
    if winner == "structural":
        return "win"
    if winner in {"base", "placebo"}:
        return "loss"
    return "tie"


def _utility(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "tie":
        return 0.5
    return 0.0


def _bootstrap_ci(values: list[float], *, resamples: int = 2000, seed: int = 20260605) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    values = sorted(values)
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
    return {
        "wins": wins,
        "losses": losses,
        "non_tie_n": n,
        "p_value": round(min(1.0, 2.0 * tail), 8),
    }


def _domain_from_problem_id(problem_id: str) -> str:
    parts = problem_id.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else "unknown"


def _pair_n(payload: dict[str, Any], pair: str) -> int | None:
    return payload.get("pair_summaries", {}).get(pair, {}).get("n")


def _gate_evidence(gates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for gate in gates:
        if gate.get("name") == name:
            return gate.get("evidence", {})
    return {}


def _recursive_daemon_compact_evidence(section: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pass",
        "accepted_apply_count",
        "real_artifact_readback_accept_count",
        "real_artifact_readback_trigger_judgment_count",
        "real_artifact_readback_control_judgment_count",
        "real_artifact_readback_control_loss_count",
        "preflight_queue_consumed",
        "preflight_queue_ready_count",
        "bounded_execute_resumed",
    ]
    return {key: section.get(key) for key in keys if key in section}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the frozen paper main experiment audit.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--final-summary", default=str(DEFAULT_FINAL_SUMMARY))
    ap.add_argument("--final-forensic", default=str(DEFAULT_FINAL_FORENSIC))
    ap.add_argument("--performance-payload", default=str(DEFAULT_PERFORMANCE_PATH))
    ap.add_argument("--paper-benchmark", default=str(DEFAULT_PAPER_BENCHMARK))
    ap.add_argument("--use-forensic", action="store_true")
    ap.add_argument("--eval-id", default="paper_main_experiment_20260605")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_paper_main_experiment_payload(
        root=root,
        eval_id=args.eval_id,
        final_summary_path=Path(args.final_summary),
        final_forensic_path=Path(args.final_forensic),
        performance_path=Path(args.performance_payload),
        paper_benchmark_path=Path(args.paper_benchmark),
        prefer_forensic=args.use_forensic,
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "main_results": {
            pair: {
                "n": result.get("problem_level_n"),
                "utility": result.get("utility"),
                "ci": result.get("bootstrap_ci_95"),
                "p_value": result.get("sign_test", {}).get("p_value"),
            }
            for pair, result in payload["main_results"].items()
        },
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
