"""Paired comparison for two HLE shard runs.

The comparison is intentionally content-blind: it reads only problem hashes,
variant names, correctness booleans, error metadata, and persistence flags from
existing shard artifacts.  It never reads or writes raw HLE questions, answers,
or prediction text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_VARIANT = "assumption_agent_recursive_verify"


def compare_hle_runs(
    *,
    candidate_run_dir: Path,
    baseline_run_dir: Path,
    candidate_variant: str = DEFAULT_CANDIDATE_VARIANT,
    baseline_profile: str = "",
    baseline_variants: list[str] | None = None,
    eval_id: str = "",
    expected_sample_size: int | None = None,
    primary_baseline_variant: str = "raw",
    bootstrap_samples: int = 4000,
    seed: int = 20260618,
) -> dict[str, Any]:
    candidate_rows = _load_rows_from_run(candidate_run_dir, profile="")
    baseline_rows = _load_rows_from_run(baseline_run_dir, profile=baseline_profile)
    candidate_by_problem = _rows_by_problem(candidate_rows, candidate_variant)
    available_baseline_variants = sorted({str(row.get("variant")) for row in baseline_rows if row.get("variant")})
    selected_baseline_variants = baseline_variants or available_baseline_variants
    baseline_by_variant = {
        variant: _rows_by_problem(baseline_rows, variant)
        for variant in selected_baseline_variants
    }
    comparisons = {
        variant: _paired_delta(
            candidate_by_problem=candidate_by_problem,
            candidate_variant=candidate_variant,
            baseline_by_problem=baseline_by_variant.get(variant, {}),
            baseline_variant=variant,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, variant in enumerate(selected_baseline_variants)
    }
    oracle = _oracle_summary(
        candidate_by_problem=candidate_by_problem,
        baseline_by_variant=baseline_by_variant,
        baseline_variants=selected_baseline_variants,
    )
    pollution = _pollution_summary(candidate_rows=candidate_rows, baseline_rows=baseline_rows)
    performance = _performance_validation(
        candidate_by_problem=candidate_by_problem,
        comparisons=comparisons,
        pollution=pollution,
        expected_sample_size=expected_sample_size,
        primary_baseline_variant=primary_baseline_variant,
    )
    payload = {
        "eval_id": eval_id or f"{candidate_run_dir.name}_vs_{baseline_run_dir.name}",
        "eval_kind": "hle_paired_run_comparison",
        "candidate": {
            "run_dir": str(candidate_run_dir),
            "variant": candidate_variant,
            "problem_count": len(candidate_by_problem),
            "correct_count": sum(1 for row in candidate_by_problem.values() if row.get("correct") is True),
            "error_count": sum(1 for row in candidate_by_problem.values() if row.get("error")),
            "accuracy": _accuracy(candidate_by_problem.values()),
        },
        "baseline": {
            "run_dir": str(baseline_run_dir),
            "profile": baseline_profile,
            "variants": selected_baseline_variants,
            "available_variants": available_baseline_variants,
        },
        "comparisons": comparisons,
        "oracle_summary": oracle,
        "pollution_summary": pollution,
        "performance_validation": performance,
        "raw_content_persisted": pollution["raw_content_persisted"],
        "pass": performance["overall_pass"],
        "failed_gates": [
            name for name, passed in performance["gates"].items() if passed is not True
        ],
    }
    return payload


def _load_rows_from_run(run_dir: Path, *, profile: str = "") -> list[dict[str, Any]]:
    directory = run_dir / profile if profile else run_dir
    rows: list[dict[str, Any]] = []
    for shard in sorted(directory.glob("*_shard_*.json")):
        if shard.name.endswith(".problem_hashes.json"):
            continue
        try:
            payload = json.loads(shard.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched["_source_shard"] = str(shard)
            rows.append(enriched)
    return rows


def _rows_by_problem(rows: list[dict[str, Any]], variant: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem = row.get("problem_id_hash")
        if row.get("variant") == variant and problem:
            out[str(problem)] = row
    return out


def _paired_delta(
    *,
    candidate_by_problem: dict[str, dict[str, Any]],
    candidate_variant: str,
    baseline_by_problem: dict[str, dict[str, Any]],
    baseline_variant: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(candidate_by_problem) & set(baseline_by_problem))
    missing_candidate = sorted(set(baseline_by_problem) - set(candidate_by_problem))
    missing_baseline = sorted(set(candidate_by_problem) - set(baseline_by_problem))
    diffs = [
        _score_bool(candidate_by_problem[problem].get("correct"))
        - _score_bool(baseline_by_problem[problem].get("correct"))
        for problem in shared
    ]
    wins = sum(1 for value in diffs if value > 0)
    losses = sum(1 for value in diffs if value < 0)
    candidate_correct = sum(_score_bool(candidate_by_problem[problem].get("correct")) for problem in shared)
    baseline_correct = sum(_score_bool(baseline_by_problem[problem].get("correct")) for problem in shared)
    return {
        "candidate_variant": candidate_variant,
        "baseline_variant": baseline_variant,
        "shared_n": len(shared),
        "candidate_accuracy": candidate_correct / len(shared) if shared else None,
        "baseline_accuracy": baseline_correct / len(shared) if shared else None,
        "delta": sum(diffs) / len(diffs) if diffs else None,
        "wins": wins,
        "losses": losses,
        "ties": sum(1 for value in diffs if value == 0),
        "sign_test_two_sided_p": _sign_test_two_sided_p(wins=wins, losses=losses),
        "bootstrap_ci95": _bootstrap_ci(diffs, samples=bootstrap_samples, seed=seed),
        "missing_candidate_count": len(missing_candidate),
        "missing_baseline_count": len(missing_baseline),
        "missing_candidate_hashes": missing_candidate,
        "missing_baseline_hashes": missing_baseline,
    }


def _oracle_summary(
    *,
    candidate_by_problem: dict[str, dict[str, Any]],
    baseline_by_variant: dict[str, dict[str, dict[str, Any]]],
    baseline_variants: list[str],
) -> dict[str, Any]:
    shared: set[str] = set(candidate_by_problem)
    for variant in baseline_variants:
        shared &= set(baseline_by_variant.get(variant, {}))
    pattern_counts: Counter[str] = Counter()
    candidate_only = 0
    all_wrong = 0
    oracle_correct = 0
    for problem in sorted(shared):
        baseline_bits = [
            "1" if baseline_by_variant[variant][problem].get("correct") is True else "0"
            for variant in baseline_variants
        ]
        candidate_bit = "1" if candidate_by_problem[problem].get("correct") is True else "0"
        pattern = "".join(baseline_bits + [candidate_bit])
        pattern_counts[pattern] += 1
        any_baseline_correct = any(bit == "1" for bit in baseline_bits)
        if candidate_bit == "1" and not any_baseline_correct:
            candidate_only += 1
        if candidate_bit == "0" and not any_baseline_correct:
            all_wrong += 1
        if candidate_bit == "1" or any_baseline_correct:
            oracle_correct += 1
    return {
        "shared_n": len(shared),
        "baseline_variant_order": baseline_variants,
        "pattern_format": "baseline bits in order followed by candidate bit",
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "candidate_only_correct_count": candidate_only,
        "all_wrong_count": all_wrong,
        "oracle_accuracy": oracle_correct / len(shared) if shared else None,
    }


def _pollution_summary(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(candidate_rows) + list(baseline_rows)
    raw_content_persisted = any(
        row.get("raw_question_persisted")
        or row.get("gold_answer_persisted")
        or row.get("prediction_text_persisted")
        for row in rows
    )
    return {
        "raw_content_persisted": bool(raw_content_persisted),
        "candidate_row_count": len(candidate_rows),
        "baseline_row_count": len(baseline_rows),
    }


def _performance_validation(
    *,
    candidate_by_problem: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    pollution: dict[str, Any],
    expected_sample_size: int | None,
    primary_baseline_variant: str,
) -> dict[str, Any]:
    primary = comparisons.get(primary_baseline_variant) or {}
    gates = {
        "candidate_rows_present": len(candidate_by_problem) > 0,
        "expected_sample_complete": (
            expected_sample_size is None or len(candidate_by_problem) == expected_sample_size
        ),
        "primary_shared_n_positive": int(primary.get("shared_n") or 0) > 0,
        "candidate_not_below_primary": (
            primary.get("delta") is not None and float(primary.get("delta")) >= 0.0
        ),
        "candidate_error_free": sum(1 for row in candidate_by_problem.values() if row.get("error")) == 0,
        "raw_content_not_persisted": pollution.get("raw_content_persisted") is False,
    }
    return {
        "overall_pass": all(value is True for value in gates.values()),
        "gates": gates,
        "primary_baseline_variant": primary_baseline_variant,
        "note": (
            "The primary gate only checks that the candidate is not below the selected baseline; "
            "effect-size and CI should be reported separately."
        ),
    }


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float | None]:
    if not values:
        return {"lower": None, "mean": None, "upper": None}
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(max(1, samples)):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {"lower": lower, "mean": sum(values) / n, "upper": upper}


def _sign_test_two_sided_p(*, wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def _accuracy(rows: Any) -> float | None:
    rows = list(rows)
    if not rows:
        return None
    return sum(1 for row in rows if row.get("correct") is True) / len(rows)


def _score_bool(value: Any) -> int:
    return 1 if value is True else 0


def format_comparison_markdown(payload: dict[str, Any]) -> str:
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    lines = [
        "# HLE Paired Run Comparison",
        "",
        f"- eval id: `{payload['eval_id']}`",
        f"- pass: `{payload['pass']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        f"- candidate variant: `{candidate.get('variant')}`",
        f"- candidate problem count: `{candidate.get('problem_count')}`",
        f"- candidate accuracy: `{candidate.get('accuracy')}`",
        f"- baseline profile: `{baseline.get('profile')}`",
        f"- raw content persisted: `{payload.get('raw_content_persisted')}`",
        "",
        "## Paired Deltas",
        "",
        "| baseline variant | n | candidate acc | baseline acc | delta | wins | losses | p | ci95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant, row in sorted((payload.get("comparisons") or {}).items()):
        ci = row.get("bootstrap_ci95") or {}
        lines.append(
            f"| `{variant}` | `{row.get('shared_n')}` | `{row.get('candidate_accuracy')}` | "
            f"`{row.get('baseline_accuracy')}` | `{row.get('delta')}` | `{row.get('wins')}` | "
            f"`{row.get('losses')}` | `{row.get('sign_test_two_sided_p')}` | "
            f"`[{ci.get('lower')}, {ci.get('upper')}]` |"
        )
    oracle = payload.get("oracle_summary") or {}
    lines.extend([
        "",
        "## Oracle Pattern",
        "",
        f"- shared n: `{oracle.get('shared_n')}`",
        f"- baseline order: `{oracle.get('baseline_variant_order')}`",
        f"- candidate-only correct: `{oracle.get('candidate_only_correct_count')}`",
        f"- all wrong: `{oracle.get('all_wrong_count')}`",
        f"- oracle accuracy: `{oracle.get('oracle_accuracy')}`",
        "",
        "| pattern | count |",
        "| --- | ---: |",
    ])
    for pattern, count in sorted((oracle.get("pattern_counts") or {}).items()):
        lines.append(f"| `{pattern}` | `{count}` |")
    lines.extend([
        "",
        "## Validation Gates",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ])
    for gate, passed in sorted(((payload.get("performance_validation") or {}).get("gates") or {}).items()):
        lines.append(f"| `{gate}` | `{passed}` |")
    lines.append("")
    lines.append("The comparison uses hashes and correctness booleans only; raw HLE content is not persisted.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two HLE shard runs on shared problem hashes.")
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE_VARIANT)
    parser.add_argument("--baseline-profile", default="")
    parser.add_argument("--baseline-variants", default="")
    parser.add_argument("--eval-id", default="")
    parser.add_argument("--expected-sample-size", type=int, default=0)
    parser.add_argument("--primary-baseline-variant", default="raw")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()
    baseline_variants = [item.strip() for item in args.baseline_variants.split(",") if item.strip()]
    payload = compare_hle_runs(
        candidate_run_dir=Path(args.candidate_run_dir),
        baseline_run_dir=Path(args.baseline_run_dir),
        candidate_variant=args.candidate_variant,
        baseline_profile=args.baseline_profile,
        baseline_variants=baseline_variants or None,
        eval_id=args.eval_id,
        expected_sample_size=args.expected_sample_size or None,
        primary_baseline_variant=args.primary_baseline_variant,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    out = Path(args.out) if args.out else Path(args.candidate_run_dir) / f"{payload['eval_id']}_comparison.json"
    md_out = Path(args.md_out) if args.md_out else Path("reconstruction/md") / f"{payload['eval_id']}_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(format_comparison_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "md_out": str(md_out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
