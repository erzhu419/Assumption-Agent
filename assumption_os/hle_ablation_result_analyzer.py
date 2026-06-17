"""Analyze HLE module ablation shard outputs.

The live runner intentionally stores only hashes, aggregate metadata, module
trace flags, and correctness booleans.  This analyzer keeps the same boundary:
it never reads or writes raw HLE questions, gold answers, or predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AGENT_VARIANT = "assumption_agent_recursive_verify"
RAW_VARIANT = "raw"
HIPPORAG_VARIANT = "hipporag_baseline"


def analyze_hle_ablation_run(
    *,
    run_dir: Path,
    eval_id: str = "",
    bootstrap_samples: int = 2000,
    seed: int = 20260617,
) -> dict[str, Any]:
    profile_rows: dict[str, list[dict[str, Any]]] = {}
    profile_shards: dict[str, list[Path]] = {}
    top_level_shards = _shard_payload_paths(run_dir)
    if top_level_shards:
        profile = run_dir.name
        rows = _load_rows_from_shards(profile=profile, shards=top_level_shards)
        if rows:
            profile_rows[profile] = rows
            profile_shards[profile] = top_level_shards
    for profile_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        shards = _shard_payload_paths(profile_dir)
        rows = _load_rows_from_shards(profile=profile_dir.name, shards=shards)
        if rows:
            profile_rows[profile_dir.name] = rows
            profile_shards[profile_dir.name] = shards

    profiles = {
        profile: _summarize_profile(
            profile=profile,
            rows=rows,
            shards=profile_shards.get(profile, []),
            run_dir=run_dir / profile,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for profile, rows in sorted(profile_rows.items())
    }
    cross_profile = _cross_profile_agent_comparisons(
        profile_rows=profile_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    pollution = _pollution_summary(profiles)
    payload = {
        "eval_id": eval_id or run_dir.name,
        "eval_kind": "hle_ablation_result_analysis",
        "run_dir": str(run_dir),
        "profile_count": len(profiles),
        "profiles": profiles,
        "cross_profile_agent_comparisons": cross_profile,
        "pollution_summary": pollution,
        "performance_validation": _performance_validation(profiles, cross_profile, pollution),
        "raw_content_persisted": any(
            profile.get("raw_content_persisted") is True for profile in profiles.values()
        ),
    }
    payload["pass"] = bool(payload["performance_validation"]["overall_pass"])
    payload["failed_gates"] = [
        name
        for name, gate in payload["performance_validation"]["gates"].items()
        if gate is not True
    ]
    return payload


def _shard_payload_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("*_shard_*.json")
        if not path.name.endswith(".problem_hashes.json")
    )


def _load_rows_from_shards(*, profile: str, shards: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in shards:
        payload = _load_json(shard)
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched["_profile"] = profile
            enriched["_shard"] = shard.name
            rows.append(enriched)
    return rows


def _summarize_profile(
    *,
    profile: str,
    rows: list[dict[str, Any]],
    shards: list[Path],
    run_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant"))].append(row)
    variant_summary = {
        variant: _variant_summary(items)
        for variant, items in sorted(by_variant.items())
    }
    paired = {
        "agent_vs_raw": _paired_variant_delta(
            rows=rows,
            left=AGENT_VARIANT,
            right=RAW_VARIANT,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "agent_vs_hipporag": _paired_variant_delta(
            rows=rows,
            left=AGENT_VARIANT,
            right=HIPPORAG_VARIANT,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }
    contamination = _profile_contamination(profile=profile, rows=rows, run_dir=run_dir)
    raw_preserve = _raw_preserve_summary(rows)
    return {
        "shard_count": len(shards),
        "row_count": len(rows),
        "problem_count": len({row.get("problem_id_hash") for row in rows if row.get("problem_id_hash")}),
        "variant_summary": variant_summary,
        "paired_control_comparison": paired,
        "module_credit": _module_credit(rows),
        "raw_preserve_selector": raw_preserve,
        "contamination": contamination,
        "clean_for_module_ablation": contamination["contaminated"] is False,
        "raw_content_persisted": any(
            row.get("raw_question_persisted")
            or row.get("gold_answer_persisted")
            or row.get("prediction_text_persisted")
            for row in rows
        ),
    }


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct") is True)
    errors = [row for row in rows if row.get("error")]
    answer_types = Counter(str(row.get("answer_type")) for row in rows)
    categories = Counter(str(row.get("category")) for row in rows if row.get("category"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "error_count": len(errors),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "category_counts": dict(sorted(categories.items())),
    }


def _paired_variant_delta(
    *,
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        problem = row.get("problem_id_hash")
        variant = row.get("variant")
        if problem and variant:
            by_variant[str(variant)][str(problem)] = row
    shared = sorted(set(by_variant.get(left, {})) & set(by_variant.get(right, {})))
    diffs = [
        _score_bool(by_variant[left][problem].get("correct"))
        - _score_bool(by_variant[right][problem].get("correct"))
        for problem in shared
    ]
    wins = sum(1 for value in diffs if value > 0)
    losses = sum(1 for value in diffs if value < 0)
    ties = sum(1 for value in diffs if value == 0)
    left_correct = sum(_score_bool(by_variant[left][problem].get("correct")) for problem in shared)
    right_correct = sum(_score_bool(by_variant[right][problem].get("correct")) for problem in shared)
    return {
        "left": left,
        "right": right,
        "shared_n": len(shared),
        "left_accuracy": left_correct / len(shared) if shared else None,
        "right_accuracy": right_correct / len(shared) if shared else None,
        "delta": (sum(diffs) / len(diffs)) if diffs else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_two_sided_p": _sign_test_two_sided_p(wins=wins, losses=losses),
        "bootstrap_ci95": _bootstrap_ci(diffs, samples=bootstrap_samples, seed=seed),
    }


def _cross_profile_agent_comparisons(
    *,
    profile_rows: dict[str, list[dict[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    full = profile_rows.get("full")
    if not full:
        return {}
    out: dict[str, Any] = {}
    full_agent = _rows_by_problem(full, AGENT_VARIANT)
    for profile, rows in sorted(profile_rows.items()):
        if profile == "full":
            continue
        other_agent = _rows_by_problem(rows, AGENT_VARIANT)
        shared = sorted(set(full_agent) & set(other_agent))
        diffs = [
            _score_bool(full_agent[problem].get("correct")) - _score_bool(other_agent[problem].get("correct"))
            for problem in shared
        ]
        wins = sum(1 for value in diffs if value > 0)
        losses = sum(1 for value in diffs if value < 0)
        out[f"full_vs_{profile}"] = {
            "left": "full",
            "right": profile,
            "variant": AGENT_VARIANT,
            "shared_n": len(shared),
            "delta": (sum(diffs) / len(diffs)) if diffs else None,
            "wins": wins,
            "losses": losses,
            "ties": sum(1 for value in diffs if value == 0),
            "sign_test_two_sided_p": _sign_test_two_sided_p(wins=wins, losses=losses),
            "bootstrap_ci95": _bootstrap_ci(diffs, samples=bootstrap_samples, seed=seed + len(out) + 11),
        }
    return out


def _rows_by_problem(rows: list[dict[str, Any]], variant: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["problem_id_hash"]): row
        for row in rows
        if row.get("variant") == variant and row.get("problem_id_hash")
    }


def _profile_contamination(*, profile: str, rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    event_counts = Counter()
    recursive_child_start_shards: list[str] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        local_counts = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(event.get("event") or "")
            if name:
                event_counts[name] += 1
                local_counts[name] += 1
        if local_counts.get("recursive_child_start"):
            recursive_child_start_shards.append(path.name)

    disabled_module_activations: dict[str, int] = {}
    if profile in {"no_recursive", "no_recursive_runner"}:
        disabled_module_activations["recursive_child_start_events"] = event_counts.get("recursive_child_start", 0)
        disabled_module_activations["recursive_child_validation_activated_rows"] = _module_status_count(
            rows, module="recursive_child_validation", status="activated"
        )
        disabled_module_activations["multi_candidate_self_verifier_activated_rows"] = _module_status_count(
            rows, module="multi_candidate_self_verifier", status="activated"
        )
    if profile in {"no_world_model", "no_world_model_router"}:
        disabled_module_activations["world_model_router_activated_rows"] = _module_status_count(
            rows, module="world_model_router", status="activated"
        )
    if profile == "no_morphism":
        disabled_module_activations["structural_morphism_transfer_activated_rows"] = _module_status_count(
            rows, module="structural_morphism_transfer", status="activated"
        )
    raw_content_persisted = any(
        row.get("raw_question_persisted")
        or row.get("gold_answer_persisted")
        or row.get("prediction_text_persisted")
        for row in rows
    )
    contaminated = raw_content_persisted or any(value > 0 for value in disabled_module_activations.values())
    return {
        "contaminated": bool(contaminated),
        "raw_content_persisted": bool(raw_content_persisted),
        "disabled_module_activations": disabled_module_activations,
        "recursive_child_start_shards": recursive_child_start_shards,
        "event_counts": dict(sorted(event_counts.items())),
    }


def _module_status_count(rows: list[dict[str, Any]], *, module: str, status: str) -> int:
    count = 0
    for row in rows:
        for item in row.get("module_trace") or []:
            if item.get("module") == module and item.get("status") == status:
                count += 1
    return count


def _raw_preserve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_rows = [row for row in rows if row.get("variant") == AGENT_VARIANT]
    keys = [
        "raw_preserve_selector_activated",
        "raw_preserve_candidate_emitted",
        "raw_preserve_candidate_selected",
    ]
    counts = {key: 0 for key in keys}
    correct_counts = {key: 0 for key in keys}
    for row in agent_rows:
        flags = ((row.get("component_efficacy") or {}).get("flags") or {})
        for key in keys:
            if flags.get(key):
                counts[key] += 1
                if row.get("correct") is True:
                    correct_counts[key] += 1
    return {
        "agent_n": len(agent_rows),
        "flag_counts": counts,
        "flag_accuracy": {
            key: (correct_counts[key] / counts[key] if counts[key] else None)
            for key in keys
        },
    }


def _module_credit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_rows = [row for row in rows if row.get("variant") == AGENT_VARIANT]
    flags: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    modules: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in agent_rows:
        correct = 1 if row.get("correct") is True else 0
        for key, value in (((row.get("component_efficacy") or {}).get("flags") or {}).items()):
            if value is True:
                flags[str(key)]["n"] += 1
                flags[str(key)]["correct"] += correct
        for item in row.get("module_trace") or []:
            if item.get("status") == "activated":
                modules[str(item.get("module"))]["n"] += 1
                modules[str(item.get("module"))]["correct"] += correct
    return {
        "activated_flag_accuracy": _count_accuracy(flags),
        "activated_module_accuracy": _count_accuracy(modules),
    }


def _count_accuracy(items: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, value in sorted(items.items()):
        n = int(value["n"])
        out[key] = {
            "n": n,
            "correct": int(value["correct"]),
            "accuracy": (value["correct"] / n if n else None),
        }
    return out


def _pollution_summary(profiles: dict[str, Any]) -> dict[str, Any]:
    contaminated_profiles = [
        profile
        for profile, summary in sorted(profiles.items())
        if (summary.get("contamination") or {}).get("contaminated") is True
    ]
    return {
        "contaminated_profiles": contaminated_profiles,
        "clean_profile_count": len(profiles) - len(contaminated_profiles),
        "profile_count": len(profiles),
    }


def _performance_validation(
    profiles: dict[str, Any],
    cross_profile: dict[str, Any],
    pollution: dict[str, Any],
) -> dict[str, Any]:
    reference_name = "full" if "full" in profiles else (next(iter(sorted(profiles))) if profiles else "")
    reference = profiles.get(reference_name) or {}
    reference_pair = ((reference.get("paired_control_comparison") or {}).get("agent_vs_raw") or {})
    gates = {
        "reference_profile_present": bool(reference),
        "reference_agent_raw_shared_n_positive": int(reference_pair.get("shared_n") or 0) > 0,
        "reference_agent_not_below_raw": (
            reference_pair.get("delta") is not None and float(reference_pair.get("delta")) >= 0.0
        ),
        "raw_content_not_persisted": not any(
            profile.get("raw_content_persisted") is True for profile in profiles.values()
        ),
        "pollution_report_present": bool(pollution),
        "cross_profile_comparisons_present": bool(cross_profile) if len(profiles) > 1 else True,
    }
    return {
        "overall_pass": all(value is True for value in gates.values()),
        "gates": gates,
        "reference_profile": reference_name,
        "note": (
            "Contaminated ablations are reported but excluded from clean module claims; performance pass checks "
            "the reference same-batch agent-vs-raw gate."
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


def _score_bool(value: Any) -> int:
    return 1 if value is True else 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def format_analysis_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# HLE Ablation Result Analysis",
        "",
        f"- eval id: `{payload['eval_id']}`",
        f"- profile count: `{payload['profile_count']}`",
        f"- pass: `{payload['pass']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        f"- raw content persisted: `{payload['raw_content_persisted']}`",
        "",
        "## Profile Accuracy",
        "",
        "| profile | clean | variant | n | accuracy | errors |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for profile, summary in sorted(payload.get("profiles", {}).items()):
        clean = summary.get("clean_for_module_ablation")
        for variant, row in sorted((summary.get("variant_summary") or {}).items()):
            lines.append(
                f"| `{profile}` | `{clean}` | `{variant}` | `{row.get('n')}` | "
                f"`{row.get('accuracy')}` | `{row.get('error_count')}` |"
            )
    lines.extend([
        "",
        "## Same-Profile Paired Deltas",
        "",
        "| profile | pair | n | delta | wins | losses | p | ci95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for profile, summary in sorted(payload.get("profiles", {}).items()):
        for pair, row in sorted((summary.get("paired_control_comparison") or {}).items()):
            ci = row.get("bootstrap_ci95") or {}
            lines.append(
                f"| `{profile}` | `{pair}` | `{row.get('shared_n')}` | `{row.get('delta')}` | "
                f"`{row.get('wins')}` | `{row.get('losses')}` | `{row.get('sign_test_two_sided_p')}` | "
                f"`[{ci.get('lower')}, {ci.get('upper')}]` |"
            )
    lines.extend([
        "",
        "## Pollution",
        "",
        f"- contaminated profiles: `{payload.get('pollution_summary', {}).get('contaminated_profiles')}`",
        "",
        "| profile | contaminated | disabled-module activations | raw content persisted |",
        "| --- | ---: | --- | ---: |",
    ])
    for profile, summary in sorted(payload.get("profiles", {}).items()):
        contamination = summary.get("contamination") or {}
        lines.append(
            f"| `{profile}` | `{contamination.get('contaminated')}` | "
            f"`{contamination.get('disabled_module_activations')}` | "
            f"`{contamination.get('raw_content_persisted')}` |"
        )
    lines.append("")
    lines.append("The analysis uses hashes and correctness booleans only; raw HLE content is not persisted.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HLE module ablation shard outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-id", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    payload = analyze_hle_ablation_run(
        run_dir=Path(args.run_dir),
        eval_id=args.eval_id,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    out = Path(args.out) if args.out else Path(args.run_dir) / f"{payload['eval_id']}_analysis.json"
    md_out = Path(args.md_out) if args.md_out else Path("reconstruction/md") / f"{payload['eval_id']}_analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(format_analysis_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
        "md_out": str(md_out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
