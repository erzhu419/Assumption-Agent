"""First-party live trace scale audit for the trace outcome world model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import stable_id


DEFAULT_FORENSIC_DIR = Path("phase four/assumption_graph/structural_live_ablation_20260603")
DEFAULT_PERFORMANCE_PAYLOAD = Path("phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json")


def build_first_party_world_model_scale_payload(
    *,
    eval_id: str,
    forensic_dir: Path | None = None,
    trace_outcome_section: dict | None = None,
    precomputed_payload: dict | None = None,
) -> dict:
    """Audit raw first-party trace scale without storing prompts or answers."""

    if precomputed_payload:
        return {**precomputed_payload, "eval_id": eval_id}
    forensic_dir = forensic_dir or DEFAULT_FORENSIC_DIR
    live = _collect_live_forensic_rows(forensic_dir)
    trace_outcome_section = trace_outcome_section or {}
    best_brier = trace_outcome_section.get("best_brier_score")
    weighted_brier = (
        (trace_outcome_section.get("feature_leave_one_out_metrics") or {}).get("weighted_brier_score")
        or (trace_outcome_section.get("trajectory_quality_metrics") or {}).get("weighted_brier_score")
    )
    gates = [
        {
            "gate": "raw_first_party_live_trainable_scale",
            "pass": live["raw_first_party_trainable_row_count"] >= 1000,
            "observed": live["raw_first_party_trainable_row_count"],
        },
        {
            "gate": "judge_event_scale",
            "pass": live["valid_judge_event_count"] >= 1000,
            "observed": live["valid_judge_event_count"],
        },
        {
            "gate": "source_run_diversity",
            "pass": live["source_run_count"] >= 10 and live["distinct_problem_count"] >= 50,
            "observed": {
                "source_run_count": live["source_run_count"],
                "distinct_problem_count": live["distinct_problem_count"],
            },
        },
        {
            "gate": "calibrated_trace_outcome_model",
            "pass": best_brier is not None and float(best_brier) <= 0.12,
            "observed": {
                "best_brier_score": best_brier,
                "weighted_brier_score": weighted_brier,
                "trace_outcome_trainable_row_count": trace_outcome_section.get("trainable_row_count"),
            },
        },
        {
            "gate": "no_prompt_answer_or_secret_payload",
            "pass": not live["secret_leak_detected"] and live["prompt_answer_payload_stored"] is False,
            "observed": {
                "secret_leak_detected": live["secret_leak_detected"],
                "prompt_answer_payload_stored": live["prompt_answer_payload_stored"],
            },
        },
    ]
    pass_condition = all(gate["pass"] for gate in gates)
    return {
        "eval_id": eval_id,
        "eval_kind": "first_party_world_model_raw_live_trace_scale",
        "pass": pass_condition,
        "raw_first_party_trainable_row_count": live["raw_first_party_trainable_row_count"],
        "raw_first_party_live_event_count": live["raw_first_party_live_event_count"],
        "valid_judge_event_count": live["valid_judge_event_count"],
        "solver_event_count": live["solver_event_count"],
        "source_run_count": live["source_run_count"],
        "distinct_problem_count": live["distinct_problem_count"],
        "trace_source_counts": live["trace_source_counts"],
        "arm_counts": live["arm_counts"],
        "pair_counts": live["pair_counts"],
        "domain_counts": live["domain_counts"],
        "calibration": {
            "best_brier_score": best_brier,
            "weighted_brier_score": weighted_brier,
            "trace_outcome_trainable_row_count": trace_outcome_section.get("trainable_row_count"),
            "trace_outcome_weighted_trainable_row_count": trace_outcome_section.get("weighted_trainable_row_count"),
            "trace_outcome_trace_source_counts": trace_outcome_section.get("trace_source_counts"),
        },
        "live_outcome_diagnostic": live["live_outcome_diagnostic"],
        "gates": gates,
        "run_summaries": live["run_summaries"],
        "row_samples": live["row_samples"],
        "prompt_answer_payload_stored": live["prompt_answer_payload_stored"],
        "secret_leak_detected": live["secret_leak_detected"],
        "scope_note": (
            "Rows come from raw first-party structural live ablation forensic logs. "
            "The committed artifact stores only compact metadata, labels, and counts; prompts and answers are excluded."
        ),
    }


def _collect_live_forensic_rows(forensic_dir: Path) -> dict:
    trainable_rows: list[dict] = []
    run_stats: dict[str, Counter] = defaultdict(Counter)
    problem_ids: set[str] = set()
    raw_event_count = 0
    solver_event_count = 0
    judge_event_count = 0
    invalid_judge_event_count = 0
    arm_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    for path in sorted(forensic_dir.glob("*forensic.jsonl")) if forensic_dir.exists() else []:
        run_id = path.name.replace("_forensic.jsonl", "")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_event_count += 1
            role = str(event.get("role") or "unknown")
            problem_id = str(event.get("problem_id") or "")
            if problem_id:
                problem_ids.add(problem_id)
            run_stats[run_id][f"role={role}"] += 1
            if role == "solver":
                solver_event_count += 1
                continue
            if role != "judge":
                continue
            judge_event_count += 1
            rows = _rows_from_judge_event(event, run_id=run_id, line_no=line_no)
            if not rows:
                invalid_judge_event_count += 1
                continue
            for row in rows:
                trainable_rows.append(row)
                arm_counts[row["arm"]] += 1
                pair_counts[row["pair"]] += 1
                domain_counts[row["domain"]] += 1
                run_stats[run_id]["trainable_rows"] += 1
                run_stats[run_id][f"pair={row['pair']}"] += 1
                run_stats[run_id][f"arm={row['arm']}"] += 1

    diagnostic = _live_outcome_diagnostic(trainable_rows)
    return {
        "raw_first_party_live_event_count": raw_event_count,
        "raw_first_party_trainable_row_count": len(trainable_rows),
        "solver_event_count": solver_event_count,
        "valid_judge_event_count": len(trainable_rows) // 2,
        "judge_event_count": judge_event_count,
        "invalid_judge_event_count": invalid_judge_event_count,
        "source_run_count": len(run_stats),
        "distinct_problem_count": len(problem_ids),
        "trace_source_counts": {"first_party_live_forensic": len(trainable_rows)} if trainable_rows else {},
        "arm_counts": dict(sorted(arm_counts.items())),
        "pair_counts": dict(sorted(pair_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "run_summaries": _run_summaries(run_stats),
        "row_samples": trainable_rows[:12],
        "live_outcome_diagnostic": diagnostic,
        "prompt_answer_payload_stored": False,
        "secret_leak_detected": False,
    }


def _rows_from_judge_event(event: dict, *, run_id: str, line_no: int) -> list[dict]:
    if event.get("error"):
        return []
    judgment = event.get("judgment") or {}
    problem_id = str(event.get("problem_id") or "")
    pair = str(judgment.get("pair") or event.get("pair") or "")
    winner = str(judgment.get("winner") or "")
    arms = [str(judgment.get("a_arm") or ""), str(judgment.get("b_arm") or "")]
    if not problem_id or not pair or not winner or any(not arm for arm in arms):
        return []
    domain = problem_id.rsplit("_", 1)[0] if "_" in problem_id else problem_id
    rows = []
    for arm in arms:
        label = 1.0 if arm == winner else 0.0
        rows.append({
            "row_id": stable_id("live_trace_row", run_id, line_no, problem_id, pair, arm),
            "source_run_id": run_id,
            "problem_id": problem_id,
            "domain": domain,
            "pair": pair,
            "arm": arm,
            "label": label,
            "outcome": "win" if label == 1.0 else "loss",
            "trace_source": "first_party_live_forensic",
            "model_alias": judgment.get("model_alias") or event.get("model_alias"),
            "judge_model": judgment.get("model") or event.get("model"),
        })
    return rows


def _live_outcome_diagnostic(rows: list[dict]) -> dict:
    if not rows:
        return {"prediction_count": 0, "brier_score": None, "accuracy_at_half": None}
    levels = [
        ("source_run_pair_arm", lambda row: f"run={row['source_run_id']}|pair={row['pair']}|arm={row['arm']}"),
        ("domain_pair_arm", lambda row: f"domain={row['domain']}|pair={row['pair']}|arm={row['arm']}"),
        ("pair_arm", lambda row: f"pair={row['pair']}|arm={row['arm']}"),
        ("arm", lambda row: f"arm={row['arm']}"),
        ("global", lambda row: "global"),
    ]
    stats = []
    for name, fn in levels:
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            buckets[fn(row)].append(float(row["label"]))
        stats.append((name, fn, buckets))
    predictions = []
    level_counts: Counter[str] = Counter()
    for row in rows:
        label = float(row["label"])
        for name, fn, buckets in stats:
            key = fn(row)
            bucket = buckets[key]
            support_n = len(bucket) - 1
            support_wins = sum(bucket) - label
            if name == "global" or support_n >= 20:
                probability = (support_wins + 1.0) / (support_n + 2.0) if support_n >= 0 else 0.5
                predictions.append((probability, label))
                level_counts[name] += 1
                break
    brier = sum((p - y) ** 2 for p, y in predictions) / len(predictions)
    accuracy = sum((p >= 0.5) == bool(y) for p, y in predictions) / len(predictions)
    return {
        "prediction_count": len(predictions),
        "brier_score": round(brier, 4),
        "accuracy_at_half": round(accuracy, 4),
        "prediction_level_counts": dict(level_counts),
        "diagnostic_note": "Live pairwise labels are used as an independent raw-scale diagnostic; paper gate uses the calibrated trace_outcome_model Brier.",
    }


def _run_summaries(run_stats: dict[str, Counter]) -> list[dict]:
    rows = []
    for run_id, counter in sorted(run_stats.items()):
        rows.append({
            "source_run_id": run_id,
            "raw_event_count": sum(value for key, value in counter.items() if key.startswith("role=")),
            "solver_event_count": counter.get("role=solver", 0),
            "judge_event_count": counter.get("role=judge", 0),
            "trainable_row_count": counter.get("trainable_rows", 0),
            "pair_counts": {
                key.split("=", 1)[1]: value
                for key, value in sorted(counter.items())
                if key.startswith("pair=")
            },
        })
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--forensic-dir", default=str(DEFAULT_FORENSIC_DIR))
    ap.add_argument("--performance-payload", default=str(DEFAULT_PERFORMANCE_PAYLOAD))
    ap.add_argument("--eval-id", default="first_party_world_model_scale_20260604")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    perf_path = _resolve(root, args.performance_payload)
    perf = _load_json(perf_path) if perf_path.exists() else {}
    sections = perf.get("sections", perf)
    payload = build_first_party_world_model_scale_payload(
        eval_id=args.eval_id,
        forensic_dir=_resolve(root, args.forensic_dir),
        trace_outcome_section=sections.get("trace_outcome_model", {}),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        out = _resolve(root, args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
