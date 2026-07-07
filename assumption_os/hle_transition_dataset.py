"""Build state-action-outcome rows from HLE run artifacts.

This is the first data layer needed for continual learning: existing HLE runs
become transition records that can be distilled into fast policies later.  The
builder keeps question/option text out of the persisted row and stores hashes
plus after-run outcomes instead.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .autonomy_journal import stable_hash


TRANSITION_DATASET_VERSION = "hle_transition_dataset_v1"

FAILURE_FLAG_PRIORITY = (
    ("candidate_generation_missed_gold", "candidate_generation_missed_gold"),
    (
        "candidate_generation_missed_gold_with_sweep_coverage",
        "candidate_generation_missed_gold_with_sweep_coverage",
    ),
    (
        "source_grounded_verifier_blocked_no_fallback",
        "source_grounded_verifier_blocked_no_fallback",
    ),
    ("verified_or_abstain_no_fallback", "verified_or_abstain no_fallback"),
    ("gold_option_source_verifier_unaccepted", "gold_option_source_verifier_unaccepted"),
    (
        "gold_option_source_verifier_direct_source_insufficient",
        "gold_option_direct_source_insufficient",
    ),
    (
        "gold_option_source_verifier_indirect_or_generic",
        "gold_option_source_indirect_or_generic",
    ),
    ("gold_option_source_verifier_zero_quality", "gold_option_source_zero_quality"),
    ("source_verifier_cross_selection_blocked", "source_verifier_cross_selection_blocked"),
    (
        "mc_option_claim_span_directness_lexical_unique_but_generic",
        "span_directness_lexical_unique_but_generic",
    ),
    (
        "mc_option_claim_candidate_direct_relation_span_directness_rejected",
        "candidate_direct_relation_span_directness_rejected",
    ),
    ("verified_or_abstain_abstained", "verified_or_abstain abstained"),
)


@dataclass(frozen=True)
class HleTransitionRecord:
    question_id: str
    question_hash: str | None
    domain: str
    category: str
    action: str
    selected_label_hash: str | None
    gold_after_run_label_hash: str | None
    correct: bool | None
    failure_bucket: str
    cost: float | None
    latency_seconds: float | None
    path_hashes: dict[str, Any] = field(default_factory=dict)
    option_feature_hashes: dict[str, str] = field(default_factory=dict)
    fast_policy_ids: list[str] = field(default_factory=list)
    raw_content_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_version"] = TRANSITION_DATASET_VERSION
        return payload


def transition_record_from_hle_row(row: dict[str, Any]) -> HleTransitionRecord:
    """Normalize one HLE result row into a transition record."""

    question_text = _first_str(row, "question", "question_text", "prompt", "stem")
    selected = _first_str(row, "selected_label", "predicted_label", "answer", "final_label")
    gold = _first_str(row, "gold_label", "gold", "_answer", "correct_label")
    correct = _coerce_bool(row.get("correct", row.get("is_correct")))
    failure_bucket = _failure_bucket(row)
    route = row.get("route") if isinstance(row.get("route"), dict) else {}
    fast_policy = _fast_policy(row, route)
    option_feature_hashes = _option_feature_hashes(row)
    record = HleTransitionRecord(
        question_id=_question_id(row),
        question_hash=stable_hash({"question": question_text}) if question_text else _first_str(row, "question_hash"),
        domain=_first_str(row, "domain", "raw_subject"),
        category=_first_str(row, "category", "subject"),
        action=_action(row, route),
        selected_label_hash=(
            stable_hash({"option_label": selected})
            if selected
            else _first_str(row, "selected_label_hash", "predicted_label_hash", "prediction_hash") or None
        ),
        gold_after_run_label_hash=(
            stable_hash({"option_label": gold})
            if gold
            else _first_str(row, "gold_label_hash", "correct_label_hash", "answer_hash") or None
        ),
        correct=correct,
        failure_bucket=failure_bucket,
        cost=_cost(row),
        latency_seconds=_latency_seconds(row),
        path_hashes=_path_hashes(row, route),
        option_feature_hashes=option_feature_hashes,
        fast_policy_ids=list(fast_policy.get("selected_policy_ids") or []),
        raw_content_persisted=False,
    )
    return record


def build_transition_dataset(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = [transition_record_from_hle_row(row).to_dict() for row in rows]
    return {
        "dataset_version": TRANSITION_DATASET_VERSION,
        "records": records,
        "summary": summarize_transition_records(records),
        "raw_content_persisted": False,
    }


def load_hle_result_rows_from_path(path: str | Path) -> list[dict[str, Any]]:
    """Load rows from an HLE result JSON/JSONL artifact.

    Aggregate artifacts with ``shards`` are expanded recursively.  Source file
    names are kept only as hashes in the final transition records.
    """

    artifact_path = Path(path).expanduser()
    payload = _load_artifact_payload(artifact_path)
    return _rows_from_payload(payload, artifact_path, artifact_path.parent)


def build_transition_dataset_from_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_artifact_hashes: list[str] = []
    for path in paths:
        artifact_path = Path(path).expanduser()
        rows.extend(load_hle_result_rows_from_path(artifact_path))
        source_artifact_hashes.append(stable_hash({"artifact": str(artifact_path)}))
    dataset = build_transition_dataset(rows)
    dataset["source_artifact_hashes"] = source_artifact_hashes
    return dataset


def summarize_transition_records(records: Iterable[dict[str, Any] | HleTransitionRecord]) -> dict[str, Any]:
    rows = [record.to_dict() if isinstance(record, HleTransitionRecord) else dict(record) for record in records]
    total = len(rows)
    correct_count = sum(1 for row in rows if row.get("correct") is True)
    known_correct = sum(1 for row in rows if row.get("correct") is not None)
    failure_buckets = Counter(str(row.get("failure_bucket") or "none") for row in rows)
    action_counts = Counter(str(row.get("action") or "unknown") for row in rows)
    verified_gate_status_counts = Counter(
        str((row.get("path_hashes") or {}).get("verified_or_abstain_gate_status") or "unknown")
        for row in rows
    )
    no_fallback_count = sum(
        1
        for row in rows
        if "no_fallback" in str(row.get("failure_bucket") or "")
        or (row.get("path_hashes") or {}).get("verified_or_abstain_gate_status") == "no_fallback"
    )
    latency_values = [
        float(row["latency_seconds"])
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    cost_values = [
        float(row["cost"])
        for row in rows
        if isinstance(row.get("cost"), (int, float))
    ]
    return {
        "record_count": total,
        "known_correct_count": known_correct,
        "correct_count": correct_count,
        "accuracy": round(correct_count / known_correct, 4) if known_correct else None,
        "failure_buckets": dict(failure_buckets),
        "action_counts": dict(action_counts),
        "verified_or_abstain_gate_status_counts": dict(verified_gate_status_counts),
        "no_fallback_count": no_fallback_count,
        "latency_sum_seconds": round(sum(latency_values), 4) if latency_values else 0.0,
        "cost_sum": round(sum(cost_values), 4) if cost_values else 0.0,
        "raw_content_persisted": False,
    }


def _question_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "problem_id", "problem_id_hash", "id", "seed"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return stable_hash({
        "question_hash": row.get("question_hash"),
        "selected_label": row.get("selected_label"),
        "gold_label": row.get("gold_label"),
    })[:16]


def _action(row: dict[str, Any], route: dict[str, Any]) -> str:
    selection = _selection(row)
    return _first_str(
        route,
        "selection_method",
        "action",
    ) or _first_str(
        selection,
        "selection_method",
        "action",
    ) or _first_str(
        row,
        "selection_method",
        "action",
        "variant",
        "agent_path",
    ) or "unknown"


def _failure_bucket(row: dict[str, Any]) -> str:
    bucket = row.get("failure_bucket")
    if bucket:
        return str(bucket)
    buckets = row.get("failure_buckets")
    if isinstance(buckets, dict) and buckets:
        return str(max(buckets.items(), key=lambda item: item[1])[0])
    if row.get("correct") is True or row.get("is_correct") is True:
        return "none"
    flags = _flags(row)
    for flag, label in FAILURE_FLAG_PRIORITY:
        if flags.get(flag) is True:
            return label
    selection = _selection(row)
    verified_gate = selection.get("verified_or_abstain_gate")
    if isinstance(verified_gate, dict):
        status = _first_str(verified_gate, "status")
        reason = _first_str(verified_gate, "reason")
        if status == "no_fallback":
            return "verified_or_abstain no_fallback"
        if status == "abstained" and reason:
            return f"verified_or_abstain abstained:{reason}"
        if status:
            return f"verified_or_abstain {status}"
    taxonomy = _component_efficacy(row).get("operator_failure_taxonomy")
    if isinstance(taxonomy, dict):
        category = _first_str(taxonomy, "category")
        if category and category != "NotOperatorFailure":
            return category
    error = _first_str(row, "error")
    if error:
        return "error"
    reason = _first_str(row, "reason", "abstain_reason", "selection_reason")
    return reason or "unknown_failure"


def _fast_policy(row: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    for container in (route, row):
        value = container.get("fast_policy_memory")
        if isinstance(value, dict):
            return value
        value = container.get("fast_policy_decision")
        if isinstance(value, dict):
            return value
    return {}


def _path_hashes(row: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "router_payload_hash",
        "option_matrix_hash",
        "source_lane_hash",
        "solver_hash",
        "fast_policy_payload_hash",
    )
    out: dict[str, Any] = {}
    for source in (route, row):
        for key in keys:
            if source.get(key):
                out[key] = source[key]
    call_metadata = row.get("call_metadata") if isinstance(row.get("call_metadata"), dict) else {}
    for key in ("agent_plan_hash", "call_id"):
        if call_metadata.get(key):
            out[key] = call_metadata[key]
    watchdog = call_metadata.get("variant_watchdog") if isinstance(call_metadata.get("variant_watchdog"), dict) else {}
    for key in (
        "model_call_count",
        "model_router_attempt_count",
        "model_call_budget",
        "model_router_attempt_budget",
        "status",
    ):
        if watchdog.get(key) is not None:
            out[f"variant_watchdog_{key}"] = watchdog[key]
    verified_gate = _selection(row).get("verified_or_abstain_gate")
    if isinstance(verified_gate, dict):
        status = _first_str(verified_gate, "status")
        reason = _first_str(verified_gate, "reason")
        if status:
            out["verified_or_abstain_gate_status"] = status
        if reason:
            out["verified_or_abstain_gate_reason_hash"] = stable_hash({"reason": reason})
    fast_policy = _fast_policy(row, route)
    if fast_policy.get("fast_policy_payload_hash"):
        out["fast_policy_payload_hash"] = fast_policy["fast_policy_payload_hash"]
    for key in (
        "_transition_source_artifact",
        "_transition_parent_artifact",
        "_transition_eval_id",
        "_transition_eval_kind",
    ):
        if row.get(key):
            out[f"{key.removeprefix('_transition_')}_hash"] = stable_hash({key: row[key]})
    if row.get("_transition_shard_index") is not None:
        out["shard_index"] = row["_transition_shard_index"]
    return out


def _option_feature_hashes(row: dict[str, Any]) -> dict[str, str]:
    option_rows = row.get("option_rows")
    if not isinstance(option_rows, list):
        matrix = row.get("option_matrix")
        option_rows = matrix.get("option_rows") if isinstance(matrix, dict) else []
    out: dict[str, str] = {}
    for option in option_rows if isinstance(option_rows, list) else []:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label") or "").strip()
        if label:
            out[label] = str(option.get("option_hash") or stable_hash(option))
    for index, option_hash in enumerate(_collect_option_hashes(row)):
        out.setdefault(f"option_hash_{index}", option_hash)
    return out


def _collect_option_hashes(value: Any) -> list[str]:
    hashes: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "option_hash" and isinstance(child, str) and child:
                    hashes.add(child)
                elif key.endswith("_by_option_hash") and isinstance(child, dict):
                    for option_hash in child:
                        if isinstance(option_hash, str) and option_hash:
                            hashes.add(option_hash)
                elif key in {"sweep_only_option_hashes", "recovered_candidate_option_hashes", "replaced_candidate_option_hashes"}:
                    if isinstance(child, list):
                        for option_hash in child:
                            if isinstance(option_hash, str) and option_hash:
                                hashes.add(option_hash)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(hashes)


def _component_efficacy(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("component_efficacy")
    return value if isinstance(value, dict) else {}


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    value = _component_efficacy(row).get("flags")
    return value if isinstance(value, dict) else {}


def _selection(row: dict[str, Any]) -> dict[str, Any]:
    value = _component_efficacy(row).get("selection")
    if isinstance(value, dict):
        return value
    value = row.get("selection")
    return value if isinstance(value, dict) else {}


def _cost(row: dict[str, Any]) -> float | None:
    explicit = _first_float(row, "cost", "usd_cost", "model_cost", "unique_model_calls")
    if explicit is not None:
        return explicit
    call_metadata = row.get("call_metadata") if isinstance(row.get("call_metadata"), dict) else {}
    watchdog = call_metadata.get("variant_watchdog") if isinstance(call_metadata.get("variant_watchdog"), dict) else {}
    return _first_float(watchdog, "model_call_count", "model_router_attempt_count")


def _latency_seconds(row: dict[str, Any]) -> float | None:
    explicit = _first_float(row, "latency_seconds", "elapsed_seconds", "elapsed", "wall_seconds", "elapsed_sec")
    if explicit is not None:
        return explicit
    call_metadata = row.get("call_metadata") if isinstance(row.get("call_metadata"), dict) else {}
    call_latency = _first_float(call_metadata, "latency_sec", "elapsed_sec")
    if call_latency is not None:
        return call_latency
    watchdog = call_metadata.get("variant_watchdog") if isinstance(call_metadata.get("variant_watchdog"), dict) else {}
    return _first_float(watchdog, "elapsed_sec")


def _load_artifact_payload(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_payload(payload: Any, artifact_path: Path, base_dir: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_with_source_metadata(row, artifact_path) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("rows"), list):
        return [
            _with_source_metadata(row, artifact_path, payload)
            for row in payload["rows"]
            if isinstance(row, dict)
        ]
    if isinstance(payload.get("records"), list):
        return [
            _with_source_metadata(row, artifact_path, payload)
            for row in payload["records"]
            if isinstance(row, dict)
        ]
    if isinstance(payload.get("shards"), list):
        rows: list[dict[str, Any]] = []
        for shard in payload["shards"]:
            if not isinstance(shard, dict):
                continue
            if isinstance(shard.get("rows"), list):
                shard_rows = [
                    _with_source_metadata(row, artifact_path, payload, shard)
                    for row in shard["rows"]
                    if isinstance(row, dict)
                ]
            else:
                out_path = _resolve_shard_out_path(shard.get("out"), base_dir)
                if out_path is None or not out_path.exists():
                    continue
                shard_payload = _load_artifact_payload(out_path)
                shard_rows = _rows_from_payload(shard_payload, out_path, out_path.parent)
                shard_rows = [
                    _with_source_metadata(row, out_path, payload, shard, artifact_path)
                    for row in shard_rows
                ]
            rows.extend(shard_rows)
        return rows
    return []


def _with_source_metadata(
    row: dict[str, Any],
    artifact_path: Path,
    parent_payload: dict[str, Any] | None = None,
    shard: dict[str, Any] | None = None,
    parent_artifact_path: Path | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out["_transition_source_artifact"] = str(artifact_path)
    if parent_payload:
        out["_transition_parent_artifact"] = str(parent_artifact_path or artifact_path)
        if parent_payload.get("eval_id"):
            out["_transition_eval_id"] = parent_payload["eval_id"]
        if parent_payload.get("eval_kind"):
            out["_transition_eval_kind"] = parent_payload["eval_kind"]
    if shard:
        if shard.get("shard_index") is not None:
            out["_transition_shard_index"] = shard["shard_index"]
        if shard.get("status") is not None:
            out["_transition_shard_status"] = shard["status"]
        if shard.get("elapsed_sec") is not None and out.get("elapsed_sec") is None:
            out["_transition_shard_elapsed_sec"] = shard["elapsed_sec"]
    return out


def _resolve_shard_out_path(value: Any, base_dir: Path) -> Path | None:
    if not value or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _first_str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if value is not None and str(value).strip():
            try:
                return float(str(value))
            except ValueError:
                continue
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "correct"}:
        return True
    if normalized in {"false", "0", "no", "wrong", "incorrect"}:
        return False
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="HLE result JSON/JSONL artifacts to convert")
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    dataset = build_transition_dataset_from_paths(args.paths)
    text = json.dumps(
        dataset,
        ensure_ascii=True,
        indent=2 if args.pretty else None,
        sort_keys=args.pretty,
    )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
