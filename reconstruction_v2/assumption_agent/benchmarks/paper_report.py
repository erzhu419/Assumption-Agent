from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import stable_hash
from ..splits import SplitManifest
from .paper_protocol import PaperProtocol


@dataclass(frozen=True)
class PaperTrialRecord:
    item_id_hash: str
    family_hash: str
    split: str
    control_id: str
    protocol_hash: str
    manifest_hash: str
    evaluator_epoch: str
    pair_id: str
    repeat: int
    success: bool
    score: float
    valid: bool
    provider_fingerprint: str
    fairness_fingerprint: str
    total_tokens: int
    steps: int
    duration_seconds: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    attempt: int = 1
    error_type: str | None = None
    observation_hash: str = ""
    prebuilt_image_key: str = ""
    prebuilt_image_id: str = ""
    prebuilt_cache_reused: bool = False
    agent_runtime_key: str = ""
    agent_runtime_version: str = ""
    codex_agent_execution_policy_hash: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperTrialRecord":
        record = cls(
            item_id_hash=str(data.get("item_id_hash") or ""),
            family_hash=str(data.get("family_hash") or ""),
            split=str(data.get("split") or ""),
            control_id=str(data.get("control_id") or ""),
            protocol_hash=str(data.get("protocol_hash") or ""),
            manifest_hash=str(data.get("manifest_hash") or ""),
            evaluator_epoch=str(data.get("evaluator_epoch") or ""),
            pair_id=str(data.get("pair_id") or ""),
            repeat=int(data.get("repeat") or 0),
            success=bool(data.get("success")),
            score=float(data.get("score") or 0.0),
            valid=bool(data.get("valid")),
            provider_fingerprint=str(data.get("provider_fingerprint") or ""),
            fairness_fingerprint=str(data.get("fairness_fingerprint") or ""),
            total_tokens=max(0, int(data.get("total_tokens") or 0)),
            steps=max(0, int(data.get("steps") or 0)),
            duration_seconds=max(0.0, float(data.get("duration_seconds") or 0.0)),
            metrics={
                str(key): float(value)
                for key, value in dict(data.get("metrics") or {}).items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
            attempt=max(1, int(data.get("attempt") or 1)),
            error_type=str(data["error_type"]) if data.get("error_type") else None,
            observation_hash=str(data.get("observation_hash") or ""),
            prebuilt_image_key=str(data.get("prebuilt_image_key") or ""),
            prebuilt_image_id=str(data.get("prebuilt_image_id") or ""),
            prebuilt_cache_reused=bool(data.get("prebuilt_cache_reused")),
            agent_runtime_key=str(data.get("agent_runtime_key") or ""),
            agent_runtime_version=str(data.get("agent_runtime_version") or ""),
            codex_agent_execution_policy_hash=str(
                data.get("codex_agent_execution_policy_hash") or ""
            ),
        )
        issues = record.validate()
        if issues:
            raise ValueError(f"invalid paper trial record: {issues}")
        return record

    def validate(self) -> list[str]:
        issues: list[str] = []
        if len(self.item_id_hash) != 64:
            issues.append("item_id_hash_invalid")
        if len(self.family_hash) != 64:
            issues.append("family_hash_invalid")
        if self.split not in {"validation", "test"}:
            issues.append("split_invalid")
        if not self.control_id:
            issues.append("control_id_missing")
        if len(self.protocol_hash) != 64:
            issues.append("protocol_hash_invalid")
        if len(self.manifest_hash) != 64:
            issues.append("manifest_hash_invalid")
        if not self.evaluator_epoch:
            issues.append("evaluator_epoch_missing")
        if len(self.pair_id) != 20:
            issues.append("pair_id_invalid")
        if self.repeat <= 0:
            issues.append("repeat_invalid")
        if self.attempt <= 0:
            issues.append("attempt_invalid")
        if self.valid and self.error_type:
            issues.append("valid_record_has_error")
        if not 0.0 <= self.score <= 1.0:
            issues.append("score_out_of_range")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            issues.append("metric_not_finite")
        if self.prebuilt_image_key and len(self.prebuilt_image_key) != 64:
            issues.append("prebuilt_image_key_invalid")
        if self.prebuilt_image_id and not self.prebuilt_image_id.startswith("sha256:"):
            issues.append("prebuilt_image_id_invalid")
        if self.agent_runtime_key and len(self.agent_runtime_key) != 64:
            issues.append("agent_runtime_key_invalid")
        if (
            self.codex_agent_execution_policy_hash
            and len(self.codex_agent_execution_policy_hash) != 64
        ):
            issues.append("codex_agent_execution_policy_hash_invalid")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "secret_value_persisted": False,
            "raw_content_persisted": False,
        }


def build_paper_report(
    records: Sequence[PaperTrialRecord],
    *,
    protocol: PaperProtocol,
    protocol_lock: Mapping[str, Any],
    split: str = "test",
    manifest_hash: str | None = None,
    phase_name: str | None = None,
) -> dict[str, Any]:
    statistics_spec = protocol.payload["statistics"]
    confidence = float(statistics_spec["confidence"])
    bootstrap_samples = int(statistics_spec["bootstrap_samples"])
    bootstrap_seed = int(statistics_spec["bootstrap_seed"])
    expected_manifest_hash = manifest_hash or str(protocol_lock.get("primary_manifest_hash") or "")
    if phase_name is None:
        secondary = expected_manifest_hash == protocol_lock.get("secondary_manifest_hash")
        if split == "test":
            phase_name = "family_out_transfer" if secondary else "sealed_test"
        else:
            phase_name = "family_out_development" if secondary else "development"
    phase = protocol.payload["phases"][phase_name]
    expected_repeats = int(phase["repeats"])
    expected_items = int(phase.get("test_count") or phase.get("validation_count") or 0)
    controls = [str(row["id"]) for row in protocol.payload["controls"]]
    selected = [row for row in records if row.split == split]
    summaries = {
        control: _summarize_control(
            [row for row in selected if row.control_id == control],
            expected_repeats=expected_repeats,
            confidence=confidence,
        )
        for control in controls
    }
    baseline_id, primary_candidate_id = statistics_spec["primary_comparison"]
    comparisons: dict[str, dict[str, Any]] = {}
    for index, control in enumerate(controls):
        if control == baseline_id:
            continue
        comparisons[control] = _paired_comparison(
            selected,
            baseline_id=baseline_id,
            candidate_id=control,
            expected_repeats=expected_repeats,
            expected_items=expected_items,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
            confidence=confidence,
        )
    adjusted = _holm_adjust(
        {control: float(summary["mcnemar_p_value"]) for control, summary in comparisons.items()}
    )
    for control, value in adjusted.items():
        comparisons[control]["holm_adjusted_p_value"] = value
    blockers: list[str] = []
    if not protocol_lock.get("claim_eligible"):
        blockers.append("protocol_lock_not_claim_eligible")
    primary = comparisons.get(primary_candidate_id)
    if primary is None:
        blockers.append("primary_comparison_missing")
    elif not primary["claim_valid"]:
        blockers.extend(f"primary:{reason}" for reason in primary["invalid_reasons"])
    if len(selected) == 0:
        blockers.append("no_records_for_split")
    if any(row.protocol_hash != protocol.protocol_hash for row in selected):
        blockers.append("record_protocol_hash_mismatch")
    if str(protocol.payload.get("protocol_version") or "") == "3.3.0" and any(
        row.codex_agent_execution_policy_hash
        != protocol.codex_agent_execution_policy.policy_hash
        for row in selected
    ):
        blockers.append("record_codex_agent_execution_policy_hash_mismatch")
    if expected_manifest_hash and any(
        row.manifest_hash != expected_manifest_hash for row in selected
    ):
        blockers.append("record_manifest_hash_mismatch")
    evaluator_epochs = {row.evaluator_epoch for row in selected}
    if len(evaluator_epochs) > 1:
        blockers.append("record_evaluator_epoch_mismatch")
    report = {
        "report_version": "skilllearn_paper_report_v1",
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": protocol_lock.get("lock_hash"),
        "codex_agent_execution_policy_hash": (
            protocol.codex_agent_execution_policy.policy_hash
        ),
        "manifest_hash": expected_manifest_hash,
        "evaluator_epoch_hash": stable_hash({"epochs": sorted(evaluator_epochs)}),
        "split": split,
        "phase": phase_name,
        "expected_item_count": expected_items,
        "expected_repeats": expected_repeats,
        "record_count": len(selected),
        "control_summaries": summaries,
        "comparisons_vs_raw": comparisons,
        "primary_comparison": list(statistics_spec["primary_comparison"]),
        "primary_claim_eligible": not blockers,
        "claim_blockers": sorted(set(blockers)),
        "test_content_accessed": split == "test" and bool(selected),
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    report["report_hash"] = stable_hash(report)
    return report


def _summarize_control(
    records: Sequence[PaperTrialRecord],
    *,
    expected_repeats: int,
    confidence: float,
) -> dict[str, Any]:
    grouped = _group_records(records)
    complete: dict[str, list[PaperTrialRecord]] = {}
    invalid_items: list[str] = []
    for item_hash, rows in grouped.items():
        repeats = {row.repeat for row in rows}
        if (
            len(rows) != expected_repeats
            or repeats != set(range(1, expected_repeats + 1))
            or not all(row.valid for row in rows)
        ):
            invalid_items.append(item_hash)
            continue
        complete[item_hash] = sorted(rows, key=lambda row: row.repeat)
    majority = [
        sum(row.success for row in rows) >= math.ceil(expected_repeats / 2)
        for rows in complete.values()
    ]
    successes = sum(majority)
    lower, upper = _wilson_interval(successes, len(majority), confidence)
    valid_rows = [row for rows in complete.values() for row in rows]
    metric_names = sorted({key for row in valid_rows for key in row.metrics})
    return {
        "observed_item_count": len(grouped),
        "complete_item_count": len(complete),
        "invalid_item_count": len(invalid_items),
        "invalid_item_set_hash": stable_hash({"items": sorted(invalid_items)}),
        "valid_trial_count": len(valid_rows),
        "majority_successes": successes,
        "majority_success_rate": successes / len(majority) if majority else 0.0,
        "wilson_interval": [lower, upper],
        "mean_repeat_success": (
            statistics.fmean(float(row.success) for row in valid_rows) if valid_rows else 0.0
        ),
        "mean_score": statistics.fmean(row.score for row in valid_rows) if valid_rows else 0.0,
        "mean_total_tokens": (
            statistics.fmean(row.total_tokens for row in valid_rows) if valid_rows else 0.0
        ),
        "median_duration_seconds": (
            statistics.median(row.duration_seconds for row in valid_rows) if valid_rows else 0.0
        ),
        "mean_metrics": {
            metric: statistics.fmean(
                row.metrics[metric] for row in valid_rows if metric in row.metrics
            )
            for metric in metric_names
        },
    }


def _paired_comparison(
    records: Sequence[PaperTrialRecord],
    *,
    baseline_id: str,
    candidate_id: str,
    expected_repeats: int,
    expected_items: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    baseline = _group_records([row for row in records if row.control_id == baseline_id])
    candidate = _group_records([row for row in records if row.control_id == candidate_id])
    all_items = sorted(set(baseline) | set(candidate))
    deltas: list[float] = []
    gains = 0
    harms = 0
    ties = 0
    invalid_reasons: list[str] = []
    invalid_items: list[str] = []
    majority_threshold = math.ceil(expected_repeats / 2)
    for item_hash in all_items:
        base_rows = sorted(baseline.get(item_hash, ()), key=lambda row: row.repeat)
        candidate_rows = sorted(candidate.get(item_hash, ()), key=lambda row: row.repeat)
        expected_repeat_ids = list(range(1, expected_repeats + 1))
        if (
            len(base_rows) != expected_repeats
            or len(candidate_rows) != expected_repeats
            or [row.repeat for row in base_rows] != expected_repeat_ids
            or [row.repeat for row in candidate_rows] != expected_repeat_ids
        ):
            invalid_items.append(item_hash)
            continue
        if not all(row.valid for row in (*base_rows, *candidate_rows)):
            invalid_items.append(item_hash)
            continue
        if any(
            not row.prebuilt_image_key
            or not row.prebuilt_image_id
            or not row.agent_runtime_key
            or not row.agent_runtime_version
            for row in (*base_rows, *candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("execution_provenance_missing")
            continue
        if any(
            base.prebuilt_image_key != cand.prebuilt_image_key
            or base.prebuilt_image_id != cand.prebuilt_image_id
            for base, cand in zip(base_rows, candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("base_image_mismatch")
            continue
        if any(
            base.provider_fingerprint != cand.provider_fingerprint
            for base, cand in zip(base_rows, candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("provider_mismatch")
            continue
        if any(
            base.fairness_fingerprint != cand.fairness_fingerprint
            for base, cand in zip(base_rows, candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("budget_mismatch")
            continue
        if any(
            base.agent_runtime_key != cand.agent_runtime_key
            or base.agent_runtime_version != cand.agent_runtime_version
            for base, cand in zip(base_rows, candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("agent_runtime_mismatch")
            continue
        if any(base.pair_id != cand.pair_id for base, cand in zip(base_rows, candidate_rows)):
            invalid_items.append(item_hash)
            invalid_reasons.append("pair_id_mismatch")
            continue
        if any(
            base.protocol_hash != cand.protocol_hash
            or base.manifest_hash != cand.manifest_hash
            or base.evaluator_epoch != cand.evaluator_epoch
            for base, cand in zip(base_rows, candidate_rows)
        ):
            invalid_items.append(item_hash)
            invalid_reasons.append("run_identity_mismatch")
            continue
        base_mean = statistics.fmean(float(row.success) for row in base_rows)
        candidate_mean = statistics.fmean(float(row.success) for row in candidate_rows)
        deltas.append(candidate_mean - base_mean)
        base_majority = sum(row.success for row in base_rows) >= majority_threshold
        candidate_majority = sum(row.success for row in candidate_rows) >= majority_threshold
        if candidate_majority and not base_majority:
            gains += 1
        elif base_majority and not candidate_majority:
            harms += 1
        else:
            ties += 1
    if invalid_items:
        invalid_reasons.append("missing_or_invalid_item")
    if len(all_items) != expected_items:
        invalid_reasons.append("item_count_mismatch")
    if len(deltas) != expected_items:
        invalid_reasons.append("complete_pair_count_mismatch")
    lower, upper = _bootstrap_interval(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    return {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "observed_item_count": len(all_items),
        "complete_paired_item_count": len(deltas),
        "invalid_item_count": len(set(invalid_items)),
        "invalid_item_set_hash": stable_hash({"items": sorted(set(invalid_items))}),
        "mean_paired_success_delta": statistics.fmean(deltas) if deltas else 0.0,
        "clustered_bootstrap_interval": [lower, upper],
        "majority_net_gain_count": gains - harms,
        "majority_gain_count": gains,
        "majority_harm_count": harms,
        "majority_tie_count": ties,
        "mcnemar_p_value": _exact_mcnemar(gains, harms),
        "claim_valid": not invalid_reasons,
        "invalid_reasons": sorted(set(invalid_reasons)),
    }


def _group_records(records: Sequence[PaperTrialRecord]) -> dict[str, list[PaperTrialRecord]]:
    grouped: dict[str, list[PaperTrialRecord]] = {}
    for row in records:
        grouped.setdefault(row.item_id_hash, []).append(row)
    return grouped


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    draws = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    return _percentile(draws, alpha), _percentile(draws, 1.0 - alpha)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = max(0.0, min(1.0, quantile)) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _exact_mcnemar(gains: int, harms: int) -> float:
    discordant = gains + harms
    if discordant == 0:
        return 1.0
    tail = min(gains, harms)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return adjusted


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SkillLearnBench Paper Report",
        "",
        f"Protocol: `{report['protocol_id']}`",
        f"Split: `{report['split']}`",
        f"Primary claim eligible: `{str(report['primary_claim_eligible']).lower()}`",
        "",
        "| Control | Complete items | Success | 95% CI | Invalid items | Mean tokens | Median seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for control, summary in report["control_summaries"].items():
        interval = summary["wilson_interval"]
        lines.append(
            "| {control} | {items} | {success:.3f} | [{lower:.3f}, {upper:.3f}] | "
            "{invalid} | {tokens:.1f} | {seconds:.1f} |".format(
                control=control,
                items=summary["complete_item_count"],
                success=summary["majority_success_rate"],
                lower=interval[0],
                upper=interval[1],
                invalid=summary["invalid_item_count"],
                tokens=summary["mean_total_tokens"],
                seconds=summary["median_duration_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "| Comparison vs raw | Paired items | Delta | 95% clustered CI | Gains | Harms | Exact p | Holm p | Valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for control, summary in report["comparisons_vs_raw"].items():
        interval = summary["clustered_bootstrap_interval"]
        lines.append(
            "| {control} | {items} | {delta:.3f} | [{lower:.3f}, {upper:.3f}] | "
            "{gains} | {harms} | {p:.4f} | {holm:.4f} | {valid} |".format(
                control=control,
                items=summary["complete_paired_item_count"],
                delta=summary["mean_paired_success_delta"],
                lower=interval[0],
                upper=interval[1],
                gains=summary["majority_gain_count"],
                harms=summary["majority_harm_count"],
                p=summary["mcnemar_p_value"],
                holm=summary["holm_adjusted_p_value"],
                valid=str(summary["claim_valid"]).lower(),
            )
        )
    if report["claim_blockers"]:
        lines.extend(["", "Claim blockers: " + ", ".join(report["claim_blockers"])])
    return "\n".join(lines) + "\n"


def read_records(path: str | Path) -> tuple[PaperTrialRecord, ...]:
    rows: dict[tuple[str, str, int, str], PaperTrialRecord] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("paper trial JSONL rows must be objects")
        record = PaperTrialRecord.from_dict(payload)
        key = (record.item_id_hash, record.control_id, record.repeat, record.split)
        incumbent = rows.get(key)
        if incumbent is not None and record.attempt == incumbent.attempt:
            if incumbent.to_dict() != record.to_dict():
                raise ValueError("conflicting paper records share the same attempt key")
            continue
        if incumbent is None or record.attempt > incumbent.attempt:
            rows[key] = record
    return tuple(rows[key] for key in sorted(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-clean SkillLearn statistics.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "development",
            "sealed_test",
            "family_out_development",
            "family_out_transfer",
        ),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    protocol = PaperProtocol.read(args.protocol)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("protocol lock must contain one JSON object")
    report = build_paper_report(
        read_records(args.records),
        protocol=protocol,
        protocol_lock=lock,
        split=args.split,
        manifest_hash=(SplitManifest.read(args.manifest).manifest_hash if args.manifest else None),
        phase_name=args.phase,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report_hash": report["report_hash"], "primary_claim_eligible": report["primary_claim_eligible"], "claim_blockers": report["claim_blockers"]}, indent=2))


if __name__ == "__main__":
    main()
