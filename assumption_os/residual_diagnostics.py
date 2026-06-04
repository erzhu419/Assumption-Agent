"""Diagnostics for residual classification and trace coverage.

The residual analyzer is only useful if two things hold:

* residual taxonomy labels agree with independent labels on representative
  failures, instead of only being convenient deterministic tags; and
* judged bypass/non-attributed losses become residual-bearing trace rows before
  the loop synthesizes new hypotheses.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .graph_memory import JsonlGraphStore
from .residual_clusterer import collect_residual_records
from .residuals import classify_manifest
from .schema import ResidualType, TrialManifest, TrialStatus, stable_id


@dataclass(frozen=True)
class LabeledResidualExample:
    example_id: str
    expected_type: ResidualType | str
    residual: str | None
    observed_effect: str = ""
    expected_effect: str = ""
    why_selected: str = ""
    label_source: str = "curated_reconstruction_residual_gold_v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> TrialManifest:
        return TrialManifest(
            problem_id=f"residual_gold::{self.example_id}",
            action_type="residual_label_gold",
            component="residual_diagnostics",
            assumption="Residual labels should separate execution, memory, evaluator, simulator, and assumption defects before graph mutation.",
            why_selected=self.why_selected,
            expected_effect=self.expected_effect,
            observed_effect=self.observed_effect,
            residual=self.residual,
            status=TrialStatus.OBSERVED if not self.residual else TrialStatus.FAILED,
            metadata={"label_source": self.label_source, **self.metadata},
            trial_id=stable_id("trial", "residual_gold", self.example_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "expected_type": _value(self.expected_type),
            "residual": self.residual,
            "observed_effect": self.observed_effect,
            "expected_effect": self.expected_effect,
            "why_selected": self.why_selected,
            "label_source": self.label_source,
            "metadata": self.metadata,
        }


DEFAULT_LABELED_RESIDUAL_EXAMPLES: tuple[LabeledResidualExample, ...] = (
    LabeledResidualExample(
        example_id="no_residual_reinforce",
        expected_type=ResidualType.NO_RESIDUAL,
        residual=None,
        observed_effect="candidate won and no failure residual was recorded",
    ),
    LabeledResidualExample(
        example_id="execution_not_applied",
        expected_type=ResidualType.EXECUTION_LAPSE,
        residual="The checklist was valid but did not apply the final verification step in the actual answer.",
        observed_effect="answer skipped the required audit",
    ),
    LabeledResidualExample(
        example_id="execution_decorative",
        expected_type=ResidualType.EXECUTION_LAPSE,
        residual="The method became decorative and was not applied to the concrete constraints.",
        why_selected="active assumption was selected but not executed faithfully",
    ),
    LabeledResidualExample(
        example_id="optimization_partial",
        expected_type=ResidualType.OPTIMIZATION,
        residual="The direction was partly right but not concrete enough; refine the payload and rerun.",
        observed_effect="baseline was more actionable",
    ),
    LabeledResidualExample(
        example_id="optimization_bypass_bridge",
        expected_type=ResidualType.OPTIMIZATION,
        residual="No graph ids fired, but bypass/cache route science_mechanism lost; optimize the bypass bridge.",
        expected_effect="math/science bypass should become an actionable bridge repair",
    ),
    LabeledResidualExample(
        example_id="evaluator_style_bias",
        expected_type=ResidualType.EVALUATOR_DEFECT,
        residual="The judge over-weighted verbosity and style instead of the objective success criterion.",
        observed_effect="verifier may be measuring the wrong target",
    ),
    LabeledResidualExample(
        example_id="memory_wrong_context",
        expected_type=ResidualType.MEMORY_DEFECT,
        residual="Retrieval selected irrelevant memory and missed the trigger concept from the sample.",
        why_selected="active memories did not match the problem",
    ),
    LabeledResidualExample(
        example_id="simulator_bad_rollout",
        expected_type=ResidualType.SIMULATOR_DEFECT,
        residual="World model rollout predicted the wrong outcome and the brier score worsened.",
        observed_effect="cheap predictor was miscalibrated",
    ),
    LabeledResidualExample(
        example_id="assumption_false",
        expected_type=ResidualType.ASSUMPTION_DEFECT,
        residual="The parent assumption is false under this condition and contradicts the observed result.",
        expected_effect="narrow or deprecate the assumption",
    ),
    LabeledResidualExample(
        example_id="discovery_novel",
        expected_type=ResidualType.DISCOVERY,
        residual="A novel failure pattern appears that is not explained by retrieval, execution, evaluator, or simulator categories.",
        observed_effect="cluster before creating a new method",
    ),
)


def build_residual_label_agreement_payload(
    *,
    eval_id: str,
    examples: Iterable[LabeledResidualExample | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score rule-based residual labels against independent gold examples."""

    rows = []
    for example in examples or DEFAULT_LABELED_RESIDUAL_EXAMPLES:
        labeled = _coerce_example(example)
        assessment = classify_manifest(labeled.to_manifest())
        expected = _value(labeled.expected_type)
        predicted = assessment.residual_type.value
        rows.append({
            "example_id": labeled.example_id,
            "expected_type": expected,
            "predicted_type": predicted,
            "match": expected == predicted,
            "reason": assessment.reason,
            "recommended_action": assessment.recommended_action,
            "label_source": labeled.label_source,
        })
    labels = sorted({row["expected_type"] for row in rows} | {row["predicted_type"] for row in rows})
    per_label = _per_label_metrics(rows, labels)
    accuracy = _safe_div(sum(1 for row in rows if row["match"]), len(rows))
    macro_f1 = _safe_div(sum(row["f1"] for row in per_label.values()), len(per_label))
    return {
        "eval_id": eval_id,
        "label_source_counts": dict(Counter(row["label_source"] for row in rows)),
        "example_count": len(rows),
        "label_count": len(labels),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "expected_type_counts": dict(Counter(row["expected_type"] for row in rows)),
        "predicted_type_counts": dict(Counter(row["predicted_type"] for row in rows)),
        "confusion": _confusion(rows),
        "per_label": per_label,
        "rows": rows,
        "pass": len(rows) >= 8 and accuracy >= 0.85 and macro_f1 >= 0.8,
    }


def build_large_residual_label_calibration_payload(
    *,
    eval_id: str,
    store: JsonlGraphStore | None = None,
    trace_dataset_payload: dict[str, Any] | None = None,
    target_examples: int = 120,
) -> dict[str, Any]:
    """Build a larger residual calibration set from first-party graph/trace labels.

    This is not a substitute for human adjudication.  It records label sources
    explicitly so paper-readiness gates can distinguish curated gold, graph
    writeback labels, and trace-derived labels.
    """

    examples: list[LabeledResidualExample] = list(DEFAULT_LABELED_RESIDUAL_EXAMPLES)
    source_counts = Counter(example.label_source for example in examples)
    if store is not None:
        graph_examples = _examples_from_graph_residual_records(store)
        examples.extend(graph_examples)
        source_counts.update(example.label_source for example in graph_examples)
    if trace_dataset_payload is not None and len(examples) < target_examples:
        trace_examples = _examples_from_trace_dataset(
            trace_dataset_payload,
            max_examples=max(0, target_examples - len(examples)),
        )
        examples.extend(trace_examples)
        source_counts.update(example.label_source for example in trace_examples)

    examples = _dedupe_examples(examples)[:target_examples]
    agreement = build_residual_label_agreement_payload(
        eval_id=f"{eval_id}_agreement",
        examples=examples,
    )
    expected_counts = agreement.get("expected_type_counts", {})
    coverage = {
        "example_count": agreement["example_count"],
        "target_examples": target_examples,
        "label_count": agreement["label_count"],
        "label_source_counts": agreement["label_source_counts"],
        "expected_type_counts": expected_counts,
        "has_curated_gold": expected_counts.get(ResidualType.EXECUTION_LAPSE.value, 0) >= 1
        and expected_counts.get(ResidualType.SIMULATOR_DEFECT.value, 0) >= 1,
        "has_graph_residuals": any(
            str(source).startswith("first_party_graph")
            for source in agreement.get("label_source_counts", {})
        ),
        "has_trace_residuals": any(
            str(source).startswith("first_party_trace") or str(source).startswith("trace_dataset")
            for source in agreement.get("label_source_counts", {})
        ),
    }
    pass_condition = (
        agreement["example_count"] >= 100
        and agreement["accuracy"] >= 0.85
        and agreement["macro_f1"] >= 0.80
        and coverage["label_count"] >= 5
        and coverage["has_curated_gold"]
        and coverage["has_graph_residuals"]
    )
    return {
        "eval_id": eval_id,
        "target_examples": target_examples,
        "calibration_kind": "large_residual_label_calibration",
        "agreement": agreement,
        "coverage": coverage,
        "example_count": agreement["example_count"],
        "label_count": agreement["label_count"],
        "accuracy": agreement["accuracy"],
        "macro_f1": agreement["macro_f1"],
        "label_source_counts": agreement["label_source_counts"],
        "expected_type_counts": expected_counts,
        "pass": pass_condition,
        "notes": [
            "Large calibration uses curated gold plus first-party graph/trace labels.",
            "It improves scale coverage but still does not replace future human adjudication.",
        ],
    }


def build_trace_residual_coverage_payload(
    *,
    trace_dataset_payload: dict[str, Any],
    eval_id: str,
) -> dict[str, Any]:
    """Check that loss rows needing attribution have residual-bearing traces."""

    rows = [row for row in trace_dataset_payload.get("rows", []) if isinstance(row, dict)]
    loss_rows = [row for row in rows if row.get("outcome") == "loss"]
    non_attributed_losses = [
        row for row in loss_rows
        if not row.get("activated_assumption_ids")
    ]
    bypass_losses = [
        row for row in non_attributed_losses
        if _is_bypass_or_skipped_loss(row)
    ]
    skipped_losses = [
        row for row in non_attributed_losses
        if str(row.get("source_kind") or row.get("features", {}).get("source_kind") or "").startswith("skipped")
    ]
    uncovered = [
        row for row in non_attributed_losses
        if not _has_residual_trace(row)
    ]
    bypass_uncovered = [
        row for row in bypass_losses
        if not _has_residual_trace(row) or row.get("residual_type") == ResidualType.NO_RESIDUAL.value
    ]
    bypass_not_trainable = [
        row for row in bypass_losses
        if not (row.get("trainable") or row.get("trace_source") in {"artifact_replay", "first_party_runtime"})
    ]
    residual_type_counts = Counter(str(row.get("residual_type") or "missing") for row in loss_rows)
    coverage_rate = _safe_div(len(non_attributed_losses) - len(uncovered), len(non_attributed_losses))
    bypass_coverage_rate = _safe_div(len(bypass_losses) - len(bypass_uncovered), len(bypass_losses))
    skipped_coverage_rate = _safe_div(
        sum(1 for row in skipped_losses if _has_residual_trace(row)),
        len(skipped_losses),
    )
    pass_condition = (
        len(loss_rows) > 0
        and not uncovered
        and not bypass_uncovered
        and not bypass_not_trainable
        and (not bypass_losses or residual_type_counts.get(ResidualType.OPTIMIZATION.value, 0) >= len(bypass_losses))
    )
    return {
        "eval_id": eval_id,
        "source_trace_dataset_eval_id": trace_dataset_payload.get("eval_id"),
        "loss_row_count": len(loss_rows),
        "non_attributed_loss_count": len(non_attributed_losses),
        "non_attributed_loss_residual_count": len(non_attributed_losses) - len(uncovered),
        "non_attributed_loss_coverage_rate": round(coverage_rate, 4),
        "bypass_loss_count": len(bypass_losses),
        "bypass_loss_residual_count": len(bypass_losses) - len(bypass_uncovered),
        "bypass_loss_trainable_count": len(bypass_losses) - len(bypass_not_trainable),
        "bypass_loss_coverage_rate": round(bypass_coverage_rate, 4),
        "skipped_loss_count": len(skipped_losses),
        "skipped_loss_residual_count": sum(1 for row in skipped_losses if _has_residual_trace(row)),
        "skipped_loss_coverage_rate": round(skipped_coverage_rate, 4),
        "loss_residual_type_counts": dict(residual_type_counts),
        "uncovered_problem_ids": _problem_ids(uncovered),
        "bypass_uncovered_problem_ids": _problem_ids(bypass_uncovered),
        "bypass_not_trainable_problem_ids": _problem_ids(bypass_not_trainable),
        "pass": pass_condition,
    }


def _examples_from_graph_residual_records(store: JsonlGraphStore) -> list[LabeledResidualExample]:
    examples = []
    for record in collect_residual_records(store):
        expected = _expected_type_from_record(record.residual_type, residual=record.residual)
        examples.append(LabeledResidualExample(
            example_id=f"graph_{record.record_id}",
            expected_type=expected,
            residual=record.residual,
            observed_effect=f"action={record.action_type}; first_party_graph_residual=true",
            expected_effect="Graph residual labels should remain stable under the residual taxonomy.",
            why_selected=f"first-party graph residual record for {record.problem_id}",
            label_source=f"first_party_graph_residual::{record.residual_type}",
            metadata={
                "record_id": record.record_id,
                "problem_id": record.problem_id,
                "component": record.component,
                "action_type": record.action_type,
                "source_residual_type": record.residual_type,
            },
        ))
    return examples


def _examples_from_trace_dataset(trace_dataset_payload: dict[str, Any], *, max_examples: int) -> list[LabeledResidualExample]:
    if max_examples <= 0:
        return []
    rows = [
        row for row in trace_dataset_payload.get("rows", [])
        if isinstance(row, dict) and row.get("trainable")
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rtype = str(row.get("residual_type") or "")
        if rtype:
            buckets[rtype].append(row)
    examples = []
    per_bucket = max(1, max_examples // max(1, len(buckets)))
    for rtype in sorted(buckets):
        for row in buckets[rtype][:per_bucket]:
            expected = _expected_type_from_record(rtype, residual=row.get("residual"))
            residual = row.get("residual")
            if expected == ResidualType.NO_RESIDUAL.value:
                residual = None
            examples.append(LabeledResidualExample(
                example_id=f"trace_{row.get('row_id') or stable_id('trace_row', row.get('problem_id'), rtype)}",
                expected_type=expected,
                residual=residual,
                observed_effect=f"outcome={row.get('outcome')}; winner={row.get('winner')}; trace_source={row.get('trace_source')}",
                expected_effect="Trace-derived residual labels should classify consistently with judged outcomes.",
                why_selected=f"trace dataset row for {row.get('problem_id')}",
                label_source=f"trace_dataset_residual::{rtype}",
                metadata={
                    "row_id": row.get("row_id"),
                    "problem_id": row.get("problem_id"),
                    "trace_source": row.get("trace_source"),
                    "source_residual_type": rtype,
                },
            ))
            if len(examples) >= max_examples:
                return examples
    return examples


def _expected_type_from_record(residual_type: str, *, residual: str | None = None) -> str:
    if residual_type == "unknown":
        text = str(residual or "").lower()
        if any(token in text for token in ["retrieval", "memory", "检索", "记忆", "wrong memory", "irrelevant"]):
            return ResidualType.MEMORY_DEFECT.value
        if any(token in text for token in [
            "没",
            "漏",
            "错过",
            "缺少",
            "不够",
            "仍",
            "只",
            "miss",
            "missed",
            "not convert",
            "did not convert",
            "optimize",
            "优化",
        ]):
            return ResidualType.OPTIMIZATION.value
        return ResidualType.DISCOVERY.value
    try:
        return ResidualType(residual_type).value
    except ValueError:
        return ResidualType.DISCOVERY.value


def _dedupe_examples(examples: list[LabeledResidualExample]) -> list[LabeledResidualExample]:
    seen = set()
    out = []
    for example in examples:
        key = (example.example_id, _value(example.expected_type), example.residual)
        if key in seen:
            continue
        seen.add(key)
        out.append(example)
    return out


def _coerce_example(example: LabeledResidualExample | dict[str, Any]) -> LabeledResidualExample:
    if isinstance(example, LabeledResidualExample):
        return example
    return LabeledResidualExample(
        example_id=str(example["example_id"]),
        expected_type=example["expected_type"],
        residual=example.get("residual"),
        observed_effect=str(example.get("observed_effect") or ""),
        expected_effect=str(example.get("expected_effect") or ""),
        why_selected=str(example.get("why_selected") or ""),
        label_source=str(example.get("label_source") or "external_residual_gold"),
        metadata=dict(example.get("metadata") or {}),
    )


def _per_label_metrics(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, dict[str, float]]:
    metrics = {}
    for label in labels:
        tp = sum(1 for row in rows if row["expected_type"] == label and row["predicted_type"] == label)
        fp = sum(1 for row in rows if row["expected_type"] != label and row["predicted_type"] == label)
        fn = sum(1 for row in rows if row["expected_type"] == label and row["predicted_type"] != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return metrics


def _confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row["expected_type"]][row["predicted_type"]] += 1
    return {expected: dict(predicted) for expected, predicted in sorted(matrix.items())}


def _is_bypass_or_skipped_loss(row: dict[str, Any]) -> bool:
    features = row.get("features") or {}
    components = set(row.get("component_counts") or {})
    residual_text = str(row.get("residual") or "").lower()
    return bool(
        row.get("bypass_route")
        or features.get("bypass_route")
        or row.get("source_kind") == "skipped_judgment"
        or features.get("source_kind") == "skipped_judgment"
        or "phase2_math_science_bypass" in components
        or "phase2_cache_hit" in components
        or any(str(component).startswith("artifact_replay") for component in components)
        or "bypass/cache route" in residual_text
    )


def _has_residual_trace(row: dict[str, Any]) -> bool:
    return bool(
        row.get("residual")
        and row.get("residual_type")
        and row.get("residual_type") != ResidualType.NO_RESIDUAL.value
        and (
            row.get("trace_event_count", 0) > 0
            or row.get("trace_source") in {"artifact_replay", "first_party_runtime", "first_party_distilled_transition"}
            or row.get("distilled_transition")
        )
    )


def _problem_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(str(row.get("problem_id") or row.get("row_id") or "") for row in rows if row.get("problem_id") or row.get("row_id"))


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _value(value: ResidualType | str) -> str:
    return value.value if isinstance(value, ResidualType) else str(value)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Residual label diagnostics.")
    ap.add_argument("--eval-id", default="residual_label_agreement_cli")
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--trace-dataset", default=None)
    ap.add_argument("--target-examples", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.large:
        store = JsonlGraphStore(args.graph_dir) if args.graph_dir else None
        trace_dataset = _load_json(Path(args.trace_dataset)) if args.trace_dataset else None
        payload = build_large_residual_label_calibration_payload(
            eval_id=args.eval_id,
            store=store,
            trace_dataset_payload=trace_dataset,
            target_examples=args.target_examples,
        )
    else:
        payload = build_residual_label_agreement_payload(eval_id=args.eval_id)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
