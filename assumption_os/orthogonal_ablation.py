"""Ablate the orthogonal-new-family integration gate.

The goal is narrow: hold the proposal batch fixed, toggle only the
orthogonal novelty predicate, and measure whether a genuinely new explanation
axis collapses back into the old parent family.  This is a deterministic
recursive-retention proxy, not a live downstream QA run.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .novelty_integration import (
    NoveltyClass,
    _build_fixture_proposals,
    _build_fixture_store,
    build_novelty_integration_payload,
)
from .schema import EdgeType


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/orthogonal_ablation_20260608.json")


def build_orthogonal_ablation_payload(*, eval_id: str | None = None) -> dict[str, Any]:
    """Compare orthogonal-gate ON/OFF on the same residual/proposal batch."""

    with tempfile.TemporaryDirectory() as td:
        store = _build_fixture_store(Path(td) / "graph")
        proposal_payload, gold_labels = _build_fixture_proposals()
        enabled = build_novelty_integration_payload(
            store,
            proposal_payload,
            eval_id=f"{eval_id or 'orthogonal_ablation'}_enabled",
            enable_orthogonal=True,
        )
        disabled = build_novelty_integration_payload(
            store,
            proposal_payload,
            eval_id=f"{eval_id or 'orthogonal_ablation'}_disabled",
            enable_orthogonal=False,
        )

    enabled_rows = {row["proposal_id"]: row for row in enabled["rows"]}
    disabled_rows = {row["proposal_id"]: row for row in disabled["rows"]}
    proposal_ids = sorted(gold_labels)
    label_comparison = [
        {
            "proposal_id": proposal_id,
            "expected": gold_labels[proposal_id],
            "enabled": enabled_rows[proposal_id]["classification"],
            "disabled": disabled_rows[proposal_id]["classification"],
            "enabled_correct": enabled_rows[proposal_id]["classification"] == gold_labels[proposal_id],
            "disabled_correct": disabled_rows[proposal_id]["classification"] == gold_labels[proposal_id],
            "changed_by_toggle": (
                enabled_rows[proposal_id]["classification"]
                != disabled_rows[proposal_id]["classification"]
            ),
        }
        for proposal_id in proposal_ids
    ]
    enabled_retention = _retention_payload(enabled_rows, gold_labels)
    disabled_retention = _retention_payload(disabled_rows, gold_labels)
    enabled_recursive = _recursive_proxy_payload(enabled_rows)
    disabled_recursive = _recursive_proxy_payload(disabled_rows)
    metrics = {
        "proposal_count": len(proposal_ids),
        "classification_accuracy_enabled": _accuracy(label_comparison, "enabled_correct"),
        "classification_accuracy_disabled": _accuracy(label_comparison, "disabled_correct"),
        "orthogonal_recall_enabled": _orthogonal_recall(label_comparison, "enabled"),
        "orthogonal_recall_disabled": _orthogonal_recall(label_comparison, "disabled"),
        "non_orthogonal_stability": _non_orthogonal_stability(label_comparison, gold_labels),
        "orthogonal_edge_count_enabled": enabled["recommended_edge_counts"].get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "orthogonal_edge_count_disabled": disabled["recommended_edge_counts"].get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "axis_retention_rate_enabled": enabled_retention["axis_retention_rate"],
        "axis_retention_rate_disabled": disabled_retention["axis_retention_rate"],
        "axis_contamination_count_enabled": enabled_recursive["axis_contamination_count"],
        "axis_contamination_count_disabled": disabled_recursive["axis_contamination_count"],
        "recursive_metaproductivity_proxy_enabled": enabled_recursive["metaproductivity_proxy"],
        "recursive_metaproductivity_proxy_disabled": disabled_recursive["metaproductivity_proxy"],
    }
    metrics["classification_accuracy_delta"] = round(
        metrics["classification_accuracy_enabled"] - metrics["classification_accuracy_disabled"],
        4,
    )
    metrics["axis_retention_delta"] = round(
        metrics["axis_retention_rate_enabled"] - metrics["axis_retention_rate_disabled"],
        4,
    )
    metrics["metaproductivity_proxy_delta"] = round(
        metrics["recursive_metaproductivity_proxy_enabled"]
        - metrics["recursive_metaproductivity_proxy_disabled"],
        4,
    )
    gates = {
        "same_proposal_set": sorted(enabled_rows) == sorted(disabled_rows) == proposal_ids,
        "only_orthogonal_candidate_changes": all(
            row["changed_by_toggle"]
            == (row["expected"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value)
            for row in label_comparison
        ),
        "orthogonal_enabled_improves_gold_accuracy": metrics["classification_accuracy_delta"] > 0.0,
        "orthogonal_enabled_retains_new_axis": (
            enabled_retention["orthogonal_new_axis_retained"]
            and not disabled_retention["orthogonal_new_axis_retained"]
        ),
        "orthogonal_disabled_collapses_to_parent_family": bool(
            disabled_retention["collapsed_orthogonal_proposals"]
        ),
        "no_nonorthogonal_regression": metrics["non_orthogonal_stability"] == 1.0,
        "recursive_proxy_improves": metrics["metaproductivity_proxy_delta"] > 0.0,
        "disabled_has_axis_contamination": (
            metrics["axis_contamination_count_disabled"]
            > metrics["axis_contamination_count_enabled"]
        ),
    }
    return {
        "eval_id": eval_id or "orthogonal_ablation_20260608",
        "eval_kind": "orthogonal_new_family_gate_toggle_ablation",
        "performance_validation": True,
        "validation_scope": (
            "deterministic recursive-retention proxy over the same proposal/residual fixture; "
            "not a live downstream QA or LLM-judge ablation"
        ),
        "pass": all(gates.values()),
        "proposal_batch_eval_id": "novelty_fixture_source",
        "gold_labels": gold_labels,
        "enabled_summary": _condition_summary(enabled),
        "disabled_summary": _condition_summary(disabled),
        "label_comparison": label_comparison,
        "retention": {
            "enabled": enabled_retention,
            "disabled": disabled_retention,
        },
        "recursive_proxy": {
            "enabled": enabled_recursive,
            "disabled": disabled_recursive,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "interpretation": (
            "With the orthogonal gate enabled, the evaluator-drift hypothesis is retained as a separate "
            "new-family axis linked by orthogonal_to.  With the gate disabled, the same proposal falls back "
            "to a specializes edge under controlled-variable reasoning, which pollutes that family with a "
            "different residual explanation axis and lowers the recursive metaproductivity proxy."
        ),
    }


def _condition_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "orthogonal_gate_enabled": payload.get("orthogonal_gate_enabled"),
        "classification_counts": payload.get("classification_counts"),
        "recommended_edge_counts": payload.get("recommended_edge_counts"),
        "pass": payload.get("pass"),
    }


def _retention_payload(rows_by_id: dict[str, dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    retained = 0
    collapsed: list[dict[str, Any]] = []
    rows = []
    for proposal_id, expected in sorted(labels.items()):
        row = rows_by_id[proposal_id]
        expected_new_axis = expected in {
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            NoveltyClass.GENUINELY_NEW_FAMILY.value,
        }
        observed_new_axis = bool(row.get("is_new_family"))
        ok = observed_new_axis == expected_new_axis
        retained += int(ok)
        if expected == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value and not observed_new_axis:
            collapsed.append({
                "proposal_id": proposal_id,
                "observed_classification": row.get("classification"),
                "observed_action": row.get("recommended_action"),
                "collapsed_into": row.get("existing_node_id") or row.get("parent_node_id"),
            })
        rows.append({
            "proposal_id": proposal_id,
            "expected_new_axis": expected_new_axis,
            "observed_new_axis": observed_new_axis,
            "passed": ok,
            "classification": row.get("classification"),
            "family_anchor": _family_anchor(row),
        })
    return {
        "axis_retention_rate": round(retained / len(labels), 4) if labels else 0.0,
        "orthogonal_new_axis_retained": not collapsed,
        "collapsed_orthogonal_proposals": collapsed,
        "rows": rows,
    }


def _recursive_proxy_payload(rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    outcomes_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    axes_by_anchor: dict[str, set[str]] = defaultdict(set)
    for outcome in _recursive_descendant_fixture():
        row = rows_by_id[outcome["seed_proposal_id"]]
        anchor = _family_anchor(row)
        axes_by_anchor[anchor].add(outcome["intended_axis"])
        outcomes_by_anchor[anchor].append({
            **outcome,
            "family_anchor": anchor,
            "seed_classification": row.get("classification"),
        })

    family_rows = []
    score_terms = []
    contamination_count = 0
    for anchor, outcomes in sorted(outcomes_by_anchor.items()):
        axes = sorted(axes_by_anchor[anchor])
        contamination = max(0, len(axes) - 1)
        contamination_count += contamination
        accepted = [row for row in outcomes if row["decision"] == "accept"]
        rejected_harm = [row for row in outcomes if row["decision"] == "reject_harm"]
        mean_delta = sum(float(row["utility_delta"]) for row in outcomes) / len(outcomes)
        accepted_rate = len(accepted) / len(outcomes)
        harm_rate = len(rejected_harm) / len(outcomes)
        family_score = mean_delta + 0.16 * accepted_rate - 0.18 * harm_rate - 0.10 * contamination
        family_score = max(0.0, min(1.0, family_score))
        family_rows.append({
            "family_anchor": anchor,
            "intended_axes": axes,
            "axis_contamination": contamination,
            "outcome_count": len(outcomes),
            "accepted_count": len(accepted),
            "rejected_harm_count": len(rejected_harm),
            "mean_utility_delta": round(mean_delta, 4),
            "family_metaproductivity_proxy": round(family_score, 4),
        })
        score_terms.append(family_score)

    return {
        "metric": "family-separated next-generation accepted-descendant utility proxy",
        "metaproductivity_proxy": round(sum(score_terms) / len(score_terms), 4) if score_terms else 0.0,
        "axis_contamination_count": contamination_count,
        "productive_family_count": sum(1 for row in family_rows if row["family_metaproductivity_proxy"] >= 0.10),
        "family_rows": family_rows,
        "descendant_rows": [
            row
            for rows in outcomes_by_anchor.values()
            for row in rows
        ],
    }


def _recursive_descendant_fixture() -> list[dict[str, Any]]:
    return [
        {
            "seed_proposal_id": "prop_specialization",
            "descendant_id": "desc_control_scope_measurement",
            "intended_axis": "controlled_variable_scope",
            "decision": "accept",
            "utility_delta": 0.10,
        },
        {
            "seed_proposal_id": "prop_specialization",
            "descendant_id": "desc_control_overfit_warning",
            "intended_axis": "controlled_variable_scope",
            "decision": "reject_harm",
            "utility_delta": -0.04,
        },
        {
            "seed_proposal_id": "prop_orthogonal_family",
            "descendant_id": "desc_eval_drift_detector",
            "intended_axis": "evaluator_drift",
            "decision": "accept",
            "utility_delta": 0.14,
        },
        {
            "seed_proposal_id": "prop_orthogonal_family",
            "descendant_id": "desc_eval_cross_judge_calibration",
            "intended_axis": "evaluator_drift",
            "decision": "accept",
            "utility_delta": 0.11,
        },
        {
            "seed_proposal_id": "prop_orthogonal_family",
            "descendant_id": "desc_eval_placebo_disagreement",
            "intended_axis": "evaluator_drift",
            "decision": "accept",
            "utility_delta": 0.08,
        },
    ]


def _family_anchor(row: dict[str, Any]) -> str:
    if row.get("classification") in {
        NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
        NoveltyClass.GENUINELY_NEW_FAMILY.value,
    }:
        return str(row.get("candidate_node_id") or row.get("proposal_id"))
    return str(row.get("existing_node_id") or row.get("parent_node_id") or row.get("candidate_node_id"))


def _accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(1 for row in rows if row[key]) / len(rows), 4) if rows else 0.0


def _orthogonal_recall(rows: list[dict[str, Any]], key: str) -> float:
    gold = [row for row in rows if row["expected"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value]
    if not gold:
        return 0.0
    hits = sum(1 for row in gold if row[key] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value)
    return round(hits / len(gold), 4)


def _non_orthogonal_stability(rows: list[dict[str, Any]], labels: dict[str, str]) -> float:
    non_orthogonal = [
        row
        for row in rows
        if labels[row["proposal_id"]] != NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
    ]
    if not non_orthogonal:
        return 0.0
    stable = sum(1 for row in non_orthogonal if row["enabled"] == row["disabled"])
    return round(stable / len(non_orthogonal), 4)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run orthogonal-new-family ON/OFF ablation.")
    parser.add_argument("--eval-id", default="orthogonal_ablation_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = build_orthogonal_ablation_payload(eval_id=args.eval_id)
    _write_json(Path(args.out), payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": args.out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
