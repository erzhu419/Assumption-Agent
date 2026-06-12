"""Leakage audit for the production graph-action simulator evidence.

The production simulator panel is intentionally strong, so it needs a separate
audit that checks the evidence is not strong because labels, outcomes, or best
arms leaked into simulator state/features or prediction fields.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .autonomy_journal import PAPER_DIR
from .simulator_production_evidence_expansion import DEFAULT_DATASET_V1
from .simulator_transition_schema import validate_transition_rows


DEFAULT_OUT = PAPER_DIR / "simulator_no_leakage_audit_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/simulator_no_leakage_audit_20260612.md")
PRODUCTION_EVIDENCE_PATH = PAPER_DIR / "simulator_production_evidence_20260612.json"
SOURCE_PATH = Path("assumption_os/simulator_production_evidence_expansion.py")

LEAK_TOKEN_RE = re.compile(
    r"(accepted|label|outcome|utility|regress|control_harm|best[_-]?arm|selected[_-]?arm|oracle|gold|answer|prompt|judge|secret|api[_-]?key|token)",
    re.IGNORECASE,
)
ALLOWED_ACTION_TOKENS = {"arm", "arm_family"}


def build_simulator_no_leakage_audit_payload(
    *,
    root: Path,
    eval_id: str = "simulator_no_leakage_audit_20260612",
    dataset_path: Path | None = None,
    production_evidence_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    dataset_path = dataset_path or DEFAULT_DATASET_V1
    production_evidence_path = production_evidence_path or PRODUCTION_EVIDENCE_PATH
    dataset_abs = dataset_path if dataset_path.is_absolute() else root / dataset_path
    evidence_abs = production_evidence_path if production_evidence_path.is_absolute() else root / production_evidence_path
    rows = _load_jsonl(dataset_abs)
    production_evidence = _load_json(evidence_abs)
    validation = validate_transition_rows(rows)
    source_scan = _scan_source(root / SOURCE_PATH)
    feature_scan = _scan_rows(rows)
    group_scan = _scan_same_state_groups(rows)
    prediction_scan = _scan_prediction_outcome_independence(rows)
    evidence_metrics = production_evidence.get("metrics", {})
    metrics = {
        "row_count": len(rows),
        "valid_row_count": validation.valid_row_count,
        "production_evidence_pass": bool(production_evidence.get("pass")),
        "state_feature_leak_count": feature_scan["state_feature_leak_count"],
        "state_feature_checked_count": feature_scan["state_feature_checked_count"],
        "provenance_leak_count": feature_scan["provenance_leak_count"],
        "row_id_leak_count": feature_scan["row_id_leak_count"],
        "best_or_selected_token_count": feature_scan["best_or_selected_token_count"],
        "same_state_group_count": group_scan["same_state_group_count"],
        "same_state_group_key_leak_count": group_scan["same_state_group_key_leak_count"],
        "min_arm_count_per_group": group_scan["min_arm_count_per_group"],
        "prediction_outcome_exact_identity_count": prediction_scan["prediction_outcome_exact_identity_count"],
        "prediction_outcome_near_identity_rate": prediction_scan["prediction_outcome_near_identity_rate"],
        "mean_abs_prediction_outcome_gap": prediction_scan["mean_abs_prediction_outcome_gap"],
        "prediction_accept_label_exact_count": prediction_scan["prediction_accept_label_exact_count"],
        "best_arm_agreement_rate": float(evidence_metrics.get("best_arm_agreement_rate", 0.0)),
        "counterfactual_mae": float(evidence_metrics.get("counterfactual_mae", 1.0)),
        "global_baseline_mae": float(evidence_metrics.get("global_baseline_mae", 0.0)),
        "feature_model_loo_brier": float(evidence_metrics.get("feature_model_loo_brier", 1.0)),
        "base_rate_loo_brier": float(evidence_metrics.get("base_rate_loo_brier", 0.0)),
        "source_direct_prediction_outcome_alias_count": source_scan["direct_prediction_outcome_alias_count"],
        "source_oracle_selector_alias_count": source_scan["oracle_selector_alias_count"],
        "source_forbidden_token_count": source_scan["forbidden_token_count"],
        "raw_simulator_promoted": bool(evidence_metrics.get("raw_simulator_promoted")),
        "gate_router_promoted": bool(evidence_metrics.get("gate_router_promoted")),
        "production_simulator_candidate_allowed": bool(evidence_metrics.get("production_simulator_candidate_allowed")),
        "claim_boundary_preserved": (
            production_evidence.get("claim_boundary", {}).get("blocked_claim")
            == "task-world simulator replacing live validation or judges"
        ),
    }
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["row_count"] and metrics["row_count"] >= 2000,
        "production_evidence_passes": metrics["production_evidence_pass"] is True,
        "no_state_feature_label_leak": metrics["state_feature_leak_count"] == 0,
        "no_provenance_label_leak": metrics["provenance_leak_count"] == 0,
        "no_row_id_label_leak": metrics["row_id_leak_count"] == 0,
        "no_best_or_selected_feature_leak": metrics["best_or_selected_token_count"] == 0,
        "same_state_group_keys_clean": metrics["same_state_group_key_leak_count"] == 0,
        "same_state_groups_multiarm": metrics["same_state_group_count"] >= 100
        and metrics["min_arm_count_per_group"] >= 4,
        "prediction_not_exact_outcome": metrics["prediction_outcome_exact_identity_count"] == 0,
        "prediction_not_near_identity": metrics["prediction_outcome_near_identity_rate"] <= 0.02,
        "prediction_outcome_gap_nontrivial": metrics["mean_abs_prediction_outcome_gap"] >= 0.025,
        "prediction_not_binary_label": metrics["prediction_accept_label_exact_count"] == 0,
        "best_arm_agreement_plausible_not_perfect": 0.80 <= metrics["best_arm_agreement_rate"] <= 0.98,
        "counterfactual_still_beats_global": metrics["counterfactual_mae"] < metrics["global_baseline_mae"],
        "brier_still_beats_base": metrics["feature_model_loo_brier"] < metrics["base_rate_loo_brier"],
        "source_no_direct_prediction_outcome_alias": metrics["source_direct_prediction_outcome_alias_count"] == 0,
        "source_no_oracle_selector_alias": metrics["source_oracle_selector_alias_count"] == 0,
        "source_forbidden_tokens_only_in_audit_safe_context": metrics["source_forbidden_token_count"] == 0,
        "claim_boundary_preserved": metrics["claim_boundary_preserved"] is True
        and metrics["raw_simulator_promoted"] is False
        and metrics["gate_router_promoted"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_no_leakage_audit",
        "last_three_part_ticket": "B8_simulator_leakage_audit",
        "performance_validation": True,
        "validation_scope": (
            "Audits the production graph-action simulator evidence for label, outcome, oracle, best-arm, and "
            "prediction/outcome construction leakage.  Passing this artifact supports only the bounded "
            "triage/router claim; it does not allow replacing live validation or judges."
        ),
        "source": {
            "dataset_path": _display_path(root, dataset_abs),
            "production_evidence_path": _display_path(root, evidence_abs),
            "source_path": str(SOURCE_PATH),
        },
        "feature_scan": feature_scan,
        "same_state_group_scan": group_scan,
        "prediction_outcome_scan": prediction_scan,
        "source_scan": source_scan,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "production graph-action simulator evidence is leakage-audited for triage/routing",
        "blocked_claims": [
            "oracle simulator replacing live validation",
            "label-leaking production world model",
            "best-arm policy selected from outcome labels",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Simulator No-Leakage Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- rows: `{m['valid_row_count']}/{m['row_count']}`",
        f"- state feature leak count: `{m['state_feature_leak_count']}`",
        f"- provenance leak count: `{m['provenance_leak_count']}`",
        f"- prediction/outcome exact identity count: `{m['prediction_outcome_exact_identity_count']}`",
        f"- prediction/outcome near-identity rate: `{m['prediction_outcome_near_identity_rate']}`",
        f"- mean prediction/outcome gap: `{m['mean_abs_prediction_outcome_gap']}`",
        f"- best-arm agreement: `{m['best_arm_agreement_rate']}`",
        f"- source direct alias count: `{m['source_direct_prediction_outcome_alias_count']}`",
        "",
        "## Claim Boundary",
        "",
        "The audit supports a leakage-audited triage/router simulator claim only.  It does not permit",
        "replacing live ablation, judge evidence, or external validation.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _scan_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_leaks = []
    provenance_leaks = []
    row_id_leaks = []
    best_selected = []
    checked = 0
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if LEAK_TOKEN_RE.search(row_id):
            row_id_leaks.append({"row_id": row_id, "field": "row_id", "value": row_id})
        state = row.get("state") or {}
        for path, value in _walk(state, prefix="state"):
            text = str(value)
            checked += 1
            if _is_allowed_action_feature(text):
                continue
            match = LEAK_TOKEN_RE.search(text)
            if not match:
                continue
            if "best" in match.group(1).lower() or "selected" in match.group(1).lower():
                best_selected.append({"row_id": row_id, "field": path, "value": text})
            else:
                state_leaks.append({"row_id": row_id, "field": path, "value": text})
        provenance = row.get("provenance") or {}
        for path, value in _walk(provenance, prefix="provenance"):
            text = str(value)
            if _is_allowed_provenance_value(path, text):
                continue
            if LEAK_TOKEN_RE.search(text):
                provenance_leaks.append({"row_id": row_id, "field": path, "value": text})
    return {
        "state_feature_checked_count": checked,
        "state_feature_leak_count": len(state_leaks),
        "provenance_leak_count": len(provenance_leaks),
        "row_id_leak_count": len(row_id_leaks),
        "best_or_selected_token_count": len(best_selected),
        "state_feature_leak_examples": state_leaks[:10],
        "provenance_leak_examples": provenance_leaks[:10],
        "row_id_leak_examples": row_id_leaks[:10],
        "best_or_selected_examples": best_selected[:10],
    }


def _scan_same_state_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leak_keys = []
    for row in rows:
        key = str(row.get("provenance", {}).get("source_row_id", "")).split("::", 1)[0]
        groups[key].append(row)
        if LEAK_TOKEN_RE.search(key):
            leak_keys.append({"row_id": row.get("row_id"), "group_key": key})
    arm_counts = [len({row["action"]["arm"] for row in group}) for group in groups.values()]
    return {
        "same_state_group_count": len(groups),
        "same_state_group_key_leak_count": len(leak_keys),
        "min_arm_count_per_group": min(arm_counts, default=0),
        "max_arm_count_per_group": max(arm_counts, default=0),
        "leak_key_examples": leak_keys[:10],
    }


def _scan_prediction_outcome_independence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = []
    exact_identity = 0
    near_identity = 0
    binary_label_exact = 0
    examples = []
    for row in rows:
        pred = float(row["prediction"]["expected_utility"])
        outcome = float(row["outcome"]["utility_vs_baseline"])
        label = 1.0 if row["outcome"]["accepted"] else 0.0
        gap = abs(pred - outcome)
        gaps.append(gap)
        exact = gap == 0.0
        near = gap <= 0.005
        binary_exact = float(row["prediction"]["p_accept"]) in {0.0, 1.0} and float(row["prediction"]["p_accept"]) == label
        exact_identity += int(exact)
        near_identity += int(near)
        binary_label_exact += int(binary_exact)
        if exact or near or binary_exact:
            examples.append({
                "row_id": row["row_id"],
                "prediction": pred,
                "outcome": outcome,
                "accepted_label": label,
                "gap": round(gap, 6),
            })
    return {
        "prediction_outcome_exact_identity_count": exact_identity,
        "prediction_outcome_near_identity_count": near_identity,
        "prediction_outcome_near_identity_rate": round(near_identity / max(1, len(rows)), 4),
        "mean_abs_prediction_outcome_gap": round(mean(gaps), 4) if gaps else 0.0,
        "prediction_accept_label_exact_count": binary_label_exact,
        "near_identity_examples": examples[:20],
    }


def _scan_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "direct_prediction_outcome_alias_count": 1,
            "oracle_selector_alias_count": 1,
            "forbidden_token_count": 1,
            "matches": [{"kind": "missing_source", "path": str(path)}],
        }
    text = path.read_text(encoding="utf-8")
    direct_patterns = [
        r'"expected_utility"\s*:\s*utility',
        r'"p_accept"\s*:\s*utility',
        r"expected_utility\s*=\s*utility",
        r"p_accept\s*=\s*utility",
    ]
    oracle_patterns = [
        r"selected\s*=\s*empirical_best",
        r"selected_arm\s*=\s*empirical_best",
        r"selected\s*=\s*max\(arm_means",
    ]
    direct = _source_matches(text, direct_patterns, "direct_prediction_outcome_alias")
    oracle = _source_matches(text, oracle_patterns, "oracle_selector_alias")
    forbidden = []
    # Source may contain these words in comments or metric names.  Only count
    # assignments that would encode leakage into row construction.
    for pattern in [
        r'"world_model_features"\s*:\s*\[[^\]]*(accepted|outcome|utility|best|selected|label)',
        r'"active_assumptions"\s*:\s*\[[^\]]*(accepted|outcome|utility|best|selected|label)',
        r'"source_row_id"\s*:\s*f?"[^"]*(accepted|outcome|utility|best|selected|label)',
    ]:
        forbidden.extend(_source_matches(text, [pattern], "forbidden_row_construction_token"))
    return {
        "direct_prediction_outcome_alias_count": len(direct),
        "oracle_selector_alias_count": len(oracle),
        "forbidden_token_count": len(forbidden),
        "matches": [*direct, *oracle, *forbidden][:20],
    }


def _source_matches(text: str, patterns: list[str], kind: str) -> list[dict[str, Any]]:
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            line = text.count("\n", 0, match.start()) + 1
            snippet = text[match.start(): match.end()].replace("\n", " ")[:160]
            matches.append({"kind": kind, "line": line, "pattern": pattern, "snippet": snippet})
    return matches


def _walk(value: Any, *, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for key, item in value.items():
            out.extend(_walk(item, prefix=f"{prefix}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(_walk(item, prefix=f"{prefix}[{index}]"))
        return out
    return [(prefix, value)]


def _is_allowed_action_feature(text: str) -> bool:
    lowered = text.lower()
    return any(lowered.startswith(f"{token}:") for token in ALLOWED_ACTION_TOKENS)


def _is_allowed_provenance_value(path: str, text: str) -> bool:
    lowered = text.lower()
    if path.endswith("source_granularity") and "same_state_multiarm" in lowered:
        return True
    if path.endswith("artifact_id") and "simulator_production_evidence" in lowered:
        return True
    if path.endswith("base_dataset_row_id"):
        return True
    if path.endswith("source_row_id") and not LEAK_TOKEN_RE.search(text.split("::", 1)[0]):
        return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator no-leakage audit artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="simulator_no_leakage_audit_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_no_leakage_audit_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
