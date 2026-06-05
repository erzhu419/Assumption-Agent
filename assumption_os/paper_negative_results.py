"""Negative-result and boundary-condition audit for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STRUCTURAL_LIVE_DIR = Path("phase four/assumption_graph/structural_live_ablation_20260603")
PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_MAIN_EXPERIMENT = PAPER_DIR / "paper_main_experiment_20260605.json"
DEFAULT_MORPHISM_CLAIMS = PAPER_DIR / "morphism_claim_bundle_20260605.json"
DEFAULT_OUT = PAPER_DIR / "paper_negative_results_20260605.json"

FAILURE_SOURCES = {
    "bottleneck_first_margin_failure": STRUCTURAL_LIVE_DIR / "structural_live_bottleneck_margin_v1_gpt54mini_gpt55_20260604_summary.json",
    "signal_first_repair_failure": STRUCTURAL_LIVE_DIR / "structural_live_signal_repair_v1_gpt54mini_gpt55_20260603_summary.json",
    "natural_one_shot_failure": STRUCTURAL_LIVE_DIR / "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json",
}


def build_paper_negative_results_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    main_experiment_path: Path | None = None,
    morphism_claim_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    main_path = _resolve(root, main_experiment_path or DEFAULT_MAIN_EXPERIMENT)
    morphism_path = _resolve(root, morphism_claim_path or DEFAULT_MORPHISM_CLAIMS)
    main = _load_json(main_path)
    morphism = _load_json(morphism_path)
    domain_boundaries = _domain_boundaries(main)
    pattern_boundaries = _pattern_boundaries(root=root, main=main)
    historical_failures = _historical_failure_rows(root=root)
    formal_boundaries = _formal_boundaries(morphism)
    abstain_policy = _abstain_policy(domain_boundaries=domain_boundaries, pattern_boundaries=pattern_boundaries)
    gates = [
        {
            "gate": "math_science_boundary_recorded",
            "pass": any(row["domain"] == "science" and row["utility_vs_raw"] < 0.50 for row in domain_boundaries)
            and any(row["domain"] == "mathematics" for row in domain_boundaries),
            "observed": domain_boundaries,
        },
        {
            "gate": "repair_failures_recorded",
            "pass": all(name in {row["failure_id"] for row in historical_failures} for name in [
                "bottleneck_first_margin_failure",
                "signal_first_repair_failure",
            ]),
            "observed": historical_failures,
        },
        {
            "gate": "formal_overstructure_boundary_recorded",
            "pass": bool(formal_boundaries.get("forbidden_claims"))
            and formal_boundaries.get("strict_category_theory_theorem_prover") is False,
            "observed": formal_boundaries,
        },
        {
            "gate": "abstain_or_gate_recommendations_present",
            "pass": len(abstain_policy) >= 4,
            "observed": abstain_policy,
        },
    ]
    return {
        "eval_id": eval_id or "paper_negative_results_20260605",
        "eval_kind": "paper_negative_results_and_boundary_audit",
        "pass": all(gate["pass"] for gate in gates),
        "sources": {
            "main_experiment": _display_path(root, main_path),
            "morphism_claim_bundle": _display_path(root, morphism_path),
            "historical_failure_summaries": {
                name: _display_path(root, _resolve(root, path))
                for name, path in FAILURE_SOURCES.items()
            },
        },
        "domain_boundaries": domain_boundaries,
        "pattern_boundaries": pattern_boundaries,
        "historical_repair_failures": historical_failures,
        "formal_layer_boundaries": formal_boundaries,
        "abstain_or_gate_policy": abstain_policy,
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
    }


def _domain_boundaries(main: dict[str, Any]) -> list[dict[str, Any]]:
    base = main.get("main_results", {}).get("structural_vs_base", {}).get("domain_breakdown", {})
    placebo = main.get("main_results", {}).get("structural_vs_placebo", {}).get("domain_breakdown", {})
    rows = []
    for domain in sorted(base):
        b = base.get(domain, {})
        p = placebo.get(domain, {})
        risk = []
        if float(b.get("utility") or 0.0) < 0.55:
            risk.append("weak_vs_raw")
        if float(p.get("utility") or 0.0) < 0.60:
            risk.append("weak_vs_placebo")
        rows.append({
            "domain": domain,
            "n": b.get("n"),
            "utility_vs_raw": b.get("utility"),
            "utility_vs_placebo": p.get("utility"),
            "outcomes_vs_raw": b.get("outcomes"),
            "outcomes_vs_placebo": p.get("outcomes"),
            "boundary_tags": risk,
            "paper_interpretation": (
                "Retain as a boundary condition; do not claim uniform domain improvement."
                if risk else "No boundary flag under current thresholds."
            ),
        })
    return rows


def _pattern_boundaries(*, root: Path, main: dict[str, Any]) -> list[dict[str, Any]]:
    source = _resolve(root, STRUCTURAL_LIVE_DIR / "structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_summary.json")
    summary = _load_json(source)
    base_patterns = summary.get("pair_summaries", {}).get("structural_vs_base", {}).get("by_pattern", {})
    placebo_patterns = summary.get("pair_summaries", {}).get("structural_vs_placebo", {}).get("by_pattern", {})
    rows = []
    for pattern in sorted(base_patterns):
        b = base_patterns.get(pattern, {})
        p = placebo_patterns.get(pattern, {})
        tags = []
        if int(b.get("n") or 0) < 5:
            tags.append("low_sample")
        if float(b.get("utility") or 0.0) < 0.50:
            tags.append("weak_vs_raw")
        if float(p.get("utility") or 0.0) < 0.55:
            tags.append("weak_vs_placebo")
        rows.append({
            "pattern_id": pattern,
            "n": b.get("n"),
            "utility_vs_raw": b.get("utility"),
            "utility_vs_placebo": p.get("utility"),
            "outcomes_vs_raw": b.get("outcomes"),
            "outcomes_vs_placebo": p.get("outcomes"),
            "boundary_tags": tags,
            "recommended_policy": _policy_from_tags(tags),
        })
    return rows


def _historical_failure_rows(*, root: Path) -> list[dict[str, Any]]:
    rows = []
    for failure_id, rel_path in FAILURE_SOURCES.items():
        path = _resolve(root, rel_path)
        summary = _load_json(path)
        pairs = summary.get("pair_summaries", {})
        base = pairs.get("structural_vs_base", {})
        placebo = pairs.get("structural_vs_placebo", {})
        rows.append({
            "failure_id": failure_id,
            "source": _display_path(root, path),
            "pass": bool(summary.get("pass")),
            "n": base.get("n"),
            "utility_vs_raw": base.get("utility"),
            "utility_vs_placebo": placebo.get("utility"),
            "outcomes_vs_raw": base.get("outcomes"),
            "outcomes_vs_placebo": placebo.get("outcomes"),
            "retained_reason": (
                "Boundary/negative result retained for manuscript credibility and future gating."
            ),
        })
    return rows


def _formal_boundaries(morphism: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_short_claim": morphism.get("recommended_short_claim"),
        "forbidden_claims": morphism.get("forbidden_claims"),
        "strict_category_theory_theorem_prover": morphism.get("evidence", {}).get("scope_flags", {}).get(
            "strict_category_theory_theorem_prover"
        ),
        "true_blackwell_or_fisher_engine": morphism.get("evidence", {}).get("scope_flags", {}).get(
            "true_blackwell_or_fisher_engine"
        ),
        "overstructure_risk": (
            "The formal layer can over-structure answers; require invariants, negative controls, "
            "and downstream behavior gain before promotion."
        ),
    }


def _abstain_policy(*, domain_boundaries: list[dict[str, Any]], pattern_boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in domain_boundaries:
        if row["boundary_tags"]:
            rows.append({
                "scope": f"domain::{row['domain']}",
                "boundary_tags": row["boundary_tags"],
                "policy": "conditioned_activation_only",
                "reason": "Domain-level utility is weak on at least one frozen comparison.",
            })
    for row in pattern_boundaries:
        if row["boundary_tags"]:
            rows.append({
                "scope": f"pattern::{row['pattern_id']}",
                "boundary_tags": row["boundary_tags"],
                "policy": row["recommended_policy"],
                "reason": "Pattern-level frozen comparison is weak or under-sampled.",
            })
    rows.append({
        "scope": "formal_layer",
        "boundary_tags": ["overstructure_risk"],
        "policy": "require_invariant_negative_control_and_behavior_gain",
        "reason": "Bounded morphism is useful but not a theorem prover or semantic-equivalence proof.",
    })
    return rows


def _policy_from_tags(tags: list[str]) -> str:
    if "weak_vs_raw" in tags:
        return "default_off_until_repaired"
    if "weak_vs_placebo" in tags:
        return "gated_off_against_placebo"
    if "low_sample" in tags:
        return "needs_more_sample_before_claim"
    return "allowed_under_current_gate"


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
    ap = argparse.ArgumentParser(description="Build paper negative results and boundary audit.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--main-experiment", default=str(DEFAULT_MAIN_EXPERIMENT))
    ap.add_argument("--morphism-claim", default=str(DEFAULT_MORPHISM_CLAIMS))
    ap.add_argument("--eval-id", default="paper_negative_results_20260605")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_negative_results_payload(
        root=root,
        eval_id=args.eval_id,
        main_experiment_path=Path(args.main_experiment),
        morphism_claim_path=Path(args.morphism_claim),
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
