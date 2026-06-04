"""Paper claim boundaries for the structural morphism layer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .formal_mapping import build_formal_engine_depth_payload
from .graph_memory import JsonlGraphStore
from .morphism_benchmark import build_morphism_independent_benchmark_payload


DEFAULT_FORMAL_DEPTH_PATH = Path(
    "phase four/assumption_graph/paper_readiness_20260604/formal_engine_depth_audit_20260604.json"
)
DEFAULT_OUT = Path(
    "phase four/assumption_graph/paper_readiness_20260604/morphism_claim_bundle_20260605.json"
)


def build_morphism_claim_bundle_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    eval_id: str | None = None,
    morphism_payload: dict[str, Any] | None = None,
    formal_depth_payload: dict[str, Any] | None = None,
    formal_depth_path: Path | None = None,
) -> dict[str, Any]:
    """Build a bounded, manuscript-safe morphism claim bundle."""

    root = root.resolve()
    eval_id = eval_id or "morphism_claim_bundle_20260605"
    morphism_payload = morphism_payload or build_morphism_independent_benchmark_payload(
        eval_id=f"{eval_id}_morphism",
        neural_embedding_backend="none",
    )
    formal_depth_path = _resolve(root, formal_depth_path or DEFAULT_FORMAL_DEPTH_PATH)
    if formal_depth_payload is None:
        if formal_depth_path.exists():
            formal_depth_payload = _load_json(formal_depth_path)
        else:
            if graph_dir is None:
                raise ValueError("graph_dir is required when formal_depth_payload is not provided")
            formal_depth_payload = build_formal_engine_depth_payload(
                eval_id=f"{eval_id}_formal_depth",
                store=JsonlGraphStore(_resolve(root, graph_dir)),
                morphism_benchmark_payload=morphism_payload,
            )

    evidence = _morphism_evidence_summary(morphism_payload=morphism_payload, formal_depth=formal_depth_payload)
    gates = _claim_gates(evidence=evidence, formal_depth=formal_depth_payload)
    pass_condition = all(gate["pass"] for gate in gates)
    return {
        "eval_id": eval_id,
        "eval_kind": "bounded_structural_morphism_claim_bundle",
        "pass": pass_condition,
        "recommended_short_claim": "category-inspired bounded structural morphism layer",
        "recommended_manuscript_claim": (
            "We implement a category-inspired bounded structural morphism layer: "
            "finite typed objects, morphism/operator labels, composition cues, preserved invariants, "
            "negative controls, and transfer gates for cross-domain analogy retrieval and task routing."
        ),
        "allowed_claims": [
            "The layer retrieves cross-domain structural analogies that KG triples and surface embedding retrieval miss on the benchmark cases.",
            "Preserved invariants, broken-invariant checks, and negative controls reduce analogy hallucination risk.",
            "The bounded formal layer improves downstream transfer and answer-quality probes in the current audit suite.",
        ],
        "forbidden_claims": [
            "complete category-theory theorem prover",
            "general categorical reasoning engine",
            "exact Blackwell order engine",
            "exact Fisher information geometry engine",
            "morphism proves semantic equivalence",
            "morphism guarantees causal transfer",
        ],
        "claim_boundaries": {
            "category_theory_status": "category-inspired, finite, typed structural audit; not theorem proving",
            "blackwell_fisher_status": "entropy/kernel proxies only; not exact Blackwell/Fisher computation",
            "retrieval_scope": "cross-domain structural analogy retrieval and gated downstream prompt/routing support",
            "safety_scope": "negative controls and invariant gates lower false analogy risk but do not eliminate it",
        },
        "evidence": evidence,
        "formal_depth_source": {
            "path": _display_path(root, formal_depth_path),
            "precomputed_used": formal_depth_path.exists(),
        },
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
    }


def _morphism_evidence_summary(*, morphism_payload: dict[str, Any], formal_depth: dict[str, Any]) -> dict[str, Any]:
    rows = morphism_payload.get("rows", [])
    expected_rows = []
    false_candidate_count = 0
    false_with_broken_invariant = 0
    preserved_invariant_count = 0
    expected_broken_invariant_count = 0
    kg_embedding_miss_count = 0
    domain_pairs = Counter()
    for row in rows:
        expected_id = row.get("expected_candidate_id")
        expected = None
        for candidate in row.get("candidate_scores", []):
            if candidate.get("candidate_id") == expected_id:
                expected = candidate
            else:
                false_candidate_count += 1
                evidence = candidate.get("morphism_evidence", {})
                if evidence.get("broken_query_invariants"):
                    false_with_broken_invariant += 1
        if expected is None:
            continue
        evidence = expected.get("morphism_evidence", {})
        preserved_invariant_count += len(evidence.get("invariant_overlap") or [])
        expected_broken_invariant_count += len(evidence.get("broken_query_invariants") or [])
        if row.get("passed_by", {}).get("morphism") and not (
            row.get("passed_by", {}).get("kg_triple") or row.get("passed_by", {}).get("embedding_proxy")
        ):
            kg_embedding_miss_count += 1
        domain_pairs[f"{row.get('query_domain')}->{expected.get('domain')}"] += 1
        expected_rows.append({
            "case_id": row.get("case_id"),
            "query_domain": row.get("query_domain"),
            "expected_candidate_id": expected_id,
            "expected_domain": expected.get("domain"),
            "morphism_score": expected.get("scores", {}).get("morphism"),
            "kg_triple_score": expected.get("scores", {}).get("kg_triple"),
            "embedding_proxy_score": expected.get("scores", {}).get("embedding_proxy"),
            "preserved_invariants": evidence.get("invariant_overlap"),
            "broken_query_invariants": evidence.get("broken_query_invariants"),
            "nonlexical_structural_success": row.get("nonlexical_structural_success"),
        })

    scorer_rates = morphism_payload.get("scorer_hit_rates", {})
    formal_summary = formal_depth.get("summary", {})
    answer_quality = formal_depth.get("answer_quality", {})
    downstream_transfer = formal_depth.get("downstream_transfer", {})
    negative_control_count = int(formal_summary.get("negative_control_application_count") or 0)
    return {
        "cross_domain_retrieval": {
            "case_count": morphism_payload.get("case_count"),
            "scorer_hit_rates": scorer_rates,
            "morphism_margin_over_best_baseline": morphism_payload.get("morphism_margin_over_best_baseline"),
            "nonlexical_success_rate": morphism_payload.get("nonlexical_success_rate"),
            "kg_embedding_miss_count": kg_embedding_miss_count,
            "domain_pair_counts": dict(domain_pairs),
        },
        "invariant_and_negative_control_checks": {
            "expected_candidate_count": len(expected_rows),
            "preserved_invariant_count": preserved_invariant_count,
            "expected_broken_invariant_count": expected_broken_invariant_count,
            "false_candidate_count": false_candidate_count,
            "false_candidate_with_broken_invariant_count": false_with_broken_invariant,
            "formal_negative_control_application_count": negative_control_count,
            "expected_rows": expected_rows,
        },
        "downstream_effect": {
            "downstream_transfer_auc": downstream_transfer.get("pairwise_auc"),
            "downstream_positive_mean_transfer_score": downstream_transfer.get("positive_mean_transfer_score"),
            "downstream_negative_mean_transfer_score": downstream_transfer.get("negative_mean_transfer_score"),
            "answer_quality_mean_delta": answer_quality.get("mean_delta"),
            "answer_quality_guided_win_rate": answer_quality.get("guided_win_rate"),
            "answer_quality_probe_count": answer_quality.get("probe_count"),
        },
        "scope_flags": {
            "bounded_formal_engine_depth_pass": formal_depth.get("bounded_formal_engine_depth_pass"),
            "strict_category_theory_theorem_prover": formal_depth.get("strict_category_theory_theorem_prover"),
            "true_blackwell_or_fisher_engine": formal_depth.get("true_blackwell_or_fisher_engine"),
            "scope_note": formal_depth.get("scope_note"),
        },
    }


def _claim_gates(*, evidence: dict[str, Any], formal_depth: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = evidence.get("cross_domain_retrieval", {})
    invariant = evidence.get("invariant_and_negative_control_checks", {})
    downstream = evidence.get("downstream_effect", {})
    flags = evidence.get("scope_flags", {})
    rates = retrieval.get("scorer_hit_rates", {})
    return [
        {
            "gate": "bounded_claim_scope",
            "pass": (
                flags.get("bounded_formal_engine_depth_pass") is True
                and flags.get("strict_category_theory_theorem_prover") is False
                and flags.get("true_blackwell_or_fisher_engine") is False
            ),
            "observed": flags,
        },
        {
            "gate": "kg_embedding_miss_cross_domain_structure",
            "pass": (
                float(rates.get("morphism") or 0.0) >= 0.80
                and float(retrieval.get("morphism_margin_over_best_baseline") or 0.0) >= 0.20
                and int(retrieval.get("kg_embedding_miss_count") or 0) >= 7
            ),
            "observed": retrieval,
        },
        {
            "gate": "invariants_and_negative_controls",
            "pass": (
                int(invariant.get("expected_candidate_count") or 0) >= 8
                and int(invariant.get("preserved_invariant_count") or 0) >= 8
                and int(invariant.get("expected_broken_invariant_count") or 0) == 0
                and int(invariant.get("formal_negative_control_application_count") or 0) >= 200
            ),
            "observed": {
                key: value
                for key, value in invariant.items()
                if key != "expected_rows"
            },
        },
        {
            "gate": "downstream_transfer_and_answer_quality",
            "pass": (
                float(downstream.get("downstream_transfer_auc") or 0.0) >= 0.90
                and float(downstream.get("answer_quality_mean_delta") or 0.0) >= 0.35
                and float(downstream.get("answer_quality_guided_win_rate") or 0.0) >= 0.80
            ),
            "observed": downstream,
        },
        {
            "gate": "formal_depth_audit_passes",
            "pass": bool(formal_depth.get("pass")),
            "observed": {
                "pass": formal_depth.get("pass"),
                "failed_gates": [gate.get("gate") for gate in formal_depth.get("gates", []) if not gate.get("pass")],
            },
        },
    ]


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
    ap = argparse.ArgumentParser(description="Build the bounded morphism manuscript claim bundle.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--formal-depth", default=str(DEFAULT_FORMAL_DEPTH_PATH))
    ap.add_argument("--eval-id", default="morphism_claim_bundle_20260605")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_morphism_claim_bundle_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        formal_depth_path=Path(args.formal_depth),
        eval_id=args.eval_id,
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "recommended_short_claim": payload["recommended_short_claim"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
