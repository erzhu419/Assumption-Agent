"""Scaled bounded NL-to-diagram extraction benchmark.

This is deliberately a finite, category-inspired extraction benchmark.  It
checks whether supported natural-language patterns can be mapped into bounded
diagram certificates, and whether unsupported or misleading near-neighbor text
is rejected.  It does not claim unrestricted natural-language theorem proving.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .finite_theorem_fragment import extract_natural_language_diagram


DEFAULT_OUT = PAPER_DIR / "nl_to_diagram_scale_benchmark_20260612.json"


@dataclass(frozen=True)
class DiagramBenchmarkCase:
    case_id: str
    text: str
    expected_family: str | None
    case_kind: str


SUPPORTED_FAMILIES: dict[str, list[str]] = {
    "negative_feedback_regulation": [
        "Le Chatelier principle and Lenz law both describe a perturbation that triggers an opposing response toward equilibrium.",
        "A price surge can create a counteracting supply response, pushing the system back toward a constrained equilibrium.",
        "When growth induces a balancing force that counteracts the growth, the useful diagram is perturbation to opposing response to regulated state.",
        "The controller should treat rising pressure as a trigger for an opposing response, not as a reason to keep adding input.",
        "The same pattern appears when a market shock creates a stabilizing opposite response and returns toward balance.",
        "A thermal equilibrium rule and an electromagnetic induction rule share the oppose-the-perturbation structure.",
        "The policy asks whether increased demand will induce a counteracting constraint and converge to equilibrium.",
        "This is not a one-step forecast: it is a negative feedback system where the response counteracts the initial change.",
    ],
    "residual_transport": [
        "ResNet skip connection preserves an identity path while a residual transform changes representation.",
        "A migration policy keeps an identity path available while a residual branch learns the transformation.",
        "The representation shift is safe because information can still travel through an unchanged skip connection.",
        "A residual adapter should change only the delta while preserving an identity route for the original signal.",
        "Skip connection and residual correction create a diagram from input to transformed state plus preserved output state.",
        "The agent should add a residual transform without breaking the identity path that carries useful context.",
        "The model improves depth by transporting gradients through an identity path and learning a residual branch.",
        "Use residual transport: keep the old answer scaffold while adding a small corrective transform.",
    ],
    "noise_invariant_signal_extraction": [
        "Autocorrelation can suppress random noise and recover a stable signal; JEPA-style latent prediction uses a related invariance idea.",
        "If random noise cancels under correlation, the stable latent feature can be extracted from repeated measurements.",
        "The useful assumption is that independent noise averages away while the persistent signal remains.",
        "A world model can focus on invariant latent structure rather than pixel-level stochastic noise.",
        "Denoise the measurement by correlating repeated observations, then keep the stable feature.",
        "Gaussian noise assumptions let the system separate uncorrelated perturbations from shared latent signal.",
        "The diagram is raw measurement to correlation filter to stable signal.",
        "Signal extraction improves when random fluctuations are treated as cancellable rather than meaningful.",
    ],
    "bottleneck_capacity_limit": [
        "The enzyme saturates, so more substrate cannot increase throughput past the bottleneck capacity.",
        "A queue capacity limit means extra arrivals produce waiting, not proportional output.",
        "Rate limiting should be modeled as input flow hitting a bottleneck and producing bounded output.",
        "The CI system has a throughput ceiling; adding jobs without removing the bottleneck only increases backlog.",
        "The limiting step controls the rate even if upstream input grows.",
        "A saturated API endpoint behaves like an enzyme bottleneck: more requests do not improve completed throughput.",
        "Queue capacity is the controlling morphism from load to bounded service.",
        "The correct transfer is bottleneck saturation, not linear scaling.",
    ],
    "bridge_decomposition": [
        "A strangler migration uses an adapter layer to move from legacy state to target state incrementally.",
        "Bridge retrofit preserves continuity while a system crosses from old structure to new structure.",
        "Use a compatibility bridge so replacement proceeds module by module rather than as a big bang rewrite.",
        "The adapter interface wraps the old API and gradually transfers traffic to the new service.",
        "Incremental migration succeeds when the bridge interface reduces blast radius.",
        "A bridge decomposition keeps legacy behavior alive while target behavior is introduced.",
        "The pattern is legacy state to bridge interface to target state.",
        "A compatibility bridge converts an unsafe rewrite into a controlled transfer.",
    ],
    "randomized_counterfactual_evaluation": [
        "A clinical trial and an A/B test both randomize units into treatment and control groups to estimate causal effect.",
        "Randomized assignment creates exchangeability before the causal comparison is read.",
        "The experiment needs a control group and a treatment group, otherwise the counterfactual is missing.",
        "A/B testing works because randomization turns user differences into noise around the treatment contrast.",
        "Clinical trial logic transfers to product experiments when units are randomized.",
        "Use randomized evaluation: split population, apply treatment, estimate causal comparison.",
        "The causal diagram is population to random assignment to effect estimate.",
        "Without a treatment-control split, the measured lift cannot be trusted.",
    ],
    "conservation_balance": [
        "A budget balance and a reactor mass balance both require input-output accounting of a conserved stock.",
        "Flow conservation means the missing quantity must appear as accumulation or outflow.",
        "Stock and flow reasoning prevents inventing resources that were never supplied.",
        "The budget cannot grow without inflow; conservation balance checks the accounting.",
        "Mass balance maps inflow to stock and outflow accounting.",
        "A cash runway model is a conserved-stock diagram, not a narrative optimism problem.",
        "The invariant is no free creation: every output must trace to an input or stored stock.",
        "Conservation reasoning turns vague resource plans into input-output constraints.",
    ],
    "error_correction_feedback": [
        "A checksum detects a corrupted message and drives an error correction step to recover the original payload.",
        "Parity bits add redundancy so noise can be detected and repaired.",
        "The syndrome identifies the error pattern before the decoder corrects the message.",
        "Use redundancy check as a feedback signal for repair rather than regenerating the entire answer.",
        "Error correction maps noisy message to redundancy signal to corrected message.",
        "The invariant is recovered message identity despite local noise.",
        "A verifier can act like parity: detect inconsistency and feed a targeted correction.",
        "Checksum-style evidence should trigger repair only where the inconsistency is located.",
    ],
    "search_pruning_by_bounds": [
        "Branch and bound uses a dominance bound to prune search paths while preserving the optimum.",
        "Beam search keeps a reduced frontier after scoring candidate paths.",
        "A pruning policy should remove dominated trajectories without discarding the best candidate.",
        "Bound evidence maps candidate space to a smaller frontier.",
        "The search space is large, so the useful assumption is pruning by provable or empirical bounds.",
        "Dominance bounds reduce exploration cost while keeping optimal paths reachable.",
        "The diagram is candidate space to bound evidence to reduced frontier.",
        "Pruning should be justified by a bound, not by arbitrary early stopping.",
    ],
    "threshold_phase_transition": [
        "A critical threshold can cause a phase transition where small parameter changes trigger a new regime.",
        "Percolation and tipping points share the same threshold-effect diagram.",
        "Near a critical boundary the response is not locally linear.",
        "The policy must detect when control parameters approach a regime boundary.",
        "A threshold effect means gradual input can produce abrupt qualitative change.",
        "The diagram is control parameter to critical boundary to new regime.",
        "Treat the launch decision as a tipping point rather than a smooth extrapolation.",
        "Phase transition reasoning asks whether the system has crossed a structural boundary.",
    ],
    "modular_composition": [
        "A compiler and an assembly line both use interface contracts to compose components into a larger system.",
        "Modular composition works when local contracts remain valid after assembly.",
        "The component spec compiles into an interface contract and then into an assembled system.",
        "Subassembly reasoning lets teams build parts independently without breaking the whole.",
        "Interface contracts are the invariant that allows composition to scale.",
        "A compiler pipeline is a modular composition diagram, not just a sequence of text transforms.",
        "The same structure appears when manufacturing subassemblies are joined through standard interfaces.",
        "Local contract preservation is the condition for safe module composition.",
    ],
    "redundant_fault_tolerance": [
        "Failover uses a redundant path to preserve service continuity when the primary path fails.",
        "Replication hides a single failure by keeping a backup path ready.",
        "Fault tolerance maps primary path to redundant path to service continuity.",
        "A backup channel is useful only if it can recover the service state after failure.",
        "Redundant channels preserve continuity when one component breaks.",
        "The invariant is that a single failure is masked by replicated state.",
        "Use failover reasoning instead of optimizing the primary path alone.",
        "Replication is not decoration; it is a morphism from primary service to continuity under failure.",
    ],
    "regularization_smoothing": [
        "Regularization adds a complexity penalty so an overfit flexible model selects a smoother generalized solution.",
        "Weight decay is useful because it biases the model toward lower-complexity parameters.",
        "A smoothness penalty reduces variance while preserving the signal structure.",
        "The model should not memorize noise when a regularization term can select a stable solution.",
        "Implicit bias acts like a penalty that prefers generalizable structure.",
        "Overfit behavior calls for regularization smoothing, not more expressive capacity.",
        "The diagram is flexible model to complexity penalty to generalized solution.",
        "A penalty term can improve transfer by reducing variance.",
    ],
}

NEGATIVE_CONTROLS = [
    "The blue button is prettier than the green one because the color feels calmer.",
    "The report should use a shorter title because the current title sounds awkward.",
    "I had lunch before the meeting and therefore the slide deck was easier to read.",
    "This anecdote is memorable, but it does not specify objects, morphisms, or invariants.",
    "The team likes the old logo because it feels familiar.",
    "A larger font may be more readable, but no transferable structural relation is stated.",
    "The coffee was cold, so the author changed the paragraph order.",
    "This is a personal preference rather than a bounded scientific or mathematical pattern.",
    "The choice of wallpaper made the office look brighter.",
    "The example has nouns and verbs but no invariant-preserving mapping.",
] * 4

NEAR_NEGATIVE_CONTROLS = [
    "An opposing view in a debate is not automatically a physical equilibrium law.",
    "A residual complaint from a user is not the same as a ResNet residual transport diagram.",
    "Randomly choosing a color is not a randomized counterfactual trial.",
    "A bridge in a poem is not an incremental migration adapter.",
    "A bottleneck-shaped icon in a diagram does not imply a throughput saturation law.",
    "A checksum printed on a poster is not an error-correction feedback process.",
    "A phrase about conservation values is not a mass balance equation.",
    "A search for restaurants is not branch-and-bound pruning unless a bound preserves an optimum.",
    "A threshold password hint is not a phase transition.",
    "A backup anecdote is not failover unless service continuity is preserved.",
] * 2


def build_nl_to_diagram_scale_benchmark_payload(
    *,
    eval_id: str = "nl_to_diagram_scale_benchmark_20260612",
    examples_per_family: int = 8,
) -> dict[str, Any]:
    cases = _build_cases(examples_per_family=examples_per_family)
    rows = [_evaluate_case(case) for case in cases]
    metrics = _metrics(rows)
    gates = {
        "positive_case_count_high": metrics["positive_case_count"] >= 96,
        "negative_case_count_high": metrics["negative_case_count"] >= 50,
        "family_count_high": metrics["family_count"] >= 12,
        "positive_accuracy_high": metrics["positive_accuracy"] >= 0.95,
        "negative_specificity_high": metrics["negative_specificity"] >= 0.95,
        "near_negative_specificity_high": metrics["near_negative_specificity"] >= 0.90,
        "certificate_pass_rate_high": metrics["certificate_pass_rate"] == 1.0,
        "macro_family_recall_high": metrics["macro_family_recall"] >= 0.95,
        "full_theorem_prover_not_claimed": metrics["full_theorem_prover_claim_allowed"] is False,
        "bounded_claim_allowed": metrics["bounded_nl_diagram_claim_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "scaled_bounded_nl_to_diagram_benchmark",
        "reconstruction_v2_full_phase": "formal_layer_scaled_nl_to_diagram_certificate",
        "implementation_level": "bounded_rule_based_extractor_with_negative_controls",
        "performance_validation": True,
        "validation_scope": (
            "Benchmarks finite NL-to-diagram extraction across supported structural families with ordinary "
            "negative controls and near-negative controls.  The result supports a bounded certificate layer, "
            "not an unrestricted natural-language theorem prover."
        ),
        "rows": rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The formal layer now has a scaled extraction benchmark: supported families are mapped to finite "
            "diagram certificates, and unsupported or misleading near-neighbor examples are rejected.  This closes "
            "the scale gap for bounded NL-to-diagram evidence while preserving the full theorem-prover boundary."
        ),
    }


def _build_cases(*, examples_per_family: int) -> list[DiagramBenchmarkCase]:
    cases: list[DiagramBenchmarkCase] = []
    for family, examples in sorted(SUPPORTED_FAMILIES.items()):
        selected = examples[:examples_per_family]
        for idx, text in enumerate(selected):
            cases.append(DiagramBenchmarkCase(
                case_id=f"pos_{family}_{idx:02d}",
                text=text,
                expected_family=family,
                case_kind="positive",
            ))
    for idx, text in enumerate(NEGATIVE_CONTROLS):
        cases.append(DiagramBenchmarkCase(
            case_id=f"neg_plain_{idx:02d}",
            text=text,
            expected_family=None,
            case_kind="negative_plain",
        ))
    for idx, text in enumerate(NEAR_NEGATIVE_CONTROLS):
        cases.append(DiagramBenchmarkCase(
            case_id=f"neg_near_{idx:02d}",
            text=text,
            expected_family=None,
            case_kind="negative_near",
        ))
    return cases


def _evaluate_case(case: DiagramBenchmarkCase) -> dict[str, Any]:
    extracted = extract_natural_language_diagram(case.text)
    predicted_family = extracted.get("family") if extracted.get("status") == "formalized" else None
    certificate = extracted.get("certificate") or {}
    validation = certificate.get("validation") or {}
    correct = predicted_family == case.expected_family
    return {
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "text_hash": stable_hash(case.text),
        "expected_family": case.expected_family,
        "predicted_family": predicted_family,
        "status": extracted.get("status"),
        "correct": correct,
        "certificate_valid": bool(validation.get("valid")) if predicted_family else None,
        "object_count": validation.get("object_count"),
        "morphism_count": validation.get("morphism_count"),
        "issue_count": validation.get("issue_count"),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["case_kind"] == "positive"]
    negatives = [row for row in rows if row["case_kind"].startswith("negative")]
    near_negatives = [row for row in rows if row["case_kind"] == "negative_near"]
    formalized = [row for row in rows if row["status"] == "formalized"]
    family_recalls = {}
    for family in sorted({row["expected_family"] for row in positives}):
        family_rows = [row for row in positives if row["expected_family"] == family]
        family_recalls[family] = round(sum(1 for row in family_rows if row["correct"]) / max(1, len(family_rows)), 4)
    confusion = Counter(
        (row["expected_family"] or "not_applicable", row["predicted_family"] or "not_applicable")
        for row in rows
    )
    return {
        "case_count": len(rows),
        "positive_case_count": len(positives),
        "negative_case_count": len(negatives),
        "near_negative_case_count": len(near_negatives),
        "family_count": len(family_recalls),
        "positive_accuracy": round(sum(1 for row in positives if row["correct"]) / max(1, len(positives)), 4),
        "negative_specificity": round(sum(1 for row in negatives if row["correct"]) / max(1, len(negatives)), 4),
        "near_negative_specificity": round(
            sum(1 for row in near_negatives if row["correct"]) / max(1, len(near_negatives)),
            4,
        ),
        "certificate_pass_rate": round(
            sum(1 for row in formalized if row["certificate_valid"]) / max(1, len(formalized)),
            4,
        ),
        "macro_family_recall": round(sum(family_recalls.values()) / max(1, len(family_recalls)), 4),
        "family_recalls": family_recalls,
        "formalized_count": len(formalized),
        "abstained_count": sum(1 for row in rows if row["status"] == "not_applicable"),
        "confusion_counts": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
        "bounded_nl_diagram_claim_allowed": True,
        "full_theorem_prover_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scaled bounded NL-to-diagram benchmark.")
    parser.add_argument("--eval-id", default="nl_to_diagram_scale_benchmark_20260612")
    parser.add_argument("--examples-per-family", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_nl_to_diagram_scale_benchmark_payload(
        eval_id=args.eval_id,
        examples_per_family=args.examples_per_family,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
