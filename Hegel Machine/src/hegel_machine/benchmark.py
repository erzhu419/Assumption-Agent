"""Deterministic Phase-2 benchmark with anti-semantic controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .hashing import stable_hash
from .laws import LawEvaluation, evaluate_law
from .schema import LawKind


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    kind: LawKind
    episode: Mapping[str, Any]
    relation_present: bool
    semantic_overlap: float
    control: str


def controlled_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            "symmetry_low_semantic_positive",
            LawKind.SYMMETRY,
            {
                "forward": (2.0, 4.0),
                "transformed": (2.0, 4.0),
                "common_codomains": True,
                "entity_names": ("xq-17", "violet-gear"),
            },
            True,
            0.05,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "symmetry_high_semantic_negative",
            LawKind.SYMMETRY,
            {
                "forward": (2.0, 4.0),
                "transformed": (2.0, 5.0),
                "common_codomains": True,
                "entity_names": ("perfect symmetry", "mirror invariance"),
            },
            False,
            0.95,
            "high_semantic_negative",
        ),
        BenchmarkCase(
            "monotonicity_low_semantic_positive",
            LawKind.MONOTONICITY,
            {
                "x_low": 1.0,
                "x_high": 2.0,
                "y_low": 3.0,
                "y_high": 5.0,
                "direction": 1.0,
                "entity_names": ("blue", "quartz"),
            },
            True,
            0.04,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "monotonicity_high_semantic_negative",
            LawKind.MONOTONICITY,
            {
                "x_low": 1.0,
                "x_high": 2.0,
                "y_low": 5.0,
                "y_high": 3.0,
                "direction": 1.0,
                "entity_names": ("increasing order", "monotone growth"),
            },
            False,
            0.96,
            "sign_flip",
        ),
        BenchmarkCase(
            "conservation_low_semantic_positive",
            LawKind.CONSERVATION,
            {
                "storage_delta": 1.0,
                "inflows": (10.0,),
                "outflows": (9.0,),
                "sources": (),
                "sinks": (),
                "boundary_observed": True,
                "entity_names": ("kappa", "station-9"),
            },
            True,
            0.03,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "conservation_high_semantic_negative",
            LawKind.CONSERVATION,
            {
                "storage_delta": 1.0,
                "inflows": (10.0,),
                "outflows": (7.0,),
                "sources": (),
                "sinks": (),
                "boundary_observed": True,
                "entity_names": ("mass balance", "closed conservation"),
            },
            False,
            0.97,
            "hidden_sink",
        ),
        BenchmarkCase(
            "complementarity_low_semantic_positive",
            LawKind.COMPLEMENTARITY,
            {
                "u_empty": 0.0,
                "u_a": 1.0,
                "u_b": 1.0,
                "u_ab": 3.0,
                "expected_interaction": 1.0,
                "interaction_margin": 0.5,
                "entity_names": ("r17", "tulip"),
            },
            True,
            0.02,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "complementarity_high_semantic_negative",
            LawKind.COMPLEMENTARITY,
            {
                "u_empty": 0.0,
                "u_a": 1.0,
                "u_b": 1.0,
                "u_ab": 1.0,
                "expected_interaction": 1.0,
                "interaction_margin": 0.5,
                "entity_names": ("synergistic pair", "perfect complement"),
            },
            False,
            0.98,
            "interaction_sign_flip",
        ),
        BenchmarkCase(
            "feedback_low_semantic_positive",
            LawKind.NEGATIVE_FEEDBACK,
            {
                "disturbance_delta": 2.0,
                "response_delta": -1.0,
                "deviation_before_response": 2.0,
                "deviation_after_response": 1.0,
                "controlled_quantity_observed": True,
                "disturbance_precedes_response": True,
                "system_induced_response": True,
                "same_controlled_quantity": True,
                "local_stability_window_observed": True,
                "response_margin": 0.5,
                "mitigation_margin": 0.5,
                "entity_names": ("omega", "coil-7"),
            },
            True,
            0.06,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "feedback_high_semantic_negative",
            LawKind.NEGATIVE_FEEDBACK,
            {
                "disturbance_delta": 2.0,
                "response_delta": 1.0,
                "deviation_before_response": 2.0,
                "deviation_after_response": 3.0,
                "controlled_quantity_observed": True,
                "disturbance_precedes_response": True,
                "system_induced_response": True,
                "same_controlled_quantity": True,
                "local_stability_window_observed": True,
                "response_margin": 0.5,
                "mitigation_margin": 0.5,
                "entity_names": ("negative feedback", "restoring response"),
            },
            False,
            0.94,
            "role_or_sign_flip",
        ),
        BenchmarkCase(
            "locality_low_semantic_positive",
            LawKind.LOCALITY,
            {
                "conditional_a": (0.7, 0.3),
                "conditional_b": (0.7, 0.3),
                "blanket_observed": True,
                "same_blanket_state": True,
                "entity_names": ("room-2", "amber"),
            },
            True,
            0.01,
            "low_semantic_positive",
        ),
        BenchmarkCase(
            "locality_high_semantic_negative",
            LawKind.LOCALITY,
            {
                "conditional_a": (0.7, 0.3),
                "conditional_b": (0.2, 0.8),
                "blanket_observed": True,
                "same_blanket_state": True,
                "entity_names": ("Markov locality", "irrelevant outside context"),
            },
            False,
            0.99,
            "outside_context_effect",
        ),
    )


def recognize_laws(
    episode: Mapping[str, Any], *, tolerance: float = 0.01
) -> tuple[LawEvaluation, ...]:
    """Evaluate the whole frozen library; missing schemas abstain."""

    completed = [
        evaluate_law(kind, episode, tolerance=tolerance) for kind in LawKind
    ]
    return tuple(item for item in completed if item.passed and not item.abstained)


def run_phase2_benchmark() -> dict[str, Any]:
    cases = controlled_cases()
    records: list[dict[str, Any]] = []
    correct = 0
    semantic_correct = 0
    positive_total = 0
    positive_correct = 0
    negative_total = 0
    negative_correct = 0
    rename_invariant = True

    for case in cases:
        recognized = recognize_laws(case.episode)
        recognized_kinds = tuple(sorted(item.kind.value for item in recognized))
        structural_prediction = case.kind.value in recognized_kinds
        case_correct = structural_prediction == case.relation_present
        correct += int(case_correct)
        semantic_prediction = case.semantic_overlap >= 0.5
        semantic_correct += int(semantic_prediction == case.relation_present)
        if case.relation_present:
            positive_total += 1
            positive_correct += int(case_correct)
        else:
            negative_total += 1
            negative_correct += int(case_correct)

        renamed = dict(case.episode)
        renamed["entity_names"] = ("entity_A", "entity_B")
        renamed_kinds = tuple(
            sorted(item.kind.value for item in recognize_laws(renamed))
        )
        rename_invariant = rename_invariant and renamed_kinds == recognized_kinds
        records.append(
            {
                "case_id": case.case_id,
                "expected_relation": case.relation_present,
                "expected_kind": case.kind.value,
                "recognized_kinds": recognized_kinds,
                "structural_correct": case_correct,
                "semantic_only_prediction": semantic_prediction,
                "semantic_overlap": case.semantic_overlap,
                "control": case.control,
            }
        )

    incomplete_boundary = {
        "storage_delta": 1.0,
        "inflows": (10.0,),
        "outflows": (9.0,),
        "sources": (),
        "sinks": (),
        "boundary_observed": False,
    }
    boundary_result = evaluate_law(
        LawKind.CONSERVATION, incomplete_boundary, tolerance=0.01
    )
    report: dict[str, Any] = {
        "benchmark": "phase2_known_law_structural_recognition_v1",
        "claim_scope": "controlled offline verification in a frozen six-law library",
        "synthetic": True,
        "case_count": len(cases),
        "structural_accuracy": correct / len(cases),
        "positive_recall": positive_correct / positive_total,
        "hard_negative_rejection": negative_correct / negative_total,
        "semantic_only_accuracy": semantic_correct / len(cases),
        "entity_rename_invariance": rename_invariant,
        "missing_boundary_abstention": boundary_result.abstained,
        "records": records,
    }
    report["report_id"] = stable_hash(report, prefix="benchmark_")
    return report
