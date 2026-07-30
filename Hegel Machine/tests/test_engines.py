import pytest

from hegel_machine.engines import (
    IdealizationContract,
    RobustModelKind,
    RobustificationCandidate,
    UncertaintyKind,
    evaluate_idealization,
    evaluate_robustification,
    pareto_frontier,
)
from hegel_machine.schema import ReductionMap


def reduction(candidate_id: str) -> ReductionMap:
    return ReductionMap(
        f"reduction_{candidate_id}",
        "parent",
        candidate_id,
        ("old_scope",),
        "ambiguity singleton recovers nominal model",
        "check_singleton",
        0.0,
    )


def robust(candidate_id: str, tail: float, cost: float) -> RobustificationCandidate:
    return RobustificationCandidate(
        candidate_id,
        "parent",
        RobustModelKind.AMBIGUITY_SET,
        UncertaintyKind.EPISTEMIC,
        ("tail_scope",),
        ("stress_1",),
        reduction(candidate_id),
        nominal_utility=0.90,
        tail_utility=tail,
        safety_utility=0.80,
        coverage=0.95,
        complexity_cost=cost,
    )


def test_robustification_uses_vector_contract():
    decision = evaluate_robustification(
        robust("robust_good", 0.75, 0.2),
        minimum_tail_gain=0.10,
        minimum_safety_gain=0.10,
        minimum_coverage=0.90,
        maximum_nominal_loss=0.05,
        parent_nominal_utility=0.92,
        parent_tail_utility=0.50,
        parent_safety_utility=0.60,
    )
    assert decision.accepted
    assert dict(decision.utility_vector)["tail"] == 0.75


def test_pareto_frontier_drops_dominated_candidate():
    strong = robust("strong", 0.8, 0.1)
    weak = robust("weak", 0.7, 0.2)
    assert pareto_frontier((strong, weak)) == (strong,)


def test_idealization_requires_probe_preservation_and_complexity_gain():
    contract = IdealizationContract(
        "ideal_1",
        "parent",
        "candidate",
        "quotient",
        ("gauge_coordinate",),
        ("decision_probe",),
        ("scale",),
        ("task",),
        maximum_probe_discrepancy=0.01,
        discrepancy_budget=0.02,
        failure_boundary=("fine_scale",),
        counterexample_ids=("counter_1",),
        compute_gain=0.4,
        sample_gain=0.0,
        complexity_gain=0.3,
        full_model_recovery="select representative and restore gauge",
    )
    assert evaluate_idealization(contract, minimum_complexity_gain=0.2).accepted


def test_robustification_rejects_nonfinite_utility():
    with pytest.raises(ValueError, match="finite"):
        RobustificationCandidate(
            "nonfinite",
            "parent",
            RobustModelKind.AMBIGUITY_SET,
            UncertaintyKind.EPISTEMIC,
            ("scope",),
            ("stress",),
            reduction("nonfinite"),
            nominal_utility=float("inf"),
            tail_utility=0.8,
            safety_utility=0.8,
            coverage=0.9,
            complexity_cost=0.1,
        )


def test_idealization_rejects_nonfinite_discrepancy():
    with pytest.raises(ValueError, match="finite"):
        IdealizationContract(
            "nonfinite",
            "parent",
            "candidate",
            "projection",
            ("coordinate",),
            ("probe",),
            ("scale",),
            ("task",),
            maximum_probe_discrepancy=float("-inf"),
            discrepancy_budget=0.1,
            failure_boundary=("boundary",),
            counterexample_ids=("counter",),
            compute_gain=0.1,
            sample_gain=0.1,
            complexity_gain=0.1,
            full_model_recovery="restore",
        )
