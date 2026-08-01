from dataclasses import replace

import pytest

from hegel_machine.phase2b_selector import (
    CandidateEvaluation,
    CandidateGridCommitment,
    CandidateGridCell,
    CandidateIntervalStatus,
    ClosedInterval,
    Phase2BSelectionPolicy,
    TypedSelectionDisposition,
    _select_against_grid_commitment,
)
from hegel_machine.schema import LawKind


def candidate(
    kind,
    binding_label,
    scale,
    residual,
    tolerance=(1.0, 1.0),
    *,
    suffix="",
):
    binding = (("subject", f"entity_{binding_label}"),)
    return CandidateEvaluation(
        candidate_id=f"candidate_{kind.name}_{binding_label}_{scale}{suffix}",
        law_kind=kind,
        role_binding=binding,
        scale_hypothesis_id=scale,
        residual=ClosedInterval(*residual),
        tolerance=ClosedInterval(*tolerance),
        completed=True,
        footprint_id=f"footprint_{kind.name}_{binding_label}",
    )


def complete_grid(*, two_passing_scales=False):
    items = []
    for kind in LawKind:
        for binding in ("a", "b"):
            for scale in ("s1", "s2"):
                passing = (
                    kind is LawKind.SYMMETRY
                    and binding == "a"
                    and (scale == "s1" or two_passing_scales)
                )
                residual = (0.1, 0.2) if passing else (3.0, 4.0)
                items.append(candidate(kind, binding, scale, residual))
    return tuple(items)


def grid_commitment(evaluations):
    return CandidateGridCommitment(
        adapter_result_id="phase2b_adapter_result_" + "a" * 64,
        bundle_content_id="phase2b_evidence_" + "b" * 64,
        registry_id="phase2b_adapter_registry_" + "c" * 64,
        expected_cells=tuple(
            CandidateGridCell(
                candidate_id=item.candidate_id,
                law_kind=item.law_kind,
                role_binding=item.role_binding,
                scale_hypothesis_id=item.scale_hypothesis_id,
                footprint_id=item.footprint_id,
            )
            for item in sorted(evaluations, key=lambda item: item.candidate_id)
        ),
    )


def select_typed_candidate_evaluations(evaluations, *, policy=None):
    kwargs = {"grid_commitment": grid_commitment(evaluations)}
    if policy is not None:
        kwargs["policy"] = policy
    return _select_against_grid_commitment(evaluations, **kwargs)


def test_interval_status_uses_conservative_candidate_specific_tolerance():
    passing = candidate(
        LawKind.SYMMETRY,
        "a",
        "s1",
        residual=(0.5, 0.8),
        tolerance=(1.0, 2.0),
    )
    assert passing.normalized_interval == ClosedInterval(0.25, 0.8)
    assert passing.status is CandidateIntervalStatus.PASS
    overlap = replace(passing, residual=ClosedInterval(0.5, 1.5))
    assert overlap.normalized_interval == ClosedInterval(0.25, 1.5)
    assert overlap.status is CandidateIntervalStatus.INCONCLUSIVE
    failing = replace(passing, residual=ClosedInterval(2.1, 3.0))
    assert failing.normalized_interval == ClosedInterval(1.05, 3.0)
    assert failing.status is CandidateIntervalStatus.FAIL


def test_selector_returns_unique_structure_and_scale():
    decision = select_typed_candidate_evaluations(complete_grid())
    assert decision.disposition is TypedSelectionDisposition.UNIQUE_IDENTIFICATION
    assert decision.selected_law_kind is LawKind.SYMMETRY
    assert decision.selected_role_binding == (("subject", "entity_a"),)
    assert decision.admissible_scale_hypothesis_ids == ("s1",)
    assert decision.normalized_structural_margin_lcb == pytest.approx(2.8)


def test_selector_can_return_a_preregistered_admissible_scale_set():
    decision = select_typed_candidate_evaluations(
        complete_grid(two_passing_scales=True)
    )
    assert decision.disposition is TypedSelectionDisposition.ADMISSIBLE_SCALE_SET
    assert decision.admissible_scale_hypothesis_ids == ("s1", "s2")


def test_multiple_passing_bindings_fail_closed():
    grid = list(complete_grid())
    index = next(
        index
        for index, item in enumerate(grid)
        if item.law_kind is LawKind.SYMMETRY
        and item.role_binding == (("subject", "entity_b"),)
        and item.scale_hypothesis_id == "s1"
    )
    grid[index] = replace(grid[index], residual=ClosedInterval(0.1, 0.2))
    decision = select_typed_candidate_evaluations(tuple(grid))
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN
    assert decision.reason == "multiple_passing_structures"


def test_inconclusive_scale_and_candidate_error_fail_closed():
    grid = list(complete_grid())
    scale_index = next(
        index
        for index, item in enumerate(grid)
        if item.law_kind is LawKind.SYMMETRY
        and item.role_binding == (("subject", "entity_a"),)
        and item.scale_hypothesis_id == "s2"
    )
    grid[scale_index] = replace(
        grid[scale_index],
        residual=ClosedInterval(0.5, 1.5),
    )
    decision = select_typed_candidate_evaluations(tuple(grid))
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN
    assert decision.reason == "selected_structure_has_inconclusive_scale"

    grid = list(complete_grid())
    grid[0] = replace(
        grid[0],
        completed=False,
        residual=None,
        tolerance=None,
        error_code="verifier_exception",
    )
    assert (
        select_typed_candidate_evaluations(tuple(grid)).reason
        == "candidate_evaluation_error"
    )


def test_missing_family_competitor_and_budget_overflow_abstain():
    full_grid = complete_grid()
    grid = tuple(
        item for item in full_grid if item.law_kind is not LawKind.LOCALITY
    )
    assert (
        _select_against_grid_commitment(
            grid,
            grid_commitment=grid_commitment(full_grid),
        ).reason
        == "incomplete_or_drifted_candidate_grid"
    )
    decision = select_typed_candidate_evaluations(
        complete_grid(),
        policy=Phase2BSelectionPolicy(maximum_candidate_count=1),
    )
    assert decision.reason == "candidate_budget_exceeded"


def test_selector_is_invariant_to_candidate_input_order():
    grid = complete_grid()
    assert select_typed_candidate_evaluations(grid) == (
        select_typed_candidate_evaluations(tuple(reversed(grid)))
    )


def test_duplicate_candidate_ids_are_rejected_without_exception():
    grid = complete_grid()
    duplicated = grid + (replace(grid[0]),)
    decision = _select_against_grid_commitment(
        duplicated,
        grid_commitment=grid_commitment(grid),
    )
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN
    assert decision.reason == "duplicate_candidate_id"


def test_selector_requires_and_exactly_matches_adapter_grid_commitment():
    grid = complete_grid()
    missing_commitment = _select_against_grid_commitment(grid)
    assert missing_commitment.disposition is TypedSelectionDisposition.ABSTAIN
    assert missing_commitment.reason == "missing_candidate_grid_commitment"

    truncated = grid[:-1]
    drifted = _select_against_grid_commitment(
        truncated,
        grid_commitment=grid_commitment(grid),
    )
    assert drifted.disposition is TypedSelectionDisposition.ABSTAIN
    assert drifted.reason == "incomplete_or_drifted_candidate_grid"

    changed = list(grid)
    changed[0] = replace(changed[0], footprint_id="forged_footprint")
    metadata_drift = _select_against_grid_commitment(
        tuple(changed),
        grid_commitment=grid_commitment(grid),
    )
    assert metadata_drift.disposition is TypedSelectionDisposition.ABSTAIN
    assert metadata_drift.reason == "candidate_metadata_drift"


def test_inconclusive_structural_competitor_never_counts_as_margin():
    grid = list(complete_grid())
    index = next(
        index
        for index, item in enumerate(grid)
        if item.law_kind is LawKind.LOCALITY
        and item.role_binding == (("subject", "entity_a"),)
        and item.scale_hypothesis_id == "s1"
    )
    grid[index] = replace(grid[index], residual=ClosedInterval(1.0, 2.0))
    grid = tuple(grid)
    decision = select_typed_candidate_evaluations(grid)
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN
    assert decision.reason == "inconclusive_structural_competitor"
