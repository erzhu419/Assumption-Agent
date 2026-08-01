"""A measured, synthetic end-to-end conservative scope extension.

This is an integration qualification, not downstream efficacy evidence.
"""

from __future__ import annotations

from typing import Any

from .benchmark import BenchmarkCase, controlled_cases
from .bootstrap import initial_theory
from .domain import LawBinding, StructuralEpisode
from .governance import (
    BranchRecord,
    EvidenceLedger,
    TheoryVersionGraph,
    compile_patch,
    evaluate_conservative_extension,
)
from .hashing import canonical_json, stable_hash
from .pipeline import verify_against_frozen_library
from .schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceKind,
    EvidenceReceipt,
    EvidenceSplit,
    FrameworkStatus,
    PatchCoordinate,
    PreregisteredPrediction,
    ReductionMap,
    TheoryPatch,
    TheoryState,
    freeze_pairs,
)

OLD_SCOPE = "controlled_offline_structural_laws"
NEW_SCOPE = "controlled_renamed_domain"
CANDIDATE_ID = "candidate_scoped_structural_transfer_v1"
PREDICTION_ID = "prediction_renamed_domain_transfer_v1"


def _episode_and_binding(
    theory: TheoryState,
    case: BenchmarkCase,
    *,
    scope: str,
    split: EvidenceSplit,
    suffix: str,
) -> tuple[StructuralEpisode, LawBinding]:
    law = next(item for item in theory.relation_laws if item.kind is case.kind)
    observation_id = f"obs_{case.case_id}_{suffix}"
    entity_ids = tuple(f"entity_{index}" for index in range(len(law.roles)))
    episode = StructuralEpisode.from_mapping(
        episode_id=f"episode_{case.case_id}_{suffix}",
        observation_ids=(observation_id,),
        object_types={
            entity_id: f"type_{role}"
            for role, entity_id in zip(law.roles, entity_ids, strict=True)
        },
        role_candidates=dict(zip(law.roles, entity_ids, strict=True)),
        role_observable_witnesses=dict(law.role_observable_requirements),
        observables=dict(case.episode),
        scale_id="phase2_default",
        scope=(scope,),
        split=split,
        data_cutoff=theory.data_cutoff,
    )
    binding = LawBinding(
        binding_id=f"binding_{case.case_id}_{suffix}",
        law_id=law.law_id,
        law_kind=law.kind,
        role_assignments=tuple(zip(law.roles, entity_ids, strict=True)),
        source_span_ids=(observation_id,),
        scale_id="phase2_default",
    )
    return episode, binding


def _measure_case(
    theory: TheoryState,
    case: BenchmarkCase,
    *,
    scope: str,
    split: EvidenceSplit,
    suffix: str,
) -> tuple[bool | None, str]:
    episode, binding = _episode_and_binding(
        theory, case, scope=scope, split=split, suffix=suffix
    )
    try:
        outcome = verify_against_frozen_library(
            theory=theory, episode=episode, bindings=(binding,)
        )[0]
    except ValueError as exc:
        if "outside the frozen theory scope" in str(exc):
            return None, episode.observation_ids[0]
        raise
    return outcome.match is not None, episode.observation_ids[0]


def _receipt(
    parent: TheoryState,
    *,
    metric: str,
    value: float,
    threshold: float,
    higher_is_better: bool,
    split: EvidenceSplit,
    observation_ids: tuple[str, ...],
    preregistration_id: str | None = None,
) -> EvidenceReceipt:
    passed = value >= threshold if higher_is_better else value <= threshold
    probe_id = (
        "probe_hard_negative"
        if metric == "hard_negative_rejection"
        else "probe_exact_residual"
    )
    registered_probes = {probe.probe_id: probe for probe in parent.probes}
    if probe_id not in registered_probes:
        raise ValueError(f"vertical-slice probe is not registered: {probe_id}")
    return EvidenceReceipt(
        receipt_id=f"receipt_{metric}",
        theory_version_id=parent.version_id,
        candidate_id=CANDIDATE_ID,
        evaluator_epoch=parent.evaluator.epoch,
        probe_id=probe_id,
        probe_version=registered_probes[probe_id].version,
        data_cutoff=parent.data_cutoff,
        split=split,
        kind=EvidenceKind.EXECUTABLE_TEST,
        metric=metric,
        value=value,
        threshold=threshold,
        higher_is_better=higher_is_better,
        passed=passed,
        independent=True,
        observation_ids=observation_ids,
        preregistration_id=(
            preregistration_id if split is EvidenceSplit.HOLDOUT else None
        ),
        actor_id=(
            "controlled_falsifier"
            if metric == "hard_negative_rejection"
            else "controlled_evaluator"
        ),
    )


def run_controlled_vertical_slice() -> dict[str, Any]:
    parent = initial_theory()
    cases = controlled_cases()

    payload = freeze_pairs(
        {
            "operation": "add_scope",
            "scope": NEW_SCOPE,
            "law_library": "unchanged",
        }
    )
    unified_bytes = len(canonical_json(payload).encode("utf-8"))
    local_patch_bytes = len(
        canonical_json(
            tuple(
                {"case_id": case.case_id, "rule": "accept_if_fixture_id"}
                for case in cases
            )
        ).encode("utf-8")
    )
    compression_gain = (local_patch_bytes - unified_bytes) / local_patch_bytes
    complexity_cost = unified_bytes / 4096.0
    prediction = PreregisteredPrediction(
        prediction_id=PREDICTION_ID,
        input_condition="renamed typed episodes in the added controlled scope",
        outcome_name="structural verifier classification",
        expected_direction="preserved",
        expected_range=(0.95, 1.0),
        failure_criterion="accuracy below 0.95 or any entity-name dependence",
        registered_at_cutoff="2026-07-30T08:00:00+08:00",
    )
    hard_negative_ids = tuple(
        f"obs_{case.case_id}_hard_negative"
        for case in cases
        if not case.relation_present
    )
    patch = TheoryPatch(
        patch_id="patch_scoped_structural_transfer_v1",
        candidate_id=CANDIDATE_ID,
        parent_version_id=parent.version_id,
        coordinate=PatchCoordinate.SCOPE,
        claim="the frozen structural law library transfers under entity renaming",
        scope=(NEW_SCOPE,),
        failure_boundary=("new_observable_schema", "unregistered_scale"),
        predictions=(prediction,),
        hard_negative_ids=hard_negative_ids,
        reduction_map_id="reduction_scoped_transfer_v1",
        conditional_description_length=complexity_cost,
        payload=payload,
        authority_assignments=tuple(
            AuthorityAssignment(role, f"controlled_{role.value}")
            for role in AuthorityRole
        ),
    )
    candidate_theory = compile_patch(parent, patch)

    old_predictions: list[bool] = []
    candidate_old_predictions: list[bool] = []
    old_ids: list[str] = []
    for case in cases:
        expected, observation_id = _measure_case(
            parent,
            case,
            scope=OLD_SCOPE,
            split=EvidenceSplit.OLD_SUCCESS,
            suffix="old",
        )
        candidate, _ = _measure_case(
            candidate_theory,
            case,
            scope=OLD_SCOPE,
            split=EvidenceSplit.OLD_SUCCESS,
            suffix="candidate_old",
        )
        old_predictions.append(bool(expected))
        candidate_old_predictions.append(bool(candidate))
        old_ids.append(observation_id)

    train_correct: list[bool] = []
    train_ids: list[str] = []
    parent_out_of_scope: list[bool] = []
    for case in cases:
        parent_result, _ = _measure_case(
            parent,
            case,
            scope=NEW_SCOPE,
            split=EvidenceSplit.TRAIN,
            suffix="parent_new",
        )
        candidate_result, observation_id = _measure_case(
            candidate_theory,
            case,
            scope=NEW_SCOPE,
            split=EvidenceSplit.TRAIN,
            suffix="candidate_new",
        )
        parent_out_of_scope.append(parent_result is None)
        train_correct.append(candidate_result is case.relation_present)
        train_ids.append(observation_id)

    # These outcomes are opened only after PREDICTION_ID is fixed above.
    holdout_correct: list[bool] = []
    holdout_ids: list[str] = []
    for case in cases:
        renamed_case = BenchmarkCase(
            case_id=case.case_id + "_renamed",
            kind=case.kind,
            episode={**case.episode, "entity_names": ("holdout_A", "holdout_B")},
            relation_present=case.relation_present,
            semantic_overlap=1.0 - case.semantic_overlap,
            control="entity_rename_and_semantic_inversion",
        )
        result, observation_id = _measure_case(
            candidate_theory,
            renamed_case,
            scope=NEW_SCOPE,
            split=EvidenceSplit.HOLDOUT,
            suffix="holdout",
        )
        holdout_correct.append(result is renamed_case.relation_present)
        holdout_ids.append(observation_id)

    preservation = sum(
        left == right
        for left, right in zip(
            old_predictions, candidate_old_predictions, strict=True
        )
    ) / len(cases)
    residual_explanation = sum(train_correct) / len(train_correct)
    limiting_reduction = preservation
    expressivity_gain = sum(parent_out_of_scope) / len(parent_out_of_scope)
    unseen_success = sum(holdout_correct) / len(holdout_correct)
    regression_cost = 1.0 - preservation

    hard_negative_correct: list[bool] = []
    measured_hard_negative_ids: list[str] = []
    for case in cases:
        if case.relation_present:
            continue
        result, observation_id = _measure_case(
            candidate_theory,
            case,
            scope=NEW_SCOPE,
            split=EvidenceSplit.HARD_NEGATIVE,
            suffix="hard_negative",
        )
        hard_negative_correct.append(result is False)
        measured_hard_negative_ids.append(observation_id)
    hard_negative_rejection = sum(hard_negative_correct) / len(
        hard_negative_correct
    )
    assert tuple(measured_hard_negative_ids) == patch.hard_negative_ids

    reduction = ReductionMap(
        reduction_id=patch.reduction_map_id,
        parent_version_id=parent.version_id,
        child_candidate_id=CANDIDATE_ID,
        old_scope=(OLD_SCOPE,),
        mapping_description="remove the added scope; law code and outputs are unchanged",
        executable_check_id="paired_old_scope_prediction_equality",
        maximum_error=regression_cost,
    )

    receipts = (
        _receipt(
            parent,
            metric="residual_explanation",
            value=residual_explanation,
            threshold=0.75,
            higher_is_better=True,
            split=EvidenceSplit.TRAIN,
            observation_ids=tuple(train_ids),
        ),
        _receipt(
            parent,
            metric="old_success_preservation",
            value=preservation,
            threshold=0.95,
            higher_is_better=True,
            split=EvidenceSplit.OLD_SUCCESS,
            observation_ids=tuple(old_ids),
        ),
        _receipt(
            parent,
            metric="limiting_case_reduction",
            value=limiting_reduction,
            threshold=0.90,
            higher_is_better=True,
            split=EvidenceSplit.OLD_SUCCESS,
            observation_ids=tuple(old_ids),
        ),
        _receipt(
            parent,
            metric="expressivity_gain",
            value=expressivity_gain,
            threshold=0.01,
            higher_is_better=True,
            split=EvidenceSplit.TRAIN,
            observation_ids=tuple(train_ids),
        ),
        _receipt(
            parent,
            metric="compression_gain",
            value=compression_gain,
            threshold=0.01,
            higher_is_better=True,
            split=EvidenceSplit.TRAIN,
            observation_ids=tuple(train_ids),
        ),
        _receipt(
            parent,
            metric="unseen_prediction_success",
            value=unseen_success,
            threshold=0.60,
            higher_is_better=True,
            split=EvidenceSplit.HOLDOUT,
            observation_ids=tuple(holdout_ids),
            preregistration_id=prediction.content_id,
        ),
        _receipt(
            parent,
            metric="hard_negative_rejection",
            value=hard_negative_rejection,
            threshold=0.95,
            higher_is_better=True,
            split=EvidenceSplit.HARD_NEGATIVE,
            observation_ids=patch.hard_negative_ids,
        ),
        _receipt(
            parent,
            metric="regression_cost",
            value=regression_cost,
            threshold=0.02,
            higher_is_better=False,
            split=EvidenceSplit.OLD_SUCCESS,
            observation_ids=tuple(old_ids),
        ),
        _receipt(
            parent,
            metric="complexity_cost",
            value=complexity_cost,
            threshold=1.0,
            higher_is_better=False,
            split=EvidenceSplit.TRAIN,
            observation_ids=tuple(train_ids),
        ),
    )
    ledger = EvidenceLedger(
        parent.version_id,
        CANDIDATE_ID,
        parent.evaluator.epoch,
        parent.data_cutoff,
        stable_hash(
            tuple(case.case_id for case in cases),
            prefix="controlled_manifest_",
        ),
        None,
        receipts,
    )
    certificate = evaluate_conservative_extension(
        parent=parent,
        patch=patch,
        ledger=ledger,
        reduction_map=reduction,
    )
    shadow_graph = TheoryVersionGraph(
        states=(parent,),
        branches=(
            BranchRecord(
                "branch_scoped_structural_transfer_v1",
                parent.version_id,
                patch.candidate_id,
                patch.content_id,
                FrameworkStatus.CANDIDATE_BRANCH,
            ),
        ),
    ).record_evaluation(
        branch_id="branch_scoped_structural_transfer_v1",
        parent=parent,
        patch=patch,
        ledger=ledger,
        reduction_map=reduction,
    )
    report: dict[str, Any] = {
        "qualification": "controlled_vertical_slice_v1",
        "synthetic": True,
        "claim_scope": (
            "integration qualification only; no downstream efficacy or invention claim"
        ),
        "parent_version_id": parent.version_id,
        "candidate_preview_version_id": candidate_theory.version_id,
        "promoted_child_version_id": None,
        "patch_id": patch.patch_id,
        "certificate_id": certificate.certificate_id,
        "decision": certificate.decision.value,
        "metrics": {receipt.metric: receipt.value for receipt in receipts},
        "receipt_ids": [receipt.content_id for receipt in receipts],
        "shadow_graph_edge_count": len(shadow_graph.edges),
        "shadow_graph_authoritative": shadow_graph.authoritative,
        "hard_negative_count": len(patch.hard_negative_ids),
        "negative_evidence_retained": len(shadow_graph.negative_evidence_ids),
        "certificate_recorded": certificate.certificate_id
        in shadow_graph.certificate_ids,
        "sealed_holdout": ledger.sealed_holdout,
        "data_cutoff": parent.data_cutoff,
    }
    report["report_id"] = stable_hash(report, prefix="vertical_slice_")
    return report
