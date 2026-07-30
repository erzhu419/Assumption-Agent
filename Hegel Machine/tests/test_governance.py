from dataclasses import replace

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.hashing import stable_hash
from hegel_machine.governance import (
    BranchRecord,
    ConservativeExtensionCertificate,
    DEFAULT_GATE_THRESHOLDS,
    EvidenceLedger,
    EvaluationRecord,
    GateCheck,
    PromotionDecision,
    SealedHoldoutManifest,
    TheoryVersionGraph,
    authorize_promotion,
    compile_patch,
    evaluate_conservative_extension,
)
from hegel_machine.schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceKind,
    EvidenceReceipt,
    EvidenceSplit,
    FrameworkStatus,
    Observation,
    PatchCoordinate,
    PreregisteredPrediction,
    ReductionMap,
    TheoryPatch,
    freeze_pairs,
)

CANDIDATE = "candidate_scope_v1"
OBSERVATIONS = {}


def observation(label, split, cutoff):
    item = Observation(
        observation_id=label,
        source_uri=f"fixture://{label}",
        split=split,
        data_cutoff=cutoff,
        observables=freeze_pairs({"label": label}),
        provenance_hash=stable_hash(
            (label, split.value, cutoff)
        ),
    )
    OBSERVATIONS[item.content_id] = item
    return item


def observation_id(label, split, cutoff):
    return observation(label, split, cutoff).content_id


def assignments():
    return tuple(
        AuthorityAssignment(role, f"actor_{role.value}") for role in AuthorityRole
    )


def prediction(*, registered_at="2026-07-30T08:00:00+08:00"):
    return PreregisteredPrediction(
        "prediction_1",
        "new typed episode",
        "structural match",
        "preserved",
        (0.60, 1.0),
        "score below 0.60",
        registered_at,
    )


def patch(parent, *, coordinate=PatchCoordinate.SCOPE, pred=None):
    pred = pred or prediction()
    payload = (
        freeze_pairs(
            {
                "operation": "add_scope",
                "scope": "new_scope",
                "law_library": "unchanged",
            }
        )
        if coordinate is PatchCoordinate.SCOPE
        else freeze_pairs({"symbol": "R_new"})
    )
    return TheoryPatch(
        "patch_1",
        CANDIDATE,
        parent.version_id,
        coordinate,
        "extend the scoped structural law",
        ("new_scope",),
        ("fine_scale_failure",),
        (pred,),
        (
            observation_id(
                "negative_1",
                EvidenceSplit.HARD_NEGATIVE,
                parent.data_cutoff,
            ),
        ),
        "reduction_1",
        0.20,
        payload,
        assignments(),
        ontology_report_id=(
            "future_phase3_report"
            if coordinate is PatchCoordinate.LANGUAGE
            else None
        ),
    )


def reduction(parent, *, maximum_error=0.01):
    return ReductionMap(
        "reduction_1",
        parent.version_id,
        CANDIDATE,
        ("controlled_offline_structural_laws",),
        "new scoped rule equals parent in old scope",
        "check_old_scope_reduction",
        maximum_error,
    )


def receipt(
    parent,
    candidate_patch,
    metric,
    value,
    threshold,
    *,
    higher=True,
    split=EvidenceSplit.VALIDATION,
    kind=EvidenceKind.EXECUTABLE_TEST,
    epoch=None,
    cutoff=None,
    actor=None,
    probe_id=None,
    independent=True,
    observation_ids=None,
):
    passed = value >= threshold if higher else value <= threshold
    hard_negative = metric == "hard_negative_rejection"
    receipt_cutoff = cutoff or parent.data_cutoff
    return EvidenceReceipt(
        receipt_id=f"receipt_{metric}_{value}",
        theory_version_id=parent.version_id,
        candidate_id=CANDIDATE,
        evaluator_epoch=epoch or parent.evaluator.epoch,
        probe_id=probe_id
        or ("probe_hard_negative" if hard_negative else "probe_exact_residual"),
        probe_version="1",
        data_cutoff=receipt_cutoff,
        split=split,
        kind=kind,
        metric=metric,
        value=value,
        threshold=threshold,
        higher_is_better=higher,
        passed=passed,
        independent=independent,
        observation_ids=observation_ids
        or (
            (candidate_patch.hard_negative_ids[0],)
            if hard_negative
            else (
                observation_id(
                    f"partition_{split.value}",
                    split,
                    receipt_cutoff,
                ),
            )
        ),
        preregistration_id=(
            candidate_patch.prediction_ids[0]
            if split is EvidenceSplit.HOLDOUT
            else None
        ),
        actor_id=actor
        or (
            "actor_generator"
            if kind is EvidenceKind.SEMANTIC_RETRIEVAL
            else "actor_falsifier"
            if hard_negative
            else "actor_evaluator"
        ),
    )


def passing_receipts(parent, candidate_patch, *, semantic_value=0.99, cutoff=None):
    return (
        receipt(
            parent,
            candidate_patch,
            "residual_explanation",
            0.90,
            0.75,
            split=EvidenceSplit.VALIDATION,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "old_success_preservation",
            0.99,
            0.95,
            split=EvidenceSplit.OLD_SUCCESS,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "limiting_case_reduction",
            0.96,
            0.90,
            split=EvidenceSplit.OLD_SUCCESS,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "expressivity_gain",
            0.20,
            0.01,
            split=EvidenceSplit.VALIDATION,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "compression_gain",
            0.10,
            0.01,
            split=EvidenceSplit.VALIDATION,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "unseen_prediction_success",
            0.80,
            0.60,
            split=EvidenceSplit.HOLDOUT,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "hard_negative_rejection",
            1.0,
            0.95,
            split=EvidenceSplit.HARD_NEGATIVE,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "regression_cost",
            0.01,
            0.02,
            higher=False,
            split=EvidenceSplit.OLD_SUCCESS,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "complexity_cost",
            0.20,
            1.00,
            higher=False,
            split=EvidenceSplit.VALIDATION,
            cutoff=cutoff,
        ),
        receipt(
            parent,
            candidate_patch,
            "semantic_retrieval_score",
            semantic_value,
            0.50,
            split=EvidenceSplit.TRAIN,
            kind=EvidenceKind.SEMANTIC_RETRIEVAL,
            cutoff=cutoff,
            independent=False,
        ),
    )


def ledger(parent, candidate_patch, receipts, *, sealed=True, cutoff=None):
    def ids_for(split):
        return tuple(
            sorted(
                {
                    observation_id
                    for item in receipts
                    if item.split is split
                    for observation_id in item.observation_ids
                }
            )
        )

    holdout_ids = tuple(
        observation_id
        for item in receipts
        if item.metric == "unseen_prediction_success"
        for observation_id in item.observation_ids
    )
    hard_negative_ids = tuple(
        observation_id
        for item in receipts
        if item.metric == "hard_negative_rejection"
        for observation_id in item.observation_ids
    )
    train_ids = ids_for(EvidenceSplit.TRAIN) or (
        observation_id(
            "partition_train",
            EvidenceSplit.TRAIN,
            cutoff or parent.data_cutoff,
        ),
    )
    validation_ids = ids_for(EvidenceSplit.VALIDATION)
    old_success_ids = ids_for(EvidenceSplit.OLD_SUCCESS)
    manifest_observation_ids = set().union(
        train_ids,
        validation_ids,
        old_success_ids,
        holdout_ids,
        hard_negative_ids,
    )
    manifest = (
        SealedHoldoutManifest(
            theory_version_id=parent.version_id,
            candidate_id=candidate_patch.candidate_id,
            patch_content_id=candidate_patch.content_id,
            evaluator_epoch=parent.evaluator.epoch,
            evaluator_version=parent.evaluator.version,
            gate_policy_id=DEFAULT_GATE_THRESHOLDS.policy_id,
            probe_registry_id=stable_hash(
                parent.probes,
                prefix="probe_registry_",
            ),
            observations=tuple(
                sorted(
                    (OBSERVATIONS[item] for item in manifest_observation_ids),
                    key=lambda item: item.content_id,
                )
            ),
            train_observation_ids=train_ids,
            validation_observation_ids=validation_ids,
            old_success_observation_ids=old_success_ids,
            holdout_observation_ids=holdout_ids,
            hard_negative_ids=hard_negative_ids,
            registered_at="2026-07-30T06:00:00+08:00",
            opened_at="2026-07-30T12:00:00+08:00",
            independent_custodian_id="external_custodian",
            generator_excluded=True,
            opening_nonce=stable_hash(
                (parent.version_id, candidate_patch.content_id),
                prefix="opening_",
            ),
        )
        if sealed
        else None
    )
    return EvidenceLedger(
        parent.version_id,
        CANDIDATE,
        parent.evaluator.epoch,
        cutoff or parent.data_cutoff,
        manifest.manifest_id if manifest else "controlled_manifest_content_hash",
        manifest,
        tuple(receipts),
    )


def certificate(parent, *, sealed=True, receipts=None, candidate_patch=None):
    candidate_patch = candidate_patch or patch(parent)
    receipts = receipts or passing_receipts(parent, candidate_patch)
    return evaluate_conservative_extension(
        parent=parent,
        patch=candidate_patch,
        ledger=ledger(
            parent,
            candidate_patch,
            receipts,
            sealed=sealed,
        ),
        reduction_map=reduction(parent),
    )


def branch_graph(
    parent,
    candidate_patch,
    *,
    status=FrameworkStatus.CANDIDATE_BRANCH,
):
    return TheoryVersionGraph(
        states=(parent,),
        branches=(
            BranchRecord(
                "branch_1",
                parent.version_id,
                candidate_patch.candidate_id,
                candidate_patch.content_id,
                status,
            ),
        ),
    )


def test_structurally_sealed_receipts_stop_without_external_trust_root():
    parent = initial_theory()
    candidate_patch = patch(parent)
    cert = certificate(parent, candidate_patch=candidate_patch)
    assert cert.decision is PromotionDecision.CANDIDATE
    failed = [check.name for check in cert.checks if not check.passed]
    assert failed == ["external_trust_root"]
    assert cert.required_next_tests == (
        "implement_and_verify_external_trust_root",
    )
    assert cert.patch_content_id == candidate_patch.content_id
    assert cert.proposed_child_version_id == compile_patch(
        parent, candidate_patch
    ).version_id


def test_unsealed_controlled_evidence_stops_at_candidate():
    parent = initial_theory()
    cert = certificate(parent, sealed=False)
    assert cert.decision is PromotionDecision.CANDIDATE
    assert cert.required_next_tests == (
        "run_sealed_external_holdout",
        "implement_and_verify_external_trust_root",
    )


def test_semantic_retrieval_score_cannot_change_certification():
    parent = initial_theory()
    candidate_patch = patch(parent)
    high = certificate(parent, candidate_patch=candidate_patch)
    low_receipts = list(passing_receipts(parent, candidate_patch))
    low_receipts[-1] = receipt(
        parent,
        candidate_patch,
        "semantic_retrieval_score",
        0.01,
        0.50,
        split=EvidenceSplit.TRAIN,
        kind=EvidenceKind.SEMANTIC_RETRIEVAL,
        independent=False,
    )
    low = certificate(
        parent, receipts=low_receipts, candidate_patch=candidate_patch
    )
    assert high.decision is low.decision is PromotionDecision.CANDIDATE
    assert high.checks == low.checks
    assert high.receipt_ids == low.receipt_ids


def test_semantic_retrieval_receipt_is_optional_for_structural_candidate():
    parent = initial_theory()
    candidate_patch = patch(parent)
    structural_receipts = passing_receipts(parent, candidate_patch)[:-1]
    cert = certificate(
        parent,
        receipts=structural_receipts,
        candidate_patch=candidate_patch,
    )
    assert cert.decision is PromotionDecision.CANDIDATE
    assert [
        check.name for check in cert.checks if not check.passed
    ] == ["external_trust_root"]


def test_cross_epoch_aggregation_is_rejected():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[0] = receipt(
        parent,
        candidate_patch,
        "residual_explanation",
        0.90,
        0.75,
        split=EvidenceSplit.VALIDATION,
        epoch="new_epoch",
    )
    with pytest.raises(ValueError, match="cross-epoch"):
        ledger(parent, candidate_patch, receipts)


@pytest.mark.parametrize(
    ("mutation", "needle"),
    (
        ({"probe_id": "not_registered"}, "unregistered_probe"),
        ({"actor": "actor_generator"}, "wrong_authority"),
        ({"independent": False}, "not_independent"),
    ),
)
def test_evidence_contract_rejects_probe_actor_and_independence_bypass(
    mutation, needle
):
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[0] = receipt(
        parent,
        candidate_patch,
        "residual_explanation",
        0.90,
        0.75,
        split=EvidenceSplit.VALIDATION,
        **mutation,
    )
    cert = certificate(
        parent, receipts=receipts, candidate_patch=candidate_patch
    )
    evidence_check = next(
        check for check in cert.checks if check.name == "evidence_contract"
    )
    assert not evidence_check.passed
    assert needle in evidence_check.reason
    assert cert.decision is PromotionDecision.BRANCH_ONLY


def test_future_or_mismatched_cutoff_is_rejected():
    parent = initial_theory()
    candidate_patch = patch(parent)
    future = "2999-12-31T23:59:59+00:00"
    receipts = passing_receipts(parent, candidate_patch, cutoff=future)
    cert = evaluate_conservative_extension(
        parent=parent,
        patch=candidate_patch,
        ledger=ledger(
            parent,
            candidate_patch,
            receipts,
            cutoff=future,
        ),
        reduction_map=reduction(parent),
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert not check.passed
    assert "cutoff_mismatch" in check.reason


def test_old_success_regression_forces_rejection():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[1] = receipt(
        parent,
        candidate_patch,
        "old_success_preservation",
        0.70,
        0.95,
        split=EvidenceSplit.OLD_SUCCESS,
    )
    assert certificate(
        parent, receipts=receipts, candidate_patch=candidate_patch
    ).decision is PromotionDecision.REJECT


def test_phase2_cannot_compile_a_language_patch_even_with_report_id():
    parent = initial_theory()
    language_patch = patch(parent, coordinate=PatchCoordinate.LANGUAGE)
    cert = certificate(parent, candidate_patch=language_patch)
    compiler = next(
        check for check in cert.checks if check.name == "single_coordinate_compiler"
    )
    assert not compiler.passed
    assert cert.decision is PromotionDecision.BRANCH_ONLY
    with pytest.raises(ValueError, match="no active compiler"):
        compile_patch(parent, language_patch)


def test_reduction_error_bound_is_enforced():
    parent = initial_theory()
    candidate_patch = patch(parent)
    cert = evaluate_conservative_extension(
        parent=parent,
        patch=candidate_patch,
        ledger=ledger(
            parent,
            candidate_patch,
            passing_receipts(parent, candidate_patch),
        ),
        reduction_map=reduction(parent, maximum_error=0.20),
    )
    assert cert.decision is PromotionDecision.BRANCH_ONLY
    assert not next(
        check for check in cert.checks if check.name == "reduction_map"
    ).passed


def test_authorization_fails_closed_without_external_trust_root():
    parent = initial_theory()
    candidate_patch = patch(parent)
    evidence = ledger(
        parent,
        candidate_patch,
        passing_receipts(parent, candidate_patch),
    )
    graph = branch_graph(parent, candidate_patch).record_evaluation(
        branch_id="branch_1",
        parent=parent,
        patch=candidate_patch,
        ledger=evidence,
        reduction_map=reduction(parent),
    )
    assert dict(graph.branch_statuses)["branch_1"] is FrameworkStatus.CANDIDATE
    with pytest.raises(ValueError, match="external trust root"):
        authorize_promotion(
            graph=graph,
            branch_id="branch_1",
            parent=parent,
            patch=candidate_patch,
            ledger=evidence,
            reduction_map=reduction(parent),
            promoter_actor_id="actor_promoter",
        )
    assert graph.states == (parent,)
    assert graph.edges == ()


def test_caller_supplied_certificate_is_not_an_authorization_api():
    with pytest.raises(ValueError, match="checks"):
        ConservativeExtensionCertificate(
            "parent",
            CANDIDATE,
            "patch",
            "patch_hash",
            "epoch",
            "reduction",
            "reduction_hash",
            (),
            "ledger",
            "policy",
            "child",
            (),
            PromotionDecision.ACTIVE_SCOPED,
            (),
        )


def test_promotion_requires_candidate_lifecycle_state():
    parent = initial_theory()
    candidate_patch = patch(parent)
    graph = branch_graph(
        parent,
        candidate_patch,
        status=FrameworkStatus.DRAFT,
    )
    with pytest.raises(ValueError, match="candidate_framework"):
        authorize_promotion(
            graph=graph,
            branch_id="branch_1",
            parent=parent,
            patch=candidate_patch,
            ledger=ledger(
                parent,
                candidate_patch,
                passing_receipts(parent, candidate_patch),
            ),
            reduction_map=reduction(parent),
            promoter_actor_id="actor_promoter",
        )


def test_only_assigned_promoter_can_authorize_graph_write():
    parent = initial_theory()
    candidate_patch = patch(parent)
    evidence = ledger(
        parent,
        candidate_patch,
        passing_receipts(parent, candidate_patch),
    )
    graph = branch_graph(parent, candidate_patch).record_evaluation(
        branch_id="branch_1",
        parent=parent,
        patch=candidate_patch,
        ledger=evidence,
        reduction_map=reduction(parent),
    )
    with pytest.raises(ValueError, match="promotion authority"):
        authorize_promotion(
            graph=graph,
            branch_id="branch_1",
            parent=parent,
            patch=candidate_patch,
            ledger=evidence,
            reduction_map=reduction(parent),
            promoter_actor_id="actor_generator",
        )


def test_nonfinite_receipt_is_rejected_at_construction():
    parent = initial_theory()
    candidate_patch = patch(parent)
    with pytest.raises(ValueError, match="finite"):
        receipt(
            parent,
            candidate_patch,
            "residual_explanation",
            float("inf"),
            0.75,
            split=EvidenceSplit.VALIDATION,
        )


def test_missing_hard_negative_coverage_blocks_gate():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[6] = receipt(
        parent,
        candidate_patch,
        "hard_negative_rejection",
        1.0,
        0.95,
        split=EvidenceSplit.HARD_NEGATIVE,
        observation_ids=(
            observation_id(
                "different_negative",
                EvidenceSplit.HARD_NEGATIVE,
                parent.data_cutoff,
            ),
        ),
    )
    cert = certificate(
        parent, receipts=receipts, candidate_patch=candidate_patch
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "missing_hard_negatives" in check.reason
    assert cert.decision is PromotionDecision.BRANCH_ONLY


def test_prediction_registered_after_outcome_is_rejected():
    parent = initial_theory()
    late_patch = patch(
        parent,
        pred=prediction(registered_at="2026-07-31T00:00:00+08:00"),
    )
    cert = certificate(parent, candidate_patch=late_patch)
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "prediction_not_preregistered" in check.reason


def test_illegal_lifecycle_transition_is_blocked():
    parent = initial_theory()
    candidate_patch = patch(parent)
    graph = branch_graph(
        parent,
        candidate_patch,
        status=FrameworkStatus.DRAFT,
    )
    with pytest.raises(ValueError, match="illegal"):
        graph.record_evaluation(
            branch_id="branch_1",
            parent=parent,
            patch=candidate_patch,
            ledger=ledger(
                parent,
                candidate_patch,
                passing_receipts(parent, candidate_patch),
            ),
            reduction_map=reduction(parent),
        )


def test_rejected_branch_retains_certificate_and_negative_evidence():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[0] = receipt(
        parent,
        candidate_patch,
        "residual_explanation",
        0.10,
        0.75,
        split=EvidenceSplit.VALIDATION,
    )
    cert = certificate(
        parent, receipts=receipts, candidate_patch=candidate_patch
    )
    assert cert.decision is PromotionDecision.REJECT
    graph = branch_graph(parent, candidate_patch).record_evaluation(
        branch_id="branch_1",
        parent=parent,
        patch=candidate_patch,
        ledger=ledger(parent, candidate_patch, receipts),
        reduction_map=reduction(parent),
    )
    assert dict(graph.branch_statuses)["branch_1"] is FrameworkStatus.REJECTED
    assert candidate_patch.hard_negative_ids[0] in graph.negative_evidence_ids
    assert cert.certificate_id in graph.certificate_ids


def test_generator_cannot_carry_holdout_semantic_receipt():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    holdout_id = receipts[5].observation_ids[0]
    receipts.append(
        receipt(
            parent,
            candidate_patch,
            "semantic_retrieval_score",
            0.77,
            0.50,
            split=EvidenceSplit.HOLDOUT,
            kind=EvidenceKind.SEMANTIC_RETRIEVAL,
            independent=False,
            observation_ids=(holdout_id,),
        )
    )
    cert = certificate(
        parent,
        receipts=receipts,
        candidate_patch=candidate_patch,
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "semantic_holdout_access" in check.reason


def test_failed_or_policy_mismatched_receipt_cannot_satisfy_gate():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[0] = receipt(
        parent,
        candidate_patch,
        "residual_explanation",
        0.90,
        0.95,
        split=EvidenceSplit.VALIDATION,
    )
    cert = certificate(
        parent,
        receipts=receipts,
        candidate_patch=candidate_patch,
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "wrong_receipt_threshold" in check.reason
    assert "failed_receipt" in check.reason
    assert cert.decision is PromotionDecision.BRANCH_ONLY


def test_proof_does_not_replace_holdout_evaluator_authority():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = list(passing_receipts(parent, candidate_patch))
    receipts[5] = receipt(
        parent,
        candidate_patch,
        "unseen_prediction_success",
        0.80,
        0.60,
        split=EvidenceSplit.HOLDOUT,
        kind=EvidenceKind.PROOF,
        actor="actor_formalizer",
        observation_ids=receipts[5].observation_ids,
    )
    cert = certificate(
        parent,
        receipts=receipts,
        candidate_patch=candidate_patch,
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "wrong_authority" in check.reason
    assert "empirical_metric_requires_execution" in check.reason


def test_every_preregistered_prediction_requires_holdout_evaluation():
    parent = initial_theory()
    first_patch = patch(parent)
    second_prediction = PreregisteredPrediction(
        "prediction_2",
        "second condition",
        "second outcome",
        "preserved",
        (0.60, 1.0),
        "score below 0.60",
        "2026-07-30T09:00:00+08:00",
    )
    candidate_patch = replace(
        first_patch,
        predictions=first_patch.predictions + (second_prediction,),
    )
    cert = certificate(parent, candidate_patch=candidate_patch)
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "incomplete_preregistered_prediction_coverage" in check.reason


def test_manifest_is_bound_to_exact_patch_context():
    parent = initial_theory()
    first_patch = patch(parent)
    receipts = passing_receipts(parent, first_patch)
    first_ledger = ledger(parent, first_patch, receipts)
    changed_patch = replace(first_patch, claim="different scoped claim")
    cert = evaluate_conservative_extension(
        parent=parent,
        patch=changed_patch,
        ledger=first_ledger,
        reduction_map=reduction(parent),
    )
    check = next(item for item in cert.checks if item.name == "evidence_contract")
    assert "sealed_manifest_context_binding_mismatch" in check.reason


def test_graph_rejects_mutable_collections_and_multiple_genesis_roots():
    parent = initial_theory()
    with pytest.raises(TypeError, match="immutable tuple"):
        TheoryVersionGraph(states=[parent])
    forged_root = replace(parent, scope=("forged_unreviewed_scope",))
    with pytest.raises(ValueError, match="exactly one genesis"):
        TheoryVersionGraph(states=(parent, forged_root))


def test_graph_rejects_direct_active_branch_without_replay():
    parent = initial_theory()
    candidate_patch = patch(parent)
    with pytest.raises(ValueError, match="replayable"):
        branch_graph(
            parent,
            candidate_patch,
            status=FrameworkStatus.ACTIVE_SCOPED,
        )


def test_graph_replays_full_inputs_and_rejects_forged_certificate():
    parent = initial_theory()
    candidate_patch = patch(parent)
    receipts = passing_receipts(parent, candidate_patch)
    evidence = ledger(parent, candidate_patch, receipts)
    cert = evaluate_conservative_extension(
        parent=parent,
        patch=candidate_patch,
        ledger=evidence,
        reduction_map=reduction(parent),
    )
    forged = replace(cert, gate_policy_id="forged_policy")
    event = EvaluationRecord(
        "branch_1",
        FrameworkStatus.CANDIDATE_BRANCH,
        FrameworkStatus.CANDIDATE,
        candidate_patch,
        evidence,
        reduction(parent),
        forged,
    )
    branch = replace(
        branch_graph(parent, candidate_patch).branches[0],
        status=FrameworkStatus.CANDIDATE,
    )
    with pytest.raises(ValueError, match="deterministic replay"):
        TheoryVersionGraph(
            states=(parent,),
            branches=(branch,),
            evaluation_records=(event,),
        )
