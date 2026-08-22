from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_c1_statistical_validation_v1 as validation
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/hegel_machine/phase2b_c1_statistical_validation_v1.py"
DOC = ROOT / "docs/Hegel_Machine_Phase2B_C1_Statistical_Validation_Profile_v1.md"


def _digest(tag: int) -> bytes:
    return hashlib.sha256(f"validation-test-{tag}".encode("ascii")).digest()


def _profile_bytes() -> bytes:
    return canonicalize_rfc8785_v1(validation._profile_payload_v1())


def _prerequisites() -> validation.OriginalPrerequisiteGraphV1:
    return validation.OriginalPrerequisiteGraphV1(
        r06_original_candidate_bytes=b'{"id":"r06-candidate"}',
        r06_original_result_bytes=b'{"id":"r06-result"}',
        r07_original_candidate_bytes=b'{"id":"r07-candidate"}',
        r07_original_result_bytes=b'{"id":"r07-result"}',
    )


def _keys():
    private_by_purpose = {}
    public_records = []
    for index, purpose in enumerate(validation.ValidationKeyPurposeV1, start=1):
        private = Ed25519PrivateKey.from_private_bytes(bytes([index]) * 32)
        raw_public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        key_id = validation._content_id(
            validation._KEY_PREFIX,
            validation._KEY_DOMAIN,
            {"purpose": purpose.value, "raw_public_key_hex": raw_public.hex()},
        )
        private_by_purpose[purpose] = private
        public_records.append(
            validation.ValidationCandidatePublicKeyV1(
                purpose=purpose,
                key_id=key_id,
                raw_public_key=raw_public,
            )
        )
    return tuple(public_records), private_by_purpose


def _seed_roots(version_index: int = 1):
    rows = []
    for index, purpose in enumerate(validation.ValidationSeedPurposeV1, start=1):
        raw = _digest(version_index * 100 + index)
        commitment = hashlib.sha256(
            validation._SEED_DOMAIN + purpose.value.encode("ascii") + b"\x00" + raw
        ).digest()
        rows.append(
            validation.ValidationSeedRootCandidateV1(
                purpose=purpose,
                raw_seed_root=raw,
                domain_commitment_sha256=commitment,
            )
        )
    return tuple(rows)


def _cohort(version_index: int = 1):
    base = version_index * 20
    return validation.ValidationCohortBindingV1(
        statistical_validation_profile_id=validation._profile_payload_v1()["profile_id"],
        prerequisite_graph_root_sha256=validation._prerequisite_root(_prerequisites()),
        primary_original_bytes_sha256=_digest(base + 1),
        primary_batch_root_sha256=_digest(base + 2),
        preservation_original_bytes_sha256=_digest(base + 3),
        preservation_batch_root_sha256=_digest(base + 4),
        generation_receipt_root_sha256=_digest(base + 5),
    )


def _systems(version_index: int = 1):
    return tuple(
        validation.ValidationSystemRunCandidateV1(
            system=system,
            primary_prediction_root_sha256=_digest(version_index * 40 + index),
            preservation_prediction_root_sha256=_digest(
                version_index * 40 + 10 + index
            ),
            raw_output_byte_count=100,
        )
        for index, system in enumerate(validation.ValidationSystemV1, start=1)
    )


def _main_rows(*, answerable_components_pass: bool = True):
    rows = []
    for family in validation._FAMILIES:
        for scale in validation._SCALES:
            for case_type, count in validation._CASE_TYPE_CELL_COUNTS:
                for ordinal in range(count):
                    answerable = case_type in validation._ANSWERABLE_CASE_TYPES
                    if answerable:
                        predicted = (
                            validation.ValidationPredictionDecisionV1.ANSWER_SET
                            if case_type
                            is validation.ValidationMainCaseTypeV1.ADMISSIBLE_SCALE_SET_ANSWERABLE
                            else validation.ValidationPredictionDecisionV1.ANSWER
                        )
                        family_exact = answerable_components_pass
                        binding_exact = True
                        scale_exact = True
                        decision_exact = True
                        joint_exact = answerable_components_pass
                    else:
                        predicted = validation.ValidationPredictionDecisionV1.ABSTAIN
                        family_exact = None
                        binding_exact = None
                        scale_exact = None
                        decision_exact = True
                        joint_exact = False
                    rows.append(
                        validation.MainEvaluationRowV1(
                            input_row_id=(
                                f"main:{family}:{scale}:{case_type.value}:{ordinal}"
                            ),
                            family=family,
                            scale=scale,
                            case_type=case_type,
                            cell_ordinal=ordinal,
                            predicted_decision=predicted,
                            family_exact=family_exact,
                            binding_exact=binding_exact,
                            scale_set_exact=scale_exact,
                            decision_exact=decision_exact,
                            joint_exact=joint_exact,
                        )
                    )
    return tuple(rows)


def _preservation_rows(*, legal_correct: int = 496, invalid_correct: int = 76):
    return tuple(
        validation.PreservationEvaluationRowV1(
            variant_row_id=f"variant:legal:{ordinal}",
            preservation_kind=validation.ValidationPreservationKindV1.LEGAL,
            kind_ordinal=ordinal,
            recognizer_correct=ordinal < legal_correct,
        )
        for ordinal in range(496)
    ) + tuple(
        validation.PreservationEvaluationRowV1(
            variant_row_id=f"variant:invalid:{ordinal}",
            preservation_kind=validation.ValidationPreservationKindV1.INVALID_CONTROL,
            kind_ordinal=ordinal,
            recognizer_correct=ordinal < invalid_correct,
        )
        for ordinal in range(76)
    )


def _challenge_rows(*, low_correct: int = 120, high_correct: int = 120):
    rows = []
    low_seen = 0
    high_seen = 0
    for family in validation._FAMILIES:
        for scale in validation._SCALES:
            for kind in validation.ValidationChallengeKindV1:
                for ordinal in range(10):
                    if kind is validation.ValidationChallengeKindV1.LOW_OVERLAP_STRUCTURAL_POSITIVE:
                        system_correct = low_seen < low_correct
                        low_seen += 1
                    else:
                        system_correct = high_seen < high_correct
                        high_seen += 1
                    baseline_correct = (
                        kind
                        is validation.ValidationChallengeKindV1.LOW_OVERLAP_STRUCTURAL_POSITIVE
                        and ordinal < 5
                    )
                    rows.append(
                        validation.ChallengeEvaluationRowV1(
                            input_row_id=(
                                f"challenge:{family}:{scale}:{kind.value}:{ordinal}"
                            ),
                            family=family,
                            scale=scale,
                            challenge_kind=kind,
                            cell_ordinal=ordinal,
                            system_correct=system_correct,
                            embedding_correct=baseline_correct,
                            frozen_llm_correct=baseline_correct,
                        )
                    )
    return tuple(rows)


def _reported_gates(main_rows, preservation_rows, total_event_count: int):
    def abstained(row):
        return row.predicted_decision is validation.ValidationPredictionDecisionV1.ABSTAIN

    answerable = tuple(
        row for row in main_rows if row.case_type in validation._ANSWERABLE_CASE_TYPES
    )
    counts = {
        "family_exact": sum(row.family_exact is True for row in answerable),
        "binding_exact": sum(row.binding_exact is True for row in answerable),
        "scale_set_accuracy": sum(row.scale_set_exact is True for row in answerable),
        "joint_exact": sum(row.joint_exact for row in answerable),
        "hard_negative_rejection": sum(
            abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.WRONG_FAMILY_HARD_NEGATIVE
        ),
        "binding_counterfactual_rejection": sum(
            abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.BINDING_COUNTERFACTUAL
        ),
        "scale_counterfactual_rejection": sum(
            abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.SCALE_COUNTERFACTUAL
        ),
        "sign_or_invariant_break_rejection": sum(
            abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.SIGN_OR_INVARIANT_BREAK
        ),
        "abstention_specificity": sum(
            not abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE
        ),
        "fail_closed_rate": total_event_count,
        "preservation_consistency": sum(
            row.recognizer_correct for row in preservation_rows
        ),
        "nonidentifiable_scale_abstention": sum(
            abstained(row)
            for row in main_rows
            if row.case_type
            is validation.ValidationMainCaseTypeV1.INSUFFICIENT_OR_NONIDENTIFIABLE
        ),
    }
    overall = tuple(
        validation.BinaryGateCandidateV1(
            metric=metric,
            scope_axis="overall",
            scope_value="all",
            correct_count=counts[metric],
            total_count=total_event_count if denominator is None else denominator,
        )
        for metric, denominator, *_ in validation._OVERALL_GATE_SPECS
    )
    slices = []
    for metric, axis, value, denominator, *_ in validation._expected_slice_layout():
        members = tuple(
            row
            for row in main_rows
            if (row.family if axis == "family" else row.scale) == value
        )
        if metric == "answerable_joint_exact":
            correct = sum(
                row.joint_exact
                for row in members
                if row.case_type in validation._ANSWERABLE_CASE_TYPES
            )
        elif metric == "all_control_rejection":
            correct = sum(
                abstained(row)
                for row in members
                if row.case_type in validation._CONTROL_CASE_TYPES
            )
        else:
            correct = sum(
                not abstained(row)
                for row in members
                if row.case_type
                is validation.ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE
            )
        slices.append(
            validation.BinaryGateCandidateV1(
                metric=metric,
                scope_axis=axis,
                scope_value=value,
                correct_count=correct,
                total_count=denominator,
            )
        )
    return overall, tuple(slices)


def _metric_bundle(
    total_event_count: int,
    *,
    pass_metrics: bool = True,
    legal_correct: int = 496,
    invalid_correct: int = 76,
    low_correct: int = 120,
    high_correct: int = 120,
):
    main_rows = _main_rows(answerable_components_pass=pass_metrics)
    preservation_rows = _preservation_rows(
        legal_correct=legal_correct,
        invalid_correct=invalid_correct,
    )
    overall, slices = _reported_gates(
        main_rows,
        preservation_rows,
        total_event_count,
    )
    zero = validation.ExactNonnegativeRationalV1(0, 1)
    regret_rows = tuple(
        validation.ScaleRegretEvaluationRowV1(
            input_row_id=row.input_row_id,
            family=row.family,
            scale=row.scale,
            regret=zero,
        )
        for row in main_rows
        if row.case_type in validation._ANSWERABLE_CASE_TYPES
    )
    return validation.StatisticalMetricBundleCandidateV1(
        main_rows=main_rows,
        preservation_rows=preservation_rows,
        scale_regret_rows=regret_rows,
        challenge_rows=_challenge_rows(
            low_correct=low_correct,
            high_correct=high_correct,
        ),
        overall_gates=overall,
        slice_gates=slices,
    )


def _execution_retry(terminal_code):
    if terminal_code in validation._RETRYABLE_INFRA_CODES:
        return (
            validation.ValidationExecutionRetryActionV1.REEXECUTE,
            validation.ValidationExecutionRetryReasonV1(terminal_code.value),
        )
    return (
        validation.ValidationExecutionRetryActionV1.NO_RETRY,
        validation.ValidationExecutionRetryReasonV1.NONE,
    )


def _provisional_attempt(
    *,
    version_index,
    attempt_index,
    terminal_code,
    execution_retry_action,
    execution_retry_reason,
    cohort,
    seed_roots,
    system_runs,
    metric_bundle,
    observer_root,
    any_raw_prediction_byte_produced,
    raw_output_byte_count,
    cas_receipt=None,
):
    return SimpleNamespace(
        validation_version_index=version_index,
        attempt_index=attempt_index,
        terminal_code=terminal_code,
        execution_retry_action=execution_retry_action,
        execution_retry_reason=execution_retry_reason,
        cohort=cohort,
        seed_roots=seed_roots,
        system_runs=system_runs,
        metric_bundle=metric_bundle,
        observer_transcript_root_sha256=observer_root,
        any_raw_prediction_byte_produced=any_raw_prediction_byte_produced,
        raw_output_byte_count=raw_output_byte_count,
        cas_receipt=cas_receipt,
    )


def _cas_receipt(provisional, keys, private_by_purpose, *, reupload: bool):
    archive_root = validation._attempt_archive_root(provisional)
    store_key = next(
        row for row in keys if row.purpose is validation.ValidationKeyPurposeV1.STORE
    )
    base = validation.ValidationCasReceiptCandidateV1(
        validation_version_index=provisional.validation_version_index,
        attempt_index=provisional.attempt_index,
        object_key=(
            "phase2b/c1/statistical-validation/v1/version-"
            f"{provisional.validation_version_index}/attempt-{provisional.attempt_index}"
        ),
        previous_generation=0,
        new_generation=1,
        value_digest_sha256=archive_root,
        upload_digest_attempts=(archive_root, archive_root) if reupload else (archive_root,),
        upload_retry_action=(
            validation.ValidationUploadRetryActionV1.REUPLOAD_COMMITTED_OUTPUT
            if reupload
            else validation.ValidationUploadRetryActionV1.NO_RETRY
        ),
        upload_retry_reason=(
            validation.ValidationUploadRetryReasonV1.OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED
            if reupload
            else validation.ValidationUploadRetryReasonV1.NONE
        ),
        store_key_id=store_key.key_id,
        signature=bytes(64),
        receipt_id="temporary",
    )
    with_id = replace(base, receipt_id=validation._cas_receipt_id(base))
    return replace(
        with_id,
        signature=private_by_purpose[validation.ValidationKeyPurposeV1.STORE].sign(
            validation._cas_signature_message(with_id)
        ),
    )


def _events(provisional, keys, private_by_purpose, logical_start: int, second_start: int):
    payloads = validation._expected_event_payloads(provisional)
    key_by_purpose = {row.purpose: row for row in keys}
    rows = []
    previous = None
    for sequence, (kind, purpose, payload) in enumerate(
        zip(validation._EVENT_KINDS, validation._EVENT_PURPOSES, payloads)
    ):
        base = validation.ValidationSignedEventCandidateV1(
            validation_version_index=provisional.validation_version_index,
            attempt_index=provisional.attempt_index,
            sequence=sequence,
            event_kind=kind,
            payload_root_sha256=payload,
            previous_event_id=previous,
            logical_time=logical_start + sequence,
            declared_unix_seconds=second_start + sequence,
            declared_nanosecond=sequence,
            signer_key_id=key_by_purpose[purpose].key_id,
            signature=bytes(64),
            event_id="temporary",
        )
        with_id = replace(base, event_id=validation._event_id(base))
        event = replace(
            with_id,
            signature=private_by_purpose[purpose].sign(
                validation._event_signature_message(with_id)
            ),
        )
        rows.append(event)
        previous = event.event_id
    return tuple(rows)


def _attempt(
    *,
    version_index: int,
    attempt_index: int,
    terminal_code: validation.ValidationAttemptTerminalCodeV1,
    total_event_count: int,
    keys,
    private_by_purpose,
    cohort=None,
    seed_roots=None,
    pass_metrics: bool = True,
    metric_bundle=None,
    logical_start: int = 0,
    second_start: int = 1_800_000_000,
    reupload: bool = False,
):
    cohort = _cohort(version_index) if cohort is None else cohort
    seed_roots = _seed_roots(version_index) if seed_roots is None else seed_roots
    complete = terminal_code in {
        validation.ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
        validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
    }
    system_runs = _systems(version_index) if complete else ()
    if complete and metric_bundle is None:
        metric_bundle = _metric_bundle(total_event_count, pass_metrics=pass_metrics)
    if not complete:
        metric_bundle = None
    raw_output_byte_count = 400 if complete else 0
    observer_root = _digest(version_index * 80 + attempt_index + 1)
    execution_retry_action, execution_retry_reason = _execution_retry(terminal_code)
    provisional = _provisional_attempt(
        version_index=version_index,
        attempt_index=attempt_index,
        terminal_code=terminal_code,
        execution_retry_action=execution_retry_action,
        execution_retry_reason=execution_retry_reason,
        cohort=cohort,
        seed_roots=seed_roots,
        system_runs=system_runs,
        metric_bundle=metric_bundle,
        observer_root=observer_root,
        any_raw_prediction_byte_produced=complete,
        raw_output_byte_count=raw_output_byte_count,
    )
    receipt = _cas_receipt(
        provisional,
        keys,
        private_by_purpose,
        reupload=reupload,
    )
    provisional.cas_receipt = receipt
    events = _events(
        provisional,
        keys,
        private_by_purpose,
        logical_start,
        second_start,
    )
    return validation.StatisticalValidationAttemptCandidateV1(
        validation_version_index=version_index,
        attempt_index=attempt_index,
        terminal_code=terminal_code,
        execution_retry_action=execution_retry_action,
        execution_retry_reason=execution_retry_reason,
        cohort=cohort,
        seed_roots=seed_roots,
        system_runs=system_runs,
        metric_bundle=metric_bundle,
        observer_transcript_root_sha256=observer_root,
        any_raw_prediction_byte_produced=complete,
        raw_output_byte_count=raw_output_byte_count,
        cas_receipt=receipt,
        events=events,
    )


def _graph(*, pass_gate: bool = True, retry: bool = False):
    keys, private_by_purpose = _keys()
    if retry:
        cohort = _cohort(1)
        seeds = _seed_roots(1)
        attempts = (
            _attempt(
                version_index=1,
                attempt_index=0,
                terminal_code=(
                    validation.ValidationAttemptTerminalCodeV1.CONTAINER_START_FAILURE
                ),
                total_event_count=12,
                keys=keys,
                private_by_purpose=private_by_purpose,
                cohort=cohort,
                seed_roots=seeds,
                logical_start=0,
            ),
            _attempt(
                version_index=1,
                attempt_index=1,
                terminal_code=validation.ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
                total_event_count=12,
                keys=keys,
                private_by_purpose=private_by_purpose,
                cohort=cohort,
                seed_roots=seeds,
                logical_start=6,
                second_start=1_800_000_006,
                reupload=True,
            ),
        )
    else:
        attempts = (
            _attempt(
                version_index=1,
                attempt_index=0,
                terminal_code=(
                    validation.ValidationAttemptTerminalCodeV1.COMPLETE_PASS
                    if pass_gate
                    else validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL
                ),
                total_event_count=6,
                keys=keys,
                private_by_purpose=private_by_purpose,
                pass_metrics=pass_gate,
            ),
        )
    return validation.StatisticalValidationCandidateGraphV1(
        profile_bytes=_profile_bytes(),
        prerequisites=_prerequisites(),
        candidate_keys=keys,
        versions=(
            validation.StatisticalValidationVersionCandidateV1(
                validation_version_index=1,
                attempts=attempts,
            ),
        ),
        version_bump=None,
    )


def _two_version_graph(*, version_two_pass: bool, reuse_seeds: bool = False):
    graph = _graph(pass_gate=False)
    keys = graph.candidate_keys
    _, private_by_purpose = _keys()
    version_one_attempt = graph.versions[0].attempts[-1]
    version_two_attempt = _attempt(
        version_index=2,
        attempt_index=0,
        terminal_code=(
            validation.ValidationAttemptTerminalCodeV1.COMPLETE_PASS
            if version_two_pass
            else validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL
        ),
        total_event_count=6,
        keys=keys,
        private_by_purpose=private_by_purpose,
        seed_roots=(
            version_one_attempt.seed_roots if reuse_seeds else _seed_roots(2)
        ),
        pass_metrics=version_two_pass,
        logical_start=6,
        second_start=1_800_000_006,
    )
    bump_base = validation.StatisticalValidationVersionBumpCandidateV1(
        from_validation_version_index=1,
        to_validation_version_index=2,
        old_protocol_version_id=validation._PROTOCOL_VERSION_IDS[0],
        new_protocol_version_id=validation._PROTOCOL_VERSION_IDS[1],
        old_selector_version_id=validation._SELECTOR_VERSION_IDS[0],
        new_selector_version_id=validation._SELECTOR_VERSION_IDS[1],
        old_validation_version_id=validation._VALIDATION_VERSION_IDS[0],
        new_validation_version_id=validation._VALIDATION_VERSION_IDS[1],
        immutable_version_one_failure_root_sha256=validation._attempt_archive_root(
            version_one_attempt
        ),
        version_one_failure_cas_receipt_id=version_one_attempt.cas_receipt.receipt_id,
        version_one_full_attempt_chain_root_sha256=(
            validation._version_attempt_chain_root(graph.versions[0])
        ),
        version_bump_core_id="temporary",
        public_development_change_set_sha256=_digest(7_777),
        public_failure_development_report_bytes=b"{}",
        version_two_issuance_authority_bytes=b"{}",
        version_two_cohort_root_sha256=validation._cohort_root(
            version_two_attempt.cohort
        ),
        version_two_seed_roots=tuple(
            row.raw_seed_root for row in version_two_attempt.seed_roots
        ),
        version_bump_id="temporary",
    )
    bump_with_core = replace(
        bump_base,
        version_bump_core_id=validation._version_bump_core_id(bump_base),
    )
    bump_with_artifacts = replace(
        bump_with_core,
        public_failure_development_report_bytes=canonicalize_rfc8785_v1(
            validation._public_failure_development_report_payload(
                bump_with_core,
                version_one_final=version_one_attempt,
                version_two_first=version_two_attempt,
            )
        ),
        version_two_issuance_authority_bytes=canonicalize_rfc8785_v1(
            validation._version_two_issuance_authority_payload(
                bump_with_core,
                version_one_final=version_one_attempt,
                version_two_first=version_two_attempt,
            )
        ),
    )
    bump = replace(
        bump_with_artifacts,
        version_bump_id=validation._version_bump_id(bump_with_artifacts),
    )
    return replace(
        graph,
        versions=(
            graph.versions[0],
            validation.StatisticalValidationVersionCandidateV1(
                validation_version_index=2,
                attempts=(version_two_attempt,),
            ),
        ),
        version_bump=bump,
    )


@pytest.fixture(scope="module")
def verified_pass_pair():
    graph = _graph()
    result = validation.verify_c1_statistical_validation_candidate_v1(graph)
    assert type(result) is validation.StatisticalValidationCandidateMechanicsResultV1
    return graph, result


def test_profile_bytes_are_strict_rfc8785_and_claims_remain_false():
    payload = _profile_bytes()
    result = validation.verify_c1_statistical_validation_profile_bytes_v1(payload)
    assert result.profile_candidate_mechanics_verified is True
    assert result.final_wire_pin_verified is False
    assert result.final_r06_pin_verified is False
    assert result.r08_satisfied is False
    assert payload[-1:] != b"\n"
    with pytest.raises(validation.StatisticalValidationProfileErrorV1) as exc_info:
        validation.verify_c1_statistical_validation_profile_bytes_v1(payload + b"\n")
    assert exc_info.value.code == "PROFILE_NOT_CANONICAL"
    with pytest.raises(validation.StatisticalValidationProfileErrorV1):
        validation.verify_c1_statistical_validation_profile_bytes_v1(
            b'{"profile_body":{},"profile_body":{},"profile_id":"x"}'
        )


def test_bootstrap_sampler_kats_rejection_types_and_nearest_rank_ties():
    root = bytes(range(32))
    assert validation.paired_cluster_bootstrap_indices_v1(
        cohort_root_sha256=root,
        metric_name="kat",
        replicate_index=0,
    ) == (8, 4, 2, 9, 5, 2, 0, 8, 7, 10, 11, 9)
    assert validation.paired_cluster_bootstrap_indices_v1(
        cohort_root_sha256=root,
        metric_name="kat",
        replicate_index=9_999,
    ) == (10, 3, 1, 4, 11, 6, 1, 8, 7, 1, 3, 3)
    with pytest.raises(TypeError):
        validation.paired_cluster_bootstrap_indices_v1(
            cohort_root_sha256=root,
            metric_name="kat",
            replicate_index=True,
        )
    tied = [Fraction(0, 1)] * 500 + [Fraction(1, 1)] * 9_000 + [Fraction(2, 1)] * 500
    assert validation._nearest_rank_value(tied.copy(), 499) == 0
    assert validation._nearest_rank_value(tied.copy(), 500) == 1
    assert validation._nearest_rank_value(tied.copy(), 9_499) == 1
    assert validation._nearest_rank_value(tied.copy(), 9_500) == 2


def test_every_binary_gate_boundary_and_special_wilson_boundaries():
    overall_minima = {
        "family_exact": 216,
        "binding_exact": 216,
        "scale_set_accuracy": 209,
        "joint_exact": 204,
        "hard_negative_rejection": 92,
        "binding_counterfactual_rejection": 92,
        "scale_counterfactual_rejection": 90,
        "sign_or_invariant_break_rejection": 92,
        "abstention_specificity": 217,
        "preservation_consistency": 555,
        "nonidentifiable_scale_abstention": 92,
    }
    for metric, denominator, point_n, point_d, lower_n, lower_d in validation._OVERALL_GATE_SPECS:
        if denominator is None:
            passing = validation.BinaryGateCandidateV1(metric, "overall", "all", 6, 6)
            failing = validation.BinaryGateCandidateV1(metric, "overall", "all", 5, 6)
        else:
            minimum = overall_minima[metric]
            passing = validation.BinaryGateCandidateV1(
                metric, "overall", "all", minimum, denominator
            )
            failing = validation.BinaryGateCandidateV1(
                metric, "overall", "all", minimum - 1, denominator
            )
        assert validation._binary_gate_pass(
            passing,
            point_n=point_n,
            point_d=point_d,
            lower_n=lower_n,
            lower_d=lower_d,
        )
        assert not validation._binary_gate_pass(
            failing,
            point_n=point_n,
            point_d=point_d,
            lower_n=lower_n,
            lower_d=lower_d,
        )

    slice_minima = {
        ("family", "answerable_joint_exact"): 33,
        ("family", "all_control_rejection"): 57,
        ("family", "abstention_specificity"): 33,
        ("scale", "answerable_joint_exact"): 96,
        ("scale", "all_control_rejection"): 169,
        ("scale", "abstention_specificity"): 97,
    }
    for metric, axis, value, denominator, point_n, point_d, lower_n, lower_d in validation._expected_slice_layout():
        minimum = slice_minima[(axis, metric)]
        for correct, expected in ((minimum, True), (minimum - 1, False)):
            row = validation.BinaryGateCandidateV1(
                metric, axis, value, correct, denominator
            )
            assert validation._binary_gate_pass(
                row,
                point_n=point_n,
                point_d=point_d,
                lower_n=lower_n,
                lower_d=lower_d,
            ) is expected

    assert 482 * 100 >= 496 * 97
    assert validation._wilson_lower(482, 496) >= 0.94
    assert 481 * 100 < 496 * 97
    assert validation._wilson_lower(109, 120) >= 0.85
    assert validation._wilson_lower(108, 120) < 0.85
    assert validation._wilson_lower(114, 120) >= 0.90
    assert validation._wilson_lower(113, 120) < 0.90


def test_complete_pass_is_only_candidate_mechanics_and_fresh_result_verifies(
    verified_pass_pair,
):
    graph, result = verified_pass_pair
    assert result.outcome is validation.StatisticalValidationOutcomeV1.COMPLETE_PASS
    assert result.verified_signature_count == 7
    assert result.total_event_count == 6
    assert result.fail_closed_denominators_by_version == (6,)
    assert result.fail_closed_correct_by_version == (6,)
    assert result.aggregate_fail_closed_passed is True
    assert len(result.per_version_outcome_gate_root_vector) == 1
    assert result.per_version_outcome_gate_root_vector[0][0] is (
        validation.StatisticalValidationOutcomeV1.COMPLETE_PASS
    )
    assert len(result.per_version_outcome_gate_root_vector[0][1]) == 64
    assert result.evaluator_manifest_row_counts == (720, 240, 572, 240)
    assert result.scale_regret_point == (0, 1)
    assert result.scale_regret_bootstrap_upper == (0, 1)
    assert result.semantic_advantage_point == (3, 4)
    assert result.semantic_advantage_bootstrap_lower == (3, 4)
    assert all(result.overall_gate_passes)
    assert all(result.slice_gate_passes)
    assert all(result.challenge_gate_passes)
    assert result.final_validation_version_all_statistical_gates_passed is True
    assert result.r06_qualified is False
    assert result.r07_frozen is False
    assert result.r08_satisfied is False
    assert result.actual_execution_verified is False
    assert result.durable_cas_verified is False
    assert result.authoritative_time_verified is False
    assert validation.verify_c1_statistical_validation_candidate_result_v1(
        candidate=graph,
        result=result,
    )


def test_statistical_failure_is_complete_fail_not_retryable():
    result = validation.verify_c1_statistical_validation_candidate_v1(
        _graph(pass_gate=False)
    )
    assert type(result) is validation.StatisticalValidationCandidateMechanicsResultV1
    assert result.outcome is validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL
    assert result.per_version_outcome_gate_root_vector[0][0] is (
        validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL
    )
    assert result.overall_gate_passes[0] is False
    assert result.final_validation_version_all_statistical_gates_passed is False
    assert result.r08_satisfied is False


def test_exact_zero_byte_reexecute_and_same_digest_reupload_are_distinct():
    assert {item.value for item in validation.ValidationAttemptTerminalCodeV1} == {
        "COMPLETE_PASS",
        "COMPLETE_FAIL",
        "CONTAINER_START_FAILURE",
        "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
        "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
        "INPUT_TRANSFER_CHECKSUM_MISMATCH",
    }
    assert {item.value for item in validation.ValidationExecutionRetryReasonV1} == {
        "NONE",
        "CONTAINER_START_FAILURE",
        "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
        "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
        "INPUT_TRANSFER_CHECKSUM_MISMATCH",
    }
    assert {item.value for item in validation.ValidationUploadRetryActionV1} == {
        "NO_RETRY",
        "REUPLOAD_COMMITTED_OUTPUT",
    }
    assert {item.value for item in validation.ValidationUploadRetryReasonV1} == {
        "NONE",
        "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
    }
    keys, private_by_purpose = _keys()
    for code in validation._RETRYABLE_INFRA_CODES:
        attempt = _attempt(
            version_index=1,
            attempt_index=0,
            terminal_code=code,
            total_event_count=6,
            keys=keys,
            private_by_purpose=private_by_purpose,
        )
        assert attempt.execution_retry_action is (
            validation.ValidationExecutionRetryActionV1.REEXECUTE
        )
        assert attempt.execution_retry_reason.value == code.value
        assert attempt.any_raw_prediction_byte_produced is False
    graph = _graph(retry=True)
    first, second = graph.versions[0].attempts
    assert first.execution_retry_action is validation.ValidationExecutionRetryActionV1.REEXECUTE
    assert first.any_raw_prediction_byte_produced is False
    assert first.execution_retry_reason is (
        validation.ValidationExecutionRetryReasonV1.CONTAINER_START_FAILURE
    )
    assert second.cas_receipt.upload_retry_action is (
        validation.ValidationUploadRetryActionV1.REUPLOAD_COMMITTED_OUTPUT
    )
    assert second.cas_receipt.upload_retry_reason is (
        validation.ValidationUploadRetryReasonV1.OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED
    )
    assert second.cas_receipt.upload_digest_attempts == (
        second.cas_receipt.value_digest_sha256,
        second.cas_receipt.value_digest_sha256,
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(graph)
    assert type(result) is validation.StatisticalValidationCandidateMechanicsResultV1
    assert result.outcome is validation.StatisticalValidationOutcomeV1.COMPLETE_PASS
    assert result.final_attempt_index == 1
    assert result.fail_closed_denominators_by_version == (12,)
    assert result.total_event_count == 12
    assert result.verified_signature_count == 14


def test_any_raw_byte_statistical_retry_or_retry_binding_change_rejects():
    raw_byte_graph = copy.deepcopy(_graph(retry=True))
    first = raw_byte_graph.versions[0].attempts[0]
    object.__setattr__(first, "raw_output_byte_count", 1)
    result = validation.verify_c1_statistical_validation_candidate_v1(raw_byte_graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    raw_prediction_graph = copy.deepcopy(_graph(retry=True))
    object.__setattr__(
        raw_prediction_graph.versions[0].attempts[0],
        "any_raw_prediction_byte_produced",
        True,
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        raw_prediction_graph
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    parser_failure_graph = copy.deepcopy(_graph(retry=True))
    object.__setattr__(
        parser_failure_graph.versions[0].attempts[0],
        "terminal_code",
        "PARSER_FAILURE_BEFORE_FIRST_PREDICTION_BYTE",
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        parser_failure_graph
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    statistical_retry_graph = copy.deepcopy(_graph(retry=True))
    first = statistical_retry_graph.versions[0].attempts[0]
    object.__setattr__(
        first,
        "terminal_code",
        validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        statistical_retry_graph
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    changed_binding_graph = copy.deepcopy(_graph(retry=True))
    second = changed_binding_graph.versions[0].attempts[1]
    object.__setattr__(second, "seed_roots", _seed_roots(2))
    result = validation.verify_c1_statistical_validation_candidate_v1(
        changed_binding_graph
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1
    assert result.partial_output_published is False

    upload_action_graph = copy.deepcopy(_graph(retry=True))
    object.__setattr__(
        upload_action_graph.versions[0].attempts[1].cas_receipt,
        "upload_retry_action",
        validation.ValidationUploadRetryActionV1.NO_RETRY,
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        upload_action_graph
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1


def test_signature_cas_time_and_concurrent_mutation_attacks_fail_closed(monkeypatch):
    graph = _graph()
    attempt = graph.versions[0].attempts[0]
    bad_event = replace(
        attempt.events[2],
        signature=bytes([attempt.events[2].signature[0] ^ 1])
        + attempt.events[2].signature[1:],
    )
    object.__setattr__(attempt, "events", attempt.events[:2] + (bad_event,) + attempt.events[3:])
    result = validation.verify_c1_statistical_validation_candidate_v1(graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1
    assert result.reason is validation.StatisticalValidationCandidateReasonV1.SIGNATURE_INVALID
    assert result.partial_output_published is False

    cas_graph = _graph()
    object.__setattr__(
        cas_graph.versions[0].attempts[0].cas_receipt,
        "previous_generation",
        1,
    )
    cas_result = validation.verify_c1_statistical_validation_candidate_v1(cas_graph)
    assert type(cas_result) is validation.StatisticalValidationCandidateRejectionV1

    time_graph = _graph()
    object.__setattr__(time_graph.versions[0].attempts[0].events[0], "logical_time", True)
    time_result = validation.verify_c1_statistical_validation_candidate_v1(time_graph)
    assert type(time_result) is validation.StatisticalValidationCandidateRejectionV1

    mutation_graph = _graph()
    original_verify = validation._verify_ed25519_signature
    mutated = False

    def verify_then_mutate(raw_public_key, signature, message):
        nonlocal mutated
        original_verify(raw_public_key, signature, message)
        if not mutated:
            mutated = True
            object.__setattr__(
                mutation_graph,
                "profile_bytes",
                mutation_graph.profile_bytes + b" ",
            )

    monkeypatch.setattr(validation, "_verify_ed25519_signature", verify_then_mutate)
    mutation_result = validation.verify_c1_statistical_validation_candidate_v1(
        mutation_graph
    )
    assert type(mutation_result) is validation.StatisticalValidationCandidateRejectionV1
    assert mutation_result.reason is (
        validation.StatisticalValidationCandidateReasonV1.INPUT_MUTATED_DURING_VERIFICATION
    )


def test_metric_manifest_recomputation_rejects_aggregate_slice_and_row_algebra_attacks():
    keys, private_by_purpose = _keys()
    base_bundle = _metric_bundle(6)
    bad_overall = replace(
        base_bundle.overall_gates[0],
        correct_count=base_bundle.overall_gates[0].correct_count - 1,
    )
    aggregate_bundle = replace(
        base_bundle,
        overall_gates=(bad_overall,) + base_bundle.overall_gates[1:],
    )
    aggregate_attempt = _attempt(
        version_index=1,
        attempt_index=0,
        terminal_code=validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
        total_event_count=6,
        keys=keys,
        private_by_purpose=private_by_purpose,
        metric_bundle=aggregate_bundle,
    )
    aggregate_graph = validation.StatisticalValidationCandidateGraphV1(
        profile_bytes=_profile_bytes(),
        prerequisites=_prerequisites(),
        candidate_keys=keys,
        versions=(
            validation.StatisticalValidationVersionCandidateV1(1, (aggregate_attempt,)),
        ),
        version_bump=None,
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(aggregate_graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    bad_slice = replace(
        base_bundle.slice_gates[0],
        correct_count=base_bundle.slice_gates[0].correct_count - 1,
    )
    slice_bundle = replace(
        base_bundle,
        slice_gates=(bad_slice,) + base_bundle.slice_gates[1:],
    )
    slice_attempt = _attempt(
        version_index=1,
        attempt_index=0,
        terminal_code=validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
        total_event_count=6,
        keys=keys,
        private_by_purpose=private_by_purpose,
        metric_bundle=slice_bundle,
    )
    slice_graph = replace(
        aggregate_graph,
        versions=(
            validation.StatisticalValidationVersionCandidateV1(1, (slice_attempt,)),
        ),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(slice_graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    algebra_bundle = copy.deepcopy(base_bundle)
    object.__setattr__(algebra_bundle.main_rows[0], "family_exact", False)
    algebra_attempt = _attempt(
        version_index=1,
        attempt_index=0,
        terminal_code=validation.ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
        total_event_count=6,
        keys=keys,
        private_by_purpose=private_by_purpose,
        metric_bundle=algebra_bundle,
    )
    algebra_graph = replace(
        aggregate_graph,
        versions=(
            validation.StatisticalValidationVersionCandidateV1(1, (algebra_attempt,)),
        ),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(algebra_graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1


def test_full_version_bump_pass_and_version_two_failure_no_go():
    passing_graph = _two_version_graph(version_two_pass=True)
    bump = passing_graph.version_bump
    assert bump is not None
    assert (
        bump.old_protocol_version_id,
        bump.new_protocol_version_id,
    ) == validation._PROTOCOL_VERSION_IDS
    assert (
        bump.old_selector_version_id,
        bump.new_selector_version_id,
    ) == validation._SELECTOR_VERSION_IDS
    assert (
        bump.old_validation_version_id,
        bump.new_validation_version_id,
    ) == validation._VALIDATION_VERSION_IDS
    report = json.loads(bump.public_failure_development_report_bytes)
    issuance = json.loads(bump.version_two_issuance_authority_bytes)
    for payload in (report, issuance):
        bindings = payload["bindings"]
        assert bindings["immutable_version_one_failure_root_sha256"] == (
            bump.immutable_version_one_failure_root_sha256.hex()
        )
        assert bindings["version_one_failure_cas_receipt_id"] == (
            bump.version_one_failure_cas_receipt_id
        )
        assert bindings["version_one_full_attempt_chain_root_sha256"] == (
            bump.version_one_full_attempt_chain_root_sha256.hex()
        )
        assert bindings["version_bump_core_id"] == bump.version_bump_core_id
        assert bindings["version_two_cohort_root_sha256"] == (
            bump.version_two_cohort_root_sha256.hex()
        )
        assert bindings["version_two_seed_roots"] == [
            value.hex() for value in bump.version_two_seed_roots
        ]
    passing = validation.verify_c1_statistical_validation_candidate_v1(
        passing_graph
    )
    assert type(passing) is validation.StatisticalValidationCandidateMechanicsResultV1
    assert passing.outcome is validation.StatisticalValidationOutcomeV1.COMPLETE_PASS
    assert passing.validation_version_count == 2
    assert passing.version_bump_id is not None
    assert passing.fail_closed_denominators_by_version == (6, 6)
    assert passing.no_go_after_version_two_failure is False
    assert tuple(row[0] for row in passing.per_version_outcome_gate_root_vector) == (
        validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL,
        validation.StatisticalValidationOutcomeV1.COMPLETE_PASS,
    )
    assert passing.final_validation_version_all_statistical_gates_passed is True
    assert passing.r08_satisfied is False

    failing = validation.verify_c1_statistical_validation_candidate_v1(
        _two_version_graph(version_two_pass=False)
    )
    assert type(failing) is validation.StatisticalValidationCandidateMechanicsResultV1
    assert failing.outcome is validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL
    assert failing.no_go_after_version_two_failure is True
    assert failing.validation_version_count == 2
    assert tuple(row[0] for row in failing.per_version_outcome_gate_root_vector) == (
        validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL,
        validation.StatisticalValidationOutcomeV1.COMPLETE_FAIL,
    )
    assert failing.final_validation_version_all_statistical_gates_passed is False
    assert failing.r08_satisfied is False


def test_version_bump_requires_fresh_seeds_immutable_failure_and_canonical_authority():
    reused = validation.verify_c1_statistical_validation_candidate_v1(
        _two_version_graph(version_two_pass=True, reuse_seeds=True)
    )
    assert type(reused) is validation.StatisticalValidationCandidateRejectionV1

    missing = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(missing, "version_bump", None)
    result = validation.verify_c1_statistical_validation_candidate_v1(missing)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    stale_failure = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        stale_failure.version_bump,
        "immutable_version_one_failure_root_sha256",
        _digest(9_999),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(stale_failure)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    stale_attempt_chain = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        stale_attempt_chain.version_bump,
        "version_one_full_attempt_chain_root_sha256",
        _digest(9_998),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        stale_attempt_chain
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    noncanonical = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        noncanonical.version_bump,
        "version_two_issuance_authority_bytes",
        b'{"z":0, "a":1}',
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(noncanonical)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    arbitrary = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        arbitrary.version_bump,
        "public_failure_development_report_bytes",
        b'{"arbitrary":true}',
    )
    object.__setattr__(
        arbitrary.version_bump,
        "version_bump_id",
        validation._version_bump_id(arbitrary.version_bump),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(arbitrary)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    arbitrary_issuance = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        arbitrary_issuance.version_bump,
        "version_two_issuance_authority_bytes",
        b'{"arbitrary":true}',
    )
    object.__setattr__(
        arbitrary_issuance.version_bump,
        "version_bump_id",
        validation._version_bump_id(arbitrary_issuance.version_bump),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(
        arbitrary_issuance
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    partial_bump = copy.deepcopy(_two_version_graph(version_two_pass=True))
    object.__setattr__(
        partial_bump.version_bump,
        "new_selector_version_id",
        validation._SELECTOR_VERSION_IDS[0],
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(partial_bump)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1


def test_exact_integer_and_resource_preflight_happen_before_snapshot(monkeypatch):
    calls = 0

    def forbidden_snapshot(_candidate):
        nonlocal calls
        calls += 1
        raise AssertionError("snapshot must not run after preflight rejection")

    monkeypatch.setattr(validation, "_snapshot_bytes", forbidden_snapshot)

    too_many = _graph()
    attempt = too_many.versions[0].attempts[0]
    object.__setattr__(attempt, "system_runs", attempt.system_runs + (attempt.system_runs[0],))
    result = validation.verify_c1_statistical_validation_candidate_v1(too_many)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    oversized = _graph()
    attempt = oversized.versions[0].attempts[0]
    object.__setattr__(
        attempt.system_runs[0],
        "raw_output_byte_count",
        validation._MAX_DECLARED_SYSTEM_OUTPUT_BYTES + 1,
    )
    object.__setattr__(
        attempt,
        "raw_output_byte_count",
        sum(row.raw_output_byte_count for row in attempt.system_runs),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(oversized)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    bool_index = _graph()
    object.__setattr__(bool_index.versions[0], "validation_version_index", True)
    result = validation.verify_c1_statistical_validation_candidate_v1(bool_index)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    long_id = _graph()
    object.__setattr__(
        long_id.versions[0].attempts[0].metric_bundle.main_rows[0],
        "input_row_id",
        "x" * (validation._MAX_SHORT_ID_BYTES + 1),
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(long_id)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1

    canonical_budget = _graph()
    monkeypatch.setattr(validation, "_MAX_SNAPSHOT_BYTES", 1)
    result = validation.verify_c1_statistical_validation_candidate_v1(
        canonical_budget
    )
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1
    assert result.reason is (
        validation.StatisticalValidationCandidateReasonV1.RESOURCE_BUDGET_EXCEEDED
    )
    assert calls == 0


def test_fresh_result_verifier_rejects_hash_cardinality_plain_string_and_custom_equality(
    verified_pass_pair,
    monkeypatch,
):
    graph, result = verified_pass_pair
    original_projection = validation._result_projection_from_object

    deep: object = True
    for _ in range(4_096):
        deep = (deep,)
    projection_calls = 0

    def forbidden_projection(_result):
        nonlocal projection_calls
        projection_calls += 1
        raise AssertionError("projection must follow bounded shape preflight")

    monkeypatch.setattr(
        validation,
        "_result_projection_from_object",
        forbidden_projection,
    )
    deeply_nested = copy.copy(result)
    object.__setattr__(
        deeply_nested,
        "overall_gate_passes",
        (deep,) + (True,) * 11,
    )
    assert not validation.verify_c1_statistical_validation_candidate_result_v1(
        candidate=graph,
        result=deeply_nested,
    )
    assert projection_calls == 0
    monkeypatch.setattr(
        validation,
        "_result_projection_from_object",
        original_projection,
    )

    oversized = copy.copy(result)
    object.__setattr__(
        oversized,
        "profile_id",
        "x" * (validation._MAX_SHORT_ID_BYTES + 1),
    )
    assert not validation.verify_c1_statistical_validation_candidate_result_v1(
        candidate=graph,
        result=oversized,
    )

    for raised in (RecursionError, MemoryError):
        def resource_failure(_result, exception_type=raised):
            raise exception_type

        monkeypatch.setattr(
            validation,
            "_result_projection_from_object",
            resource_failure,
        )
        assert not validation.verify_c1_statistical_validation_candidate_result_v1(
            candidate=graph,
            result=result,
        )
    monkeypatch.setattr(
        validation,
        "_result_projection_from_object",
        original_projection,
    )

    class EqualityTrap:
        def __eq__(self, _other):
            raise AssertionError("custom equality must not be invoked")

    forged = copy.copy(result)
    object.__setattr__(forged, "overall_gate_passes", (EqualityTrap(),) * 12)
    assert not validation.verify_c1_statistical_validation_candidate_result_v1(
        candidate=graph,
        result=forged,
    )

    for field, value in (
        ("outcome", "COMPLETE_PASS"),
        ("total_event_count", True),
        ("slice_gate_passes", (True,) * 23),
        (
            "per_version_outcome_gate_root_vector",
            ((validation.StatisticalValidationOutcomeV1.COMPLETE_PASS, "0" * 64),),
        ),
        ("final_metric_bundle_root_sha256", "0" * 64),
        ("result_id", result.result_id[:-1] + ("0" if result.result_id[-1] != "0" else "1")),
    ):
        forged = copy.copy(result)
        object.__setattr__(forged, field, value)
        assert not validation.verify_c1_statistical_validation_candidate_result_v1(
            candidate=graph,
            result=forged,
        )


def test_original_prerequisite_bytes_must_be_canonical_and_cross_bound():
    graph = _graph()
    object.__setattr__(
        graph.prerequisites,
        "r06_original_candidate_bytes",
        b'{"z":0, "a":1}',
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(graph)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1
    assert result.partial_output_published is False

    transplanted = _graph()
    object.__setattr__(
        transplanted.prerequisites,
        "r06_original_candidate_bytes",
        b'{"id":"different-r06-candidate"}',
    )
    result = validation.verify_c1_statistical_validation_candidate_v1(transplanted)
    assert type(result) is validation.StatisticalValidationCandidateRejectionV1


def test_public_surface_is_closed_and_module_has_no_effectful_primitive():
    expected = {
        "verify_c1_statistical_validation_profile_bytes_v1",
        "paired_cluster_bootstrap_indices_v1",
        "verify_c1_statistical_validation_candidate_v1",
        "verify_c1_statistical_validation_candidate_result_v1",
    }
    assert expected <= set(validation.__all__)
    source = SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "subprocess",
        "requests.",
        "socket.",
        "random.",
        "secrets.",
        "time.time",
        "os.urandom",
        "docker ",
        "Path(",
        "open(",
    )
    assert not any(token in source for token in forbidden)
    assert "PENDING" not in source
    assert "phase2b_formal_trusted_wire_v3_" not in source
    assert "phase2b_actual_unsealed_960_generator_construction_profile_v1_" not in source


def test_docs_state_candidate_only_boundary_and_closed_retry_version_semantics():
    text = DOC.read_text(encoding="utf-8")
    for phrase in (
        "STATISTICAL_VALIDATION_CANDIDATE_MECHANICS_VERIFIED_NOT_R08",
        "960",
        "572",
        "496",
        "76",
        "10,000",
        "COMPLETE_PASS",
        "COMPLETE_FAIL",
        "REJECTED",
        "REEXECUTE",
        "REUPLOAD_COMMITTED_OUTPUT",
        "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
        "NO-GO",
        "does not satisfy R08",
    ):
        assert phrase in text
