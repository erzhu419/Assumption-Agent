"""One full synthetic raw-960 scorer pipeline test.

The graph built here is deliberately local and unbacked.  It exercises public
wire decoders and the frozen scorer/replay/gate mechanics, but it is not an
actual recognizer run, an authority ceremony, formal scoring evidence, effect
evidence, or C1 exit evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import hegel_machine.phase2b_actual_unsealed_960_replay_input_contract_v2 as replay_v2
import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as answer_v2
import hegel_machine.phase2b_recognizer_input_archive_v2 as input_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v2 as prediction_v2
import hegel_machine.phase2b_recognizer_runtime_cli_v2 as runtime_v2
import hegel_machine.phase2b_scorer_bundle_v1 as bundle_v1
import hegel_machine.phase2b_trusted_wire_batch_v2 as batch_v2
import hegel_machine.phase2b_unsealed_960_available_gate_mechanics_v2 as gate_v2
import hegel_machine.phase2b_unsealed_960_prediction_scoring_mechanics_v2 as scoring_v2
import hegel_machine.phase2b_unsealed_prediction_evaluator_v2 as evaluator_v2
from hegel_machine.phase2b_protocol import salted_answer_commitment_sha256


CODEC_SOURCE_SHA256 = (
    "8c2c6ae00d55274f38e8f346b2ababd2beaaee25ade9e9fccd23f4db7d7d63e1"
)
SCORING_SOURCE_SHA256 = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
EXPECTED_BUNDLE_FALSE_CLAIMS = (
    "actual_960_case_run_verified",
    "authority_requirement_satisfied",
    "execution_registry_verified",
    "campaign_authority_verified",
    "attempt_authority_verified",
    "organizational_independence_verified",
    "independent_human_actors_verified",
    "answer_authority_verified",
    "gate_input_authority_verified",
    "prediction_receipt_signature_verified",
    "reveal_token_signature_verified",
    "scorer_image_runtime_verified",
    "scorer_source_root_verified",
    "pre_reveal_commitment_timing_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "rollback_resistance_verified",
    "authoritative_time_verified",
    "input_archive_membership_verified",
    "source_registry_projection_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "recognizer_executed",
    "runtime_executed",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "c1_exit_evidence",
)
COMMON_AUTHORITY_AND_ACTUAL_NONCLAIMS = (
    "formal_gate_evaluation_performed",
    "answer_manifest_authority_verified",
    "answer_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified",
    "input_archive_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "effect_evidence",
    "c1_exit_evidence",
)


def test_full_synthetic_raw_960_bundle_encode_decode_evaluate_pipeline() -> None:
    """Cross the real public raw-wire and mechanics pipeline without stubs."""

    tests_root = Path(__file__).resolve().parent
    source_root = Path(bundle_v1.__file__).resolve().parent
    assert hashlib.sha256(Path(bundle_v1.__file__).read_bytes()).hexdigest() == (
        CODEC_SOURCE_SHA256
    )
    assert bundle_v1.PHASE2B_SCORING_SOURCE_SHA256_V1 == SCORING_SOURCE_SHA256
    assert hashlib.sha256(
        (source_root / "phase2b_unsealed_960_prediction_scoring_mechanics_v2.py")
        .read_bytes()
    ).hexdigest() == SCORING_SOURCE_SHA256

    # Reuse only checked synthetic source constructors.  The resulting 960-row
    # input is nevertheless genuine public wire: every later decoder sees its
    # bytes, not a forged DecodedRecognizerInputArchiveV2 surface.
    input_fixture = runpy.run_path(
        str(tests_root / "test_phase2b_recognizer_input_archive_v1.py")
    )
    source_registry, source_authority = input_fixture["_six_family_source_fixture"]()
    count = 960
    run_id = b"F" * 32
    keys = batch_v2.TrustedWireKeySourcesV2(b"S" * 32, b"I" * 32, b"P" * 32)
    source_authorities = (source_authority,) * count
    trusted_batch = batch_v2.build_trusted_wire_batch_v2(
        authorities=source_authorities,
        run_id=run_id,
        key_sources=keys,
    )
    assert type(trusted_batch) is batch_v2.TrustedWireBatchV2
    source_case = input_v2.TrustedRecognizerSourceCaseV2(
        authority=source_authority,
        adapter_registry=source_registry,
    )
    decoded_input = input_v2.issue_trusted_recognizer_input_archive_v2(
        batch=trusted_batch,
        run_id=run_id,
        key_sources=keys,
        source_cases=(source_case,) * count,
    )
    assert type(decoded_input) is input_v2.DecodedRecognizerInputArchiveV2
    input_archive_bytes = decoded_input.archive
    assert len(decoded_input.rows) == len(set(decoded_input.row_ids)) == count
    assert (
        input_v2.decode_public_recognizer_input_archive_v2(input_archive_bytes)
        == decoded_input
    )

    prediction_fixture = runpy.run_path(
        str(tests_root / "test_phase2b_recognizer_prediction_archive_v2.py")
    )
    execution_freeze = prediction_fixture[
        "execution_freeze_manifest"
    ].__wrapped__()
    execution_freeze_bytes = runtime_v2.encode_execution_freeze_manifest_json_v2(
        execution_freeze
    )
    context = prediction_v2.PublicPredictionRunContextV2._issue(
        prediction_v2._CONTEXT_ISSUE_TOKEN_V2,
        batch_id=decoded_input.batch_id,
        input_archive_id=decoded_input.archive_id,
        input_archive_sha256=hashlib.sha256(input_archive_bytes).hexdigest(),
        input_row_ids_root=prediction_v2._input_row_ids_root_v2(
            decoded_input.row_ids
        ),
        execution_freeze_manifest_id=execution_freeze.manifest_id,
    )
    prediction_records = tuple(
        prediction_v2.PublicRecognizerPredictionRecordV2._issue(
            prediction_v2._RECORD_ISSUE_TOKEN_V2,
            context=context,
            input_row=row,
            outcome=prediction_fixture["_outcome"](
                row_id=row.row_id,
                payload_sha256=row.payload_sha256,
                freeze_manifest_id=execution_freeze.manifest_id,
                index=index,
            ),
        )
        for index, row in enumerate(decoded_input.rows)
    )
    prediction_archive_bytes = prediction_v2._encode_prediction_archive_v2(
        context=context,
        records=prediction_records,
    )
    decoded_prediction = (
        prediction_v2.decode_public_recognizer_prediction_archive_v2(
            prediction_archive_bytes
        )
    )
    assert len(decoded_prediction.records) == count

    main_row_ids = tuple(sorted(decoded_prediction.input_row_ids)[:720])
    semantic_conflict_row_ids = tuple(
        sorted(decoded_prediction.input_row_ids)[720:]
    )
    structural_receipt = bundle_v1.verify_strict_recognizer_io_structure_v2(
        input_archive=input_archive_bytes,
        prediction_archive=prediction_archive_bytes,
    )
    partition = evaluator_v2.build_unsealed_prediction_partition_manifest_v2(
        prediction_archive=decoded_prediction,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
    )

    replay_fixture = runpy.run_path(
        str(tests_root / "test_phase2b_actual_unsealed_960_replay_input_contract_v2.py")
    )
    template = replay_fixture["_synthetic_fixture_base"]()
    answer_rows = []
    row_index = 0
    for family_id, _scale_slice, assignments in template.cells:
        for case_type, _margin in assignments:
            answer_rows.append(
                replay_fixture["_synthetic_answer_row"](
                    index=row_index,
                    input_row_id=main_row_ids[row_index],
                    case_type=case_type,
                    family_id=family_id,
                )
            )
            row_index += 1
    assert row_index == len(answer_rows) == 720
    answer_manifest = answer_v2.build_formal_unsealed_answer_manifest_v2(
        input_archive_id=structural_receipt.input_archive_id,
        input_archive_sha256=structural_receipt.input_archive_sha256,
        input_archive_version=structural_receipt.input_archive_version,
        input_archive_policy_id=structural_receipt.input_archive_policy_id,
        batch_id=structural_receipt.batch_id,
        batch_policy_id=structural_receipt.batch_policy_id,
        exact_freeze_id=partition.exact_freeze_id,
        phase2b_protocol_id=structural_receipt.protocol_id,
        execution_freeze_manifest_id=(
            structural_receipt.execution_freeze_manifest_id
        ),
        ordered_archive_input_row_ids_root=(
            partition.ordered_archive_input_row_ids_root
        ),
        main_row_ids_root=partition.main_row_ids_root,
        semantic_conflict_row_ids_root=(
            partition.semantic_conflict_row_ids_root
        ),
        partition_union_row_ids_root=partition.partition_union_row_ids_root,
        main_answer_rows=tuple(answer_rows),
    )
    live_gate_fixture = replay_fixture["_SyntheticFixtureV2"](
        answer=answer_manifest,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
        cells=template.cells,
    )
    gate_input_manifest = replay_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=answer_manifest,
        main_gate_input_rows=replay_fixture["_gate_rows"](live_gate_fixture),
    )
    answer_salt = "synthetic-full-raw-960-answer-opening-0123456789abcdef"
    gate_salt = "synthetic-full-raw-960-gate-opening-0123456789abcdef"
    answer_commitment = salted_answer_commitment_sha256(
        answer_manifest.answer_manifest_sha256,
        answer_salt,
    )
    gate_commitment = replay_v2.salted_gate_input_commitment_sha256_v2(
        gate_input_manifest.gate_input_manifest_sha256,
        gate_salt,
    )

    # The requested pipeline: raw public archives enter the encoder, are
    # independently reconstructed by the decoder, then reach the real scorer,
    # replay validator, and available-gate evaluator.
    bundle_bytes = bundle_v1.encode_phase2b_scorer_bundle_v1(
        authority_requirement_id=(
            bundle_v1.PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
        ),
        actor_profile_content_id=(
            bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1
        ),
        actor_profile_canonical_sha256=(
            bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1
        ),
        campaign_id=(
            "phase2b_actual_unsealed_960_campaign_v2_"
            + hashlib.sha256(b"synthetic-full-pipeline-campaign").hexdigest()
        ),
        attempt_id=(
            "phase2b_actual_unsealed_960_attempt_v2_"
            + hashlib.sha256(b"synthetic-full-pipeline-attempt").hexdigest()
        ),
        execution_registry_id=(
            "phase2b_c1_owner_accepted_execution_registry_v1_"
            + hashlib.sha256(b"synthetic-full-pipeline-registry").hexdigest()
        ),
        scorer_image_digest=(
            "sha256:"
            + hashlib.sha256(b"synthetic-full-pipeline-image").hexdigest()
        ),
        scorer_source_root_sha256=hashlib.sha256(
            b"synthetic-full-pipeline-source-root"
        ).hexdigest(),
        execution_freeze_bytes=execution_freeze_bytes,
        input_archive=input_archive_bytes,
        prediction_archive=prediction_archive_bytes,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
        answer_manifest=answer_manifest,
        answer_commitment_salt=answer_salt,
        salted_answer_commitment_sha256=answer_commitment,
        gate_input_manifest=gate_input_manifest,
        gate_input_commitment_salt=gate_salt,
        salted_gate_input_commitment_sha256=gate_commitment,
        prediction_if_absent_receipt_id=(
            "phase2b_prediction_if_absent_receipt_v1_"
            + hashlib.sha256(b"synthetic-full-pipeline-receipt").hexdigest()
        ),
        signed_reveal_token_id=(
            "phase2b_signed_reveal_token_v1_"
            + hashlib.sha256(b"synthetic-full-pipeline-reveal").hexdigest()
        ),
    )
    decoded_bundle = bundle_v1.decode_phase2b_scorer_bundle_v1(bundle_bytes)
    assert decoded_bundle.input_archive_bytes == input_archive_bytes
    assert decoded_bundle.prediction_archive_bytes == prediction_archive_bytes
    assert decoded_bundle.decoded_input_archive.archive == input_archive_bytes
    assert decoded_bundle.decoded_prediction_archive.archive == prediction_archive_bytes
    assert dict(decoded_bundle.metadata["claims"]) == {
        name: False for name in EXPECTED_BUNDLE_FALSE_CLAIMS
    }

    evaluated = bundle_v1.evaluate_phase2b_scorer_bundle_v1(decoded_bundle)
    scoring = evaluated.scoring_mechanics
    available = evaluated.available_gate_mechanics
    assert type(scoring) is scoring_v2.Unsealed960PredictionScoringMechanicsV2
    assert type(available) is gate_v2.Unsealed960AvailableGateMechanicsV2
    assert (
        scoring.total_prediction_count,
        scoring.main_row_result_count,
        scoring.metric_eligible_main_row_count,
        scoring.control_row_without_frozen_metric_count,
        scoring.semantic_conflict_excluded_count,
    ) == (960, 720, 336, 384, 240)
    assert len(scoring.metric_results) == 9
    assert len(scoring.main_row_results) == 720
    assert {len(row.metric_outcomes) for row in scoring.main_row_results} == {9}
    assert sum(
        len(row.metric_outcomes) for row in scoring.main_row_results
    ) == 720 * 9
    assert (
        available.overall_result_count,
        available.slice_result_count,
        available.unavailable_result_count,
    ) == (10, 24, 2)
    assert (
        len(available.available_overall_gate_mechanics_results),
        len(available.available_slice_gate_mechanics_results),
        len(available.unavailable_gate_mechanics),
    ) == (10, 24, 2)

    for name in COMMON_AUTHORITY_AND_ACTUAL_NONCLAIMS:
        assert getattr(scoring, name) is False
        assert getattr(available, name) is False
    for name in (
        "gate_input_manifest_authority_verified",
        "gate_input_commitment_authority_verified",
        "one_shot_policy_enforced",
        "durable_attempt_ledger_verified",
        "formal_c1_report_verified",
    ):
        assert getattr(available, name) is False

    report_specs = (
        (
            evaluated.score_report_bytes,
            evaluated.score_report_id,
            bundle_v1.SCORE_REPORT_ID_PREFIX_V1,
            bundle_v1.SCORE_REPORT_DOMAIN_V1,
            "PHASE2B_SCORE_MECHANICS_V1",
        ),
        (
            evaluated.available_gate_report_bytes,
            evaluated.available_gate_report_id,
            bundle_v1.GATE_REPORT_ID_PREFIX_V1,
            bundle_v1.GATE_REPORT_DOMAIN_V1,
            "PHASE2B_AVAILABLE_GATE_MECHANICS_V1",
        ),
    )
    for report_bytes, report_id, prefix, domain, report_kind in report_specs:
        assert report_bytes.endswith(b"\n")
        assert report_bytes.count(b"\n") == 1
        assert b"\r" not in report_bytes
        assert report_id == prefix + hashlib.sha256(domain + report_bytes).hexdigest()
        report = json.loads(report_bytes)
        assert report == {
            "actual_scoring_claimed": False,
            "formal_gate_evaluation_claimed": False,
            "report_kind": report_kind,
            "result": report["result"],
        }
