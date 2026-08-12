import json
from dataclasses import replace
from pathlib import Path

import pytest

from hegel_machine.milestones import PHASE2B
from hegel_machine.phase2b_freeze_v1 import (
    PreservationTransform,
    frozen_phase2b_exact_freeze,
)
from hegel_machine.phase2b_exact_bridge_v1 import (
    DEFAULT_EXACT_BRIDGE_POLICY,
    DEFAULT_EXACT_SELECTION_POLICY,
)
from hegel_machine.phase2b_covert_audit_v1 import (
    DEFAULT_COVERT_AUDIT_POLICY,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    SEMANTICS_ID as COVERT_AUDIT_SEMANTICS_ID,
)
from hegel_machine.phase2b_exact_derived_witness_bridge_v1 import (
    EXACT_DERIVED_WITNESS_BRIDGE_VERSION,
    EXACT_DERIVED_WITNESS_MATCHER_VERSION,
)
from hegel_machine.phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
    EXACT_TRANSFORM_POLICY_ID,
    EXACT_TRANSFORM_SEMANTICS_VERSION,
    PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
)
from hegel_machine.phase2b_trusted_wire_v1 import (
    FIELD_MANIFEST_ID as TRUSTED_WIRE_FIELD_MANIFEST_ID,
    JCS_PROFILE_ID as TRUSTED_WIRE_JCS_PROFILE_ID,
    NON_AUTHORITATIVE_CLAIM_LEVEL as TRUSTED_WIRE_CLAIM_LEVEL,
)
from hegel_machine.phase2b_trusted_wire_batch_v1 import (
    EXACT_TRANSFORM_VALIDATOR_POLICY_ID,
    TRUSTED_WIRE_BATCH_POLICY_ID,
    TRUSTED_WIRE_KEY_SCHEDULE_VERSION,
    TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
)
from hegel_machine.phase2b_trusted_wire_typed_authority_v1 import (
    TYPED_AUTHORITY_CODEC_POLICY_ID,
    TYPED_AUTHORITY_CODEC_VERSION,
    TYPED_AUTHORITY_SCHEMA_ID,
)
from hegel_machine.phase2b_trusted_wire_typed_replay_v1 import (
    TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID,
    TYPED_TRUSTED_WIRE_REPLAY_VERSION,
)
from hegel_machine.phase2b_recognizer_input_archive_v1 import (
    PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION,
)
from hegel_machine.phase2b_recognizer_input_archive_v2 import (
    PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from hegel_machine.phase2b_recognizer_prediction_archive_v1 import (
    PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID,
    PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION,
    PUBLIC_RUN_CONTEXT_SCHEMA_ID,
    PUBLIC_RUN_CONTEXT_SCHEMA_VERSION,
    RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID,
    RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION,
)
from hegel_machine.phase2b_recognizer_prediction_v2 import (
    PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID,
    PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION,
    RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2,
)
from hegel_machine.phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from hegel_machine.phase2b_unsealed_prediction_evaluator_v1 import (
    UNSEALED_PREDICTION_EVALUATOR_POLICY_ID,
    UNSEALED_PREDICTION_EVALUATOR_VERSION,
)
from hegel_machine.phase2b_trusted_wire_batch_v2 import (
    TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
    TRUSTED_WIRE_ENVELOPE_V2_MAGIC,
    TRUSTED_WIRE_ENVELOPE_V2_VERSION,
)
from hegel_machine.phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
)
from hegel_machine.phase2b_trusted_wire_typed_replay_v2 import (
    TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
)
from hegel_machine.phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    CandidateFootprint,
    ExecutionFreezeManifest,
    HoldoutLifecycle,
    MeasurementUse,
    SealedRunLedger,
    evaluate_binary_gate,
    evaluate_shared_footprint,
    frozen_phase2b_protocol,
    one_sided_wilson_lower_bound,
    phase2b_preregistration_report,
    salted_answer_commitment_sha256,
)
from hegel_machine.phase2b_uncertainty_compiler import (
    DEFAULT_EXACT_UNCERTAINTY_POLICY,
    FROZEN_RATIONAL_GRID_ID,
)
from hegel_machine.phase2b_wire import TransformOperation


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
ANSWER_SALT = "sealed-answer-salt-" + "x" * 32


def test_phase2b_protocol_freezes_720_independent_latent_cases():
    protocol = frozen_phase2b_protocol()
    assert protocol.milestone_id == PHASE2B.machine_id
    assert protocol.milestone_name == PHASE2B.name
    assert len(protocol.law_families) == 6
    assert protocol.scale_cell_count == 2
    assert protocol.cases_per_family_scale_cell == 60
    assert protocol.independent_latent_case_count == 720
    assert dict(protocol.case_type_totals) == {
        "unique_scale_answerable": 228,
        "admissible_scale_set_answerable": 12,
        "wrong_family_hard_negative": 96,
        "binding_counterfactual": 96,
        "scale_counterfactual": 96,
        "sign_or_invariant_break": 96,
        "insufficient_or_nonidentifiable": 96,
    }
    assert protocol.separate_preservation_denominator is True
    assert protocol.holdout_run_limit == 1


def test_exact_margin_quota_resolves_conflict_but_implementation_blocks_generation():
    protocol = frozen_phase2b_protocol()
    assert dict(protocol.margin_stratum_per_cell) == {
        "clear_interior": 21,
        "moderate": 18,
        "near_boundary_identifiable": 12,
        "nonunique_or_insufficient": 9,
    }
    assert dict(protocol.margin_stratum_totals) == {
        "clear_interior": 252,
        "moderate": 216,
        "near_boundary_identifiable": 144,
        "nonunique_or_insufficient": 108,
    }
    assert (
        "margin_strata_require_108_ambiguous_or_admissible_cases_but_case_table_allocates_96"
        not in protocol.unresolved_freeze_questions
    )
    assert protocol.unresolved_freeze_questions == ()
    assert (
        "trusted_rfc8785_wire_builder_and_namespace_aware_formal_"
        "covert_auditor_not_implemented"
    ) in protocol.implementation_blockers
    assert (
        "formal_preservation_pair_generator_evaluator_and_complete_"
        "transform_to_verifier_coverage_not_implemented"
    ) in protocol.implementation_blockers
    assert (
        "formal_wire_builder_and_covert_channel_auditor_not_implemented"
        not in protocol.implementation_blockers
    )
    assert (
        "functional_recognizer_cli_signed_minimal_image_and_formal_"
        "scoring_evaluator_not_implemented"
        in protocol.implementation_blockers
    )
    assert protocol.ready_for_holdout_generation is False


def test_protocol_uses_stricter_scale_thresholds_and_requires_real_inference():
    protocol = frozen_phase2b_protocol()
    gates = {gate.metric: gate for gate in protocol.overall_gates}
    assert (
        gates["scale_set_accuracy"].minimum_point_estimate,
        gates["scale_set_accuracy"].minimum_one_sided_wilson_lcb,
    ) == (0.87, 0.82)
    assert (
        gates["scale_counterfactual_rejection"].minimum_point_estimate,
        gates["scale_counterfactual_rejection"].minimum_one_sided_wilson_lcb,
    ) == (0.93, 0.88)
    assert (
        gates["nonidentifiable_scale_abstention"].minimum_point_estimate,
        gates["nonidentifiable_scale_abstention"].minimum_one_sided_wilson_lcb,
    ) == (0.95, 0.90)
    assert protocol.scale_regret_gate.maximum_point_estimate == 0.05
    assert protocol.scale_regret_gate.maximum_bootstrap_upper_bound == 0.08
    assert protocol.scale_hypothesis_generation_required is True


def test_isolation_contract_is_complete_but_does_not_self_attest_enforcement():
    isolation = frozen_phase2b_protocol().isolation_profile
    assert isolation.contract_complete is True
    assert isolation.missing_controls == ()
    assert isolation.proves_external_enforcement is False
    weakened = replace(isolation, network_disabled=False)
    assert weakened.contract_complete is False
    assert weakened.missing_controls == ("network_disabled",)


def test_wilson_lower_bound_and_binary_gate_are_fail_closed():
    assert one_sided_wilson_lower_bound(100, 100) == pytest.approx(
        0.9736579873
    )
    assert one_sided_wilson_lower_bound(0, 100) == 0.0
    with pytest.raises(TypeError, match="integers"):
        one_sided_wilson_lower_bound(True, 100)
    with pytest.raises(ValueError, match="valid range"):
        one_sided_wilson_lower_bound(101, 100)

    threshold = next(
        gate
        for gate in frozen_phase2b_protocol().overall_gates
        if gate.metric == "family_exact"
    )
    assert evaluate_binary_gate(
        threshold,
        successes=240,
        total=240,
    ).passed
    assert not evaluate_binary_gate(
        threshold,
        successes=200,
        total=240,
    ).passed


def test_shared_footprint_requires_real_nonconstant_shared_measurements():
    correct = CandidateFootprint(
        "candidate_a",
        (
            MeasurementUse("m1", "numeric"),
            MeasurementUse("m2", "order"),
            MeasurementUse("m3", "context"),
        ),
    )
    competitor = CandidateFootprint(
        "candidate_b",
        (
            MeasurementUse("m1", "numeric"),
            MeasurementUse("m2", "order"),
            MeasurementUse("m4", "context"),
        ),
    )
    result = evaluate_shared_footprint(correct, competitor)
    assert result.passed is True
    assert result.shared_measurement_count == 2
    assert result.correct_shared_fraction == pytest.approx(2 / 3)
    assert result.competitor_shared_fraction == pytest.approx(2 / 3)

    private_competitor = replace(
        competitor,
        measurements=competitor.measurements
        + (MeasurementUse("oracle", "numeric", candidate_private=True),),
    )
    assert not evaluate_shared_footprint(correct, private_competitor).passed

    mismatched_kinds = CandidateFootprint(
        "candidate_c",
        (
            MeasurementUse("m1", "order"),
            MeasurementUse("m2", "numeric"),
            MeasurementUse("m4", "context"),
        ),
    )
    mismatch_result = evaluate_shared_footprint(correct, mismatched_kinds)
    assert mismatch_result.shared_measurement_count == 0
    assert mismatch_result.passed is False


def _freeze_manifest():
    exact_freeze = frozen_phase2b_exact_freeze()
    spec_id_by_kind = {
        BaselineKind(spec.baseline_id): spec.content_id
        for spec in exact_freeze.baselines
    }
    baselines = tuple(
        BaselineRegistration(
            kind,
            baseline_spec_id=spec_id_by_kind[kind],
            implementation_id=f"baseline_{kind.value}_v1",
            artifact_sha256=digest,
            frozen_before_holdout_generation=True,
        )
        for kind, digest in zip(BaselineKind, (SHA_A, SHA_B, SHA_C), strict=True)
    )
    protocol = frozen_phase2b_protocol()
    return ExecutionFreezeManifest(
        protocol_id=protocol.protocol_id,
        exact_freeze_id=exact_freeze.freeze_id,
        git_commit="1" * 40,
        recognizer_image_digest="sha256:" + SHA_A,
        configuration_sha256=SHA_B,
        theory_version_id="theory_frozen_v1",
        adapter_implementation_sha256=SHA_C,
        selector_implementation_sha256=SHA_D,
        verifier_registry_sha256=SHA_A,
        baseline_registrations=baselines,
        isolation_profile_id=protocol.isolation_profile.profile_id,
    )


def test_execution_freeze_requires_all_three_pinned_baselines():
    manifest = _freeze_manifest()
    assert manifest.manifest_id.startswith("phase2b_execution_freeze_")
    with pytest.raises(ValueError, match="exact freeze"):
        replace(manifest, exact_freeze_id="phase2b_exact_freeze_" + SHA_D)
    with pytest.raises(ValueError, match="all three baseline"):
        replace(manifest, baseline_registrations=manifest.baseline_registrations[:2])
    with pytest.raises(ValueError, match="BaselineSpec"):
        replace(
            manifest,
            baseline_registrations=(
                replace(
                    manifest.baseline_registrations[0],
                    baseline_spec_id="phase2b_baseline_spec_" + SHA_D,
                ),
                *manifest.baseline_registrations[1:],
            ),
        )
    with pytest.raises(ValueError, match="before holdout"):
        replace(
            manifest,
            baseline_registrations=(
                replace(
                    manifest.baseline_registrations[0],
                    frozen_before_holdout_generation=False,
                ),
                *manifest.baseline_registrations[1:],
            ),
        )


def test_sealed_run_state_machine_requires_commit_before_reveal_and_is_one_shot():
    protocol = frozen_phase2b_protocol()
    ledger = SealedRunLedger(
        run_id="external_custodian_run_001",
        protocol_id=protocol.protocol_id,
        freeze_manifest_id=_freeze_manifest().manifest_id,
        independent_custodian_id="external_custodian",
    )
    assert ledger.lifecycle is HoldoutLifecycle.PREREGISTERED
    with pytest.raises(ValueError, match="predictions_committed"):
        ledger.consume(
            revealed_answer_manifest_sha256=SHA_C,
            score_report_sha256=SHA_D,
            answer_commitment_salt=ANSWER_SALT,
        )

    generated = ledger.record_generated_holdout(
        input_commitment_sha256=SHA_A,
        salted_answer_commitment_sha256=salted_answer_commitment_sha256(
            SHA_A,
            ANSWER_SALT,
        ),
    )
    with pytest.raises(ValueError, match="already transitioned"):
        ledger.record_generated_holdout(
            input_commitment_sha256=SHA_C,
            salted_answer_commitment_sha256=SHA_D,
        )
    committed = generated.commit_predictions(
        prediction_archive_sha256=SHA_C,
        audit_archive_sha256=SHA_D,
    )
    consumed = committed.consume(
        revealed_answer_manifest_sha256=SHA_A,
        score_report_sha256=SHA_B,
        answer_commitment_salt=ANSWER_SALT,
    )
    assert consumed.lifecycle is HoldoutLifecycle.CONSUMED
    assert consumed.prior_ledger_id == committed.ledger_id
    with pytest.raises(ValueError, match="generated_sealed"):
        consumed.commit_predictions(
            prediction_archive_sha256=SHA_C,
            audit_archive_sha256=SHA_D,
        )
    with pytest.raises(ValueError, match="terminal"):
        consumed.invalidate("attempted reuse")


def test_sealed_ledger_rejects_direct_terminal_construction_and_bad_reveal():
    protocol = frozen_phase2b_protocol()
    with pytest.raises(ValueError, match="transition authority"):
        SealedRunLedger(
            run_id="forged_consumed_run",
            protocol_id=protocol.protocol_id,
            freeze_manifest_id=_freeze_manifest().manifest_id,
            independent_custodian_id="forged_custodian",
            lifecycle=HoldoutLifecycle.CONSUMED,
            holdout_input_commitment_sha256=SHA_A,
            salted_answer_commitment_sha256=SHA_B,
            prediction_archive_sha256=SHA_C,
            audit_archive_sha256=SHA_D,
            revealed_answer_manifest_sha256=SHA_A,
            score_report_sha256=SHA_B,
            prior_ledger_id="phase2b_run_ledger_" + "e" * 64,
        )

    ledger = SealedRunLedger(
        run_id="bad_reveal_run",
        protocol_id=protocol.protocol_id,
        freeze_manifest_id=_freeze_manifest().manifest_id,
        independent_custodian_id="external_custodian",
    )
    generated = ledger.record_generated_holdout(
        input_commitment_sha256=SHA_A,
        salted_answer_commitment_sha256=salted_answer_commitment_sha256(
            SHA_A,
            ANSWER_SALT,
        ),
    )
    committed = generated.commit_predictions(
        prediction_archive_sha256=SHA_C,
        audit_archive_sha256=SHA_D,
    )
    with pytest.raises(ValueError, match="does not open"):
        committed.consume(
            revealed_answer_manifest_sha256=SHA_B,
            score_report_sha256=SHA_C,
            answer_commitment_salt=ANSWER_SALT,
        )


def test_phase2b_report_is_explicitly_unsealed_and_nonqualifying():
    report = phase2b_preregistration_report()
    assert report["artifact"] == "phase2b_preregistration_readiness_v1"
    assert report["formal_phase2b_exit_claim"] is False
    assert report["status"] == "exact_parameter_freeze_with_implementation_blockers"
    assert report["normative_parameter_freeze_complete"] is True
    assert report["formal_holdout_generation_authorized"] is False
    assert report["unresolved_freeze_questions"] == []
    assert report["implementation_blockers"]
    assert report["sealed_holdout_generated"] is False
    assert report["sealed_holdout_consumed"] is False
    assert report["process_local_ledger_fork_guard_implemented"] is True
    assert report["durable_external_one_shot_ledger_implemented"] is False
    assert report["answer_commitment_opening_validation_implemented"] is True
    assert report["independent_custodian_attested"] is False
    assert report["external_isolation_attested"] is False
    assert report["unsealed_pipeline_validation_run"] is False
    assert report["typed_evidence_to_prediction_pipeline_complete"] is False
    assert report["projection_compiler_implemented"] is False
    assert (
        report[
            "bounded_binary64_dimensionless_point_root_identity_"
            "projection_mechanics_implemented"
        ]
        is True
    )
    assert (
        report["binary64_absolute_bound_envelope_mechanics_implemented"] is True
    )
    assert (
        report["formal_rational_grid_uncertainty_compiler_implemented"] is True
    )
    assert (
        report["absolute_bound_uncertainty_semantics_compiler_implemented"]
        is True
    )
    assert (
        report["standard_error_uncertainty_semantics_compiler_implemented"]
        is False
    )
    assert report["bundle_atomic_exact_uncertainty_receipt_implemented"] is True
    assert report["exact_uncertainty_compiler_policy_id"] == (
        DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id
    )
    assert report["formal_rational_grid_id"] == FROZEN_RATIONAL_GRID_ID
    assert (
        report["exact_uncertainty_receipt_consumed_by_projection_compiler"]
        is False
    )
    assert (
        report["exact_rational_residual_interval_semantics_implemented"]
        is False
    )
    assert (
        report[
            "root_identity_six_law_exact_rational_residual_interval_"
            "semantics_implemented"
        ]
        is True
    )
    assert report["exact_rational_selector_bridge_implemented"] is True
    assert (
        report[
            "authoritative_exact_bridge_recomputes_uncertainty_and_"
            "adapter_internally"
        ]
        is True
    )
    assert (
        report[
            "oversized_bundle_theory_or_registry_rejected_before_content_hash"
        ]
        is True
    )
    assert (
        report["nested_authority_exact_type_enforced_before_content_hash"]
        is True
    )
    assert (
        report["exact_uncertainty_receipt_consumed_by_root_identity_bridge"]
        is True
    )
    assert report["exact_bridge_policy_id"] == (
        DEFAULT_EXACT_BRIDGE_POLICY.policy_id
    )
    assert report["exact_selector_policy_id"] == (
        DEFAULT_EXACT_SELECTION_POLICY.policy_id
    )
    assert report["exact_verifier_semantics_id"] == (
        DEFAULT_EXACT_BRIDGE_POLICY.verifier_semantics_id
    )
    assert report["public_transform_evidence_v2_authority_implemented"] is True
    assert report["public_transform_evidence_schema_version"] == (
        PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
    )
    assert report["exact_transform_semantics_version"] == (
        EXACT_TRANSFORM_SEMANTICS_VERSION
    )
    assert report["exact_transform_policy_id"] == EXACT_TRANSFORM_POLICY_ID
    assert (
        report[
            "eight_wire_transform_operation_exact_kernel_mechanics_implemented"
        ]
        is True
    )
    assert report["bundle_atomic_exact_transform_receipt_implemented"] is True
    assert report["exact_transform_recomputes_uncertainty_internally"] is True
    assert report["complete_transform_semantics_implemented"] is False
    assert report["formal_preservation_transform_suite_implemented"] is False
    assert report["exact_derived_observation_witness_bridge_implemented"] is True
    assert report["exact_derived_witness_bridge_version"] == (
        EXACT_DERIVED_WITNESS_BRIDGE_VERSION
    )
    assert report["exact_derived_witness_matcher_version"] == (
        EXACT_DERIVED_WITNESS_MATCHER_VERSION
    )
    assert (
        report[
            "authoritative_derived_witness_bridge_recomputes_transform_"
            "internally"
        ]
        is True
    )
    assert (
        report[
            "strict_scope_complete_law_binding_scale_support_slice_grid_"
            "implemented"
        ]
        is True
    )
    assert (
        report["scale_selector_aggregates_exact_support_slices_before_selection"]
        is True
    )
    assert (
        report["exact_transform_receipt_consumed_by_derived_witness_bridge"]
        is True
    )
    assert (
        report["all_eight_transform_operations_covered_by_derived_six_law_bridge"]
        is False
    )
    assert (
        report["nondimensionless_derived_verifier_semantics_implemented"]
        is False
    )
    assert report["prediction_archive_evaluator_implemented"] is False
    assert report["public_wire_contract_implemented"] is True
    assert report["public_wire_is_family_neutral_shaped_only"] is True
    assert report["semantic_family_neutrality_audited"] is False
    assert report["allowed_field_answer_correlation_audit_implemented"] is False
    assert report["schema_closed_accepted_jcs_profile_mechanics_implemented"] is True
    assert report["accepted_jcs_profile_id"] == TRUSTED_WIRE_JCS_PROFILE_ID
    assert (
        report["explicit_v2_uuid_namespace_path_manifest_mechanics_implemented"]
        is True
    )
    assert (
        report["uuid_namespace_path_manifest_id"]
        == TRUSTED_WIRE_FIELD_MANIFEST_ID
    )
    assert (
        report["fixed_65536_public_padding_envelope_mechanics_implemented"]
        is True
    )
    assert report["trusted_wire_profile_claim_level"] == TRUSTED_WIRE_CLAIM_LEVEL
    assert (
        report["trusted_wire_profile_transform_policy_id"]
        == EXACT_TRANSFORM_POLICY_ID
    )
    assert report["keyed_trusted_wire_batch_mechanics_implemented"] is True
    assert report["trusted_wire_batch_policy_id"] == TRUSTED_WIRE_BATCH_POLICY_ID
    assert report["trusted_wire_key_schedule_version"] == (
        TRUSTED_WIRE_KEY_SCHEDULE_VERSION
    )
    assert report["trusted_wire_public_provenance_version"] == (
        TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
    )
    assert report["trusted_wire_exact_transform_validator_policy_id"] == (
        EXACT_TRANSFORM_VALIDATOR_POLICY_ID
    )
    assert report["typed_authority_codec_version"] == TYPED_AUTHORITY_CODEC_VERSION
    assert report["typed_authority_schema_id"] == TYPED_AUTHORITY_SCHEMA_ID
    assert report["typed_authority_codec_policy_id"] == (
        TYPED_AUTHORITY_CODEC_POLICY_ID
    )
    assert report["strict_closed_typed_authority_codec_mechanics_implemented"] is True
    assert report["exact_transform_provenance_compiler_version"] == (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION
    )
    assert report["exact_transform_provenance_compiler_policy_id"] == (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
    )
    assert report["native_v2_provenance_compile_before_framing_implemented"] is True
    assert report["typed_trusted_wire_replay_version"] == (
        TYPED_TRUSTED_WIRE_REPLAY_VERSION
    )
    assert report["typed_trusted_wire_replay_policy_id"] == (
        TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
    )
    assert (
        report[
            "direct_payload_authority_exact_transform_complete_replay_implemented"
        ]
        is True
    )
    assert report["whole_batch_atomic_typed_replay_mechanics_implemented"] is True
    assert (
        report["source_order_bound_stage_b_secret_replay_receipt_implemented"]
        is True
    )
    assert report["public_recognizer_registry_schema_version"] == (
        PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION
    )
    assert report["public_recognizer_registry_schema_id"] == (
        PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID
    )
    assert report["public_recognizer_family_alias_policy_id"] == (
        PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID
    )
    assert report["trusted_recognizer_input_archive_version"] == (
        TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION
    )
    assert report["recognizer_input_archive_policy_id"] == (
        RECOGNIZER_INPUT_ARCHIVE_POLICY_ID
    )
    assert report["recognizer_input_archive_claim_level"] == (
        TRUSTED_WIRE_CLAIM_LEVEL
    )
    for field_name in (
        "strict_public_recognizer_registry_codec_mechanics_implemented",
        "live_post_hmac_recognizer_registry_projection_mechanics_implemented",
        "registry_envelope_exact_scope_bijection_replay_implemented",
        "global_source_public_uuid_disjointness_gate_implemented",
        (
            "whole_batch_atomic_custodian_gated_recognizer_input_archive_"
            "issuer_mechanics_implemented"
        ),
        "public_recognizer_input_archive_structural_decode_replay_implemented",
        "recognizer_input_archive_success_is_false_claim_public_decode",
    ):
        assert report[field_name] is True
    for field_name in (
        "durable_trusted_recognizer_input_archive_receipt_implemented",
        "recognizer_input_archive_batch_policy_membership_verified",
        "recognizer_input_archive_source_registry_projection_verified",
        "recognizer_input_archive_secret_custodian_replay_verified",
        "recognizer_input_archive_origin_authenticated",
        "recognizer_input_archive_formal_covert_audit",
        "recognizer_input_archive_sealed_holdout_eligible",
        "recognizer_input_archive_recognizer_executed",
        "recognizer_input_archive_prediction_archive_evaluated",
        "recognizer_input_archive_c1_exit_evidence",
    ):
        assert report[field_name] is False
    assert report["public_run_context_schema_version"] == (
        PUBLIC_RUN_CONTEXT_SCHEMA_VERSION
    )
    assert report["public_run_context_schema_id"] == PUBLIC_RUN_CONTEXT_SCHEMA_ID
    assert report["public_recognizer_prediction_record_schema_version"] == (
        PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION
    )
    assert report["public_recognizer_prediction_record_schema_id"] == (
        PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID
    )
    assert report["recognizer_prediction_archive_schema_version"] == (
        RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION
    )
    assert report["recognizer_prediction_archive_policy_id"] == (
        RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID
    )
    assert report["recognizer_prediction_archive_claim_level"] == (
        TRUSTED_WIRE_CLAIM_LEVEL
    )
    assert report["unsealed_prediction_evaluator_version"] == (
        UNSEALED_PREDICTION_EVALUATOR_VERSION
    )
    assert report["unsealed_prediction_evaluator_policy_id"] == (
        UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
    )
    assert report["recognizer_runner_total_case_count"] == (
        TOTAL_RECOGNIZER_CASE_COUNT
    )
    for field_name in (
        "public_run_context_structural_schema_mechanics_implemented",
        "closed_public_prediction_record_schema_mechanics_implemented",
        (
            "record_framed_exact_960_prediction_archive_structural_codec_"
            "mechanics_implemented"
        ),
        "internal_derived_to_prediction_mapping_gate_mechanics_implemented",
        "decoded_prediction_semantic_fields_exclude_split_gold_index_labels",
        (
            "unsealed_720_240_sorted_disjoint_exhaustive_structural_"
            "evaluator_implemented"
        ),
        "recognizer_runner_total_960_contract_implemented",
    ):
        assert report[field_name] is True
    assert report["minimum_constructed_positive_typed_profile_bytes"] == 125_582
    assert report["trusted_wire_maximum_payload_bytes"] == 65_424
    assert report["real_positive_typed_profile_fits_trusted_wire"] is False
    assert report["compact_typed_authority_codec_v2_version"] == (
        COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    )
    assert report["compact_typed_authority_schema_id_v2"] == (
        COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    )
    assert report["compact_typed_authority_codec_policy_id_v2"] == (
        COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
    )
    assert report["trusted_wire_batch_v2_schema_version"] == (
        TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
    )
    assert report["trusted_wire_batch_v2_payload_schema_version"] == (
        TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION
    )
    assert report["trusted_wire_envelope_v2_version"] == (
        TRUSTED_WIRE_ENVELOPE_V2_VERSION
    )
    assert report["trusted_wire_envelope_v2_magic_hex"] == (
        TRUSTED_WIRE_ENVELOPE_V2_MAGIC.hex()
    )
    assert TRUSTED_WIRE_ENVELOPE_V2_MAGIC.hex() == "4847503242573200"
    assert report["trusted_wire_batch_v2_policy_id"] == (
        TRUSTED_WIRE_BATCH_V2_POLICY_ID
    )
    assert report["typed_trusted_wire_replay_v2_version"] == (
        TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION
    )
    assert report["typed_trusted_wire_replay_v2_policy_id"] == (
        TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
    )
    assert report["public_recognizer_registry_v2_schema_version"] == (
        PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION
    )
    assert report["public_recognizer_registry_v2_schema_id"] == (
        PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2
    )
    assert report["public_recognizer_family_alias_policy_id_v2"] == (
        PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2
    )
    assert report["trusted_recognizer_input_archive_v2_version"] == (
        TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
    )
    assert report["recognizer_input_archive_v2_policy_id"] == (
        RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
    )
    assert report["recognizer_input_archive_v2_claim_level"] == (
        TRUSTED_WIRE_CLAIM_LEVEL
    )
    for field_name in (
        "lossless_compact_typed_authority_codec_v2_mechanics_implemented",
        "compact_v2_fixed_65536_envelope_mechanics_implemented",
        "public_typed_trusted_wire_v2_replay_mechanics_implemented",
        "typed_trusted_wire_replay_v2_batch_policy_membership_mechanics_implemented",
        "typed_trusted_wire_replay_v2_whole_batch_atomic_mechanics_implemented",
        "typed_trusted_wire_replay_v2_compact_authority_canonical_mechanics_implemented",
        "typed_trusted_wire_replay_v2_public_provenance_mechanics_implemented",
        "typed_trusted_wire_replay_v2_direct_exact_transform_mechanics_implemented",
        "recognizer_input_archive_v2_structural_archive_mechanics_implemented",
        "recognizer_input_archive_v2_row_bijection_mechanics_implemented",
        "recognizer_input_archive_v2_registry_schema_mechanics_implemented",
        "recognizer_input_archive_v2_registry_authority_exact_scope_mechanics_implemented",
        "recognizer_input_archive_v2_compact_typed_replay_mechanics_implemented",
        "recognizer_input_archive_v2_direct_payload_transform_replay_mechanics_implemented",
        "recognizer_input_archive_v2_cross_row_unlinkable_public_uuid_disjoint_mechanics_implemented",
        "recognizer_input_archive_v2_private_single_live_allocation_gate_mechanics_implemented",
        "recognizer_input_archive_v2_private_source_public_uuid_disjointness_gate_mechanics_implemented",
        "real_positive_compact_v2_payload_fits_trusted_wire",
        "single_constructed_positive_compact_v2_mechanics_verified",
        "real_positive_compact_v2_exact_transform_replay_implemented",
        "real_positive_compact_v2_recognizer_input_archive_replay_implemented",
        "real_positive_compact_v2_derived_bridge_compilation_parity_implemented",
        "real_positive_compact_v2_derived_bridge_decision_parity_implemented",
    ):
        assert report[field_name] is True
    assert report["real_positive_expanded_typed_profile_bytes"] == 125_582
    assert report["real_positive_compact_v2_payload_bytes"] == 50_255
    assert report["real_positive_compact_v2_payload_cap_headroom_bytes"] == 15_169
    assert report["real_positive_compact_v2_secret_padding_bytes"] == 15_201
    assert report["real_positive_compact_v2_fixed_envelope_bytes"] == 65_536
    for field_name in (
        "typed_trusted_wire_replay_v2_secret_custodian_replay_verified",
        "typed_trusted_wire_replay_v2_whole_batch_shuffle_publicly_verified",
        "typed_trusted_wire_replay_v2_purpose_separated_keys_publicly_verified",
        "typed_trusted_wire_replay_v2_post_shuffle_hmac_uuidv4_publicly_verified",
        "typed_trusted_wire_replay_v2_secret_hmac_padding_publicly_verified",
        "typed_trusted_wire_replay_v2_source_authority_binding_verified",
        "typed_trusted_wire_replay_v2_live_allocation_schedule_verified",
        "typed_trusted_wire_replay_v2_recognizer_capacity_evidence",
        "typed_trusted_wire_replay_v2_origin_authenticated",
        "typed_trusted_wire_replay_v2_formal_uuid_audit",
        "typed_trusted_wire_replay_v2_formal_covert_audit",
        "typed_trusted_wire_replay_v2_sealed_holdout_eligible",
        "typed_trusted_wire_replay_v2_c1_exit_evidence",
        "recognizer_input_archive_v2_batch_policy_membership_verified",
        "recognizer_input_archive_v2_source_registry_projection_verified",
        "recognizer_input_archive_v2_source_public_disjoint_verified",
        "recognizer_input_archive_v2_single_live_allocation_verified",
        "recognizer_input_archive_v2_secret_custodian_replay_verified",
        "recognizer_input_archive_v2_origin_authenticated",
        "recognizer_input_archive_v2_formal_uuid_audit",
        "recognizer_input_archive_v2_formal_covert_audit",
        "recognizer_input_archive_v2_sealed_holdout_eligible",
        "recognizer_input_archive_v2_recognizer_executed",
        "recognizer_input_archive_v2_prediction_archive_evaluated",
        "recognizer_input_archive_v2_capacity_evidence",
        "recognizer_input_archive_v2_c1_exit_evidence",
    ):
        assert report[field_name] is False
    assert report["public_recognizer_prediction_outcome_v2_schema_version"] == (
        PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION
    )
    assert report["public_recognizer_prediction_outcome_v2_schema_id"] == (
        PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID
    )
    assert report["recognizer_prediction_row_v2_policy_id"] == (
        RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2
    )
    assert report["recognizer_prediction_row_v2_claim_level"] == (
        TRUSTED_WIRE_CLAIM_LEVEL
    )
    for field_name in (
        "public_recognizer_prediction_outcome_v2_ephemeral_schema_mechanics_implemented",
        "v2_single_row_prediction_mapping_mechanics_implemented",
        "recognizer_prediction_row_v2_exact_input_and_freeze_binding_mechanics_implemented",
        "recognizer_prediction_row_v2_compact_typed_replay_mechanics_implemented",
        "recognizer_prediction_row_v2_public_registry_adapter_mechanics_implemented",
        "recognizer_prediction_row_v2_exact_derived_bridge_mechanics_implemented",
        "recognizer_prediction_row_v2_closed_decision_reason_mapping_mechanics_implemented",
        "recognizer_prediction_row_v2_cross_version_rejection_mechanics_implemented",
        "recognizer_prediction_row_v2_private_ephemeral_issue_mechanics_implemented",
        "real_positive_compact_v2_single_row_prediction_mapping_mechanics_verified",
        "real_positive_compact_v2_prediction_decision_parity_implemented",
        "real_positive_compact_v2_prediction_bundle_identity_parity_implemented",
        "real_positive_compact_v2_prediction_family_binding_scale_parity_implemented",
        "real_positive_compact_v2_prediction_input_protocol_freeze_root_parity_implemented",
    ):
        assert report[field_name] is True
    for field_name in (
        "recognizer_prediction_row_v2_durable_receipt_implemented",
        "recognizer_prediction_row_v2_input_archive_membership_verified",
        "recognizer_prediction_row_v2_batch_policy_membership_verified",
        "recognizer_prediction_row_v2_execution_manifest_authority_verified",
        "recognizer_prediction_row_v2_recognizer_executed",
        "recognizer_prediction_row_v2_runtime_executed",
        "recognizer_prediction_row_v2_capacity_evidence",
        "recognizer_prediction_row_v2_prediction_scoring_implemented",
        "recognizer_prediction_row_v2_effect_evidence",
        "recognizer_prediction_row_v2_origin_authenticated",
        "recognizer_prediction_row_v2_formal_uuid_audit",
        "recognizer_prediction_row_v2_formal_covert_audit",
        "recognizer_prediction_row_v2_sealed_holdout_eligible",
        "recognizer_prediction_row_v2_c1_exit_evidence",
        "v2_full_960_prediction_archive_structural_codec_implemented",
        "v2_unsealed_prediction_evaluator_implemented",
    ):
        assert report[field_name] is False
    assert report["next_phase2b_construction_slice"] == (
        "recognizer_prediction_archive_v2_exact_960_structural_codec"
    )
    for field_name in (
        "real_positive_prediction_end_to_end_replay_implemented",
        "recognizer_prediction_capacity_evidence",
        "prediction_scoring_implemented",
        "prediction_effect_evidence",
        "actual_960_case_prediction_archive_run",
        "recognizer_runtime_executed",
        "prediction_archive_input_membership_verified",
        "prediction_archive_execution_manifest_authority_verified",
        "prediction_archive_derived_mapping_verified_by_public_decode",
        "prediction_archive_origin_authenticated",
        "prediction_archive_formal_covert_audit",
        "prediction_archive_sealed_holdout_eligible",
        "prediction_archive_c1_exit_evidence",
    ):
        assert report[field_name] is False
    assert report["pairwise_distinct_key_source_contract_implemented"] is True
    assert report["key_source_statistical_independence_attested"] is False
    assert report["whole_batch_unbiased_fisher_yates_mechanics_implemented"] is True
    assert report["post_shuffle_namespace_hmac_uuidv4_mechanics_implemented"] is True
    assert report["case_local_latent_id_anti_link_allocation_implemented"] is True
    assert report["renamed_authority_schema_recanonicalization_implemented"] is True
    assert (
        report["wire_only_public_provenance_rebinding_mechanics_implemented"]
        is True
    )
    assert (
        report["secret_hmac_padding_custodian_replay_mechanics_implemented"]
        is True
    )
    assert report["batch_atomic_keyed_trusted_wire_mechanics_implemented"] is True
    assert report["uuid_collision_retry_warning_mechanics_implemented"] is True
    assert (
        report["trusted_wire_custodian_secret_replay_mechanics_implemented"]
        is True
    )
    assert report["trusted_wire_1024_authority_capacity_qualified"] is False
    assert report["global_batch_shuffle_implemented"] is False
    assert report["post_shuffle_hmac_uuidv4_assignment_implemented"] is False
    assert report["provenance_rebound_to_public_payload_implemented"] is False
    assert report["secret_padding_replay_implemented"] is False
    assert report["batch_atomic_trusted_wire_builder_implemented"] is False
    assert (
        report["typed_trusted_wire_authority_decode_replay_implemented"]
        is True
    )
    assert (
        report["typed_trusted_wire_authority_decode_replay_claim_level"]
        == TRUSTED_WIRE_CLAIM_LEVEL
    )
    assert report["trusted_wire_origin_authenticated"] is False
    assert report["trusted_rfc8785_wire_builder_implemented"] is False
    assert report["formal_uuid_namespace_field_audit_implemented"] is False
    assert report["randomized_identifier_assignment_attested"] is False
    assert report["uncertainty_semantics_compiler_implemented"] is False
    assert report["recognizer_entrypoint_implemented"] is False
    assert report["formal_recognizer_run_runnable"] is False
    assert report["signed_sbom_validation_implemented"] is False
    assert report["runtime_attestation_signature_verifier_implemented"] is False
    assert report["internal_candidate_enumeration_implemented"] is True
    assert report["interval_selector_core_implemented"] is True
    assert report["public_selector_reenumerates_adapter_grid_from_bundle"] is True
    assert report["inconclusive_structural_competitor_forces_abstention"] is True
    assert report["admissible_scale_set_output_supported"] is True
    assert report["oci_isolation_launch_contract_implemented"] is True
    assert set(report["component_source_ids"]) == {
        "adapter",
        "covert_audit_mechanics",
        "exact_derived_witness_bridge",
        "exact_bridge",
        "exact_transform_semantics",
        "projection_compiler",
        "recognizer_input_archive_mechanics",
        "recognizer_input_archive_mechanics_v2",
        "recognizer_prediction_archive_mechanics",
        "recognizer_prediction_row_mapping_mechanics_v2",
        "runner",
        "selector",
        "trusted_wire_keyed_batch_mechanics",
        "trusted_wire_keyed_batch_mechanics_v2",
        "trusted_wire_profile_mechanics",
        "trusted_wire_compact_typed_authority_codec_v2",
        "trusted_wire_typed_authority_codec",
        "trusted_wire_typed_replay_mechanics",
        "trusted_wire_typed_replay_mechanics_v2",
        "uncertainty_compiler",
        "unsealed_prediction_structural_evaluator",
        "wire",
    }
    assert report["component_source_ids"][
        "recognizer_prediction_row_mapping_mechanics_v2"
    ] == (
        "sha256:"
        "e32133bfe7d8848b56fcbcf6f68849908b7d9a203a6e6196aae6d954e011c023"
    )
    assert report["ready_for_holdout_generation"] is False
    assert report["independent_latent_case_count"] == 720
    assert report["legal_preservation_pair_count"] == 496
    assert report["invalid_transform_control_count"] == 76
    assert report["total_preservation_sensitivity_pair_count"] == 572
    assert report["semantic_conflict_challenge_case_count"] == 240
    assert report["validation_attempts_per_version"] == 2
    assert report["maximum_validation_versions_before_no_go"] == 2
    assert "maximum_validation_protocol_runs" not in report
    assert report["baseline_specs_frozen"] is True
    assert report["exact_baseline_revisions_registered"] is False
    assert report["covert_channel_audit_frozen"] is True
    assert report["fixed_envelope_covert_audit_mechanics_implemented"] is True
    assert (
        report["fixed_envelope_covert_statistics_mechanics_implemented"]
        is True
    )
    assert report["fixed_envelope_covert_audit_semantics_id"] == (
        COVERT_AUDIT_SEMANTICS_ID
    )
    assert report["fixed_envelope_covert_audit_policy_id"] == (
        DEFAULT_COVERT_AUDIT_POLICY.policy_id
    )
    assert report["fixed_envelope_covert_audit_claim_level"] == (
        NON_AUTHORITATIVE_CLAIM_LEVEL
    )
    assert report["covert_channel_audit_implemented"] is False
    assert report["formal_covert_channel_audit_passed"] is False
    assert report["covert_channel_audit_executed"] is False
    assert report["formal_uncertainty_models_allowed"] == ["absolute_bound"]
    assert report["preservation_pairs_counted_in_720"] is False
    assert report["active_promotion_enabled"] is False


def test_protocol_preservation_transform_ids_match_the_exact_freeze():
    protocol_ids = tuple(
        requirement.transform
        for requirement in frozen_phase2b_protocol().preservation_requirements
    )
    freeze_ids = tuple(
        rule.transform for rule in frozen_phase2b_exact_freeze().preservation_rules
    )
    assert protocol_ids == freeze_ids == tuple(PreservationTransform)


def test_wire_operations_are_not_the_formal_preservation_taxonomy():
    wire_operations = {item.value for item in TransformOperation}
    preservation_transforms = {item.value for item in PreservationTransform}
    assert len(wire_operations) == len(preservation_transforms) == 8
    assert wire_operations != preservation_transforms
    assert wire_operations & preservation_transforms == {"unit_conversion"}


def test_checked_in_phase2b_preregistration_artifact_matches_runtime():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase2b_preregistration_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact == phase2b_preregistration_report()
