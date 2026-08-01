import json
from dataclasses import replace
from pathlib import Path

import pytest

from hegel_machine.milestones import PHASE2B
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
        "answerable_positive": 240,
        "wrong_family_hard_negative": 96,
        "binding_counterfactual": 96,
        "scale_counterfactual": 96,
        "sign_or_invariant_break": 96,
        "insufficient_or_ambiguous": 96,
    }
    assert protocol.separate_preservation_denominator is True
    assert protocol.holdout_run_limit == 1


def test_margin_quota_conflict_is_machine_visible_and_blocks_generation():
    protocol = frozen_phase2b_protocol()
    assert dict(protocol.margin_stratum_totals) == {
        "clear_interior": 252,
        "moderate": 216,
        "near_boundary_identifiable": 144,
        "ambiguous_or_insufficient": 108,
    }
    assert (
        "margin_strata_require_108_ambiguous_or_admissible_cases_but_case_table_allocates_96"
        in protocol.unresolved_freeze_questions
    )
    assert (
        "allowed_field_side_channel_and_identifier_randomization_audit_not_frozen"
        in protocol.unresolved_freeze_questions
    )
    assert (
        "functional_recognizer_cli_signed_minimal_image_and_archive_evaluator_not_implemented"
        in protocol.unresolved_freeze_questions
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


def _freeze_manifest():
    baselines = tuple(
        BaselineRegistration(
            kind,
            implementation_id=f"baseline_{kind.value}_v1",
            artifact_sha256=digest,
            frozen_before_holdout_generation=True,
        )
        for kind, digest in zip(BaselineKind, (SHA_A, SHA_B, SHA_C), strict=True)
    )
    protocol = frozen_phase2b_protocol()
    return ExecutionFreezeManifest(
        protocol_id=protocol.protocol_id,
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
    with pytest.raises(ValueError, match="all three baseline"):
        replace(manifest, baseline_registrations=manifest.baseline_registrations[:2])
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
    assert report["prediction_archive_evaluator_implemented"] is False
    assert report["public_wire_contract_implemented"] is True
    assert report["public_wire_is_family_neutral_shaped_only"] is True
    assert report["semantic_family_neutrality_audited"] is False
    assert report["allowed_field_answer_correlation_audit_implemented"] is False
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
        "runner",
        "selector",
        "wire",
    }
    assert report["ready_for_holdout_generation"] is False
    assert report["independent_latent_case_count"] == 720
    assert report["preservation_pairs_counted_in_720"] is False
    assert report["active_promotion_enabled"] is False


def test_checked_in_phase2b_preregistration_artifact_matches_runtime():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase2b_preregistration_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact == phase2b_preregistration_report()
