from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine import (
    phase2b_actual_unsealed_960_case_prediction_scoring_completion_replay_v2
    as replay,
)
from hegel_machine import (
    phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2 as core,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


ROOT = Path(__file__).resolve().parents[1]


def _digest(seed: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(seed.encode("ascii")).hexdigest()


def _state_inputs() -> tuple[
    replay.LoadedJsonV2,
    replay.LoadedJsonV2,
    dict[str, object],
    dict[str, object],
    core.ActualUnsealed960PreconsumptionAdmissionV2,
]:
    prereg_value = {
        "schema_version": replay.PREREGISTRATION_SCHEMA_V2,
        "action_id": replay.ACTION_ID_V2,
    }
    prereg_raw = json.dumps(prereg_value, sort_keys=True).encode("ascii")
    prereg = replay.LoadedJsonV2(
        Path("preregistration.json"),
        prereg_raw,
        replay.raw_sha256(prereg_raw),
        prereg_value,
    )
    contract_value = {
        "schema_version": replay.CONTRACT_SCHEMA_V2,
        "action_id": replay.ACTION_ID_V2,
        "preregistration_raw_sha256": prereg.raw_sha256,
    }
    contract_raw = json.dumps(contract_value, sort_keys=True).encode("ascii")
    contract = replay.LoadedJsonV2(
        Path("contract.json"),
        contract_raw,
        replay.raw_sha256(contract_raw),
        contract_value,
    )
    rows = [
        {
            "relative_path": path,
            "byte_count": index + 1,
            "raw_sha256": _digest(f"source-{index}"),
        }
        for index, path in enumerate(replay.SOURCE_PATHS_V2)
    ]
    binding_rows = [
        {"name": name, "identity": f"fixture-{index}-{name}"}
        for index, name in enumerate(core.REQUIRED_STATIC_BINDING_NAMES_V2)
    ]
    capsule = replay.build_capsule(
        contract,
        prereg,
        source_commit="1" * 40,
        source_closure_rows=rows,
        static_binding_rows=binding_rows,
        dependency_receipt_root=_digest("dependencies"),
        run_id="2" * 32,
        created_at_unix_seconds=1_800_000_000,
    )
    authorization = replay.build_authorization(contract, prereg, capsule)
    admitted = core.ActualUnsealed960PreconsumptionAdmissionV2(
        static_contract_closure_verified=True,
        authoritative_schema_verifier_available=True,
        authoritative_r01_r16_bundle_verified=True,
        external_custodian_checkpoint_verified=True,
        three_baseline_outputs_verified=True,
        exact_scale_bridge_verified=True,
        recognizer_executor_binding_verified=True,
        scorer_executor_binding_verified=True,
    )
    return contract, prereg, capsule, authorization, admitted


def _complete_snapshot_and_evidence(
    capsule: dict[str, object],
    authorization: dict[str, object],
    consumption: dict[str, object],
    admission: core.ActualUnsealed960PreconsumptionAdmissionV2,
) -> tuple[
    core.ActualUnsealed960ReplaySnapshotV2,
    core.ActualUnsealed960ReplaySealedEvidenceV2,
]:
    snapshot = core.initial_actual_unsealed_960_replay_snapshot_v2(
        replay.static_bindings_from_capsule(capsule)
    )
    snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot, core.ActualUnsealed960ReplayEventV2.RECORD_AUTHORIZATION
    )
    snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        core.ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=admission,
    )
    for event in (
        core.ActualUnsealed960ReplayEventV2.CONSUME_AUTHORIZATION,
        core.ActualUnsealed960ReplayEventV2.SEAL_RECOGNIZER_RESULT,
        core.ActualUnsealed960ReplayEventV2.ADMIT_BASELINE_RESULTS,
        core.ActualUnsealed960ReplayEventV2.ADMIT_PREDICTION_ARCHIVE,
        core.ActualUnsealed960ReplayEventV2.ADMIT_ANSWER_MANIFEST,
        core.ActualUnsealed960ReplayEventV2.SEAL_SCORER_REPLAY,
        core.ActualUnsealed960ReplayEventV2.COMPLETE_MECHANICS,
    ):
        snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
            snapshot, event
        )
    evidence = core.ActualUnsealed960ReplaySealedEvidenceV2(
        authorization_digest=authorization["authorization_digest"],
        consumption_record_digest=consumption["record_digest"],
        authoritative_r16_bundle_id="r16-authoritative-bundle-fixture",
        campaign_id="imported-campaign-fixture",
        attempt_id="imported-attempt-fixture",
        recognizer_result_id="recognizer-result-fixture",
        baseline_result_ids=("baseline-a", "baseline-b", "baseline-c"),
        prediction_archive_id="prediction-archive-fixture",
        answer_manifest_id="answer-manifest-fixture",
        scorer_bundle_id="scorer-bundle-fixture",
        score_report_id="score-report-fixture",
        available_gate_report_id="available-gate-report-fixture",
        upstream_admission_manifest_raw_sha256=_digest("upstream-manifest"),
        scorer_bundle_raw_sha256=_digest("scorer-bundle-raw"),
        total_predictions=960,
        main_scored_rows=720,
        semantic_conflict_excluded=240,
        metric_results=9,
        available_overall_gate_mechanics=10,
        available_slice_gate_mechanics=24,
        unavailable_formal_overall_gates=2,
    )
    return snapshot, evidence


def test_preconsumption_pending_is_same_capsule_zero_action_and_not_consumable() -> None:
    contract, prereg, capsule, authorization, _ = _state_inputs()
    blocked = core.ActualUnsealed960PreconsumptionAdmissionV2(
        static_contract_closure_verified=True,
        authoritative_schema_verifier_available=True,
        authoritative_r01_r16_bundle_verified=False,
        external_custodian_checkpoint_verified=True,
        three_baseline_outputs_verified=True,
        exact_scale_bridge_verified=True,
        recognizer_executor_binding_verified=True,
        scorer_executor_binding_verified=True,
    )
    pending = replay.build_preconsumption_pending(
        contract, prereg, capsule, authorization, blocked
    )
    assert replay.validate_preconsumption_pending(
        pending, contract, prereg, capsule, authorization, blocked
    ) == pending
    assert pending["resume_capsule_digest"] == capsule["capsule_digest"]
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert pending["campaign_id_allocation_count"] == 0
    assert pending["attempt_id_allocation_count"] == 0
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="cannot be consumed",
    ):
        replay.build_consumption(
            contract, prereg, capsule, authorization, blocked
        )


def test_canonical_sealed_artifact_replays_to_mechanics_only_terminal() -> None:
    contract, prereg, capsule, authorization, admitted = _state_inputs()
    consumption = replay.build_consumption(
        contract, prereg, capsule, authorization, admitted
    )
    snapshot, evidence = _complete_snapshot_and_evidence(
        capsule, authorization, consumption, admitted
    )
    artifact = replay.build_sealed_completion_artifact_v2(snapshot, evidence)
    decoded = replay.decode_sealed_completion_artifact_v2(artifact.canonical_bytes)
    assert decoded == artifact
    assert replay.validate_artifact_binding(
        artifact.canonical_bytes, capsule, authorization, consumption
    ) == artifact

    terminal = replay.build_complete_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        admitted,
        artifact.canonical_bytes,
    )
    assert replay.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        admitted,
        artifact_payload=artifact.canonical_bytes,
    ) == terminal
    outcome = terminal["outcome"]
    assert outcome["status"] == core.MECHANICS_COMPLETE_STATUS_V2
    assert outcome["mechanics_complete"] is True
    assert outcome["formal_gate_evaluation_performed"] is False
    assert outcome["formal_c1_report_verified"] is False
    assert outcome["c1_pass"] is False
    assert outcome["effect_evidence"] is False
    assert outcome["external_authority"] is False
    assert outcome["upstream_admission_manifest_raw_sha256"] == _digest("upstream-manifest")
    assert outcome["scorer_bundle_raw_sha256"] == _digest("scorer-bundle-raw")
    assert (
        outcome["total_predictions"],
        outcome["main_scored_rows"],
        outcome["semantic_conflict_excluded"],
        outcome["metric_results"],
        outcome["available_overall_gate_mechanics"],
        outcome["available_slice_gate_mechanics"],
        outcome["unavailable_formal_overall_gates"],
    ) == (960, 720, 240, 9, 10, 24, 2)


def test_artifact_binding_claim_and_canonical_tampering_are_rejected() -> None:
    contract, prereg, capsule, authorization, admitted = _state_inputs()
    consumption = replay.build_consumption(
        contract, prereg, capsule, authorization, admitted
    )
    snapshot, evidence = _complete_snapshot_and_evidence(
        capsule, authorization, consumption, admitted
    )
    payload = replay.encode_sealed_completion_artifact_v2(snapshot, evidence)

    wire = list(canonical_cbor_decode(payload))
    wire[4] = b"\xff" * 32
    wrong_capsule = canonical_cbor_encode(tuple(wire))
    replay.decode_sealed_completion_artifact_v2(wrong_capsule)
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="state binding",
    ):
        replay.validate_artifact_binding(
            wrong_capsule, capsule, authorization, consumption
        )

    wire = list(canonical_cbor_decode(payload))
    wire[32] = True
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="wire differs|claim boundary",
    ):
        replay.decode_sealed_completion_artifact_v2(
            canonical_cbor_encode(tuple(wire))
        )
    wire = list(canonical_cbor_decode(payload))
    wire[20] = 959
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="claim boundary",
    ):
        replay.decode_sealed_completion_artifact_v2(
            canonical_cbor_encode(tuple(wire))
        )

    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="not canonical",
    ):
        replay.decode_sealed_completion_artifact_v2(payload + b"\xf6")


@pytest.mark.parametrize("raw_sha_index", [18, 19])
def test_terminal_rejects_sealed_raw_sha_drift(raw_sha_index: int) -> None:
    contract, prereg, capsule, authorization, admitted = _state_inputs()
    consumption = replay.build_consumption(
        contract, prereg, capsule, authorization, admitted
    )
    snapshot, evidence = _complete_snapshot_and_evidence(
        capsule, authorization, consumption, admitted
    )
    artifact = replay.build_sealed_completion_artifact_v2(snapshot, evidence)
    terminal = replay.build_complete_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        admitted,
        artifact.canonical_bytes,
    )
    wire = list(canonical_cbor_decode(artifact.canonical_bytes))
    wire[raw_sha_index] = b"\xee" * 32
    drifted = canonical_cbor_encode(tuple(wire))
    replay.decode_sealed_completion_artifact_v2(drifted)
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="terminal projection",
    ):
        replay.validate_terminal(
            terminal,
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            admitted,
            artifact_payload=drifted,
        )


def test_terminal_replay_is_exact_and_invalidated_terminal_has_no_resume() -> None:
    contract, prereg, capsule, authorization, admitted = _state_inputs()
    consumption = replay.build_consumption(
        contract, prereg, capsule, authorization, admitted
    )
    invalidated = replay.build_invalidated_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        admitted,
        failure_phase="SEALED_ARTIFACT_ADMISSION",
        failure_code="SEALED_ARTIFACT_MISSING_AFTER_CONSUMPTION",
        failure_detail="sealed public artifact was absent",
    )
    assert replay.validate_terminal(
        invalidated,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        admitted,
    ) == invalidated
    assert invalidated["outcome"]["terminal_no_resume"] is True

    changed = deepcopy(invalidated)
    changed["outcome"]["c1_pass"] = True
    with pytest.raises(
        replay.ActualUnsealed960CompletionReplayErrorV2,
        match="terminal projection",
    ):
        replay.validate_terminal(
            changed,
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            admitted,
        )


def test_completion_module_has_no_producer_or_process_import_surface() -> None:
    policy = replay.EXECUTION_POLICY_V2
    assert policy["campaign_id_allocation_limit"] == 0
    assert policy["attempt_id_allocation_limit"] == 0
    assert policy["actual_cohort_generation_limit"] == 0
    assert policy["actual_cohort_producer_read_limit"] == 0
    assert policy["recognizer_execution_limit"] == 0
    assert policy["recognizer_result_admission_limit"] == 1
    assert policy["baseline_result_admission_limit"] == 3
    assert policy["prediction_archive_admission_limit"] == 1
    assert policy["answer_manifest_admission_limit"] == 1
    assert policy["scorer_replay_limit"] == 1
    assert policy["formal_gate_evaluation_limit"] == 0
    assert policy["c1_pass_publication_limit"] == 0

    source_path = (
        ROOT
        / "src/hegel_machine/phase2b_actual_unsealed_960_case_prediction_scoring_completion_replay_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        forbidden in name
        for name in imported
        for forbidden in (
            "recognizer_prediction",
            "recognizer_input",
            "scorer_bundle",
            "baseline",
            "subprocess",
            "docker",
        )
    )
