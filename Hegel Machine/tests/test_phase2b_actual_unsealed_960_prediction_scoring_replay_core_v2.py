from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hegel_machine.phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2 import (
    CLAIM_LEVEL_V2,
    MECHANICS_COMPLETE_STATUS_V2,
    READY_STATUS_V2,
    REQUIRED_STATIC_BINDING_NAMES_V2,
    ActualUnsealed960PendingReasonV2,
    ActualUnsealed960PreconsumptionAdmissionV2,
    ActualUnsealed960ReplayActionAuditV2,
    ActualUnsealed960ReplayCoreErrorV2,
    ActualUnsealed960ReplayEventV2,
    ActualUnsealed960ReplaySealedEvidenceV2,
    ActualUnsealed960ReplayStateV2,
    ActualUnsealed960ReplayStaticBindingsV2,
    ZERO_ACTION_AUDIT_V2,
    build_actual_unsealed_960_replay_completion_projection_v2,
    initial_actual_unsealed_960_replay_snapshot_v2,
    reduce_actual_unsealed_960_prediction_scoring_replay_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _bindings() -> ActualUnsealed960ReplayStaticBindingsV2:
    return ActualUnsealed960ReplayStaticBindingsV2(
        capsule_digest=_digest("a"),
        source_closure_root=_digest("b"),
        binding_ids=tuple(
            (name, f"bound:{index:02d}:{name}")
            for index, name in enumerate(REQUIRED_STATIC_BINDING_NAMES_V2)
        ),
    )


def _admission(**overrides: bool) -> ActualUnsealed960PreconsumptionAdmissionV2:
    values = {
        "static_contract_closure_verified": True,
        "authoritative_schema_verifier_available": True,
        "authoritative_r01_r16_bundle_verified": True,
        "external_custodian_checkpoint_verified": True,
        "three_baseline_outputs_verified": True,
        "exact_scale_bridge_verified": True,
        "recognizer_executor_binding_verified": True,
        "scorer_executor_binding_verified": True,
    }
    values.update(overrides)
    return ActualUnsealed960PreconsumptionAdmissionV2(**values)


def _authorized_snapshot():
    snapshot = initial_actual_unsealed_960_replay_snapshot_v2(_bindings())
    return reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        ActualUnsealed960ReplayEventV2.RECORD_AUTHORIZATION,
    )


def _complete_snapshot():
    snapshot = _authorized_snapshot()
    snapshot = reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=_admission(),
    )
    for event in (
        ActualUnsealed960ReplayEventV2.CONSUME_AUTHORIZATION,
        ActualUnsealed960ReplayEventV2.SEAL_RECOGNIZER_RESULT,
        ActualUnsealed960ReplayEventV2.ADMIT_BASELINE_RESULTS,
        ActualUnsealed960ReplayEventV2.ADMIT_PREDICTION_ARCHIVE,
        ActualUnsealed960ReplayEventV2.ADMIT_ANSWER_MANIFEST,
        ActualUnsealed960ReplayEventV2.SEAL_SCORER_REPLAY,
        ActualUnsealed960ReplayEventV2.COMPLETE_MECHANICS,
    ):
        snapshot = reduce_actual_unsealed_960_prediction_scoring_replay_v2(
            snapshot,
            event,
        )
    return snapshot


def test_core_is_pure_and_has_no_execution_or_target_import_surface() -> None:
    path = (
        ROOT
        / "src/hegel_machine/phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_roots.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    assert imported_roots == {"__future__", "dataclasses", "enum", "typing"}
    assert not {
        "open",
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
        "Popen",
        "run",
        "system",
    }.intersection(called_names)
    for forbidden in (
        "subprocess",
        "pathlib",
        "socket",
        "urllib",
        "requests",
        "docker",
        "phase2b_recognizer_host_executor",
        "phase2b_scorer_host_executor",
        "phase2b_scorer_bundle",
        "actual cohort",
    ):
        assert forbidden not in source.lower()


def test_engineering_freeze_snapshot_is_exactly_action_free() -> None:
    snapshot = initial_actual_unsealed_960_replay_snapshot_v2(_bindings())

    assert READY_STATUS_V2.endswith("CAPSULE_READY_NOT_AUTHORIZED")
    assert snapshot.state is ActualUnsealed960ReplayStateV2.FROZEN_NOT_AUTHORIZED
    assert snapshot.audit == ZERO_ACTION_AUDIT_V2
    assert snapshot.completed_prefix == ()
    assert snapshot.pending_reasons == ()
    assert snapshot.admission_satisfied is False
    assert snapshot.formal_gate_evaluation_performed is False
    assert snapshot.formal_c1_report_verified is False
    assert snapshot.c1_pass is False
    assert snapshot.effect_evidence is False


def test_missing_authoritative_verifier_baselines_and_scale_bridge_stay_pending() -> None:
    snapshot = _authorized_snapshot()
    snapshot = reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=_admission(
            authoritative_schema_verifier_available=False,
            authoritative_r01_r16_bundle_verified=False,
            three_baseline_outputs_verified=False,
            exact_scale_bridge_verified=False,
        ),
    )

    assert snapshot.state is ActualUnsealed960ReplayStateV2.PRECONSUMPTION_PENDING
    assert snapshot.pending_reasons == (
        ActualUnsealed960PendingReasonV2.AUTHORITATIVE_SCHEMA_VERIFIER_UNAVAILABLE.value,
        ActualUnsealed960PendingReasonV2.AUTHORITATIVE_R01_R16_BUNDLE_UNVERIFIED.value,
        ActualUnsealed960PendingReasonV2.THREE_BASELINE_OUTPUTS_UNVERIFIED.value,
        ActualUnsealed960PendingReasonV2.EXACT_SCALE_BRIDGE_UNVERIFIED.value,
    )
    assert snapshot.audit == ZERO_ACTION_AUDIT_V2

    with pytest.raises(ActualUnsealed960ReplayCoreErrorV2) as error:
        reduce_actual_unsealed_960_prediction_scoring_replay_v2(
            snapshot,
            ActualUnsealed960ReplayEventV2.CONSUME_AUTHORIZATION,
        )
    assert error.value.code == "REJECT_REPLAY_TRANSITION"


def test_same_capsule_pending_replay_can_clear_without_consumption() -> None:
    snapshot = _authorized_snapshot()
    snapshot = reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=_admission(exact_scale_bridge_verified=False),
    )
    snapshot = reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=_admission(),
    )

    assert snapshot.state is ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED
    assert snapshot.bindings.capsule_digest == _digest("a")
    assert snapshot.admission_satisfied is True
    assert snapshot.audit.readiness_admissions == 1
    assert snapshot.audit.logical_campaigns == 0
    assert snapshot.audit.authorization_consumptions == 0


def test_ordered_one_shot_reducer_finishes_only_as_nonformal_non_c1() -> None:
    snapshot = _complete_snapshot()

    assert snapshot.state is (
        ActualUnsealed960ReplayStateV2.MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1
    )
    assert snapshot.completed_prefix == (
        "recognizer",
        "baselines",
        "prediction_archive",
        "answer_manifest",
        "scorer_replay",
    )
    assert snapshot.audit == ActualUnsealed960ReplayActionAuditV2(
        logical_campaigns=1,
        authorization_consumptions=1,
        readiness_admissions=1,
        recognizer_result_admissions=1,
        baseline_output_admissions=3,
        prediction_archive_admissions=1,
        answer_manifest_admissions=1,
        scorer_replays=1,
    )
    assert snapshot.formal_gate_evaluation_performed is False
    assert snapshot.formal_c1_report_verified is False
    assert snapshot.c1_pass is False
    assert snapshot.effect_evidence is False


def test_completion_projection_contains_only_mechanics_success_and_nonclaims() -> None:
    projection = build_actual_unsealed_960_replay_completion_projection_v2(
        _complete_snapshot(),
        ActualUnsealed960ReplaySealedEvidenceV2(
            authorization_digest=_digest("c"),
            consumption_record_digest=_digest("d"),
            authoritative_r16_bundle_id="r16:sealed-authoritative-bundle",
            campaign_id="campaign:externally-bound",
            attempt_id="attempt:externally-bound",
            recognizer_result_id="recognizer:sealed-result",
            baseline_result_ids=(
                "baseline:embedding-nearest-prototype",
                "baseline:frozen-llm-semantic-only",
                "baseline:flat-learned-typed",
            ),
            prediction_archive_id="prediction:sealed-archive",
            answer_manifest_id="answer:independent-reveal",
            scorer_bundle_id="scorer:sealed-bundle",
            score_report_id="score:mechanics-report",
            available_gate_report_id="gate:available-mechanics-report",
            upstream_admission_manifest_raw_sha256=_digest("e"),
            scorer_bundle_raw_sha256=_digest("f"),
            total_predictions=960,
            main_scored_rows=720,
            semantic_conflict_excluded=240,
            metric_results=9,
            available_overall_gate_mechanics=10,
            available_slice_gate_mechanics=24,
            unavailable_formal_overall_gates=2,
        ),
    )

    assert projection.status == MECHANICS_COMPLETE_STATUS_V2
    assert projection.claim_level == CLAIM_LEVEL_V2
    assert projection.mechanics_complete is True
    assert projection.formal_gate_evaluation_performed is False
    assert projection.formal_c1_report_verified is False
    assert projection.c1_pass is False
    assert projection.effect_evidence is False
    assert projection.external_authority is False
    assert projection.total_predictions == 960
    assert projection.main_scored_rows == 720
    assert projection.semantic_conflict_excluded == 240
    assert projection.metric_results == 9
    assert projection.available_overall_gate_mechanics == 10
    assert projection.available_slice_gate_mechanics == 24
    assert projection.unavailable_formal_overall_gates == 2


def test_reducer_rejects_out_of_order_repeat_and_terminal_resume() -> None:
    snapshot = _authorized_snapshot()
    with pytest.raises(ActualUnsealed960ReplayCoreErrorV2):
        reduce_actual_unsealed_960_prediction_scoring_replay_v2(
            snapshot,
            ActualUnsealed960ReplayEventV2.SEAL_RECOGNIZER_RESULT,
        )

    complete = _complete_snapshot()
    for event in ActualUnsealed960ReplayEventV2:
        with pytest.raises(ActualUnsealed960ReplayCoreErrorV2) as error:
            reduce_actual_unsealed_960_prediction_scoring_replay_v2(complete, event)
        assert error.value.code == "REJECT_REPLAY_TERMINAL"


def test_static_binding_order_and_forbidden_claim_counts_fail_closed() -> None:
    rows = tuple(
        (name, f"bound:{index:02d}:{name}")
        for index, name in enumerate(REQUIRED_STATIC_BINDING_NAMES_V2)
    )
    with pytest.raises(ActualUnsealed960ReplayCoreErrorV2) as error:
        ActualUnsealed960ReplayStaticBindingsV2(
            capsule_digest=_digest("a"),
            source_closure_root=_digest("b"),
            binding_ids=tuple(reversed(rows)),
        )
    assert error.value.code == "REJECT_REPLAY_BINDINGS"

    with pytest.raises(ActualUnsealed960ReplayCoreErrorV2) as error:
        ActualUnsealed960ReplayActionAuditV2(formal_gate_evaluations=1)
    assert error.value.code == "REJECT_REPLAY_BOUNDARY"
